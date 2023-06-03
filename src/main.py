#!/usr/bin/env python3
import argparse
import base64
import http.client
import json
import logging
import math
import pickle
import requests
import signal
import ssl
import threading
import time
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Dict, Generator, List, Tuple

import cv2
import grpc
import numpy
from PIL import Image, UnidentifiedImageError

import image_classification_pb2
from image_classification_pb2_grpc import ImageClassifierStub

# Classification server configuration.
CLASSIFY_HOST="localhost"
CLASSIFY_PORT=8000

# Relative path to the model and configs, from which the script was invoked.
IMAGE_STORAGE_PATH = '/mnt/apt_cam_captures'
API_HOST = '4bit.local'
API_PORT = 3000
MESSAGE_ENDPOINT = '/telegram/message'
CLIENT_CERT_PATH = 'certs/video0.crt'
CLIENT_KEY_PATH = 'certs/video0.key'
TRUSTED_CA_PATH = 'certs/4bit_server_chain.crt'

# Classification configuration.
from common import IMAGE_HEIGHT, IMAGE_WIDTH

IS_RUNNING = True
VIDEO0_DEVICE_NAME = "video0"

# Map of type:
# [cam_ip: string]: {
#   "name": string,  # Name of the device.
#   "data": bytes,   # Current device image stored as base64.
# }
API_CAMERA_MAP = {}

# Map of type:
# [cam_ip: string]: {
#   "CropFrameHeight": float,
#   "CropFrameWidth": float,
#   "CropFrameX": int,
#   "CropFrameY": int,
# }
API_CAMERA_ADJUSTMENTS_MAP = {}

# Cooldown period for quering connected cameras. This is used for getting the current
# state of all connected cameras without data going stale. The cooldown period is for
# periodic state queries.
API_CAMERA_COOLDOWN = 60 * 20 # Every 20min.

def streamApiCameras_thread():
    """
        Intended to run in a thread, which subscribes to the API streaming endpoint
        consuming live data from all cameras to be consumed and classified.

        Requirements:
            IS_RUNNING: This thread listens on that global variable to stop running.
    """
    global IS_RUNNING
    snap_endpoint = f'https://{API_HOST}:{API_PORT}/camera/snap'
    list_endpoint = f'https://{API_HOST}:{API_PORT}/camera/list'

    def snap_helper():
        """
            Helper function for invoking the /snap endpoint to gather images of connected
            cameras.
        """
        with requests.get(
            snap_endpoint,
            cert=(CLIENT_CERT_PATH, CLIENT_KEY_PATH),
            verify=TRUSTED_CA_PATH,
            json={},
        ) as response:
            if response.status_code == 200:
                obj = response.json()
                # Store the current state.
                global API_CAMERA_MAP
                API_CAMERA_MAP = obj['cameras']

            else:
                raise Exception(f"API Endpoint connection failed with status code {response.status_code}")

    def list_helper():
        """
            Helper function for getting the list of connected cameras.
        """
        logging.info(f'Querying the state of connected cameras')

        with requests.get(
            list_endpoint,
            cert=(CLIENT_CERT_PATH, CLIENT_KEY_PATH),
            verify=TRUSTED_CA_PATH,
            json={},
        ) as response:
            if response.status_code == 200:
                obj = response.json()
                # Store the current state.
                global API_CAMERA_ADJUSTMENTS_MAP

                for cam in obj['Cameras']:
                    cam_ip = cam['IP']
                    cam_adjustments = cam['Adjustment']
                    API_CAMERA_ADJUSTMENTS_MAP[cam_ip] = cam_adjustments
                logging.debug(f'Camera list query result: {API_CAMERA_ADJUSTMENTS_MAP}')

            else:
                raise Exception(f"API Endpoint connection failed with status code {response.status_code}")


    # Track the cooldown query based on upcoming query time.
    next_query_time = 0

    while IS_RUNNING:
        # Set a timeout.
        time.sleep(1)

        # Grab snaps of all connected cameras.
        try:
            snap_helper()
        except Exception as e:
            logging.exception(f'API Streaming connection failed (retrying): ', exc_info=e)
            time.sleep(5)

        # Query the state of connected cameras.
        if next_query_time <= time.time():
            try:
                list_helper()
                # Calculate the next query time.
                next_query_time = time.time() + API_CAMERA_COOLDOWN
            except Exception as e:
                logging.exception(f'API List endpoint failed: ', exc_info=e)
                time.sleep(5)

def clean_up(video: cv2.VideoCapture) -> None:
    """
    Takes care of cleaning up the VideoCapture instance and releasing any
    memory held by OpenCV
    """
    global IS_RUNNING

    logging.info('Cleaning up')
    IS_RUNNING = False
    video.release()
    cv2.destroyAllWindows()

def send_message(body: str, b64_image: bytes) -> None:
    request_headers = {
        'Content-Type': 'application/json',
    }
    request_body = {
        "chatId": 5039741009,
        "message": body,
        "image": b64_image.decode('utf-8'),
    }

    context= ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    context.load_cert_chain(CLIENT_CERT_PATH, CLIENT_KEY_PATH)
    connection = http.client.HTTPSConnection(API_HOST, port=API_PORT, context=context)

    connection.request(
        method='POST',
        url=MESSAGE_ENDPOINT,
        headers=request_headers,
        body=json.dumps(request_body)
    )
    resp = connection.getresponse()
    logging.info("[API] Response -> status({}) reason({})".format(
        resp.status,
        resp.reason,
    ))

def apply_image_adjustments(image: Image, cam_ip: str) -> numpy.ndarray:
    """
        Adjusts a given image with respect to the stored camera adjustments.

        Args:
            image: The image instance to adjust.
            cam_ip: The camera's IP used as a key to lookup corresponding adjustments for.

        Return:
            Adjusted image as a numpy array.
    """
    # Convert the image to a numpy array.
    image_np = numpy.asarray(image)

    if cam_ip in API_CAMERA_ADJUSTMENTS_MAP:
        cam_adjustments = API_CAMERA_ADJUSTMENTS_MAP[cam_ip]
        image_frame_crop_height = cam_adjustments['CropFrameHeight']
        image_frame_crop_width = cam_adjustments['CropFrameWidth']
        image_frame_crop_x = cam_adjustments['CropFrameX']
        image_frame_crop_y = cam_adjustments['CropFrameY']

        # Crop the image
        h, w = image_np.shape[:2]
        h_ajustment = math.floor(h * image_frame_crop_height)
        w_ajustment = math.floor(w * image_frame_crop_width)
        x, y = image_frame_crop_x, image_frame_crop_y

        logging.debug(f'Applying frame crop H={h_ajustment} H={w_ajustment} on x={x}, y={y}')
        new_h = h - h_ajustment
        new_w = w - w_ajustment
        image_np = image_np[y:y+new_h, x:x+new_w]

    return image_np

def capture_images(video: cv2.VideoCapture) -> Generator[image_classification_pb2.ClassifyImageRequest, None, None]:
    """
        Capture images from the variuos video input devices, generating gRPC request stream messages.

        Args:
            video: VideoCapture instance to real camera hardware.

        Returns:
            Image classify request streams.
    """
    # Captured images are tuples of the device name and the ndarray that was captured.
    captured_images: List[Tuple[str, numpy.ndarray]] = []

    # Capture frame from the connected video device.
    ret, captured_frame = video.read()
    if ret != True:
        logging.error('[loop] ERROR: Run loop failed to read video frame')
    else:
        captured_images.append((VIDEO0_DEVICE_NAME, captured_frame))

    # Query frame from connected devices.
    # Shallow copy to prevent thread ownership issues.
    api_cam_mp = API_CAMERA_MAP.copy()
    for camera_ip in api_cam_mp:
        cam = api_cam_mp[camera_ip]
        cam_name = cam['name']
        cam_data = cam['data']
        device_name = f"{cam_name}:{camera_ip}"

        # Extract current camera's image.
        try:
            img_b64 = cam_data
            if not img_b64:
                continue

            img = base64.b64decode(img_b64)
            _img = Image.open(BytesIO(img))
            nd_array = apply_image_adjustments(_img, camera_ip)
            logging.debug(f'[loop] ESP Device {device_name} captured image -> {nd_array.shape}')
            captured_images.append((device_name, nd_array))
        except UnidentifiedImageError:
                logging.error(f'[loop] Failed to interpret image from ESP Device {device_name}')
        except Exception as e:
            logging.warning(f'[loop] Unknown exception while interpreting image from ESP Device {device_name}: {e}')


    # Create a generator for the captured images.
    for device_name, captured_frame in captured_images:
        logging.info(f'[loop] Classifying device: "{device_name}" Frame: {captured_frame.shape}')
        yield image_classification_pb2.ClassifyImageRequest(
            image=pickle.dumps(captured_frame),
            device=device_name,
        )

def start_loop(video: cv2.VideoCapture, cooldown: int) -> int:
    """
    Run loop which starts reading frames from the video and invoke the
    dnn model through a gRPC server on what's being presented in each frame.

    Args:
        video (VideoCapture): OpenCV VideoCapture instance to read frames from.
        cooldown (int): Cooldown in minutes between notifying the user of a
            image detection.
    """
    # Setup gRPC client with the classification server.
    # A helper function in which can be invoked to re-establish the channel upon a disconnect.
    def _setup_grpc_client() -> ImageClassifierStub:
        channel = grpc.insecure_channel(
            f'{CLASSIFY_HOST}:{CLASSIFY_PORT}',
            options=[
                ("grpc.max_receive_message_length", IMAGE_WIDTH*IMAGE_HEIGHT*20),
                ("grpc.max_send_message_length", IMAGE_WIDTH*IMAGE_HEIGHT*20),
            ],
        )
        classify_client: ImageClassifierStub = ImageClassifierStub(channel)
        return classify_client

    # Checks the results and issues notification to the user.
    def _check_and_notify_dection(device_name: str, matches: List[int], match_scores: List[float], frame: numpy.ndarray, last_notified: datetime) -> datetime:
        # Check if HOOOMAN was detected.
        ACCEPTED_MATCHES = [ 1, 17, 18 ]
        if [i for i in matches if i in ACCEPTED_MATCHES]:
            # Convert image to a blob that can be send in a post request
            # to 4bit.api on the /message endpoint.
            now = datetime.now()
            img_path = Path(IMAGE_STORAGE_PATH, 'hooman-{}-{}.jpg'.format(str(now), device_name))
            cv2.imwrite(str(img_path), frame)
            raw_image = cv2.imencode('.jpg', frame)[1]
            logging.info(f'[loop] Hooman detected by {device_name}! Saving to {img_path}')

            # Have a cooldown for invoking the endpoint. However, save
            # those images to the external flash drive with timestamps.
            cooldown_s = cooldown * 60
            if last_notified == None or (now - last_notified).seconds >= cooldown_s:
                last_notified = now
                b64_image = base64.b64encode(raw_image)
                send_message('[Device={}|Scores={}] Hoooman detected at {}.'.format(
                    device_name,
                    match_scores,
                    str(now),
                ), b64_image)
                return now


    # Track cooldowns based on device name.
    last_notified: Dict[str, datetime] = {
        VIDEO0_DEVICE_NAME: None,
    }

    classify_client = _setup_grpc_client()
    global IS_RUNNING
    while IS_RUNNING:
        # Lets detect captured images.
        try:
            for classfied_image in classify_client.ClassifyImage(capture_images(video)):
                # classfied_image.matches
                logging.info(f'[loop] Match results from {classfied_image.device} = {classfied_image.matches}')

                # Extract the ndarray frame from the response.
                frame = pickle.loads(classfied_image.image)

                # Check if device was entered into cooldown map.
                if classfied_image.device not in last_notified:
                    last_notified[classfied_image.device] = None

                # Check if matches merit notifying a detection.
                notified = _check_and_notify_dection(
                    classfied_image.device,
                    classfied_image.matches,
                    classfied_image.match_scores,
                    frame,
                    last_notified[classfied_image.device],
                )
                if notified:
                    last_notified[classfied_image.device] = notified
        except grpc.RpcError as e:
            e.details()

            # Handle disconnected errors.
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                logging.warning(f'gRPC Server possibly disconnected with an UNAVAILABLE status code. Retrying connection...')
                classify_client = _setup_grpc_client()

        except Exception as e:
            logging.error("Unhandled exception occurred")
            raise e

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start detecting those HOOMANS")
    parser.add_argument(
        '--cooldown',
        '-c',
        type=int,
        default=5,
        help='Cooldown in minutes between notifying the user of detected image.'
    )
    parser.add_argument(
        '--skip-esp',
        action='store_true',
        help='Skips setting up ESP cameras.'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        help='Sets the log verbosity',
    )
    return parser.parse_args()

def signal_safe_exit(video: cv2.VideoCapture):
    def _sig_helper_func(signal, frame):
        logging.info('[Signal] Interrupt detected: Exiting safely')
        clean_up(video)
    return _sig_helper_func

def main():
    # Parse those args!
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO))

    # Open and check that we can read from the video camera.
    video_file = f"/dev/{VIDEO0_DEVICE_NAME}"
    video_capture = cv2.VideoCapture(video_file)

    if not video_capture.isOpened():
        logging.error(f'[main] {video_file} failed to open.')
        clean_up(video_capture)
        return 1

    # Adjust frame resolution.
    video_capture.set(cv2.CAP_PROP_FPS, 1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    # Register signal handler for safely exiting on interrupt.
    signal.signal(signal.SIGINT, signal_safe_exit(video_capture))

    # Start threads.
    api_camera_query_thread = threading.Thread(target=streamApiCameras_thread)
    api_camera_query_thread.start()

    # Start the run loop.
    logging.info('[main] Starting detection loop')
    loop_status = False
    try:
        loop_status = start_loop(
            video_capture,
            args.cooldown,
        )
        logging.info(f'[main] Run loop status returned {loop_status}')
    except Exception as e:
        logging.error(f'[main] Unexpected exception: {e}')
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        clean_up(video_capture)

        # Wait for threads to stop.
        api_camera_query_thread.join()

    return loop_status

if __name__ == '__main__':
    ret = main()
    exit(ret)
