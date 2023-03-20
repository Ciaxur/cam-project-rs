#!/usr/bin/env python3
import argparse
import base64
import http.client
import json
import logging
import pickle
import signal
import ssl
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Generator, List, Tuple, Dict

import cv2
import grpc
import numpy
from PIL import Image, UnidentifiedImageError

import image_classification_pb2
from image_classification_pb2_grpc import ImageClassifierStub
from VideoDevice import ESP32_CAM

# Classification server configuration.
CLASSIFY_HOST="localhost"
CLASSIFY_PORT=8000

# Relative path to the model and configs, from which the script was invoked.
IMAGE_STORAGE_PATH = '/mnt/apt_cam_captures'
API_HOST = 'localhost'
API_PORT = 3000
MESSAGE_ENDPOINT = '/telegram/message'
CLIENT_CERT_PATH = 'certs/cam-client1.crt'
CLIENT_KEY_PATH = 'certs/cam-client1.key'

# Classification configuration.
from common import (
    IMAGE_HEIGHT, IMAGE_WIDTH
)
IS_RUNNING = True

# Array of tuples containing IP and port to poll from ESP devices.
# Device name is optional. Default it to empty string.
ESP_DEVICES = [
    ("door_cam0", "192.168.0.9", 3000)
]
VIDEO0_DEVICE_NAME = "video0"


def clean_up(video: cv2.VideoCapture, esp_cams: List[ESP32_CAM]) -> None:
    """
    Takes care of cleaning up the VideoCapture instance and releasing any
    memory held by OpenCV
    """
    global IS_RUNNING

    logging.info('Cleaning up')
    IS_RUNNING = False
    video.release()
    cv2.destroyAllWindows()

    for esp_cam in esp_cams:
        logging.info(f'Stopping ESP Camera {esp_cam._device_ip}')
        esp_cam.stop()

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

def capture_images(video: cv2.VideoCapture, esp_cams: List[ESP32_CAM]) -> Generator[image_classification_pb2.ClassifyImageRequest, None, None]:
    """
        Capture images from the variuos video input devices, generating gRPC request stream messages.

        Args:
            video: VideoCapture instance to real camera hardware.
            esp_cams: List of known ESP32 cameras to query image from.

        Returns:
            Image classify request streams.
    """
    # Captured images are tuples of the device name and the ndarray that was captured.
    captured_images: List[Tuple[str, numpy.ndarray]] = []

    # Capture frame from the connected video device.
    ret, frame = video.read()
    if ret != True:
        logging.error('[loop] ERROR: Run loop failed to read video frame')
    else:
        captured_images.append((VIDEO0_DEVICE_NAME, frame))

    # Query frame from ESP32 devices.
    for esp_cam in esp_cams:
        # Grab
        if esp_cam.is_running():
            img = esp_cam.get_image(blocking=False)
            esp_cam.clear_images()
            if img:
                # Convert requested image to an ndarray (numpy) of which the DNN can understand.
                try:
                    _img = Image.open(BytesIO(img))
                    _img = _img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                    nd_array = numpy.asarray(_img)
                    logging.debug(f'[loop] ESP Device {esp_cam._device_ip} captured image -> {nd_array.shape}')
                    captured_images.append((esp_cam._device_ip, nd_array))
                except UnidentifiedImageError:
                    logging.error(f'[loop] Failed interpret image from ESP Device {esp_cam._device_ip}')
                except Exception as e:
                    logging.error(f'[loop] Unknown exception while interpreting image from ESP Device {esp_cam._device_ip}: {e}')

        # Retry the connection if the device is not running.
        else:
            try:
                esp_cam.start()
            except Exception:
                logging.error(f'[loop] Failed to start ESP Cam connection with {esp_cam._device_ip}')

    # Create a generator for the captured images.
    for device_name, frame in captured_images:
        yield image_classification_pb2.ClassifyImageRequest(
            image=pickle.dumps(frame),
            device=device_name,
        )

def start_loop(video: cv2.VideoCapture, esp_cams: List[ESP32_CAM], cooldown: int) -> int:
    """
    Run loop which starts reading frames from the video and invoke the
    dnn model through a gRPC server on what's being presented in each frame.

    Args:
        video (VideoCapture): OpenCV VideoCapture instance to read frames from.
        esp_cams (List[ESP32_CAM]): List of ESP32 camera instances to query from.
        cooldown (int): Cooldown in minutes between notifying the user of a
            image detection.
    """
    # Setup gRPC client with the classification server.
    # A helper function in which can be invoked to re-establish the channel upon a disconnect.
    def _setup_grpc_client() -> ImageClassifierStub:
        channel = grpc.insecure_channel(f'{CLASSIFY_HOST}:{CLASSIFY_PORT}')
        classify_client: ImageClassifierStub = ImageClassifierStub(channel)
        return classify_client

    # Checks the results and issues notification to the user.
    def _check_and_notify_dection(device_name: str, matches: List[int], match_scores: List[float], frame: numpy.ndarray, last_notified: datetime) -> datetime:
        # Check if HOOOMAN was detected.
        if 1 in matches:
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
        **{ f'{dev.device_name}': None for dev in esp_cams },
    }

    classify_client = _setup_grpc_client()
    global IS_RUNNING
    while IS_RUNNING:
        # Lets detect captured images.
        try:
            for classfied_image in classify_client.ClassifyImage(capture_images(video, esp_cams)):
                # classfied_image.matches
                logging.debug(f'[loop] Match results from {classfied_image.device} = {classfied_image.matches}')

                # Extract the ndarray frame from the response.
                frame = pickle.loads(classfied_image.image)

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
    return parser.parse_args()

def signal_safe_exit(video: cv2.VideoCapture, esp_cams: List[ESP32_CAM]):
    def _sig_helper_func(signal, frame):
        logging.info('[Signal] Interrupt detected: Exiting safely')
        clean_up(video, esp_cams)
    return _sig_helper_func

def main():
    # Parse those args!
    args = parse_args()

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

    # Register ESP Cameras.
    esp_cams = []
    if not args.skip_esp:
        esp_cams = [
            ESP32_CAM(dev_ip, dev_port, device_name=dev_name)
            for dev_name, dev_ip, dev_port in ESP_DEVICES
        ]
        for esp_cam in esp_cams:
            try:
                esp_cam.start()
            except Exception:
                logging.error(f"[main] Failed to start ESP Cam connection with {esp_cam._device_ip}")

    # Register signal handler for safely exiting on interrupt.
    signal.signal(signal.SIGINT, signal_safe_exit(video_capture, esp_cams))

    # Start the run loop.
    logging.info('[main] Starting detection loop')
    loop_status = False
    try:
        loop_status = start_loop(
            video_capture,
            esp_cams,
            args.cooldown,
        )
        logging.info(f'[main] Run loop status returned {loop_status}')
    except Exception as e:
        logging.error(f'[main] Unexpected exception: {e}')
        traceback.print_exception(type(e), e, e.__traceback__)
    finally:
        clean_up(video_capture, esp_cams)
    return loop_status

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ret = main()
    exit(ret)
