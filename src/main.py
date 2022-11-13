import argparse
import base64
import http.client
import json
import logging
import signal
import ssl
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import List, Tuple

import cv2
import numpy
from PIL import Image, UnidentifiedImageError

from VideoDevice import ESP32_CAM

# Relative path to the model and configs, from which the script was invoked.
MODEL_WEIGHTS_PATH = 'models/SSD_MobileNet_v2/frozen_inference_graph.pb'
MODEL_CONFIG_PATH = 'models/SSD_MobileNet_v2/graph.pbtxt'
IMAGE_STORAGE_PATH = '/mnt/apt_cam_captures'
API_HOST = 'localhost'
API_PORT = 3000
MESSAGE_ENDPOINT = '/telegram/message'
CLIENT_CERT_PATH = 'certs/cam-client1.crt'
CLIENT_KEY_PATH = 'certs/cam-client1.key'

# Classification configuration.
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
IS_RUNNING = True


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

def detect(frame, model: cv2.dnn.Net, score_thresh: int) -> list:
    """
    NOTE: I think 1 refers to human.
    Run the DNN model on the frame, detecting objects.

    Args:
        frame: Captured frame to run through dnn.
        model: The DNN instance.
        score_thresh: The score's minimum threshold.
    Returns:
        A list of detected objects.
    """
    rows = frame.shape[0]
    cols = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, size=(rows,cols), swapRB=True, crop=True)

    # Apply the image through the DNN.
    model.setInput(blob)
    cvOut = model.forward()

    # Iterate through the results, checking if there was a match and
    # apply the matches to the image.
    matches = []
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        w = detection[1]
        if score > score_thresh:
            logging.info(f'Detected: {w}')
            matches.append(w)
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    return matches

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

def start_loop(video: cv2.VideoCapture, esp_cams: List[ESP32_CAM], model: cv2.dnn.Net, score_thresh: float, cooldown: int) -> int:
    """
    Run loop which starts reading frames from the video and invoke the
    dnn model on what's being presented in each frame.

    Args:
        video (VideoCapture): OpenCV VideoCapture instance to read frames from.
        esp_cams (List[ESP32_CAM]): List of ESP32 camera instances to query from.
        model (Net): Deep Neural Network model instance that will be used to
            run on each frame.
        score_thresh (float): Image detection score threshold (0.0-1.0).
        cooldown (int): Cooldown in minutes between notifying the user of a
            image detection.
    """
    def _check_and_notify_dection(device_name: str, frame: numpy.ndarray, last_notified: datetime) -> datetime:
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
                send_message('[Device={}] Hoooman detected at {}.'.format(
                    device_name,
                    str(now),
                ), b64_image)
                return now
    
    last_notified = None
    global IS_RUNNING
    while IS_RUNNING:
        # Captured images are tuples of the device name and the ndarray that was captured.
        captured_images: List[Tuple[str, numpy.ndarray]] = []
        
        # Capture frame from the connected video device (video0).
        ret, frame = video.read()
        if ret != True:
            logging.error('[loop] ERROR: Run loop failed to read video frame')
        else:
            captured_images.append(('video0', frame))

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

            # Retry the connection if the device is not running.
            else:
                try:
                    esp_cam.start()
                except Exception:
                    logging.error(f'[loop] Failed to start ESP Cam connection with {esp_cam._device_ip}')

        # Lets detect captured images.
        logging.info(f'[loop] Running DNN on {len(captured_images)} captured images')
        for device_name, frame in captured_images:
            matches = detect(frame, model, score_thresh)
            logging.debug(f'[loop] Match results from {device_name} = {matches}')

            notified = _check_and_notify_dection(device_name, frame, last_notified)
            if notified:
                last_notified = notified
        
        sleep(0.5)

def range_limited_float_type(min: float, max: float):
    """ Type function for argparse - a float within some predefined bounds """
    def _inner(arg):
        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("Must be a floating point number")
        if f < min or f > max:
            raise argparse.ArgumentTypeError("Argument must be < " + str(max) + "and > " + str(min))
        return f
    return _inner

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start detecting those HOOMANS")
    parser.add_argument(
        '--threshold', 
        '-t',
        type=range_limited_float_type(0.0, 1.0),
        default=0.2,
        help='Matching score threshold'
    )
    parser.add_argument(
        '--cooldown', 
        '-c',
        type=int,
        default=5,
        help='Cooldown in minutes between notifying the user of detected image.'
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
    video_file = "/dev/video0"
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
    esp_cams = [
        ESP32_CAM("192.168.0.9", 3000),
    ]
    for esp_cam in esp_cams:
        try:
            esp_cam.start()
        except Exception:
            logging.error(f"[main] Failed to start ESP Cam connection with {esp_cam._device_ip}")

    # Load model.
    cvNet = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
    
    # Register signal handler for safely exiting on interrupt.
    signal.signal(signal.SIGINT, signal_safe_exit(video_capture, esp_cams))
    
    # Start the run loop.
    logging.info('[main] Starting detection loop')
    loop_status = False
    try:
        loop_status = start_loop(
            video_capture, 
            esp_cams,
            cvNet,
            args.threshold,
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
