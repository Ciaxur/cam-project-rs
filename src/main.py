import argparse
import base64
import http.client
import json
import logging
import signal
import ssl
from datetime import datetime
from pathlib import Path
from time import sleep

import cv2

# Relative path to the model and configs, from which the script was invoked.
MODEL_WEIGHTS_PATH = 'models/SSD_MobileNet_v2/frozen_inference_graph.pb'
MODEL_CONFIG_PATH = 'models/SSD_MobileNet_v2/graph.pbtxt'
IMAGE_STORAGE_PATH = '/mnt/apt_cam_captures'
API_HOST = 'localhost'
API_PORT = 3000
MESSAGE_ENDPOINT = '/telegram/message'
CLIENT_CERT_PATH = 'certs/cam-client1.crt'
CLIENT_KEY_PATH = 'certs/cam-client1.key'


def clean_up(video: cv2.VideoCapture) -> None:
    """
    Takes care of cleaning up the VideoCapture instance and releasing any
    memory held by OpenCV
    """
    logging.info('Cleaning up')
    video.release()
    cv2.destroyAllWindows()

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

def start_loop(video: cv2.VideoCapture, model: cv2.dnn.Net, score_thresh: float, cooldown: int) -> int:
    """
    Run loop which starts reading frames from the video and invoke the
    dnn model on what's being presented in each frame.

    Args:
        video (VideoCapture): OpenCV VideoCapture instance to read frames from.
        model (Net): Deep Neural Network model instance that will be used to
            run on each frame.
        score_thresh (float): Image detection score threshold (0.0-1.0).
        cooldown (int): Cooldown in minutes between notifying the user of a
            image detection.
    """
    last_notified = None
    while True:
        ret, frame = video.read()

        if ret != True:
            logging.error('[Loop] ERROR: Run loop failed to read video frame')
            return ret

        # Lets detect that frame!
        logging.debug('[Loop] Detecting frame')
        matches = detect(frame, model, score_thresh)
        logging.debug(f'[Loop] Resulting matches = {matches}')

        # Check if HOOOMAN was detected.
        if 1 in matches:
            # Convert image to a blob that can be send in a post request
            # to 4bit.api on the /message endpoint.
            now = datetime.now()
            img_path = Path(IMAGE_STORAGE_PATH, 'hooman-{}.jpg'.format(str(now)))
            cv2.imwrite(str(img_path), frame)
            raw_image = cv2.imencode('.jpg', frame)[1]
            logging.info(f'[Loop] Hooman detected! Saving to {img_path}')

            # Have a cooldown for invoking the endpoint. However, save
            # those images to the external flash drive with timestamps.
            cooldown_s = cooldown * 60
            if last_notified == None or (now - last_notified).seconds >= cooldown_s:
                last_notified = now
                b64_image = base64.b64encode(raw_image)
                send_message('Hoooman detected at {}.'.format(
                    str(now)
                ), b64_image)
        
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

def signal_safe_exit(video: cv2.VideoCapture):
    def _sig_helper_func(signal, frame):
        logging.info('[Signal] Interrupt detected: Exiting safely')
        clean_up(video)
    return _sig_helper_func

def main():
    # Parse those args!
    args = parse_args()

    # Open and check that we can read from the video camera.
    video_file = "/dev/video0"
    video_capture = cv2.VideoCapture(video_file)

    # Register signal handler for safely exiting on interrupt.
    signal.signal(signal.SIGINT, signal_safe_exit(video_capture))

    if not video_capture.isOpened():
        logging.error(f'[main] {video_file} failed to open.')
        clean_up(video_capture)
        return 1

    # Adjust frame resolution.
    video_capture.set(cv2.CAP_PROP_FPS, 1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    # Load model.
    cvNet = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
    
    # Start the run loop.
    logging.info('[main] Starting detection loop')
    loop_status = start_loop(
        video_capture, 
        cvNet,
        args.threshold,
        args.cooldown,
    )
    logging.info(f'[main] Run loop status returned {loop_status}')
    return loop_status

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ret = main()
    exit(ret)
