import argparse
import cv2
import signal

# Relative path to the model and configs, from which the script was invoked.
MODEL_WEIGHTS_PATH = 'models/SSD_MobileNet_v2/frozen_inference_graph.pb'
MODEL_CONFIG_PATH = 'models/SSD_MobileNet_v2/graph.pbtxt'

def clean_up(video: cv2.VideoCapture) -> None:
    """
    Takes care of cleaning up the VideoCapture instance and releasing any
    memory held by OpenCV
    """
    print('Cleaning up...')
    video.release()
    cv2.destroyAllWindows()

def detect(frame, model: cv2.dnn.Net, score_thresh: int) -> list:
    """
    NOTE: I think 72 refers to human.
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
            print('Detected: ', w)
            matches.append(w)
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    return matches

def start_loop(video: cv2.VideoCapture, model: cv2.dnn.Net, score_thresh: int) -> int:
    """
    Run loop which starts reading frames from the video and invoke the
    dnn model on what's being presented in each frame.

    Args:
        video (VideoCapture): OpenCV VideoCapture instance to read frames from.
        model (Net): Deep Neural Network model instance that will be used to
            run on each frame.
    """
    i = 0
    while i < 5:
        ret, frame = video.read()

        if ret != True:
            print('ERROR: Run loop failed to read video frame.')
            return ret

        # Lets detect that frame!
        print('Detecting frame.')
        matches = detect(frame, model, score_thresh)
        print(f'Resulting matches = {matches}.')

        # Check if HOOOMAN was detected.
        if 72 in matches:
            print('Detected a HOOOOOMAN!')
            cv2.imwrite(f'out-dnned_{i}.jpg', frame)
            i += 1

    return 0

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
    return parser.parse_args()

def signal_safe_exit(video: cv2.VideoCapture):
    def _sig_helper_func(signal, frame):
        print('Interrupt detected: Exiting safely.')
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
        print(f'{video_file} failed to open.')
        clean_up(video_capture)
        return 1

    # Adjust frame resolution.
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    # Load model.
    cvNet = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
    
    # Start the run loop.
    loop_status = start_loop(
        video_capture, 
        cvNet,
        args.threshold,
    )
    print('Run loop returned ', loop_status)
    return loop_status

if __name__ == '__main__':
    ret = main()
    exit(ret)
