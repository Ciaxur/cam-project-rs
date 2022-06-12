import cv2

IMAGE_OUT = 'out.jpg'

def clean_up(video: cv2.VideoCapture) -> None:
    """
    Takes care of cleaning up the VideoCapture instance and releasing any
    memory held by OpenCV
    """
    print('Cleaning up...')
    video.release()
    cv2.destroyAllWindows()


def main():
    # Open and check that we can read from the video camera.
    video_file = "/dev/video0"
    video_capture = cv2.VideoCapture(video_file)

    if not video_capture.isOpened():
        print(f'{video_file} failed to open.')
        clean_up(video_capture)
        return 1

    # Adjust frame resolution.
    video_capture.set(cv2.CAP_PROP_FPS, 5)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Capture one frame.
    ret, frame = video_capture.read()
    if not ret:
        print('Failed to capture video :)')
        clean_up(video_capture)
        return 1
    # Save captured frame.
    ret = cv2.imwrite(IMAGE_OUT, frame)
    if not ret:
        print(f'Failed to write image to {IMAGE_OUT}.')
        clean_up(video_capture)
        return 1
    print(f'Successfuly save image to {IMAGE_OUT}')
    clean_up(video_capture)
    return 0

if __name__ == '__main__':
    ret = main()
    exit(ret)



