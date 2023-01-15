#!/usr/bin/env python3
# The server consumes images produced by the client, which carries the task
# of classifying the image taken, and responds only with the classified image
# which contins a HOOMAN.
import argparse
import logging
import pickle
from concurrent import futures

import cv2
import grpc
import numpy

import image_classification_pb2
import image_classification_pb2_grpc

# DNN model paths.
MODEL_WEIGHTS_PATH = 'models/SSD_MobileNet_v3/frozen_inference_graph.pb'
MODEL_CONFIG_PATH = 'models/SSD_MobileNet_v3/graph.pbtxt'

# Server configs.
PORT=6969

# Configure logging.
logging.basicConfig(level=logging.INFO)


class ImageClassifierServer(image_classification_pb2_grpc.ImageClassifierServicer):
    model: cv2.dnn.Net
    threshold: int

    def __init__(self, threshold: int) -> None:
        # Load up the DNN model.
        logging.debug(f"Loading model from {MODEL_CONFIG_PATH} | {MODEL_WEIGHTS_PATH}")
        self.model = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
        self.threshold = threshold

        super().__init__()

    @staticmethod
    def detect(frame: numpy.ndarray, model: cv2.dnn.Net, score_thresh: int) -> list:
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
        logging.debug(f"Detecting image of shape rows={rows} cols={cols}")

        # Apply the image through the DNN.
        logging.debug("Applying image through the DNN")
        model.setInput(blob)
        cvOut = model.forward()

        # Iterate through the results, checking if there was a match and
        # apply the matches to the image.
        matches = []
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            logging.debug(f"Resolved to a detection score of {score}")
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

    def ClassifyImage(self, request_iterator: image_classification_pb2.ClassifyImageRequest, context):
        """
            Classifies the request image, yielding matches found within the image.

            Args:
            request_iterator: A stream of classification requests.
            context: gRPC request context.

            Returns:
            - Empty message if there were no matches found.
            - List of matches and the corresponding image with an outline around the matches.
        """
        for request in request_iterator:
            # Extract the image into an ndarray which the model would understand.
            frame = pickle.loads(request.image)

            # Classify the image.
            matches: numpy.array = self.detect(frame, self.model, self.threshold)
            logging.info(f'Image for device={request.device} matched={len(matches)}')
            logging.debug(f'Matches={list(matches)}')

            # Encode the new frame with a potentially outlined match(es).
            encoded_frame = pickle.dumps(frame)

            # Return result.
            yield image_classification_pb2.ClassifyImageResponse(
                matches=list(matches),
                image=encoded_frame,
                device=request.device,
            )


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

def main():
    args = parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    image_classification_pb2_grpc.add_ImageClassifierServicer_to_server(ImageClassifierServer(
        threshold=args.threshold,
    ), server)
    server.add_insecure_port(f'[::]:{PORT}')
    print(f'Starting server on :{PORT}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    main()