#!/usr/bin/env python3
# The server consumes images produced by the client, which carries the task
# of classifying the image taken, and responds only with the classified image
# which contins a HOOMAN.
import argparse
import logging
import pickle
from concurrent import futures
from typing import List, Tuple

import cv2
import grpc
import numpy

import image_classification_pb2
import image_classification_pb2_grpc

# DNN model paths.
MODEL_WEIGHTS_PATH = 'models/SSD_MobileNet_v3/frozen_inference_graph.pb'
MODEL_CONFIG_PATH = 'models/SSD_MobileNet_v3/graph.pbtxt'
from common import IMAGE_HEIGHT, IMAGE_WIDTH, LABELS_MP

# Server configs.
PORT=6969

# Configure logging.
logging.basicConfig(level=logging.INFO)


class ImageClassifierServer(image_classification_pb2_grpc.ImageClassifierServicer):
    model: cv2.dnn.Net
    threshold: int

    def __init__(self, threshold: int) -> None:
        # Load & configure the DNN model.
        logging.debug(f"Loading model from {MODEL_CONFIG_PATH} | {MODEL_WEIGHTS_PATH}")
        self.model = cv2.dnn_DetectionModel(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
        self.model.setInputSize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.model.setInputScale(1.0 / 127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)

        self.threshold = threshold

        super().__init__()

    @staticmethod
    def detect(frame: numpy.ndarray, model: cv2.dnn_DetectionModel, score_thresh: int) -> Tuple[List[int], List[float]]:
        """
        NOTE: I think 1 refers to human.
        Run the DNN model on the frame, detecting objects.

        Args:
            frame: Captured frame to run through dnn.
            model: The DNN instance.
            score_thresh: The score's minimum threshold.
        Returns:
            A tuple containing a list of match ids and a list of their corresponding confidence scores.
        """
        # Apply the image through the DNN.
        logging.debug("Applying image through the DNN")
        classes, confidences, boxes = model.detect(frame, confThreshold=score_thresh)

        # Early return on no results.
        matches = []
        scores = []
        if len(classes) == 0:
            return (matches, scores,)

        # Label results
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            className = LABELS_MP[f'{classId}']
            logging.info(f'Detection score: classId={classId} | className={className} | confidence={confidence}')
            matches.append(classId)
            scores.append(confidence)
            cv2.rectangle(frame, box, color=(0, 255, 0))

        return (matches, scores,)

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
            matches, scores = self.detect(frame, self.model, self.threshold)
            logging.info(f'Image for device={request.device} matched={len(matches)}')

            # Encode the new frame with a potentially outlined match(es).
            encoded_frame = pickle.dumps(frame)

            # Return result.
            yield image_classification_pb2.ClassifyImageResponse(
                matches=list(matches),
                image=encoded_frame,
                device=request.device,
                match_scores=list(scores),
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

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", IMAGE_WIDTH*IMAGE_HEIGHT*20),
            ("grpc.max_send_message_length", IMAGE_WIDTH*IMAGE_HEIGHT*20),
        ]
    )
    image_classification_pb2_grpc.add_ImageClassifierServicer_to_server(ImageClassifierServer(
        threshold=args.threshold,
    ), server)
    server.add_insecure_port(f'[::]:{PORT}')
    print(f'Starting server on :{PORT}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    main()