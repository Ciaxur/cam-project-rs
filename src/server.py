#!/usr/bin/env python3
# The server consumes images produced by the client, which carries the task
# of classifying the image taken, and responds only with the classified image
# which contins a HOOMAN.
import argparse
import logging
import os
import pickle
import queue
import signal
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import List, Optional, Tuple

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
DEFAULT_PORT=6969
IMAGE_WRITER_TIMEOUT = 2



@dataclass
class ClassifiedImage:
    image: numpy.ndarray
    name: str

class ImageClassifierServer(image_classification_pb2_grpc.ImageClassifierServicer):
    model: cv2.dnn.Net
    threshold: int

    # Image store settings.
    image_store_active: bool
    image_store_t: Optional[Thread]
    image_store_queue: queue.Queue[ClassifiedImage]
    image_store_dir: str

    def __init__(self, threshold: int, image_store_dir: Optional[str] = None) -> None:
        # Load & configure the DNN model.
        logging.debug(f"Loading model from {MODEL_CONFIG_PATH} | {MODEL_WEIGHTS_PATH}")
        self.model = cv2.dnn_DetectionModel(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
        self.model.setInputSize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.model.setInputScale(1.0 / 127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)

        self.threshold = threshold

        # Spin up a thread to consume images to be stored in a given directory.
        self.image_store_queue = queue.Queue(2048)
        if image_store_dir:
            self.image_store_active = True
            self.image_store_dir = image_store_dir
            self.image_store_t = Thread(target=self.image_writer_thread)
            self.image_store_t.start()
            self.setup_sigint()
        else:
            self.image_store_active = False


        super().__init__()

    def image_writer_thread(self):
        """
            Function intended to run within a thread, which handles queued up images ready to
            be written to a file in a directory.
        """
        logging.info('Starting up image writer thread')
        while self.image_store_active:
            # Consume image from queue with a timeout in order to check closing this thread.
            try:
                image = self.image_store_queue.get(timeout=IMAGE_WRITER_TIMEOUT)
            except queue.Empty:
                logging.debug('Image store thread timed out: Image queue empty')
                continue

            now = datetime.now()
            filename = f'{now}-{image.name}.jpg'

            # Create a sub-directory to organize image captures by date.
            image_subdir = datetime.now().strftime("%d-%m-%Y")
            image_dir = f'{self.image_store_dir}/{image_subdir}'
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            cv2.imwrite(f'{image_dir}/{filename}', image.image)
        logging.info('Shutting down image writer thread')

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
            logging.debug(f'Recieved request for {request.device}. Classifying...')

            # Extract the image into an ndarray which the model would understand.
            frame: numpy.ndarray = pickle.loads(request.image)

            # Classify the image.
            matches, scores = self.detect(frame, self.model, self.threshold)
            logging.info(f'Image for device={request.device} matched={len(matches)}')

            # Encode the new frame with a potentially outlined match(es).
            encoded_frame = pickle.dumps(frame)

            # Offload storing the classified image to a thread, if one's active.
            if self.image_store_active:
                self.image_store_queue.put(ClassifiedImage(
                    image=frame,
                    name=request.device,
                ))

            # Return result.
            yield image_classification_pb2.ClassifyImageResponse(
                matches=list(matches),
                image=encoded_frame,
                device=request.device,
                match_scores=list(scores),
            )

    def setup_sigint(self):
        """Sets up SIGINT signal to clean things up."""
        def _handler(s, f):
            self.stop()
        signal.signal(signal.SIGINT, _handler)

    def stop(self):
        """Clean up."""
        if self.image_store_active and self.image_store_t:
            self.image_store_active = False
            self.image_store_t.join()
        exit(0)


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
        '--loglevel',
        type=str,
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--threshold',
        '-t',
        type=range_limited_float_type(0.0, 1.0),
        default=0.2,
        help='Matching score threshold'
    )
    parser.add_argument(
        '--image_store_dir',
        type=str,
        default=None,
        help='Path to a directory in which to store images to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help='Server port to listen on'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Configure logging.
    logging.basicConfig(level=getattr(logging, args.loglevel))

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ("grpc.max_receive_message_length", IMAGE_WIDTH*IMAGE_HEIGHT*20),
            ("grpc.max_send_message_length", IMAGE_WIDTH*IMAGE_HEIGHT*20),
        ]
    )
    image_classification_pb2_grpc.add_ImageClassifierServicer_to_server(ImageClassifierServer(
        threshold=args.threshold,
        image_store_dir=args.image_store_dir,
    ), server)
    server.add_insecure_port(f'[::]:{args.port}')
    logging.info(f'Starting server on :{args.port}')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    main()