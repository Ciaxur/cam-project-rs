#!/usr/bin/env python3
# The server consumes images produced by the client, which carries the task
# of classifying the image taken, and responds only with the classified image
# which contins a HOOMAN.
import argparse
import logging
import os
import queue
import signal
import time
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from threading import Thread
from typing import List, Optional, Tuple

import cv2
import grpc
import image_classification_pb2
import image_classification_pb2_grpc
import numpy
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

# Common model configs
from common import IMAGE_HEIGHT, IMAGE_WIDTH, get_random_color

# Server configs.
DEFAULT_PORT=6969
IMAGE_WRITER_TIMEOUT = 2
YOLOV8_MODEL_FILEPATH = 'models/yolov8/yolov8n.pt'


@dataclass
class ClassifiedImage:
    image: numpy.ndarray
    name: str

class ImageClassifierServer(image_classification_pb2_grpc.ImageClassifierServicer):
    model: YOLO
    threshold: int

    # Image store settings.
    image_store_active: bool
    image_store_t: Optional[Thread]
    image_store_queue: queue.Queue[ClassifiedImage]
    image_store_dir: str

    def __init__(self, threshold: int, image_store_dir: Optional[str] = None) -> None:
        # Load in Yolov8.
        if not os.path.exists(YOLOV8_MODEL_FILEPATH):
            raise FileExistsError(f'File {YOLOV8_MODEL_FILEPATH} does not exist')

        self.model = YOLO(YOLOV8_MODEL_FILEPATH)
        self.threshold = threshold
        logging.info(f'Model Device set to -> {self.model.device}')


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

    def detect(self, np_image: numpy.ndarray) -> Tuple[List[str], List[int], List[float], numpy.array]:
        """
            Run the image classifier model on the frame, detecting objects.
            This image classifier does not include a bbox.

            Args:
                np_image: Captured frame to run through the model.
            Returns:
                A tuple containing a list of labels, match ids and a list of their corresponding confidence scores
                and a modified image with bbox.
        """
        # Apply the image through the pipeline
        img = Image.fromarray(np_image)

        t0 = time.time()
        with torch.no_grad():
            # https://docs.ultralytics.com/modes/predict/#inference-arguments
            results: List[Results] = self.model.predict(img, verbose=False)
        t1 = time.time()
        dt = t1 - t0
        logging.info(f'Model object detection took {dt}s with {len(results)} detections')


        # Early return on no results.
        classes = []
        labels = []
        scores = []
        if len(results) == 0:
            return (labels, classes, scores, np_image)

        # Populate prediction info.
        for result in results:
            # Draw bbox around the image.
            np_image: numpy.array = self.draw_bbox(np_image, result.boxes, result)

            # Extract features.
            for box in result.boxes:
                class_id = box.cls.item()
                label = result.names[class_id]
                score = box.conf.item()

                # Filter classifications based on score.
                if score < self.threshold:
                    continue

                logging.info(f"Detected object above {self.threshold} threshold -> label=[{label}] | score={score} | class={class_id}")
                classes.append(class_id)
                labels.append(label)
                scores.append(score)

        return (labels, classes, scores, np_image)

    @staticmethod
    def draw_bbox(img_np: numpy.array, boxes: Boxes, prediction: Results) -> numpy.array:
        """
            Helper function that applies a bounding box and labels onto the given image
            and result of object detection on that image.

            Args:
                img: Pillow Image instance.
                boxes: Resulting boxes instance being drawn on image.
                prediction: Results instance.

            Returns:
                Modified Pillow image instance with bbox and labels.
        """
        for box in boxes:
            bbox = box.xyxy[0]
            class_id = box.cls.item()
            score = box.conf.item()
            label = prediction.names[class_id]
            color = get_random_color()

            # Extract coordinates as ints
            xmin, ymin = int(bbox[0]), int(bbox[1]),
            xmax, ymax = int(bbox[2]), int(bbox[3])

            # Draw bounding box
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), color, 1)

            # Display label and score
            label_text = f"{label}: {score:.2f}"
            cv2.putText(img_np, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img_np

    @staticmethod
    def parse_image_to_nparray(image: bytes) -> numpy.ndarray:
        """
            This funciton consumes the image, returning an ndarray representation of the
            parsed image.

            Args:
                image: Bytes image obtained from the classification request.

            Returns:
                An ndarray representation of the parsed image.
        """
        # images within the requests are expected to be raw jpeg images.
        frame = Image.open(BytesIO(image))
        img_np = numpy.asarray(frame)
        logging.info("Converted raw image byte to numpy array -> {}".format(img_np.shape))
        return img_np

    @staticmethod
    def convert_numpy_to_jpeg_bytes(image_np: numpy.ndarray) -> bytes:
        """
            Helper function which supports converting a given numpy array to jpeg bytes array.

            Args:
                image_np: Image in a numpy array.

            Returns:
                Bytes array encoded as a jpeg image.
        """
        # Parse the numpy array into a PIL Image, saving that image into an in-memory buffer.
        buffer = BytesIO()
        Image.fromarray(image_np).save(buffer, format='jpeg')
        return buffer.getvalue()

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
            frame_np: numpy.ndarray = self.parse_image_to_nparray(request.image)

            # Classify the image.
            labels, matches, scores, frame_np = self.detect(frame_np)
            logging.info(f'Image for device={request.device} matched={len(labels)}')

            # Encode the new frame as an encoded image byte array.
            logging.info(f"Serializing classified image for {request.device} as an image byte array")
            serlialized_frame: bytes = self.convert_numpy_to_jpeg_bytes(frame_np)

            # Offload storing the classified image to a thread, if one's active.
            if self.image_store_active:
                self.image_store_queue.put(ClassifiedImage(
                    image=frame_np,
                    name=request.device,
                ))

            # Return result.
            yield image_classification_pb2.ClassifyImageResponse(
                labels=labels,
                matches=list(matches),
                match_scores=list(scores),
                image=serlialized_frame,
                device=request.device,
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
