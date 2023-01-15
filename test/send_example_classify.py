#!/usr/bin/env python3
# This script is intended to be explicitly invoked with an exmaple image to verify
# the model works as expected.
import pickle
from typing import Generator

import cv2
import grpc

import image_classification_pb2
from image_classification_pb2_grpc import ImageClassifierStub

# Classification server configuration.
CLASSIFY_HOST="4bit.local"
CLASSIFY_PORT=8000

def main():
    channel = grpc.insecure_channel(f'{CLASSIFY_HOST}:{CLASSIFY_PORT}')
    classify_client: ImageClassifierStub = ImageClassifierStub(channel)

    def _gen() -> Generator[image_classification_pb2.ClassifyImageRequest, None, None]:
        # Open example image & load as an ndarray.
        frame = cv2.imread('example.jpg')
        frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_NEAREST)
        print('Loaded frame: ', frame)

        yield image_classification_pb2.ClassifyImageRequest(
            image=pickle.dumps(frame),
            device='example_test',
        )

    for resp in classify_client.ClassifyImage(_gen()):
        frame = pickle.loads(resp.image)
        print('Matches:', resp.matches)
        print('Scores:', resp.match_scores)
        print('Saved result to result.jpg')
        cv2.imwrite('result.jpg', frame)


if __name__ == '__main__':
    main()