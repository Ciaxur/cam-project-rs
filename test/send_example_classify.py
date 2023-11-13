#!/usr/bin/env python3
# This script is intended to be explicitly invoked with an exmaple image to verify
# the model works as expected.
import pickle
from typing import Generator

import cv2
import grpc
import numpy as np
import math

import image_classification_pb2
from image_classification_pb2_grpc import ImageClassifierStub

# Classification server configuration.
CLASSIFY_HOST="localhost"
CLASSIFY_PORT=6969

def main():
    channel = grpc.insecure_channel(
        f'{CLASSIFY_HOST}:{CLASSIFY_PORT}',
        # options=[
        #     ("grpc.max_receive_message_length", 1920*1080*69),
        #     ("grpc.max_send_message_length", 1920*1080*69),
        # ]
    )
    classify_client: ImageClassifierStub = ImageClassifierStub(channel)

    def _gen() -> Generator[image_classification_pb2.ClassifyImageRequest, None, None]:
        # Open example image & load as an ndarray.
        frame = cv2.imread('test.jpg')
        # frame = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_NEAREST)
        print('Loaded frame: ', frame.shape, frame.dtype)

        # # Crop the image to get the peep hole.
        # h, w = frame.shape[:2]
        # h_ajustment = math.floor(h * 0.69)
        # w_ajustment = math.floor(w * 0.75)
        # print(h_ajustment, w_ajustment)


        # x, y = 220, 180
        # new_h = h - h_ajustment
        # new_w = w - w_ajustment
        # img = frame[y:y+new_h, x:x+new_w]

        # Define the camera matrix and distortion coefficients
        # K = np.array([[600, 0, new_h], [0, 600, new_h], [0, 0, 1]])
        # D = np.array([0.1, 0.1, 0.1, 0.0])
        # img = cv2.undistort(img, K, D)

        # cv2.imwrite('result.jpg', img)

        yield image_classification_pb2.ClassifyImageRequest(
            image=pickle.dumps(frame),
            device='POTATO:192.168.0.15',
        )

    for resp in classify_client.ClassifyImage(_gen()):
        frame = pickle.loads(resp.image)
        print('Matches:', resp.matches)
        print('Scores:', resp.match_scores)
        print('Saved result to result.jpg')
        cv2.imwrite('result.jpg', frame)


if __name__ == '__main__':
    main()