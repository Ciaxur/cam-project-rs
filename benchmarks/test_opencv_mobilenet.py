from urllib.request import urlopen
import cv2
import numpy as np
from PIL import Image
import os
import time


# Loaded image.
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))

# Model configuration file.
# DNN model paths.
MODEL_PATH = "./models/SSD_MobileNet_v3"
MODEL_WEIGHTS_PATH = f'{MODEL_PATH}/frozen_inference_graph.pb'
MODEL_CONFIG_PATH = f'{MODEL_PATH}/graph.pbtxt'
IMAGE_HEIGHT, IMAGE_WIDTH = [300, 300]
SCORE_THRESHOLD = 0.7

# Verify models exist
for file in [MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH]:
  if not os.path.exists(file):
    raise Exception(f'{file} not found')


# Load thy model
print(f"Loading model from {MODEL_CONFIG_PATH} | {MODEL_WEIGHTS_PATH}")
model = cv2.dnn_DetectionModel(MODEL_WEIGHTS_PATH, MODEL_CONFIG_PATH)
model.setInputSize(IMAGE_WIDTH, IMAGE_HEIGHT)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Convert loaded image to a numpy array
image_np = np.array(img)
print(f'Image shape -> {image_np.shape}')

# Apply the image through the DNN.
print("Applying image through the DNN")
t0 = time.time()
classes, confidences, boxes = model.detect(image_np, confThreshold=SCORE_THRESHOLD)
t1 = time.time()
dt = t1 - t0
print(f'Model classification took {dt}s')

# Interpret results.
matches = []
scores = []

# Label results
for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
  matches.append(classId)
  scores.append(confidence)
print(f'Matches -> {matches}')
print(f'Scores -> {scores}')
