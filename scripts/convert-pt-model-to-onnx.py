# https://docs.ultralytics.com/integrations/onnx/#installation
import os
from ultralytics import YOLO
from typing import Optional
from PIL import Image

# Some image to test.
EXAMPLE_IMG_PATH = "keanu.jpg"
img: Optional[Image.Image] = None
if os.path.exists(EXAMPLE_IMG_PATH):
  img = Image.open(EXAMPLE_IMG_PATH)
  print(img)

PT_MODEL_PATH="models/yolov8/yolov8n.pt"
ONNX_MODEL_PATH="models/yolov8/yolov8n.onnx"

# Convert pytorch model to onnx using ultralytics' YOLO model.
model = YOLO(PT_MODEL_PATH)
model.export(format='onnx')

# Load the exported ONNX model
onnx_model = YOLO(ONNX_MODEL_PATH)
print('ONNX Model successfuly loaded')

if img:
  results = onnx_model.predict(img)
  print(len(results))
