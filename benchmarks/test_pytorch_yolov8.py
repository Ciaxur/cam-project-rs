from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import intel_extension_for_pytorch as ipex
import torch
import time
from PIL import Image
import os
from urllib.request import urlopen
from typing import List
from random import randrange
import numpy as np
import cv2

# TEST:
import os
args = os.sys.argv
if len(args) < 2:
    raise Exception("Expected path to image as an arg")
img_path = args[-1]

# Colors.
COLORS = [
  (19,  103, 138),  # Blue
  (154, 235, 163),  # Green
  (255, 107, 26),   # Orange
  (255, 0, 0),      # Red
  (0, 255, 0),      # Green
  (0, 0, 255),      # Blue
  (0, 255, 255),    # Cyan
  (255, 0, 255),    # Magenta
  (255, 255, 0),    # Yellow
  (0, 0, 0),        # Black
  (255, 255, 255),  # White
  (128, 0, 0),      # Dark Red
  (0, 128, 0),      # Dark Green
  (0, 0, 128),      # Dark Blue
  (192, 192, 192),  # Light Gray
  (255, 165, 0),    # Orange
  (165, 42, 42),    # Brown
]

def draw_bbox(img: Image, boxes: Boxes, results: Results) -> Image:
  """
    Helper function that applies a bounding box and labels onto the given image
    and result of object detection on that image.

    Args:
      img: Pillow Image instance.
      boxes: Resulting boxes instance being drawn on image.
    
    Returns:
      Modified Pillow image instance with bbox and labels.
  """
  img_np = np.array(img)
  
  for box in boxes:
    bbox = box.xyxy[0]
    class_id = box.cls.item()
    label = results.names[class_id]
    score = box.conf.item()
    color = COLORS[ randrange(0, len(COLORS)) ]

    # Extract coordinates as ints
    xmin, ymin = int(bbox[0]), int(bbox[1]), 
    xmax, ymax = int(bbox[2]), int(bbox[3])

    # Draw bounding box
    cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), color, 1)

    # Display label and score
    label_text = f"{label}: {score:.2f}"
    cv2.putText(img_np, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

  return Image.fromarray(img_np)


# Verify Intel DG2 extention works.
print(f'PyTorch Version: {torch.__version__}')
#print(f'Intel PyTorch Extension Version: {ipex.__version__}')

# List the available XPu Devices
#xpu_devices = ipex.xpu.device_count()
#assert xpu_devices > 0, 'Expected an available XPU device'
#print('Available XPU Devices:')
#for i in range(xpu_devices):
#  print(f"  - Device {i}: {ipex.xpu.get_device_name(i)}")
#print()


# Construct an iterable of test images.
test_urls = [
  ('beignets', 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'),
  ('keanu', 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F4.bp.blogspot.com%2F-1SBVTLdwHqc%2FVTutdWprmXI%2FAAAAAAAAbhw%2F-i_dV2h-aPQ%2Fs1600%2Fkeanu-reeves-birthday.jpg&f=1&nofb=1&ipt=6c5096ad424a3d3aad4f6473e301ca6976ef2efac1f333387f80794c4beb112c&ipo=images'),
  ('mario', 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fvignette.wikia.nocookie.net%2Fmario%2Fimages%2F7%2F75%2FMario.png%2Frevision%2Flatest%3Fcb%3D20170604110020%26path-prefix%3Dda&f=1&nofb=1&ipt=7efdec9a625c5440c966d14a7f46dad39b6fc80ba724021553bd9127c547d0a3&ipo=images'),
]
# TEST:
test_urls = [
  ('apt-test', img_path,)
]

def image_iter(urls: list[str]) -> iter:
  for (name, url) in urls:
    img = Image.open(url)
    #img = Image.open(urlopen(url))
    img = img.convert('RGB')
    yield (name, img)

# Load in Yolov8.
yolov8_model_filepath = 'models/yolov8/yolov8n.pt'
if not os.path.exists(yolov8_model_filepath):
  raise FileExistsError(f'File {yolov8_model_filepath} does not exist')

model = YOLO(yolov8_model_filepath)
print('Model Device set to -> ', model.device)

# Test
print('Running test...')
for (name, img) in image_iter(test_urls):
  t0 = time.time()
  with torch.no_grad():
    # https://docs.ultralytics.com/modes/predict/#inference-arguments
    res: List[Results] = model.predict(img, verbose=False)#, conf=0.75)
  t1 = time.time()
  dt = t1 - t0
  print(f'Model classification took {dt}s for {name}')

  # Extract predictions.
  for pred in res:
    print()
    print('Total Boxes -> ', len(pred.boxes))
    classes = []
    labels = []
    scores = []

    # Populate prediction info.
    for box in pred.boxes:
      class_id = box.cls.item()
      label = pred.names[class_id]
      score = box.conf.item()

      classes.append(class_id)
      labels.append(label)
      scores.append(score)

    print(f'classes[{classes}] | labels[{labels}] | scores[{scores}]')

    # Draw bboxes
    img_res = draw_bbox(img, pred.boxes, pred)
    print(f'Saving result for {name} -> /tmp/{name}.jpg')
    img_res.save(f'/tmp/{name}.jpg')

# Move model to evaluation mode and set the device in use
# to be the XPU.
print('\n\n')
#print('Using XPU Device [0]{}'.format(ipex.xpu.get_device_name(0)))
model.xpu('xpu:0')
print('Model Device set to -> ', model.device)

print('Optimizing mode to run on XPU')
model.training = False
#mode = ipex.optimize(model)


print('Running test with XPU optimizations...')
for (name, img) in image_iter(test_urls):
  t0 = time.time()
  with torch.no_grad():
      res = model.predict(img)
  t1 = time.time()
  dt = t1 - t0
  print(f'Model classification took {dt}s')
