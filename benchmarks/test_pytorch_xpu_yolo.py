"""
    Exmaple use of Intel DG2 on an Object Detection model for which
    results in labeling an image and contains bounding boxes.

    The model used is https://huggingface.co/hustvl/yolos-tiny.
"""
from urllib.request import urlopen
from PIL import Image
from transformers import (
    # Object detection model.
    AutoModelForObjectDetection,
    YolosForObjectDetection,

    # Model Configuration.
    AutoConfig,
    YolosConfig,

    # Pre-Processors.
    AutoImageProcessor,
    YolosImageProcessor,

    # Pipeline.
    ObjectDetectionPipeline,
    pipeline,
)
from typing import List, Dict

import cv2
import numpy as np
import torch
import os
import intel_extension_for_pytorch as ipex
import time
from random import randrange

# Verify Intel DG2 extention works.
print(f'PyTorch Version: {torch.__version__}')
print(f'Intel PyTorch Extension Version: {ipex.__version__}')

# Construct an iterable of test images.
test_urls = [
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png',
    'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F4.bp.blogspot.com%2F-1SBVTLdwHqc%2FVTutdWprmXI%2FAAAAAAAAbhw%2F-i_dV2h-aPQ%2Fs1600%2Fkeanu-reeves-birthday.jpg&f=1&nofb=1&ipt=6c5096ad424a3d3aad4f6473e301ca6976ef2efac1f333387f80794c4beb112c&ipo=images',
]
def image_iter(urls: list[str]) -> iter:
    for url in urls:
        img = Image.open(urlopen(url))
        yield img

# Model configuration file.
MODEL_BASE_PATH = "./models/yolos-tiny"

# Verify model exist
for file in [MODEL_BASE_PATH]:
   if not os.path.exists(file): raise Exception(f'{file} not found') 

# Load pre-processor configurations
processor: YolosImageProcessor = AutoImageProcessor.from_pretrained(MODEL_BASE_PATH)

# Load the model with weights applied
model_config: YolosConfig = AutoConfig.from_pretrained(MODEL_BASE_PATH)
model: YolosForObjectDetection = AutoModelForObjectDetection.from_pretrained(MODEL_BASE_PATH, config=model_config)

# Adjust model to work with Intel GPU.
model: YolosForObjectDetection = ipex.optimize(model)

# Construct a pipeline with all of these configurations bundled in.
# https://huggingface.co/docs/transformers/v4.36.1/en/quicktour#pipeline
detect_pipe: ObjectDetectionPipeline = pipeline(
    task='object-detection',
    model=model,
    config=model_config,
    image_processor=processor,
)

# Colors.
COLORS = [
    (255, 0, 0), # Red
    (0, 255, 0), # Green
    (0, 0, 255), # Blue
    (0, 255, 255), # Cyan
    (255, 0, 255), # Magenta
    (255, 255, 0), # Yellow
    (0, 0, 0), # Black
    (255, 255, 255), # White
    (128, 128, 128), # Gray
    (128, 0, 0), # Dark Red
    (0, 128, 0), # Dark Green
    (0, 0, 128), # Dark Blue
    (192, 192, 192), # Light Gray
    (255, 165, 0), # Orange
    (165, 42, 42), # Brown
]

def apply_detection(img: Image, results: List[Dict]) -> Image:
    """
        Helper function that applies a bounding box and labels onto the given image
        and result of object detection on that image.

        Args:
            img: Pillow Image instance.
            results: List of maps which is the result of running the object detection model.
        
        Returns:
            Modified Pillow image instance with bbox and labels.
    """
    img_np = np.array(img)

    for result in results:
        score = result['score']
        label = result['label']
        bbox = result['box']
        color = COLORS[ randrange(0, len(COLORS)) ]

        # Convert box coordinates to integers
        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])

        # Draw bounding box
        cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), color, 2)

        # Display label and score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img_np, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return Image.fromarray(img_np)
        

# Run the model!
for img in image_iter(test_urls):
    t0 = time.time()
    res = detect_pipe(img)
    _img = apply_detection(img, res)
    t1 = time.time()
    dt = t1 - t0
    print('=====================================================================================')
    print(f'Model detection took {dt}s')
    print('Result -> ', res)
    print('=====================================================================================\n')