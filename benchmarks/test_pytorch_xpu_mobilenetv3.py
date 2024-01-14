from urllib.request import urlopen
from PIL import Image
from transformers import AutoConfig, AutoModelForImageClassification, AutoTokenizer
import json
import numpy as np
import torch
import os

# Loaded image.
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))
# print(img)

# Model configuration file.
MOBILENET_V3_PATH = "./models/mobilenetv3_large_100.ra_in1k"
MOBILENET_V3_CONFIG = f"{MOBILENET_V3_PATH}/config.json"
MOBILENET_V3_MODEL = f"{MOBILENET_V3_PATH}/pytorch_model.bin"
MOBILENET_V3_MODEL_SAFETENSORS = f"{MOBILENET_V3_PATH}/model.safetensors"

# Verify models exist
if not os.path.exists(MOBILENET_V3_PATH): raise Exception(f'{MOBILENET_V3_PATH} not found')
if not os.path.exists(MOBILENET_V3_CONFIG): raise Exception(f'{MOBILENET_V3_CONFIG} not found')
if not os.path.exists(MOBILENET_V3_MODEL): raise Exception(f'{MOBILENET_V3_MODEL} not found')
if not os.path.exists(MOBILENET_V3_MODEL_SAFETENSORS): raise Exception(f'{MOBILENET_V3_MODEL_SAFETENSORS} not found')

# Grab pretrained config of the model's input size.
config = json.load(open(MOBILENET_V3_CONFIG, 'r'))
[shape, width, height] = config['pretrained_cfg']['input_size']

# Resize loaded image.
resized_img = img.resize((width, height,))

# Load configuration
model_config = AutoConfig.from_pretrained(MOBILENET_V3_CONFIG)
# print('config -> ', model_config)

# Load the model with weights applied
model = AutoModelForImageClassification.from_pretrained(MOBILENET_V3_MODEL, config=model_config)
# print('model -> ', model)

# Convert loaded image to a numpy array
image_np = np.array(img)
resized_image_np = np.array(resized_img)
print(f'Original Image shape -> {image_np.shape}')
print(f'Resized Image shape -> {resized_image_np.shape}')

# Perform inference on model
with torch.no_grad():
    outputs = model(resized_image_np)

print('outputs -> ', outputs)