"""
    Exmaple use of Intel DG2 on an Image Classification model for which
    results in only labeling an image.

    The model used is https://huggingface.co/microsoft/resnet-50
"""
from urllib.request import urlopen
from PIL import Image
from transformers import (
    # Classification model.
    AutoModelForImageClassification,
    ResNetForImageClassification,

    # Model Configuration.
    AutoConfig,
    ResNetConfig,

    # Pre-Processors.
    AutoImageProcessor,
    AutoFeatureExtractor,
    ConvNextImageProcessor,
    ConvNextFeatureExtractor,

    # Pipeline.
    ImageClassificationPipeline,
    pipeline,
)
from transformers import pipeline
from typing import Iterator

import json
import numpy as np
import torch
import os
import intel_extension_for_pytorch as ipex
import time

# Verify Intel DG2 extention works.
print(f'PyTorch Version: {torch.__version__}')
print(f'Intel PyTorch Extension Version: {ipex.__version__}')

# Construct an iterable of test images.
test_urls = [
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png',
    'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2F4.bp.blogspot.com%2F-1SBVTLdwHqc%2FVTutdWprmXI%2FAAAAAAAAbhw%2F-i_dV2h-aPQ%2Fs1600%2Fkeanu-reeves-birthday.jpg&f=1&nofb=1&ipt=6c5096ad424a3d3aad4f6473e301ca6976ef2efac1f333387f80794c4beb112c&ipo=images',
]
def image_iter(urls: list[str]) -> Iterator:
    for url in urls:
        img = Image.open(urlopen(url))
        yield img

# Model configuration file.
RESNET50_PATH = "./models/resnet-50"

# Verify models exist
for file in [ RESNET50_PATH ]:
   if not os.path.exists(file): raise Exception(f'{file} not found') 

# Load pre-processor configurations
processor: ConvNextImageProcessor = AutoImageProcessor.from_pretrained(RESNET50_PATH)
feature_extractor: ConvNextFeatureExtractor = AutoFeatureExtractor.from_pretrained(RESNET50_PATH)

# Load the model with weights applied
model_config: ResNetConfig = AutoConfig.from_pretrained(RESNET50_PATH)
model = AutoModelForImageClassification.from_pretrained(RESNET50_PATH, config=model_config)

# Adjust model to work with Intel GPU.
model: ResNetForImageClassification = ipex.optimize(model)


# Construct a pipeline with all of these configurations bundled in.
# https://huggingface.co/docs/transformers/v4.36.1/en/quicktour#pipeline
classify_pipe: ImageClassificationPipeline = pipeline(
    task='image-classification',
    model=model,
    config=model_config,
    image_processor=processor,
    feature_extractor=feature_extractor,
)

# Perform inference on model
for img in image_iter(test_urls):
    t0 = time.time()
    with torch.no_grad():
        res = classify_pipe(img)
    t1 = time.time()
    dt = t1 - t0
    print(f'Model classification took {dt}s')
    print('Result -> ', res)