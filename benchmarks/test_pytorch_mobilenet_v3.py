from torchvision.models import (
  # MobileNet instance types
  MobileNetV3,

  # MobileNet weight types
  MobileNet_V3_Large_Weights,
  MobileNet_V3_Small_Weights,

  # MobileNet Model Constructors.
  mobilenet_v3_small,
  mobilenet_v3_large,
)
import torchvision.transforms as transforms
from PIL import Image
from urllib.request import (
  urlopen,
  urlretrieve,
)
from time import time

import torch
import intel_extension_for_pytorch as ipex

# Verify Intel DG2 extention works.
print(f'PyTorch Version: {torch.__version__}')
print(f'Intel PyTorch Extension Version: {ipex.__version__}')


def get_mobilenet_classes() -> list[str]:
  # Get imagenet class mappings
  url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
  urlopen(url) 

  categories = []
  with urlopen(url) as f:
    categories = [s.strip() for s in f.readlines()]
  return categories

def evaluate_result(probabilities: torch.Tensor, ids: torch.Tensor):
  categories = get_mobilenet_classes()
  for i in range(probabilities.size(0)):
    score = probabilities[i].item()
    label = categories[ids[i]]
    print(f'label={label}, score={score}')


# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Construct thy model.
model: MobileNetV3 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

# Convert model to use XPU.
model.eval()
model = ipex.optimize(model)

## Test the model ##
# Load an image.
img = Image.open(urlopen(
  'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# Process the image.
t0 = time()
print('Processing image')
model_input: torch.Tensor = preprocess(img)

# Add a batch dimension
input_batch = model_input.unsqueeze(0)  

with torch.no_grad():
  output: torch.Tensor = model(input_batch)

# Extract prediction results.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(f'Probabilities -> {probabilities.shape}')

top5_prob, top5_ids = torch.topk(probabilities, 5)
top5_ids: torch.Tensor
top5_prob: torch.Tensor
evaluate_result(top5_prob, top5_ids)

t1 = time()
dt = t1 - t0
print(f'Image processing time took {dt}s')