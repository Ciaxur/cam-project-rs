from random import randrange
from typing import Callable, Tuple

# Expected image.
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# List of bbox colors.
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
get_random_color: Callable[[], Tuple[int, int, int]] = lambda: COLORS[ randrange(0, len(COLORS)) ]