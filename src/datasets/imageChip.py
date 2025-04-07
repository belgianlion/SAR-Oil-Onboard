from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class ImageChip:
    x_start: int
    y_start: int
    image: np.ndarray
    contains_oil: bool = False