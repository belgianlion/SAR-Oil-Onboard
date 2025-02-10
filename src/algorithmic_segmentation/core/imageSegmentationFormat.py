from dataclasses import dataclass
import numpy as np

@dataclass
class ImageSegmentationFormat:
    image: np.ndarray
    filename: str