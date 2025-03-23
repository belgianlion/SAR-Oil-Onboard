from typing import Tuple

import cv2
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod

class NormalizationForSegmentation(SegmentationMethod):
    def __init__(self):
        pass
    
    def segment(self, image):
        if len(image.shape) == 2:
            gray_image = image
        elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return normalized_image
        
    
    def __str__(self):
        return f"Normalization"
