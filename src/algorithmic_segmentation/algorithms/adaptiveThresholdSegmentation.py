from typing import Tuple

import cv2
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod

class AdaptiveThresholdSegmentation(SegmentationMethod):
    def __init__(self, max_value: int = 255, block_size: int = 11, c: float = 2.0):
        self.max_value = max_value
        self.block_size = block_size
        self.c = c
    
    def segment(self, image):
        if len(image.shape) == 2:
            gray_image = image
        elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        
        # Apply adaptive thresholding
        adaptive_thresholded_image = cv2.adaptiveThreshold(gray_image, self.max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.c)
        
        return adaptive_thresholded_image
    
    def __str__(self):
        return f"Adaptive_Threshold_Segmentation_max{self.max_value}_block{self.block_size}_c{self.c}"

