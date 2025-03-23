from typing import Tuple

import cv2
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod

class ThresholdSegmentation(SegmentationMethod):
    def __init__(self, threshold: int = 100, max_value: int = 255, invert_result: bool = True):
        self.threshold = threshold
        self.max_value = max_value
        self.invert_result = invert_result
    
    def segment(self, image):
        if len(image.shape) == 2:
            gray_image = image
        elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        thresholded_image = cv2.threshold(gray_image, self.threshold, self.max_value, cv2.THRESH_BINARY)[1]
        if self.invert_result:
            thresholded_image = cv2.bitwise_not(thresholded_image)
        return thresholded_image
    
    def __str__(self):
        return f"Threshold_Segmentation_t{self.threshold}_max{self.max_value}"

