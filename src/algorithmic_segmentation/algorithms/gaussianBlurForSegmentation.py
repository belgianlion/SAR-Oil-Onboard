from typing import Tuple

import cv2
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod

class GaussianBlurForSegmentation(SegmentationMethod):
    def __init__(self, kernel_size: Tuple[int, int] = (5, 5)):
        self.kernel_size = kernel_size
    
    def segment(self, image):
        if len(image.shape) == 2:
            gray_image = image
        elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        blurred_image = cv2.GaussianBlur(gray_image, self.kernel_size, 0)
        return blurred_image
    
    def __str__(self):
        return f"Gaussian_Blur_k{self.kernel_size}"

