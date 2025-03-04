from typing import Tuple
import cv2
import numpy as np

from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod


# Logically based off the implementation found here: https://learnopencv.com/otsu-thresholding-with-opencv/
class AdaptiveOtsuSegmentation(SegmentationMethod):
    def __init__(self, kernel_size: Tuple[int, int] = (5, 5), adaptive_block_size: int = 21, mean_bias: int = 2):
        self.kernel_size = kernel_size
        self.adaptive_block_size = adaptive_block_size
        self.mean_bias = mean_bias

    def reduce_noise(self, gray_image):
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, self.kernel_size, 0)
        return blurred_image
    
    def segment(self, image):
        # Reduce noise
        preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.kernel_size is not None and self.kernel_size != (0, 0):
            preprocessed_image = self.reduce_noise(preprocessed_image)

        equalized = cv2.equalizeHist(preprocessed_image)

        # Calculate threshold
        threshold = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptive_block_size, self.mean_bias)

        # Apply threshold to segment the image
        return threshold

    def __str__(self):
        return f"Adaptive_Otsu_Segmentation_k{self.kernel_size}_a{self.adaptive_block_size}_mb{self.mean_bias}"