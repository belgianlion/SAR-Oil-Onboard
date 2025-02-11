from typing import Tuple
import cv2
import numpy as np

from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod


# Logically based off the implementation found here: https://learnopencv.com/otsu-thresholding-with-opencv/
class OtsuSegmentation(SegmentationMethod):
    def __init__(self, kernel_size: Tuple[int, int] = (5, 5)):
        self.kernel_size = kernel_size

    def reduce_noise(self, gray_image):
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, self.kernel_size, 0)
        return blurred_image
    
    def segment(self, image):
        # Reduce noise
        preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.kernel_size is not None and self.kernel_size != (0, 0):
            preprocessed_image = self.reduce_noise(preprocessed_image)

        # Calculate threshold
        _, threshold = cv2.threshold(preprocessed_image, 0, 255, cv2.THRESH_OTSU)

        # Apply threshold to segment the image
        return threshold

    def __str__(self):
        return f"Otsu_Segmentation_k{self.kernel_size}"