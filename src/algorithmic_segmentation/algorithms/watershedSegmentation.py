import cv2
import numpy as np
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod
from skimage import morphology, color
from skimage.segmentation import flood, flood_fill


class WatershedSegmentation(SegmentationMethod):
    def __init__(self, kernel_size: int = 5, iterations: int = 1):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def reduce_noise(self, gray_image):
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (self.kernel_size, self.kernel_size), 0)
        return blurred_image

    def segment(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        

    def __str__(self):
        return f"Watershed_Segmentation_k{self.kernel_size}_i{self.iterations}"