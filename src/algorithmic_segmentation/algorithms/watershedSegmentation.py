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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self.reduce_noise(gray)

        # Apply thresholding (using Otsu's method for automatic threshold selection)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Remove noise using morphological opening (erosion followed by dilation)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Determine the sure background by dilating the opening
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Determine the sure foreground using the distance transform and thresholding
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Convert sure_fg to uint8 (if it's not already)
        sure_fg = np.uint8(sure_fg)

        # Identify the unknown region (neither foreground nor background)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that the background is not 0, but 1
        markers = markers + 1

        # Mark the unknown region with zero
        markers[unknown == 255] = 0

        # Apply the watershed algorithm
        markers = cv2.watershed(image, markers)

        # Boundaries are marked with -1 after watershed. Let's color them red.
        image[markers == -1] = [0, 0, 255]

        return image

    def __str__(self):
        return f"Watershed_Segmentation_k{self.kernel_size}_i{self.iterations}"