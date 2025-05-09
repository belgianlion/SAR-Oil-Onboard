from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod
import cv2

class CannyEdgeDetection(SegmentationMethod):
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def segment(self, image):
        # Check if the image has 3 or 4 channels before converting to grayscale
        if len(image.shape) == 2:
            gray_image = image
        elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        high_thresh, _ = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
        lowThresh = 0.10 * high_thresh

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, lowThresh, high_thresh)

        return edges

    def __str__(self):
        return f"Canny_Edge_Detection_low{self.low_threshold}_high{self.high_threshold}"