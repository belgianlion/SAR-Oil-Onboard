import cv2
import numpy as np

from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod


# Logically based off the implementation found here: https://learnopencv.com/otsu-thresholding-with-opencv/
class ImprovedOtsuSegmentation(SegmentationMethod):
    def __init__(self, kernel_size_x=5, kernel_size_y=5):
        self.kernel_size_x = kernel_size_x
        self.kernel_size_y = kernel_size_y

    def calculate_threshold(self, blurred_image):
        # Calculate histogram
        hist, bin_edges = np.histogram(blurred_image, bins=256, range=(0, 256))

        # Normalize the histogram
        hist = hist.ravel()

        bin_middles = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate cumulative sums
        w1 = np.cumsum(hist)
        w2 = np.cumsum(hist[::-1])[::-1]

        # Avoid division by zero
        w1[w1 == 0] = 1
        w2[w2 == 0] = 1

        # Calculate cumulative means
        m1 = np.cumsum(hist * bin_middles) / w1
        m2 = (np.cumsum((hist * bin_middles)[::-1]) / w2[::-1])[::-1]

        # Calculate between-class variance
        sigma_b_squared = w1 * w2 * (m1 - m2) ** 2

        # Find the threshold that maximizes the between-class variance
        threshold = np.argmax(sigma_b_squared)

        return threshold
        

    def reduce_noise(self, gray_image):
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (self.kernel_size_x, self.kernel_size_y), 0)
        return blurred_image
    
    def segment(self, image):
        # Reduce noise
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurred_image = self.reduce_noise(gray_image)

        # Calculate threshold
        threshold = self.calculate_threshold(gray_image)

        # Apply threshold to segment the image
        _, self.mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return self.mask

    def __str__(self):
        return "Improved_Otsu_Segmentation"