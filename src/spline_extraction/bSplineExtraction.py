from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as spi

from src.algorithmic_segmentation.algorithms.common import Common

class BSplineExtraction:
    def __init__(self, num_points: int = 20):
        self.num_points = num_points

    def extract_spline(self, image) -> List[Tuple[float, float]]:
        gray_image = Common.tryConvertBGRToGray(image)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        splines = []
        for contour in contours:
            # Finds unique contours within the image and creates a spline from it
            # This process was found with the help of github copilot.
            # None of the code worked out of the box, and was modified by me
            if contour.shape[0] < 2:  # Ignore contours with fewer than 2 points
                continue

            contour = np.squeeze(contour)  # Only remove single dimensions
            x = contour[:, 0]
            y = contour[:, 1]

            # Fit a B-spline to the contour points
            try:
                tck, _ = spi.splprep([x, y], s=0, per=False)
                xnew, ynew = spi.splev(np.linspace(0, 1, 20), tck)
                splines.append((xnew, ynew))
            except:
                print("Unable to extract spline from current line, going to next.")
                continue
        return splines