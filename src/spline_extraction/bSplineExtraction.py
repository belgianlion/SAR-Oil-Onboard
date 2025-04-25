from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as spi

from src.spline_extraction.spline import Spline
from src.algorithmic_segmentation.algorithms.common import Common

class BSplineExtraction:
    def __init__(self, num_points: int = 20):
        self.num_points = num_points

    def extract_spline(self, image) -> List[Tuple[float, float]]:
        gray_image = Common.tryConvertBGRToGray(image)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
        inverted_image = cv2.bitwise_not(binary_image)

        contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        splines = []
        for contour in contours:
            if contour.shape[0] < 2:
                continue

            contour = np.squeeze(contour)
            x = contour[:, 0]
            y = contour[:, 1]

            try:
                tck, _ = spi.splprep([x, y], s=0, per=False)
                linspace = np.linspace(0, 1, 20)
                xnew, ynew = spi.splev(linspace, tck)
                spline = Spline(list(zip(xnew, ynew)), tck, linspace)
                splines.append(spline)
            except:
                print("Unable to extract spline from current line, going to next.")
                continue
        return splines
    
    @staticmethod
    def try_add_splines_to_image(image: np.ndarray, splines: List[Spline], curve_point_count: int = 100):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if not len(splines):
            return image
        
        for spline in splines:
            # Based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splrep.html#scipy.interpolate.splrep
            # & with some help from copilot to understand what everything does
            fidelity = np.linspace(0, 1, curve_point_count)
            smooth_x, smooth_y = spi.splev(fidelity, spline.tck)

            points = np.column_stack((smooth_x, smooth_y)).astype(np.int32)
            cv2.polylines(image, [points], False, (0, 255, 0), 2)
            
        return image