import numpy as np
import cv2

from src.datasets.images.baseImage import BaseImage

class PngImage(BaseImage):
    def __init__(self, path: str):
        super().__init__(path)
        self.__find_inscribed_corners()

    def try_convert_to_grayscale(self) -> np.ndarray:
        """
        Converts the image to greyscale using OpenCV.
        If the image is already greyscale, then we return the image as is.
        Otherwise, we convert it to greyscale.
        """
        if len(self.image.shape) == 2:
            gray_image = self.image
        elif len(self.image.shape) == 3 and self.image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        
        return gray_image
    
    def find_angle(self) -> float:
        upper_right = self.corner_points[1]
        upper_left = self.corner_points[0]
        if upper_left is None or upper_right is None:
            return

        angle = np.arctan2(upper_right[1] - upper_left[1], upper_right[0] - upper_left[0])

        angle = np.degrees(angle)  # Convert to degrees
        return angle
    
    def __find_inscribed_corners(self):
        grey = self.try_convert_to_grayscale()

        height, width = grey.shape[:2]

        upper_left = None
        upper_right = None
        lower_left = None
        lower_right = None
        # Find upper right
        for i in range(height):
            for j in range(width):
                if grey[i, j] != 0:
                    upper_right = np.array([j, i, 1], dtype=np.float32)
                    break
            if upper_right is not None:
                break

        # Find upper left
        for j in range(width):
            for i in range(height):
                if grey[i, j] != 0:
                    upper_left = np.array([j, i, 1], dtype=np.float32)
                    break
            if upper_left is not None:
                break

        # Find lower left
        for i in range(height-1, -1, -1):
            for j in range(width-1, -1, -1):
                if grey[i, j] != 0:
                    lower_left = np.array([j, i, 1], dtype=np.float32)
                    break
            if lower_left is not None:
                break

        # Find lower right
        for j in range(width-1, -1, -1):
            for i in range(height-1, -1, -1):
                if grey[i, j] != 0:
                    lower_right = np.array([j, i, 1], dtype=np.float32)
                    break
            if lower_right is not None:
                break

        self.corner_points = np.vstack([upper_left, upper_right, lower_left, lower_right])
        upper_left_point = (int(upper_left[0]), int(upper_left[1]))
        upper_right_point = (int(upper_right[0]), int(upper_right[1]))
        lower_left_point = (int(lower_left[0]), int(lower_left[1]))
        lower_right_point = (int(lower_right[0]), int(lower_right[1]))
        color = (0, 255, 0)
        thickness = 100
        out_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        cv2.line(out_img, upper_left_point, upper_right_point, color, thickness)
        cv2.line(out_img, upper_left_point, lower_left_point, color, thickness)
        cv2.line(out_img, lower_right_point, upper_right_point, color, thickness)
        cv2.line(out_img, lower_right_point, lower_left_point, color, thickness)
        # cv2.imwrite(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\results\edges_of_image.png", out_img)