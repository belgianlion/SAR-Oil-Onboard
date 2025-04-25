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
        # Find upper left
        for i in range(height):
            for j in range(width):
                if grey[i, j] != 0:
                    upper_left = np.array([j, i, 1], dtype=np.float32)
                    break
            if upper_left is not None:
                break

        # Find upper right
        for j in range(width):
            for i in range(height):
                if grey[i, j] != 0:
                    upper_right = np.array([j, i, 1], dtype=np.float32)
                    break
            if upper_right is not None:
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