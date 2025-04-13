import cv2
import numpy as np

from src.datasets.images.baseImage import BaseImage

class JpgImage(BaseImage):
    def __init__(self, path: str):
        super().__init__(path)

    def try_convert_to_grayscale(self) -> np.ndarray:
        """
        Converts the image to black and white using OpenCV.
        """
        if len(self.image.shape) == 2:
            gray_image = self.image
        elif len(self.image.shape) == 3 and self.image.shape[2] in [3, 4]:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have 2, 3, or 4 channels")
        
        
        return gray_image
    