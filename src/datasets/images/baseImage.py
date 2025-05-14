from abc import abstractmethod
import cv2
from PIL import Image
import numpy as np

class BaseImage:
    def __init__(self, path: str):
        image = Image.open(path).resize((image.width // 4, image.height // 4)).convert("LA")
        self.image = np.array(image)

    def width(self):
        return self.image.shape[1]
    
    def height(self):
        return self.image.shape[0]

    @abstractmethod
    def convert_to_grayscale(self):
        pass
   