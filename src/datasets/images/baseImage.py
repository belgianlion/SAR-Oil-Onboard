from abc import abstractmethod
import cv2

class BaseImage:
    def __init__(self, path: str):
        image = cv2.imread(path)
        self.image = cv2.resize(image, cv2.IMREAD_REDUCED_COLOR_2)

    def width(self):
        return self.image.shape[1]
    
    def height(self):
        return self.image.shape[0]

    @abstractmethod
    def convert_to_grayscale(self):
        pass
   