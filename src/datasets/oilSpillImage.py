import cv2

class OilSpillImage():

    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.oil_images = ImageChipCollection()