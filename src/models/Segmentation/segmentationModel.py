import numpy as np
from ultralytics import YOLO
import cv2

from src.datasets.imageChipCollection import ImageChipCollection

class SegmentationModel():
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def run_model(self, image_path: str):
        results = self.model(image_path)
        return results[0].masks.data.cpu().numpy() if results[0].masks else None
    
    def run_on_collection(self, image_collection: ImageChipCollection):
        for image_chip in image_collection:
            color_chip = cv2.cvtColor(image_chip.image, cv2.COLOR_GRAY2BGR)
            results = self.model(color_chip)
            if results[0].masks:
                masks = results[0].masks.data.cpu().numpy()
                if len(masks) > 0:
                    image_chip.contains_oil = 1
                else:
                    image_chip.contains_oil = 0
                    continue
                combined_mask = np.zeros((color_chip.shape[0], color_chip.shape[1]), dtype=np.uint8)
                for mask in masks:
                    mask = cv2.resize(mask, (color_chip.shape[1], color_chip.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    combined_mask = np.maximum(combined_mask, mask)
                image_chip.mask = combined_mask
            else:
                image_chip.contains_oil = 0
