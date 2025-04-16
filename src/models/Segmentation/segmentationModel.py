from ultralytics import YOLO

class SegmentationModel():
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def run_model(self, image_path: str):
        results = self.model(image_path)
        return results[0].masks.data.cpu().numpy() if results[0].masks else None
