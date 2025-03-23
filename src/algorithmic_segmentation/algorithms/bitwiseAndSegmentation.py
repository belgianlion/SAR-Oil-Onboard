import cv2
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod

class BitwiseAndSegmentation(SegmentationMethod):
    def __init__(self, *segmentation_methods):
        self.segmentation_methods = segmentation_methods

    def segment(self, image):
        images = []
        for segmentation_method in self.segmentation_methods:
            images.append(segmentation_method.segment(image))
        # Perform bitwise OR operation on the segmented images
        image = images[0]
        for i in range(1, len(images)):
            image = cv2.bitwise_and(image, images[i])
            
        return image

    def __str__(self):
        string="Bitwise_And_Segmentation"
        for segmentation_method in self.segmentation_methods:
            string += "_" + str(segmentation_method)
        return string
