from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod

class CascadedSegmentation(SegmentationMethod):
    def __init__(self, *segmentation_methods):
        self.segmentation_methods = segmentation_methods

    def segment(self, image):
        for segmentation_method in self.segmentation_methods:
            image = segmentation_method.segment(image)
        return image

    def __str__(self):
        string="Cascaded_Segmentation"
        for segmentation_method in self.segmentation_methods:
            string += "_" + str(segmentation_method)
        return string
