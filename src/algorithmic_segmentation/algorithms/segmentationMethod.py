from abc import ABC, abstractmethod

class SegmentationMethod(ABC):
    """Base class for segmentation methods."""
    @abstractmethod
    def segment(self, image):
        """Segment the given image."""
        pass

    @abstractmethod
    def __str__(self):
        return super().__str__()