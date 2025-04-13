from typing import Iterator
import numpy as np
from src.datasets.imageChip import ImageChip
import cv2

class ImageChipCollection:
    def __init__(self):
        """
        Initializes an empty collection of image chips.
        """
        self.image_chips = []

    def try_add_chip(self, image_chip: ImageChip) -> bool:
        """
        Tries to add an image chip to the collection if it is not empty.

        :param image_chip: The image chip to add.
        :return: True if the chip was added, False if it was empty.
        """
        if self.__is_chip_empty(image_chip):
            print("Chip is empty and will not be added.")
            return False
        self.add_chip(image_chip)
        return True

    def add_chip(self, image_chip: ImageChip):
        """
        Adds an image chip to the collection.

        :param image_chip: The image chip to add.
        """
        self.image_chips.append(image_chip)

    def remove_chip(self, index):
        """
        Removes an image chip from the collection by index.

        :param index: The index of the image chip to remove.
        :raises IndexError: If the index is out of range.
        """
        if 0 <= index < len(self.image_chips):
            del self.image_chips[index]
        else:
            raise IndexError("Index out of range.")

    def get_chip(self, index: int) -> ImageChip:
        """
        Retrieves an image chip by index.

        :param index: The index of the image chip to retrieve.
        :return: The image chip at the specified index.
        :raises IndexError: If the index is out of range.
        """
        if 0 <= index < len(self.image_chips):
            return self.image_chips[index]
        else:
            raise IndexError("Index out of range.")
        
    @staticmethod
    def __is_chip_empty(chip: ImageChip) -> bool:
        chip_content = chip.image
        if cv2.countNonZero(chip_content) == 0:
            return True
        return False

    def __len__(self):
        """
        Returns the number of image chips in the collection.

        :return: The number of image chips.
        """
        return len(self.image_chips)

    def __iter__(self) -> Iterator[ImageChip]:
        """
        Returns an iterator over the image chips in the collection.

        :return: An iterator over the image chips.
        """
        return iter(self.image_chips)