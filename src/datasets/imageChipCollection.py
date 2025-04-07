from typing import Iterator
import numpy as np
from src.datasets.imageChip import ImageChip
import cv2

class ImageChipCollection:
    def __init__(self, whole_image_width=0, whole_image_height=0):
        """
        Initializes an empty collection of image chips.
        """
        self.image_chips = []
        self.whole_image_width = whole_image_width
        self.whole_image_height = whole_image_height

    def try_add_chip_bw(self, image_chip: ImageChip) -> bool:
        """
        Tries to add an image chip to the collection if it is not empty.

        :param image_chip: The image chip to add.
        :return: True if the chip was added, False if it was empty.
        """
        if self.__is_chip_empty(image_chip):
            print("Chip is empty and will not be added.")
            return False
        bw_chip = ImageChipCollection.__convert_to_grey_white_background(image_chip)
        self.add_chip(bw_chip)
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
        if cv2.countNonZero(cv2.cvtColor(chip_content, cv2.COLOR_BGRA2GRAY)) == 0:
            return True
        return False
    
    @staticmethod
    def __convert_to_grey_white_background(chip: ImageChip) -> ImageChip:
        # Assuming 'chip' is a BGRA image
        image = chip.image
        alpha_channel = image[:, :, 3]  # Extract the alpha channel
        white_background = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255  # Create a white background
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Convert to BGR
        image[alpha_channel == 0] = white_background[alpha_channel == 0]  # Set transparent areas to white

        # Now convert the BGR image to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        chip.image = grey_image
        
        return chip

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