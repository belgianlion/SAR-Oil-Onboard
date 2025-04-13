import cv2
import numpy as np

from src.datasets.images.jpgImage import JpgImage
from src.datasets.images.pngImage import PngImage
from src.datasets.imageChip import ImageChip
from src.datasets.imageChipCollection import ImageChipCollection

class OilSpillImage():

    def __init__(self, png_image_path: str, jpg_image_path: str, relative_scale: float = 1.0, with_rotation: bool = False, padding: int = 0):
        self.png_image = PngImage(png_image_path)
        self.jpg_image = JpgImage(jpg_image_path)

        self.chips = ImageChipCollection()
        self.class_1_chips = ImageChipCollection()
        if with_rotation:
            angle = self.__find_angle()
            self.jpg_image.image = OilSpillImage.__rotate_image(self.jpg_image.image, angle, padding)
        self.jpg_image.image = cv2.resize(self.jpg_image.image, (0, 0), fx=relative_scale, fy=relative_scale)


    def split_jpg_into_chips(self, chip_size: int = 400, overlap_percentage: float = 0.5):
        image_content = self.jpg_image.try_convert_to_grayscale()
        frame_separation = overlap_percentage * chip_size
        height, width = image_content.shape[:2]

        def split_vertically(y, x):
            while y + chip_size <= height:
                # This is the default path to split, scanning the image from
                # top to bottom, left to right and extracting images of size chip_size
                chip = ImageChip(int(x), int(y), image_content[
                                int(y):int(y + chip_size),
                                int(x):int(x + chip_size)])
                self.chips.try_add_chip(chip)
                y += frame_separation

            if y < height:
                # If there is still a small area at the bottom of the image that
                # is not covered by any previous chips, we want to take a chip
                # that sits perfectly on the bottom
                y = height - chip_size
                chip = ImageChip(int(x), int(y), image_content[
                                int(y):int(y + chip_size),
                                int(x):int(x + chip_size)])
                self.chips.try_add_chip(chip)


        x_start = 0
        while x_start + chip_size <= width:
            y_start = 0
            split_vertically(y_start, x_start)
            x_start += frame_separation
            
        if x_start < width:
            # If there is still a small area at the right of the image that
            # is not covered by any previous chips, we want to take a chip
            # that sits perfectly on the right
            y_start = 0
            x_start = width - chip_size
            split_vertically(y_start, x_start)


    def __find_angle(self) -> float:
        grey = self.png_image.try_convert_to_grayscale()

        upper_left = (0, 0)
        upper_right = (0, 0)
        height, width = grey.shape[:2]
        for i in range(height):
            for j in range(width):
                if grey[i, j] != 0:
                    upper_left = (j, i)
                    break
                if upper_left != (0, 0):
                    break

        for j in range(width):
            for i in range(height):
                if grey[i, j] != 0:
                    upper_right = (j, i)
                    break
                if upper_right != (0, 0):
                    break
        cv2.line(grey, tuple(upper_left), tuple(upper_right), (0, 255, 0), 2)

        angle = np.arctan2(upper_right[1] - upper_left[1], upper_right[0] - upper_left[0])

        angle = np.degrees(angle)  # Convert to degrees
        return angle
        # rotated_image = OilSpillImage.__rotate_images(self.png_image, angle)
        # return rotated_image


    @staticmethod
    def __rotate_image(image, angle, padding=0):
        # TODO: I had help from Copilot on this. I want to increase the 
        # efficiency of this code in the future
        height, width = image.shape[:2]

        max_size = 32767

        width_with_padding = width + 2*padding
        height_with_padding = height + 2*padding

        if width + 2*padding > max_size or height + 2*padding > max_size:
            scale_factor = min(max_size / (width + 2*padding), max_size / (height + 2*padding))
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
            width, height = new_width, new_height

        shift_matrix = np.float32([[1, 0, padding], [0, 1, padding]])
        shifted_image = cv2.warpAffine(image, shift_matrix, (width+padding, height+padding))

        # Ensure the image is properly scaled and padded

        rotated_width = int(width_with_padding * abs(np.cos(np.radians(angle))) + height_with_padding * abs(np.sin(np.radians(angle))))
        rotated_height = int(height_with_padding * abs(np.cos(np.radians(angle))) + width_with_padding * abs(np.sin(np.radians(angle))))

        if rotated_width > max_size or rotated_height > max_size:
            # If the rotated image exceeds the maximum size, scale down the original image
            scale_factor = min(max_size / rotated_width, max_size / rotated_height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            shifted_image = cv2.resize(shifted_image, (width, height))
            rotated_width = int(rotated_width * scale_factor)
            rotated_height = int(rotated_height * scale_factor)
            
        center = (width_with_padding // 2, height_with_padding // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Adjust the bounding box to ensure the entire rotated image fits
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int(height_with_padding * sin + width_with_padding * cos)
        new_height = int(height_with_padding * cos + width_with_padding * sin)

        # Update the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width - width_with_padding) / 2
        rotation_matrix[1, 2] += (new_height - height_with_padding) / 2

        rotated_image = cv2.warpAffine(shifted_image, rotation_matrix, (new_width, new_height))

        # Crop the image to remove black borders
        gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY) if len(rotated_image.shape) == 3 else rotated_image

        _, binary = cv2.threshold(gray_rotated, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour to ensure the correct bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            rotated_image = rotated_image[y:y+h, x:x+w]
        else:
            # If no contours are found, return the original rotated image
            print("Warning: No contours found. Returning the original rotated image.")

        return rotated_image