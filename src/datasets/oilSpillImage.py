from typing import List, Tuple
import cv2
import numpy as np

from src.datasets.images.imageXmlData import ImageXmlData
from src.datasets.images.jpgImage import JpgImage
from src.datasets.images.pngImage import PngImage
from src.datasets.imageChip import ImageChip
from src.datasets.imageChipCollection import ImageChipCollection

MAX_SIZE = 32767

class OilSpillImage():

    def __init__(self, png_image_path: str, jpg_image_path: str, xml_jpg_image_path: str, relative_scale: float = 1.0, with_rotation: bool = False, padding: int = 0):
        self.png_image = PngImage(png_image_path)
        self.jpg_image = JpgImage(jpg_image_path)
        self.xml_data = ImageXmlData(xml_jpg_image_path)
        self.inverse_transformation_matrix = None

        self.chips = ImageChipCollection()
        self.classed_chips = ImageChipCollection()
        if with_rotation:
            base_scale_matrix, scaled_dims= self.__get_base_scale_matrix(padding) 
            scaled_image = cv2.resize(self.jpg_image.image, (scaled_dims[1], scaled_dims[0]))
            new_corners = OilSpillImage.__apply_transform_to_points(self.png_image.corner_points, base_scale_matrix)

            transformation_matrix, dimensions = OilSpillImage.__generate_matrix(scaled_image, self.png_image.find_angle(), padding)
            new_corners = OilSpillImage.__apply_transform_to_points(new_corners, transformation_matrix)
            transformed_image = cv2.warpAffine(scaled_image, transformation_matrix, (dimensions[1], dimensions[0]))

            crop_shift_matrix, dimensions = OilSpillImage.__crop_from_corners(new_corners)
            cropped_image = cv2.warpAffine(transformed_image, crop_shift_matrix, (dimensions[1]+padding, dimensions[0]+padding))

            display_scale = 0.25
            scaled_cropped = cv2.resize(cropped_image, (0, 0), fx=display_scale, fy=display_scale)
            cv2.imshow("scaled_cropped_image", scaled_cropped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.jpg_image.image = cropped_image
            self.transformation_matrix = OilSpillImage.__combine_matricies([base_scale_matrix, transformation_matrix, crop_shift_matrix])
            
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

        # rotated_image = OilSpillImage.__rotate_images(self.png_image, angle)
        # return rotated_image

    def __get_base_scale_matrix(self, padding=0):
        height, width = self.jpg_image.image.shape[:2]

        scale = 1.0
        if width + 2*padding > MAX_SIZE or height + 2*padding > MAX_SIZE:
            sx = MAX_SIZE / (width + 2*padding)
            sy = MAX_SIZE / (height + 2*padding)
            scale = min(sx, sy)
            scale_matrix = np.array([[scale, 0, 0], [0, scale, 0]])
            scaled_height = int(height * scale)
            scaled_width = int(width * scale)

        return scale_matrix, (scaled_height, scaled_width)
    
    @staticmethod
    def __combine_matricies(matricies: List):
        aligned_matricies = []
        for matrix in matricies:
            if matrix.shape == (2, 3):
                matrix = np.vstack([matrix, [0, 0, 1]])
            aligned_matricies.append(matrix)

        combined_matrix = aligned_matricies[0]
        for matrix in aligned_matricies[1:]:
            combined_matrix = np.matmul(matrix, combined_matrix)
        return combined_matrix
    
    def map_location(self, splines: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not len(splines):
            return []
        lat_long_positions = []
        for spline in splines:
            for i in range(len(spline[0])):
                rearranged_point = [int(spline[0][i]), int(spline[1][i])]
                point_on_original = self.get_inverse_transform(rearranged_point)
                xml_data = self.xml_data
                longitude = xml_data.x_origin + (xml_data.pixel_width * point_on_original[0]) + (xml_data.x_rotation * point_on_original[1]) 
                latitude = xml_data.y_origin + (xml_data.y_rotation * point_on_original[0]) + (xml_data.pixel_height * point_on_original[1])
                lat_long_positions.append((latitude, longitude))
        return lat_long_positions




    def get_inverse_transform(self, point):
        if self.inverse_transformation_matrix is None:
            self.__get_inverse_transform_matrix()
        point = np.array(point)
        if len(point.shape) == 1:
            point = np.array([point])
        if point.shape[1] == 2:
            point = np.hstack([point, np.ones((point.shape[0], 1))])
        return np.dot(point, self.inverse_transformation_matrix.T)[0]

    def __get_inverse_transform_matrix(self):
        if self.transformation_matrix is None:
            raise Exception("Image not transformed, cannot get inverse transform matrix")
        full_transform_matrix = self.transformation_matrix
        if full_transform_matrix.shape == (2, 3):
            full_transform_matrix = np.vstack([full_transform_matrix, [0, 0, 1]])

        self.inverse_transformation_matrix = np.linalg.inv(full_transform_matrix)

    @staticmethod
    def __apply_transform_to_points(points, transform_matrix):
        if points.shape[1] == 2:
            points = np.hstack([points, np.ones((points.shape[0], 1))])
        return np.dot(points, transform_matrix.T)
    
    @staticmethod
    def __crop_from_corners(corners):
        min_x = np.min(corners[:, 0])
        min_y = np.min(corners[:, 1])
        max_x = np.max(corners[:, 0])
        max_y = np.max(corners[:, 1])

        crop_shift_matrix = np.float32([
            [1, 0, -min_x],
            [0, 1, -min_y]
        ])

        width = int(max_x-min_x)
        height = int(max_y-min_y)

        return crop_shift_matrix, (height, width)



    @staticmethod
    def __generate_matrix(image, angle, padding=0, relative_scale: float = 1.0):
        height, width = image.shape[:2]

        width_with_padding = width + 2*padding
        height_with_padding = height + 2*padding

        rotated_width = int(width_with_padding * abs(np.cos(np.radians(angle))) + height_with_padding * abs(np.sin(np.radians(angle))))
        rotated_height = int(height_with_padding * abs(np.cos(np.radians(angle))) + width_with_padding * abs(np.sin(np.radians(angle))))

        scale = 1.0
        if rotated_width > MAX_SIZE or rotated_height > MAX_SIZE:
            sx = MAX_SIZE / rotated_width
            sy = MAX_SIZE / rotated_height
            scale = min(sx, sy)
            new_width = int(width_with_padding * scale)
            new_height = int(height_with_padding * scale)
        
        shift_matrix = np.float32([[1, 0, padding], [0, 1, padding]])

        center = (width_with_padding // 2, height_with_padding // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        final_width = int(new_height * sin + new_width * cos)
        final_height = int(new_height * cos + new_width * sin)

        # Add rotation around center 
        rotation_matrix[0, 2] += (final_width - width_with_padding) / 2
        rotation_matrix[1, 2] += (final_height - height_with_padding) / 2

        scale_matrix = np.array([[relative_scale, 0, 0], [0, relative_scale, 0], [0, 0, 1]])

        combo_matrix = np.matmul(rotation_matrix, np.vstack([shift_matrix, [0, 0, 1]]))
        combo_matrix = np.matmul(combo_matrix, scale_matrix)[:2]
        final_height = int(final_height * relative_scale)
        final_width = int(final_width * relative_scale)

        return combo_matrix, (final_height, final_width)
        

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

        # Store the combined transformation matrix (shift followed by rotation)
        combo_matrix = np.dot(rotation_matrix, shift_matrix)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour to ensure the correct bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            crop_matrix = np.float32([[1, 0, -x], [0, 1, -y]])
            rotated_image = rotated_image[y:y+h, x:x+w]
            return rotated_image, crop_matrix
        else:
            # If no contours are found, return the original rotated image
            print("Warning: No contours found. Returning the original rotated image.")

        return rotated_image