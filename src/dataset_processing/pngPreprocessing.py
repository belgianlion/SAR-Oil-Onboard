import math
import cv2
import numpy as np

from src.algorithmic_segmentation.algorithms.common import Common



class PNGProcessing():
    def __init__(self):
        pass

    @staticmethod
    def process_png(png_path):
        try:
            image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not open or find the image: {png_path}")
            if image.shape[-1] == 4:  # Check if the image has an alpha channel
                if image.shape[0] > 10000 or image.shape[1] > 10000:  # Check if dimensions are too large
                    resized_image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)  # Resize alpha channel to half its size
                else:
                    resized_image = image  # Use original alpha channel if dimensions are manageable
                # color_alpha = cv2.cvtColor(resized_alpha, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image
                # cv2.imshow("alpha", color_alpha)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # corners = PNGProcessing.find_angle(color_alpha)  # Find corners in the alpha channel
                
                # cv2.imshow("Processed PNG", corners)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                chips = PNGProcessing.split_into_chips(resized_image, chip_size=400, overlap_percentage=0.5)
            return image
        except Exception as e:
            print(f"Error opening PNG file with OpenCV: {e}")
            return None
        
    @staticmethod
    def split_into_chips(image, chip_size: int = 400, overlap_percentage: float = 0.5):
        height, width = image.shape[:2]
        frame_separation = overlap_percentage * chip_size
        chips = []

        x_start = 0
        while x_start + chip_size <= width:
            y_start = 0
            while y_start + chip_size <= height:
                chip = image[int(y_start):int(y_start + chip_size),
                            int(x_start):int(x_start + chip_size)]
                
                if not PNGProcessing.is_chip_empty([chip]):
                    cv2.imshow(f"Chip at ({int(height - chip_size)}, {int(x_start)})", chip)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    chips.append(chip)
                y_start += frame_separation

            # Handle the last row if it doesn't fit perfectly
            if y_start < height:
                chip = image[int(height - chip_size):int(height),
                            int(x_start):int(x_start + chip_size)]
            if not PNGProcessing.is_chip_empty([chip]):
                cv2.imshow(f"Chip at ({int(height - chip_size)}, {int(x_start)})", chip)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                chips.append(chip)

            x_start += frame_separation

        # Handle the last column if it doesn't fit perfectly
        if x_start < width:
            y_start = 0
            while y_start + chip_size <= height:
                chip = image[int(y_start):int(y_start + chip_size),
                            int(width - chip_size):int(width)]
                if not PNGProcessing.is_chip_empty([chip]):
                    cv2.imshow(f"Chip at ({int(height - chip_size)}, {int(x_start)})", chip)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    chips.append(chip)
                y_start += frame_separation

            # Handle the last row in the last column
            if y_start < height:
                chip = image[int(height - chip_size):int(height),
                            int(width - chip_size):int(width)]
            if not PNGProcessing.is_chip_empty([chip]):
                cv2.imshow(f"Chip at ({int(height - chip_size)}, {int(x_start)})", chip)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                chips.append(chip)
        return chips

    @staticmethod
    def is_chip_empty(chips) -> bool:
        for i, chip in enumerate(chips):
            if cv2.countNonZero(cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)) == 0:
                print(f"Chip {i} is empty.")
                return True
            print(f"Chip {i} is not empty.")
        return False


        
    @staticmethod
    def find_angle(image, padding=5):
        grey = Common.tryConvertBGRToGray(image)

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
        cv2.line(image, tuple(upper_left), tuple(upper_right), (0, 255, 0), 2)

        angle = np.arctan2(upper_right[1] - upper_left[1], upper_right[0] - upper_left[0])

        angle = np.degrees(angle)  # Convert to degrees
        rotated_image = PNGProcessing.rotate_image(image, angle)
        return rotated_image
    
    @staticmethod
    def rotate_image(image, angle):
        height, width = image.shape[:2]
        center = (width//2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

                    