import math
import cv2
import numpy as np


class ChunkedSARImage():
    def __init__(self, png_image: np.ndarray):
        self.chunk_size = 400
        interleaving = 0.5
        separation_distance = self.chunk_size * interleaving

        self.image = png_image
        image_size_x = png_image.shape[1]
        image_size_y = png_image.shape[0]
        chunk_count_x = image_size_x / separation_distance
        chunk_count_y = image_size_y / separation_distance
        for i in range(math.floor(chunk_count_y)-2):
            for j in range(math.floor(chunk_count_x)-2):
                start_y = i * separation_distance
                start_x = j * separation_distance            

    @staticmethod
    def load_png_white_transparency(image: np.ndarray) -> np.ndarray:
        if image.shape[-1] == 4:
            b, g, r, a = cv2.split(image)

            white_bg = np.ones_like(a, dtype=np.uint8) * 255

            b = cv2.bitwise_or(b, white_bg, mask=a)
            g = cv2.bitwise_or(g, white_bg, mask=a)
            r = cv2.bitwise_or(r, white_bg, mask=a)

            image_white_bg = cv2.merge([b, g, r])
        else:
            image_white_bg = image
        return image_white_bg

    @staticmethod
    def auto_rotate_full_image():
        