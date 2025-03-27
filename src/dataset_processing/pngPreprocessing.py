import cv2
import numpy as np



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
                _, _, _, a = cv2.split(image)
                if a.shape[0] > 10000 or a.shape[1] > 10000:  # Check if dimensions are too large
                    resized_alpha = cv2.resize(a, (0, 0), fx=0.1, fy=0.1)  # Resize alpha channel to half its size
                else:
                    resized_alpha = a  # Use original alpha channel if dimensions are manageable
                color_alpha = cv2.cvtColor(resized_alpha, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image
                cv2.imshow("alpha", color_alpha)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                
                cv2.imshow("Processed PNG", color_alpha)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return image
        except Exception as e:
            print(f"Error opening PNG file with OpenCV: {e}")
            return None