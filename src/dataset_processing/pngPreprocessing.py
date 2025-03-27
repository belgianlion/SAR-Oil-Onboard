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
                edges = cv2.Canny(resized_alpha, 50, 150)
                cv2.imshow("edges", edges)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if edges is not None and edges.any():
                    hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=50, maxLineGap=0)  # Adjusted parameters for better detection of thin lines
                else:
                    hough_lines = None
                if hough_lines is not None:
                    color_alpha = cv2.cvtColor(resized_alpha, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image
                    for line in hough_lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(color_alpha, (x1, y1), (x2, y2), (0, 255, 0), 20)
                    cv2.imshow("Processed PNG", color_alpha)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            return image
        except Exception as e:
            print(f"Error opening PNG file with OpenCV: {e}")
            return None