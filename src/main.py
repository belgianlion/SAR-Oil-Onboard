import sys
import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import cv2

from src.models.Segmentation.segmentationModel import SegmentationModel
from src.spline_extraction.bSplineExtraction import BSplineExtraction
from src.datasets.oilSpillImage import OilSpillImage
from src.datasets.imageChipCollection import ImageChipCollection, ImageChip
from src.dataset_processing.pngPreprocessing import PNGProcessing
from src.models.ResNet.customModel import CustomModel
from src.tui.tui import TUI
from src.tui.tuiCore import TUICore

app_path = "/Users/mitchellsylvia/SAR-Oil-Onboard/"

def combine_chips(oil_spill_image: OilSpillImage) -> np.ndarray:
    chips = oil_spill_image.classed_chips
    final_image = np.zeros((oil_spill_image.jpg_image.height(), oil_spill_image.jpg_image.width(), 3), dtype=np.uint8)
    for chip in chips:
        if chip.contains_oil == 1:
            x_start = chip.x_start
            y_start = chip.y_start
            chip_image = chip.image
            final_image[y_start:y_start + chip_image.shape[0], x_start:x_start + chip_image.shape[1]] = chip_image
    return final_image

def combine_masks(oil_spill_image: OilSpillImage) -> np.ndarray:
    chips = oil_spill_image.chips
    final_mask = np.zeros((oil_spill_image.jpg_image.height(), oil_spill_image.jpg_image.width()), dtype=np.uint8)
    for chip in chips:
        if chip.contains_oil == 1 and chip.mask is not None:
            x_start = chip.x_start
            y_start = chip.y_start
            mask = chip.mask
            final_mask[y_start:y_start + mask.shape[0], x_start:x_start + mask.shape[1]] = mask
    return final_mask
        


if __name__ == "__main__":
    tuiCore = TUICore()
    # Here are the list of segmentation methods I have run before: [OtsuSegmentation((21, 21)), OtsuSegmentation(None), AdaptiveOtsuSegmentation(), AdaptiveOtsuSegmentation((9, 9), adaptive_block_size=81, mean_bias=10), WatershedSegmentation(9, 3)]
    # segmentation_methods = [CascadedSegmentation(AdaptiveOtsuSegmentation((9, 9), adaptive_block_size=81, mean_bias=10), CannyEdgeDetection())]
    # CascadedSegmentation(GaussianBlurForSegmentation((9,9)), AdaptiveThresholdSegmentation(block_size=17, c=3), AdaptiveOtsuSegmentation((0, 0), adaptive_block_size=101, mean_bias=10))

    jpg_image_path = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\UAVSAR_IMG_XML\output.jpg"
    png_image_path = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\UAVSAR_IMG_XML\output.png"
    xml_jpg_image_path = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\UAVSAR_IMG_XML\output.jpg.aux.xml"

    oil_spill_image = OilSpillImage(png_image_path, jpg_image_path, xml_jpg_image_path, 0.4, with_rotation=True, padding=10)
    oil_spill_image.split_jpg_into_chips(chip_size=400, overlap_percentage=0.5)

    model = SegmentationModel(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\Segmentation\weights.pt")

    # Run the chip through the model
    model.run_on_collection(oil_spill_image.chips)

    # Extract Splines
    bspline_extractor = BSplineExtraction()

    chips = oil_spill_image.classed_chips

    # for chip in chips:
    #     if chip.contains_oil == 1:
    #         spline = bspline_extractor.extract_spline(chip.image)
    #         chip.image = BSplineExtraction.try_add_splines_to_image(chip.image, spline)

    
    # Filter chips to only include those with contains_oil = 1
    combined_mask = combine_masks(oil_spill_image)
    masked_image = cv2.bitwise_and(oil_spill_image.jpg_image.image, oil_spill_image.jpg_image.image, mask=combined_mask)
    scaled_mask = combined_mask * 255
    _, binary_mask = cv2.threshold(scaled_mask, 1, 255, cv2.THRESH_BINARY)
    inverted_mask = cv2.bitwise_not(binary_mask)
    blurred_mask = cv2.GaussianBlur(inverted_mask, (9,9), 0)
    splines = bspline_extractor.extract_spline(blurred_mask)
    spline_image = BSplineExtraction.try_add_splines_to_image(blurred_mask, splines)
    lat_long_data = oil_spill_image.map_location(splines)

    # Get size in bytes
    lat_long_size = lat_long_data.nbytes if isinstance(lat_long_data, np.ndarray) else sys.getsizeof(lat_long_data)
    print(f"Spline image size: {spline_image.nbytes} bytes")  # For numpy arrays
    print(f"Lat long data size: {lat_long_size} bytes")
    print(f"Percentage reduction of: {((spline_image.nbytes-lat_long_size)/spline_image.nbytes)*100}")

    cv2.imshow("Splines", spline_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display or process the filtered chips
    

    # model = CustomModel(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", num_epochs=10)
    # segmentation_methods =  [CascadedSegmentation(NormalizationForSegmentation(), ThresholdSegmentation(55))]
    # batchAlgorithmRunner = BatchAlgorithmRunner(app_path, segmentation_methods, tuiCore, samples=5)
    # bSplineExtractor = BSplineExtraction()
    # tui = TUI(tuiCore, app_path, batchAlgorithmRunner, bSplineExtractor, model)
    # tui.startup()

