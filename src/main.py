import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import cv2

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
        


if __name__ == "__main__":
    tuiCore = TUICore()
    # Here are the list of segmentation methods I have run before: [OtsuSegmentation((21, 21)), OtsuSegmentation(None), AdaptiveOtsuSegmentation(), AdaptiveOtsuSegmentation((9, 9), adaptive_block_size=81, mean_bias=10), WatershedSegmentation(9, 3)]
    # segmentation_methods = [CascadedSegmentation(AdaptiveOtsuSegmentation((9, 9), adaptive_block_size=81, mean_bias=10), CannyEdgeDetection())]
    # CascadedSegmentation(GaussianBlurForSegmentation((9,9)), AdaptiveThresholdSegmentation(block_size=17, c=3), AdaptiveOtsuSegmentation((0, 0), adaptive_block_size=101, mean_bias=10))

    jpg_image_path = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\UAVSAR_IMG_XML\output.jpg"
    png_image_path = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\UAVSAR_IMG_XML\output.png"

    oil_spill_image = OilSpillImage(png_image_path, jpg_image_path, 0.4, with_rotation=True, padding=5)
    oil_spill_image.split_jpg_into_chips(chip_size=400, overlap_percentage=0.5)

    model = CustomModel(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", num_epochs=10)

    # Run the chip through the model
    model.run_on_collection(oil_spill_image, r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet\resnet18_sar_cross_val.pt")

    # Extract Splines
    bspline_extractor = BSplineExtraction()

    chips = oil_spill_image.classed_chips

    for chip in chips:
        if chip.contains_oil == 1:
            spline = bspline_extractor.extract_spline(chip.image)
            chip.image = BSplineExtraction.try_add_splines_to_image(chip.image, spline)

    
    # Filter chips to only include those with contains_oil = 1
    combined_chips = combine_chips(oil_spill_image)

    cv2.imshow("Filtered Chips", combined_chips)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display or process the filtered chips
    

    # model = CustomModel(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", num_epochs=10)
    # segmentation_methods =  [CascadedSegmentation(NormalizationForSegmentation(), ThresholdSegmentation(55))]
    # batchAlgorithmRunner = BatchAlgorithmRunner(app_path, segmentation_methods, tuiCore, samples=5)
    # bSplineExtractor = BSplineExtraction()
    # tui = TUI(tuiCore, app_path, batchAlgorithmRunner, bSplineExtractor, model)
    # tui.startup()

