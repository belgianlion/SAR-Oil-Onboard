import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import cv2

from src.datasets.imageChipCollection import ImageChipCollection, ImageChip
from src.dataset_processing.pngPreprocessing import PNGProcessing
from src.models.ResNet.customModel import CustomModel
from src.spline_extraction.bSplineExtraction import BSplineExtraction
from src.algorithmic_segmentation.algorithms.normalizationForSegmentation import NormalizationForSegmentation
from src.algorithmic_segmentation.algorithms.thresholdSegmentation import ThresholdSegmentation
from src.algorithmic_segmentation.algorithms.gaussianBlurForSegmentation import GaussianBlurForSegmentation
from src.algorithmic_segmentation.algorithms.adaptiveThresholdSegmentation import AdaptiveThresholdSegmentation
from src.algorithmic_segmentation.algorithms.cascadedSegmentation import CascadedSegmentation
from src.algorithmic_segmentation.algorithms.adaptiveOtsuSegmentation import AdaptiveOtsuSegmentation
from src.algorithmic_segmentation.algorithms.watershedSegmentation import WatershedSegmentation
from src.algorithmic_segmentation.algorithms.cannyEdgeDetection import CannyEdgeDetection
from src. algorithmic_segmentation.core.batchAlgorithmRunner import BatchAlgorithmRunner
from src.algorithmic_segmentation.algorithms.otsuSegmentation import OtsuSegmentation
from src.datasets.sarImageDataset import SARImageDataset
from src.models.model import Model
from src.tui.selectMenu import SelectMenu
from src.tui.tui import TUI
from src.tui.tuiCore import TUICore
from src.tui.deepLearningTui import DeepLearningTUI

app_path = "/Users/mitchellsylvia/SAR-Oil-Onboard/"

def combine_chips(chips: ImageChipCollection) -> np.ndarray:
    final_image = np.zeros((chips.whole_image_height, chips.whole_image_width), dtype=np.uint8)
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

    image = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\UAVSAR_IMG_XML\output.png"

    chips = PNGProcessing.split_into_chips(image)

    model = CustomModel(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", num_epochs=10)

    # Run the chip through the model
    output = model.run_on_collection(chips, r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet\resnet18_sar_cross_val.pt")

    # Filter chips to only include those with contains_oil = 1
    combined_chips = combine_chips(output)

    cv2.imshow("Filtered Chips", combined_chips)
    cv2.waitKey(0)

    # Display or process the filtered chips
    

    # model = CustomModel(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", num_epochs=10)
    # segmentation_methods =  [CascadedSegmentation(NormalizationForSegmentation(), ThresholdSegmentation(55))]
    # batchAlgorithmRunner = BatchAlgorithmRunner(app_path, segmentation_methods, tuiCore, samples=5)
    # bSplineExtractor = BSplineExtraction()
    # tui = TUI(tuiCore, app_path, batchAlgorithmRunner, bSplineExtractor, model)
    # tui.startup()

