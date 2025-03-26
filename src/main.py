from torchvision import transforms
import torch
import torch.nn as nn

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

if __name__ == "__main__":
    tuiCore = TUICore()
    # Here are the list of segmentation methods I have run before: [OtsuSegmentation((21, 21)), OtsuSegmentation(None), AdaptiveOtsuSegmentation(), AdaptiveOtsuSegmentation((9, 9), adaptive_block_size=81, mean_bias=10), WatershedSegmentation(9, 3)]
    # segmentation_methods = [CascadedSegmentation(AdaptiveOtsuSegmentation((9, 9), adaptive_block_size=81, mean_bias=10), CannyEdgeDetection())]
    # CascadedSegmentation(GaussianBlurForSegmentation((9,9)), AdaptiveThresholdSegmentation(block_size=17, c=3), AdaptiveOtsuSegmentation((0, 0), adaptive_block_size=101, mean_bias=10))
    segmentation_methods =  [CascadedSegmentation(NormalizationForSegmentation(), ThresholdSegmentation(55))]
    batchAlgorithmRunner = BatchAlgorithmRunner(app_path, segmentation_methods, tuiCore, samples=5)
    bSplineExtractor = BSplineExtraction()
    tui = TUI(tuiCore, app_path, batchAlgorithmRunner, bSplineExtractor)
    tui.startup()

    # deepLearningTui = DeepLearningTUI(tuiCore)
    # device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    # deepLearningTui.deep_learning_startup(device_string)
    # model_name = 'resnet18'
    # torch_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, weights='ResNet18_Weights.DEFAULT')

    # # Modify torch model to only output binary values (is or is not oil spill):
    # torch_model.fc = nn.Linear(512, 2)
    

    # model = Model(torch_model, device_string, ["layer4", "fc"], model_name)

    # # Transform defined as resizing the image to 224x224, and creating tensor representations of the image.
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])

    # dataset = SARImageDataset(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", transform=transform)
    # sample_image_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\Samples"

    # # Explained in docs at https://pytorch.org/docs/stable/generated/torch.optim.Adam.html. Adam was the recommended optimizer by GitHub Copilot. The filter simply
    # # only makes modifications to the weights on parameters that have the requires_grad (requires gradient) flag set to true.
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.model.parameters()), lr=0.001)
    # # This is just basic Cross Entropy Loss with standard params
    # criterion = nn.CrossEntropyLoss()


    # deepLearningTui.print_model_pre_summary(model)
    # model.train(epochs=10, dataset=dataset, batch_size=256, shuffle=True, optimizer=optimizer, criterion=criterion)


    # filename = "resnet18_sar.pt"
    # base_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet"
    # model.output_losses(base_dir, 10)
    # deepLearningTui.print_model_post_summary(model)
    # model.save(f"{base_dir}\{filename}")
    # model.predict(sample_image_dir, transform)

