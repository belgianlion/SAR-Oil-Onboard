from matplotlib import pyplot as plt
import numpy as np
from src.spline_extraction.bSplineExtraction import BSplineExtraction
from src.algorithmic_segmentation.core.imageSegmentationFormat import ImageSegmentationFormat
from src.tui.tuiCore import TUICore
import os
import scipy.interpolate as spi

from src.tui.textColors import TextColors
from src.tui.selectMenu import SelectMenu
from src.algorithmic_segmentation.core.batchAlgorithmRunner import BatchAlgorithmRunner
import glob
import cv2
from scipy.interpolate import interp1d

from src.tui.terminalSettings import TerminalSettings


class TUI:
    def __init__(self, tuiCore: TUICore, base_path: str, batch_algorithm_runner: BatchAlgorithmRunner, bSplineExtractor: BSplineExtraction):
        self.tuiCore = tuiCore
        self.base_path = base_path
        self.batch_algorithm_runner = batch_algorithm_runner
        self.bSplineExtractor = bSplineExtractor

    def startup(self):
        with TerminalSettings():
            TUICore.clear_terminal()
            terminal_buffer = []
            terminal_buffer.append(self.tuiCore.create_header("SAR Oil Onboard TUI"))
            terminal_buffer.append(self.tuiCore.create_message("This TUI is intended for use for Mitchell Sylvia (belgianlion)'s SAR Oil Onboard Thesis Project", TextColors.LIGHT))
            terminal_buffer.append("")
            TUICore.print_buffer(terminal_buffer)
            input("Press enter to continue...")
            terminal_buffer.clear()
            terminal_buffer.append(self.tuiCore.create_header("What would you like to do?"))
            options = ["Train a Model", "Test a Model", "Test Algorithmic Segmentation", "Test bSplineExtraction", "Exit"]
            selected_index = 0
            menu = SelectMenu(options, self.tuiCore, terminal_buffer)
            selected_index = menu.run()

            if selected_index == 0:
                print("Train a Model")
            elif selected_index == 1:
                print("Test a Model")
            elif selected_index == 2:
                TUICore.clear_terminal()
                images = []
                image_folder = os.path.join(self.base_path, "Datasets", "training_data", "1")
                image_files = glob.glob(os.path.join(image_folder, '*.jpg'))  # Adjust the extension if needed

                for image_file in image_files:
                    image_data = cv2.imread(image_file)
                    filename = os.path.splitext(os.path.basename(image_file))[0]
                    images.append(ImageSegmentationFormat(image=image_data, filename=filename))
                self.batch_algorithm_runner.run(images)
            elif selected_index == 3:
                TUICore.clear_terminal()
                results_folder = os.path.join(self.base_path, "results")
                first_folder = sorted([f for f in os.listdir(results_folder) if f != '.DS_Store'])[0]
                image_folder = os.path.join(results_folder, first_folder)
                image_files = glob.glob(os.path.join(image_folder, '*.png'))  # Adjust the extension if needed

                images = []
                for image_file in image_files:
                    image_data = cv2.imread(image_file)
                    filename = os.path.splitext(os.path.basename(image_file))[0]
                    images.append(ImageSegmentationFormat(image=image_data, filename=filename))
                    

                for image in images:
                    splines = self.bSplineExtractor.extract_spline(image.image)
                    output_folder = os.path.join(self.base_path, "spline_results")
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"{image.filename}_splines.png")

                    plt.figure()
                    plt.imshow(cv2.cvtColor(image.image, cv2.COLOR_BGR2RGB))
                    for xnew, ynew in splines:
                        plt.plot(xnew, ynew, '-', lw=2)
                    plt.title(f"Image: {image.filename}")
                    plt.savefig(output_path)
                    plt.close()


            elif selected_index == 4:
                terminal_buffer.clear()
                os._exit(0)