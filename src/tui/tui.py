from src.algorithmic_segmentation.core.imageSegmentationFormat import ImageSegmentationFormat
from src.tui.tuiCore import TUICore
import os

from src.tui.textColors import TextColors
from src.tui.selectMenu import SelectMenu
from src.algorithmic_segmentation.core.batchAlgorithmRunner import BatchAlgorithmRunner
import glob
import cv2

from src.tui.terminalSettings import TerminalSettings


class TUI:
    def __init__(self, tuiCore: TUICore, base_path: str):
        self.tuiCore = tuiCore
        self.base_path = base_path

    def startup(self, batch_algorithm_runner: BatchAlgorithmRunner):
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
            options = ["Train a Model", "Test a Model", "Test Algorithmic Segmentation", "Exit"]
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
                batch_algorithm_runner.run(images)
            elif selected_index == 3:
                terminal_buffer.clear()
                os._exit(0)