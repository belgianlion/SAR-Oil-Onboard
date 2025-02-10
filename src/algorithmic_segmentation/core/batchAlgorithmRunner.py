from tkinter import Image
from typing import List

import numpy as np
from src.algorithmic_segmentation.core.imageSegmentationFormat import ImageSegmentationFormat
from src.algorithmic_segmentation.algorithms.segmentationMethod import SegmentationMethod
from torchvision import transforms
import os
import cv2
import concurrent.futures

class BatchAlgorithmRunner:
    def __init__(self, base_path: str, segmentation_methods: List[SegmentationMethod], tuiCore):
        self.output_base_path = os.path.join(base_path, "results")
        self.segmentation_methods = segmentation_methods
        self.tuiCore = tuiCore
    
    def run(self, images: List[ImageSegmentationFormat]):
        print(self.tuiCore.create_header("Batch Algorithmic Segmentation"))
        print(self.tuiCore.create_message(f"Results will be saved to {self.output_base_path}"))
        print(self.tuiCore.create_message("This will take a while, please be patient."))
        def process_image(img, filename:str):
            for segmentation_method in self.segmentation_methods:
                segmented_img = segmentation_method.segment(img)
                self.save_segmented_image(segmented_img, segmentation_method.__class__.__name__, filename)

        for image in images:
            process_image(image.image, image.filename)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image.image, image.filename) for image in images]
            concurrent.futures.wait(futures)
    
    def save_segmented_image(self, img, segmentation_method_name: str, filename: str):
        save_dir = os.path.join(self.output_base_path, segmentation_method_name)
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"segmented_{filename}.png")
        cv2.imwrite(img_path, img)