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
    def __init__(self, base_path: str, segmentation_methods: List[SegmentationMethod], tuiCore, samples: int = None):
        self.output_base_path = os.path.join(base_path, "results")
        self.segmentation_methods = segmentation_methods
        self.tuiCore = tuiCore
        self.samples = samples
    
    def run(self, images: List[ImageSegmentationFormat]):
        print(self.tuiCore.create_header("Batch Algorithmic Segmentation"))
        print(self.tuiCore.create_message(f"Results will be saved to {self.output_base_path}"))
        print(self.tuiCore.create_message("This will take a while, please be patient."))

        if self.samples is not None:
            images = images[:self.samples]
            print(self.tuiCore.create_message(f"Only processing first {self.samples} samples of {len(images)} images."))
        else:
            print(self.tuiCore.create_message(f"Processing all {len(images)} images."))

        def process_image(img, filename: str):
            for segmentation_method in self.segmentation_methods:
                print(self.tuiCore.create_message(f"Processing image {filename} with {segmentation_method}"))
                segmented_img = segmentation_method.segment(img)
                save_dir = os.path.join(self.output_base_path, segmentation_method.__str__())
                os.makedirs(save_dir, exist_ok=True)
                img_path = os.path.join(save_dir, f"segmented_{filename}.png")
                cv2.imwrite(img_path, segmented_img)

        for image in images:
            process_image(image.image, image.filename)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(process_image, image.image, image.filename) for image in images]
        #     concurrent.futures.wait(futures)
