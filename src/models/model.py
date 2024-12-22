from typing import List
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class Model:
    def __init__(self, model: object, device_string: str, layers_to_train: List[str], model_name: str = None):
        self.model = model
        self.model_name = model_name
        # Enable GPU acceleration if a cuda core enabled GPU is available for use.
        self.device = torch.device(device_string)
        self.__enable_train_layers(layers_to_train)
        self.model.to(self.device)
        self.trained = False
        self.total_losses = []

    def save(self, path: str):
        if not self.trained:
            raise Exception("Model should be trained before saving. \
                            Without training, the model will be saved in an untrained state. \
                            If you want to save the model in an untrained state, use the torch.save() function.")
        torch.save(self.model.state_dict(), path)
        print(f"model saved at {path}")

    def output_losses(self, path: str, epoches: int):
        plt.plot(range(epoches), self.total_losses)
        x_axis = np.arange(0, epoches, 1)
        plt.xlabel('Epoch')
        plt.xticks(x_axis)
        plt.yscale('log')
        plt.ylabel('Running Loss')
        plt.grid(True, which='both', ls="--")
        plt.title('Training Loss at Each Epoch')
        plt.savefig(f"{path}\{self.model_name}_training_loss.png")

    def __load_test_image(self, image_path: str, transform):
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        # We need to add the batch dimension, as it was
        # utilized in training, but since we are not batching
        # evaluation, it should be set to 0.
        image = image.unsqueeze(0)
        return image

    def predict(self, image_dir: str, transform):
        self.model.eval()
        # Run the model on each image in the directory
        
        #TODO: create a dataset class like SARImageDataset for example data.
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            # load images and send them to the specified device
            image = self.__load_test_image(image_path, transform).to(self.device)
            with torch.no_grad():
                output = self.model(image)
                # Help from Copilot for this:
                probabilities = torch.nn.functional.softmax(output, dim=1)
                # From https://discuss.pytorch.org/t/get-probabilities-from-resnet-output/97445: The main reason for
                # argmax is to compress predictions on a 0-1 scale to the binary 1 or 0 output
                predicted = torch.argmax(probabilities, dim=1).item()
                # gives percentage in 0-100 range, from tensor probailities on dim 1
                percentage = probabilities[0][predicted].item() * 100
            expected_class = image_name.split('_')[7].split('.')[0]
            print(f"Output for {image_name}: {predicted} (expected {expected_class}) with {percentage:.2f}% confidence")
        

    def train(self, epochs: int, dataset: Dataset, batch_size: int, shuffle: bool, optimizer: object, criterion: object):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        self.model.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for epoch in range(epochs):
            running_loss = float(0)
            # TQDM is a library that adds a progress bar, which is useful for my TUI.
            for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # From my research, zero_grad() resets the gradients between subsequent batches of the dataloader
                device_images, device_labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(device_images)

                # The criterion calculates the loss
                loss = criterion(outputs, device_labels)
                loss.backward()
                # The optimizer propagates the loss defined by the criterion
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} loss: {running_loss}")
            self.total_losses.append(running_loss)
        self.trained = True



    def __enable_train_layers(self, layers_to_train: List[str]):

        # Enable all layers for training if layers_to_train is None.
        # The default should be to train the model from scratch.
        for param in self.model.parameters():
            param.requires_grad = False
        if layers_to_train is None:
            for param in self.model.named_parameters():
                param.requires_grad = True
            return
        # Enable training for the layers specified in layers_to_train.
        # this is mostly used for transfer learning / fine-tuning
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layers_to_train):
                param.requires_grad = True
            else:
                param.requires_grad = False