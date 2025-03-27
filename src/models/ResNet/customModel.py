import os
from typing import Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.datasets.sarImageDataset import SARImageDataset
from src.models.ResNet.baseModel import BaseModel


class CustomModel(BaseModel):
    def __init__(self, dataset_path: str, num_epochs: int = 10):
        super(CustomModel, self).__init__()
        self.num_epochs = num_epochs
        self.dataset_path = dataset_path
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(512, 2)
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def train(self):

        transform = v2.Compose([
            v2.RandomRotation(degrees=15, fill=(255, 255, 255)),  # Background will be white
            v2.RandomHorizontalFlip(),
            v2.Resize((224, 224)),
            v2.ToTensor(),
        ])


        training_dataset = SARImageDataset(self.dataset_path, transform=transform)
        dataloader = DataLoader(training_dataset, batch_size=256, shuffle=True)

        device, criterion, optimizer = self.__training_standard_setup()

        # Training loop
        losses = self.__training_loop(dataloader, criterion, optimizer, device)

        plt.plot(range(self.num_epochs), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.show()

        filename = "resnet18_sar.pt"
        base_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet"
        torch.save(self.model.state_dict(), f"{base_dir}/{filename}")
        print(f"Finished Training! File saved as {filename}")

    def train_with_k_fold(self, k: int = 5, seed: int = None, shuffle: bool = True):
        k_fold = KFold(n_splits=k, shuffle=shuffle)
        if seed is not None:
            k_fold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)

        transform = v2.Compose([
            v2.RandomRotation(degrees=15, fill=(255, 255, 255)),  # Background will be white
            v2.RandomHorizontalFlip(),
            v2.Resize((224, 224)),
            v2.ToTensor(),
        ])

        training_dataset = SARImageDataset(self.dataset_path, transform=transform)

        device, criterion, optimizer = self.__training_standard_setup()

        # K-Fold Cross Validation
        accuracies = []
        for fold, (train_idx, val_idx) in enumerate(k_fold.split(training_dataset)):
            print(f"Fold {fold + 1}/{k}")
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = DataLoader(training_dataset, batch_size=256, sampler=train_sampler)
            val_loader = DataLoader(training_dataset, batch_size=256, sampler=val_sampler)

            # Training loop for this fold
            losses = self.__training_loop(train_loader, criterion, optimizer, device)
            val_loss, accuracy = self.__validation_loop(val_loader, criterion, device)
            accuracies.append(accuracy)

            print(f"Finished Fold {fold + 1}/{k}!")
            plt.plot(range(self.num_epochs), losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss over Epochs (Fold {fold + 1})')
            plt.show()
        print(f"Average Accuracy over {k} folds: {sum(accuracies) / k}%")
        print(f"Standard Deviation of Accuracy over {k} folds: {torch.std(torch.tensor(accuracies)).item()}%")

        filename = "resnet18_sar_cross_val.pt"
        base_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet"
        torch.save(self.model.state_dict(), f"{base_dir}/{filename}")
        

    def run(self, test_dataset_path: str, model_weight_path: str):
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {device} device')
        self.model.load_state_dict(torch.load(model_weight_path, weights_only=False))
        self.model = self.model.to(device)
        self.model.eval()

        # Define the image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Function to load and preprocess the image
        def load_image(image_path):
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            return image

        # Run the model on each image in the directory
        for image_name in os.listdir(test_dataset_path):
            image_path = os.path.join(test_dataset_path, image_name)
            image = load_image(image_path).to(device)
            with torch.no_grad():
                output = self.model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted = torch.argmax(probabilities, dim=1).item()
                percentage = probabilities[0][predicted].item() * 100
            print(f"Output for {image_name}: {predicted} with {percentage:.2f}% confidence")

        # PRIVATE METHODS

    def __training_standard_setup(self) -> Tuple[torch.device, nn.CrossEntropyLoss, torch.optim.Optimizer]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {device} device')

        self.model.to(device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        return device, criterion, optimizer


    def __training_loop(self, dataloader, criterion, optimizer, device):
        losses = [0] * self.num_epochs
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, labels, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                # Zero the parameter gradients
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item()
            losses[epoch] = running_loss/len(dataloader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {running_loss/len(dataloader)}")
        return losses
    
    def __validation_loop(self, dataloader, criterion, device):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {avg_loss}, Accuracy: {accuracy}%')
        return avg_loss, accuracy


    