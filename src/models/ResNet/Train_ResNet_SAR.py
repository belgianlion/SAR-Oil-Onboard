import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 2)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
print(model)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

class SARImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        for label in os.listdir(img_dir):
            label_dir = os.path.join(img_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.img_labels.append((os.path.join(label_dir, img_file), int(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

training_dataset = SARImageDataset("/Users/mitchellsylvia/SAR-Oil-Onboard/Datasets/CSIRO_Sentinel-1_SAR_image_dataset/training_data/", transform=transform)
dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

filename = "resnet18_sar.pt"
torch.save(model.state_dict(), f"/Users/mitchellsylvia/SAR-Oil-Onboard/src/models/ResNet/{filename}")
print(f"Finished Training! File saved as {filename}")
    