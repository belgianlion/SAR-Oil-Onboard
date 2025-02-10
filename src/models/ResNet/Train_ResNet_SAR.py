import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.sarImageDataset import SARImageDataset

print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print(f'_CUDA version: ')
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device')

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='ResNet18_Weights.DEFAULT')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512, 2)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

training_dataset = SARImageDataset(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", transform=transform)
dataloader = DataLoader(training_dataset, batch_size=256, shuffle=True)

# Training loop
num_epochs = 10
losses = [0] * num_epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Zero the parameter gradients
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
    losses[epoch] = running_loss/len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

filename = "resnet18_sar.pt"
base_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet"
torch.save(model.state_dict(), f"{base_dir}/{filename}")
print(f"Finished Training! File saved as {filename}")
    