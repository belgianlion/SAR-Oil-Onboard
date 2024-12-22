from torchvision import transforms
import torch
import torch.nn as nn
from models.ResNet.sarimagedataset import SARImageDataset
from models.model import Model
from tui.tui import TUI


if __name__ == "__main__":
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    TUI.system_startup(device_string)
    model_name = 'resnet18'
    torch_model = torch.hub.load('pytorch/vision:v0.10.0', model_name, weights='ResNet18_Weights.DEFAULT')

    # Modify torch model to only output binary values (is or is not oil spill):
    torch_model.fc = nn.Linear(512, 2)
    

    model = Model(torch_model, device_string, ["layer4", "fc"], model_name)

    # Transform defined as resizing the image to 224x224, and creating tensor representations of the image.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SARImageDataset(r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\training_data", transform=transform)
    sample_image_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\Datasets\Samples"

    # Explained in docs at https://pytorch.org/docs/stable/generated/torch.optim.Adam.html. Adam was the recommended optimizer by GitHub Copilot. The filter simply
    # only makes modifications to the weights on parameters that have the requires_grad (requires gradient) flag set to true.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.model.parameters()), lr=0.001)
    # This is just basic Cross Entropy Loss with standard params
    criterion = nn.CrossEntropyLoss()


    TUI.print_model_pre_summary(model)
    model.train(epochs=10, dataset=dataset, batch_size=256, shuffle=True, optimizer=optimizer, criterion=criterion)


    filename = "resnet18_sar.pt"
    base_dir = r"C:\Users\belgi\OneDrive\Documents\GitHub\SAR-Oil-Onboard\src\models\ResNet"
    model.output_losses(base_dir, 10)
    TUI.print_model_post_summary(model)
    model.save(f"{base_dir}\{filename}")
    model.predict(sample_image_dir, transform)

