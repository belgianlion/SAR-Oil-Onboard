import torch

from models.model import Model

class TUI:
    def system_startup(device_string: str):
        print(f"Mitchell Sylvia (belgianlion)'s Model Training TUI")
        print("="*50)
        print('')
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using {device_string} device")
        if device_string == "cuda":
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Available GPU devices: {torch.cuda.device_count()}")
            print(f"More information about the CUDA version can be found at https://developer.nvidia.com/cudnn")
    
    def print_model_pre_summary(model: Model):
        print("="*50)
        print(f"Pre-Training Model Summary:")
        print("-"*50)
        print(f"Model Name:{model.model_name}")
        print(f"Device to Execute: {model.device}")
        print(f"Already Trained?: {model.trained}")
        print("Layers:")
        for name, layer in model.model.named_parameters():
            print(f"Name: {name}")
            print(f"\t Requires gradients?:{layer.requires_grad}")
        print("="*50)

        

    def print_model_post_summary(model: Model):
        print("="*50)
        print(f"Post-Training Model Summary:")
        print("-"*50)
        print(f"Model Name:{model.model_name}")
        print(f"Device: {model.device}")
        print(f"Trained: {model.trained}")
        print(f"Total Losses: {model.total_losses}")
        print("="*50)
        