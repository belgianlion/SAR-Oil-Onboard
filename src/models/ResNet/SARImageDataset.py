from PIL import Image
from torch.utils.data import Dataset

import os

class SARImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        for label in os.listdir(img_dir):
            print(f"Label: {label}")
            label_dir = os.path.join(img_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    self.img_labels.append((os.path.join(label_dir, img_file), int(label)))
        print(f"Found {len(self.img_labels)} images in {img_dir}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label