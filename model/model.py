from os.path import exists
import os
import sys
sys.path.append(r"C:\Users\Debabrata Paul\Desktop\clear_vision")

from utils.preprocessing import create_image_restoration_dataset
import sys
import types
import torch
import torchvision.transforms.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer




os.makedirs("model_after_training", exist_ok=True)


# Create a simple rgb_to_grayscale function
def rgb_to_grayscale(img):
    return F.rgb_to_grayscale(img)


# Create a module for torchvision.transforms.functional_tensor
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Add this module to sys.modules
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from basicsr.archs.rrdbnet_arch import RRDBNet
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)


import torch.nn as nn
pixel_loss = nn.L1Loss()
# Optionally, add:
# - Perceptual loss (e.g., VGG-based features)
# - Adversarial loss (if using GAN)


from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def setup_restorer():
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    model_path = "model_after_training/RealESRGAN_x2plus.pth"
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )
    return upsampler

def process_image(img_path, restorer, output_path=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use restorer (which is just RealESRGANer with RRDBNet)
    restored_img = restorer.enhance(img, outscale=2)[0]

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(restored_img)
    plt.title("Restored")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return img, restored_img

base_dir = r"dataset" # paste your directory

create_image_restoration_dataset(base_dir)


restorer = setup_restorer()

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr = Image.open(lr_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        return lr, hr

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])



# Load data
train_dataset = FaceDataset("dataset/train/corrupted", "dataset/train/original", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

for epoch in range(10):  # or more
    model.train()
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        loss = criterion(pred, hr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model_after_training/RealESRGAN_x2plus.pth")
