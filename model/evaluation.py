import os
import cv2
import sys
import types
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.transforms import ToTensor
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import lpips
import torchvision.transforms.functional as F

from model import setup_restorer


functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

# Metrics
loss_fn_alex = lpips.LPIPS(net='alex')  

def preprocess_for_metrics(img, size=(256, 256)):
    return cv2.resize(img, size)

def calculate_psnr(gt, pred):
    return psnr(gt, pred, data_range=255)

def calculate_ssim(gt, pred):
    return ssim(gt, pred, data_range=255, channel_axis=-1)

def calculate_lpips(gt, pred):
    gt_tensor = ToTensor()(gt).unsqueeze(0) * 2 - 1
    pred_tensor = ToTensor()(pred).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        d = loss_fn_alex(gt_tensor, pred_tensor)
    return d.item()


# Image process + evaluation
def process_and_evaluate(image_name, restorer, base_dir):
    corrupted_path = os.path.join(base_dir, "val", "corrupted", image_name)
    gt_path = os.path.join(base_dir, "val", "original", image_name)

    # Load corrupted image
    img = cv2.imread(corrupted_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Restore image
    _, _, restored_img = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

    # Load ground truth
    gt_img = cv2.imread(gt_path)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

    # Ensure size match
    gt_resized = preprocess_for_metrics(gt_img)
    restored_resized = preprocess_for_metrics(restored_img)

    # Metrics
    psnr_val = calculate_psnr(gt_resized, restored_resized)
    ssim_val = calculate_ssim(gt_resized, restored_resized)
    lpips_val = calculate_lpips(gt_resized, restored_resized)

    return psnr_val, ssim_val, lpips_val

base_dir = r"dataset"
restorer = setup_restorer()

results = []
val_folder = os.path.join(base_dir, "val", "corrupted")
for filename in os.listdir(val_folder):
    try:
        psnr_val, ssim_val, lpips_val = process_and_evaluate(filename, restorer, base_dir)
        print(f"{filename} --> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")
        results.append((filename, psnr_val, ssim_val, lpips_val))
    except Exception as e:
        print(f"Failed on {filename}: {e}")

# Average metrics
if results:
    psnrs, ssims, lpips_vals = zip(*[(r[1], r[2], r[3]) for r in results])
    print(f"\nAverage PSNR: {np.mean(psnrs):.2f}")
    print(f"Average SSIM: {np.mean(ssims):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_vals):.4f}")