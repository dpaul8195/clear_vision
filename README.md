# 🖼️ Image Restoration Using Deep Generative Models

## 🧠 Problem Statement

Image degradation is common in real-world visual data due to noise, compression artifacts, occlusion, or partial corruption. This project builds an **image restoration system** that can recover and reconstruct high-quality images from degraded inputs using **deep generative models**.

The goal is to design a model that restores corrupted images to clean, high-fidelity versions and provides a user-friendly interface for image upload, restoration, and download.

---

## 📁 Project Structure

```
image-restoration-project/
├── app.py # Streamlit web app
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── dataset/ # Original and corrupted image dataset
│ ├── original/
│ ├── corrupted/
│ ├── train/
│ ├── val/
│ ├── test/
│ └── corrupted_info/
├── model_training/
│ └── model_after_training/ # Trained RealESRGAN model
├── utils/
│ ├── image_scraper.py # Selenium-based image downloader
│ ├── corruption_utils.py # Functions to apply 10 image corruptions
│ ├── split_dataset.py # Script to split dataset into train/val/test
│ └── evaluation.py # Computes PSNR, SSIM, LPIPS, and latency
├── visuals/
│ ├── varify.py # Side-by-side visualization (original vs corrupted)
│ └── visualize_samples.py # Random dataset sample viewer
```
---

## 📦 Deliverables

- ✅ **Selenium-based image scraper** to gather raw image data
- ✅ **Custom image corruption module** (`corruption_utils.py`) with:
  - Gaussian Noise  
  - Salt & Pepper Noise  
  - Speckle Noise  
  - Mild Blur  
  - Motion Blur  
  - JPEG Compression  
  - Low Brightness  
  - Low Contrast  
  - Occlusion  
  - Compression Artifacts  
- ✅ **CelebA-HQ dataset** as the clean image base
- ✅ Train/Validation/Test splits with separate `original/` and `corrupted/` folders
- ✅ Visual comparison scripts (`visuals/`) to view corrupted vs original side-by-side
- ✅ Model training using **RealESRGANer + RRDBNet**
- ✅ Streamlit-based **web application** to upload, restore, and download images
- ✅ `evaluation.py` script to compute:
  - PSNR  
  - SSIM  
  - LPIPS  
  - Inference Latency  

---

## 🛠️ Tech Stack / Frameworks

### 💻 Machine Learning
- PyTorch  
- RealESRGAN  
- RRDBNet

### 🌐 Web Interface
- Streamlit

### 🕸 Web Scraping
- Selenium

---

## 📊 Evaluation Metrics

| Metric        | Description |
|---------------|-------------|
| **PSNR**      | Measures the signal-to-noise ratio between restored and original images |
| **SSIM**      | Structural Similarity Index for perceptual quality |
| **LPIPS**     | Learned Perceptual Image Patch Similarity for feature-space accuracy |
| **Latency**   | Time taken for model inference during image restoration |

Use the following command to run evaluations:

```bash
python utils/evaluation.py

```
---

## 🚀 How It Works
---
1. **Image Collection**  
   - Use `utils/image_scraper.py` to collect original images from the web.

2. **Dataset Preparation**  
   - Corrupt images using `utils/corruption_utils.py`.  
   - Split the dataset using `utils/split_dataset.py`.

3. **Training**  
   - Train your RealESRGANer model on the dataset.  
   - Save checkpoints in `model_training/model_after_training/`.

4. **Visualization**  
   - View original vs corrupted samples using:
     - `visuals/varify.py`  
     - `visuals/visualize_samples.py`

5. **Web App**  
   - Launch the interface using:
     ```bash
     streamlit run app.py
     ```
   - Upload a corrupted image to preview and download the restored version.

---
## ✅ Example Commands

---
```bash
# Run the web application
streamlit run app.py

# Evaluate model performance
python utils/evaluation.py

# Visualize original vs corrupted samples
python visuals/varify.py
```
---
