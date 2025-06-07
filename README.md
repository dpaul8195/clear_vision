# 🔍 Clear Vision: Deep Generative Image Restoration

Clear Vision is a deep learning-based image restoration project that aims to reconstruct high-quality images from degraded inputs using generative models. This system simulates real-world image corruptions (noise, compression artifacts, occlusion, etc.) and restores them using powerful models like Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Diffusion Models.

---

## 📌 Problem Statement

Image degradation is a common challenge in real-world data. Factors such as noise, lossy compression, occlusions, and missing pixels significantly reduce image quality. The goal of this project is to develop an image restoration pipeline that:
- Simulates realistic image degradation.
- Restores degraded images to high-fidelity outputs using deep generative models.
- Evaluates restoration quality using standard metrics.
- Offers a user-friendly interface for image upload and restoration preview.

---

## ✅ Deliverables

- ✔️ Selenium-based image scraper for dataset generation.
- ✔️ Custom image corruption module to simulate real-world degradation.
- ✔️ Implementation of one or more restoration models:
  - Variational Autoencoder (VAE)
  - Generative Adversarial Network (GAN)
  - Diffusion Model
- ✔️ Evaluation metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - Inference latency measurement
- ✔️ Web interface (Streamlit) or Mobile interface (Flutter) for real-time testing.

---

## 🧪 Tech Stack / Frameworks

| Component        | Stack                  |
|------------------|------------------------|
| Machine Learning | PyTorch / TensorFlow   |
| Web Interface    | Streamlit              |
| Web Scraping     | Selenium               |
| Evaluation       | NumPy, scikit-image, LPIPS |

---

## 📂 Dataset

- **Source**: Scraped using a Selenium-based automated crawler.
- **Structure**:
  - `train/`: Clean images for training
  - `test/`: Clean images for testing
  - `corrupted/`: Corresponding degraded versions

---

## 🧠 Models

### ✅ Variational Autoencoder (VAE)
- Latent space reconstruction  
- Optimized for pixel-wise loss

### ✅ Generative Adversarial Network (GAN)
- Generator-Discriminator framework  
- Superior perceptual quality with adversarial loss

### ✅ Diffusion Model
- Iterative denoising process  
- State-of-the-art performance on image generation/restoration

---

## 📊 Evaluation Metrics

| Metric              | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| **PSNR**            | Measures pixel-level similarity (higher is better)                |
| **SSIM**            | Structural similarity index for perceptual quality                |
| **LPIPS**           | Learned perceptual similarity using deep features                 |
| **Inference Latency** | Time taken to restore one image (evaluates speed of inference)     |

## 🖥 Web Interface (Streamlit)

The Streamlit-based web interface allows users to:

- 📤 Upload a degraded image  
- 👁️ Preview the restored output  
- 📊 Compare restoration quality metrics  
- 💾 Download the restored result

### ▶️ To Run the Streamlit App:

```bash
streamlit run app.py