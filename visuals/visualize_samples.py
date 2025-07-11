# Step 5: Display some sample images to verify
import matplotlib.pyplot as plt
import random
import os
from PIL import Image

def display_samples(dataset_dir="dataset", num_samples=3):
    """Display some sample image pairs from the dataset"""

    # Get random samples from the training set
    train_originals = os.path.join(dataset_dir, "train", "original")
    train_corrupted = os.path.join(dataset_dir, "train", "corrupted")
    train_info_dir = os.path.join(dataset_dir, "train", "corrupted_info")

    if not os.path.isdir(train_originals) or not os.path.isdir(train_corrupted):
        print(f"Directory not found: {train_originals} or {train_corrupted}")
        return

    image_files = os.listdir(train_originals)
    if not image_files:
        print("No images found in training set")
        return

    # Select random samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 4 * num_samples))

    for i, img_file in enumerate(samples):
        # Load original image
        original_path = os.path.join(train_originals, img_file)
        original_img = Image.open(original_path)

        # Load corrupted image
        corrupted_path = os.path.join(train_corrupted, img_file)
        corrupted_img = Image.open(corrupted_path)

        # Get corruption info if available
        base_name, _ = os.path.splitext(img_file)
        info_path = os.path.join(train_info_dir, f"{base_name}.txt")
        corruption_info = ""
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                corruption_info = f.read().strip()

        # Display images
        if num_samples == 1:
            axes[0].imshow(original_img)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(corrupted_img)
            axes[1].set_title("Corrupted", fontsize=10)

            # Add corruption info as text below the title
            if corruption_info:
                axes[1].text(
                    0.5,
                    -0.05,
                    corruption_info,
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=axes[1].transAxes,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7),
                )
            axes[1].axis("off")
        else:
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(corrupted_img)
            axes[i, 1].set_title("Corrupted", fontsize=10)

            # Add corruption info as text below the title
            if corruption_info:
                axes[i, 1].text(
                    0.5,
                    -0.05,
                    corruption_info,
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=axes[i, 1].transAxes,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7),
                )
            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Add more space between rows for corruption text
    plt.show()


