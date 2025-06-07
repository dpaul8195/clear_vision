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


# Function to display multiple image pairs
def display_image_comparison(dataset_dir="dataset", num_samples=10):
    """
    Original aur corrupted images ke 10 pairs ko display karta hai

    Parameters:
    - dataset_dir: Dataset ka main directory
    - num_samples: Kitne image pairs dikhane hain
    """
    # Check if dataset exists
    if not os.path.isdir(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found!")
        return

    # Check for original and corrupted directories
    original_dir = os.path.join(dataset_dir, "original")
    corrupted_dir = os.path.join(dataset_dir, "corrupted")

    if not os.path.isdir(original_dir) or not os.path.isdir(corrupted_dir):
        print("Original ya corrupted directory nahi mili!")
        # Try train directory instead
        original_dir = os.path.join(dataset_dir, "train", "original")
        corrupted_dir = os.path.join(dataset_dir, "train", "corrupted")

        if not os.path.isdir(original_dir) or not os.path.isdir(corrupted_dir):
            print("Train directory mein bhi original ya corrupted folders nahi mile!")
            return

    # Get list of image files
    original_files = os.listdir(original_dir)
    corrupted_files = os.listdir(corrupted_dir)

    # Find common files (jo dono directories mein hain)
    common_files = list(set(original_files).intersection(set(corrupted_files)))

    if not common_files:
        print("Koi common image files nahi mili!")
        return

    # Select random samples
    selected_files = random.sample(common_files, min(num_samples, len(common_files)))

    # Create figure for display
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 3))

    # Check for corruption info directory
    info_dir = os.path.join(dataset_dir, "corrupted_info")
    if not os.path.isdir(info_dir):
        info_dir = os.path.join(dataset_dir, "train", "corrupted_info")

    # Display each pair
    for i, img_file in enumerate(selected_files):
        # Load images
        original_img = Image.open(os.path.join(original_dir, img_file))
        corrupted_img = Image.open(os.path.join(corrupted_dir, img_file))

        # Get corruption info if available
        corruption_info = ""
        base_name, _ = os.path.splitext(img_file)
        info_path = (
            os.path.join(info_dir, f"{base_name}.txt")
            if os.path.isdir(info_dir)
            else None
        )

        if info_path and os.path.exists(info_path):
            with open(info_path, "r") as f:
                corruption_info = f.read().strip()

                # Format multi-line corruption info for better display
                if "Multiple corruptions applied:" in corruption_info:
                    lines = corruption_info.split("\n")
                    header = lines[0]
                    corruptions = lines[1:]
                    # Limit to first 3 corruptions in title with ellipsis if more
                    if len(corruptions) > 3:
                        corruption_info = (
                            header + "\n" + "\n".join(corruptions[:3]) + "\n..."
                        )

        # Display images
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(corrupted_img)
        if not corruption_info:
            title = "Corrupted Image"
        else:
            # For multi-line titles, use a smaller font size
            axes[i, 1].set_title(f"Corrupted Image", fontsize=10)
            # Add corruption info as text below the title
            if "Multiple corruptions applied:" in corruption_info:
                lines = corruption_info.split("\n")
                corruption_text = "\n".join(lines)
                axes[i, 1].text(
                    0.5,
                    -0.05,
                    corruption_text,
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=axes[i, 1].transAxes,
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7),
                )
            else:
                title = f"Corrupted: {corruption_info}"
                axes[i, 1].set_title(title, fontsize=9)

        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Add more space between rows for corruption text
    plt.show()

    print(f"{len(selected_files)} image pairs successfully displayed!")
