import os
import matplotlib.pyplot as plt
from PIL import Image
import random


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


def validate_dataset_structure(dataset_dir="dataset"):
    """Verify dataset structure and count images in each split"""
    structure = {}
    splits = ["train", "val", "test"]
    image_types = ["original", "corrupted"]

    # Check main directories
    for img_type in image_types:
        main_path = os.path.join(dataset_dir, img_type)
        if os.path.exists(main_path) and os.path.isdir(main_path):
            files = [
                f
                for f in os.listdir(main_path)
                if os.path.isfile(os.path.join(main_path, f))
            ]
            structure[f"main_{img_type}"] = len(files)
        else:
            structure[f"main_{img_type}"] = 0

    # Check split directories
    for split in splits:
        structure[split] = {}
        for img_type in image_types:
            path = os.path.join(dataset_dir, split, img_type)
            if os.path.exists(path) and os.path.isdir(path):
                files = [
                    f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                ]
                structure[split][img_type] = len(files)
            else:
                structure[split][img_type] = 0

    # Print summary
    print("Dataset Structure Validation:")
    print("=" * 50)

    # Check main directories
    print("\nMain directories:")
    for img_type in image_types:
        count = structure[f"main_{img_type}"]
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {img_type}: {count} images")

    # Check split directories
    print("\nSplit directories:")
    for split in splits:
        print(f"  {split}:")
        for img_type in image_types:
            count = structure[split][img_type]
            status = "✓" if count > 0 else "✗"
            print(f"    {status} {img_type}: {count} images")

    # Verify matching counts within splits
    print("\nPair matching validation:")
    for split in splits:
        orig_count = structure[split]["original"]
        corr_count = structure[split]["corrupted"]
        if orig_count == corr_count and orig_count > 0:
            print(
                f"  ✓ {split}: Original and corrupted counts match ({orig_count} pairs)"
            )
        else:
            print(
                f"  ✗ {split}: Mismatch - {orig_count} original vs {corr_count} corrupted images"
            )

    return structure


def verify_image_pairs(dataset_dir="dataset"):
    """Verify that original and corrupted images are properly paired"""
    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        original_dir = os.path.join(dataset_dir, split, "original")
        corrupted_dir = os.path.join(dataset_dir, split, "corrupted")

        if not os.path.isdir(original_dir) or not os.path.isdir(corrupted_dir):
            results[split] = {"status": "missing_directories"}
            continue

        original_files = set(os.listdir(original_dir))
        corrupted_files = set(os.listdir(corrupted_dir))

        # Find missing pairs
        missing_in_corrupted = original_files - corrupted_files
        missing_in_original = corrupted_files - original_files
        common_files = original_files.intersection(corrupted_files)

        results[split] = {
            "total_original": len(original_files),
            "total_corrupted": len(corrupted_files),
            "matching_pairs": len(common_files),
            "missing_in_corrupted": list(missing_in_corrupted)[
                :5
            ],  # Show first 5 examples
            "missing_in_original": list(missing_in_original)[:5],
            "status": (
                "complete"
                if not missing_in_corrupted and not missing_in_original
                else "incomplete"
            ),
        }

    # Print summary
    print("Image Pair Validation:")
    print("=" * 50)

    for split in splits:
        info = results[split]
        if info.get("status") == "missing_directories":
            print(f"\n{split}: ✗ Missing directories")
            continue

        print(f"\n{split}:")
        print(f"  Total original images: {info['total_original']}")
        print(f"  Total corrupted images: {info['total_corrupted']}")
        print(f"  Matching pairs: {info['matching_pairs']}")

        if info["status"] == "complete":
            print(f"  ✓ All images have matching pairs")
        else:
            if info["missing_in_corrupted"]:
                print(
                    f"  ✗ {len(info['missing_in_corrupted'])} original images missing corrupted versions"
                )
                print(f"    Examples: {', '.join(info['missing_in_corrupted'])}")

            if info["missing_in_original"]:
                print(
                    f"  ✗ {len(info['missing_in_original'])} corrupted images missing original versions"
                )
                print(f"    Examples: {', '.join(info['missing_in_original'])}")

    return results


def inspect_random_pairs(dataset_dir="dataset", num_samples=5, split="train"):
    """Display random pairs of original and corrupted images for visual inspection"""
    original_dir = os.path.join(dataset_dir, split, "original")
    corrupted_dir = os.path.join(dataset_dir, split, "corrupted")
    info_dir = os.path.join(dataset_dir, split, "corrupted_info")

    if not os.path.isdir(original_dir) or not os.path.isdir(corrupted_dir):
        print(f"Dataset directories for {split} split not found.")
        return False

    original_files = os.listdir(original_dir)
    corrupted_files = os.listdir(corrupted_dir)

    common_files = list(set(original_files).intersection(set(corrupted_files)))

    if not common_files:
        print(f"No matching image pairs found in {split} split.")
        return False

    selected_files = random.sample(common_files, min(num_samples, len(common_files)))

    fig, axes = plt.subplots(
        len(selected_files), 2, figsize=(12, 4 * len(selected_files))
    )

    for i, img_file in enumerate(selected_files):
        # Load images
        orig_img = Image.open(os.path.join(original_dir, img_file))
        corr_img = Image.open(os.path.join(corrupted_dir, img_file))

        # Get corruption info if available
        corruption_info = "No info available"
        base_name, _ = os.path.splitext(img_file)
        info_path = os.path.join(info_dir, f"{base_name}.txt")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                corruption_info = f.read().strip()

        # Display images
        if len(selected_files) == 1:
            axes[0].imshow(orig_img)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(corr_img)
            axes[1].set_title("Corrupted")
            axes[1].axis("off")
        else:
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(corr_img)
            axes[i, 1].set_title("Corrupted")
            axes[i, 1].axis("off")

        # Add corruption info as text
        if len(selected_files) == 1:
            plt.figtext(
                0.5,
                0.01,
                corruption_info,
                ha="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
            )
        else:
            plt.figtext(
                0.5,
                0.99 - (i * 1 / len(selected_files)),
                f"Image {i+1}: {corruption_info}",
                ha="center",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
            )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.3)
    plt.show()

    print(f"Displayed {len(selected_files)} random image pairs from {split} split")
    return True


def validate_corruption_info(dataset_dir="dataset"):
    """Verify that corruption info files exist for all corrupted images"""
    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        corrupted_dir = os.path.join(dataset_dir, split, "corrupted")
        info_dir = os.path.join(dataset_dir, split, "corrupted_info")

        if not os.path.isdir(corrupted_dir) or not os.path.isdir(info_dir):
            results[split] = {"status": "missing_directories"}
            continue

        corrupted_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(corrupted_dir)
            if os.path.isfile(os.path.join(corrupted_dir, f))
        ]
        info_files = [
            os.path.splitext(f)[0]
            for f in os.listdir(info_dir)
            if os.path.isfile(os.path.join(info_dir, f))
        ]

        # Find missing info files
        missing_info = set(corrupted_files) - set(info_files)

        results[split] = {
            "total_corrupted": len(corrupted_files),
            "total_info": len(info_files),
            "missing_info": list(missing_info)[:5],  # Show first 5 examples
            "status": "complete" if not missing_info else "incomplete",
        }

    # Print summary
    print("Corruption Info Validation:")
    print("=" * 50)

    for split in splits:
        info = results[split]
        if info.get("status") == "missing_directories":
            print(f"\n{split}: ✗ Missing directories")
            continue

        print(f"\n{split}:")
        print(f"  Total corrupted images: {info['total_corrupted']}")
        print(f"  Total info files: {info['total_info']}")

        if info["status"] == "complete":
            print(f"  ✓ All corrupted images have corresponding info files")
        else:
            print(
                f"  ✗ {len(info['missing_info'])} corrupted images missing info files"
            )
            if info["missing_info"]:
                print(f"    Examples: {', '.join(info['missing_info'])}")

    return results




# Function ko call karein
base_dir = r"C:\Users\Debabrata Paul\Desktop\clear_vision\Data"
# display_image_comparison(base_dir)


# Run validation
# dataset_info = validate_dataset_structure(base_dir)


# Run pair verification
# pair_info = verify_image_pairs(base_dir)


# Inspect random pairs
# inspect_random_pairs(base_dir, num_samples=3)


# Validate corruption info
# info_validation = validate_corruption_info(base_dir)
