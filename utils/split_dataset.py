import os
import random
import shutil


def split_dataset(base_dir="dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split the dataset into train, validation, and test sets"""
    # Get list of valid image files (original + corrupted must exist)
    original_dir = os.path.join(base_dir, "original")
    corrupted_dir = os.path.join(base_dir, "corrupted")

    image_files = [
        f
        for f in os.listdir(original_dir)
        if os.path.exists(os.path.join(corrupted_dir, f))
    ]

    if not image_files:
        print("No images found to split")
        return False

    # Shuffle the list
    random.shuffle(image_files)

    # Calculate split indices
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    # Split the list
    train_files = image_files[:train_count]
    val_files = image_files[train_count : train_count + val_count]
    test_files = image_files[train_count + val_count :]

    print(
        f"Splitting dataset: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test"
    )

    # Copy files to respective directories
    for file_list, split_name in [
        (train_files, "train"),
        (val_files, "val"),
        (test_files, "test"),
    ]:
        for img_file in file_list:
            # Copy original image
            src_path = os.path.join(base_dir, "original", img_file)
            dst_path = os.path.join(base_dir, split_name, "original", img_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

            # Copy corrupted image
            src_path = os.path.join(base_dir, "corrupted", img_file)
            dst_path = os.path.join(base_dir, split_name, "corrupted", img_file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

            # Copy corruption info if available
            base_name, _ = os.path.splitext(img_file)
            info_path = os.path.join(base_dir, "corrupted_info", f"{base_name}.txt")
            if os.path.exists(info_path):
                dst_info_dir = os.path.join(base_dir, split_name, "corrupted_info")
                os.makedirs(dst_info_dir, exist_ok=True)
                shutil.copy2(info_path, os.path.join(dst_info_dir, f"{base_name}.txt"))

    return True
