import os
from utils.generate_corrupted_dataset import process_images
from utils.split_dataset import split_dataset
from visuals.visualize_samples import display_samples
from utils.image_scraper import image_scraper

def create_dataset_directories(base_dir="dataset"):
    """Create necessary directories for the dataset"""
    directories = [
        f"{base_dir}/original",
        f"{base_dir}/corrupted",
        f"{base_dir}/train/original",
        f"{base_dir}/train/corrupted",
        f"{base_dir}/val/original",
        f"{base_dir}/val/corrupted",
        f"{base_dir}/test/original",
        f"{base_dir}/test/corrupted",
        f"{base_dir}/corrupted_info",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"Created dataset directories in {base_dir}")
    return base_dir


def create_image_restoration_dataset(base_dir="dataset"):
    """Main function to create the image restoration dataset"""
    dataset_dir = create_dataset_directories(base_dir)

    # images_dir = image_scraper(f"{base_dir}/original")
    
    # But Here we do CelebA-HQ dataset for model training

    images_dir = r"C:\Users\Debabrata Paul\Desktop\clear_vision\CelebA-HQ\celeba_hq_256"

    if not images_dir:
        print("No image directory found.")
        return

    success = process_images(images_dir, dataset_dir)
    if not success:
        print("Failed to process images")
        return

    success = split_dataset(dataset_dir)
    if not success:
        print("Failed to split dataset")
        return

    try:
        display_samples(dataset_dir)
    except Exception as e:
        print(f"Could not display samples: {e}")

    print("Dataset creation completed successfully!")
    return dataset_dir
