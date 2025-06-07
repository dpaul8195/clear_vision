# Step 3: Process images - create corrupted versions with multiple corruptions
import os
from utils.corruption import *
from tqdm import tqdm
from PIL import Image
import random

def process_images(input_dir, output_dir="dataset"):
    """Process images from input directory, create corrupted versions with multiple corruptions"""
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            ):  # Added .webp
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return False

    print(f"Found {len(image_files)} images to process")

    # Define corruption types
    corruption_types = [
        ("gaussian_noise", add_gaussian_noise),
        ("salt_pepper", add_salt_pepper_noise),
        ("speckle_noise", add_speckle_noise),
        ("mild_blur", add_mild_blur),
        ("motion_blur", add_motion_blur),
        ("jpeg_compression", add_jpeg_compression),
        ("low_brightness", adjust_brightness),
        ("low_contrast", adjust_contrast),
        ("occlusion", add_occlusion),
        ("compression_blocks", add_compression_blocks),
    ]

    # Process each image
    count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Generate output filename
            img_filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(img_filename)

            # Open the image
            img = Image.open(img_path).convert("RGB")

            # Save original image with jpg extension for consistency
            original_path = os.path.join(output_dir, "original", f"{base_name}.jpg")
            img.save(original_path)

            # Choose number of corruptions to apply (3-4)
            num_corruptions = random.randint(3, 4)

            # Select random corruption types without replacement
            selected_corruptions = random.sample(corruption_types, num_corruptions)

            # Start with the original image
            corrupted_img = img.copy()
            corruption_info_list = []

            # Apply each selected corruption sequentially
            for corruption_name, corruption_func in selected_corruptions:
                if corruption_name == "gaussian_noise":
                    severity = random.uniform(0.02, 0.08)  # Reduced severity
                    corrupted_img = corruption_func(corrupted_img, severity)
                    corruption_info_list.append(
                        f"Gaussian noise (severity={severity:.2f})"
                    )

                elif corruption_name == "salt_pepper":
                    severity = random.uniform(0.003, 0.015)  # Reduced severity
                    corrupted_img = corruption_func(corrupted_img, severity)
                    corruption_info_list.append(
                        f"Salt & pepper noise (severity={severity:.2f})"
                    )

                elif corruption_name == "speckle_noise":
                    severity = random.uniform(0.02, 0.08)  # Reduced severity
                    corrupted_img = corruption_func(corrupted_img, severity)
                    corruption_info_list.append(
                        f"Speckle noise (severity={severity:.2f})"
                    )

                elif corruption_name == "mild_blur":
                    # Increased blur range slightly as requested
                    radius = random.uniform(0.8, 1.8)
                    corrupted_img = corruption_func(corrupted_img, radius)
                    corruption_info_list.append(
                        f"Mild Gaussian blur (radius={radius:.2f})"
                    )

                elif corruption_name == "motion_blur":
                    kernel_size = random.choice(
                        [3, 5, 7]
                    )  # Added 7 for slightly more blur
                    corrupted_img = corruption_func(corrupted_img, kernel_size)
                    corruption_info_list.append(
                        f"Motion blur (kernel_size={kernel_size})"
                    )

                elif corruption_name == "jpeg_compression":
                    quality = random.randint(50, 85)
                    corrupted_img = corruption_func(corrupted_img, quality)
                    corruption_info_list.append(f"JPEG compression (quality={quality})")

                elif corruption_name == "low_brightness":
                    factor = random.uniform(0.7, 0.9)
                    corrupted_img = corruption_func(corrupted_img, factor)
                    corruption_info_list.append(f"Low brightness (factor={factor:.2f})")

                elif corruption_name == "low_contrast":
                    factor = random.uniform(0.7, 0.9)
                    corrupted_img = corruption_func(corrupted_img, factor)
                    corruption_info_list.append(f"Low contrast (factor={factor:.2f})")

                elif corruption_name == "occlusion":
                    # Reduced occlusion size parameters
                    occlusion_type = random.choice(["random", "horizontal", "vertical"])
                    corrupted_img = corruption_func(corrupted_img, occlusion_type)
                    corruption_info_list.append(f"Occlusion (type={occlusion_type})")

                elif corruption_name == "compression_blocks":
                    block_size = random.choice([8, 16])
                    severity = random.uniform(0.15, 0.35)  # Reduced severity
                    corrupted_img = corruption_func(corrupted_img, block_size, severity)
                    corruption_info_list.append(
                        f"Compression blocks (size={block_size}, severity={severity:.2f})"
                    )

            # Save corrupted image with jpg extension for consistency
            corrupted_path = os.path.join(output_dir, "corrupted", f"{base_name}.jpg")
            corrupted_img.save(corrupted_path)

            # Save corruption info to a text file
            info_dir = os.path.join(output_dir, "corrupted_info")
            os.makedirs(info_dir, exist_ok=True)
            with open(os.path.join(info_dir, f"{base_name}.txt"), "w") as f:
                f.write(
                    "Multiple corruptions applied:\n" + "\n".join(corruption_info_list)
                )

        except Exception as e:
            print(f"Error processing: {e}")
            count = count + 1  
            os.remove(img_path)  # Optionally delete bad original image
            continue


    print(f"Processed and saved {len(image_files) - count} images")
    return True
