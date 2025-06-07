# Helper functions for various corruptions
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import cv2
import io

def add_gaussian_noise(image, severity=0.1):
    """Add Gaussian noise to image"""
    img_array = np.array(image) / 255.0
    noise = np.random.normal(0, severity, img_array.shape)
    noisy_img = img_array + noise
    noisy_img = np.clip(noisy_img, 0, 1) * 255
    return Image.fromarray(noisy_img.astype(np.uint8))


def add_salt_pepper_noise(image, severity=0.02):
    """Add salt and pepper noise to image"""
    img_array = np.array(image)
    s_vs_p = 0.5
    amount = severity

    # Salt (white) noise
    num_salt = np.ceil(amount * img_array.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 255

    # Pepper (black) noise
    num_pepper = np.ceil(amount * img_array.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 0

    return Image.fromarray(img_array)


def add_speckle_noise(image, severity=0.1):
    """Add speckle noise to image"""
    img_array = np.array(image) / 255.0
    noise = np.random.normal(0, severity, img_array.shape)
    noisy_img = img_array + img_array * noise
    noisy_img = np.clip(noisy_img, 0, 1) * 255
    return Image.fromarray(noisy_img.astype(np.uint8))


def add_mild_blur(image, radius=1.2):
    """Add mild Gaussian blur"""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def add_motion_blur(image, kernel_size=7):
    """Add motion blur using OpenCV"""
    img_array = np.array(image)

    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Apply motion blur
    blurred = cv2.filter2D(img_array, -1, kernel)

    return Image.fromarray(blurred)


def add_jpeg_compression(image, quality=50):
    """Add JPEG compression artifacts"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def adjust_brightness(image, factor=0.7):
    """Adjust image brightness"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def adjust_contrast(image, factor=0.7):
    """Adjust image contrast"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def add_occlusion(image, occlusion_type="random"):
    """Add occlusion to image with reduced parameters"""
    img_array = np.array(image)
    height, width, _ = img_array.shape

    if occlusion_type == "random":
        # Reduced rectangle occlusion size
        rect_width = random.randint(width // 10, width // 4)  # Smaller than before
        rect_height = random.randint(height // 10, height // 4)  # Smaller than before
        x = random.randint(0, width - rect_width)
        y = random.randint(0, height - rect_height)

        # Random color for occlusion (black, white, or random)
        color_choice = random.choice(["black", "white", "random"])
        if color_choice == "black":
            color = [0, 0, 0]
        elif color_choice == "white":
            color = [255, 255, 255]
        else:
            color = [
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ]

        img_array[y : y + rect_height, x : x + rect_width, :] = color

    elif occlusion_type == "horizontal":
        # Smaller horizontal stripe occlusion
        stripe_height = random.randint(height // 50, height // 20)  # Much smaller
        y = random.randint(0, height - stripe_height)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        img_array[y : y + stripe_height, :, :] = color

    elif occlusion_type == "vertical":
        # Smaller vertical stripe occlusion
        stripe_width = random.randint(width // 50, width // 20)  # Much smaller
        x = random.randint(0, width - stripe_width)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        img_array[:, x : x + stripe_width, :] = color

    return Image.fromarray(img_array)


def add_compression_blocks(image, block_size=8, severity=0.5):
    """Simulate compression blocks similar to DCT artifacts"""
    img_array = np.array(image)
    height, width, channels = img_array.shape

    # Process the image in blocks
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Get current block
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = img_array[y:y_end, x:x_end, :]

            # Average color in block
            avg_color = np.mean(block, axis=(0, 1))

            # Mix original block with average color based on severity
            mix = block * (1 - severity) + avg_color * severity
            img_array[y:y_end, x:x_end, :] = mix

    return Image.fromarray(img_array.astype(np.uint8))
