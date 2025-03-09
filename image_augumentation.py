import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import random
from tqdm import tqdm

def apply_augmentations(image):
    """
    Apply a series of augmentations using PIL & OpenCV.

    :param image: PIL Image object.
    :return: Augmented PIL Image.
    """

    # Convert PIL Image to NumPy array
    img_array = np.array(image)

    # Random Horizontal Flip
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random Vertical Flip
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Random Rotation
    angle = random.randint(-40, 40)
    image = image.rotate(angle)

    # Random Brightness Adjustment
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random Contrast Adjustment
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random Gaussian Noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 15, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array + noise, 0, 255)
        image = Image.fromarray(img_array)

    return image


def augment_dataset(input_dir, output_dir, num_augmented=5):
    """
    Apply augmentations to all images in the dataset recursively.

    :param input_dir: Root folder containing the original images.
    :param output_dir: Folder where augmented images will be stored.
    :param num_augmented: Number of augmented copies per image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_image_paths = []
    
    print(f"Scanning directory: {input_dir}")

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        save_folder = os.path.join(output_dir, relative_path)
        os.makedirs(save_folder, exist_ok=True)

        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                all_image_paths.append(image_path)

                try:
                    # Open image
                    image = Image.open(image_path).convert("RGB")

                    for i in range(num_augmented):
                        augmented_image = apply_augmentations(image)
                        output_path = os.path.join(save_folder, f"{os.path.splitext(file)[0]}_aug_{i}.jpg")
                        augmented_image.save(output_path)

                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    print(f" Augmentation complete. Total images processed (original + augmented): {len(all_image_paths)}")


if __name__ == "__main__":
    input_dir = "/home/xwang1/image_dataset/google-landmark/test"  # Change this to your dataset path
    output_dir = "/home/xwang1/image_dataset/google-landmark/test_augmented"  # Change this to save augmented images
    num_augmented = 5  # Number of augmented versions per image

    print("Starting image augmentation...")
    augment_dataset(input_dir, output_dir, num_augmented)
    print(" Image augmentation completed!")
