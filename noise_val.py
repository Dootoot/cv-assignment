import os
import cv2
import shutil
import random
import numpy as np

INPUT_VAL_DIR = "data/EWS-Dataset/validation"
OUTPUT_VAL_NOISE_DIR = "data/EWS-Dataset/noise_val"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def full_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def recreate_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def get_png_files(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return mask


def collect_pairs(folder):
    pairs = []

    for file_name in get_png_files(folder):
        stem, ext = os.path.splitext(file_name)

        if stem.endswith("_mask"):
            continue

        mask_name = f"{stem}_mask{ext}"
        image_path = os.path.join(folder, file_name)
        mask_path = os.path.join(folder, mask_name)

        if os.path.exists(mask_path):
            pairs.append((file_name, image_path, mask_path))
        else:
            print(f"Skipping {file_name}: mask not found.")

    return pairs


def add_gaussian_noise(image, mean=0, std=12):
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def change_exposure(image, alpha=1.25, beta=10):
    # alpha controls contrast/exposure, beta controls brightness
    adjusted = image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def create_val_noise_exposure(input_val_dir, output_dir):
    input_val_dir = full_path(input_val_dir)
    output_dir = full_path(output_dir)

    recreate_folder(output_dir)

    pairs = collect_pairs(input_val_dir)
    if not pairs:
        raise ValueError("No valid validation image-mask pairs found.")

    total_saved = 0

    for image_name, image_path, mask_path in pairs:
        image = read_image(image_path)
        mask = read_mask(mask_path)

        base_name, ext = os.path.splitext(image_name)

        versions = {
            "noise": add_gaussian_noise(image, std=12),
            "bright": change_exposure(image, alpha=1.25, beta=15),
            "dark": change_exposure(image, alpha=0.75, beta=-10),
            "noise_bright": change_exposure(add_gaussian_noise(image, std=10), alpha=1.2, beta=10),
        }

        for suffix, aug_image in versions.items():
            out_image_name = f"{base_name}_{suffix}{ext}"
            out_mask_name = f"{base_name}_{suffix}_mask{ext}"

            cv2.imwrite(os.path.join(output_dir, out_image_name), aug_image)
            cv2.imwrite(os.path.join(output_dir, out_mask_name), mask)

            total_saved += 1

    print("=" * 60)
    print("Validation noise/exposure set created.")
    print(f"Input folder  : {input_val_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Original pairs: {len(pairs)}")
    print(f"Saved pairs   : {total_saved}")
    print(f"PNG files     : {total_saved * 2}")
    print("=" * 60)


if __name__ == "__main__":
    create_val_noise_exposure(INPUT_VAL_DIR, OUTPUT_VAL_NOISE_DIR)
