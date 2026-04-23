import os
import cv2
import shutil
import random
import numpy as np

INPUT_VAL_DIR = "data/EWS-Dataset/validation"
OUTPUT_VAL_DIR = "data/EWS-Dataset/val_robustness"

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


def change_exposure(image, alpha=1.25, beta=15):
    adjusted = image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def add_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_jpeg_compression(image, quality=35):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", image, encode_param)

    if not success:
        raise RuntimeError("JPEG compression failed.")

    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return compressed


def save_pair(output_dir, image_name, image, mask):
    base_name, ext = os.path.splitext(image_name)
    image_path = os.path.join(output_dir, image_name)
    mask_path = os.path.join(output_dir, f"{base_name}_mask{ext}")

    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)


def save_augmented_pair(output_dir, base_name, ext, suffix, image, mask):
    image_name = f"{base_name}_{suffix}{ext}"
    mask_name = f"{base_name}_{suffix}_mask{ext}"

    cv2.imwrite(os.path.join(output_dir, image_name), image)
    cv2.imwrite(os.path.join(output_dir, mask_name), mask)


def create_robustness_validation_set(input_val_dir, output_val_dir):
    input_val_dir = full_path(input_val_dir)
    output_val_dir = full_path(output_val_dir)

    recreate_folder(output_val_dir)

    pairs = collect_pairs(input_val_dir)
    if not pairs:
        raise ValueError("No valid validation image-mask pairs found.")

    total_pairs_saved = 0

    for image_name, image_path, mask_path in pairs:
        image = read_image(image_path)
        mask = read_mask(mask_path)

        base_name, ext = os.path.splitext(image_name)

        # save original image and mask
        save_pair(output_val_dir, image_name, image, mask)
        total_pairs_saved += 1

        # save four separate robustness versions
        augmented_versions = {
            "gaussian_noise": add_gaussian_noise(image, std=12),
            "exposure": change_exposure(image, alpha=1.25, beta=15),
            "blur": add_blur(image, kernel_size=5),
            "compression": add_jpeg_compression(image, quality=35),
        }

        for suffix, aug_image in augmented_versions.items():
            save_augmented_pair(
                output_val_dir,
                base_name,
                ext,
                suffix,
                aug_image,
                mask,
            )
            total_pairs_saved += 1

    print("=" * 60)
    print("Validation robustness set created.")
    print(f"Input validation folder : {input_val_dir}")
    print(f"Output folder           : {output_val_dir}")
    print(f"Original pairs          : {len(pairs)}")
    print(f"Total pairs saved       : {total_pairs_saved}")
    print(f"Actual PNG files        : {total_pairs_saved * 2}")
    print("=" * 60)


if __name__ == "__main__":
    create_robustness_validation_set(INPUT_VAL_DIR, OUTPUT_VAL_DIR)
