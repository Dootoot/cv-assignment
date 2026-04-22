import os
import cv2
import shutil
import random
import numpy as np
import albumentations as A

# ----------------------------
# settings
# ----------------------------
INPUT_TRAIN_DIR = "data/EWS-Dataset/train"
OUTPUT_AUG_DIR = "data/EWS-Dataset/aug_train"
NUM_AUG_PER_IMAGE = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------
# augmentation
# ----------------------------
def build_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.05,
            contrast_limit=0.05,
            p=0.4
        ),
        A.OneOf([
            A.RandomRain(
                slant_range=(-5, 5),
                drop_length=8,
                drop_width=1,
                blur_value=3,
                brightness_coefficient=0.95,
                p=1.0
            ),
            A.RandomFog(
                fog_coef_range=(0.05, 0.12),
                alpha_coef=0.05,
                p=1.0
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=60,
                p=1.0
            )
        ], p=0.2)
    ])


# ----------------------------
# helper functions
# ----------------------------
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
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return mask


def save_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)


def save_mask(path, mask):
    cv2.imwrite(path, mask)


# ----------------------------
# collect image-mask pairs
# input format:
# image.png
# image_mask.png
# ----------------------------
def collect_pairs(train_dir):
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train folder not found: {train_dir}")

    all_files = get_png_files(train_dir)
    if not all_files:
        raise ValueError(f"No PNG files found in: {train_dir}")

    pairs = []

    for file_name in all_files:
        stem, ext = os.path.splitext(file_name)

        # skip mask files
        if stem.endswith("_mask"):
            continue

        mask_name = f"{stem}_mask{ext}"
        image_path = os.path.join(train_dir, file_name)
        mask_path = os.path.join(train_dir, mask_name)

        if os.path.exists(mask_path):
            pairs.append((file_name, image_path, mask_path))
        else:
            print(f"Skipping {file_name}: matching mask not found.")

    return pairs


# ----------------------------
# main
# ----------------------------
def augment_training_set(input_train_dir, output_aug_dir, num_aug_per_image):
    input_train_dir = full_path(input_train_dir)
    output_aug_dir = full_path(output_aug_dir)

    recreate_folder(output_aug_dir)
    pairs = collect_pairs(input_train_dir)

    if not pairs:
        raise ValueError("No valid image-mask pairs found.")

    transform = build_transform()
    total_pairs_saved = 0

    for image_name, image_path, mask_path in pairs:
        image = read_image(image_path)
        mask = read_mask(mask_path)

        base_name, ext = os.path.splitext(image_name)

        # save original
        save_image(os.path.join(output_aug_dir, image_name), image)
        save_mask(os.path.join(output_aug_dir, f"{base_name}_mask{ext}"), mask)
        total_pairs_saved += 1

        # save augmented copies
        for i in range(1, num_aug_per_image + 1):
            result = transform(image=image, mask=mask)
            aug_image = result["image"]
            aug_mask = result["mask"]

            save_image(
                os.path.join(output_aug_dir, f"{base_name}_aug{i}{ext}"),
                aug_image
            )
            save_mask(
                os.path.join(output_aug_dir, f"{base_name}_aug{i}_mask{ext}"),
                aug_mask
            )
            total_pairs_saved += 1

    print("=" * 60)
    print("Augmentation finished.")
    print(f"Input train folder : {input_train_dir}")
    print(f"Output aug folder  : {output_aug_dir}")
    print(f"Original images    : {len(pairs)}")
    print(f"Aug per image      : {num_aug_per_image}")
    print(f"Total pairs saved  : {total_pairs_saved}")
    print(f"Actual PNG files   : {total_pairs_saved * 2}")
    print("=" * 60)


if __name__ == "__main__":
    augment_training_set(INPUT_TRAIN_DIR, OUTPUT_AUG_DIR, NUM_AUG_PER_IMAGE)