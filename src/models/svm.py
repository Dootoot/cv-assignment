import cv2
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .helpers import predict_from_trained_model, save_model, load_model


RANDOM_SEED = 42
USE_ALL_TRAIN_IMAGES = True
MAX_IMAGES_FOR_TRAIN = 20


def load_trained_svm(model_dir: str):
    return load_model(model_dir, "svm_model.joblib")


def generate_trained_svm(model_dir: str, image_folder_path: str):
    if image_folder_path is None:
        raise RuntimeError("Invalid image folder path")

    images_dir = Path(image_folder_path)

    all_pairs = get_image_mask_pairs(images_dir)
    if len(all_pairs) == 0:
        raise RuntimeError(f"No image/mask pairs found in {image_folder_path}")

    pairs = all_pairs if USE_ALL_TRAIN_IMAGES else all_pairs[:MAX_IMAGES_FOR_TRAIN]

    X_list = []
    y_list = []

    for img_p, msk_p, _ in pairs:
        image_rgb, mask = load_rgb_and_mask(str(img_p), str(msk_p))
        features = extract_raw_high_fidelity_features(image_rgb)

        X = features.reshape(-1, features.shape[-1])
        y = mask.flatten()

        X_list.append(X)
        y_list.append(y)

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    svm_model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.2,
            class_weight="balanced",
            max_iter=30000,
            random_state=RANDOM_SEED,
            dual=False
        )
    )
    svm_model.fit(X_train, y_train)

    save_model(svm_model, model_dir, "svm_model.joblib")
    return svm_model


def predict_from_trained_svm(output_dir: str, svm_model, testing_folder: str):
    return predict_from_trained_model(
        output_dir,
        testing_folder,
        svm_model,
        single_image_predict_from_trained_svm
    )


def single_image_predict_from_trained_svm(svm_model, image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    features = extract_raw_high_fidelity_features(image_rgb)

    h, w, c = features.shape
    raw_pred = svm_model.predict(features.reshape(-1, c)).reshape(h, w).astype(np.uint8)
    refined_pred = postprocess_mask(raw_pred)

    output_mask = (refined_pred * 255).astype(np.uint8)
    return output_mask


def get_image_mask_pairs(split_dir: Path):
    if not split_dir.exists():
        return []

    files = sorted([f for f in split_dir.glob("*.png") if not f.name.endswith("_mask.png")])
    pairs = []

    for img_p in files:
        mask_p = img_p.with_name(img_p.stem + "_mask.png")
        if mask_p.exists():
            pairs.append((img_p, mask_p, img_p.name))

    return pairs


def load_rgb_and_mask(img_path: str, mask_path: str):
    image_bgr = cv2.imread(img_path)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None or mask_gray is None:
        raise RuntimeError(f"Failed to read image or mask: {img_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = (mask_gray > 0).astype(np.uint8)
    return image_rgb, mask


def extract_raw_high_fidelity_features(image_rgb: np.ndarray):
    img_f = image_rgb.astype(np.float32) / 255.0
    r, g, b = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    exg = 2 * g - r - b

    exg_norm = np.clip((exg + 1) / 2 * 255, 0, 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(
        exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    dynamic_gate = (otsu_thresh / 255.0 * 2 - 1) * 1.05
    spectral_mask = (exg > dynamic_gate).astype(np.float32)

    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    neighbor_support = cv2.filter2D(spectral_mask, -1, kernel)

    suppression_mask = ((spectral_mask > 0) & (neighbor_support >= 2)).astype(np.float32)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0

    feature_stack = [
        r, g, b,
        lab[:, :, 1],
        lab[:, :, 1] * suppression_mask,
        exg,
        exg * suppression_mask,
        neighbor_support,
        hsv[:, :, 1],
        cv2.erode(exg.astype(np.float32), np.ones((3, 3))),
        cv2.GaussianBlur(exg.astype(np.float32), (3, 3), 0)
    ]

    return np.stack(feature_stack, axis=-1).astype(np.float32)


def postprocess_mask(pred_mask: np.ndarray):
    mask = pred_mask.astype(np.uint8).copy()

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= 8:
            mask[labels == i] = 0

    inv = 1 - mask
    num_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num_inv):
        x, y, w, h, area = stats_inv[i]
        touches_border = (x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0])
        if (not touches_border) and area <= 100:
            mask[labels_inv == i] = 1


