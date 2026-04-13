import os
import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

DATASET_DIR = r"..\data\EWS-Dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

RANDOM_SEED = 42
USE_ALL_TRAIN_IMAGES = True
MAX_IMAGES_FOR_TRAIN = 20

RUN_VALIDATION = False
RUN_TEST = True

OUTPUT_DIR = "svm_final_results"
GOOD_CASES_DIR = os.path.join(OUTPUT_DIR, "good_cases")
BAD_CASES_DIR = os.path.join(OUTPUT_DIR, "bad_cases")
TOP_K = 3
BOTTOM_K = 3


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_raw_high_fidelity_features(image_rgb):
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
        lab[:, :, 1], lab[:, :, 1] * suppression_mask,
        exg, exg * suppression_mask,
        neighbor_support,
        hsv[:, :, 1],
        cv2.erode(exg.astype(np.float32), np.ones((3, 3))),
        cv2.GaussianBlur(exg.astype(np.float32), (3, 3), 0)
    ]
    return np.stack(feature_stack, axis=-1).astype(np.float32)


def postprocess_mask(pred_mask):
    mask = pred_mask.astype(np.uint8).copy()

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labs, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= 8:
            mask[labs == i] = 0

    inv = 1 - mask
    num_inv, labs_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, num_inv):
        x, y, w, h, area = stats_inv[i]
        border = (x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0])
        if (not border) and area <= 100:
            mask[labs_inv == i] = 1

    return mask


def load_rgb_and_mask(img_p, msk_p):
    img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
    msk = (cv2.imread(msk_p, 0) >= 128).astype(np.uint8)
    return img, msk


def get_image_mask_pairs(split_dir):
    if not os.path.exists(split_dir):
        return []
    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".png") and "_mask" not in f])
    pairs = []
    for f in files:
        img_p = os.path.join(split_dir, f)
        msk_p = img_p.replace(".png", "_mask.png")
        if os.path.exists(msk_p):
            pairs.append((img_p, msk_p, f))
    return pairs


def save_result_figure(image_np, gt_np, pred_np, save_path, title_suffix):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap="gray", vmin=0, vmax=1)
    plt.title(title_suffix)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close()


def evaluate_on_split(model, split_name, output_dir):
    split_dir = os.path.join(DATASET_DIR, split_name)
    pairs = get_image_mask_pairs(split_dir)
    print(f"\nEvaluating on {split_name} ({len(pairs)} images)...")

    all_results = []
    total_test_time = 0.0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(GOOD_CASES_DIR, exist_ok=True)
    os.makedirs(BAD_CASES_DIR, exist_ok=True)

    for idx, (img_p, msk_p, fname) in enumerate(pairs, 1):
        img, gt = load_rgb_and_mask(img_p, msk_p)
        feats = extract_raw_high_fidelity_features(img)
        h, w, c = feats.shape

        start_t = time.perf_counter()
        raw_pred = model.predict(feats.reshape(-1, c)).reshape(h, w).astype(np.uint8)
        pred = postprocess_mask(raw_pred)
        infer_time = time.perf_counter() - start_t
        total_test_time += infer_time

        y_true = gt.flatten()
        y_pred = pred.flatten()

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        iou = jaccard_score(y_true, y_pred, zero_division=0)

        all_results.append({
            "name": fname,
            "image": img,
            "gt": gt,
            "pred": pred,
            "precision": p,
            "recall": r,
            "f1": f1,
            "iou": iou,
            "time": infer_time,
        })

        save_path = os.path.join(output_dir, fname.replace(".png", "_result.png"))
        save_result_figure(
            img,
            gt,
            pred,
            save_path,
            title_suffix=f"SVM Prediction\nP:{p:.4f} R:{r:.4f}\nF1:{f1:.4f} IoU:{iou:.4f}"
        )

        print(
            f"[{idx:02d}/{len(pairs)}] {fname} | "
            f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | "
            f"Time: {infer_time:.4f}s"
        )

    mean_precision = float(np.mean([x["precision"] for x in all_results]))
    mean_recall = float(np.mean([x["recall"] for x in all_results]))
    mean_f1 = float(np.mean([x["f1"] for x in all_results]))
    mean_iou = float(np.mean([x["iou"] for x in all_results]))
    avg_time = total_test_time / max(len(all_results), 1)

    print(f"\n{split_name.upper()} Summary")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall:    {mean_recall:.4f}")
    print(f"Mean F1-score:  {mean_f1:.4f}")
    print(f"Mean IoU:       {mean_iou:.4f}")
    print(f"Total test time: {total_test_time:.4f}s")
    print(f"Avg/image time:  {avg_time:.4f}s")

    sorted_results = sorted(all_results, key=lambda x: x["iou"], reverse=True)
    good_cases = sorted_results[:TOP_K]
    bad_cases = sorted_results[-BOTTOM_K:]

    for item in good_cases:
        save_path = os.path.join(GOOD_CASES_DIR, item["name"].replace(".png", "_good.png"))
        save_result_figure(
            item["image"],
            item["gt"],
            item["pred"],
            save_path,
            title_suffix=f"Good Candidate\nP:{item['precision']:.4f} R:{item['recall']:.4f}\nF1:{item['f1']:.4f} IoU:{item['iou']:.4f}"
        )

    for item in bad_cases:
        save_path = os.path.join(BAD_CASES_DIR, item["name"].replace(".png", "_bad.png"))
        save_result_figure(
            item["image"],
            item["gt"],
            item["pred"],
            save_path,
            title_suffix=f"Bad Candidate\nP:{item['precision']:.4f} R:{item['recall']:.4f}\nF1:{item['f1']:.4f} IoU:{item['iou']:.4f}"
        )

    csv_path = os.path.join(output_dir, f"{split_name}_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "precision", "recall", "f1", "iou", "inference_time_s"])
        for item in all_results:
            writer.writerow([
                item["name"],
                f"{item['precision']:.6f}",
                f"{item['recall']:.6f}",
                f"{item['f1']:.6f}",
                f"{item['iou']:.6f}",
                f"{item['time']:.6f}",
            ])

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SVM Final Evaluation Summary\n")
        f.write("============================\n")
        f.write(f"Mean Precision: {mean_precision:.6f}\n")
        f.write(f"Mean Recall:    {mean_recall:.6f}\n")
        f.write(f"Mean F1-score:  {mean_f1:.6f}\n")
        f.write(f"Mean IoU:       {mean_iou:.6f}\n")
        f.write(f"Total test time: {total_test_time:.6f}s\n")
        f.write(f"Avg/image time:  {avg_time:.6f}s\n\n")
        f.write("Good case candidates:\n")
        for item in good_cases:
            f.write(f"  - {item['name']}\n")
        f.write("Bad case candidates:\n")
        for item in bad_cases:
            f.write(f"  - {item['name']}\n")

    print(f"\nResults saved to: {output_dir}")
    print(f"Good cases folder: {GOOD_CASES_DIR}")
    print(f"Bad cases folder:  {BAD_CASES_DIR}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dir = os.path.join(DATASET_DIR, TRAIN_SPLIT)
    val_dir = os.path.join(DATASET_DIR, VAL_SPLIT)
    test_dir = os.path.join(DATASET_DIR, TEST_SPLIT)

    all_train_pairs = get_image_mask_pairs(train_dir)
    pairs = all_train_pairs if USE_ALL_TRAIN_IMAGES else all_train_pairs[:MAX_IMAGES_FOR_TRAIN]

    X_list, y_list = [], []
    print("Building training set...")
    print(f"Using {len(pairs)} training images.")

    t_train_start = time.perf_counter()

    for img_p, msk_p, fname in pairs:
        img, msk = load_rgb_and_mask(img_p, msk_p)
        feats = extract_raw_high_fidelity_features(img)
        X = feats.reshape(-1, feats.shape[-1])
        y = msk.flatten()

        X_list.append(X)
        y_list.append(y)

    print("Training SVM Classifier...")
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.2,
            class_weight="balanced",
            max_iter=30000,
            random_state=RANDOM_SEED,
            dual=False
        )
    )
    model.fit(np.vstack(X_list), np.concatenate(y_list))
    training_time = time.perf_counter() - t_train_start

    print("\nTraining finished.")
    print(f"Total training time: {training_time:.2f} sec")

    if RUN_VALIDATION:
        print("\nRunning validation evaluation...")
        evaluate_on_split(model, VAL_SPLIT, OUTPUT_DIR)

    if RUN_TEST:
        print("\nLoading model for final test evaluation...")
        evaluate_on_split(model, TEST_SPLIT, OUTPUT_DIR)


if __name__ == "__main__":
    main()