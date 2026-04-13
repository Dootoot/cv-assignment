import os
import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import segmentation_models_pytorch as smp

# ==========================================
# Config
# ==========================================
DATASET_DIR = r"C:\Users\Administrator\Desktop\project\data\EWS-Dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

IMG_SIZE = 512
BATCH_SIZE = 12
NUM_EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
THRESHOLD = 0.6

BEST_MODEL_PATH = "best_fpn_ews.pth"
RESULTS_DIR = "fpn_final_results"
GOOD_CASES_DIR = os.path.join(RESULTS_DIR, "good_cases")
BAD_CASES_DIR = os.path.join(RESULTS_DIR, "bad_cases")
TOP_K = 3
BOTTOM_K = 3

# ==========================================
# Dataset
# ==========================================
class EWSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(".png") and "_mask" not in f]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = img_path.replace(".png", "_mask.png")

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        # 官方 mask: 白底黑叶 -> wheat = 1, background = 0
        mask = (mask < 128).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            img_name,
        )

# ==========================================
# Transforms
# ==========================================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

eval_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

# ==========================================
# Metrics
# ==========================================
def compute_binary_metrics(pred_np, mask_np):
    pred_np = pred_np.astype(np.uint8).reshape(-1)
    mask_np = mask_np.astype(np.uint8).reshape(-1)

    tp = np.sum((pred_np == 1) & (mask_np == 1))
    fp = np.sum((pred_np == 1) & (mask_np == 0))
    fn = np.sum((pred_np == 0) & (mask_np == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return precision, recall, f1, iou

def compute_batch_metrics_from_logits(logits, masks, threshold=0.55):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_np = preds.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    masks_np = masks.detach().cpu().numpy().astype(np.uint8).reshape(-1)

    tp = np.sum((preds_np == 1) & (masks_np == 1))
    fp = np.sum((preds_np == 1) & (masks_np == 0))
    fn = np.sum((preds_np == 0) & (masks_np == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return precision, recall, f1, iou

# ==========================================
# Train / Validation
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_validation(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    precisions, recalls, f1s, ious = [], [], [], []

    for images, masks, _ in tqdm(loader, desc="Validation", leave=False):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        p, r, f1, iou = compute_batch_metrics_from_logits(logits, masks, threshold=THRESHOLD)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ious.append(iou)

    return {
        "loss": total_loss / len(loader.dataset),
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "iou": float(np.mean(ious)),
    }

# ==========================================
# Save result figure
# ==========================================
def save_result_figure(image_np, gt_np, pred_np, save_path, title_suffix):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(1 - gt_np, cmap="gray", vmin=0, vmax=1)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(1 - pred_np, cmap="gray", vmin=0, vmax=1)
    plt.title(title_suffix)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close()

# ==========================================
# Final test evaluation
# ==========================================
@torch.no_grad()
def evaluate_on_test(model, dataset, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(GOOD_CASES_DIR, exist_ok=True)
    os.makedirs(BAD_CASES_DIR, exist_ok=True)

    all_results = []
    total_test_time = 0.0

    print(f"\nEvaluating on test ({len(dataset)} images)...")

    for idx in range(len(dataset)):
        image, mask, img_name = dataset[idx]

        x = image.unsqueeze(0).to(DEVICE)

        start_t = time.perf_counter()
        logits = model(x)
        infer_time = time.perf_counter() - start_t
        total_test_time += infer_time

        pred = (torch.sigmoid(logits) > THRESHOLD).float().cpu().numpy()[0, 0]

        image_np = image.numpy().transpose(1, 2, 0)
        gt_np = mask.numpy()[0]
        pred_np = pred.astype(np.uint8)

        p, r, f1, iou = compute_binary_metrics(pred_np, gt_np)

        all_results.append({
            "name": img_name,
            "image": image_np,
            "gt": gt_np,
            "pred": pred_np,
            "precision": p,
            "recall": r,
            "f1": f1,
            "iou": iou,
            "time": infer_time,
        })

        save_path = os.path.join(output_dir, img_name.replace(".png", "_result.png"))
        save_result_figure(
            image_np,
            gt_np,
            pred_np,
            save_path,
            title_suffix=f"FPN Prediction\nP:{p:.4f} R:{r:.4f}\nF1:{f1:.4f} IoU:{iou:.4f}"
        )

        print(
            f"[{idx+1:02d}/{len(dataset)}] {img_name} | "
            f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | "
            f"Time: {infer_time:.4f}s"
        )

    mean_precision = float(np.mean([x["precision"] for x in all_results]))
    mean_recall = float(np.mean([x["recall"] for x in all_results]))
    mean_f1 = float(np.mean([x["f1"] for x in all_results]))
    mean_iou = float(np.mean([x["iou"] for x in all_results]))
    avg_time = total_test_time / max(len(dataset), 1)

    print("\nTEST Summary")
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
            item["image"], item["gt"], item["pred"], save_path,
            title_suffix=f"Good Candidate\nP:{item['precision']:.4f} R:{item['recall']:.4f}\nF1:{item['f1']:.4f} IoU:{item['iou']:.4f}"
        )

    for item in bad_cases:
        save_path = os.path.join(BAD_CASES_DIR, item["name"].replace(".png", "_bad.png"))
        save_result_figure(
            item["image"], item["gt"], item["pred"], save_path,
            title_suffix=f"Bad Candidate\nP:{item['precision']:.4f} R:{item['recall']:.4f}\nF1:{item['f1']:.4f} IoU:{item['iou']:.4f}"
        )

    csv_path = os.path.join(output_dir, "test_metrics.csv")
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
        f.write("FPN Final Evaluation Summary\n")
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

# ==========================================
# Main
# ==========================================
def main():
    print("Using device:", DEVICE)

    train_dir = os.path.join(DATASET_DIR, TRAIN_SPLIT)
    val_dir = os.path.join(DATASET_DIR, VAL_SPLIT)
    test_dir = os.path.join(DATASET_DIR, TEST_SPLIT)

    train_dataset = EWSDataset(train_dir, transform=train_transform)
    val_dataset = EWSDataset(val_dir, transform=eval_transform)
    test_dataset = EWSDataset(test_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    model = smp.FPN(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def criterion(logits, masks):
        return 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    print("Building training set...")
    print(f"Using {len(train_dataset)} training images.")

    training_start = time.perf_counter()
    best_val_iou = -1.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate_validation(model, val_loader, criterion)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model to {BEST_MODEL_PATH}")

    total_training_time = time.perf_counter() - training_start
    print("\nTraining finished.")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Total training time: {total_training_time:.2f} sec")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    evaluate_on_test(model, test_dataset, RESULTS_DIR)

if __name__ == "__main__":
    main()