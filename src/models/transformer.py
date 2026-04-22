import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from transformers import SegformerForSemanticSegmentation

import cv2

from .helpers import predict_from_trained_model, compute_metrics

NUM_CLASSES = 2
IMAGE_SIZE = (384, 384)
BATCH_SIZE = 4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
LR = 1e-4
EARLY_STOPPING_MIN_DELTA = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 3. Dataset
# =========================================================
class CropSegDataset(Dataset):
    def __init__(self, folder_path, image_size=(384, 384), train=False, has_masks=True):
        self.folder_path = Path(folder_path)
        self.image_size = image_size
        self.train = train

        all_files = sorted(self.folder_path.glob("*.png"))
        self.image_paths = [p for p in all_files if "_mask" not in p.stem]

    def __len__(self):
        return len(self.image_paths)

    def _get_mask_path(self, img_path: Path) -> Path:
        return img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        mask_path = self._get_mask_path(img_path)
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, self.image_size, antialias=True)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        mask = np.array(mask, dtype=np.uint8)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        mask = torch.as_tensor(mask, dtype=torch.long)
        return image, mask


# load a pretrained model segformer B2
def get_model(num_classes):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model = model.to(DEVICE)
    return model


def forward_logits(model, images, target_size):
    images = images.to(DEVICE, non_blocking=True)
    outputs = model(pixel_values=images)
    logits = outputs.logits

    if logits.shape[-2:] != target_size:
        logits = F.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

    return logits.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        logits = forward_logits(model, images, masks.shape[-2:])
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_iou = 0.0

    for images, masks in loader:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        logits = forward_logits(model, images, masks.shape[-2:])
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        for pred, mask in zip(preds, masks):
            pred_np = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
            mask_np = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
            metrics = compute_metrics(pred_np, mask_np)
            total_precision += metrics["precision"]
            total_recall += metrics["recall"]
            total_f1 += metrics["f1_score"]
            total_iou += metrics["intersection_over_union"]

    n_batches = len(loader)
    n_samples = len(loader.dataset)

    return (
        total_loss / n_batches,
        total_iou / n_samples,
        total_precision / n_samples,
        total_recall / n_samples,
        total_f1 / n_samples,
    )


def generate_trained_transformer(model_dir: str, image_folder_path: str, val_path: str):
    print("Using device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = DataLoader(
        CropSegDataset(image_folder_path, image_size=IMAGE_SIZE, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        CropSegDataset(val_path, image_size=IMAGE_SIZE, train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = get_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    save_path = Path(model_dir) / "segformer_b2_best.pth"
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_iou, val_precision, val_recall, val_f1 = evaluate(
            model, val_loader, criterion
        )
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mIoU: {val_iou:.4f} | "
            f"P: {val_precision:.4f} | "
            f"R: {val_recall:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    print(f"\nLowest validation loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def load_trained_transformer(model_dir: str):
    path = Path(model_dir) / "segformer_b2_best.pth"
    if not path.exists():
        return None

    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def _single_image_predict(model, image_path: str):
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size   # (width, height)

    image = TF.resize(original_image, IMAGE_SIZE, antialias=True)
    image = TF.to_tensor(image)
    image = TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image = image.unsqueeze(0).to(DEVICE, non_blocking=True)

    model.eval()
    with torch.no_grad():
        logits = forward_logits(model, image, IMAGE_SIZE)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # convert {0,1} -> {0,255}
    mask = preds * 255

    # resize prediction back to original image size
    mask = cv2.resize(
        mask,
        original_size,   # (width, height)
        interpolation=cv2.INTER_NEAREST
    )

    return mask.astype(np.uint8)


def predict_from_trained_transformer(output_dir: str, model, testing_folder: str):
    return predict_from_trained_model(output_dir, testing_folder, model, _single_image_predict)