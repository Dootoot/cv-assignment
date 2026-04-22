# File path: src/models/deeplabv3plus.py

import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm


class EWSWheatDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")

        # White background + black leaves -> leaf = 1, background = 0
        mask = (mask < 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        mask = mask.unsqueeze(0).float()
        return image, mask


def get_data_paths(split_dir):
    all_files = os.listdir(split_dir)

    img_names = sorted([
        f for f in all_files
        if not f.endswith("_mask.png") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    img_paths = [os.path.join(split_dir, n) for n in img_names]
    mask_paths = [os.path.join(split_dir, os.path.splitext(n)[0] + "_mask.png") for n in img_names]

    for mp in mask_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Missing corresponding mask file: {mp}")

    return img_paths, mask_paths


def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),

        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.0, 0.05),
            rotate=(-20, 20),
            shear=(-8, 8),
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3
        ),

        A.GaussNoise(p=0.2),

        A.Normalize(),
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
        ToTensorV2()
    ])


# Model loading
def load_trained_deeplabv3plus(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    model_path = os.path.join(model_dir, "deeplabv3plus_model.pth")
    if not os.path.exists(model_path):
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# TTA inference
def tta_predict_logits(model, images):
    """
    images: [B, C, H, W]
    Return the averaged logits.
    """
    logits_list = []

    # Original image
    logits = model(images)
    logits_list.append(logits)

    # Horizontal flip
    images_h = torch.flip(images, dims=[3])
    logits_h = model(images_h)
    logits_h = torch.flip(logits_h, dims=[3])
    logits_list.append(logits_h)

    # Vertical flip
    images_v = torch.flip(images, dims=[2])
    logits_v = model(images_v)
    logits_v = torch.flip(logits_v, dims=[2])
    logits_list.append(logits_v)

    # Horizontal + vertical flip
    images_hv = torch.flip(images, dims=[2, 3])
    logits_hv = model(images_hv)
    logits_hv = torch.flip(logits_hv, dims=[2, 3])
    logits_list.append(logits_hv)

    mean_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
    return mean_logits


# IoU evaluation
def evaluate_iou(model, loader, device, threshold=0.5, use_tta=False):
    model.eval()
    tp, fp, fn = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if use_tta:
                logits = tta_predict_logits(model, images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            tp += (preds * masks).sum().item()
            fp += (preds * (1 - masks)).sum().item()
            fn += ((1 - preds) * masks).sum().item()

    iou = tp / (tp + fp + fn + 1e-7)
    return iou


def search_best_threshold(model, loader, device, use_tta=False):
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_iou = -1.0
    best_threshold = 0.5

    for th in thresholds:
        iou = evaluate_iou(model, loader, device, threshold=th, use_tta=use_tta)
        if iou > best_iou:
            best_iou = iou
            best_threshold = th

    return best_threshold, best_iou


# Training
def generate_trained_deeplabv3plus(model_dir, train_path, val_path):
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_imgs, train_masks = get_data_paths(train_path)
    val_imgs, val_masks = get_data_paths(val_path)

    print(f"Number of training images: {len(train_imgs)}")
    print(f"Number of validation images: {len(val_imgs)}")

    train_dataset = EWSWheatDataset(
        train_imgs,
        train_masks,
        transform=get_train_transform()
    )

    val_dataset = EWSWheatDataset(
        val_imgs,
        val_masks,
        transform=get_val_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,           # More stable for 512 input size
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    dice_loss = smp.losses.DiceLoss(
        mode=smp.losses.BINARY_MODE,
        from_logits=True
    )

    # Positive class weighting for sparse foreground pixels
    pos_weight = torch.tensor([3.0], device=device)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def loss_fn(logits, masks):
        loss_dice = dice_loss(logits, masks)
        loss_bce = bce_loss(logits, masks)
        return 0.7 * loss_dice + 0.3 * loss_bce

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        verbose=True
    )

    best_iou = 0.0
    best_threshold = 0.5
    best_epoch = 0

    early_stop_patience = 12
    no_improve_count = 0

    num_epochs = 60

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Use TTA + best threshold search during validation
        current_threshold, current_iou = search_best_threshold(
            model,
            val_loader,
            device,
            use_tta=True
        )

        scheduler.step(current_iou)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"TrainLoss={avg_train_loss:.4f}, "
            f"ValIoU={current_iou:.4f}, "
            f"BestTh={current_threshold:.2f}, "
            f"LR={current_lr:.7f}"
        )

        if current_iou > best_iou:
            best_iou = current_iou
            best_threshold = current_threshold
            best_epoch = epoch + 1
            no_improve_count = 0

            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "deeplabv3plus_model.pth")
            )

            with open(os.path.join(model_dir, "deeplabv3plus_best_threshold.txt"), "w", encoding="utf-8") as f:
                f.write(str(best_threshold))

            with open(os.path.join(model_dir, "deeplabv3plus_best_iou.txt"), "w", encoding="utf-8") as f:
                f.write(str(best_iou))
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Training completed. Best Epoch = {best_epoch}, Best IoU = {best_iou:.4f}, Best Threshold = {best_threshold:.2f}")

    best_model = load_trained_deeplabv3plus(model_dir)
    return best_model


# Prediction and metric evaluation
def predict_from_trained_deeplabv3plus(output_dir, model, val_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_imgs, val_masks = get_data_paths(val_path)
    val_dataset = EWSWheatDataset(
        val_imgs,
        val_masks,
        transform=get_val_transform()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    threshold = 0.5
    threshold_file = None

    if output_dir:
        threshold_file = os.path.join(output_dir, "deeplabv3plus_best_threshold.txt")

    if threshold_file is not None and os.path.exists(threshold_file):
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                threshold = float(f.read().strip())
        except:
            threshold = 0.5
    else:
        threshold, _ = search_best_threshold(
            model,
            val_loader,
            device,
            use_tta=True
        )

    iou = evaluate_iou(
        model,
        val_loader,
        device,
        threshold=threshold,
        use_tta=True
    )

    return {
        "IoU": iou,
        "BestThreshold": threshold
    }


# Optional: local debugging entry point
if __name__ == "__main__":
    model_dir = "./checkpoints"
    train_path = "./dataset_v2_random/train"
    val_path = "./dataset_v2_random/val"

    model = generate_trained_deeplabv3plus(
        model_dir=model_dir,
        train_path=train_path,
        val_path=val_path
    )

    metrics = predict_from_trained_deeplabv3plus(
        output_dir=model_dir,
        model=model,
        val_path=val_path
    )

    print("Final validation results:", metrics)
