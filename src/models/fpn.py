import os
import cv2
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .helpers import predict_from_trained_model


IMG_SIZE = 512
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
THRESHOLD = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        mask = (mask > 0).astype(np.float32)

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


def build_fpn_model():
    model = smp.FPN(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )
    return model.to(DEVICE)


def get_model_path(model_dir: str):
    return os.path.join(model_dir, "fpn_model.pth")


def load_trained_fpn(model_dir: str):
    model_path = get_model_path(model_dir)

    if not os.path.exists(model_path):
        return None

    model = build_fpn_model()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def generate_trained_fpn(model_dir: str, image_folder_path: str, val_path: str):
    if image_folder_path is None or val_path is None:
        raise RuntimeError("Invalid training or validation folder path")

    os.makedirs(model_dir, exist_ok=True)

    train_dataset = EWSDataset(image_folder_path, transform=train_transform)
    val_dataset = EWSDataset(val_path, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = build_fpn_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")
    best_val_loss = float("inf")

    print(f"Using device: {DEVICE}")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss)
        val_loss = evaluate_one_epoch(model, val_loader, bce_loss, dice_loss)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), get_model_path(model_dir))

    best_model = load_trained_fpn(model_dir)
    return best_model


def predict_from_trained_fpn(output_dir: str, model, testing_folder: str):
    return predict_from_trained_model(
        output_dir,
        testing_folder,
        model,
        single_image_predict_from_trained_fpn
    )


def single_image_predict_from_trained_fpn(model, image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented = eval_transform(image=image_rgb)
    image = augmented["image"]

    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > THRESHOLD).float().cpu().numpy()[0, 0]

    pred = (pred * 255).astype(np.uint8)

    original_h, original_w = image_bgr.shape[:2]
    pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return pred


def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss):
    model.train()
    total_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)

        loss = bce_loss(logits, masks) + dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate_one_epoch(model, loader, bce_loss, dice_loss):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)