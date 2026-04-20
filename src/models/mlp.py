import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import cv2
import numpy as np
from pathlib import Path

import albumentations

from .helpers import predict_from_trained_model


class PatchMLPSegmenter(nn.Module):
    def __init__(self, in_channels=3, patch_size=5, hidden_dims=(64, 32), dropout=0.2):
        super().__init__()

        self.patch_size = patch_size
        patch_dim = in_channels * patch_size * patch_size

        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        # extract local patches around each pixel
        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            padding=self.patch_size // 2,
        )  # [B, C*K*K, H*W]

        patch_dim = patches.shape[1]
        patches = patches.transpose(1, 2).reshape(-1, patch_dim)  # [B*H*W, C*K*K]

        logits = self.mlp(patches)  # [B*H*W, 1]
        logits = logits.view(b, h * w, 1).transpose(1, 2).reshape(b, 1, h, w)

        return logits


class SegmentationDataset(Dataset):
    def __init__(self, folder_path, augment=False):
        self.folder_path = Path(folder_path)
        self.image_paths = sorted(
            p for p in self.folder_path.glob("*.png") if not p.name.endswith("_mask.png")
        )

        if augment:
            self.transform = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.RandomRotate90(p=0.5),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3,
                ),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = str(image_path).removesuffix(".png") + "_mask.png"

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image at {image_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask at {mask_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image_rgb, mask=mask)
            image_rgb = augmented["image"]
            mask = augmented["mask"]

        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1)

        mask = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def _run_validation(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    return val_loss / len(dataloader)


def generate_trained_mlp(model_dir: str, image_folder_path: str, val_path: str):
    device = _get_device()
    print(f"Using device: {device}")

    train_loader = DataLoader(
        SegmentationDataset(image_folder_path, augment=True),
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SegmentationDataset(val_path, augment=False),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = PatchMLPSegmenter(
        in_channels=3,
        patch_size=5,
        hidden_dims=(64, 32),
        dropout=0.2,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=4,
    )

    save_path = Path(model_dir) / "mlp_model.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = 30
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 8

    for epoch in range(num_epochs):
        train_loss = _run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_validation(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Stopped early due to no validation improvement.")
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_trained_mlp(model_dir: str):
    path = Path(model_dir) / "mlp_model.pth"
    if not path.exists():
        return None

    device = _get_device()
    model = PatchMLPSegmenter(
        in_channels=3,
        patch_size=5,
        hidden_dims=(64, 32),
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _single_image_predict(model, image_path: str):
    device = next(model.parameters()).device

    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)

    probs = torch.sigmoid(logits[0, 0]).cpu().numpy()

    mask = (probs > 0.5).astype(np.uint8) * 255

    return mask


def predict_from_trained_mlp(output_dir: str, model, testing_folder: str):
    return predict_from_trained_model(output_dir, testing_folder, model, _single_image_predict)