import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import cv2
import numpy as np
from pathlib import Path

import albumentations as A

from .helpers import predict_from_trained_model

# check unet original architecture paper for reference

# Double convolution is stacking 2 3x3-kernels to provide a larger receptive field of 5x5
# whilst keeping the kernel smaller, i.e. 18 vs 25 parameters
# Each level is 2 layers of neurons: Conv -> BatchNorm -> ReLU -> (repeat)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # bias=False because BatchNorm already has its own bias
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    # pytorch calls this during forward propagation
    def forward(self, x):
        return self.double_conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.double_conv(x)
        p = self.pool(x)
        # returns both: x for skip connection, p for next encoder level
        return x, p


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 2, stride = 2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        # concatenate along channel dimension (skip has matching spatial dims)
        x = torch.cat([x, skip_connection], dim = 1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    # out_channels = 1 for binary segmentation
    # in_channels = 3 for RGB image
    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()

        self.down_sample1 = DownSampleBlock(in_channels, 64)
        self.down_sample2 = DownSampleBlock(64, 128)
        self.down_sample3 = DownSampleBlock(128, 256)
        self.down_sample4 = DownSampleBlock(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up_sample1 = UpSampleBlock(1024, 512)
        self.up_sample2 = UpSampleBlock(512, 256)
        self.up_sample3 = UpSampleBlock(256, 128)
        self.up_sample4 = UpSampleBlock(128, 64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x):
        # encoder: each level saves output for skip connection
        down1, p1 = self.down_sample1(x)
        down2, p2 = self.down_sample2(p1)
        down3, p3 = self.down_sample3(p2)
        down4, p4 = self.down_sample4(p3)

        bottleneck = self.bottleneck(p4)

        # decoder: upsample + skip connection from encoder
        up1 = self.up_sample1(bottleneck, down4)
        up2 = self.up_sample2(up1, down3)
        up3 = self.up_sample3(up2, down2)
        up4 = self.up_sample4(up3, down1)

        # raw logits (no sigmoid — handled by BCEWithLogitsLoss during training)
        return self.output_layer(up4)


# dataset

class SegmentationDataset(Dataset):
    def __init__(self, folder_path, augment = False):
        self.folder_path = Path(folder_path)
        self.image_paths = [
            p for p in self.folder_path.glob("*.png") if not p.name.endswith("_mask.png")
        ]

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.RandomRotate90(p = 0.5),
                A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.2, p = 0.3),
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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask at {mask_path}")

        # augmentation (on numpy arrays, before tensor conversion)
        if self.transform:
            augmented = self.transform(image = image_rgb, mask = mask)
            image_rgb = augmented["image"]
            mask = augmented["mask"]

        # normalise to [0, 1] and convert to tensors
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW for pytorch
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)  # add channel dim

        # pad 350 -> 352 (1px each side) so dimensions are divisible by 16
        image_tensor = F.pad(image_tensor, (1, 1, 1, 1), mode = 'reflect')
        mask_tensor = F.pad(mask_tensor, (1, 1, 1, 1), mode = 'reflect')

        return image_tensor, mask_tensor


# training helpers

def _get_device():
    # MPS can segfault on some macOS versions, uncomment below to try whether urs is ok or not
    # use cuda if u have nvidia

    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_one_epoch(model, dataloader, criterion, optimizer, device):
    # Run one training epoch, returns average loss
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()       # clear old gradients
        outputs = model(images)     # forward pass
        loss = criterion(outputs, masks)
        loss.backward()             # backprop: compute gradients for all weights
        optimizer.step()            # update weights using gradients

        running_loss += loss.item()

    return running_loss / len(dataloader)


def _run_validation(model, dataloader, criterion, device):
    # eval without gradient, only check loss since we don't want to improve model with
    # validation data
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


def generate_trained_unet(model_dir: str, image_folder_path: str, val_path: str):
    device = _get_device()
    print(f"Using device: {device}")

    train_loader = DataLoader(
        SegmentationDataset(image_folder_path, augment = True),
        batch_size = 4, shuffle = True, num_workers = 0
    )
    val_loader = DataLoader(
        SegmentationDataset(val_path, augment = False),
        batch_size = 4, shuffle = False, num_workers = 0
    )

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 5)

    save_path = Path(model_dir) / "unet_model.pth"
    num_epochs = 50
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(num_epochs):
        train_loss = _run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_validation(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # only saving val loss because train loss only represents under or overfitting
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Stopped early due to patience counter hitting threshold (no improvements in 10 epochs)")
                break

    print(f"\nlowest validation loss: {best_val_loss:.4f}")

    # load the best checkpoint (not necessarily the last epoch)
    model.load_state_dict(torch.load(save_path, map_location = device, weights_only = True))
    model.eval()
    return model


# loading and prediction helpers

def load_trained_unet(model_dir: str):
    path = Path(model_dir) / "unet_model.pth"
    if not path.exists():
        return None

    device = _get_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load(path, map_location = device, weights_only = True))
    model.eval()
    return model


def _single_image_predict(model, image_path: str):
    # predict segmentation mask for a single image. Returns uint8 mask (0 or 255)
    device = next(model.parameters()).device

    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # same preprocessing as dataset: normalise, permute, pad
    img_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    # height width channel to channel height width for pytorch conv
    img_tensor = img_tensor.permute(2, 0, 1)  # CHW

    # 1 pixel frame edge for padding
    img_tensor = F.pad(img_tensor, (1, 1, 1, 1), mode = 'reflect')

    img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dim: [1, 3, 352, 352]

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)  # [1, 1, 352, 352]

    # crop padding back to 350x350, apply sigmoid, threshold at 0.5
    probs = torch.sigmoid(logits[0, 0, 1:-1, 1:-1]).cpu().numpy()
    mask = (probs > 0.5).astype(np.uint8) * 255

    return mask


def predict_from_trained_unet(output_dir: str, model, testing_folder: str):
    # Predict on all images in folder, compute metrics, save comparison images
    return predict_from_trained_model(output_dir, testing_folder, model, _single_image_predict)
