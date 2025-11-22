#!/usr/bin/env python3
"""
train_model_beast.py — BEAST MODE
Unsupervised training with Contrast Enhancement, Deep CNN, and Heavy Augmentation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import warnings

warnings.simplefilter("ignore")

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
# UPDATE THIS PATH to your actual image folder
IMG_DIR = Path("/Users/vedanshagarwal/Downloads/data/CYCLONE_DATASET_INFRARED")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "model_data"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "final_model.pth"

IMG_SIZE = (224, 224)
EPOCHS = 15  # Increased for better convergence
BATCH_SIZE = 32  # Increased batch size for batch normalization stability
LR = 1e-3


# --------------------------------------------------
# 1. ADVANCED PREPROCESSING
# --------------------------------------------------
def enhance_image(img):
    """
    Converts to grayscale and applies contrast enhancement (CLAHE-like effect)
    to make the cyclone eye and bands stand out against the background.
    """
    gray = ImageOps.grayscale(img)
    # Increase contrast to separate deep convection from thin clouds
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(1.5)
    return gray


def get_storm_metrics(gray_arr):
    """
    Calculates statistical 'intensity' metrics.
    Organized storms have: High max brightness, High standard deviation (contrast).
    """
    # Normalize
    arr = gray_arr / 255.0

    mean_val = float(arr.mean())
    max_val = float(arr.max())
    std_val = float(arr.std())

    # Laplacian variance (detects edges/spiral bands)
    gy, gx = np.gradient(arr)
    gxx = np.gradient(gx, axis=1)
    gyy = np.gradient(gy, axis=0)
    laplacian = gxx + gyy
    sharpness = float(np.var(laplacian))

    # Heuristic formula for cyclone intensity (Unsupervised)
    # High Max + High Variance (Organization) + High Sharpness (Eye wall)
    intensity_score = (max_val * 1.0) + (std_val * 2.0) + (sharpness * 50.0)

    return intensity_score


# --------------------------------------------------
# 2. BEAST ARCHITECTURE
# --------------------------------------------------
class BeastBlock(nn.Module):
    """Conv -> BatchNorm -> LeakyReLU -> MaxPool"""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CycloneBeast(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            BeastBlock(1, 32),  # 112x112
            BeastBlock(32, 64),  # 56x56
            BeastBlock(64, 128),  # 28x28
            BeastBlock(128, 256),  # 14x14
            BeastBlock(256, 512),  # 7x7
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> 1x1
        )

        # Head 1: Classification (Danger Level)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 3)  # Low, Med, High
        )

        # Head 2: Regression (Intensity Score)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch, 1, 224, 224]
        feats = self.features(x)
        return self.classifier(feats), self.regressor(feats).view(-1)


# --------------------------------------------------
# 3. DATA PIPELINE (WITH AUGMENTATION)
# --------------------------------------------------
class BeastDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["file"]

        # Load and Preprocess
        img = Image.open(img_path).convert("RGB")
        img = enhance_image(img)  # Convert to 1-channel Enhanced Grayscale

        if self.transform:
            img = self.transform(img)

        return img, int(row["label"]), np.float32(row["intensity"])


def get_transforms():
    # Training: Heavy Augmentation to learn rotation invariance
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(180),  # Cyclones can be any orientation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

    # Validation: Just resize and tensor
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
def main():
    # Setup Directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Generate Pseudo-Labels
    print("🔍 Scanning images and generating intensity metrics...")
    imgs = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not imgs:
        raise FileNotFoundError(f"No images found in {IMG_DIR}")

    data_rows = []
    for p in imgs:
        img = Image.open(p).convert("RGB")
        gray = enhance_image(img)
        arr = np.array(gray)

        score = get_storm_metrics(arr)
        data_rows.append({"file": p.name, "intensity": score})

    df = pd.DataFrame(data_rows)

    # 2. Clustering (K-Means)
    print("🧠 Clustering data into 3 Danger Levels...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df[["intensity"]])

    # Reorder clusters so 0=Low, 1=Med, 2=High based on mean intensity
    cluster_map = df.groupby("cluster")["intensity"].mean().sort_values().index
    mapping = {old: new for new, old in enumerate(cluster_map)}
    df["label"] = df["cluster"].map(mapping)

    # Save mapping for inference later
    print(f"   Label Mapping (Low->High): {mapping}")
    df.to_csv(OUT_DIR / "training_meta.csv", index=False)

    # 3. Setup Training
    train_tf, val_tf = get_transforms()

    # Split 80/20
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_ds = BeastDataset(train_df, IMG_DIR, train_tf)
    val_ds = BeastDataset(val_df, IMG_DIR, val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Init Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Launching training on {device}...")

    model = CycloneBeast().to(device)

    # AdamW handles weight decay better than Adam
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler: Starts low, goes high, ends low (Super convergence)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # 5. Training Loop
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for images, labels, intensities in train_loader:
            images, labels, intensities = images.to(device), labels.to(device), intensities.to(device)

            optimizer.zero_grad()

            pred_cls, pred_reg = model(images)

            # Loss = Classification + (0.1 * Regression)
            loss_c = criterion_cls(pred_cls, labels)
            loss_r = criterion_reg(pred_reg, intensities)
            total_loss = loss_c + (0.1 * loss_r)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += total_loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save Best Model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({"state": model.state_dict(), "mapping": mapping}, MODEL_PATH)

    print("\n🐯 BEAST MODE TRAINING COMPLETE.")
    print(f"💾 Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()