"""
ml/train_autoencoder.py
=======================
Training script for Model B: Convolutional Autoencoder (Anomaly Detector).

Key design decisions:
- Trained ONLY on clean images (quality_label == "ACCEPTABLE").
  This forces the AE to learn the manifold of pristine images;
  anything anomalous produces elevated reconstruction error at inference.
- Reconstruction target = normalised input (pixel values in [0, 1]).
- Loss = MSE per pixel, averaged over the batch.
- Input images are resized to 128x128 RGB.
- Saves reconstruction error statistics (mean + 2*std from training set)
  as the anomaly detection threshold. This threshold selection is reported
  during evaluation.
"""

import os, sys, json, glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.models.autoencoder import ConvAutoencoder, IMG_SIZE

MANIFEST_CSV  = os.path.join(BASE_DIR, "data", "manifest.csv")
CHECKPOINT    = os.path.join(BASE_DIR, "ml", "models", "autoencoder_best.pt")
THRESHOLD_FILE = os.path.join(BASE_DIR, "ml", "models", "ae_threshold.json")

# Training hyper-parameters
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE    = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


class CleanImageDataset(Dataset):
    """Loads clean images, resizes to IMG_SIZE x IMG_SIZE, normalises to [0,1]."""
    def __init__(self, file_paths: list[str]):
        self.paths = file_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bgr = cv2.imread(self.paths[idx])
        if bgr is None:
            bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        return tensor


def main():
    print(f"[*] Training Convolutional Autoencoder on device: {DEVICE}")

    df = pd.read_csv(MANIFEST_CSV)
    clean_df = df[df["quality_label"] == "ACCEPTABLE"].reset_index(drop=True)

    train_paths = clean_df[clean_df["split"] == "train"]["filepath"].tolist()
    val_paths   = clean_df[clean_df["split"] == "val"]["filepath"].tolist()

    print(f"    Clean train images: {len(train_paths)} | Clean val images: {len(val_paths)}")

    train_ds = CleanImageDataset(train_paths)
    val_ds   = CleanImageDataset(val_paths)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model     = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss    = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Training ──
        model.train()
        train_loss = 0.0
        for imgs in train_loader:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * imgs.shape[0]
        train_loss /= len(train_loader.dataset)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(DEVICE)
                recon = model(imgs)
                val_loss += criterion(recon, imgs).item() * imgs.shape[0]
        val_loss /= max(len(val_loader.dataset), 1)

        scheduler.step(val_loss)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr_now:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "img_size":    IMG_SIZE,
            }, CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    [Early Stop] Stopping at epoch {epoch}.")
                break

    # ── Compute anomaly detection threshold on full training set ──
    # Load best model
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\n[*] Computing anomaly threshold from training set reconstruction errors...")
    all_errors = []
    all_ds = CleanImageDataset(train_paths + val_paths)
    all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for imgs in all_loader:
            imgs = imgs.to(DEVICE)
            errs = model.reconstruction_error(imgs)
            all_errors.extend(errs.cpu().numpy().tolist())

    err_arr = np.array(all_errors)
    threshold = float(err_arr.mean() + 2.0 * err_arr.std())
    threshold_json = {
        "mean_clean_error":   float(err_arr.mean()),
        "std_clean_error":    float(err_arr.std()),
        "threshold_2sigma":   threshold,
        "calibration_scale":  float(threshold),   # normalisation scale for score.py
    }
    with open(THRESHOLD_FILE, "w") as f:
        json.dump(threshold_json, f, indent=2)

    # Save history
    history_path = os.path.join(BASE_DIR, "ml", "models", "ae_training_history.json")
    with open(history_path, "w") as hf:
        json.dump(history, hf, indent=2)

    print(f"[+] Autoencoder training complete. Best val_loss={best_val_loss:.6f}")
    print(f"    Anomaly threshold (mean+2*std): {threshold:.6f}")
    print(f"    Checkpoint:  {CHECKPOINT}")
    print(f"    Threshold:   {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()
