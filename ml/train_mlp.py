"""
ml/train_mlp.py
===============
Training script for Model A: Multi-Head MLP Issue Classifier.
Includes class balance tuning and validation threshold calibration.
Saves best checkpoint to ml/models/mlp_best.pt and thresholds to ml/models/mlp_thresholds.json.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.feature_extractor import FEATURE_NAMES
from ml.models.mlp import MultiLabelMLP, NUM_CLASSES

FEATURES_CSV   = os.path.join(BASE_DIR, "data", "features.csv")
CHECKPOINT     = os.path.join(BASE_DIR, "ml", "models", "mlp_best.pt")
THRESHOLDS_OUT = os.path.join(BASE_DIR, "ml", "models", "mlp_thresholds.json")

LABEL_COLS = ["has_blur", "has_underexposure", "has_overexposure",
              "has_noise", "has_corruption", "has_defect"]

# Training hyper-parameters
EPOCHS        = 80
BATCH_SIZE    = 64
LR            = 3e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 12          # early-stopping patience (epochs)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        X = df[FEATURE_NAMES].values.astype(np.float32)
        Y = df[LABEL_COLS].values.astype(np.float32)
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def compute_pos_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute moderate positive class weights using square root scaling to
    balance recall without causing excessive false positive sensitivity.
    """
    weights = []
    for col in LABEL_COLS:
        pos = train_df[col].sum()
        neg = len(train_df) - pos
        ratio = neg / max(pos, 1)
        # Moderate square-root scaling
        w = np.sqrt(ratio)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            feats = model.input_bn(X_b)
            feats = model.backbone(feats)
            logits = torch.cat([head(feats) for head in model.heads], dim=1)
            loss = criterion(logits, Y_b)
            total_loss += loss.item() * len(X_b)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(Y_b.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    acc = ((preds >= 0.5) == targets).mean()
    return avg_loss, float(acc)


def calibrate_thresholds(model, val_loader, device) -> dict:
    """
    Finds the threshold on the validation split that maximizes F1 for each class.
    """
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for X_b, Y_b in val_loader:
            X_b = X_b.to(device)
            probs = model(X_b)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(Y_b.numpy())

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)

    calibrated = {}
    print("\n[*] Calibrating per-class decision thresholds on Validation set:")
    for i, col in enumerate(LABEL_COLS):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.20, 0.85, 66):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        calibrated[col] = {
            "threshold": round(best_t, 3),
            "val_f1": round(float(best_f1), 4)
        }
        print(f"    - {col:<20}: threshold = {best_t:.2f} (Val F1 = {best_f1:.3f})")

    return calibrated


def main():
    print(f"[*] Training Multi-Head MLP on device: {DEVICE}")
    df = pd.read_csv(FEATURES_CSV)

    for col in FEATURE_NAMES:
        df[col] = df[col].fillna(df[col].median())

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    print(f"    Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = FeatureDataset(train_df)
    val_ds   = FeatureDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiLabelMLP().to(DEVICE)

    pos_weights = compute_pos_weights(train_df).to(DEVICE)
    print(f"    Class pos_weights: {dict(zip(LABEL_COLS, pos_weights.cpu().numpy().round(2)))}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    def forward_logits(x):
        x = model.input_bn(x)
        feats = model.backbone(x)
        return torch.cat([head(feats) for head in model.heads], dim=1)

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = forward_logits(X_b)
            loss = criterion(logits, Y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)

        train_loss /= len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_acc"].append(round(val_acc, 5))

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr_now:.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "feature_names": FEATURE_NAMES,
                "label_cols":    LABEL_COLS,
            }, CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    [Early Stop] No improvement for {PATIENCE} epochs. Stopping at epoch {epoch}.")
                break

    # Save training history
    history_path = os.path.join(BASE_DIR, "ml", "models", "mlp_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[+] MLP training complete. Best val_loss={best_val_loss:.4f}")
    print(f"    Checkpoint saved to: {CHECKPOINT}")

    # Load best checkpoint and perform threshold calibration
    best_ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])
    thresholds = calibrate_thresholds(model, val_loader, DEVICE)
    with open(THRESHOLDS_OUT, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"    Calibrated thresholds saved to: {THRESHOLDS_OUT}")

if __name__ == "__main__":
    main()
