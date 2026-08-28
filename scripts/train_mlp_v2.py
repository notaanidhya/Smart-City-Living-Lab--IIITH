"""
scripts/train_mlp_v2.py
=======================
Trains Candidate Model A v2 with 24 features, Hard Negative Mining & Focal Loss on Defect.
Saves checkpoint to ml/models/mlp_v2.pt and thresholds to ml/models/mlp_thresholds_v2.json.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

sys.path.insert(0, ".")
from ml.feature_extractor import FEATURE_NAMES
from ml.models.mlp import MultiLabelMLP, NUM_CLASSES

FEATURES_CSV   = "data/features_v2.csv"
CHECKPOINT     = "ml/models/mlp_v2.pt"
THRESHOLDS_OUT = "ml/models/mlp_thresholds_v2.json"

LABEL_COLS = ["has_blur", "has_underexposure", "has_overexposure",
              "has_noise", "has_corruption", "has_defect"]

EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 3e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 14
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        return loss.mean()

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
    weights = []
    for col in LABEL_COLS:
        pos = train_df[col].sum()
        neg = len(train_df) - pos
        ratio = neg / max(pos, 1)
        w = np.sqrt(ratio) if col != "has_defect" else np.sqrt(ratio) * 1.2
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, device, pos_weights):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    focal_defect = FocalLoss(gamma=2.0, pos_weight=pos_weights[5])
    
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            feats = model.input_bn(X_b)
            feats = model.backbone(feats)
            logits = torch.cat([head(feats) for head in model.heads], dim=1)
            
            # Loss computation
            loss_standard = F.binary_cross_entropy_with_logits(logits[:, :5], Y_b[:, :5], pos_weight=pos_weights[:5])
            loss_defect   = focal_defect(logits[:, 5:6], Y_b[:, 5:6])
            loss = loss_standard + 1.2 * loss_defect
            
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
    print(f"[*] Training Multi-Head MLP v2 (24 features) on: {DEVICE}")
    df = pd.read_csv(FEATURES_CSV)

    for col in FEATURE_NAMES:
        df[col] = df[col].fillna(df[col].median())

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Features: {len(FEATURE_NAMES)}")

    train_ds = FeatureDataset(train_df)
    val_ds   = FeatureDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiLabelMLP(input_dim=len(FEATURE_NAMES)).to(DEVICE)
    pos_weights = compute_pos_weights(train_df).to(DEVICE)
    focal_defect = FocalLoss(gamma=2.0, pos_weight=pos_weights[5])

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
            optimizer.zero_grad()
            
            feats = model.input_bn(X_b)
            feats = model.backbone(feats)
            logits = torch.cat([head(feats) for head in model.heads], dim=1)

            loss_standard = F.binary_cross_entropy_with_logits(logits[:, :5], Y_b[:, :5], pos_weight=pos_weights[:5])
            loss_defect   = focal_defect(logits[:, 5:6], Y_b[:, 5:6])
            loss = loss_standard + 1.2 * loss_defect

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)

        train_loss /= len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE, pos_weights)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "feature_names": FEATURE_NAMES,
                "label_cols": LABEL_COLS,
            }, CHECKPOINT)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    [Early Stop] Epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

    print(f"\n[+] Training complete. Best val_loss={best_val_loss:.4f}. Checkpoint: {CHECKPOINT}")
    best_ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])
    thresholds = calibrate_thresholds(model, val_loader, DEVICE)
    with open(THRESHOLDS_OUT, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"    Saved thresholds to: {THRESHOLDS_OUT}")

if __name__ == "__main__":
    main()
