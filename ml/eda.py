"""
ml/eda.py
=========
Quick EDA sanity pass on features.csv:
  1. Feature correlation matrix → docs/sample_images/eda_correlation.png
  2. Per-label feature distribution → docs/sample_images/eda_distributions.png
  3. Feature importance proxy via mutual information → printed table

forgot what eda was :( [Exploratory Data Analysis]
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from ml.feature_extractor import FEATURE_NAMES

FEATURES_CSV = os.path.join(BASE_DIR, "data", "features.csv")
DOCS_DIR = os.path.join(BASE_DIR, "docs", "sample_images")
os.makedirs(DOCS_DIR, exist_ok=True)

df = pd.read_csv(FEATURES_CSV)
X = df[FEATURE_NAMES].fillna(0).replace([np.inf, -np.inf], 0)

# ── 1. correlation matrix for features 
fig, ax = plt.subplots(figsize=(14, 12))
corr = X.corr()
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURE_NAMES)))
ax.set_yticks(range(len(FEATURE_NAMES)))
ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.03)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "eda_correlation.png"), dpi=150)
plt.close()
print("[+] Saved eda_correlation.png")

# ── 2. Per-label boxplots for the 6 most diagnostic features 
key_features = [
    "laplacian_variance", "mean_luminance", "bright_pixel_ratio",
    "noise_sigma_immerkaar", "dct_blockiness", "mean_saturation"
]
label_order = ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
colors = {"ACCEPTABLE": "#4caf50", "DEGRADED": "#ff9800", "DEFECTIVE": "#f44336"}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    ax = axes[i]
    data_by_label = [df[df["quality_label"] == lbl][feat].dropna().values for lbl in label_order]
    bp = ax.boxplot(data_by_label, patch_artist=True, notch=False, medianprops=dict(color="black", linewidth=2))
    for patch, lbl in zip(bp["boxes"], label_order):
        patch.set_facecolor(colors[lbl])
    ax.set_xticklabels(label_order, fontsize=9)
    ax.set_title(feat, fontsize=10, fontweight="bold")
    ax.set_ylabel("Value", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Feature Distributions by Quality Label", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(DOCS_DIR, "eda_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved eda_distributions.png")



label_cols = ["has_blur", "has_underexposure", "has_overexposure", "has_noise", "has_corruption", "has_defect"]
print("\n[EDA] Mutual Information — feature relevance per label (top 8 per label):\n")
for col in label_cols:
    y = df[col].values
    mi = mutual_info_classif(X.values, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=FEATURE_NAMES).sort_values(ascending=False)
    top = mi_series.head(8)
    print(f"  [{col}]")
    for feat, score in top.items():
        print(f"    {feat:<30} {score:.4f}")
    print()


print("[EDA] High-correlation feature pairs (|r| > 0.85):")
found = False
for i in range(len(FEATURE_NAMES)):
    for j in range(i+1, len(FEATURE_NAMES)):
        r = corr.iloc[i, j]
        if abs(r) > 0.85:
            print(f"    {FEATURE_NAMES[i]:30} <-> {FEATURE_NAMES[j]:30}  r={r:.3f}")
            found = True
if not found:
    print("    None — all features have |r| < 0.85. No redundancy detected.")

print("\n[EDA] NaN check after fix:")
nan_counts = df[FEATURE_NAMES].isna().sum()
bad = nan_counts[nan_counts > 0]
if len(bad):
    print(bad.to_string())
else:
    print("    [OK] Zero NaN values in features.csv")
