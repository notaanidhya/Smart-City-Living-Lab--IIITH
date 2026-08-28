"""
scripts/extract_features_v2.py
==============================
Extracts 24 CV features across all 1,000 images in data/manifest.csv -> data/features_v2.csv.
"""

import os, sys, cv2, time, pandas as pd, numpy as np
sys.path.insert(0, ".")
from ml.feature_extractor import extract_features, FEATURE_NAMES

MANIFEST_PATH = "data/manifest.csv"
OUTPUT_CSV = "data/features_v2.csv"

manifest = pd.read_csv(MANIFEST_PATH)
print(f"[*] Extracting 24 features for {len(manifest)} images in {MANIFEST_PATH}...")

records = []
start_t = time.time()

for idx, row in manifest.iterrows():
    img_path = os.path.join("data", "degraded", row["filename"])
    if not os.path.exists(img_path):
        img_path = os.path.join("data", "raw", row["filename"])
    
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"[!] Warning: Could not read {img_path}")
        continue

    feats = extract_features(bgr)
    rec = dict(row)
    rec.update(feats)
    records.append(rec)

    if (idx + 1) % 200 == 0 or (idx + 1) == len(manifest):
        print(f"    Extracted {idx+1}/{len(manifest)} ({(idx+1)/len(manifest)*100:.0f}%)...")

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
elapsed = time.time() - start_t
print(f"[+] Feature extraction complete in {elapsed:.1f}s. Saved to {OUTPUT_CSV}")
print(f"    Shape: {df.shape} | Feature count: {len(FEATURE_NAMES)}")
