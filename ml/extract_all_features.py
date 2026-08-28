"""
ml/extract_all_features.py
==========================
Runs extract_features() over every sample in data/manifest.csv and
produces data/features.csv — the feature matrix (X) + labels (Y) used
for MLP training, EDA, and evaluation.
"""
import os, sys, time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from ml.feature_extractor import extract_features, FEATURE_NAMES

MANIFEST = os.path.join(BASE_DIR, "data", "manifest.csv")
FEATURES_OUT = os.path.join(BASE_DIR, "data", "features.csv")


def process_row(row):
    try:
        feats = extract_features(row["filepath"])
        feats["filename"] = row["filename"]
        feats["has_blur"] = row["has_blur"]
        feats["has_underexposure"] = row["has_underexposure"]
        feats["has_overexposure"] = row["has_overexposure"]
        feats["has_noise"] = row["has_noise"]
        feats["has_corruption"] = row["has_corruption"]
        feats["has_defect"] = row["has_defect"]
        feats["quality_score"] = row["quality_score"]
        feats["quality_label"] = row["quality_label"]
        feats["split"] = row["split"]
        return True, feats, None
    except Exception as e:
        return False, None, f"{row['filename']}: {e}"


def main():
    df = pd.read_csv(MANIFEST)
    total = len(df)
    print(f"[*] Extracting features for {total} samples (using 8 threads)...")

    rows = df.to_dict("records")
    results = []
    errors = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_row, r): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            ok, feats, err = fut.result()
            done += 1
            if ok:
                results.append(feats)
            else:
                errors.append(err)
            if done % 100 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"    -> {done}/{total} processed ({elapsed:.1f}s elapsed, {len(errors)} errors)")

    out_df = pd.DataFrame(results)
    # Reorder: features first, then labels
    col_order = FEATURE_NAMES + ["filename", "has_blur", "has_underexposure", "has_overexposure",
                                  "has_noise", "has_corruption", "has_defect",
                                  "quality_score", "quality_label", "split"]
    out_df = out_df[[c for c in col_order if c in out_df.columns]]
    out_df.to_csv(FEATURES_OUT, index=False)

    print(f"\n[+] Feature extraction complete!")
    print(f"    - Total samples: {len(out_df)}")
    print(f"    - Features: {len(FEATURE_NAMES)}")
    print(f"    - Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"      [!] {e}")
    print(f"    - Saved to: {FEATURES_OUT}")

    # Quick EDA sanity output
    feat_df = out_df[FEATURE_NAMES]
    print(f"\n[EDA] Feature stats across all samples:")
    print(feat_df.describe().round(4).to_string())

    # Check for NaN / Inf
    nan_counts = feat_df.isna().sum()
    inf_counts = np.isinf(feat_df.values).sum(axis=0)
    bad = [(f, nan_counts[f], inf_counts[i]) for i, f in enumerate(FEATURE_NAMES) if nan_counts[f] > 0 or inf_counts[i] > 0]
    if bad:
        print("\n[!] Features with NaN/Inf:")
        for f, nans, infs in bad:
            print(f"    {f}: NaN={nans}, Inf={infs}")
    else:
        print("\n[OK] No NaN or Inf values detected in any feature.")

if __name__ == "__main__":
    main()
