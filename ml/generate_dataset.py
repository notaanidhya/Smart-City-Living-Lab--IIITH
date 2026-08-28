import os
import sys
import glob
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.degrade import DEGRADATION_MAP

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DEGRADED_DIR = os.path.join(BASE_DIR, "data", "degraded")
DOCS_SAMPLES_DIR = os.path.join(BASE_DIR, "docs", "sample_images")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "manifest.csv")

SEVERITY_LEVELS = ["mild", "moderate", "severe"]
SEVERITY_WEIGHTS = {"none": 0.0, "mild": 12.0, "moderate": 24.0, "severe": 38.0}
DEFECT_WEIGHTS = {"none": 0.0, "mild": 20.0, "moderate": 40.0, "severe": 60.0}

def compute_quality_score(issues_dict: dict) -> float:
    penalty = 0.0
    for issue, sev in issues_dict.items():
        if issue == "defect":
            penalty += DEFECT_WEIGHTS.get(sev, 0.0)
        else:
            penalty += SEVERITY_WEIGHTS.get(sev, 0.0)
    score = max(0.0, min(100.0, 100.0 - penalty))
    return round(score, 1)

def determine_label(score: float, issues_dict: dict) -> str:
    if issues_dict.get("defect", "none") in ["moderate", "severe"] or score < 40.0:
        return "DEFECTIVE"
    elif score < 75.0:
        return "DEGRADED"
    else:
        return "ACCEPTABLE"

def main():
    random.seed(42)
    np.random.seed(42)

    os.makedirs(DEGRADED_DIR, exist_ok=True)
    os.makedirs(DOCS_SAMPLES_DIR, exist_ok=True)

    raw_images = sorted(glob.glob(os.path.join(RAW_DIR, "*.jpg")))
    if not raw_images:
        print(f"[!] No images found in {RAW_DIR}")
        return

    print(f"[*] Processing {len(raw_images)} clean base images to generate multi-label synthetic dataset...")
    records = []

    # 1. Add all clean images as ACCEPTABLE reference class
    for img_path in raw_images:
        fname = os.path.basename(img_path)
        dest_path = os.path.join(DEGRADED_DIR, fname)
        img = cv2.imread(img_path)
        cv2.imwrite(dest_path, img)

        issues = {k: "none" for k in DEGRADATION_MAP.keys()}
        q_score = 100.0
        records.append({
            "filename": fname,
            "filepath": dest_path,
            "base_image": fname,
            "blur_severity": "none",
            "underexposure_severity": "none",
            "overexposure_severity": "none",
            "noise_severity": "none",
            "corruption_severity": "none",
            "defect_present": "none",
            "has_blur": 0,
            "has_underexposure": 0,
            "has_overexposure": 0,
            "has_noise": 0,
            "has_corruption": 0,
            "has_defect": 0,
            "quality_score": q_score,
            "quality_label": "ACCEPTABLE"
        })

    # 2. Generate single-issue degradations across all severities
    deg_idx = 1
    for img_path in raw_images:
        img_bgr = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # For each clean image, generate 3 single-issue variants
        chosen_issues = random.sample(list(DEGRADATION_MAP.keys()), 3)
        for issue_name in chosen_issues:
            sev = random.choice(SEVERITY_LEVELS)
            transform_fn = DEGRADATION_MAP[issue_name]
            deg_img = transform_fn(img_bgr, severity=sev)

            out_name = f"{base_name}_{issue_name}_{sev}_{deg_idx:04d}.jpg"
            dest_path = os.path.join(DEGRADED_DIR, out_name)
            cv2.imwrite(dest_path, deg_img)

            issues = {k: "none" for k in DEGRADATION_MAP.keys()}
            issues[issue_name] = sev
            q_score = compute_quality_score(issues)
            q_label = determine_label(q_score, issues)

            record = {
                "filename": out_name,
                "filepath": dest_path,
                "base_image": os.path.basename(img_path),
                "blur_severity": issues["blur"],
                "underexposure_severity": issues["underexposure"],
                "overexposure_severity": issues["overexposure"],
                "noise_severity": issues["noise"],
                "corruption_severity": issues["corruption"],
                "defect_present": issues["defect"],
                "has_blur": int(issues["blur"] != "none"),
                "has_underexposure": int(issues["underexposure"] != "none"),
                "has_overexposure": int(issues["overexposure"] != "none"),
                "has_noise": int(issues["noise"] != "none"),
                "has_corruption": int(issues["corruption"] != "none"),
                "has_defect": int(issues["defect"] != "none"),
                "quality_score": q_score,
                "quality_label": q_label
            }
            records.append(record)
            deg_idx += 1

    # 3. Generate multi-issue co-occurring degradations
    for img_path in raw_images:
        img_bgr = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        available_pairs = [
            ("blur", "noise"),
            ("underexposure", "noise"),
            ("overexposure", "corruption"),
            ("blur", "corruption"),
            ("noise", "defect"),
            ("underexposure", "defect")
        ]
        chosen_pair = random.choice(available_pairs)
        sev1 = random.choice(SEVERITY_LEVELS)
        sev2 = random.choice(SEVERITY_LEVELS)

        deg_img = DEGRADATION_MAP[chosen_pair[0]](img_bgr, severity=sev1)
        deg_img = DEGRADATION_MAP[chosen_pair[1]](deg_img, severity=sev2)

        out_name = f"{base_name}_multi_{chosen_pair[0]}_{chosen_pair[1]}_{deg_idx:04d}.jpg"
        dest_path = os.path.join(DEGRADED_DIR, out_name)
        cv2.imwrite(dest_path, deg_img)

        issues = {k: "none" for k in DEGRADATION_MAP.keys()}
        issues[chosen_pair[0]] = sev1
        issues[chosen_pair[1]] = sev2
        q_score = compute_quality_score(issues)
        q_label = determine_label(q_score, issues)

        record = {
            "filename": out_name,
            "filepath": dest_path,
            "base_image": os.path.basename(img_path),
            "blur_severity": issues["blur"],
            "underexposure_severity": issues["underexposure"],
            "overexposure_severity": issues["overexposure"],
            "noise_severity": issues["noise"],
            "corruption_severity": issues["corruption"],
            "defect_present": issues["defect"],
            "has_blur": int(issues["blur"] != "none"),
            "has_underexposure": int(issues["underexposure"] != "none"),
            "has_overexposure": int(issues["overexposure"] != "none"),
            "has_noise": int(issues["noise"] != "none"),
            "has_corruption": int(issues["corruption"] != "none"),
            "has_defect": int(issues["defect"] != "none"),
            "quality_score": q_score,
            "quality_label": q_label
        }
        records.append(record)
        deg_idx += 1

    df = pd.DataFrame(records)

    # 4. Perform stratified train (70%) / val (15%) / test (15%) split by base_image
    unique_bases = sorted(df["base_image"].unique())
    random.shuffle(unique_bases)
    n_total = len(unique_bases)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_bases = set(unique_bases[:n_train])
    val_bases = set(unique_bases[n_train:n_train+n_val])
    test_bases = set(unique_bases[n_train+n_val:])

    def assign_split(base_img):
        if base_img in train_bases:
            return "train"
        elif base_img in val_bases:
            return "val"
        else:
            return "test"

    df["split"] = df["base_image"].apply(assign_split)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[+] Dataset generated successfully: {len(df)} total samples.")
    print(f"    - Split distribution: Train={sum(df['split']=='train')}, Val={sum(df['split']=='val')}, Test={sum(df['split']=='test')}")
    print(f"    - Class distribution:\n{df['quality_label'].value_counts().to_string()}")

    # 5. Generate sample visual degradation grid
    print("\n[*] Generating sample condition grid for documentation...")
    sample_clean_path = raw_images[0]
    sample_img = cv2.imread(sample_clean_path)
    sample_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    grid_items = [
        ("Pristine / Clean", sample_rgb),
        ("Blur (Defocus)", cv2.cvtColor(DEGRADATION_MAP["blur"](sample_img, "severe"), cv2.COLOR_BGR2RGB)),
        ("Underexposure", cv2.cvtColor(DEGRADATION_MAP["underexposure"](sample_img, "severe"), cv2.COLOR_BGR2RGB)),
        ("Overexposure", cv2.cvtColor(DEGRADATION_MAP["overexposure"](sample_img, "severe"), cv2.COLOR_BGR2RGB)),
        ("Gaussian Noise", cv2.cvtColor(DEGRADATION_MAP["noise"](sample_img, "severe"), cv2.COLOR_BGR2RGB)),
        ("JPEG Corruption", cv2.cvtColor(DEGRADATION_MAP["corruption"](sample_img, "severe"), cv2.COLOR_BGR2RGB)),
        ("Synthetic Defect", cv2.cvtColor(DEGRADATION_MAP["defect"](sample_img, "severe"), cv2.COLOR_BGR2RGB)),
        ("Multi-Degradation", cv2.cvtColor(DEGRADATION_MAP["noise"](DEGRADATION_MAP["blur"](sample_img, "moderate"), "moderate"), cv2.COLOR_BGR2RGB))
    ]

    for ax, (title, img_data) in zip(axes, grid_items):
        ax.imshow(img_data)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    grid_out_path = os.path.join(DOCS_SAMPLES_DIR, "degradations_grid.png")
    plt.savefig(grid_out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved degradation demonstration grid to: {grid_out_path}")

    # Also save individual curated sample conditions for evaluation and README
    for title, img_data in grid_items:
        clean_title = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        cv2.imwrite(os.path.join(DOCS_SAMPLES_DIR, f"sample_{clean_title}.jpg"), cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
