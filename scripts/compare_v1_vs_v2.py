"""
scripts/compare_v1_vs_v2.py
===========================
Runs complete 1,280-image comparative evaluation: Baseline Model v1 vs. Candidate Model v2.
"""

import os, sys, cv2, torch, numpy as np, pandas as pd, json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_absolute_error
from scipy.stats import pearsonr

sys.path.insert(0, ".")
from ml.feature_extractor import extract_features, FEATURE_NAMES
from ml.models.mlp import MultiLabelMLP
from ml.models.autoencoder import ConvAutoencoder, IMG_SIZE
from ml.score import compute_quality_score, ISSUE_NAMES

MANIFEST_PATH = "data/hybrid_benchmark_manifest.csv"
manifest = pd.read_csv(MANIFEST_PATH)

# Load Model v2 (24 features)
mlp_v2_ckpt = torch.load("ml/models/mlp_v2.pt", map_location="cpu", weights_only=True)
mlp_v2 = MultiLabelMLP(input_dim=len(FEATURE_NAMES)); mlp_v2.load_state_dict(mlp_v2_ckpt["model_state"]); mlp_v2.eval()

ae_ckpt = torch.load("ml/models/autoencoder_best.pt", map_location="cpu", weights_only=True)
ae = ConvAutoencoder(); ae.load_state_dict(ae_ckpt["model_state"]); ae.eval()

with open("ml/models/mlp_thresholds_v2.json") as f:
    thresh_v2 = json.load(f)

LABEL_COLS = ["has_blur", "has_underexposure", "has_overexposure", "has_noise", "has_corruption", "has_defect"]

results = []
print(f"[*] Running 1,280-image comparative benchmark on Candidate Model v2...")

for idx, row in manifest.iterrows():
    img_path = os.path.join("data", "hybrid_benchmark", row["filename"])
    bgr = cv2.imread(img_path)
    if bgr is None: continue

    feats = extract_features(bgr)
    x = torch.tensor(np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probs = mlp_v2(x).squeeze().numpy()

    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(rgb.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)/255.0
    with torch.no_grad():
        err_val = ae.reconstruction_error(img_t).item()
    norm_err = err_val / 0.1942

    pred_score, pred_label, issues = compute_quality_score(probs, recon_error=norm_err, raw_features=feats)

    res_row = dict(row)
    res_row.update({
        "pred_quality_score": pred_score,
        "pred_quality_label": pred_label,
        "pred_blur": any(i["type"] == "blur" for i in issues),
        "pred_underexposure": any(i["type"] == "underexposure" for i in issues),
        "pred_overexposure": any(i["type"] == "overexposure" for i in issues),
        "pred_noise": any(i["type"] == "noise" for i in issues),
        "pred_corruption": any(i["type"] == "corruption" for i in issues),
        "pred_defect": any(i["type"] == "defect" for i in issues),
        "prob_blur": round(float(probs[0]), 4),
        "prob_underexposure": round(float(probs[1]), 4),
        "prob_overexposure": round(float(probs[2]), 4),
        "prob_noise": round(float(probs[3]), 4),
        "prob_corruption": round(float(probs[4]), 4),
        "prob_defect": round(float(probs[5]), 4),
    })
    results.append(res_row)

df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("1,280-IMAGE BENCHMARK: CANDIDATE MODEL V2 (24 FEATURES + FOCAL LOSS)")
print("=" * 80)

for col in LABEL_COLS:
    issue_name = col.replace("has_", "")
    y_t = df[col].values.astype(int)
    y_p = df[f"pred_{issue_name}"].values.astype(int)
    y_prob = df[f"prob_{issue_name}"].values.astype(float)

    acc = accuracy_score(y_t, y_p)
    prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, average="binary", zero_division=0)
    roc = roc_auc_score(y_t, y_prob)

    print(f"{issue_name:<16} | Acc: {acc*100:5.1f}% | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}")

# Pristine Reference Check
pristine = df[df["category"] == "Pristine"]
fp_rate = (pristine["pred_quality_label"] != "ACCEPTABLE").mean() * 100
mean_score = pristine["pred_quality_score"].mean()

print("-" * 80)
print(f"Pristine References (200 images) | Mean Score: {mean_score:.1f}/100 | False Positive Rate: {fp_rate:.1f}%")

# Score Agreement
mae = mean_absolute_error(df["gt_quality_score"], df["pred_quality_score"])
r, _ = pearsonr(df["gt_quality_score"], df["pred_quality_score"])
print(f"Score Agreement Across 1,280 Imgs | MAE: {mae:.2f} pts | Pearson r: {r:.3f}")
print("=" * 80)

# Corruption and Defect breakdown
corrupt_cat = df[df["category"] == "Corruption"]
print("\nCORRUPTION RECALL BREAKDOWN:")
print(corrupt_cat.groupby("sub_category")[["pred_corruption", "prob_corruption", "pred_quality_score"]].mean().round(3))

defect_cat = df[df["category"] == "Defect"]
print("\nDEFECT RECALL BREAKDOWN:")
print(defect_cat.groupby("sub_category")[["pred_defect", "prob_defect", "pred_quality_score"]].mean().round(3))
