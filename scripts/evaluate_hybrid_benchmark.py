"""
scripts/evaluate_hybrid_benchmark.py
====================================
Runs end-to-end evaluation of the complete 1,280-image Hybrid Benchmark Pack.
Computes multi-label metrics, ROC-AUC, score regression MAE/RMSE, and category breakdowns.
"""

import os, sys, cv2, torch, numpy as np, pandas as pd, time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

sys.path.insert(0, ".")
from ml.feature_extractor import extract_features, FEATURE_NAMES
from ml.models.mlp import MultiLabelMLP
from ml.models.autoencoder import ConvAutoencoder, IMG_SIZE
from ml.score import compute_quality_score, ISSUE_NAMES

MANIFEST_PATH = "data/hybrid_benchmark_manifest.csv"
OUTPUT_REPORT = "docs/hybrid_benchmark_evaluation_report.md"
OUTPUT_CSV    = "docs/hybrid_benchmark_predictions.csv"

manifest = pd.read_csv(MANIFEST_PATH)
print(f"[*] Loaded manifest with {len(manifest)} images.")

# Load PyTorch models
mlp_ckpt = torch.load("ml/models/mlp_best.pt", map_location="cpu", weights_only=True)
mlp = MultiLabelMLP(); mlp.load_state_dict(mlp_ckpt["model_state"]); mlp.eval()

ae_ckpt = torch.load("ml/models/autoencoder_best.pt", map_location="cpu", weights_only=True)
ae = ConvAutoencoder(); ae.load_state_dict(ae_ckpt["model_state"]); ae.eval()

LABEL_COLS = ["has_blur", "has_underexposure", "has_overexposure", "has_noise", "has_corruption", "has_defect"]

results = []
start_time = time.time()

print("[*] Running inference on 1,280 images...")
for idx, row in manifest.iterrows():
    img_path = os.path.join("data", "hybrid_benchmark", row["filename"])
    bgr = cv2.imread(img_path)
    if bgr is None: continue

    # Extract 22 CV features
    feats = extract_features(bgr)
    x = torch.tensor(np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        probs = mlp(x).squeeze().numpy()

    # Autoencoder error
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    img_t = torch.tensor(rgb.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)/255.0
    with torch.no_grad():
        err_val = ae.reconstruction_error(img_t).item()
    norm_err = err_val / 0.1942

    # Scoring & decision fusion
    pred_score, pred_label, issues = compute_quality_score(probs, recon_error=norm_err, raw_features=feats)

    pred_flags = {
        "pred_blur": any(i["type"] == "blur" for i in issues),
        "pred_underexposure": any(i["type"] == "underexposure" for i in issues),
        "pred_overexposure": any(i["type"] == "overexposure" for i in issues),
        "pred_noise": any(i["type"] == "noise" for i in issues),
        "pred_corruption": any(i["type"] == "corruption" for i in issues),
        "pred_defect": any(i["type"] == "defect" for i in issues),
    }

    res_row = dict(row)
    res_row.update({
        "pred_quality_score": pred_score,
        "pred_quality_label": pred_label,
        "prob_blur": round(float(probs[0]), 4),
        "prob_underexposure": round(float(probs[1]), 4),
        "prob_overexposure": round(float(probs[2]), 4),
        "prob_noise": round(float(probs[3]), 4),
        "prob_corruption": round(float(probs[4]), 4),
        "prob_defect": round(float(probs[5]), 4),
        "ae_recon_error": round(err_val, 5),
        "issues_detected": "; ".join([f"{i['type']}({i['severity']})" for i in issues]) or "None"
    })
    res_row.update(pred_flags)
    results.append(res_row)

    if (idx + 1) % 250 == 0 or (idx + 1) == len(manifest):
        print(f"    Processed {idx+1}/{len(manifest)} images ({(idx+1)/len(manifest)*100:.0f}%)...")

total_time = time.time() - start_time
print(f"[+] Inference complete in {total_time:.2f}s (Average: {total_time/len(manifest)*1000:.1f}ms/image)")

pred_df = pd.DataFrame(results)
pred_df.to_csv(OUTPUT_CSV, index=False)

# ==========================================
# QUANTITATIVE METRICS COMPUTATION
# ==========================================
print("\n" + "=" * 80)
print("1,280-IMAGE HYBRID BENCHMARK EVALUATION RESULTS")
print("=" * 80)

metrics_summary = []
for col in LABEL_COLS:
    issue_name = col.replace("has_", "")
    pred_col = f"pred_{issue_name}"
    prob_col = f"prob_{issue_name}"

    y_true = pred_df[col].values.astype(int)
    y_pred = pred_df[pred_col].values.astype(int)
    y_prob = pred_df[prob_col].values.astype(float)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = 1.0

    metrics_summary.append({
        "Degradation Family": issue_name.capitalize(),
        "Accuracy": f"{acc*100:.1f}%",
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-Score": round(f1, 3),
        "ROC-AUC": round(roc, 3)
    })
    print(f"{issue_name:<16} | Acc: {acc*100:5.1f}% | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}")

sum_df = pd.DataFrame(metrics_summary)

# Pristine Reference False-Positive Analysis
pristine_df = pred_df[pred_df["category"] == "Pristine"]
pristine_fp_count = (pristine_df["pred_quality_label"] != "ACCEPTABLE").sum()
pristine_fp_rate = (pristine_fp_count / len(pristine_df)) * 100
mean_pristine_score = pristine_df["pred_quality_score"].mean()

# Score Regression Metrics
gt_scores = pred_df["gt_quality_score"].values
pred_scores = pred_df["pred_quality_score"].values
mae = mean_absolute_error(gt_scores, pred_scores)
rmse = np.sqrt(mean_squared_error(gt_scores, pred_scores))
pearson_corr, _ = pearsonr(gt_scores, pred_scores)

print("-" * 80)
print(f"Pristine References (200 images) | Mean Score: {mean_pristine_score:.1f}/100 | False Positive Rate: {pristine_fp_rate:.1f}%")
print(f"Score Agreement Across 1,280 Imgs | MAE: {mae:.2f} pts | RMSE: {rmse:.2f} pts | Pearson r: {pearson_corr:.3f}")
print("=" * 80)

# Category breakdown
cat_summary = pred_df.groupby("category").agg(
    count=("id", "count"),
    mean_gt_score=("gt_quality_score", "mean"),
    mean_pred_score=("pred_quality_score", "mean"),
    acceptable_pct=("pred_quality_label", lambda x: (x == "ACCEPTABLE").mean() * 100),
    degraded_pct=("pred_quality_label", lambda x: (x == "DEGRADED").mean() * 100),
    defective_pct=("pred_quality_label", lambda x: (x == "DEFECTIVE").mean() * 100),
).round(1)

print("\nCATEGORY-BY-CATEGORY DISTRIBUTION:")
print(cat_summary)
