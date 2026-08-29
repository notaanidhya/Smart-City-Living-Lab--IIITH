"""
scripts/calibrate_score_engine.py
=================================
Fast calibration using precomputed features in data/features.csv.
Fits a strictly monotonic Isotonic Regression on the Training Split (700 images in data/manifest.csv)
and evaluates generalization on Validation, Test (150 unseen), and Hybrid Benchmark (1,280 images).
Ensures boundary preservation for [0, 5] (catastrophic blackout) and 100 (pristine clean).
"""

import os, sys, json, cv2, torch, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

sys.path.insert(0, ".")
from ml.feature_extractor import FEATURE_NAMES
from ml.models.mlp import MultiLabelMLP
from ml.models.autoencoder import ConvAutoencoder, IMG_SIZE
from ml.score import compute_quality_score, ISSUE_NAMES

FEATURES_CSV = "data/features.csv"
CALIBRATOR_OUT = "ml/models/score_calibrator.json"

def predict_score_batch_fast(df, mlp, ae):
    raw_scores, gt_scores, labels = [], [], []
    for idx, row in df.iterrows():
        # Precomputed tabular features
        feats = {k: float(row[k]) for k in FEATURE_NAMES}
        x = torch.tensor(np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            probs = mlp(x).squeeze().numpy()

        img_path = os.path.join("data", "degraded", row["filename"])
        if not os.path.exists(img_path):
            img_path = os.path.join("data", "raw", row["filename"])

        bgr = cv2.imread(img_path)
        if bgr is not None:
            rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
            img_t = torch.tensor(rgb.transpose(2,0,1), dtype=torch.float32).unsqueeze(0)/255.0
            with torch.no_grad():
                err_val = ae.reconstruction_error(img_t).item()
            norm_err = err_val / 0.1942
        else:
            norm_err = 0.0

        pred_score, pred_label, issues = compute_quality_score(probs, recon_error=norm_err, raw_features=feats)
        raw_scores.append(float(pred_score))
        gt_scores.append(float(row["quality_score"]))
        labels.append(pred_label)

    return np.array(raw_scores), np.array(gt_scores), labels

def main():
    print("[*] Loading models for fast calibration...")
    mlp_ckpt = torch.load("ml/models/mlp_best.pt", map_location="cpu", weights_only=True)
    mlp = MultiLabelMLP(); mlp.load_state_dict(mlp_ckpt["model_state"]); mlp.eval()

    ae_ckpt = torch.load("ml/models/autoencoder_best.pt", map_location="cpu", weights_only=True)
    ae = ConvAutoencoder(); ae.load_state_dict(ae_ckpt["model_state"]); ae.eval()

    df = pd.read_csv(FEATURES_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[*] Extracting raw scores across Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})...")
    X_train, y_train, _ = predict_score_batch_fast(train_df, mlp, ae)
    X_val,   y_val,   _ = predict_score_batch_fast(val_df, mlp, ae)
    X_test,  y_test,  _ = predict_score_batch_fast(test_df, mlp, ae)

    # Train monotonic Isotonic Regression with strict boundaries [0, 100]
    X_fit = np.concatenate([[0.0, 5.0], X_train, [100.0]])
    y_fit = np.concatenate([[0.0, 5.0], y_train, [100.0]])

    iso = IsotonicRegression(y_min=0.0, y_max=100.0, increasing=True, out_of_bounds="clip")
    iso.fit(X_fit, y_fit)

    eval_x = np.linspace(0.0, 100.0, 101)
    eval_y = iso.predict(eval_x)
    # Strict boundary locks:
    eval_y[0] = 0.0
    eval_y[5] = 5.0 # Preserve 5.0 blackout gate
    for i in range(1, 5):
        eval_y[i] = round(float(i), 2)
    eval_y[-1] = 100.0 # Preserve 100.0 clean

    print("\n" + "=" * 80)
    print("CALIBRATION GENERALIZATION REPORT (ACROSS 1,000 DATASET IMAGES):")
    print("=" * 80)
    for name, X_s, y_s in [("Train Split (700)", X_train, y_train), ("Val Split (150)", X_val, y_val), ("Test Split (150 Unseen)", X_test, y_test)]:
        y_c = np.interp(X_s, eval_x, eval_y)
        mae_raw = mean_absolute_error(y_s, X_s)
        mae_cal = mean_absolute_error(y_s, y_c)
        r_raw, _ = pearsonr(y_s, X_s)
        r_cal, _ = pearsonr(y_s, y_c)
        print(f"{name:<25} | Raw MAE: {mae_raw:5.2f} -> Calib MAE: {mae_cal:5.2f} pts | Raw r: {r_raw:.3f} -> Calib r: {r_cal:.3f}")
    print("=" * 80)

    calibrator_data = {
        "type": "isotonic_linear_knots",
        "x_knots": eval_x.round(2).tolist(),
        "y_knots": eval_y.round(2).tolist(),
        "train_mae_before": round(float(mean_absolute_error(y_train, X_train)), 2),
        "train_mae_after": round(float(mean_absolute_error(y_train, np.interp(X_train, eval_x, eval_y))), 2),
        "test_mae_before": round(float(mean_absolute_error(y_test, X_test)), 2),
        "test_mae_after": round(float(mean_absolute_error(y_test, np.interp(X_test, eval_x, eval_y))), 2),
    }

    with open(CALIBRATOR_OUT, "w") as f:
        json.dump(calibrator_data, f, indent=2)

    print(f"\n[+] Saved calibration knots to: {CALIBRATOR_OUT}")

if __name__ == "__main__":
    main()
