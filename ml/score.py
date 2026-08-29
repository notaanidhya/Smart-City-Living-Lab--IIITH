"""
ml/score.py
===========
Quality Score Derivation Formula & Decision Fusion Engine.
"""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "ml", "models", "mlp_thresholds.json")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "ml", "models", "score_calibrator.json")

ISSUE_NAMES = ["blur", "underexposure", "overexposure", "noise", "corruption", "defect"]

DEFAULT_THRESHOLDS = {
    "has_blur": 0.35,
    "has_underexposure": 0.60,
    "has_overexposure": 0.50,
    "has_noise": 0.40,
    "has_corruption": 0.50,
    "has_defect": 0.60,
}

SEVERITY_BANDS = {
    "blur":          {"medium_offset": 0.20, "high_offset": 0.50},
    "underexposure": {"medium_offset": 0.15, "high_offset": 0.40},
    "overexposure":  {"medium_offset": 0.15, "high_offset": 0.40},
    "noise":         {"medium_offset": 0.25, "high_offset": 0.55},
    "corruption":    {"medium_offset": 0.20, "high_offset": 0.45},
    "defect":        {"medium_offset": 0.08, "high_offset": 0.30},
}


ISSUE_PENALTIES = {
    "blur":          {"none": 0, "low": 8,  "medium": 22, "high": 42},
    "underexposure": {"none": 0, "low": 10, "medium": 26, "high": 48},
    "overexposure":  {"none": 0, "low": 10, "medium": 26, "high": 48},
    "noise":         {"none": 0, "low": 5,  "medium": 14, "high": 30},
    "corruption":    {"none": 0, "low": 8,  "medium": 22, "high": 40},
    "defect":        {"none": 0, "low": 25, "medium": 45, "high": 65},
}

LABEL_THRESHOLDS = {
    "ACCEPTABLE": 75.0,
    "DEGRADED":   40.0,
}

MLP_WEIGHT = 0.70
AE_WEIGHT  = 0.30
AE_PENALTY_SCALE = 30.0
AE_PENALTY_MAX   = 35.0

_CALIBRATOR_CACHE = None

def load_score_calibrator() -> tuple[np.ndarray, np.ndarray] | None:

    global _CALIBRATOR_CACHE
    if _CALIBRATOR_CACHE is not None:
        return _CALIBRATOR_CACHE
    if os.path.exists(CALIBRATOR_PATH):
        try:
            with open(CALIBRATOR_PATH, "r") as f:
                data = json.load(f)
                x_k = np.array(data["x_knots"], dtype=np.float64)
                y_k = np.array(data["y_knots"], dtype=np.float64)
                _CALIBRATOR_CACHE = (x_k, y_k)
                return _CALIBRATOR_CACHE
        except Exception:
            pass
    return None

def calibrate_quality_score(raw_score: float) -> float:

    if raw_score <= 5.0:
        return round(raw_score, 1)
    if raw_score >= 99.5:
        return 100.0
    calib = load_score_calibrator()
    if calib is not None:
        x_k, y_k = calib
        calibrated = float(np.interp(raw_score, x_k, y_k))
        return round(float(np.clip(calibrated, 0.0, 100.0)), 1)
    return round(raw_score, 1)


def load_calibrated_thresholds() -> dict:

    if os.path.exists(THRESHOLDS_PATH):
        try:
            with open(THRESHOLDS_PATH, "r") as f:
                data = json.load(f)
                res = {}
                for k, v in data.items():
                    thresh = v.get("threshold", DEFAULT_THRESHOLDS.get(k, 0.5))
                    if k == "has_defect":
                        thresh = 0.55
                    elif k == "has_blur":
                        thresh = max(0.35, thresh)
                    res[k] = thresh
                return res
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()


def compute_defect_threshold(norm_error: float) -> float:
    """
    Continuous Power-Exponential Gating for physical defect detection.
    Smoothly decays from 0.70 (pristine baseline) down to 0.38 (high-anomaly),
    eliminating discrete step cliffs while strictly protecting textured clean images.
    """
    ne = max(0.0, float(norm_error))
    return float(0.38 + 0.32 * np.exp(-3.5 * (ne ** 1.5)))


def determine_issue_severity(issue: str, prob: float, threshold: float) -> str:
    if prob < threshold:
        return "none"
    margin = 1.0 - threshold
    bands = SEVERITY_BANDS.get(issue, {"medium_offset": 0.15, "high_offset": 0.40})
    if prob >= threshold + bands["high_offset"] * margin:
        return "high"
    elif prob >= threshold + bands["medium_offset"] * margin:
        return "medium"
    else:
        return "low"


def compute_quality_score(
    mlp_probs: list | np.ndarray,
    recon_error: float = 0.0,
    raw_features: dict = None,
) -> tuple[float, str, list[dict]]:
    """
    Compute composite quality score, label, and per-issue detail list.
    Includes catastrophic blackout/blowout gate and multi-issue compounding cap.
    """
    # 1. Catastrophic Information Loss Gate (Total Blackout / Blowout)
    if raw_features is not None:
        mean_lum = float(raw_features.get("mean_luminance", 128.0))
        dark_ratio = float(raw_features.get("dark_pixel_ratio", 0.0))
        bright_ratio = float(raw_features.get("bright_pixel_ratio", 0.0))
        if mean_lum < 3.0 and dark_ratio > 0.98:
            return 5.0, "DEFECTIVE", [{
                "type": "underexposure",
                "severity": "high",
                "confidence": 1.0,
                "details": "Catastrophic sensor blackout / total information loss (100% black frame)",
            }]
        if mean_lum > 252.0 and bright_ratio > 0.98:
            return 5.0, "DEFECTIVE", [{
                "type": "overexposure",
                "severity": "high",
                "confidence": 1.0,
                "details": "Catastrophic sensor blowout / total information loss (100% white frame)",
            }]

    probs = np.asarray(mlp_probs, dtype=np.float32)
    assert probs.shape == (6,), f"Expected 6 MLP probabilities, got {probs.shape}"

    calib_thresholds = load_calibrated_thresholds()

    raw_issues = {}
    for i, issue in enumerate(ISSUE_NAMES):
        prob = float(probs[i])
        thresh_key = f"has_{issue}"
        thresh = calib_thresholds.get(thresh_key, 0.5)

        # Dynamic AE Gating for Defect
        if issue == "defect":
            thresh = compute_defect_threshold(recon_error)
            if recon_error > 1.2:
                prob = max(prob, min(0.95, 0.60 + (recon_error - 1.0) * 0.30))

        sev = determine_issue_severity(issue, prob, thresh)
        raw_issues[issue] = {"severity": sev, "confidence": prob, "threshold": thresh}

    # Blur-Noise Cross-Class Suppression
    # If severe noise is present and blur probability is borderline (<0.55), suppress noise-induced blur artifact
    if raw_issues["noise"]["severity"] == "high":
        if raw_issues["blur"]["severity"] == "low" or (raw_issues["blur"]["severity"] == "medium" and raw_issues["blur"]["confidence"] < 0.55):
            raw_issues["blur"]["severity"] = "none"

    penalty_list = []
    issues = []
    has_high_issue = False

    for issue, data in raw_issues.items():
        sev = data["severity"]
        prob = data["confidence"]
        if sev != "none":
            penalty = ISSUE_PENALTIES[issue][sev]
            penalty_list.append(penalty)
            if sev == "high":
                has_high_issue = True
            issues.append({
                "type":       issue,
                "severity":   sev,
                "confidence": round(prob, 4),
                "details":    f"Confidence {prob:.2%}; severity={sev}; penalty={penalty}",
            })

    # Multi-Issue Compounding Cap (Diminishing returns on secondary & tertiary issues)
    if not penalty_list:
        mlp_penalty = 0.0
    elif len(penalty_list) == 1:
        mlp_penalty = float(penalty_list[0])
    else:
        sorted_penalties = sorted(penalty_list, reverse=True)
        mlp_penalty = float(sorted_penalties[0])
        if len(sorted_penalties) > 1:
            mlp_penalty += float(sorted_penalties[1]) * 0.70  # 70% weight on secondary issue
        for extra in sorted_penalties[2:]:
            mlp_penalty += float(extra) * 0.50                # 50% weight on tertiary+ issues

    # Anomaly penalty based on autoencoder reconstruction residual
    ae_excess = max(0.0, recon_error - 0.9)
    ae_penalty = float(np.clip(ae_excess * AE_PENALTY_SCALE, 0.0, AE_PENALTY_MAX))

    total_penalty = MLP_WEIGHT * mlp_penalty + AE_WEIGHT * ae_penalty
    raw_score = float(np.clip(100.0 - total_penalty, 0.0, 100.0))
    quality_score = calibrate_quality_score(raw_score)

    # Confidence-Weighted Label Assignment & Consistency
    defect_issue = next((iss for iss in issues if iss["type"] == "defect"), None)
    has_severe_defect = (
        defect_issue is not None 
        and defect_issue["severity"] == "high" 
        and defect_issue["confidence"] >= 0.70
    ) or recon_error > 1.8

    if quality_score < LABEL_THRESHOLDS["DEGRADED"] or has_severe_defect:
        quality_label = "DEFECTIVE"
    elif quality_score < LABEL_THRESHOLDS["ACCEPTABLE"] or defect_issue is not None or has_high_issue:
        quality_label = "DEGRADED"
    else:
        quality_label = "ACCEPTABLE"

    return round(quality_score, 1), quality_label, issues
