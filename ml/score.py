"""
ml/score.py
===========
Quality Score Derivation Formula & Label Thresholds.

This module defines the authoritative formula for computing a composite
image quality score (0-100) from Model A predictions and the Model B
reconstruction error.
"""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS_PATH = os.path.join(BASE_DIR, "ml", "models", "mlp_thresholds.json")

ISSUE_NAMES = ["blur", "underexposure", "overexposure", "noise", "corruption", "defect"]

DEFAULT_THRESHOLDS = {
    "has_blur": 0.35,
    "has_underexposure": 0.60,
    "has_overexposure": 0.50,
    "has_noise": 0.40,
    "has_corruption": 0.50,
    "has_defect": 0.50,
}

ISSUE_PENALTIES = {
    "blur":          {"none": 0, "low":  8, "medium": 18, "high": 30},
    "underexposure": {"none": 0, "low":  8, "medium": 18, "high": 30},
    "overexposure":  {"none": 0, "low":  8, "medium": 18, "high": 30},
    "noise":         {"none": 0, "low":  8, "medium": 16, "high": 28},
    "corruption":    {"none": 0, "low":  8, "medium": 16, "high": 28},
    "defect":        {"none": 0, "low": 12, "medium": 24, "high": 42},
}

LABEL_THRESHOLDS = {
    "ACCEPTABLE": 75.0,
    "DEGRADED":   40.0,
}

MLP_WEIGHT = 0.70
AE_WEIGHT  = 0.30
AE_PENALTY_SCALE = 30.0
AE_PENALTY_MAX   = 35.0


def load_calibrated_thresholds() -> dict:
    """Load per-class thresholds calibrated on the validation set."""
    if os.path.exists(THRESHOLDS_PATH):
        try:
            with open(THRESHOLDS_PATH, "r") as f:
                data = json.load(f)
                res = {}
                for k, v in data.items():
                    thresh = v.get("threshold", DEFAULT_THRESHOLDS.get(k, 0.5))
                    if k == "has_defect":
                        thresh = 0.50
                    elif k == "has_blur":
                        thresh = max(0.35, thresh)
                    res[k] = thresh
                return res
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()


def determine_issue_severity(prob: float, threshold: float) -> str:
    """Categorize severity based on calibrated decision threshold."""
    if prob < threshold:
        return "none"
    margin = 1.0 - threshold
    if prob >= threshold + 0.40 * margin:
        return "high"
    elif prob >= threshold + 0.15 * margin:
        return "medium"
    else:
        return "low"


def compute_quality_score(
    mlp_probs: list | np.ndarray,
    recon_error: float = 0.0,
) -> tuple[float, str, list[dict]]:
    """
    Compute composite quality score, label, and per-issue detail list.

    Parameters
    ----------
    mlp_probs : array-like, shape (6,)
        Sigmoid probabilities for [blur, underexposure, overexposure,
        noise, corruption, defect] from Model A.
    recon_error : float
        Normalised reconstruction error from Model B's autoencoder
        (divided by calibration scale, so ~1.0 = anomaly threshold).

    Returns
    -------
    quality_score : float in [0, 100]
    quality_label : str (ACCEPTABLE, DEGRADED, DEFECTIVE)
    issues        : list of dicts
    """
    probs = np.asarray(mlp_probs, dtype=np.float32)
    assert probs.shape == (6,), f"Expected 6 MLP probabilities, got {probs.shape}"

    calib_thresholds = load_calibrated_thresholds()

    mlp_penalty = 0.0
    issues = []
    
    for i, issue in enumerate(ISSUE_NAMES):
        prob = float(probs[i])
        thresh_key = f"has_{issue}"
        thresh = calib_thresholds.get(thresh_key, 0.5)

        # Defect confidence boosted if spatial anomaly is elevated
        if issue == "defect" and recon_error > 1.2:
            prob = max(prob, min(0.95, 0.55 + (recon_error - 1.0) * 0.3))

        sev = determine_issue_severity(prob, thresh)
        penalty = ISSUE_PENALTIES[issue][sev]
        mlp_penalty += penalty

        if sev != "none":
            issues.append({
                "type":       issue,
                "severity":   sev,
                "confidence": round(prob, 4),
                "details":    f"Confidence {prob:.2%}; severity={sev}; penalty={penalty}",
            })

    # Anomaly penalty based on autoencoder reconstruction residual
    ae_excess = max(0.0, recon_error - 0.9)
    ae_penalty = float(np.clip(ae_excess * AE_PENALTY_SCALE, 0.0, AE_PENALTY_MAX))

    total_penalty = MLP_WEIGHT * mlp_penalty + AE_WEIGHT * ae_penalty
    quality_score = float(np.clip(100.0 - total_penalty, 0.0, 100.0))

    # Determine Label
    defect_issue = next((iss for iss in issues if iss["type"] == "defect"), None)
    has_severe_defect = (defect_issue is not None and defect_issue["severity"] == "high") or recon_error > 1.8

    if quality_score < LABEL_THRESHOLDS["DEGRADED"] or has_severe_defect:
        quality_label = "DEFECTIVE"
    elif quality_score < LABEL_THRESHOLDS["ACCEPTABLE"]:
        quality_label = "DEGRADED"
    else:
        quality_label = "ACCEPTABLE"

    return round(quality_score, 1), quality_label, issues
