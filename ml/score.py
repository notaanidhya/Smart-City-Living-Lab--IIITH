"""
ml/score.py
===========
Quality Score Derivation Formula & Label Thresholds.

This module defines the authoritative formula for computing a composite
image quality score (0-100) from Model A predictions and the Model B
reconstruction error. Kept separate so both the training pipeline and
the backend inference service use *exactly* the same business logic.

Formula (documented for Section 7 of the assessment):
------------------------------------------------------
    issue_penalties = {
        "blur":           {none:0, low:8,  medium:18, high:32},
        "underexposure":  {none:0, low:8,  medium:18, high:32},
        "overexposure":   {none:0, low:8,  medium:18, high:32},
        "noise":          {none:0, low:6,  medium:14, high:24},
        "corruption":     {none:0, low:6,  medium:14, high:24},
    }
    anomaly_penalty = clip(reconstruction_error * 60, 0, 45)

    mlp_penalty   = sum(issue_penalties[issue][severity] for detected issues)
    total_penalty = 0.65 * mlp_penalty + 0.35 * anomaly_penalty
    quality_score = clip(100 - total_penalty, 0, 100)

Weighting rationale:
  - 65% weight to MLP: directly interpretable, maps to labelled ground truth.
  - 35% weight to autoencoder: captures spatial/structural anomalies that
    hand-crafted features miss. Deliberately lower weight because the
    autoencoder is unsupervised — its reconstruction error is a proxy
    signal, not a ground-truth label.

Quality Labels:
    ACCEPTABLE:  score >= 72
    DEGRADED:    40 <= score < 72
    DEFECTIVE:   score < 40  OR  any defect-class confidence >= 0.55
"""

import numpy as np

# MLP output index → issue name
ISSUE_NAMES = ["blur", "underexposure", "overexposure", "noise", "corruption", "defect"]

# Severity thresholds on raw MLP sigmoid probability
SEVERITY_THRESHOLDS = {
    "high":   0.75,
    "medium": 0.50,
    "low":    0.30,
}

# Penalty per issue per severity level (tuned against synthetic quality_score ground truth)
ISSUE_PENALTIES = {
    "blur":          {"none": 0, "low":  8, "medium": 18, "high": 32},
    "underexposure": {"none": 0, "low":  8, "medium": 18, "high": 32},
    "overexposure":  {"none": 0, "low":  8, "medium": 18, "high": 32},
    "noise":         {"none": 0, "low":  6, "medium": 14, "high": 24},
    "corruption":    {"none": 0, "low":  6, "medium": 14, "high": 24},
    "defect":        {"none": 0, "low": 12, "medium": 28, "high": 45},
}

# Label thresholds
LABEL_THRESHOLDS = {
    "ACCEPTABLE": 72.0,
    "DEGRADED":   40.0,
    # DEFECTIVE: below 40 OR defect confidence >= 0.55
}

DEFECT_LABEL_CONFIDENCE_THRESHOLD = 0.55
MLP_WEIGHT = 0.65
AE_WEIGHT  = 0.35
AE_PENALTY_SCALE = 60.0   # maps normalised AE recon error [0,1] → penalty points
AE_PENALTY_MAX   = 45.0   # cap to prevent a single large recon error dominating


def confidence_to_severity(confidence: float) -> str:
    """Map a sigmoid probability to a categorical severity label."""
    if confidence >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    elif confidence >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    elif confidence >= SEVERITY_THRESHOLDS["low"]:
        return "low"
    return "none"


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
        (mean pixel-level MSE across the 128x128 image, divided by a
        calibration constant so values typically fall in [0, 1]).

    Returns
    -------
    quality_score : float  in [0, 100]
    quality_label : str    one of ACCEPTABLE / DEGRADED / DEFECTIVE
    issues        : list of dicts with keys: type, severity, confidence, details
    """
    probs = np.asarray(mlp_probs, dtype=np.float32)
    assert probs.shape == (6,), f"Expected 6 MLP probabilities, got {probs.shape}"

    # 1. Compute MLP penalty
    mlp_penalty = 0.0
    issues = []
    for i, (issue, prob) in enumerate(zip(ISSUE_NAMES, probs)):
        sev = confidence_to_severity(float(prob))
        penalty = ISSUE_PENALTIES[issue][sev]
        mlp_penalty += penalty
        if sev != "none":
            issues.append({
                "type":       issue,
                "severity":   sev,
                "confidence": round(float(prob), 4),
                "details":    f"Confidence {prob:.2%}; severity={sev}; penalty_contribution={penalty}",
            })

    # 2. Compute AE penalty (bounded)
    ae_penalty = float(np.clip(recon_error * AE_PENALTY_SCALE, 0.0, AE_PENALTY_MAX))

    # 3. Fuse
    total_penalty = MLP_WEIGHT * mlp_penalty + AE_WEIGHT * ae_penalty
    quality_score = float(np.clip(100.0 - total_penalty, 0.0, 100.0))

    # 4. Determine label
    defect_prob = float(probs[ISSUE_NAMES.index("defect")])
    if quality_score < LABEL_THRESHOLDS["DEGRADED"] or defect_prob >= DEFECT_LABEL_CONFIDENCE_THRESHOLD:
        quality_label = "DEFECTIVE"
    elif quality_score < LABEL_THRESHOLDS["ACCEPTABLE"]:
        quality_label = "DEGRADED"
    else:
        quality_label = "ACCEPTABLE"

    return round(quality_score, 2), quality_label, issues
