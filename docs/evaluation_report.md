# Model Evaluation & Experimental Rigor Report
**Assessment Deliverable - Section 9 & 10 Evaluation & Explainability**

---

## 1. Executive Summary

This report documents the quantitative and qualitative evaluation of the **Hybrid AI Image Quality & Defect Detection System** across:
1. **Primary Unseen Test Split**: 150 unseen test images across 30 independent physical scenes.
2. **Comprehensive Extended Benchmark**: 1,280 benchmark images across 7 degradation categories (Noise, Blur, Underexposure, Overexposure, Compression Corruption, Physical Defects, and Pristine References).

### 🎯 Key Performance Highlights
* **Multi-Label Issue Detection**: Macro-average ROC-AUC of **0.948 (94.8%)** and Accuracy of **91.8%**.
* **Differentiated Penalties + Continuous PCHIP Monotonic Calibration**: Eliminated the flat-step plateau effect. Scores are smoothly and logically distributed across degradation severities (Noise $77.0$, Glitch $70.0$, Blur $68.6$, Defect $66.4$, Underexposure $63.4$, Overexposure $62.6$, Multi $52.9$, Clean $100.0$).
* **Autoencoder 256 Resolution Upgrade**: Increased defect classification accuracy from $68.0\% \rightarrow \mathbf{75.3\%}$ with a $38.8\%$ reduction in false alarms.
* **Pristine Reference Quality**: Real clean uncompressed photos achieve an average score of $\mathbf{87.2 / 100}$ (with $100.0$ on nominal benchmarks).
* **Zero Data Leakage**: All test variants originate from pristine base images strictly excluded from the training split.
* **Explainability**: High-resolution multi-scale quadrant-stitched anomaly heatmaps provide localized visualization of physical occlusions, cracks, and sensor defects.

---

## 2. Quantitative Classification Results (Unseen Test Split)

| Degradation Family | ROC-AUC | F1-Score | Accuracy | Precision | Recall | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| **Sensor Noise** | **`1.000`** | **`1.000`** | **`100.0%`** | `1.000` | `1.000` | 36 | 0 | 0 | 114 |
| **Underexposure** | **`1.000`** | **`0.952`** | **`98.7%`** | `1.000` | `0.909` | 20 | 0 | 2 | 128 |
| **Overexposure** | **`0.999`** | **`0.958`** | **`98.7%`** | `0.920` | `1.000` | 23 | 2 | 0 | 125 |
| **Defocus Blur** | **`0.949`** | **`0.596`** | **`87.3%`** | `0.483` | `0.778` | 14 | 15 | 4 | 117 |
| **JPEG Corruption**| **`0.937`** | **`0.667`** | **`90.7%`** | `0.667` | `0.667` | 14 | 7 | 7 | 122 |
| **Physical Defect**| **`0.802`** | **`0.448`** | **`75.3%`** | `0.405` | `0.500` | 15 | 22 | 15 | 98 |
| **Macro Average**  | **`0.948` (94.8%)** | **`0.770`** | **`91.8%`** | **`0.746`** | **`0.809`** | — | — | — | — |

---

## 3. Presets Score Distribution Matrix

| Benchmark Preset | Score | Label | Diagnosed Issues | Diagnostic Meaning |
|---|---|---|---|---|
| **Pristine Clean** | **`100.0`** | **`ACCEPTABLE`** | None (Clean) | Nominal baseline photo |
| **Gaussian Noise** | **`77.0`** | **`DEGRADED`** | `noise(high)` | Visible sensor noise; high readability |
| **JPEG Glitch** | **`70.0`** | **`DEGRADED`** | `corruption(high)` | DCT boundary blockiness |
| **Defocus Blur** | **`68.6`** | **`DEGRADED`** | `blur(high)` | Optical defocus blur |
| **Physical Defect** | **`66.4`** | **`DEGRADED`** | `defect(medium)` | Localized physical scratch |
| **Underexposure** | **`63.4`** | **`DEGRADED`** | `underexposure(high)` | Severe shadow clipping |
| **Overexposure** | **`62.6`** | **`DEGRADED`** | `overexposure(high)` | Severe highlight blowout |
| **Multi-Degraded** | **`52.9`** | **`DEGRADED`** | `blur(high), noise(high)` | Dual-issue compound penalty |

---

## 4. Verification & Automated Test Suite

* **10 / 10 Automated Unit & API Tests Passing (100%)**
* **Inference Latency**: $\approx 35\text{--}40\text{ms}$ per image on CPU at $256\times256$ resolution.
