"""
ml/feature_extractor.py
=======================
Deterministic Computer Vision feature extraction pipeline.
Extracts 22 statistical & frequency-domain metrics from an input image.

Feature Families:
  1. Sharpness   (4 features)  - Laplacian variance, Tenengrad, FFT energy ratio, edge density
  2. Exposure    (4 features)  - Mean luminance, dark pixel ratio, bright pixel ratio, histogram skewness
  3. Contrast    (2 features)  - RMS contrast, Michelson contrast
  4. Noise       (3 features)  - Immerkær Laplacian noise sigma, flat-region local variance, signal-to-noise proxy
  5. Color       (3 features)  - Mean HSV saturation, per-channel RGB imbalance, colorfulness index
  6. Texture     (3 features)  - GLCM contrast, GLCM homogeneity, GLCM energy
  7. Corruption  (3 features)  - DCT blockiness score, high-frequency energy loss, compression gradient metric
"""

import cv2
import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

FEATURE_NAMES = [
    # Sharpness
    "laplacian_variance",
    "tenengrad_mean",
    "fft_high_freq_ratio",
    "edge_density",
    # Exposure
    "mean_luminance",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "histogram_skewness",
    # Contrast
    "rms_contrast",
    "michelson_contrast",
    # Noise
    "noise_sigma_immerkaar",
    "flat_region_variance",
    "snr_proxy",
    # Color
    "mean_saturation",
    "channel_imbalance",
    "colorfulness",
    # Texture (GLCM)
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    # Corruption / Compression
    "dct_blockiness",
    "hf_energy_loss",
    "compression_gradient",
]


def _sharpness_features(gray: np.ndarray) -> dict:
    """
    Sharpness Features
    ------------------
    laplacian_variance : Variance of the Laplacian-filtered image.
        High for sharp images (strong edges), low for blurry ones.

    tenengrad_mean : Mean Sobel gradient magnitude (Tenengrad focus measure).
        Captures high-frequency edge presence; drops under defocus.

    fft_high_freq_ratio : Ratio of power in high-frequency FFT bins vs. total power.
        Blurry images attenuate high spatial frequencies, lowering this ratio.

    edge_density : Fraction of pixels classified as edges by the Canny detector.
        Provides an additional sharp-edge density measure.
    """
    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # Tenengrad (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    tenengrad = float(grad_mag.mean())

    # FFT high-frequency energy ratio
    h, w = gray.shape
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2

    # Define "high frequency" as outside centre 10% of spectrum
    cy, cx = h // 2, w // 2
    r_low = min(h, w) // 10
    y_coords, x_coords = np.ogrid[:h, :w]
    low_mask = (y_coords - cy) ** 2 + (x_coords - cx) ** 2 <= r_low ** 2
    total_power = mag.sum() + 1e-8
    hf_ratio = float((mag * ~low_mask).sum() / total_power)

    # Edge density via Canny
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(edges.sum() / (255.0 * gray.size))

    return {
        "laplacian_variance": lap_var,
        "tenengrad_mean": tenengrad,
        "fft_high_freq_ratio": hf_ratio,
        "edge_density": edge_density,
    }


def _exposure_features(gray: np.ndarray) -> dict:
    """
    Exposure Features
    -----------------
    mean_luminance : Average pixel intensity (0-255).
        Low → underexposed; High → overexposed.

    dark_pixel_ratio : Fraction of pixels with intensity < 10.
        High value indicates underexposure / crushed blacks.

    bright_pixel_ratio : Fraction of pixels with intensity > 245.
        High value indicates overexposure / clipped highlights.

    histogram_skewness : Skewness of the luminance histogram.
        Strong negative skew → overexposure; Positive skew → underexposure.
        Returns 0.0 if all pixels are identical (zero-variance edge case from
        severe under/overexposure).
    """
    g = gray.astype(np.float32).ravel()
    mean_lum = float(g.mean())
    dark_ratio = float((g < 10).sum() / g.size)
    bright_ratio = float((g > 245).sum() / g.size)
    # Guard against zero-variance arrays (all-black/white from extreme exposure)
    if g.std() < 1e-6:
        hist_skew = 0.0
    else:
        hist_skew = float(skew(g))

    return {
        "mean_luminance": mean_lum,
        "dark_pixel_ratio": dark_ratio,
        "bright_pixel_ratio": bright_ratio,
        "histogram_skewness": hist_skew,
    }


def _contrast_features(gray: np.ndarray) -> dict:
    """
    Contrast Features
    -----------------
    rms_contrast : Standard deviation of intensity / mean intensity.
        A commonly used perceptual contrast measure.

    michelson_contrast : (I_max - I_min) / (I_max + I_min).
        Captures the full dynamic range utilisation of the image.
    """
    g = gray.astype(np.float32)
    mean = g.mean() + 1e-8
    rms = float(g.std() / mean)

    i_max = float(g.max())
    i_min = float(g.min())
    michelson = float((i_max - i_min) / (i_max + i_min + 1e-8))

    return {
        "rms_contrast": rms,
        "michelson_contrast": michelson,
    }


def _noise_features(gray: np.ndarray) -> dict:
    """
    Noise Features
    --------------
    noise_sigma_immerkaar : Immerkær (2002) Laplacian-based noise sigma estimator.
        Estimates the standard deviation of additive Gaussian noise directly
        from the Laplacian filtered image. Fast and calibration-free.

    flat_region_variance : Mean local variance in low-texture (flat) regions.
        In a clean image, flat regions have near-zero variance. Noisy images
        exhibit elevated local variance even in homogeneous areas.

    snr_proxy : Signal-to-noise ratio proxy = mean_luminance / noise_sigma.
        Higher is cleaner; drops sharply with increasing sensor noise.
    """
    # Immerkær noise estimator
    h, w = gray.shape
    kernel = np.array([[1, -2, 1],
                       [-2,  4, -2],
                       [1, -2, 1]], dtype=np.float64)
    filtered = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = float(np.sqrt(np.pi / 2.0) * np.abs(filtered).mean() / (6 * (h - 2) * (w - 2)) * h * w)
    sigma = max(sigma, 1e-4)

    # Flat-region local variance
    local_var = _local_variance(gray, block_size=8)
    # Keep the bottom 20% of variance blocks (most "flat" regions)
    threshold = np.percentile(local_var, 20)
    flat_mask = local_var <= threshold
    flat_var = float(local_var[flat_mask].mean()) if flat_mask.any() else 0.0

    # SNR proxy
    snr = float(gray.astype(np.float32).mean() / sigma)

    return {
        "noise_sigma_immerkaar": sigma,
        "flat_region_variance": flat_var,
        "snr_proxy": snr,
    }


def _local_variance(gray: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Compute block-level variance across non-overlapping patches."""
    h, w = gray.shape
    g = gray.astype(np.float32)
    variances = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = g[y:y + block_size, x:x + block_size]
            variances.append(block.var())
    return np.array(variances, dtype=np.float32)


def _color_features(bgr: np.ndarray) -> dict:
    """
    Color / Saturation Features
    ---------------------------
    mean_saturation : Mean HSV saturation (0-1).
        Desaturation from exposure or processing shifts this toward 0.

    channel_imbalance : Mean absolute deviation of per-channel (R,G,B) means.
        Measures chromatic imbalance indicative of white balance errors or
        channel-specific sensor defects.

    colorfulness : Hasler & Süsstrunk (2003) colorfulness metric.
        Derived from the standard deviation and mean of the rg and yb
        opponent colour channels. Low values suggest monochrome-like images.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    mean_sat = float(sat.mean())

    b, g, r = bgr[:, :, 0].astype(np.float32), bgr[:, :, 1].astype(np.float32), bgr[:, :, 2].astype(np.float32)
    channel_means = np.array([r.mean(), g.mean(), b.mean()])
    ch_imbalance = float(np.abs(channel_means - channel_means.mean()).mean())

    # Hasler & Süsstrunk colorfulness
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    return {
        "mean_saturation": mean_sat,
        "channel_imbalance": ch_imbalance,
        "colorfulness": colorfulness,
    }


def _texture_features(gray: np.ndarray) -> dict:
    """
    Texture Features (GLCM)
    -----------------------
    Uses a Gray Level Co-occurrence Matrix (GLCM) at distance=1 and four
    angles (0°, 45°, 90°, 135°) then averages over angles.

    glcm_contrast : Measures local intensity variation.
        High in noisy or high-texture images; low in blurry or homogeneous ones.

    glcm_homogeneity : Measures closeness of the GLCM to its diagonal.
        High for uniform / blurry images; low for noisy or textured images.

    glcm_energy : Sum of squared entries (Angular Second Moment).
        High for periodic/structured textures; low for random noise.
    """
    # Quantise to 64 levels for computational tractability
    g64 = (gray // 4).astype(np.uint8)
    g64 = np.clip(g64, 0, 63)

    try:
        glcm = graycomatrix(g64, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=64, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, "contrast").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())
        energy = float(graycoprops(glcm, "energy").mean())
    except Exception:
        contrast, homogeneity, energy = 0.0, 0.0, 0.0

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
    }


def _corruption_features(bgr: np.ndarray, gray: np.ndarray) -> dict:
    """
    Compression / Corruption Features
    ----------------------------------
    dct_blockiness : Boundary discontinuity score across 8×8 DCT block edges.
        Measures the mean absolute difference between adjacent pixel rows/columns
        at 8-pixel intervals (where JPEG quantisation creates blocking artefacts).
        Higher values indicate stronger JPEG blockiness.

    hf_energy_loss : Ratio of high-frequency energy in the original vs. a
        lightly re-encoded JPEG version. Corruption typically strips high-freq
        content, making this ratio < 1 in corrupted images.

    compression_gradient : Mean gradient magnitude along 8×8 block boundaries
        relative to the global gradient. Blockiness inflates boundary gradients.
    """
    h, w = gray.shape

    # DCT blockiness: mean |intensity jump| at 8-pixel boundaries
    boundary_diffs_h = []
    for row in range(8, h, 8):
        diff = np.abs(gray[row, :].astype(np.float32) - gray[row - 1, :].astype(np.float32))
        boundary_diffs_h.append(diff.mean())
    boundary_diffs_v = []
    for col in range(8, w, 8):
        diff = np.abs(gray[:, col].astype(np.float32) - gray[:, col - 1].astype(np.float32))
        boundary_diffs_v.append(diff.mean())

    all_diffs = boundary_diffs_h + boundary_diffs_v
    dct_blockiness = float(np.mean(all_diffs)) if all_diffs else 0.0

    # HF energy loss via round-trip light JPEG encode
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        _, encimg = cv2.imencode('.jpg', bgr, encode_param)
        decoded = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

        orig_f = np.fft.fft2(gray.astype(np.float64))
        dec_f = np.fft.fft2(decoded.astype(np.float64))

        cy, cx = h // 2, w // 2
        r_lo = min(h, w) // 10
        y_g, x_g = np.ogrid[:h, :w]
        low_m = (y_g - cy) ** 2 + (x_g - cx) ** 2 <= r_lo ** 2

        orig_hf = (np.abs(np.fft.fftshift(orig_f)) ** 2 * ~low_m).sum()
        dec_hf = (np.abs(np.fft.fftshift(dec_f)) ** 2 * ~low_m).sum()
        hf_energy_loss = float(dec_hf / (orig_hf + 1e-8))
    except Exception:
        hf_energy_loss = 1.0

    # Compression gradient: boundary vs. global gradient ratio
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_global = np.sqrt(sobelx ** 2 + sobely ** 2).mean()

    boundary_mask = np.zeros_like(gray, dtype=bool)
    boundary_mask[8::8, :] = True
    boundary_mask[:, 8::8] = True
    grad_full = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_boundary = grad_full[boundary_mask].mean() if boundary_mask.any() else 0.0
    compression_gradient = float(grad_boundary / (grad_global + 1e-8))

    return {
        "dct_blockiness": dct_blockiness,
        "hf_energy_loss": hf_energy_loss,
        "compression_gradient": compression_gradient,
    }


def extract_features(image_input) -> dict:
    """
    Main entry point. Accepts either:
      - A file path (str / Path) pointing to an image file.
      - A numpy ndarray in BGR format (as returned by cv2.imread).

    Returns a flat dict of 22 named features.
    Raises ValueError if the image cannot be loaded or is empty.
    """
    if isinstance(image_input, (str, bytes)) or hasattr(image_input, "__fspath__"):
        bgr = cv2.imread(str(image_input))
    else:
        bgr = image_input

    if bgr is None or bgr.size == 0:
        raise ValueError(f"Could not load image from: {image_input!r}")

    # Standardise to a fixed working resolution to ensure feature comparability
    # across images of different sizes (Picsum delivers 800x600 but may vary)
    bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    features = {}
    features.update(_sharpness_features(gray))
    features.update(_exposure_features(gray))
    features.update(_contrast_features(gray))
    features.update(_noise_features(gray))
    features.update(_color_features(bgr))
    features.update(_texture_features(gray))
    features.update(_corruption_features(bgr, gray))

    assert set(features.keys()) == set(FEATURE_NAMES), (
        f"Feature name mismatch. Expected {sorted(FEATURE_NAMES)}, "
        f"got {sorted(features.keys())}"
    )

    return features
