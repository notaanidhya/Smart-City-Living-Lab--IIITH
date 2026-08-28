import os
import sys
import uuid
import json
import logging
import cv2
import numpy as np
import torch
from PIL import Image
import io

logger = logging.getLogger("image_quality_api.inference")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.feature_extractor import extract_features, FEATURE_NAMES
from ml.models.mlp import MultiLabelMLP
from ml.models.autoencoder import ConvAutoencoder, IMG_SIZE
from ml.score import compute_quality_score

class InferenceService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mlp_model = None
        self.ae_model = None
        self.ae_calibration_scale = 0.1942
        self.is_ready = False

    def load_models(self, model_dir: str = None):
        """Loads both PyTorch models and metadata into memory."""
        if model_dir is None:
            model_dir = os.path.join(BASE_DIR, "ml", "models")

        mlp_path = os.path.join(model_dir, "mlp_best.pt")
        ae_path = os.path.join(model_dir, "autoencoder_best.pt")
        ae_meta_path = os.path.join(model_dir, "ae_threshold.json")

        logger.info(f"Loading Model A (MLP) from: {mlp_path}")
        mlp_ckpt = torch.load(mlp_path, map_location=self.device, weights_only=True)
        self.mlp_model = MultiLabelMLP().to(self.device)
        self.mlp_model.load_state_dict(mlp_ckpt["model_state"])
        self.mlp_model.eval()

        logger.info(f"Loading Model B (Autoencoder) from: {ae_path}")
        ae_ckpt = torch.load(ae_path, map_location=self.device, weights_only=True)
        self.ae_model = ConvAutoencoder().to(self.device)
        self.ae_model.load_state_dict(ae_ckpt["model_state"])
        self.ae_model.eval()

        if os.path.exists(ae_meta_path):
            try:
                with open(ae_meta_path, "r") as f:
                    meta = json.load(f)
                    self.ae_calibration_scale = meta.get("calibration_scale", 0.1942)
            except Exception as e:
                logger.warning(f"Could not load ae_threshold.json: {e}")

        self.is_ready = True
        logger.info("InferenceService successfully initialized and models loaded.")

    def generate_heatmap_overlay(self, original_bgr: np.ndarray, pixel_err_map: np.ndarray) -> np.ndarray:
        """Upsamples 128x128 error map and applies Jet colormap overlay."""
        h, w, _ = original_bgr.shape
        norm_err = cv2.normalize(pixel_err_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm_err = norm_err.astype(np.uint8)
        resized_heatmap = cv2.resize(norm_err, (w, h), interpolation=cv2.INTER_CUBIC)
        color_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original_bgr, 0.65, color_heatmap, 0.35, 0)
        return overlay

    def analyze_image(self, image_bytes: bytes, original_filename: str, upload_dir: str) -> dict:
        """
        Executes complete end-to-end quality assessment pipeline on an uploaded image.
        """
        if not self.is_ready:
            raise RuntimeError("InferenceService models are not loaded.")

        # 1. Decode image with OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid or corrupted image format; cannot decode pixels.")

        # 2. Setup storage directories and filenames
        images_dir = os.path.join(upload_dir, "images")
        heatmaps_dir = os.path.join(upload_dir, "heatmaps")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(heatmaps_dir, exist_ok=True)

        file_uuid = uuid.uuid4().hex[:12]
        clean_ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
        stored_filename = f"{file_uuid}{clean_ext}"
        heatmap_filename = f"{file_uuid}_heatmap.png"

        image_disk_path = os.path.join(images_dir, stored_filename)
        heatmap_disk_path = os.path.join(heatmaps_dir, heatmap_filename)

        # Save original uploaded image
        cv2.imwrite(image_disk_path, image_bgr)

        # 3. Extract 22 deterministic CV features
        features = extract_features(image_bgr)
        feat_vector = np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)

        # 4. Model A (MLP) Inference
        feat_tensor = torch.tensor(feat_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mlp_probs = self.mlp_model(feat_tensor).squeeze(0).cpu().numpy()

        # 5. Model B (Autoencoder) Inference
        rgb_128 = cv2.cvtColor(cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(rgb_128.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0

        with torch.no_grad():
            recon_err, pixel_err = self.ae_model.reconstruction_error(img_tensor, return_heatmap=True)
            raw_err_val = float(recon_err.item())
            err_map_2d = pixel_err.squeeze().cpu().numpy()

        norm_err = raw_err_val / max(self.ae_calibration_scale, 1e-4)

        # 6. Generate and save explainability heatmap
        overlay_img = self.generate_heatmap_overlay(image_bgr, err_map_2d)
        cv2.imwrite(heatmap_disk_path, overlay_img)

        # 7. Decision Fusion & Scoring
        quality_score, quality_label, issues = compute_quality_score(mlp_probs, recon_error=norm_err)

        # 8. Package statistics
        stats_dict = {
            "laplacian_variance": round(float(features["laplacian_variance"]), 2),
            "mean_luminance": round(float(features["mean_luminance"]), 2),
            "rms_contrast": round(float(features["rms_contrast"]), 3),
            "noise_sigma_immerkaar": round(float(features["noise_sigma_immerkaar"]), 3),
            "mean_saturation": round(float(features["mean_saturation"]), 3),
            "dct_blockiness": round(float(features["dct_blockiness"]), 2),
            "glcm_contrast": round(float(features["glcm_contrast"]), 2),
            "reconstruction_error": round(raw_err_val, 5),
            "all_features": {k: round(float(v), 4) for k, v in features.items()}
        }

        return {
            "filename": original_filename,
            "stored_filename": stored_filename,
            "quality_score": quality_score,
            "quality_label": quality_label,
            "issues": issues,
            "statistics": stats_dict,
            "image_url": f"/uploads/images/{stored_filename}",
            "heatmap_url": f"/uploads/heatmaps/{heatmap_filename}",
        }

inference_service = InferenceService()
