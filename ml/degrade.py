import cv2
import numpy as np
from PIL import Image
import io
import random

def apply_blur(image_bgr: np.ndarray, severity: str = "moderate") -> np.ndarray:
    
    img = image_bgr.copy()
    if severity == "mild":
        ksize = 5
        sigma = 1.8
    elif severity == "moderate":
        ksize = 11
        sigma = 3.5
    else:  
        ksize = 21
        sigma = 7.0
        
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def apply_underexposure(image_bgr: np.ndarray, severity: str = "moderate") -> np.ndarray:
    
    img_f = image_bgr.astype(np.float32) / 255.0
    if severity == "mild":
        gamma = 1.8
        scale = 0.75
    elif severity == "moderate":
        gamma = 2.8
        scale = 0.50
    else:  # severe
        gamma = 4.2
        scale = 0.25
        
    degraded = np.power(img_f * scale, gamma) * 255.0
    return np.clip(degraded, 0, 255).astype(np.uint8)

def apply_overexposure(image_bgr: np.ndarray, severity: str = "moderate") -> np.ndarray:
    
    img_f = image_bgr.astype(np.float32) / 255.0
    if severity == "mild":
        gamma = 0.65
        gain = 1.3
    elif severity == "moderate":
        gamma = 0.45
        gain = 1.7
    else:  # severe
        gamma = 0.30
        gain = 2.2
        
    degraded = np.power(img_f, gamma) * gain * 255.0
    return np.clip(degraded, 0, 255).astype(np.uint8)

def apply_noise(image_bgr: np.ndarray, severity: str = "moderate") -> np.ndarray:
    
    img = image_bgr.astype(np.float32)
    if severity == "mild":
        sigma = 18.0
    elif severity == "moderate":
        sigma = 38.0
    else:  # severe
        sigma = 68.0
        
    gaussian = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img + gaussian
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_corruption(image_bgr: np.ndarray, severity: str = "moderate") -> np.ndarray:
    
    if severity == "mild":
        quality = 18
    elif severity == "moderate":
        quality = 8
    else:  # severe
        quality = 3
        
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
    if not result:
        return image_bgr
    decoded = cv2.imdecode(encimg, 1)
    
    # For severe corruption, introduce subtle horizontal line slice shifts
    if severity == "severe":
        h, w, _ = decoded.shape
        num_glitches = random.randint(2, 4)
        for _ in range(num_glitches):
            y_start = random.randint(0, h - 30)
            y_end = min(h, y_start + random.randint(8, 24))
            shift = random.randint(-15, 15)
            decoded[y_start:y_end] = np.roll(decoded[y_start:y_end], shift, axis=1)
            
    return decoded

def apply_defect(image_bgr: np.ndarray, severity: str = "moderate") -> np.ndarray:
    
    img = image_bgr.copy()
    h, w, _ = img.shape
    
    if severity == "mild":
        
        num_scratches = random.randint(1, 2)
        for _ in range(num_scratches):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (pt1[0] + random.randint(-80, 80), pt1[1] + random.randint(-80, 80))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.line(img, pt1, pt2, color, thickness=random.randint(1, 2))
    elif severity == "moderate":
        
        num_scratches = random.randint(2, 4)
        for _ in range(num_scratches):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (pt1[0] + random.randint(-120, 120), pt1[1] + random.randint(-120, 120))
            color = (0, 0, 0) if random.random() > 0.5 else (255, 255, 255)
            cv2.line(img, pt1, pt2, color, thickness=random.randint(2, 3))
        
        bx = random.randint(w // 4, 3 * w // 4)
        by = random.randint(h // 4, 3 * h // 4)
        radius = random.randint(15, 30)
        cv2.circle(img, (bx, by), radius, (30, 30, 30), -1)
    else:
        
        bx = random.randint(w // 6, 5 * w // 6)
        by = random.randint(h // 6, 5 * h // 6)
        pw = random.randint(40, 80)
        ph = random.randint(40, 80)
        patch = np.random.randint(0, 256, (ph, pw, 3), dtype=np.uint8)
        img[by:min(h, by+ph), bx:min(w, bx+pw)] = patch[:min(h-by, ph), :min(w-bx, pw)]
        
        
        for _ in range(4):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (pt1[0] + random.randint(-150, 150), pt1[1] + random.randint(-150, 150))
            cv2.line(img, pt1, pt2, (0, 255, 255), thickness=3)
            
    return img

DEGRADATION_MAP = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
    "defect": apply_defect
}
