"""
scripts/generate_stress_test_suite.py
=====================================
Generates a curated 30-sample In-The-Wild & Edge-Case Stress Test Suite in data/stress_test/
Covering all 7 distinct failure categories.
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "data/stress_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load a clean base reference
clean_base_path = "docs/sample_images/sample_pristine___clean.jpg"
base_img = cv2.imread(clean_base_path)
if base_img is None:
    # Generate synthetic landscape base if missing
    base_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(base_img, (0, 0), (640, 240), (230, 200, 150), -1)
    cv2.rectangle(base_img, (0, 240), (640, 480), (80, 150, 60), -1)

h, w, c = base_img.shape

def save(name: str, img: np.ndarray):
    p = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"[+] Generated: {name}")

# ==========================================
# CATEGORY 1: EXTREME & EDGE EXPOSURE (5)
# ==========================================
# 01. Total Sensor Blackout
save("01_total_blackout.jpg", np.zeros_like(base_img))

# 02. Total Highlight Blowout
save("02_total_whiteout.jpg", np.full_like(base_img, 255))

# 03. Night Cityscape with Bright Lights
night = np.clip((base_img.astype(np.float32) / 255.0 * 0.25) ** 1.8 * 255.0, 0, 255).astype(np.uint8)
cv2.circle(night, (w//4, h//3), 25, (255, 255, 255), -1)
cv2.circle(night, (3*w//4, h//2), 30, (200, 255, 255), -1)
cv2.rectangle(night, (w//2 - 40, h - 80), (w//2 + 40, h), (255, 200, 100), -1)
save("03_night_cityscape_neon.jpg", night)

# 04. Realistic Severe Underexposure
under = np.clip((base_img.astype(np.float32) / 255.0 * 0.45) ** 2.5 * 255.0, 0, 255).astype(np.uint8)
save("04_severe_underexposure.jpg", under)

# 05. High-Key Overexposure with Clipped Sky
over = np.clip((base_img.astype(np.float32) / 255.0) ** 0.35 * 1.8 * 255.0, 0, 255).astype(np.uint8)
save("05_high_key_overexposure.jpg", over)

# ==========================================
# CATEGORY 2: FOCUS, BOKEH & SHARPNESS (5)
# ==========================================
# 06. Portrait Bokeh (Sharp Center Face + Blurry Background)
bokeh = cv2.GaussianBlur(base_img, (35, 35), 15)
center_mask = np.zeros((h, w), dtype=np.float32)
cv2.circle(center_mask, (w//2, h//2), min(h, w)//3, 1.0, -1)
center_mask = cv2.GaussianBlur(center_mask, (41, 41), 15)[:, :, None]
portrait_bokeh = (base_img * center_mask + bokeh * (1 - center_mask)).astype(np.uint8)
save("06_portrait_shallow_bokeh.jpg", portrait_bokeh)

# 07. Severe Motion Blur (Horizontal camera shake)
kernel_motion = np.zeros((25, 25))
kernel_motion[12, :] = 1.0 / 25.0
motion_blur = cv2.filter2D(base_img, -1, kernel_motion)
save("07_severe_motion_blur.jpg", motion_blur)

# 08. Severe Defocus Blur
defocus = cv2.GaussianBlur(base_img, (31, 31), 10)
save("08_severe_defocus_blur.jpg", defocus)

# 09. Ultra Sharp Architecture (Enhanced edges)
sharpen_k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
ultra_sharp = np.clip(cv2.filter2D(base_img, -1, sharpen_k), 0, 255)
save("09_ultra_sharp_architecture.jpg", ultra_sharp)

# 10. Smooth Gradient Sky (Pristine low texture)
gradient_sky = np.zeros_like(base_img)
for r in range(h):
    frac = r / h
    gradient_sky[r, :] = [int(220 * (1 - frac) + 120 * frac), int(180 * (1 - frac) + 90 * frac), int(100 * (1 - frac) + 40 * frac)]
save("10_smooth_gradient_sky.jpg", gradient_sky)

# ==========================================
# CATEGORY 3: SENSOR NOISE & TEXTURE (4)
# ==========================================
# 11. High ISO Low-Light Noise
noise_high = base_img.astype(np.float32) + np.random.normal(0, 32, base_img.shape)
save("11_high_iso_night_grain.jpg", np.clip(noise_high, 0, 255).astype(np.uint8))

# 12. Subtle Film Grain
film_grain = base_img.astype(np.float32) + np.random.normal(0, 8, base_img.shape)
save("12_mild_film_grain.jpg", np.clip(film_grain, 0, 255).astype(np.uint8))

# 13. Heavy Gaussian Noise
heavy_noise = base_img.astype(np.float32) + np.random.normal(0, 55, base_img.shape)
save("13_heavy_gaussian_noise.jpg", np.clip(heavy_noise, 0, 255).astype(np.uint8))

# 14. Flat Uniform Studio Backdrop
flat_backdrop = np.full_like(base_img, (180, 180, 180))
save("14_flat_uniform_color.jpg", flat_backdrop)

# ==========================================
# CATEGORY 4: COMPRESSION ARTIFACTS (4)
# ==========================================
# 15. Heavy JPEG Meme Blockiness (Quality 10)
_, enc10 = cv2.imencode(".jpg", base_img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
save("15_heavy_jpeg_meme_blocking.jpg", cv2.imdecode(enc10, cv2.IMREAD_COLOR))

# 16. Standard Web JPEG (Quality 60)
_, enc60 = cv2.imencode(".jpg", base_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
save("16_mild_web_jpeg.jpg", cv2.imdecode(enc60, cv2.IMREAD_COLOR))

# 17. Pristine Lossless Image
save("17_pristine_lossless_reference.jpg", base_img.copy())

# 18. Severe Color Banding / Posterization
poster = (base_img // 64) * 64
save("18_chroma_posterization.jpg", poster)

# ==========================================
# CATEGORY 5: PHYSICAL DAMAGE & DEFECTS (4)
# ==========================================
# 19. Cracked Glass / Linear Scratch
scratch = base_img.copy()
cv2.line(scratch, (w//6, h//6), (5*w//6, 5*h//6), (255, 255, 255), 4)
cv2.line(scratch, (w//6, h//6), (5*w//6, 5*h//6), (30, 30, 30), 1)
save("19_cracked_glass_scratch.jpg", scratch)

# 20. Camera Sensor Dust Spot
dust = base_img.copy()
cv2.circle(dust, (w//3, h//3), 18, (30, 30, 30), -1)
dust = cv2.GaussianBlur(dust, (9, 9), 3)
save("20_sensor_dust_spot.jpg", dust)

# 21. Translucent Water Stain
stain = base_img.copy().astype(np.float32)
cv2.circle(stain, (2*w//3, h//2), 45, (40, 80, 140), -1)
stain_blend = (base_img.astype(np.float32) * 0.65 + stain * 0.35).astype(np.uint8)
save("21_surface_water_stain.jpg", stain_blend)

# 22. Heavy Physical Cutout Defect
cutout = base_img.copy()
cv2.rectangle(cutout, (w//2 - 50, h//2 - 50), (w//2 + 50, h//2 + 50), (0, 0, 0), -1)
save("22_punched_hole_defect.jpg", cutout)

# ==========================================
# CATEGORY 6: DIGITAL & VECTOR ART / OOD (4)
# ==========================================
# 23. Comic Art with Text
comic = base_img.copy()
cv2.rectangle(comic, (40, 40), (w - 40, 120), (0, 240, 255), -1)
cv2.rectangle(comic, (40, 40), (w - 40, 120), (0, 0, 0), 4)
cv2.putText(comic, "CRITICAL ALERT 9000", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
save("23_vector_comic_text_overlay.jpg", comic)

# 24. Software UI Screenshot Mock
ui_mock = np.full((h, w, 3), 30, dtype=np.uint8)
cv2.rectangle(ui_mock, (20, 20), (w - 20, 70), (45, 45, 45), -1)
cv2.putText(ui_mock, "def compute_quality_score(features):", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 120), 2)
cv2.putText(ui_mock, "    return quality_index, label", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 180, 100), 2)
save("24_software_ui_screenshot.jpg", ui_mock)

# 25. High Contrast Bar Chart Infographic
chart = np.full((h, w, 3), 245, dtype=np.uint8)
cv2.rectangle(chart, (100, 150), (180, 400), (200, 80, 50), -1)
cv2.rectangle(chart, (240, 220), (320, 400), (50, 150, 80), -1)
cv2.rectangle(chart, (380, 100), (460, 400), (80, 100, 220), -1)
save("25_digital_infographic_chart.jpg", chart)

# 26. Flat Vector Illustration
vector_art = np.zeros_like(base_img)
cv2.rectangle(vector_art, (0, 0), (w, h//2), (255, 180, 120), -1)
cv2.rectangle(vector_art, (0, h//2), (w, h), (100, 180, 240), -1)
cv2.circle(vector_art, (w//2, h//2), 90, (80, 80, 240), -1)
save("26_flat_vector_illustration.jpg", vector_art)

# ==========================================
# CATEGORY 7: PRISTINE NATURAL REFERENCES (4)
# ==========================================
# 27. Pristine Reference 1 (Clean base)
save("27_pristine_nature_reference.jpg", base_img.copy())

# 28. Pristine Reference 2 (Slightly adjusted tone)
base2 = np.clip(base_img.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
save("28_pristine_urban_reference.jpg", base2)

# 29. Pristine Reference 3 (Vibrant natural saturation)
hsv_p = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv_p[:, :, 1] = np.clip(hsv_p[:, :, 1] * 1.15, 0, 255)
base3 = cv2.cvtColor(hsv_p.astype(np.uint8), cv2.COLOR_HSV2BGR)
save("29_pristine_vibrant_texture.jpg", base3)

# 30. Pristine Reference 4 (Warm indoor tone)
base4 = base_img.copy()
base4[:, :, 2] = np.clip(base4[:, :, 2] * 1.08, 0, 255)
save("30_pristine_warm_indoor.jpg", base4)

print(f"\n[+] Successfully generated all 30 test images in {OUTPUT_DIR}!")
