"""
scripts/generate_hybrid_benchmark.py
====================================
Generates a comprehensive 1,200-image Hybrid Benchmark Pack in data/hybrid_benchmark/
with a structured ground-truth manifest in data/hybrid_benchmark_manifest.csv.
"""

import os, sys, cv2, numpy as np, pandas as pd, glob, random

OUTPUT_DIR = "data/hybrid_benchmark"
MANIFEST_PATH = "data/hybrid_benchmark_manifest.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

raw_paths = glob.glob("data/raw/*.jpg")
if len(raw_paths) < 50:
    raw_paths = glob.glob("data/**/*.jpg", recursive=True)

print(f"[*] Found {len(raw_paths)} base raw images for hybrid dataset generation.")

records = []
img_counter = 1

def save_img(filename, img, category, subcat, blur=0, under=0, over=0, noise=0, corrupt=0, defect=0, gt_score=100.0, gt_label="ACCEPTABLE"):
    global img_counter
    p = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    records.append({
        "id": img_counter,
        "filename": filename,
        "category": category,
        "sub_category": subcat,
        "has_blur": blur,
        "has_underexposure": under,
        "has_overexposure": over,
        "has_noise": noise,
        "has_corruption": corrupt,
        "has_defect": defect,
        "gt_quality_score": round(float(gt_score), 1),
        "gt_quality_label": gt_label
    })
    img_counter += 1

print("[*] Generating 200 Pristine Clean References...")
for i in range(200):
    base_p = raw_paths[i % len(raw_paths)]
    img = cv2.imread(base_p)
    if img is None: continue
    img = cv2.resize(img, (640, 480))
    save_img(f"clean_{i+1:04d}.jpg", img, "Pristine", "clean_reference", 0,0,0,0,0,0, 100.0, "ACCEPTABLE")

print("[*] Generating 180 Optical Focus & Blur Variants...")
for i in range(180):
    base_p = raw_paths[(i + 30) % len(raw_paths)]
    img = cv2.resize(cv2.imread(base_p), (640, 480))
    subtype = i % 4
    if subtype == 0: 
        k = random.choice([7, 9])
        b = cv2.GaussianBlur(img, (k, k), 2.5)
        save_img(f"blur_defocus_mild_{i+1:04d}.jpg", b, "Blur", "defocus_mild", 1,0,0,0,0,0, 82.0, "ACCEPTABLE")
    elif subtype == 1:
        k = random.choice([21, 29])
        b = cv2.GaussianBlur(img, (k, k), 8.0)
        save_img(f"blur_defocus_severe_{i+1:04d}.jpg", b, "Blur", "defocus_severe", 1,0,0,0,0,0, 68.0, "DEGRADED")
    elif subtype == 2: 
        k = np.zeros((21, 21))
        k[10, :] = 1.0 / 21.0
        b = cv2.filter2D(img, -1, k)
        save_img(f"blur_motion_{i+1:04d}.jpg", b, "Blur", "motion_blur", 1,0,0,0,0,0, 65.0, "DEGRADED")
    else: 
        b_full = cv2.GaussianBlur(img, (35, 35), 12)
        mask = np.zeros((480, 640), dtype=np.float32)
        cv2.circle(mask, (320, 240), 140, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (35, 35), 10)[:, :, None]
        bokeh = (img * mask + b_full * (1 - mask)).astype(np.uint8)
        save_img(f"bokeh_portrait_{i+1:04d}.jpg", bokeh, "Blur", "portrait_bokeh", 0,0,0,0,0,0, 95.0, "ACCEPTABLE")

print("[*] Generating 180 Exposure & Lighting Variants...")
for i in range(180):
    base_p = raw_paths[(i + 60) % len(raw_paths)]
    img = cv2.resize(cv2.imread(base_p), (640, 480))
    subtype = i % 4
    img_f = img.astype(np.float32) / 255.0
    if subtype == 0: 
        u = np.clip(np.power(img_f * 0.55, 2.2) * 255.0, 0, 255).astype(np.uint8)
        save_img(f"under_realistic_{i+1:04d}.jpg", u, "Exposure", "underexposure_realistic", 0,1,0,0,0,0, 70.0, "DEGRADED")
    elif subtype == 1: 
        u = np.clip(np.power(img_f * 0.35, 3.0) * 255.0, 0, 255).astype(np.uint8)
        save_img(f"under_severe_{i+1:04d}.jpg", u, "Exposure", "underexposure_severe", 0,1,0,0,0,0, 50.0, "DEGRADED")
    elif subtype == 2: 
        o = np.clip(np.power(img_f, 0.40) * 1.8 * 255.0, 0, 255).astype(np.uint8)
        save_img(f"over_blowout_{i+1:04d}.jpg", o, "Exposure", "overexposure_blowout", 0,0,1,0,0,0, 35.0, "DEFECTIVE")
    else: 
        if i % 2 == 0:
            save_img(f"blackout_sensor_{i+1:04d}.jpg", np.zeros_like(img), "Exposure", "blackout", 0,1,0,0,0,0, 5.0, "DEFECTIVE")
        else:
            save_img(f"whiteout_sensor_{i+1:04d}.jpg", np.full_like(img, 255), "Exposure", "whiteout", 0,0,1,0,0,0, 5.0, "DEFECTIVE")


print("[*] Generating 180 Sensor Noise & Grain Variants...")
for i in range(180):
    base_p = raw_paths[(i + 90) % len(raw_paths)]
    img = cv2.resize(cv2.imread(base_p), (640, 480))
    subtype = i % 3
    if subtype == 0: 
        n = np.clip(img.astype(np.float32) + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
        save_img(f"noise_film_{i+1:04d}.jpg", n, "Noise", "film_grain", 0,0,0,1,0,0, 82.0, "ACCEPTABLE")
    elif subtype == 1: 
        n = np.clip(img.astype(np.float32) + np.random.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
        save_img(f"noise_high_iso_{i+1:04d}.jpg", n, "Noise", "high_iso", 0,0,0,1,0,0, 72.0, "DEGRADED")
    else: 
        n = np.clip(img.astype(np.float32) + np.random.normal(0, 55, img.shape), 0, 255).astype(np.uint8)
        save_img(f"noise_severe_{i+1:04d}.jpg", n, "Noise", "severe_noise", 0,0,0,1,0,0, 58.0, "DEGRADED")


print("[*] Generating 180 Compression & Glitch Variants...")
for i in range(180):
    base_p = raw_paths[(i + 120) % len(raw_paths)]
    img = cv2.resize(cv2.imread(base_p), (640, 480))
    subtype = i % 3
    if subtype == 0: 
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
        c = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        save_img(f"corrupt_jpeg_heavy_{i+1:04d}.jpg", c, "Corruption", "jpeg_blockiness", 0,0,0,0,1,0, 68.0, "DEGRADED")
    elif subtype == 1: 
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        c = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        save_img(f"corrupt_jpeg_web_{i+1:04d}.jpg", c, "Corruption", "web_compression", 0,0,0,0,1,0, 78.0, "ACCEPTABLE")
    else:
        c = (img // 48) * 48
        save_img(f"corrupt_posterization_{i+1:04d}.jpg", c, "Corruption", "posterization", 0,0,0,0,1,0, 65.0, "DEGRADED")


print("[*] Generating 180 Physical Damage & Defect Variants...")
for i in range(180):
    base_p = raw_paths[(i + 150) % len(raw_paths)]
    img = cv2.resize(cv2.imread(base_p), (640, 480)).copy()
    subtype = i % 3
    if subtype == 0:
        x1, y1 = random.randint(50, 250), random.randint(50, 200)
        x2, y2 = random.randint(350, 580), random.randint(280, 420)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x1, y1), (x2, y2), (20, 20, 20), 1)
        save_img(f"defect_scratch_{i+1:04d}.jpg", img, "Defect", "scratch_crack", 0,0,0,0,0,1, 68.0, "DEGRADED")
    elif subtype == 1:
        for _ in range(random.randint(2, 5)):
            cx, cy = random.randint(80, 560), random.randint(80, 400)
            r = random.randint(8, 20)
            cv2.circle(img, (cx, cy), r, (35, 35, 35), -1)
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        save_img(f"defect_dust_{i+1:04d}.jpg", img, "Defect", "sensor_dust", 0,0,0,0,0,1, 70.0, "DEGRADED")
    else:
        cx, cy = random.randint(150, 490), random.randint(120, 360)
        cv2.rectangle(img, (cx - 35, cy - 35), (cx + 35, cy + 35), (0, 0, 0), -1)
        save_img(f"defect_void_{i+1:04d}.jpg", img, "Defect", "cutout_void", 0,0,0,0,0,1, 55.0, "DEGRADED")


print("[*] Generating 180 Compound Multi-Degradation Variants...")
for i in range(180):
    base_p = raw_paths[(i + 10) % len(raw_paths)]
    img = cv2.resize(cv2.imread(base_p), (640, 480))
    subtype = i % 3
    if subtype == 0:
        b = cv2.GaussianBlur(img, (15, 15), 5.0)
        bn = np.clip(b.astype(np.float32) + np.random.normal(0, 28, img.shape), 0, 255).astype(np.uint8)
        save_img(f"multi_blur_noise_{i+1:04d}.jpg", bn, "Multi", "blur_noise", 1,0,0,1,0,0, 52.0, "DEGRADED")
    elif subtype == 1:
        img_f = img.astype(np.float32) / 255.0
        u = np.clip(np.power(img_f * 0.50, 2.2) * 255.0, 0, 255)
        un = np.clip(u + np.random.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
        save_img(f"multi_under_noise_{i+1:04d}.jpg", un, "Multi", "under_noise", 0,1,0,1,0,0, 50.0, "DEGRADED")
    else: 
        cv2.line(img, (100, 100), (540, 380), (255, 255, 255), 4)
        b = cv2.GaussianBlur(img, (11, 11), 3.0)
        _, enc = cv2.imencode(".jpg", b, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        comp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        save_img(f"multi_defect_blur_jpeg_{i+1:04d}.jpg", comp, "Multi", "defect_blur_jpeg", 1,0,0,0,1,1, 38.0, "DEFECTIVE")

manifest_df = pd.DataFrame(records)
manifest_df.to_csv(MANIFEST_PATH, index=False)
print(f"\n[+] Hybrid Benchmark Pack successfully created!")
print(f"    Total Images: {len(manifest_df)}")
print(f"    Manifest Saved: {MANIFEST_PATH}")
print(manifest_df["category"].value_counts())
