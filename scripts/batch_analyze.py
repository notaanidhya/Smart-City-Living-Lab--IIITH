"""
scripts/batch_analyze.py
========================
Batch image analysis CLI tool for PixelShamer.
Processes all images in a directory, sends them to the local API,
and generates a summary CSV report of scores and detected issues.
"""

import os
import sys
import glob
import time
import argparse
import httpx
import pandas as pd

def batch_analyze(folder_path: str, api_url: str = "http://localhost:8000/api/analyze", output_csv: str = "batch_results.csv"):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print(f"[!] No images found in: {folder_path}")
        return

    print(f"[*] Starting batch analysis of {len(image_paths)} images against {api_url}...")
    results = []

    with httpx.Client(timeout=30.0) as client:
        for i, path in enumerate(image_paths, 1):
            filename = os.path.basename(path)
            print(f"  [{i}/{len(image_paths)}] Analyzing {filename}...", end="", flush=True)
            try:
                start_t = time.time()
                with open(path, "rb") as f:
                    resp = client.post(api_url, files={"image": (filename, f, "image/jpeg")})
                latency = (time.time() - start_t) * 1000
                
                if resp.status_code == 201:
                    data = resp.json()
                    issues_str = "; ".join([f"{iss['type']}({iss['severity']})" for iss in data.get("issues", [])]) or "None"
                    print(f" -> Score: {data['quality_score']} ({data['quality_label']}) | {issues_str} [{latency:.0f}ms]")
                    results.append({
                        "id": data["id"],
                        "filename": data["filename"],
                        "quality_score": data["quality_score"],
                        "quality_label": data["quality_label"],
                        "issues": issues_str,
                        "reconstruction_error": data.get("statistics", {}).get("reconstruction_error", 0.0),
                        "laplacian_variance": data.get("statistics", {}).get("laplacian_variance", 0.0),
                        "mean_luminance": data.get("statistics", {}).get("mean_luminance", 0.0),
                        "latency_ms": round(latency, 1),
                        "status": "SUCCESS"
                    })
                else:
                    print(f" -> ERROR: HTTP {resp.status_code}")
                    results.append({"filename": filename, "status": f"HTTP_{resp.status_code}"})
            except Exception as e:
                print(f" -> FAILED: {e}")
                results.append({"filename": filename, "status": f"EXCEPTION: {e}"})

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[+] Batch analysis complete! Summary saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelShamer Batch Image Analyzer")
    parser.add_argument("--folder", default="docs/sample_images", help="Folder containing images to analyze")
    parser.add_argument("--api", default="http://localhost:8000/api/analyze", help="API analysis endpoint")
    parser.add_argument("--out", default="batch_analysis_results.csv", help="Output CSV path")
    args = parser.parse_args()

    batch_analyze(args.folder, args.api, args.out)
