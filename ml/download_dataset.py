import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
TARGET_COUNT = 200

VALID_IDS = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100,
    101, 102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142,
    143, 144, 145, 146, 147, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184,
    185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
    206, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 225, 227, 228,
    229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 247, 248, 249, 250, 251
]

def download_and_validate_image(pic_id, index, output_dir):
    url = f"https://picsum.photos/id/{pic_id}/800/600"
    target_path = os.path.join(output_dir, f"clean_{index:03d}.jpg")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read()
                

            img = Image.open(io.BytesIO(content))
            img.verify()
            
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img.save(target_path, "JPEG", quality=95)
            return True, index, pic_id
        except Exception as e:
            time.sleep(0.5 * (attempt + 1))
            
    return False, index, pic_id

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Sourcing {TARGET_COUNT} clean benchmark images into: {OUTPUT_DIR}")
    
    success_count = 0
    futures = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, pic_id in enumerate(VALID_IDS[:TARGET_COUNT], start=1):
            futures.append(executor.submit(download_and_validate_image, pic_id, idx, OUTPUT_DIR))
            
        for future in as_completed(futures):
            success, index, pic_id = future.result()
            if success:
                success_count += 1
                if success_count % 25 == 0 or success_count == TARGET_COUNT:
                    print(f"    -> Successfully downloaded & verified {success_count}/{TARGET_COUNT} images")
            else:
                print(f"    [!] Failed to download ID {pic_id}")
                
    print(f"\n[+] Dataset acquisition complete! Total clean images ready: {success_count}/{TARGET_COUNT}")

if __name__ == "__main__":
    main()
