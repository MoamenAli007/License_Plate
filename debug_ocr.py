"""
Debug script: shows each preprocessing variant + raw OCR output.
Run: python debug_ocr.py
"""
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys, cv2, numpy as np
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='ar', use_gpu=False, show_log=False)

# ── Load the saved plate crop ─────────────────────────────────────────
import pathlib

# Use the ASCII-named copy (Arabic filenames break cv2.imread on Windows)
crop_path = "debug_plate_crop.jpg"
if not pathlib.Path(crop_path).exists():
    print("ERROR: debug_plate_crop.jpg not found.")
    print("Run detect_rec.py --save-crop first, then copy the crop:")
    print("  python -c \"import shutil,glob; shutil.copy(sorted(glob.glob('runs/detect/exp*/crops/licence/*.jpg'))[-1],'debug_plate_crop.jpg')\"")
    sys.exit(1)

print(f"Using crop: {crop_path}\n")

img = cv2.imread(crop_path)
print(f"Raw crop size: {img.shape[1]}x{img.shape[0]} px")

# ── Upscale to working size ───────────────────────────────────────────
min_h, min_w = 160, 400
ch, cw = img.shape[:2]
scale = max(min_h / ch, min_w / cw, 1.0)
img = cv2.resize(img, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
print(f"Upscaled to:  {img.shape[1]}x{img.shape[0]} px\n")

# Save upscaled for inspection
cv2.imwrite("debug_upscaled.png", img)
print("Saved: debug_upscaled.png")

# ── Preprocessing variants ────────────────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe_op   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
clahe_gray = clahe_op.apply(gray)
clahe_img  = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
kernel     = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
_, otsu    = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, otsu_i  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Strip EGYPT banner (right ~28%)
cw2 = img.shape[1]
plate_only = img[:, : int(cw2 * 0.72)]

variants = [
    ("full_original",   img),
    ("plate_only",      plate_only),
    ("clahe",           clahe_img),
    ("clahe+sharp",     cv2.filter2D(clahe_img, -1, kernel)),
    ("otsu",            cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
    ("otsu_inv",        cv2.cvtColor(otsu_i, cv2.COLOR_GRAY2BGR)),
]

# ── Run OCR on each and print raw results ─────────────────────────────
print("=" * 60)
for label, v in variants:
    res = ocr.ocr(v, cls=True)
    cv2.imwrite(f"debug_{label}.png", v)
    print(f"\n[{label}]  (saved debug_{label}.png)")
    if res and res[0]:
        for line in res[0]:
            text = line[1][0]
            conf = line[1][1]
            print(f"  text='{text}'  conf={conf:.3f}")
    else:
        print("  → NO TEXT")
print("\n" + "=" * 60)
print("Open the debug_*.png files to see what each variant looks like.")
