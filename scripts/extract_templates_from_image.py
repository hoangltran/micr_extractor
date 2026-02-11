#!/usr/bin/env python3
"""
Extract MICR E-13B reference templates from a known check image.

Uses the segmentation pipeline on MICR.png and maps each ROI
to its known ground truth character, then saves them as templates.
"""

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "micr" / "resources" / "templates"

# Ground truth for MICR.png: ⑈267084131⑈ 790319013⑆1024
# The segmentation (with threshold=11) produces these groups in order:
GROUND_TRUTH = [
    "transit",  # ⑈ (bar + 2 dots)
    "2",
    "6",
    "7",
    "0",
    "8",
    "4",
    "1",        # digit 1 (may include noise speck)
    "3",
    "1",        # digit 1
    "transit",  # ⑈ (bar + 2 dots)
    # --- large gap (space) ---
    "7",
    "9",
    "0",
    "3",
    "1",
    "9",
    "0",
    "1",
    "3",
    "on_us",    # ⑆ (2 thin bars + possible artifact)
    "1",
    "0",
    "2",
    "4",
]


def extract_templates():
    """Extract character templates from MICR.png."""
    image_path = PROJECT_ROOT / "MICR.png"
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Preprocess (same pipeline as scanner_pipeline)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_img = binary.shape[0]

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if h >= h_img * 0.10 and w >= 2 and area >= 20:
            rects.append((x, y, w, h))

    rects.sort(key=lambda r: r[0])

    # Compute gap threshold
    tall_heights = [h for _, _, _, h in rects if h > h_img * 0.3]
    median_char_height = sorted(tall_heights)[len(tall_heights) // 2] if tall_heights else h_img * 0.5
    gap_threshold = median_char_height * 0.25

    # Group contours
    groups = []
    current_group = [rects[0]]

    for i in range(1, len(rects)):
        prev_right = max(r[0] + r[2] for r in current_group)
        curr_left = rects[i][0]
        gap = curr_left - prev_right

        if gap < gap_threshold:
            current_group.append(rects[i])
        else:
            groups.append(current_group)
            current_group = [rects[i]]

    groups.append(current_group)

    # Merge each group into bounding box
    merged = []
    for group in groups:
        x_min = min(r[0] for r in group)
        y_min = min(r[1] for r in group)
        x_max = max(r[0] + r[2] for r in group)
        y_max = max(r[1] + r[3] for r in group)
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    print(f"Found {len(merged)} character groups")
    print(f"Expected {len(GROUND_TRUTH)} characters")

    if len(merged) != len(GROUND_TRUTH):
        print("\nWARNING: Count mismatch! Listing all groups:")
        for i, (x, y, w, h) in enumerate(merged):
            gt = GROUND_TRUTH[i] if i < len(GROUND_TRUTH) else "???"
            print(f"  [{i:2d}] x={x:4d} w={w:3d} h={h:3d} -> {gt}")
        return

    # Extract and save templates (best quality per character)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}

    for i, ((x, y, w, h), char_name) in enumerate(zip(merged, GROUND_TRUTH)):
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(binary.shape[1], x + w + pad)
        y1 = min(binary.shape[0], y + h + pad)
        roi = binary[y0:y1, x0:x1]

        # Keep the best quality template per character (largest area)
        if char_name not in saved or (w * h > saved[char_name][1]):
            filepath = TEMPLATES_DIR / f"{char_name}.png"
            cv2.imwrite(str(filepath), roi)
            saved[char_name] = (filepath, w * h)
            print(f"  [{i:2d}] Saved {char_name:>8s} -> {filepath.name} ({w}x{h})")
        else:
            print(f"  [{i:2d}] Skipped {char_name:>8s} (already have better)")

    print(f"\nExtracted {len(saved)} unique templates to {TEMPLATES_DIR}")


if __name__ == "__main__":
    extract_templates()
