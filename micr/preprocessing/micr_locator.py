"""Locate the MICR line region within a check image."""

import cv2
import numpy as np

from micr.preprocessing.common import resize_to_height


def _maybe_resize(roi: np.ndarray, target_height: int) -> np.ndarray:
    if target_height > 0:
        return resize_to_height(roi, target_height)
    return roi


def locate_micr_line(binary: np.ndarray, target_height: int = 0) -> np.ndarray:
    """
    Locate and extract the MICR line region from a preprocessed binary image.

    The MICR line is always printed in the bottom portion of a check.
    Uses horizontal projection profile to find the text band.

    Args:
        binary: Binary image (text=white, background=black).
        target_height: Height to normalize the extracted ROI to.
                       0 means keep original resolution.

    Returns:
        Extracted and normalized MICR line ROI (binary image).
    """
    h, w = binary.shape[:2]

    # If the image is very short (likely already a MICR line crop), use as-is
    aspect_ratio = w / h
    if aspect_ratio > 5:
        return _maybe_resize(binary, target_height)

    # Focus on bottom 30% of the image where MICR line lives
    bottom_start = int(h * 0.70)
    bottom_region = binary[bottom_start:, :]

    # Horizontal projection: sum pixel intensities per row
    h_proj = np.sum(bottom_region, axis=1).astype(float)

    if h_proj.max() == 0:
        # No content found in bottom region, try full image
        bottom_region = binary
        h_proj = np.sum(bottom_region, axis=1).astype(float)
        bottom_start = 0

    # Normalize projection
    h_proj /= h_proj.max() + 1e-10

    # Find rows with significant content (> 10% of max)
    threshold = 0.10
    active_rows = np.where(h_proj > threshold)[0]

    if len(active_rows) == 0:
        # Fallback: return bottom 15% of original image
        roi = binary[int(h * 0.85):, :]
        return _maybe_resize(roi, target_height)

    # Find the densest band of active rows (the MICR line)
    best_start = active_rows[0]
    best_end = active_rows[-1]

    # Add padding around the detected band
    bh = bottom_region.shape[0]
    pad = max(5, int((best_end - best_start) * 0.2))
    row_start = max(0, best_start - pad)
    row_end = min(bh, best_end + pad)

    roi = bottom_region[row_start:row_end, :]

    # Trim horizontal whitespace
    v_proj = np.sum(roi, axis=0)
    active_cols = np.where(v_proj > 0)[0]
    if len(active_cols) > 0:
        col_start = max(0, active_cols[0] - 10)
        col_end = min(roi.shape[1], active_cols[-1] + 10)
        roi = roi[:, col_start:col_end]

    return _maybe_resize(roi, target_height)
