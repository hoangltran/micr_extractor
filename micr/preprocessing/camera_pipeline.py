"""Preprocessing pipeline for camera/phone-captured check images."""

import cv2
import numpy as np

from micr.preprocessing.common import (
    binarize_adaptive,
    deskew,
    remove_noise_morphological,
    to_grayscale,
)


def preprocess_camera(image: np.ndarray) -> np.ndarray:
    """
    Aggressive preprocessing for camera/phone images.

    Handles variable lighting, perspective distortion, noise, and blur.

    Steps:
    1. Grayscale conversion
    2. Bilateral filtering (noise reduction, edge preservation)
    3. CLAHE (contrast enhancement under uneven illumination)
    4. Perspective correction (if check edges detected)
    5. Deskew
    6. Adaptive thresholding
    7. Morphological cleanup

    Returns:
        Binary image with text=white, background=black.
    """
    gray = to_grayscale(image)

    # Bilateral filter: reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # CLAHE: handle uneven illumination
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Attempt perspective correction
    corrected = _correct_perspective(gray)
    if corrected is not None:
        gray = corrected

    # Adaptive thresholding (better for variable lighting than Otsu)
    binary = binarize_adaptive(gray, block_size=35, c=15)

    # Deskew
    binary = deskew(binary)

    # Morphological cleanup
    binary = remove_noise_morphological(binary, kernel_size=2)

    # Close small gaps in characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def _correct_perspective(gray: np.ndarray) -> np.ndarray | None:
    """
    Attempt to detect and correct perspective distortion.

    Looks for a rectangular check boundary and warps to a top-down view.
    Returns None if no suitable rectangle is found.
    """
    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find the largest rectangular contour
    h, w = gray.shape[:2]
    image_area = h * w

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Check if it's a quadrilateral covering a significant area
        if len(approx) == 4 and cv2.contourArea(approx) > image_area * 0.3:
            pts = approx.reshape(4, 2).astype(np.float32)
            warped = _four_point_transform(gray, pts)
            return warped

    return None


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a four-point perspective transform to obtain a top-down view."""
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    # Compute output dimensions
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # Top-right has smallest difference
    rect[3] = pts[np.argmax(d)]  # Bottom-left has largest difference

    return rect
