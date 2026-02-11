"""Common image preprocessing utilities for MICR extraction."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if it isn't already."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Binarize using Otsu's method. Returns binary image (text=white, bg=black)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(gray: np.ndarray, block_size: int = 35, c: int = 15) -> np.ndarray:
    """Binarize using adaptive Gaussian thresholding. Returns binary (text=white, bg=black)."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    return binary


def denoise(gray: np.ndarray, strength: int = 10) -> np.ndarray:
    """Apply non-local means denoising."""
    return cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)


def deskew(binary: np.ndarray) -> np.ndarray:
    """
    Deskew a binary image by detecting the dominant text line angle.

    Uses minAreaRect on non-zero pixel coordinates to estimate rotation.
    """
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 10:
        return binary

    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle to get the actual skew
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only correct if skew is small (< 15 degrees)
    if abs(angle) > 15 or abs(angle) < 0.1:
        return binary

    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def remove_noise_morphological(binary: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Remove small noise using morphological opening."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to a target height while preserving aspect ratio."""
    h, w = image.shape[:2]
    if h == target_height:
        return image
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
