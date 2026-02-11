"""Preprocessing pipeline optimized for scanner/check-scanner images."""

import cv2
import numpy as np

from micr.preprocessing.common import (
    binarize_otsu,
    deskew,
    remove_noise_morphological,
    to_grayscale,
)


def preprocess_scanner(image: np.ndarray) -> np.ndarray:
    """
    Light preprocessing for clean scanner images.

    Steps:
    1. Grayscale conversion
    2. Otsu binarization
    3. Deskew (small angle correction)
    4. Morphological noise removal

    Returns:
        Binary image with text=white, background=black.
    """
    gray = to_grayscale(image)

    # Light Gaussian blur to smooth scanner artifacts
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = binarize_otsu(gray)
    binary = deskew(binary)
    binary = remove_noise_morphological(binary, kernel_size=2)

    return binary
