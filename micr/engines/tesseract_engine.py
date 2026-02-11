"""Tesseract OCR engine for MICR E-13B recognition (secondary engine)."""

import shutil

import cv2
import numpy as np

from micr.engines.base import BaseMICREngine
from micr.models import CharacterResult


def is_tesseract_available() -> bool:
    """Check if the Tesseract binary is installed."""
    return shutil.which("tesseract") is not None


class TesseractMICREngine(BaseMICREngine):
    """
    Secondary MICR recognition engine using Tesseract OCR.

    Requires:
    - tesseract binary installed (brew install tesseract)
    - pytesseract Python package (pip install pytesseract)
    - Optionally, e13b.traineddata for MICR-specific recognition
    """

    def __init__(self, tessdata_dir: str | None = None, lang: str = "eng"):
        try:
            import pytesseract  # noqa: F401

            self._pytesseract = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required for Tesseract engine. "
                "Install with: pip install pytesseract"
            )

        if not is_tesseract_available():
            raise RuntimeError(
                "Tesseract binary not found. "
                "Install with: brew install tesseract (macOS) "
                "or apt-get install tesseract-ocr (Linux)"
            )

        self._tessdata_dir = tessdata_dir
        self._lang = lang

    @property
    def name(self) -> str:
        return "tesseract"

    def recognize(self, micr_line_image: np.ndarray) -> list[CharacterResult]:
        """Run Tesseract OCR on the MICR line image."""
        # Ensure grayscale
        if len(micr_line_image.shape) == 3:
            gray = cv2.cvtColor(micr_line_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = micr_line_image

        # Ensure text is dark on light background (Tesseract preference)
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)

        # Scale up for better Tesseract recognition
        scale = max(1, 300 // max(gray.shape[0], 1))
        if scale > 1:
            gray = cv2.resize(
                gray,
                (gray.shape[1] * scale, gray.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC,
            )

        # Build Tesseract config
        config_parts = [
            f"-l {self._lang}",
            "--psm 7",  # Single text line
            "--oem 1",  # LSTM engine
            "-c tessedit_char_whitelist=0123456789",
        ]
        if self._tessdata_dir:
            config_parts.insert(0, f"--tessdata-dir {self._tessdata_dir}")

        config = " ".join(config_parts)

        # Get character-level data
        try:
            data = self._pytesseract.image_to_data(
                gray, config=config, output_type=self._pytesseract.Output.DICT
            )
        except Exception:
            return []

        results = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue

            conf = data["conf"][i]
            if isinstance(conf, str):
                try:
                    conf = float(conf)
                except ValueError:
                    conf = 0.0

            # Tesseract confidence is 0-100
            confidence = max(0.0, min(1.0, conf / 100.0))

            # Rescale bounding box back to original scale
            bbox = (
                data["left"][i] // scale,
                data["top"][i] // scale,
                data["width"][i] // scale,
                data["height"][i] // scale,
            )

            # Emit individual characters
            for j, ch in enumerate(text):
                if ch.isdigit():
                    char_w = bbox[2] // max(len(text), 1)
                    char_bbox = (
                        bbox[0] + j * char_w,
                        bbox[1],
                        char_w,
                        bbox[3],
                    )
                    results.append(
                        CharacterResult(
                            character=ch,
                            confidence=confidence,
                            bbox=char_bbox,
                            engine=self.name,
                        )
                    )

        return results
