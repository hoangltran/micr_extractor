"""Public API for MICR E-13B text extraction."""

from pathlib import Path

import cv2
import numpy as np

from micr.engines.template_engine import TemplateMatchingEngine
from micr.models import ImageSource, MICRResult
from micr.parsing.consensus import resolve_consensus
from micr.parsing.parser import parse_micr_line
from micr.parsing.validator import validate_micr_result
from micr.preprocessing.micr_locator import locate_micr_line
from micr.preprocessing.scanner_pipeline import preprocess_scanner


class MICRExtractor:
    """
    Main entry point for MICR E-13B text extraction.

    Usage:
        extractor = MICRExtractor()
        result = extractor.extract("path/to/check.png")
        print(result.routing_number)
        print(result.account_number)
        print(result.check_number)
    """

    def __init__(
        self,
        use_tesseract: bool = False,
        templates_dir: str | Path | None = None,
        tessdata_dir: str | None = None,
        tesseract_lang: str = "eng",
    ):
        """
        Initialize the MICR extractor.

        Args:
            use_tesseract: Whether to use Tesseract as a secondary engine.
            templates_dir: Path to E-13B character templates directory.
            tessdata_dir: Path to Tesseract tessdata directory.
            tesseract_lang: Tesseract language code (e.g., 'eng' or 'e13b').
        """
        self._template_engine = TemplateMatchingEngine(templates_dir)
        self._tesseract_engine = None

        if use_tesseract:
            try:
                from micr.engines.tesseract_engine import TesseractMICREngine

                self._tesseract_engine = TesseractMICREngine(
                    tessdata_dir=tessdata_dir, lang=tesseract_lang
                )
            except (ImportError, RuntimeError) as e:
                import warnings

                warnings.warn(f"Tesseract engine unavailable: {e}")

    def extract(self, image_path: str | Path) -> MICRResult:
        """
        Extract MICR data from a check image file.

        Args:
            image_path: Path to the image file.

        Returns:
            MICRResult with parsed fields and confidence scores.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            ValueError: If the image cannot be read.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        return self.extract_from_array(image)

    def extract_from_array(
        self,
        image: np.ndarray,
        image_source: ImageSource = ImageSource.UNKNOWN,
    ) -> MICRResult:
        """
        Extract MICR data from a numpy array (BGR or grayscale image).

        Args:
            image: Image as numpy array.
            image_source: Hint about the image source type.

        Returns:
            MICRResult with parsed fields and confidence scores.
        """
        # Detect source if not specified
        if image_source == ImageSource.UNKNOWN:
            image_source = self._classify_source(image)

        # Preprocess
        if image_source == ImageSource.CAMERA:
            try:
                from micr.preprocessing.camera_pipeline import preprocess_camera

                binary = preprocess_camera(image)
            except ImportError:
                binary = preprocess_scanner(image)
        else:
            binary = preprocess_scanner(image)

        # Locate MICR line
        micr_roi = locate_micr_line(binary)

        # Run primary engine (template matching)
        primary_results = self._template_engine.recognize(micr_roi)

        # Run secondary engine (Tesseract) if available
        secondary_results = None
        if self._tesseract_engine:
            try:
                secondary_results = self._tesseract_engine.recognize(micr_roi)
            except Exception:
                pass

        # Consensus
        if secondary_results:
            final_chars = resolve_consensus(primary_results, secondary_results)
        else:
            final_chars = primary_results

        # Parse into structured fields
        result = parse_micr_line(final_chars, image_source=image_source)

        # Validate
        result.warnings = validate_micr_result(result)

        return result

    def _classify_source(self, image: np.ndarray) -> ImageSource:
        """
        Simple heuristic to classify whether an image is from a scanner or camera.

        Based on noise level and color characteristics.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )

        # Compute Laplacian variance (noise measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Scanner images tend to have very high contrast and low noise
        # Camera images tend to have more noise and color variation
        if len(image.shape) == 3:
            # Check for color variation (scanners often produce near-grayscale)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation_mean = hsv[:, :, 1].mean()

            if saturation_mean > 30 or laplacian_var > 5000:
                return ImageSource.CAMERA

        return ImageSource.SCANNER
