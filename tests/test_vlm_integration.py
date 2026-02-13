"""Integration tests requiring a running Ollama instance.

Skipped by default. Run with:
    pytest tests/test_vlm_integration.py -v -m integration
"""

from pathlib import Path

import pytest

from micr.engines.vlm_engine import is_ollama_available

pytestmark = pytest.mark.integration

CHECK_IMAGE = Path(__file__).parent.parent / "Check.jpg"
MICR_IMAGE = Path(__file__).parent.parent / "MICR.png"


@pytest.fixture(autouse=True)
def _skip_if_no_ollama():
    if not is_ollama_available():
        pytest.skip("Ollama not running")


class TestVLMIntegration:
    def test_extract_full_check(self):
        from micr.api import MICRExtractor

        if not CHECK_IMAGE.exists():
            pytest.skip("Check.jpg not found")

        extractor = MICRExtractor(use_vlm=True)
        result = extractor.extract_check(str(CHECK_IMAGE))

        assert result.routing_number is not None
        assert result.legal_amount is not None
        assert result.courtesy_amount is not None
        assert result.confidence > 0.0
        assert result.extraction_method == "vlm"
        print(f"\nFull check result: {result.to_dict()}")

    def test_vlm_on_micr_line(self):
        import cv2

        from micr.engines.vlm_engine import OllamaVLMEngine

        engine = OllamaVLMEngine()
        image = cv2.imread(str(MICR_IMAGE))
        result = engine.extract(image, image_type="micr_line")

        assert result.routing_number is not None
        assert result.confidence > 0.0
        print(f"\nMICR line result: routing={result.routing_number}")

    def test_extract_check_json_output(self):
        from micr.api import MICRExtractor

        if not CHECK_IMAGE.exists():
            pytest.skip("Check.jpg not found")

        extractor = MICRExtractor(use_vlm=True)
        result = extractor.extract_check(str(CHECK_IMAGE))
        d = result.to_dict()

        assert "legal_amount" in d
        assert "courtesy_amount" in d
        assert "extraction_method" in d
        assert "confidence" in d
        assert d["routing_number"] is not None
