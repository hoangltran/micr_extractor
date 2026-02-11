"""Tests for the MICR extraction pipeline."""

from pathlib import Path

import pytest

from micr.api import MICRExtractor
from micr.models import MICRResult
from micr.parsing.validator import validate_routing_number

SAMPLE_IMAGE = Path(__file__).parent.parent / "MICR.png"


@pytest.fixture
def extractor():
    return MICRExtractor(use_tesseract=False)


class TestValidator:
    def test_valid_routing_number(self):
        assert validate_routing_number("267084131") is True

    def test_invalid_routing_number(self):
        assert validate_routing_number("123456789") is False

    def test_short_routing_number(self):
        assert validate_routing_number("12345") is False

    def test_non_digit_routing_number(self):
        assert validate_routing_number("12345678a") is False

    def test_empty_routing_number(self):
        assert validate_routing_number("") is False

    def test_none_routing_number(self):
        assert validate_routing_number(None) is False

    def test_known_valid_routing_numbers(self):
        # Well-known valid ABA routing numbers
        valid = ["021000021", "011401533", "091000019"]
        for rn in valid:
            assert validate_routing_number(rn) is True, f"{rn} should be valid"


class TestMICRExtractor:
    def test_extract_from_file(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        assert isinstance(result, MICRResult)
        assert result.routing_number == "267084131"
        assert result.account_number == "790319013"
        assert result.check_number == "1024"
        assert result.routing_valid is True
        assert result.overall_confidence > 0.90

    def test_routing_number_checksum(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        assert result.routing_valid is True
        assert validate_routing_number(result.routing_number) is True

    def test_confidence_threshold(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        # All characters should have reasonable confidence
        for char in result.characters:
            assert char.confidence > 0.5, (
                f"Character '{char.display}' has low confidence: {char.confidence}"
            )

    def test_character_count(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        # Expected: ⑈267084131⑈ 790319013⑆1024 = 25 characters
        assert len(result.characters) == 25

    def test_symbol_detection(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        symbols = [c for c in result.characters if c.is_symbol]
        # Should have 2 transit + 1 on-us = 3 symbols
        assert len(symbols) == 3
        transit_count = sum(1 for c in symbols if c.character == "transit")
        on_us_count = sum(1 for c in symbols if c.character == "on_us")
        assert transit_count == 2
        assert on_us_count == 1

    def test_to_dict(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        d = result.to_dict()
        assert d["routing_number"] == "267084131"
        assert d["account_number"] == "790319013"
        assert d["check_number"] == "1024"
        assert d["routing_valid"] is True
        assert isinstance(d["overall_confidence"], float)
        assert d["warnings"] == []

    def test_raw_micr_line(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        assert "267084131" in result.raw_micr_line
        assert "790319013" in result.raw_micr_line
        assert "1024" in result.raw_micr_line

    def test_no_warnings(self, extractor):
        result = extractor.extract(str(SAMPLE_IMAGE))
        assert result.warnings == []

    def test_file_not_found(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.extract("nonexistent.png")
