"""Tests for VLM engine and hybrid merging -- uses mocked Ollama responses."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from micr.models import (
    CheckResult,
    ImageSource,
    MICRResult,
    VLMExtractionResult,
)

SAMPLE_IMAGE = Path(__file__).parent.parent / "MICR.png"


# ---------------------------------------------------------------------------
# VLM engine unit tests (no Ollama needed)
# ---------------------------------------------------------------------------


class TestOllamaVLMParsing:
    """Test JSON parsing without a running Ollama."""

    def _make_engine(self, server_type="ollama"):
        """Create an engine instance bypassing __init__ connectivity check."""
        from micr.engines.vlm_engine import OllamaVLMEngine

        engine = OllamaVLMEngine.__new__(OllamaVLMEngine)
        engine.model = "test-model"
        engine.base_url = "http://localhost:11434"
        engine.timeout = 10.0
        engine._server_type = server_type
        return engine

    def test_parse_clean_json(self):
        engine = self._make_engine()
        raw = (
            '{"routing_number": "267084131", "account_number": "790319013", '
            '"check_number": "1024", "legal_amount": "One thousand dollars", '
            '"courtesy_amount": "1000.00", "amounts_match": true, '
            '"confidence": 0.85, "notes": null}'
        )
        result = engine._parse_response(raw)
        assert result.routing_number == "267084131"
        assert result.account_number == "790319013"
        assert result.check_number == "1024"
        assert result.legal_amount == "One thousand dollars"
        assert result.courtesy_amount == "1000.00"
        assert result.amounts_match is True
        assert result.confidence == 0.85
        assert result.notes is None

    def test_parse_no_cleanup_applied(self):
        """VLM output is returned as-is, no digit extraction or amount cleaning."""
        engine = self._make_engine()
        raw = (
            '{"routing_number": "⑈267084131⑈", '
            '"account_number": "790-319-013", '
            '"check_number": "1024", '
            '"courtesy_amount": "$1,000.00", '
            '"confidence": 0.7}'
        )
        result = engine._parse_response(raw)
        # Values returned exactly as VLM gave them — no cleanup
        assert result.routing_number == "⑈267084131⑈"
        assert result.account_number == "790-319-013"
        assert result.courtesy_amount == "$1,000.00"

    def test_parse_markdown_wrapped_json(self):
        engine = self._make_engine()
        raw = (
            'Here is the result:\n```json\n'
            '{"routing_number": "267084131", "account_number": "790319013", '
            '"confidence": 0.9}\n'
            '```\n'
        )
        result = engine._parse_response(raw)
        assert result.routing_number == "267084131"
        assert result.account_number == "790319013"
        assert result.confidence == 0.9

    def test_parse_malformed_response(self):
        engine = self._make_engine()
        result = engine._parse_response("Sorry, I cannot read this image.")
        assert result.confidence == 0.0
        assert result.routing_number is None
        assert result.notes is not None  # Should contain error info

    def test_parse_vlm_confidence(self):
        """VLM self-reported confidence is used."""
        engine = self._make_engine()
        raw = '{"routing_number": "267084131", "confidence": 0.42}'
        result = engine._parse_response(raw)
        assert result.confidence == 0.42

    def test_parse_confidence_clamped(self):
        """Confidence is clamped to [0.0, 1.0]."""
        engine = self._make_engine()
        raw = '{"routing_number": "267084131", "confidence": 1.5}'
        result = engine._parse_response(raw)
        assert result.confidence == 1.0

    def test_parse_amounts_match_false(self):
        engine = self._make_engine()
        raw = (
            '{"legal_amount": "Five hundred dollars", '
            '"courtesy_amount": "1000.00", '
            '"amounts_match": false, '
            '"confidence": 0.6, '
            '"notes": "Legal amount says 500 but courtesy says 1000"}'
        )
        result = engine._parse_response(raw)
        assert result.amounts_match is False
        assert result.notes is not None

    def test_extract_json_from_text(self):
        from micr.engines.vlm_engine import OllamaVLMEngine

        assert OllamaVLMEngine._extract_json('blah {"a": 1} blah') == '{"a": 1}'

    def test_extract_json_markdown_code_block(self):
        from micr.engines.vlm_engine import OllamaVLMEngine

        text = 'prefix\n```json\n{"key": "val"}\n```\npostfix'
        assert OllamaVLMEngine._extract_json(text) == '{"key": "val"}'


# ---------------------------------------------------------------------------
# Hybrid merge tests
# ---------------------------------------------------------------------------


class TestHybridMerge:
    """Test VLM-first merging with template as reference."""

    @staticmethod
    def _make_template(routing="267084131", valid=True, confidence=0.95):
        return MICRResult(
            raw_micr_line="⑈267084131⑈790319013⑆1024",
            routing_number=routing,
            account_number="790319013",
            check_number="1024",
            routing_valid=valid,
            overall_confidence=confidence,
            image_source=ImageSource.SCANNER,
            characters=[],
            warnings=[],
        )

    @staticmethod
    def _make_vlm(routing="267084131", confidence=0.85):
        return VLMExtractionResult(
            routing_number=routing,
            account_number="790319013",
            check_number="1024",
            legal_amount="One thousand dollars and 00/100 cent",
            courtesy_amount="1000.00",
            amounts_match=True,
            confidence=confidence,
            notes=None,
            model_name="test",
            latency_ms=500.0,
        )

    def test_vlm_drives_output(self):
        from micr.parsing.hybrid import merge_vlm_and_template

        result = merge_vlm_and_template(self._make_template(), self._make_vlm())
        assert isinstance(result, CheckResult)
        assert result.routing_number == "267084131"
        assert result.legal_amount == "One thousand dollars and 00/100 cent"
        assert result.courtesy_amount == "1000.00"
        assert result.amounts_match is True
        assert result.extraction_method == "vlm"
        assert result.confidence == 0.85

    def test_vlm_none_fallback(self):
        from micr.parsing.hybrid import merge_vlm_and_template

        result = merge_vlm_and_template(self._make_template(), None)
        assert result.extraction_method == "template"
        assert result.legal_amount is None
        assert result.courtesy_amount is None
        assert result.routing_number == "267084131"

    def test_vlm_routing_differs_from_template(self):
        from micr.parsing.hybrid import merge_vlm_and_template

        template = self._make_template(routing="267084131", valid=True)
        vlm = self._make_vlm(routing="063000047")
        result = merge_vlm_and_template(template, vlm)
        # VLM-first: VLM routing is used
        assert result.routing_number == "063000047"
        assert result.extraction_method == "vlm"
        assert any("differs" in w for w in result.warnings)

    def test_vlm_invalid_routing_warning(self):
        from micr.parsing.hybrid import merge_vlm_and_template

        template = self._make_template(routing="111111111", valid=False)
        vlm = self._make_vlm(routing="222222222")
        result = merge_vlm_and_template(template, vlm)
        # VLM output used as-is, but warning about invalid checksum
        assert result.routing_number == "222222222"
        assert any("fails checksum" in w for w in result.warnings)

    def test_amounts_mismatch_warning(self):
        from micr.parsing.hybrid import merge_vlm_and_template

        vlm = self._make_vlm()
        vlm.amounts_match = False
        vlm.notes = "Legal says 500, courtesy says 1000"
        result = merge_vlm_and_template(self._make_template(), vlm)
        assert any("do not match" in w for w in result.warnings)
        assert result.notes is not None

    def test_vlm_confidence_passed_through(self):
        from micr.parsing.hybrid import merge_vlm_and_template

        vlm = self._make_vlm(confidence=0.42)
        result = merge_vlm_and_template(self._make_template(), vlm)
        assert result.confidence == 0.42


# ---------------------------------------------------------------------------
# CheckExtractor integration test with mocked VLM
# ---------------------------------------------------------------------------


class TestCheckExtractorMocked:
    """Test extract_check() with a mocked VLM engine."""

    def test_extract_check_with_mock_vlm(self):
        from micr.api import MICRExtractor

        extractor = MICRExtractor(use_vlm=False)

        mock_engine = MagicMock()
        mock_engine.extract.return_value = VLMExtractionResult(
            routing_number="267084131",
            account_number="790319013",
            check_number="1024",
            legal_amount="One thousand dollars and 00/100 cent",
            courtesy_amount="1000.00",
            amounts_match=True,
            confidence=0.85,
            notes=None,
            model_name="test-model",
            latency_ms=500.0,
        )
        extractor._vlm_engine = mock_engine

        result = extractor.extract_check(str(SAMPLE_IMAGE))
        assert isinstance(result, CheckResult)
        assert result.routing_number == "267084131"
        assert result.account_number == "790319013"
        assert result.legal_amount == "One thousand dollars and 00/100 cent"
        assert result.courtesy_amount == "1000.00"
        assert result.confidence == 0.85
        assert result.extraction_method == "vlm"

    def test_extract_check_no_vlm_fallback(self):
        from micr.api import MICRExtractor

        extractor = MICRExtractor(use_vlm=False)
        result = extractor.extract_check(str(SAMPLE_IMAGE))
        assert isinstance(result, CheckResult)
        assert result.routing_number == "267084131"
        assert result.extraction_method == "template"
        assert result.legal_amount is None
        assert result.courtesy_amount is None

    def test_extract_check_vlm_failure_graceful(self):
        from micr.api import MICRExtractor

        extractor = MICRExtractor(use_vlm=False)

        mock_engine = MagicMock()
        mock_engine.extract.side_effect = RuntimeError("Ollama timeout")
        extractor._vlm_engine = mock_engine

        # Should not raise — falls back to template-only
        result = extractor.extract_check(str(SAMPLE_IMAGE))
        assert isinstance(result, CheckResult)
        assert result.routing_number == "267084131"
        assert result.extraction_method == "template"


# ---------------------------------------------------------------------------
# Server type detection tests
# ---------------------------------------------------------------------------


class TestServerDetection:
    """Test auto-detection of Ollama vs OpenAI-compatible servers."""

    def test_detect_ollama(self):
        from micr.engines.vlm_engine import _detect_server_type

        with patch("micr.engines.vlm_engine.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = _detect_server_type("http://localhost:11434")
            assert result == "ollama"

    def test_detect_openai(self):
        from micr.engines.vlm_engine import _detect_server_type

        def mock_urlopen(req, **kwargs):
            # Ollama endpoint fails, OpenAI endpoint succeeds
            if "/api/tags" in req.full_url:
                raise urllib.error.URLError("Connection refused")
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        import urllib.error

        with patch("micr.engines.vlm_engine.urllib.request.urlopen", side_effect=mock_urlopen):
            result = _detect_server_type("http://localhost:8000")
            assert result == "openai"

    def test_detect_no_server(self):
        from micr.engines.vlm_engine import _detect_server_type

        import urllib.error

        with patch(
            "micr.engines.vlm_engine.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            result = _detect_server_type("http://localhost:9999")
            assert result is None

    def test_engine_name_ollama(self):
        engine = TestOllamaVLMParsing()._make_engine(server_type="ollama")
        assert engine.name == "vlm_ollama(test-model)"

    def test_engine_name_openai(self):
        engine = TestOllamaVLMParsing()._make_engine(server_type="openai")
        assert engine.name == "vlm_openai(test-model)"

    def test_call_vlm_dispatches_to_ollama(self):
        """_call_vlm dispatches to _call_ollama for Ollama servers."""
        engine = TestOllamaVLMParsing()._make_engine(server_type="ollama")
        engine._call_ollama = MagicMock(return_value='{"routing_number": "267084131"}')
        engine._call_openai = MagicMock()

        engine._call_vlm("prompt", "image_b64")
        engine._call_ollama.assert_called_once_with("prompt", "image_b64")
        engine._call_openai.assert_not_called()

    def test_call_vlm_dispatches_to_openai(self):
        """_call_vlm dispatches to _call_openai for OpenAI-compatible servers."""
        engine = TestOllamaVLMParsing()._make_engine(server_type="openai")
        engine._call_ollama = MagicMock()
        engine._call_openai = MagicMock(return_value='{"routing_number": "267084131"}')

        engine._call_vlm("prompt", "image_b64")
        engine._call_openai.assert_called_once_with("prompt", "image_b64")
        engine._call_ollama.assert_not_called()
