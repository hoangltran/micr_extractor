"""VLM-based check field extraction via Ollama or OpenAI-compatible servers (vLLM)."""

import base64
import json
import time
import urllib.request
import urllib.error
from typing import Optional

import cv2
import numpy as np

from micr.models import VLMExtractionResult

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"

# Model presets: shorthand → (ollama_model, description, disk_size, min_ram)
VLM_MODEL_PRESETS = {
    "3b": (
        "qwen2.5vl:3b",
        "Qwen2.5-VL 3B — fast, fits fully in GPU on 16GB Mac",
        "3.2 GB",
        "~6 GB",
    ),
    "7b": (
        "qwen2.5vl:7b",
        "Qwen2.5-VL 7B — best accuracy/speed balance (default)",
        "6.0 GB",
        "~14 GB",
    ),
    "32b": (
        "qwen2.5vl:32b",
        "Qwen2.5-VL 32B — highest accuracy, needs 64GB+ RAM",
        "21 GB",
        "~40 GB",
    ),
    "72b": (
        "qwen2.5vl:72b",
        "Qwen2.5-VL 72B — maximum accuracy, needs 128GB+ RAM",
        "49 GB",
        "~80 GB",
    ),
    "minicpm": (
        "minicpm-v",
        "MiniCPM-V 2.6 8B — alternative VLM, good OCR",
        "4.9 GB",
        "~10 GB",
    ),
    "moondream": (
        "moondream",
        "Moondream 1.8B — ultra-fast, lower accuracy",
        "1.7 GB",
        "~4 GB",
    ),
    "granite": (
        "granite3.2-vision:2b",
        "Granite 3.2 Vision 2B — document-focused, compact",
        "1.8 GB",
        "~5 GB",
    ),
}


def resolve_model_name(model: str) -> str:
    """Resolve a preset shorthand or full model name to an Ollama model name."""
    if model in VLM_MODEL_PRESETS:
        return VLM_MODEL_PRESETS[model][0]
    return model


def _detect_server_type(base_url: str) -> Optional[str]:
    """Detect whether the server is Ollama or OpenAI-compatible (vLLM).

    Returns:
        "ollama", "openai", or None if no server is reachable.
    """
    # Try Ollama first (GET /api/tags)
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return "ollama"
    except (urllib.error.URLError, TimeoutError, OSError):
        pass

    # Try OpenAI-compatible (GET /v1/models) — used by vLLM, TGI, etc.
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return "openai"
    except (urllib.error.URLError, TimeoutError, OSError):
        pass

    return None


def _get_openai_model_name(base_url: str) -> Optional[str]:
    """Query an OpenAI-compatible server for the loaded model name."""
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
    return None


def is_ollama_available(base_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check if a VLM server (Ollama or OpenAI-compatible) is accessible."""
    return _detect_server_type(base_url) is not None


def is_model_available(
    model: str, base_url: str = DEFAULT_OLLAMA_URL
) -> bool:
    """Check if a specific model is available on the server."""
    server_type = _detect_server_type(base_url)
    if server_type == "ollama":
        try:
            req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                model_names = [m["name"] for m in data.get("models", [])]
                return any(model in name for name in model_names)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False
    elif server_type == "openai":
        name = _get_openai_model_name(base_url)
        return name is not None
    return False


class OllamaVLMEngine:
    """
    VLM-based check extraction engine using Ollama or OpenAI-compatible servers.

    Supports:
    - Ollama (development): POST /api/generate
    - vLLM / OpenAI-compatible (production): POST /v1/chat/completions

    The server type is auto-detected from the URL. Just change the URL
    to switch between Ollama and vLLM — no other code changes needed.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Auto-detect server type
        self._server_type = _detect_server_type(self.base_url)
        if self._server_type is None:
            raise RuntimeError(
                f"No VLM server found at {self.base_url}. "
                "Start Ollama with: ollama serve\n"
                "Or start vLLM with: docker run ... vllm/vllm-openai:latest ..."
            )

        # Resolve model name
        if self._server_type == "openai":
            # For OpenAI-compatible servers, query the loaded model name
            self.model = _get_openai_model_name(self.base_url) or model
        else:
            self.model = resolve_model_name(model)

    @property
    def name(self) -> str:
        return f"vlm_{self._server_type}({self.model})"

    def extract(
        self, image: np.ndarray, image_type: str = "full_check"
    ) -> VLMExtractionResult:
        """
        Extract check fields from an image using the VLM.

        Args:
            image: BGR or grayscale image (numpy array).
            image_type: "full_check" or "micr_line" (adjusts prompt).

        Returns:
            VLMExtractionResult with extracted fields.
        """
        image_b64 = self._encode_image(image)
        prompt = self._build_prompt(image_type)

        start_time = time.monotonic()
        raw_response = self._call_vlm(prompt, image_b64)
        latency_ms = (time.monotonic() - start_time) * 1000

        result = self._parse_response(raw_response)
        result.model_name = self.model
        result.latency_ms = latency_ms

        return result

    @staticmethod
    def _normalize_orientation(image: np.ndarray) -> np.ndarray:
        """Rotate portrait check images to landscape orientation.

        Checks are always wider than tall. If the image is portrait
        (height > width), it was photographed with the camera rotated.
        We rotate it to landscape so the VLM sees text in normal
        reading orientation.
        """
        h, w = image.shape[:2]
        if h <= w:
            return image

        # Portrait image — rotate to landscape.
        # Try both rotations and pick the one where the bottom edge
        # has more dark content (MICR line is always at the bottom).
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) == 3
            else image
        )
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Rotate CW: left edge → top, bottom edge → left, right → bottom, top → right
        # Rotate CCW: left edge → bottom, top edge → left, right → top, bottom → right
        # After rotation, check which has more ink at the bottom (MICR line location).
        cw = cv2.rotate(thresh, cv2.ROTATE_90_CLOCKWISE)
        ccw = cv2.rotate(thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Sum the bottom 15% of rows for each rotation
        stripe = max(1, cw.shape[0] // 7)
        cw_bottom = cw[-stripe:, :].sum()
        ccw_bottom = ccw[-stripe:, :].sum()

        if cw_bottom >= ccw_bottom:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """Encode image as base64 JPEG string for the VLM API."""
        image = OllamaVLMEngine._normalize_orientation(image)

        h, w = image.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        success, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            raise ValueError("Failed to encode image as JPEG")
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    @staticmethod
    def _build_prompt(image_type: str) -> str:
        """Build the extraction prompt for the VLM."""
        micr_standard = (
            "HOW TO READ THE MICR LINE (ANSI X9.13 standard):\n"
            "The MICR line is printed at the very BOTTOM of the check in "
            "E-13B magnetic ink font. It contains digits 0-9 and four "
            "special symbols that act as field delimiters:\n\n"
            "Special symbols (these are NOT digits — do not read them "
            "as numbers):\n"
            "  ⑈ Transit symbol — looks like two small dots with a bar "
            "between them (resembles ⑈ or |: or :). It is NOT the "
            "digit 1. Encloses the routing number.\n"
            "  ⑆ On-Us symbol — looks like a single vertical bar "
            "(resembles | or ⑆). It is NOT the digit 1. Separates "
            "account number from check number.\n"
            "  ⑇ Amount symbol — four vertical dots. Not a digit.\n"
            "  ⑉ Dash symbol — a horizontal bar. Not a digit.\n\n"
            "WARNING: The ⑈ and ⑆ symbols look similar to the digit 1 "
            "but they are delimiters, NOT digits. Do not include them "
            "in any number field. The routing number must be EXACTLY "
            "9 digits.\n\n"
            "MICR LINE LAYOUT (read left to right):\n"
            "  ⑈RRRRRRRRR⑈  AAAAAAAAAA⑆CCCC\n"
            "  |         |  |          | |  |\n"
            "  |         |  |  Account | | Check number\n"
            "  |         |  |  number  | On-Us separator\n"
            "  | Routing |  (On-Us field)\n"
            "  Transit symbols\n\n"
            "Rules:\n"
            "- Routing number: EXACTLY 9 digits between the two ⑈ "
            "symbols in the MICR line at the BOTTOM of the check.\n"
            "- Do NOT read the fractional routing printed near the top "
            "(e.g. 63-8413/2670). Only use the MICR line at the bottom.\n"
            "- On-Us field: everything after the second ⑈, split by ⑆ "
            "into account number and check number. They are SEPARATE "
            "fields.\n"
            "- The check number is typically the shorter number.\n"
            "- Common formats: ACCOUNT⑆CHECK or CHECK⑆ACCOUNT⑆\n\n"
            "Examples:\n"
            "  ⑈021000021⑈ 123456789⑆0001234\n"
            "    routing=021000021 account=123456789 check=0001234\n"
            "  ⑈063000047⑈ 001500291567⑆5030\n"
            "    routing=063000047 account=001500291567 check=5030\n\n"
        )

        orientation_note = (
            "NOTE: The check image may have been photographed in portrait "
            "orientation (rotated 90°). If the text appears sideways, "
            "mentally rotate it to the standard landscape orientation "
            "before reading.\n\n"
        )

        if image_type == "micr_line":
            return (
                "This is a MICR E-13B line from a bank check.\n\n"
                + micr_standard
                + "Extract the following fields and return ONLY valid "
                "JSON:\n"
                "{\n"
                '  "routing_number": "exactly 9 digits between ⑈ '
                'symbols",\n'
                '  "account_number": "digits before the ⑆ separator",\n'
                '  "check_number": "digits after the ⑆ separator",\n'
                '  "confidence": 0.0 to 1.0,\n'
                '  "notes": "any issues or observations"\n'
                "}\n"
                "Return ONLY the JSON object, no other text."
            )
        return (
            "This is a bank check image. Extract the fields below.\n\n"
            + orientation_note
            + micr_standard
            + "VALIDATION:\n"
            "- Compare the legal amount (written in words) against the "
            "courtesy amount (numeric). Do they represent the same "
            "dollar value?\n"
            "- If the legal amount appears misspelled, still extract it "
            "exactly as written but note the likely intended amount.\n"
            "- Rate your overall confidence from 0.0 to 1.0.\n\n"
            "Return ONLY valid JSON with these exact keys:\n"
            "{\n"
            '  "routing_number": "9 digits from MICR line at bottom",\n'
            '  "account_number": "from MICR line, before ⑆",\n'
            '  "check_number": "from MICR line, after ⑆",\n'
            '  "legal_amount": "written amount exactly as printed",\n'
            '  "courtesy_amount": "numeric amount (e.g. 1000.00)",\n'
            '  "amounts_match": true or false,\n'
            '  "confidence": 0.0 to 1.0,\n'
            '  "notes": "misspellings, unclear text, or other issues"\n'
            "}\n"
            "If a field is not visible or readable, use null. "
            "Return ONLY the JSON object, no other text."
        )

    def _call_vlm(self, prompt: str, image_b64: str) -> str:
        """Call the VLM server, auto-dispatching to the correct API format."""
        if self._server_type == "openai":
            return self._call_openai(prompt, image_b64)
        return self._call_ollama(prompt, image_b64)

    def _call_ollama(self, prompt: str, image_b64: str) -> str:
        """Call Ollama /api/generate endpoint."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 512,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read())
                return response_data.get("response", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama API call failed: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(
                f"VLM timed out after {self.timeout}s. "
                "VLM inference on CPU can be slow. "
                "Consider increasing timeout or using a smaller model."
            ) from e

    def _call_openai(self, prompt: str, image_b64: str) -> str:
        """Call OpenAI-compatible /v1/chat/completions endpoint (vLLM, TGI)."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = json.loads(resp.read())
                choices = response_data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
        except urllib.error.URLError as e:
            raise RuntimeError(f"VLM API call failed: {e}") from e
        except TimeoutError as e:
            raise RuntimeError(
                f"VLM timed out after {self.timeout}s. "
                "Consider increasing timeout or using a smaller model."
            ) from e

    def _parse_response(self, raw_response: str) -> VLMExtractionResult:
        """Parse VLM JSON response into VLMExtractionResult.

        Returns the VLM output as-is without post-processing cleanup.
        If the VLM can't extract fields correctly, that's reflected in
        a low confidence score rather than hidden by cleanup heuristics.
        """
        json_str = self._extract_json(raw_response)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return VLMExtractionResult(
                confidence=0.0,
                notes=f"Failed to parse VLM response as JSON: {raw_response[:200]}",
            )

        # Parse confidence from VLM's self-assessment
        confidence = 0.0
        try:
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            pass

        return VLMExtractionResult(
            routing_number=data.get("routing_number"),
            account_number=data.get("account_number"),
            check_number=data.get("check_number"),
            legal_amount=data.get("legal_amount"),
            courtesy_amount=data.get("courtesy_amount"),
            amounts_match=data.get("amounts_match"),
            raw_micr_line=data.get("raw_micr_line"),
            confidence=confidence,
            notes=data.get("notes"),
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON object from potentially wrapped text."""
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            return text[brace_start : brace_end + 1]
        return text
