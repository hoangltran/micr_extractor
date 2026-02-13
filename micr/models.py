"""Data models for MICR extraction results."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ImageSource(Enum):
    SCANNER = "scanner"
    CAMERA = "camera"
    UNKNOWN = "unknown"


class MICRSymbol(Enum):
    """MICR E-13B special symbols."""

    TRANSIT = "transit"  # ⑈ surrounds routing number
    ON_US = "on_us"  # ⑆ separates account/check fields
    AMOUNT = "amount"  # ⑇ surrounds amount field
    DASH = "dash"  # ⑉ separator within account number

    @classmethod
    def display_char(cls, symbol: "MICRSymbol") -> str:
        mapping = {
            cls.TRANSIT: "⑈",
            cls.ON_US: "⑆",
            cls.AMOUNT: "⑇",
            cls.DASH: "⑉",
        }
        return mapping.get(symbol, "?")


# Character name to display string mapping
CHAR_DISPLAY = {
    "transit": "⑈",
    "on_us": "⑆",
    "amount": "⑇",
    "dash": "⑉",
}

# All valid MICR E-13B character names
VALID_CHARACTERS = set("0123456789") | {"transit", "on_us", "amount", "dash"}


@dataclass
class CharacterResult:
    """Recognition result for a single MICR character."""

    character: str  # '0'-'9' or symbol name ('transit', 'on_us', 'amount', 'dash')
    confidence: float  # 0.0 to 1.0
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in the MICR line ROI
    engine: str  # which engine produced this result

    @property
    def is_symbol(self) -> bool:
        return self.character in CHAR_DISPLAY

    @property
    def display(self) -> str:
        return CHAR_DISPLAY.get(self.character, self.character)


@dataclass
class MICRResult:
    """Complete MICR extraction result."""

    raw_micr_line: str
    routing_number: Optional[str] = None
    account_number: Optional[str] = None
    check_number: Optional[str] = None
    amount: Optional[str] = None
    routing_valid: bool = False
    overall_confidence: float = 0.0
    image_source: ImageSource = ImageSource.UNKNOWN
    characters: list[CharacterResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw_micr_line": self.raw_micr_line,
            "routing_number": self.routing_number,
            "account_number": self.account_number,
            "check_number": self.check_number,
            "amount": self.amount,
            "routing_valid": self.routing_valid,
            "overall_confidence": round(self.overall_confidence, 4),
            "image_source": self.image_source.value,
            "character_count": len(self.characters),
            "warnings": self.warnings,
        }


@dataclass
class VLMExtractionResult:
    """Raw extraction result from a VLM engine.

    Fields are returned as-is from the VLM without post-processing.
    The VLM self-reports its confidence score.
    """

    routing_number: Optional[str] = None
    account_number: Optional[str] = None
    check_number: Optional[str] = None
    legal_amount: Optional[str] = None
    courtesy_amount: Optional[str] = None
    amounts_match: Optional[bool] = None
    raw_micr_line: Optional[str] = None
    confidence: float = 0.0
    notes: Optional[str] = None
    model_name: str = ""
    latency_ms: float = 0.0


@dataclass
class CheckResult:
    """Full check extraction result.

    VLM-first: fields come directly from the VLM without cleanup.
    Template engine result is kept for reference but VLM drives the output.
    """

    routing_number: Optional[str] = None
    account_number: Optional[str] = None
    check_number: Optional[str] = None
    legal_amount: Optional[str] = None
    courtesy_amount: Optional[str] = None
    amounts_match: Optional[bool] = None
    confidence: float = 0.0
    extraction_method: str = "template"
    notes: Optional[str] = None
    micr: Optional[MICRResult] = None
    vlm_result: Optional[VLMExtractionResult] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "routing_number": self.routing_number,
            "account_number": self.account_number,
            "check_number": self.check_number,
            "legal_amount": self.legal_amount,
            "courtesy_amount": self.courtesy_amount,
            "amounts_match": self.amounts_match,
            "confidence": round(self.confidence, 4),
            "extraction_method": self.extraction_method,
            "warnings": self.warnings,
        }
        if self.notes:
            d["notes"] = self.notes
        if self.vlm_result:
            d["vlm_latency_ms"] = round(self.vlm_result.latency_ms, 1)
            d["vlm_model"] = self.vlm_result.model_name
        if self.micr:
            d["routing_valid"] = self.micr.routing_valid
            d["raw_micr_line"] = self.micr.raw_micr_line
        return d
