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
