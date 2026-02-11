#!/usr/bin/env python3
"""
Generate MICR E-13B reference character templates.

E-13B characters are drawn programmatically based on the standard
character specifications. Each character occupies a 0.117" x 0.091" cell
with specific bar patterns.

The 14 characters are: 0-9, Transit, On-Us, Amount, Dash.
"""

import os
from pathlib import Path

import cv2
import numpy as np

# Template dimensions (pixels at reference scale)
CHAR_HEIGHT = 80
CHAR_WIDTH = 56  # E-13B characters are roughly 0.117" wide by 0.091" tall

TEMPLATES_DIR = Path(__file__).parent.parent / "micr" / "resources" / "templates"


def _draw_bars(canvas: np.ndarray, bars: list[tuple[int, int, int, int]]):
    """Draw rectangular bars on canvas. Each bar is (x, y, w, h)."""
    for x, y, w, h in bars:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), 255, -1)


def _make_template(bars: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Create a template image from bar specifications."""
    canvas = np.zeros((CHAR_HEIGHT, CHAR_WIDTH), dtype=np.uint8)
    _draw_bars(canvas, bars)
    return canvas


def generate_digit_0() -> np.ndarray:
    """MICR digit 0 - rectangular frame with openings at top-right and bottom-left."""
    bars = [
        (8, 0, 8, CHAR_HEIGHT),       # left vertical bar
        (40, 0, 8, CHAR_HEIGHT),       # right vertical bar
        (8, 0, 40, 8),                 # top horizontal bar
        (8, CHAR_HEIGHT - 8, 40, 8),   # bottom horizontal bar
    ]
    return _make_template(bars)


def generate_digit_1() -> np.ndarray:
    """MICR digit 1 - single thick vertical bar."""
    bars = [
        (22, 0, 12, CHAR_HEIGHT),  # center vertical bar
    ]
    return _make_template(bars)


def generate_digit_2() -> np.ndarray:
    """MICR digit 2."""
    bars = [
        (8, 0, 40, 8),                # top horizontal
        (40, 0, 8, 40),               # top-right vertical (upper half)
        (8, 36, 40, 8),               # middle horizontal
        (8, 36, 8, 44),               # bottom-left vertical (lower half)
        (8, CHAR_HEIGHT - 8, 40, 8),  # bottom horizontal
    ]
    return _make_template(bars)


def generate_digit_3() -> np.ndarray:
    """MICR digit 3."""
    bars = [
        (8, 0, 40, 8),                # top horizontal
        (40, 0, 8, CHAR_HEIGHT),       # right vertical (full height)
        (8, 36, 40, 8),               # middle horizontal
        (8, CHAR_HEIGHT - 8, 40, 8),  # bottom horizontal
    ]
    return _make_template(bars)


def generate_digit_4() -> np.ndarray:
    """MICR digit 4."""
    bars = [
        (8, 0, 8, 44),                # left vertical (upper half + middle)
        (8, 36, 40, 8),               # middle horizontal
        (40, 0, 8, CHAR_HEIGHT),       # right vertical (full height)
    ]
    return _make_template(bars)


def generate_digit_5() -> np.ndarray:
    """MICR digit 5."""
    bars = [
        (8, 0, 40, 8),                # top horizontal
        (8, 0, 8, 44),                # top-left vertical (upper half)
        (8, 36, 40, 8),               # middle horizontal
        (40, 36, 8, 44),              # bottom-right vertical (lower half)
        (8, CHAR_HEIGHT - 8, 40, 8),  # bottom horizontal
    ]
    return _make_template(bars)


def generate_digit_6() -> np.ndarray:
    """MICR digit 6."""
    bars = [
        (8, 0, 40, 8),                # top horizontal
        (8, 0, 8, CHAR_HEIGHT),        # left vertical (full height)
        (8, 36, 40, 8),               # middle horizontal
        (40, 36, 8, 44),              # bottom-right vertical (lower half)
        (8, CHAR_HEIGHT - 8, 40, 8),  # bottom horizontal
    ]
    return _make_template(bars)


def generate_digit_7() -> np.ndarray:
    """MICR digit 7."""
    bars = [
        (8, 0, 40, 8),                # top horizontal
        (40, 0, 8, CHAR_HEIGHT),       # right vertical (full height)
    ]
    return _make_template(bars)


def generate_digit_8() -> np.ndarray:
    """MICR digit 8."""
    bars = [
        (8, 0, 8, CHAR_HEIGHT),        # left vertical (full height)
        (40, 0, 8, CHAR_HEIGHT),       # right vertical (full height)
        (8, 0, 40, 8),                # top horizontal
        (8, 36, 40, 8),               # middle horizontal
        (8, CHAR_HEIGHT - 8, 40, 8),  # bottom horizontal
    ]
    return _make_template(bars)


def generate_digit_9() -> np.ndarray:
    """MICR digit 9."""
    bars = [
        (8, 0, 40, 8),                # top horizontal
        (8, 0, 8, 44),                # left vertical (upper half)
        (40, 0, 8, CHAR_HEIGHT),       # right vertical (full height)
        (8, 36, 40, 8),               # middle horizontal
        (8, CHAR_HEIGHT - 8, 40, 8),  # bottom horizontal
    ]
    return _make_template(bars)


def generate_transit() -> np.ndarray:
    """MICR Transit symbol ⑈ - vertical bar followed by two dots (colon-like)."""
    bars = [
        (8, 0, 10, CHAR_HEIGHT),       # tall left vertical bar
        (30, 10, 14, 14),              # upper dot
        (30, 56, 14, 14),              # lower dot
    ]
    return _make_template(bars)


def generate_on_us() -> np.ndarray:
    """MICR On-Us symbol ⑆ - two thin vertical bars."""
    bars = [
        (12, 0, 10, CHAR_HEIGHT),      # left vertical bar
        (34, 0, 10, CHAR_HEIGHT),      # right vertical bar
    ]
    return _make_template(bars)


def generate_amount() -> np.ndarray:
    """MICR Amount symbol ⑇ - vertical bar with horizontal extensions."""
    bars = [
        (18, 0, 20, CHAR_HEIGHT),      # thick center vertical bar
        (4, 20, 48, 10),               # middle horizontal bar
        (4, 50, 48, 10),               # lower horizontal bar
    ]
    return _make_template(bars)


def generate_dash() -> np.ndarray:
    """MICR Dash symbol ⑉ - single horizontal bar in the middle."""
    bars = [
        (4, 32, 48, 16),              # center horizontal bar
    ]
    return _make_template(bars)


def generate_all_templates() -> dict[str, np.ndarray]:
    """Generate all 14 MICR E-13B character templates."""
    generators = {
        "0": generate_digit_0,
        "1": generate_digit_1,
        "2": generate_digit_2,
        "3": generate_digit_3,
        "4": generate_digit_4,
        "5": generate_digit_5,
        "6": generate_digit_6,
        "7": generate_digit_7,
        "8": generate_digit_8,
        "9": generate_digit_9,
        "transit": generate_transit,
        "on_us": generate_on_us,
        "amount": generate_amount,
        "dash": generate_dash,
    }

    templates = {}
    for name, gen_func in generators.items():
        templates[name] = gen_func()

    return templates


def save_templates(output_dir: Path | None = None):
    """Generate and save all templates to disk."""
    if output_dir is None:
        output_dir = TEMPLATES_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    templates = generate_all_templates()
    for name, template in templates.items():
        filepath = output_dir / f"{name}.png"
        cv2.imwrite(str(filepath), template)
        print(f"Saved: {filepath}")

    print(f"\nGenerated {len(templates)} templates in {output_dir}")
    return templates


if __name__ == "__main__":
    save_templates()
