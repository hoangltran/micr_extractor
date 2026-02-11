"""Abstract base class for MICR recognition engines."""

from abc import ABC, abstractmethod

import numpy as np

from micr.models import CharacterResult


class BaseMICREngine(ABC):
    """Base interface for MICR character recognition engines."""

    @abstractmethod
    def recognize(self, micr_line_image: np.ndarray) -> list[CharacterResult]:
        """
        Recognize characters in a preprocessed MICR line image.

        Args:
            micr_line_image: Binary image of the MICR line (text=white, bg=black).

        Returns:
            List of CharacterResult, ordered left to right.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier name."""
