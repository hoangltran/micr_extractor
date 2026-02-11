"""MICR field validation utilities."""

from micr.models import MICRResult

# ABA routing number checksum weights
_ROUTING_WEIGHTS = [3, 7, 1, 3, 7, 1, 3, 7, 1]


def validate_routing_number(routing: str) -> bool:
    """
    Validate a 9-digit ABA routing number using the modulo-10 checksum.

    The checksum uses weights [3, 7, 1, 3, 7, 1, 3, 7, 1].
    The weighted sum of all 9 digits must be divisible by 10.
    """
    if not routing or len(routing) != 9 or not routing.isdigit():
        return False

    total = sum(int(d) * w for d, w in zip(routing, _ROUTING_WEIGHTS))
    return total % 10 == 0


def validate_micr_result(result: MICRResult) -> list[str]:
    """
    Validate a complete MICR extraction result and return a list of warnings.

    Also penalizes overall_confidence when structural issues are detected
    (e.g. wrong symbol counts, failed routing checksum).
    """
    warnings = []

    if not result.routing_number:
        warnings.append("No routing number detected")
    elif len(result.routing_number) != 9:
        warnings.append(
            f"Routing number has {len(result.routing_number)} digits, expected 9"
        )
    elif not validate_routing_number(result.routing_number):
        warnings.append("Routing number failed checksum validation")

    if not result.account_number:
        warnings.append("No account number detected")
    elif not result.account_number.isdigit():
        warnings.append("Account number contains non-digit characters")

    # Validate expected symbol counts
    transit_count = sum(
        1 for c in result.characters if c.character == "transit"
    )
    on_us_count = sum(
        1 for c in result.characters if c.character == "on_us"
    )

    if transit_count != 2:
        warnings.append(
            f"Expected 2 transit symbols, found {transit_count}"
        )
    if on_us_count == 0:
        warnings.append("No on-us symbol detected")
    elif on_us_count > 2:
        warnings.append(
            f"Expected 1-2 on-us symbols, found {on_us_count}"
        )

    # Penalize confidence for structural issues
    penalty = _compute_confidence_penalty(warnings)
    if penalty > 0:
        result.overall_confidence *= (1.0 - penalty)

    if result.overall_confidence < 0.7:
        warnings.append(
            f"Low overall confidence: {result.overall_confidence:.2f}"
        )

    return warnings


def _compute_confidence_penalty(warnings: list[str]) -> float:
    """
    Compute a confidence penalty based on validation warnings.

    Returns a value between 0.0 (no penalty) and 1.0 (zero confidence).
    Multiple issues stack: each issue adds to the penalty.
    """
    penalty = 0.0

    for w in warnings:
        if "checksum" in w:
            penalty += 0.40
        elif "No routing number" in w:
            penalty += 0.50
        elif "transit symbols" in w:
            penalty += 0.30
        elif "on-us symbol" in w:
            penalty += 0.30
        elif "No account number" in w:
            penalty += 0.20
        elif "digits, expected 9" in w:
            penalty += 0.20

    return min(penalty, 1.0)
