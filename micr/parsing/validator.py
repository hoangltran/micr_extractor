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

    if result.overall_confidence < 0.7:
        warnings.append(
            f"Low overall confidence: {result.overall_confidence:.2f}"
        )

    return warnings
