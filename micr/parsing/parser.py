"""Parse recognized MICR characters into structured fields."""

from micr.models import CharacterResult, MICRResult, ImageSource, CHAR_DISPLAY
from micr.parsing.validator import validate_routing_number


def parse_micr_line(
    characters: list[CharacterResult],
    image_source: ImageSource = ImageSource.UNKNOWN,
) -> MICRResult:
    """
    Parse a sequence of recognized characters into structured MICR fields.

    Standard US check MICR E-13B format:
        ⑈ROUTING⑈  ACCOUNT⑆CHECK_NUMBER
    or:
        ⑈ROUTING⑈  CHECK_NUMBER⑆ACCOUNT⑆

    The Transit symbols (⑈) always delimit the routing number.
    The On-Us symbol (⑆) separates account and check number fields.
    """
    if not characters:
        return MICRResult(
            raw_micr_line="",
            overall_confidence=0.0,
            image_source=image_source,
            characters=characters,
            warnings=["No characters recognized"],
        )

    raw_line = _format_raw_line(characters)

    # Find transit symbol positions (delimit routing number)
    transit_positions = [
        i for i, c in enumerate(characters) if c.character == "transit"
    ]

    # Find on-us symbol positions (delimit account/check fields)
    on_us_positions = [
        i for i, c in enumerate(characters) if c.character == "on_us"
    ]

    routing_number = None
    account_number = None
    check_number = None
    amount = None

    # Extract routing number (between first pair of transit symbols)
    if len(transit_positions) >= 2:
        routing_chars = characters[transit_positions[0] + 1 : transit_positions[1]]
        routing_number = "".join(c.character for c in routing_chars if c.character.isdigit())

    # Extract account and check number from on-us delimited section
    if transit_positions and on_us_positions:
        # Get the section after the last transit symbol
        after_transit = transit_positions[-1] + 1 if transit_positions else 0

        # Characters between last transit and first on-us (after transit)
        relevant_on_us = [p for p in on_us_positions if p > after_transit]

        if relevant_on_us:
            # Characters between last transit and first on-us = account number
            account_chars = characters[after_transit : relevant_on_us[0]]
            account_number = "".join(
                c.character for c in account_chars if c.character.isdigit()
            )

            # Characters after last on-us = check number
            check_chars = characters[relevant_on_us[-1] + 1 :]
            check_number = "".join(
                c.character for c in check_chars if c.character.isdigit()
            )

            # If check_number is empty, swap (some formats put check first)
            if not check_number and account_number:
                if len(relevant_on_us) >= 2:
                    # Format: ⑈ROUTING⑈ CHECK⑆ACCOUNT⑆
                    first_section = characters[after_transit : relevant_on_us[0]]
                    second_section = characters[relevant_on_us[0] + 1 : relevant_on_us[1]]
                    check_number = "".join(
                        c.character for c in first_section if c.character.isdigit()
                    )
                    account_number = "".join(
                        c.character for c in second_section if c.character.isdigit()
                    )
        else:
            # No on-us after transit — all remaining digits are account
            remaining_chars = characters[after_transit:]
            account_number = "".join(
                c.character for c in remaining_chars if c.character.isdigit()
            )

    elif not transit_positions and on_us_positions:
        # No transit symbols found — try to parse as just account info
        # Everything before first on-us
        first_section = characters[: on_us_positions[0]]
        digits = "".join(c.character for c in first_section if c.character.isdigit())
        if len(digits) >= 9:
            routing_number = digits[:9]
            account_number = digits[9:]
        else:
            account_number = digits

    elif not transit_positions and not on_us_positions:
        # No symbols at all — treat entire sequence as digits
        all_digits = "".join(c.character for c in characters if c.character.isdigit())
        if len(all_digits) >= 9:
            # Assume first 9 digits are routing
            routing_number = all_digits[:9]
            account_number = all_digits[9:]
        else:
            account_number = all_digits

    # Extract amount if present (between amount symbols)
    amount_positions = [
        i for i, c in enumerate(characters) if c.character == "amount"
    ]
    if len(amount_positions) >= 2:
        amount_chars = characters[amount_positions[0] + 1 : amount_positions[1]]
        amount = "".join(c.character for c in amount_chars if c.character.isdigit())

    # Validate routing number
    routing_valid = validate_routing_number(routing_number) if routing_number else False

    # Compute overall confidence
    overall_confidence = _compute_overall_confidence(characters)

    return MICRResult(
        raw_micr_line=raw_line,
        routing_number=routing_number,
        account_number=account_number,
        check_number=check_number,
        amount=amount,
        routing_valid=routing_valid,
        overall_confidence=overall_confidence,
        image_source=image_source,
        characters=characters,
    )


def _format_raw_line(characters: list[CharacterResult]) -> str:
    """Format characters into a human-readable MICR line string."""
    return "".join(CHAR_DISPLAY.get(c.character, c.character) for c in characters)


def _compute_overall_confidence(characters: list[CharacterResult]) -> float:
    """Compute average confidence across all characters."""
    if not characters:
        return 0.0
    return sum(c.confidence for c in characters) / len(characters)
