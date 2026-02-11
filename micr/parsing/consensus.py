"""Multi-engine consensus for MICR recognition results."""

from micr.models import CharacterResult


def resolve_consensus(
    primary: list[CharacterResult],
    secondary: list[CharacterResult],
) -> list[CharacterResult]:
    """
    Merge results from primary (template matching) and secondary (Tesseract) engines.

    Strategy:
    - Align results by bounding box position overlap
    - For symbols: always prefer primary (template matching handles symbols better)
    - For digits with agreement: boost confidence
    - For digits with disagreement: prefer higher confidence, with penalty

    Args:
        primary: Results from template matching engine.
        secondary: Results from Tesseract engine.

    Returns:
        Merged character results.
    """
    if not secondary:
        return primary
    if not primary:
        return secondary

    aligned = _align_by_position(primary, secondary)

    merged = []
    for p_char, s_char in aligned:
        if p_char is None and s_char is not None:
            merged.append(s_char)
        elif s_char is None and p_char is not None:
            merged.append(p_char)
        elif p_char is not None and s_char is not None:
            merged.append(_merge_pair(p_char, s_char))

    return merged


def _align_by_position(
    primary: list[CharacterResult],
    secondary: list[CharacterResult],
) -> list[tuple[CharacterResult | None, CharacterResult | None]]:
    """
    Align character results from two engines by their horizontal position.

    Uses bounding box x-coordinate overlap to match characters.
    """
    pairs: list[tuple[CharacterResult | None, CharacterResult | None]] = []

    secondary_used = set()

    for p in primary:
        best_match = None
        best_overlap = 0

        for j, s in enumerate(secondary):
            if j in secondary_used:
                continue

            overlap = _bbox_x_overlap(p.bbox, s.bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = j

        if best_match is not None and best_overlap > 0.3:
            pairs.append((p, secondary[best_match]))
            secondary_used.add(best_match)
        else:
            pairs.append((p, None))

    # Add unmatched secondary results
    for j, s in enumerate(secondary):
        if j not in secondary_used:
            pairs.append((None, s))

    # Sort by x position
    pairs.sort(key=lambda pair: _get_x(pair))

    return pairs


def _bbox_x_overlap(
    bbox1: tuple[int, int, int, int],
    bbox2: tuple[int, int, int, int],
) -> float:
    """Compute the x-axis overlap ratio between two bounding boxes."""
    x1_start, _, w1, _ = bbox1
    x2_start, _, w2, _ = bbox2

    x1_end = x1_start + w1
    x2_end = x2_start + w2

    overlap_start = max(x1_start, x2_start)
    overlap_end = min(x1_end, x2_end)

    if overlap_end <= overlap_start:
        return 0.0

    overlap_width = overlap_end - overlap_start
    min_width = min(w1, w2)

    return overlap_width / max(min_width, 1)


def _merge_pair(
    p: CharacterResult, s: CharacterResult
) -> CharacterResult:
    """Merge a pair of matched character results."""
    # Symbols: always prefer template matching (primary)
    if p.character in ("transit", "on_us", "amount", "dash"):
        return CharacterResult(
            character=p.character,
            confidence=p.confidence,
            bbox=p.bbox,
            engine="consensus(template)",
        )

    # Both are digits
    if p.character == s.character:
        # Agreement — boost confidence
        boosted = min(1.0, (p.confidence + s.confidence) / 1.5)
        return CharacterResult(
            character=p.character,
            confidence=boosted,
            bbox=p.bbox,
            engine="consensus(agree)",
        )

    # Disagreement — pick higher confidence with penalty
    if p.confidence >= s.confidence:
        winner = p
        engine_note = "consensus(template_wins)"
    else:
        winner = s
        engine_note = "consensus(tesseract_wins)"

    return CharacterResult(
        character=winner.character,
        confidence=winner.confidence * 0.8,  # Penalty for disagreement
        bbox=winner.bbox,
        engine=engine_note,
    )


def _get_x(pair: tuple[CharacterResult | None, CharacterResult | None]) -> int:
    """Get x position from a pair for sorting."""
    p, s = pair
    if p is not None:
        return p.bbox[0]
    if s is not None:
        return s.bbox[0]
    return 0
