"""Hybrid merging of VLM and template engine results."""

from typing import Optional

from micr.models import CheckResult, MICRResult, VLMExtractionResult
from micr.parsing.validator import validate_routing_number


def merge_vlm_and_template(
    template_result: MICRResult,
    vlm_result: Optional[VLMExtractionResult],
) -> CheckResult:
    """
    Merge VLM extraction with template engine result.

    VLM-first approach: VLM drives all fields directly.
    Template engine result is kept for reference and routing validation.
    No cleanup or post-processing is applied to VLM output.
    """
    if vlm_result is None:
        # No VLM — fall back to template-only for MICR fields
        return CheckResult(
            routing_number=template_result.routing_number,
            account_number=template_result.account_number,
            check_number=template_result.check_number,
            confidence=template_result.overall_confidence,
            extraction_method="template",
            micr=template_result,
            warnings=template_result.warnings.copy(),
        )

    # VLM-first: use VLM output directly
    warnings = []

    # Cross-reference with template for informational warnings
    if template_result.routing_valid and vlm_result.routing_number:
        if vlm_result.routing_number != template_result.routing_number:
            warnings.append(
                f"VLM routing ({vlm_result.routing_number}) differs from "
                f"template ({template_result.routing_number})"
            )

    # Check VLM routing validity for informational purposes
    vlm_routing_valid = validate_routing_number(vlm_result.routing_number)
    if vlm_result.routing_number and not vlm_routing_valid:
        warnings.append(
            f"VLM routing ({vlm_result.routing_number}) fails checksum"
        )

    if vlm_result.amounts_match is False:
        warnings.append("Legal amount and courtesy amount do not match")

    return CheckResult(
        routing_number=vlm_result.routing_number,
        account_number=vlm_result.account_number,
        check_number=vlm_result.check_number,
        legal_amount=vlm_result.legal_amount,
        courtesy_amount=vlm_result.courtesy_amount,
        amounts_match=vlm_result.amounts_match,
        confidence=vlm_result.confidence,
        extraction_method="vlm",
        notes=vlm_result.notes,
        micr=template_result,
        vlm_result=vlm_result,
        warnings=warnings,
    )
