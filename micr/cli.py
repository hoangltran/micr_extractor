"""Command-line interface for MICR extraction."""

import argparse
import json
import sys
from pathlib import Path


def _list_vlm_models():
    """Print available VLM model presets and exit."""
    from micr.engines.vlm_engine import VLM_MODEL_PRESETS

    print("Available VLM models (use with --vlm-model):\n")
    print(f"  {'Preset':<12} {'Ollama Model':<24} {'Disk':<10} {'RAM':<10} Description")
    print(f"  {'------':<12} {'------------':<24} {'----':<10} {'---':<10} -----------")
    for preset, (model, desc, disk, ram) in VLM_MODEL_PRESETS.items():
        default = " *" if preset == "7b" else ""
        print(f"  {preset:<12} {model:<24} {disk:<10} {ram:<10} {desc}{default}")
    print("\n  * = default")
    print("\n  You can also pass any Ollama model name directly,")
    print("  e.g. --vlm-model qwen2.5vl:7b-q4_0")


def main():
    parser = argparse.ArgumentParser(
        prog="micr-extract",
        description="Extract MICR E-13B text from check images",
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the check image file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Extract full check fields (MICR + amounts) using VLM",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Enable VLM engine (requires Ollama running locally)",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Model preset (3b, 7b, 32b, 72b, minicpm, moondream, granite) "
        "or full Ollama model name (default: 7b)",
    )
    parser.add_argument(
        "--list-vlm-models",
        action="store_true",
        help="List available VLM model presets and exit",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--vlm-timeout",
        type=float,
        default=120.0,
        help="VLM inference timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--tesseract",
        action="store_true",
        help="Enable Tesseract as secondary engine",
    )
    parser.add_argument(
        "--tessdata-dir",
        help="Path to Tesseract tessdata directory",
    )
    parser.add_argument(
        "--tesseract-lang",
        default="eng",
        help="Tesseract language code (default: eng)",
    )
    parser.add_argument(
        "--templates-dir",
        help="Path to custom E-13B templates directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-character details",
    )

    args = parser.parse_args()

    if args.list_vlm_models:
        _list_vlm_models()
        sys.exit(0)

    if not args.image:
        parser.error("the following arguments are required: image")

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    from micr.api import MICRExtractor

    extractor = MICRExtractor(
        use_tesseract=args.tesseract,
        use_vlm=args.vlm or args.check,
        vlm_model=args.vlm_model,
        ollama_url=args.ollama_url,
        vlm_timeout=args.vlm_timeout,
        templates_dir=args.templates_dir,
        tessdata_dir=args.tessdata_dir,
        tesseract_lang=args.tesseract_lang,
    )

    if args.check:
        result = extractor.extract_check(str(image_path))
        _print_check_result(result, args.format, args.verbose)
    else:
        result = extractor.extract(str(image_path))
        _print_micr_result(result, args.format, args.verbose)


def _print_micr_result(result, fmt, verbose):
    """Print MICR-only result."""
    if fmt == "json":
        output = result.to_dict()
        if verbose:
            output["characters"] = [
                {
                    "char": c.display,
                    "confidence": round(c.confidence, 4),
                    "bbox": list(c.bbox),
                    "engine": c.engine,
                }
                for c in result.characters
            ]
        print(json.dumps(output, indent=2))
    else:
        print(f"MICR Line:       {result.raw_micr_line}")
        print(f"Routing Number:  {result.routing_number or 'N/A'}")
        print(f"Account Number:  {result.account_number or 'N/A'}")
        print(f"Check Number:    {result.check_number or 'N/A'}")
        if result.amount:
            print(f"Amount:          {result.amount}")
        print(f"Routing Valid:   {result.routing_valid}")
        print(f"Confidence:      {result.overall_confidence:.2%}")
        print(f"Image Source:    {result.image_source.value}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        if verbose:
            print(f"\nCharacter Details ({len(result.characters)} chars):")
            for i, c in enumerate(result.characters):
                print(
                    f"  [{i:2d}] {c.display:>8s}  "
                    f"conf={c.confidence:.3f}  "
                    f"bbox={c.bbox}  "
                    f"engine={c.engine}"
                )


def _print_check_result(result, fmt, verbose):
    """Print full check extraction result."""
    if fmt == "json":
        output = result.to_dict()
        if verbose and result.micr and result.micr.characters:
            output["characters"] = [
                {
                    "char": c.display,
                    "confidence": round(c.confidence, 4),
                    "bbox": list(c.bbox),
                    "engine": c.engine,
                }
                for c in result.micr.characters
            ]
        print(json.dumps(output, indent=2))
    else:
        print(f"Routing Number:   {result.routing_number or 'N/A'}")
        print(f"Account Number:   {result.account_number or 'N/A'}")
        print(f"Check Number:     {result.check_number or 'N/A'}")
        print(f"Legal Amount:     {result.legal_amount or 'N/A'}")
        print(f"Courtesy Amount:  {result.courtesy_amount or 'N/A'}")
        if result.amounts_match is not None:
            print(f"Amounts Match:    {result.amounts_match}")
        print(f"Confidence:       {result.confidence:.2%}")
        print(f"Method:           {result.extraction_method}")

        if result.vlm_result:
            print(f"VLM Model:        {result.vlm_result.model_name}")
            print(f"VLM Latency:      {result.vlm_result.latency_ms:.0f}ms")

        if result.notes:
            print(f"\nNotes: {result.notes}")

        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")


if __name__ == "__main__":
    main()
