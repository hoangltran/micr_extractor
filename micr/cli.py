"""Command-line interface for MICR extraction."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="micr-extract",
        description="Extract MICR E-13B text from check images",
    )
    parser.add_argument(
        "image",
        help="Path to the check image file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
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

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    from micr.api import MICRExtractor

    extractor = MICRExtractor(
        use_tesseract=args.tesseract,
        templates_dir=args.templates_dir,
        tessdata_dir=args.tessdata_dir,
        tesseract_lang=args.tesseract_lang,
    )

    result = extractor.extract(str(image_path))

    if args.format == "json":
        output = result.to_dict()
        if args.verbose:
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
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")

        if args.verbose:
            print(f"\nCharacter Details ({len(result.characters)} chars):")
            for i, c in enumerate(result.characters):
                print(
                    f"  [{i:2d}] {c.display:>8s}  "
                    f"conf={c.confidence:.3f}  "
                    f"bbox={c.bbox}  "
                    f"engine={c.engine}"
                )


if __name__ == "__main__":
    main()
