#!/usr/bin/env python3
"""Run OCR on the fixed fixture with a local LightOnOCR model directory."""

from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/LightOnOCR-2-1B-ONNX"),
        help="Local model checkpoint directory (default: models/LightOnOCR-2-1B-ONNX).",
    )
    args = parser.parse_args()

    from fast_lightonocr import LightOnOCR

    image = Path("tests/fixtures/SROIE.png")
    model = LightOnOCR.from_pretrained(args.model_dir)
    result = model.process(image)
    print(f"OCR text: {result.text!r}")
    if not result.text.strip():
        raise SystemExit("baseline inference returned empty text")


if __name__ == "__main__":
    main()
