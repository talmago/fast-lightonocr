#!/usr/bin/env python3
"""Download LightOnOCR ONNX models from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "onnx-community/LightOnOCR-2-1B-ONNX"

PRESETS: dict[str, list[str]] = {
    "default": [
        "*.json",
        "onnx/embed_tokens.onnx",
        "onnx/embed_tokens.onnx_data*",
        "onnx/vision_encoder.onnx",
        "onnx/vision_encoder.onnx_data*",
        "onnx/decoder_model_merged.onnx",
        "onnx/decoder_model_merged.onnx_data*",
    ],
    "fp16": [
        "*.json",
        "onnx/embed_tokens.onnx",
        "onnx/embed_tokens.onnx_data*",
        "onnx/vision_encoder.onnx",
        "onnx/vision_encoder.onnx_data*",
        "onnx/decoder_model_merged.onnx",
        "onnx/decoder_model_merged.onnx_data*",
    ],
    "q4": [
        "*.json",
        "onnx/embed_tokens_q4.onnx",
        "onnx/embed_tokens_q4.onnx_data*",
        "onnx/vision_encoder_q4.onnx",
        "onnx/vision_encoder_q4.onnx_data*",
        "onnx/decoder_model_merged_q4.onnx",
        "onnx/decoder_model_merged_q4.onnx_data*",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LightOnOCR ONNX model."
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="default",
        help="Model preset to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/lightonocr"),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Downloading '{args.preset}' model...")

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=PRESETS[args.preset],
    )

    print(f"Model downloaded to {args.output_dir}")


if __name__ == "__main__":
    main()
