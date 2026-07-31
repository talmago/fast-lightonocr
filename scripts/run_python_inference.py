#!/usr/bin/env python3
"""
Reference ONNX inference implementation for LightOnOCR.

This script mirrors the Rust implementation as closely as possible and is
intended as the debugging reference during the Rust port.

Example:

    python scripts/inference.py examples/receipt.jpg

or

    python scripts/inference.py \
        examples/receipt.jpg \
        --variant fp16 \
        --verbose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
)


DEFAULT_MODEL_DIR = Path("models/lightonocr")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LightOnOCR ONNX inference.",
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Input image.",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model directory.",
    )

    parser.add_argument(
        "--variant",
        default=None,
        choices=[
            None,
            "fp16",
            "q4",
        ],
        help=(
            "ONNX model variant. Defaults to the unqualified model names "
            "(vision_encoder.onnx, embed_tokens.onnx, decoder_model_merged.onnx)."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    parser.add_argument(
        "--dump-vision",
        action="store_true",
        help="Write vision_python.bin",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def print_tensor_stats(
    name: str,
    tensor: np.ndarray,
    values: int = 100,
) -> None:
    flat = tensor.reshape(-1)

    print(name)
    print(f"  shape : {tensor.shape}")
    print(f"  min   : {float(flat.min())}")
    print(f"  max   : {float(flat.max())}")
    print(f"  mean  : {float(flat.mean())}")

    print("  first :", flat[:values].tolist())
    print("  last  :", flat[-values:].tolist())


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def onnx_filename(stem: str, variant: str | None) -> str:
    if variant is None:
        return f"{stem}.onnx"
    return f"{stem}_{variant}.onnx"


def load_model(model_dir: Path, variant: str | None = None):
    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    generation_config = GenerationConfig.from_pretrained(model_dir)

    vision_model = model_dir / "onnx" / onnx_filename(
        "vision_encoder",
        variant,
    )

    embed_model = model_dir / "onnx" / onnx_filename(
        "embed_tokens",
        variant,
    )

    decoder_model = model_dir / "onnx" / onnx_filename(
        "decoder_model_merged",
        variant,
    )

    providers = ["CPUExecutionProvider"]

    vision_session = ort.InferenceSession(
        str(vision_model),
        providers=providers,
    )

    embed_session = ort.InferenceSession(
        str(embed_model),
        providers=providers,
    )

    decoder_session = ort.InferenceSession(
        str(decoder_model),
        providers=providers,
    )

    if processor.chat_template is None:
        processor.chat_template = processor.tokenizer.chat_template

    return (
        config,
        processor,
        generation_config,
        vision_session,
        embed_session,
        decoder_session,
    )


# ----------------------------------------------------------------------
# Input preparation
# ----------------------------------------------------------------------

def prepare_inputs(
    processor,
    image_path: Path,
):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                }
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    return {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
        "pixel_values": inputs["pixel_values"].numpy(),
    }


# ----------------------------------------------------------------------
# KV cache
# ----------------------------------------------------------------------


def create_empty_kv_cache(
    *,
    batch_size: int,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}

    for layer in range(num_hidden_layers):
        for kind in ("key", "value"):
            cache[f"past_key_values.{layer}.{kind}"] = np.zeros(
                (
                    batch_size,
                    num_key_value_heads,
                    0,
                    head_dim,
                ),
                dtype=np.float32,
            )

    return cache


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------


def run_inference(
    *,
    config,
    processor,
    generation_config,
    vision_session: ort.InferenceSession,
    embed_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    pixel_values: np.ndarray,
    max_new_tokens: int,
    verbose: bool,
    dump_vision: bool,
) -> str:
    if max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")

    text_config = config.text_config

    num_key_value_heads = int(text_config.num_key_value_heads)
    head_dim = int(text_config.head_dim)
    num_hidden_layers = int(text_config.num_hidden_layers)

    image_token_id = int(config.image_token_id)

    eos_token_ids = np.atleast_1d(
        np.asarray(
            generation_config.eos_token_id,
            dtype=np.int64,
        )
    )

    batch_size = input_ids.shape[0]

    past_cache_values = create_empty_kv_cache(
        batch_size=batch_size,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )

    generated_tokens = np.empty(
        (batch_size, 0),
        dtype=np.int64,
    )

    image_features: np.ndarray | None = None

    for step in range(max_new_tokens):
        inputs_embeds = embed_session.run(
            None,
            {
                "input_ids": input_ids,
            },
        )[0]

        if image_features is None:
            image_features = vision_session.run(
                None,
                {
                    "pixel_values": pixel_values,
                },
            )[0]

            image_token_mask = input_ids == image_token_id
            image_token_count = int(image_token_mask.sum())
            feature_count = int(
                image_features.shape[0] * image_features.shape[1]
            )

            if image_token_count != feature_count:
                raise RuntimeError(
                    "Image placeholder count does not match vision feature count: "
                    f"{image_token_count} placeholders versus "
                    f"{feature_count} features"
                )

            if verbose:
                print_tensor_stats(
                    "Image features",
                    image_features,
                )

                print()
                print(f"Prompt tokens      : {input_ids.shape[1]}")
                print(f"Image tokens       : {image_token_count}")
                print(f"Image features     : {image_features.shape[1]}")
                print(f"Input embeddings   : {inputs_embeds.shape}")
                print()

            if dump_vision:
                output_path = Path("vision_python.bin")
                np.asarray(
                    image_features,
                    dtype=np.float32,
                ).tofile(output_path)

                if verbose:
                    print(f"Wrote {output_path}")
                    print()

            inputs_embeds[image_token_mask] = image_features.reshape(
                -1,
                image_features.shape[-1],
            )

        decoder_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            **past_cache_values,
        }

        decoder_outputs = decoder_session.run(
            None,
            decoder_inputs,
        )

        logits = decoder_outputs[0]
        present_cache_values = decoder_outputs[1:]

        next_token_ids = logits[:, -1].argmax(
            axis=-1,
            keepdims=True,
        ).astype(np.int64)

        if verbose and step < 10:
            decoded = processor.decode(
                next_token_ids[0],
                skip_special_tokens=False,
            )

            print(
                f"step={step:04d} "
                f"token_id={int(next_token_ids[0, 0])} "
                f"token={decoded!r}"
            )

        generated_tokens = np.concatenate(
            (
                generated_tokens,
                next_token_ids,
            ),
            axis=-1,
        )

        reached_eos = np.isin(
            next_token_ids,
            eos_token_ids,
        ).any()

        if reached_eos:
            break

        input_ids = next_token_ids

        attention_mask = np.concatenate(
            (
                attention_mask,
                np.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                ),
            ),
            axis=-1,
        )

        if len(present_cache_values) != len(past_cache_values):
            raise RuntimeError(
                "Decoder returned an unexpected number of KV-cache tensors: "
                f"{len(present_cache_values)} returned, "
                f"{len(past_cache_values)} expected"
            )

        for index, key in enumerate(past_cache_values):
            past_cache_values[key] = present_cache_values[index]

        if not verbose:
            print(
                processor.decode(
                    next_token_ids[0],
                    skip_special_tokens=False,
                ),
                end="",
                flush=True,
            )

    if not verbose:
        print()

    return processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {args.model_dir}"
        )

    if not args.image.is_file():
        raise FileNotFoundError(
            f"Input image does not exist: {args.image}"
        )

    (
        config,
        processor,
        generation_config,
        vision_session,
        embed_session,
        decoder_session,
    ) = load_model(
        args.model_dir,
        args.variant,
    )

    inputs = prepare_inputs(
        processor,
        args.image,
    )

    output = run_inference(
        config=config,
        processor=processor,
        generation_config=generation_config,
        vision_session=vision_session,
        embed_session=embed_session,
        decoder_session=decoder_session,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        dump_vision=args.dump_vision,
    )

    if args.verbose:
        print()
        print("-" * 80)
        print("OCR Output")
        print("-" * 80)
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())