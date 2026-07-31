#!/usr/bin/env python3
"""
Inspect the exported ONNX models for LightOnOCR-2-1B.

This script downloads the ONNX models from Hugging Face and prints:

- model metadata
- opset version
- producer
- graph inputs
- graph outputs
- initializers
- external data
- node/operator statistics

Usage:

    python inspect_models.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import onnx
from huggingface_hub import snapshot_download

MODEL_ID = "onnx-community/LightOnOCR-2-1B-ONNX"

MODELS = [
    "onnx/vision_encoder_q4.onnx",
    "onnx/embed_tokens_q4.onnx",
    "onnx/decoder_model_merged_q4.onnx",
]


ONNX_TYPES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}


def shape(value_info):
    tensor = value_info.type.tensor_type

    dims = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    return dims


def dtype(value_info):
    tensor = value_info.type.tensor_type
    return ONNX_TYPES.get(tensor.elem_type, tensor.elem_type)


def inspect(path: Path):
    print("=" * 80)
    print(path.name)
    print("=" * 80)

    model = onnx.load(path, load_external_data=False)

    print(f"IR version      : {model.ir_version}")

    if model.opset_import:
        print(
            f"Opset           : {model.opset_import[0].version}"
        )

    print(f"Producer        : {model.producer_name}")
    print(f"Producer ver    : {model.producer_version}")

    print()

    print("Inputs")
    print("-" * 80)

    for value in model.graph.input:
        print(
            f"{value.name:<40}"
            f"{dtype(value):<10}"
            f"{shape(value)}"
        )

    print()

    print("Outputs")
    print("-" * 80)

    for value in model.graph.output:
        print(
            f"{value.name:<40}"
            f"{dtype(value):<10}"
            f"{shape(value)}"
        )

    print()

    print(f"Initializers    : {len(model.graph.initializer)}")
    print(f"Nodes           : {len(model.graph.node)}")

    ops = Counter(node.op_type for node in model.graph.node)

    print()
    print("Operators")
    print("-" * 80)

    for op, count in sorted(ops.items()):
        print(f"{op:<24}{count}")


def main():
    root = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=[f"{m}*" for m in MODELS],
        )
    )

    print(f"Downloaded to: {root}")
    print()

    for model in MODELS:
        inspect(root / model)


if __name__ == "__main__":
    main()
