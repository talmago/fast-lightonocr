#!/usr/bin/env python3
"""Inspect a local LightOnOCR PyTorch/safetensors checkpoint."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

SUPPORTED_MODEL_TYPES = {"lighton_ocr", "mistral3"}

def identify_architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and describe the shared Pixtral + Qwen3 OCR architecture."""
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(value) for value in config.get("architectures", [])]
    normalized_architectures = {value.lower().replace("_", "") for value in architectures}
    text_config = config.get("text_config") or {}
    vision_config = config.get("vision_config") or {}
    text_model_type = str(text_config.get("model_type", "")).lower()
    vision_model_type = str(vision_config.get("model_type", "")).lower()
    if (model_type not in SUPPORTED_MODEL_TYPES
            or "lightonocrforconditionalgeneration" not in normalized_architectures):
        raise ValueError(
            "unsupported LightOnOCR architecture in config.json: "
            f"model_type={model_type!r}, architectures={architectures!r}"
        )
    if text_model_type != "qwen3" or vision_model_type != "pixtral":
        raise ValueError(
            "export currently supports the shared Qwen3/Pixtral LightOnOCR "
            f"architecture; got text={text_model_type!r}, vision={vision_model_type!r}"
        )
    layer_types = text_config.get("layer_types") or []
    if layer_types and any(layer_type != "full_attention" for layer_type in layer_types):
        raise ValueError(
            "export supports full-attention decoder layers only; "
            f"found layer_types={layer_types!r}"
        )
    return {
        "family": "lightonocr_pixtral_qwen3_full_attention",
        "model_type": model_type,
        "architectures": architectures,
        "text_model_type": text_model_type,
        "vision_model_type": vision_model_type,
        "transformers_version": config.get("transformers_version"),
    }


def detect_checkpoint_revision(checkpoint_dir: Path) -> str | None:
    """Read the Hub snapshot revision when local download metadata exists."""
    metadata = checkpoint_dir / ".cache/huggingface/download/config.json.metadata"
    if metadata.is_file():
        return metadata.read_text().splitlines()[0].strip()
    for index, part in enumerate(checkpoint_dir.parts[:-1]):
        if part == "snapshots":
            return checkpoint_dir.parts[index + 1]
    return None

def inspect_checkpoint(
    checkpoint_dir: Path,
    list_tensors: bool = False,
    revision: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Read architecture and tensor metadata without materializing weights."""
    from safetensors import safe_open

    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text())
    architecture = identify_architecture(config)
    detected_revision = detect_checkpoint_revision(checkpoint_dir)
    if revision and detected_revision and revision != detected_revision:
        raise ValueError(
            f"requested revision {revision} differs from local snapshot {detected_revision}"
        )
    source_revision = revision or detected_revision
    source_name = model_name or checkpoint_dir.name
    weights = sorted(checkpoint_dir.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no safetensors files found under {checkpoint_dir}")

    dtype_counts: Counter[str] = Counter()
    module_counts: Counter[str] = Counter()
    tensor_inventory: list[dict[str, Any]] = []
    for weight_file in weights:
        with safe_open(weight_file, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                view = sf.get_slice(name)
                dtype_name = str(view.get_dtype())
                tensor_shape = list(view.get_shape())
                dtype_counts[dtype_name] += 1
                module_counts[name.split(".")[0]] += 1
                tensor_inventory.append({
                    "name": name,
                    "shape": tensor_shape,
                    "dtype": dtype_name,
                    "file": weight_file.name,
                })

    print(f"Checkpoint      : {checkpoint_dir}")
    print(f"Model name      : {source_name}")
    print(f"Transformers ver: {architecture['transformers_version'] or 'unspecified'}")
    print(f"Model type      : {config.get('model_type')}")
    print(f"Architectures   : {config.get('architectures')}")
    print(f"Image token ID  : {config.get('image_token_id', config.get('image_token_index'))}")
    print(f"Hub revision    : {source_revision or 'not recorded locally'}")
    print(f"Text config     : {json.dumps(config.get('text_config', {}), sort_keys=True)}")
    print(f"Vision config   : {json.dumps(config.get('vision_config', {}), sort_keys=True)}")
    print(f"Weight files    : {len(weights)}")
    print(f"Tensor count    : {len(tensor_inventory)}")
    print(f"Tensor dtypes   : {dict(dtype_counts)}")
    print(f"Root modules    : {dict(module_counts)}")
    if list_tensors:
        print("\nSafetensors inventory\n" + "-" * 80)
        for tensor in tensor_inventory:
            print(f"{tensor['name']:<88} {tensor['dtype']:<12} {tensor['shape']}")

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "model_name": source_name,
        "revision": source_revision,
        "architecture": architecture,
        "config": config,
        "weight_files": [p.name for p in weights],
        "tensor_count": len(tensor_inventory),
        "dtype_counts": dict(dtype_counts),
        "root_module_counts": dict(module_counts),
        "tensors": tensor_inventory,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--list-tensors", action="store_true")
    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint_dir, args.list_tensors,
                       revision=args.revision, model_name=args.model_name)

if __name__ == "__main__":
    main()
