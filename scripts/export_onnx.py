#!/usr/bin/env python3
"""Export a local supported LightOnOCR PyTorch checkpoint to ONNX."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any
import onnx

EXPORT_OPSET = 20
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

ONNX_TYPES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
    11: "float64", 12: "uint32", 13: "uint64", 14: "complex64",
    15: "complex128", 16: "bfloat16",
}


def shape(value_info: Any) -> list[str]:
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


def dtype(value_info: Any) -> str | int:
    return ONNX_TYPES.get(value_info.type.tensor_type.elem_type,
                          value_info.type.tensor_type.elem_type)


def inspect_onnx(path: Path) -> dict[str, Any]:
    print("=" * 80)
    print(path.name)
    print("=" * 80)
    model = onnx.load(path, load_external_data=False)
    opset = model.opset_import[0].version if model.opset_import else None
    print(f"IR version      : {model.ir_version}")
    print(f"Opset           : {opset}")
    print(f"Producer        : {model.producer_name} {model.producer_version}")
    for title, values in (("Inputs", model.graph.input),
                          ("Outputs", model.graph.output)):
        print(f"\n{title}\n{'-' * 80}")
        for value in values:
            print(f"{value.name:<40}{str(dtype(value)):<12}{shape(value)}")
    ops = Counter(node.op_type for node in model.graph.node)
    print(f"\nInitializers    : {len(model.graph.initializer)}")
    print(f"Nodes           : {len(model.graph.node)}")
    print("Operators       : " + ", ".join(
        f"{op}={count}" for op, count in sorted(ops.items())
    ))
    return {
        "file": path.name,
        "ir_version": model.ir_version,
        "opset": opset,
        "producer": model.producer_name,
        "producer_version": model.producer_version,
        "inputs": [{"name": v.name, "dtype": dtype(v), "shape": shape(v)}
                   for v in model.graph.input],
        "outputs": [{"name": v.name, "dtype": dtype(v), "shape": shape(v)}
                    for v in model.graph.output],
        "initializer_count": len(model.graph.initializer),
        "node_count": len(model.graph.node),
        "operators": dict(ops),
    }


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remove_previous_graph(path: Path) -> None:
    """Remove only the graph and sidecars named by that graph's ONNX metadata."""
    if path.exists():
        model = onnx.load(path, load_external_data=False)
        locations = {
            item.value
            for tensor in model.graph.initializer
            for item in tensor.external_data
            if item.key == "location"
        }
        for location in locations:
            sidecar = path.parent / location
            if sidecar.is_file():
                sidecar.unlink()
        path.unlink()
    consolidated = Path(str(path) + ".data")
    if consolidated.is_file():
        consolidated.unlink()


def _consolidate_external_data(path: Path) -> None:
    """Pack legacy PyTorch per-initializer sidecars into one graph data file."""
    model = onnx.load(path, load_external_data=False)
    tensors = [tensor for tensor in model.graph.initializer if tensor.data_location == 1]
    if not tensors:
        return

    data_path = Path(str(path) + ".data")
    source_paths: set[Path] = set()
    with data_path.open("wb") as destination:
        for tensor in tensors:
            metadata = {item.key: item.value for item in tensor.external_data}
            source = path.parent / metadata["location"]
            source_paths.add(source)
            offset = int(metadata.get("offset", "0"))
            remaining = int(metadata.get("length", str(source.stat().st_size - offset)))
            new_offset = destination.tell()
            with source.open("rb") as source_stream:
                source_stream.seek(offset)
                while remaining:
                    block = source_stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise IOError(f"truncated ONNX external data file: {source}")
                    destination.write(block)
                    remaining -= len(block)
            tensor.ClearField("external_data")
            tensor.raw_data = b""
            tensor.data_location = onnx.TensorProto.EXTERNAL
            onnx.external_data_helper.set_external_data(
                tensor, location=data_path.name,
                offset=new_offset,
                length=int(metadata.get("length", str(source.stat().st_size - offset))),
            )
    onnx.save(model, path)
    for source in source_paths:
        if source != data_path and source.is_file():
            source.unlink()


def _copy_checkpoint_assets(checkpoint_dir: Path, output_dir: Path) -> list[str]:
    """Copy only the runtime configs, tokenizer data, and required template."""
    asset_names = [
        "config.json",
        "processor_config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ]
    redundant_map = output_dir / "special_tokens_map.json"
    if redundant_map.is_file():
        redundant_map.unlink()
    copied = []
    for asset_name in asset_names:
        source = checkpoint_dir / asset_name
        if not source.is_file():
            raise FileNotFoundError(f"required checkpoint asset is missing: {source}")
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination.name)
    return copied


def export_onnx(
    checkpoint_dir: Path,
    output_dir: Path,
    image_path: Path,
    inspection: dict[str, Any],
) -> None:
    """Export a supported LightOnOCR full-attention checkpoint to ONNX."""
    import torch
    import numpy as np
    from PIL import Image
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
    from transformers.cache_utils import DynamicCache

    class VisionGraph(torch.nn.Module):
        def __init__(self, ocr_model: Any) -> None:
            super().__init__()
            self.vision_encoder = ocr_model.model.vision_encoder
            self.vision_projection = ocr_model.model.vision_projection
            self.patch_size = ocr_model.config.vision_config.patch_size
            self.merge_size = ocr_model.config.spatial_merge_size
            self.max_grid_width = (
                ocr_model.config.vision_config.image_size // self.patch_size
            )

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # The Transformers Pixtral implementation loops over image_sizes
            # in Python, which specializes a trace to the sample image size.
            # This single-image export expresses the same patch/grid math with
            # tensor shape operations so ONNX keeps height and width dynamic.
            encoder = self.vision_encoder
            patch_grid = encoder.patch_conv(
                pixel_values.to(dtype=encoder.patch_conv.weight.dtype)
            )
            grid_height, grid_width = patch_grid.shape[-2:]
            hidden_states = patch_grid.flatten(2).transpose(1, 2)
            hidden_states = encoder.ln_pre(hidden_states)
            rows = torch.arange(grid_height, device=pixel_values.device)
            columns = torch.arange(grid_width, device=pixel_values.device)
            row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
            position_ids = (
                row_grid * self.max_grid_width + column_grid
            ).reshape(1, -1)
            position_embeddings = encoder.patch_positional_embedding(
                hidden_states, position_ids
            )
            sequence_length = hidden_states.shape[1]
            attention_mask = torch.zeros(
                (hidden_states.shape[0], 1, sequence_length, sequence_length),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hidden_states = encoder.transformer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            ).last_hidden_state.squeeze(0)

            projector = self.vision_projection
            hidden_states = projector.norm(hidden_states)
            image_grid = hidden_states.reshape(
                grid_height, grid_width, -1
            ).permute(2, 0, 1).unsqueeze(0)
            patches = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.merge_size,
                stride=self.merge_size,
            )
            patches = patches.reshape(
                1, -1, patches.shape[-1]
            ).transpose(1, 2)
            features = projector.patch_merger.merging_layer(patches)
            features = projector.linear_1(features)
            features = projector.act(features)
            features = projector.linear_2(features)
            return features

    class EmbeddingGraph(torch.nn.Module):
        def __init__(self, ocr_model: Any) -> None:
            super().__init__()
            self.embed_tokens = ocr_model.model.language_model.embed_tokens

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed_tokens(input_ids)

    class DecoderGraph(torch.nn.Module):
        def __init__(self, ocr_model: Any) -> None:
            super().__init__()
            self.language_model = ocr_model.model.language_model
            self.lm_head = ocr_model.lm_head
            self.num_layers = int(ocr_model.config.text_config.num_hidden_layers)

        def forward(self, inputs_embeds: torch.Tensor,
                    attention_mask: torch.Tensor,
                    *past_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
            cache_pairs = [
                (past_tensors[2 * i], past_tensors[2 * i + 1])
                for i in range(self.num_layers)
            ]
            cache = DynamicCache(
                ddp_cache_data=cache_pairs,
                config=self.language_model.config,
            )
            past_len = past_tensors[0].shape[-2]
            current_len = inputs_embeds.shape[1]
            position_ids = torch.arange(
                past_len, past_len + current_len,
                dtype=torch.long, device=inputs_embeds.device,
            ).unsqueeze(0)
            hidden_states = inputs_embeds
            position_embeddings = self.language_model.rotary_emb(
                hidden_states, position_ids
            )
            key_positions = torch.arange(
                past_len + current_len, dtype=torch.long,
                device=inputs_embeds.device,
            )
            query_positions = position_ids[0]
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            allowed = allowed.unsqueeze(0).unsqueeze(0) & attention_mask[:, None, None, :].bool()
            causal_mask = torch.zeros(
                allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device
            ).masked_fill(~allowed, torch.finfo(hidden_states.dtype).min)
            for layer in self.language_model.layers:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            hidden_states = self.language_model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            updated: list[torch.Tensor] = [logits]
            for layer in cache.layers:
                updated.extend((layer.keys, layer.values))
            return tuple(updated)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    graph_paths = {
        "vision": onnx_dir / "vision_encoder.onnx",
        "embedding": onnx_dir / "embed_tokens.onnx",
        "decoder": onnx_dir / "decoder_model_merged.onnx",
    }
    # Remove artifacts from the earlier flat layout when re-exporting.
    for filename in ("vision_encoder.onnx", "embed_tokens.onnx", "decoder_model_merged.onnx"):
        _remove_previous_graph(output_dir / filename)
    for path in graph_paths.values():
        _remove_previous_graph(path)
    processor = LightOnOcrProcessor.from_pretrained(
        str(checkpoint_dir), local_files_only=True
    )
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    text_config = model.config.text_config
    num_layers = int(text_config.num_hidden_layers)
    num_kv_heads = int(text_config.num_key_value_heads)
    head_dim = int(
        getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
    )
    hidden_size = int(text_config.hidden_size)
    vocab_size = int(text_config.vocab_size)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        raise ValueError("checkpoint config does not define image_token_id/image_token_index")
    image_token_id = int(image_token_id)
    image = Image.open(image_path).convert("RGB")
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_sizes = inputs["image_sizes"]
    attention_mask = inputs["attention_mask"]

    # The architecture and dynamic tensor shapes are derived from this actual
    # processor sample, rather than assumptions from the LightOnOCR-2 export.
    print("\nReference processor tensors")
    for name, value in inputs.items():
        print(f"{name:<20} shape={tuple(value.shape)} dtype={value.dtype}")
    if int((input_ids == image_token_id).sum()) == 0:
        raise RuntimeError("processor template contains no image placeholder IDs")

    vision = VisionGraph(model).eval()
    embedding = EmbeddingGraph(model).eval()
    decoder = DecoderGraph(model).eval()
    empty_past = [
        torch.empty((1, num_kv_heads, 0, head_dim), dtype=torch.float32)
        for _ in range(2 * num_layers)
    ]

    # Export the visual encoder and projector as one graph. This keeps the
    # exposed output at the language-model image-feature boundary.
    with torch.inference_mode():
        reference_features = vision(pixel_values)
        reference_embeddings = embedding(input_ids)
        transformers_features = model.model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            return_dict=True,
        ).pooler_output[0].unsqueeze(0)
        reference_full = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    image_positions = input_ids[0] == image_token_id
    if reference_features.shape[1] != int(image_positions.sum()):
        raise RuntimeError(
            "processor image placeholder count does not match projected features: "
            f"{int(image_positions.sum())} != {reference_features.shape[1]}"
        )
    np.testing.assert_allclose(
        reference_features.detach().numpy(),
        transformers_features.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    reference_embeddings = reference_embeddings.clone()
    reference_embeddings[0, image_positions] = reference_features[0]

    common_export_options = {
        "opset_version": EXPORT_OPSET,
        "dynamo": False,
        "external_data": True,
        "do_constant_folding": True,
    }
    torch.onnx.export(
        vision, (pixel_values,), str(graph_paths["vision"]),
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "image_features": {1: "num_image_features"},
        },
        **common_export_options,
    )
    _consolidate_external_data(graph_paths["vision"])
    torch.onnx.export(
        embedding, (input_ids,), str(graph_paths["embedding"]),
        input_names=["input_ids"], output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
        **common_export_options,
    )
    _consolidate_external_data(graph_paths["embedding"])

    cache_names = [
        f"past_key_values.{layer}.{kind}"
        for layer in range(num_layers) for kind in ("key", "value")
    ]
    present_names = [
        f"present.{layer}.{kind}"
        for layer in range(num_layers) for kind in ("key", "value")
    ]
    decoder_inputs = (reference_embeddings, attention_mask, *empty_past)
    decoder_input_names = ["inputs_embeds", "attention_mask", *cache_names]
    decoder_dynamic_axes: dict[str, dict[int, str]] = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
    }
    for name in cache_names:
        decoder_dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
    decoder_dynamic_axes["logits"] = {0: "batch_size", 1: "sequence_length"}
    for name in present_names:
        decoder_dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
    torch.onnx.export(
        decoder, decoder_inputs, str(graph_paths["decoder"]),
        input_names=decoder_input_names,
        output_names=["logits", *present_names],
        dynamic_axes=decoder_dynamic_axes,
        **common_export_options,
    )
    _consolidate_external_data(graph_paths["decoder"])

    # Validate every graph against the PyTorch modules, including a cached
    # decode step. These are architecture-level parity checks, not OCR-quality
    # evaluation; the selected fixture is also used for the Rust baseline.
    import onnxruntime as ort

    sessions = {
        name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for name, path in graph_paths.items()
    }
    ort_features = sessions["vision"].run(
        ["image_features"],
        {"pixel_values": pixel_values.numpy()},
    )[0]
    ort_embeddings = sessions["embedding"].run(
        ["inputs_embeds"], {"input_ids": input_ids.numpy()}
    )[0]
    np.testing.assert_allclose(ort_features, transformers_features.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(ort_embeddings, embedding(input_ids).detach().numpy(), rtol=1e-5, atol=1e-5)

    # Exercise a second processor-produced aspect/height to catch traces that
    # accidentally bake the sample image's patch grid or rotary positions.
    alternate_image = image.resize((700, 280))
    alternate_inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": alternate_image}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        alternate_reference = model.model.get_image_features(
            pixel_values=alternate_inputs["pixel_values"],
            image_sizes=alternate_inputs["image_sizes"],
            return_dict=True,
        ).pooler_output[0].unsqueeze(0)
    alternate_features = sessions["vision"].run(
        ["image_features"],
        {"pixel_values": alternate_inputs["pixel_values"].numpy()},
    )[0]
    np.testing.assert_allclose(
        alternate_features, alternate_reference.numpy(), rtol=1e-4, atol=1e-4
    )
    dynamic_shape_probe = {
        "pixel_values_shape": list(alternate_inputs["pixel_values"].shape),
        "image_sizes": alternate_inputs["image_sizes"].tolist(),
        "image_token_count": int(
            (alternate_inputs["input_ids"] == image_token_id).sum()
        ),
        "image_features_shape": list(alternate_features.shape),
    }
    ort_inputs = {
        "inputs_embeds": reference_embeddings.numpy(),
        "attention_mask": attention_mask.numpy(),
        **{name: tensor.numpy() for name, tensor in zip(cache_names, empty_past)},
    }
    with torch.inference_mode():
        torch_prefill = decoder(*decoder_inputs)
    ort_prefill = sessions["decoder"].run(None, ort_inputs)
    np.testing.assert_allclose(ort_prefill[0], torch_prefill[0].numpy(), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(
        ort_prefill[0][:, -1:, :], reference_full.logits.detach().numpy(),
        rtol=2e-3, atol=2e-3,
    )

    next_id = int(torch_prefill[0][0, -1].argmax())
    next_ids = torch.tensor([[next_id]], dtype=torch.long)
    next_embeddings = embedding(next_ids)
    next_mask = torch.ones((1, attention_mask.shape[1] + 1), dtype=torch.long)
    next_past = [torch.from_numpy(value) for value in ort_prefill[1:]]
    decode_feed = {
        "inputs_embeds": next_embeddings.detach().numpy(),
        "attention_mask": next_mask.numpy(),
        **{name: tensor.numpy() for name, tensor in zip(cache_names, next_past)},
    }
    ort_decode = sessions["decoder"].run(None, decode_feed)
    torch_decode_inputs = (next_embeddings, next_mask, *next_past)
    with torch.inference_mode():
        torch_decode = decoder(*torch_decode_inputs)
    np.testing.assert_allclose(ort_decode[0], torch_decode[0].numpy(), rtol=2e-3, atol=2e-3)
    next_ort_id = int(ort_decode[0][0, -1].argmax())
    next_torch_id = int(torch_decode[0][0, -1].argmax())
    if next_ort_id != next_torch_id:
        raise AssertionError(f"cached decode token mismatch: ONNX {next_ort_id}, PyTorch {next_torch_id}")

    max_new_tokens = 8
    generated_onnx: list[int] = []
    current_logits = ort_prefill[0]
    current_past = [torch.from_numpy(value) for value in ort_prefill[1:]]
    for _ in range(max_new_tokens):
        token_id = int(current_logits[0, -1].argmax())
        generated_onnx.append(token_id)
        eos_ids = model.generation_config.eos_token_id
        if token_id in (eos_ids if isinstance(eos_ids, list) else [eos_ids]):
            break
        token = torch.tensor([[token_id]], dtype=torch.long)
        token_embedding = sessions["embedding"].run(
            ["inputs_embeds"], {"input_ids": token.numpy()}
        )[0]
        total_length = input_ids.shape[1] + len(generated_onnx)
        decode_attention = np.ones((1, total_length), dtype=np.int64)
        decode_feed = {
            "inputs_embeds": token_embedding,
            "attention_mask": decode_attention,
            **{name: value.numpy() for name, value in zip(cache_names, current_past)},
        }
        decoded = sessions["decoder"].run(None, decode_feed)
        current_logits = decoded[0]
        current_past = [torch.from_numpy(value) for value in decoded[1:]]

    with torch.inference_mode():
        generated_reference = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )
    generated_reference_ids = generated_reference[0, input_ids.shape[1]:].tolist()
    if generated_onnx != generated_reference_ids:
        raise AssertionError(
            "greedy generation token mismatch: "
            f"ONNX {generated_onnx}, Transformers {generated_reference_ids}"
        )
    generated_text = processor.tokenizer.decode(
        generated_onnx, skip_special_tokens=True
    )

    copied_assets = _copy_checkpoint_assets(checkpoint_dir, output_dir)
    graph_reports = []
    for path in graph_paths.values():
        graph_report = inspect_onnx(path)
        graph_report["file"] = str(path.relative_to(output_dir))
        graph_reports.append(graph_report)
    report = {
        "source_model": inspection["model_name"],
        "source_revision": inspection["revision"],
        "model_type": inspection["architecture"]["model_type"],
        "architecture": inspection["architecture"],
        "dimensions": {
            "num_layers": num_layers,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "image_token_id": image_token_id,
        },
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "opset": EXPORT_OPSET,
        "precision": "float32",
        "fixture_image": str(image_path),
        "copied_runtime_assets": copied_assets,
        "processor_inputs": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in inputs.items()
        },
        "pixel_values_stats": {
            "sha256": hashlib.sha256(
                pixel_values.contiguous().numpy().tobytes()
            ).hexdigest(),
            "min": float(pixel_values.min()),
            "max": float(pixel_values.max()),
            "mean": float(pixel_values.mean()),
        },
        "dynamic_shape_probe": dynamic_shape_probe,
        "processor_token_ids": input_ids[0].tolist(),
        "reference_prefill_argmax_token_id": int(torch_prefill[0][0, -1].argmax()),
        "cached_decode_argmax_token_id": next_torch_id,
        "greedy_generated_token_ids": generated_onnx,
        "greedy_generated_text": generated_text,
        "parity": {
            "vision_features": "passed (rtol=1e-4, atol=1e-4)",
            "vision_dynamic_spatial_shape": "passed (second size 280x700; rtol=1e-4, atol=1e-4)",
            "token_embeddings": "passed (rtol=1e-5, atol=1e-5)",
            "prefill_logits": "passed (rtol=2e-3, atol=2e-3)",
            "cached_decode_logits": "passed (rtol=2e-3, atol=2e-3)",
            "cached_decode_argmax": "passed",
            "full_model_prefill_logits": "passed (rtol=2e-3, atol=2e-3)",
            "greedy_generated_token_ids": "passed",
        },
        "graphs": graph_reports,
    }
    report["checkpoint_inspection"] = {
        key: value for key, value in inspection.items() if key != "tensors"
    }
    for path in graph_paths.values():
        relative_path = str(path.relative_to(output_dir))
        report.setdefault("sha256", {})[relative_path] = _file_sha256(path)
        external_path = Path(str(path) + ".data")
        if external_path.exists():
            relative_external_path = str(external_path.relative_to(output_dir))
            report.setdefault("sha256", {})[relative_external_path] = _file_sha256(external_path)
    for asset_name in copied_assets:
        report.setdefault("sha256", {})[asset_name] = _file_sha256(output_dir / asset_name)
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nExport and parity report: {output_dir / 'export_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="Local Transformers/safetensors checkpoint directory.")
    parser.add_argument("--output-dir", type=Path,
                        help="Checkpoint root (default: models/<checkpoint-name>-ONNX).")
    parser.add_argument("--model-name")
    parser.add_argument("--revision")
    parser.add_argument("--fixture-image", type=Path,
                        default=Path("tests/fixtures/lightonocr2_baseline.png"))
    parser.add_argument("--list-tensors", action="store_true")
    args = parser.parse_args()
    model_name = args.model_name or args.checkpoint_dir.name
    output_dir = args.output_dir or Path("models") / f"{model_name}-ONNX"
    inspection = inspect_checkpoint(args.checkpoint_dir, args.list_tensors,
                                    revision=args.revision, model_name=model_name)
    export_onnx(args.checkpoint_dir, output_dir, args.fixture_image, inspection)

if __name__ == "__main__":
    main()
