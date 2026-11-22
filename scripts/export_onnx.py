#!/usr/bin/env python3
"""Export a local supported LightOnOCR PyTorch checkpoint to ONNX."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any
import onnx

EXPORT_OPSET = 20

class InferenceArchitecture(str, Enum):
    """Inference paths supported by the Rust runtime."""

    PIXTRAL = "pixtral"
    QWEN = "qwen"


def identify_architecture(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a checkpoint and classify it as Pixtral or Qwen."""
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(value) for value in config.get("architectures", [])]
    normalized_architectures = {value.lower().replace("_", "") for value in architectures}
    text_config = config.get("text_config") or {}
    vision_config = config.get("vision_config") or {}
    text_model_type = str(text_config.get("model_type", "")).lower()
    vision_model_type = str(vision_config.get("model_type", "")).lower()

    if (
        model_type in {"lighton_ocr", "mistral3"}
        and "lightonocrforconditionalgeneration" in normalized_architectures
        and text_model_type == "qwen3"
        and vision_model_type == "pixtral"
    ):
        layer_types = text_config.get("layer_types") or []
        if any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError(
                "Pixtral checkpoints must use full-attention decoder layers; "
                f"found layer_types={layer_types!r}"
            )
        architecture = InferenceArchitecture.PIXTRAL
    elif (
        model_type == "qwen3_5"
        and "qwen35forconditionalgeneration" in normalized_architectures
        and text_model_type == "qwen3_5_text"
        and vision_model_type == "qwen3_5"
    ):
        layer_types = text_config.get("layer_types") or []
        if not layer_types or any(
            layer_type not in {"linear_attention", "full_attention"}
            for layer_type in layer_types
        ):
            raise ValueError(
                "Qwen checkpoints must declare linear/full attention layers; "
                f"found layer_types={layer_types!r}"
            )
        architecture = InferenceArchitecture.QWEN
    else:
        raise ValueError(
            "unsupported LightOnOCR architecture in config.json: "
            f"model_type={model_type!r}, architectures={architectures!r}, "
            f"text={text_model_type!r}, vision={vision_model_type!r}"
        )

    return {
        "inference_architecture": architecture.value,
        "model_type": model_type,
        "architectures": architectures,
        "text_model_type": text_model_type,
        "vision_model_type": vision_model_type,
        "layer_types": layer_types,
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
            for tensor in _graph_tensors(model.graph)
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


def _graph_tensors(graph: Any):
    """Yield initializer and attribute tensors, including nested subgraphs."""
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graph_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from _graph_tensors(nested)


def _consolidate_external_data(path: Path) -> None:
    """Pack initializer and node-attribute sidecars into one graph data file."""
    model = onnx.load(path, load_external_data=False)
    tensors = [tensor for tensor in _graph_tensors(model.graph)
               if tensor.data_location == onnx.TensorProto.EXTERNAL]
    if not tensors:
        return

    data_path = Path(str(path) + ".data")
    source_paths: set[Path] = set()
    pack_tensors = []
    for tensor in tensors:
        metadata = {item.key: item.value for item in tensor.external_data}
        location = metadata.get("location")
        if not location:
            raise ValueError(f"external tensor {tensor.name!r} has no location in {path}")
        source = path.parent / location
        source_paths.add(source)
        if source.resolve() != data_path.resolve():
            pack_tensors.append((tensor, metadata, source))
    if not pack_tensors:
        return

    # Keep the already consolidated initializer data in place, then append
    # node-attribute constants and any remaining per-tensor sidecars.
    mode = "ab" if data_path.is_file() else "wb"
    with data_path.open(mode) as destination:
        for tensor, metadata, source in pack_tensors:
            if not source.is_file():
                raise FileNotFoundError(f"ONNX external data file is missing: {source}")
            offset = int(metadata.get("offset", "0"))
            length = int(metadata.get("length", str(source.stat().st_size - offset)))
            new_offset = destination.tell()
            remaining = length
            with source.open("rb") as source_stream:
                source_stream.seek(offset)
                while remaining:
                    block = source_stream.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise IOError(f"truncated ONNX external data file: {source}")
                    destination.write(block)
                    remaining -= len(block)
            del tensor.external_data[:]
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for key, value in (("location", data_path.name),
                               ("offset", str(new_offset)),
                               ("length", str(length))):
                entry = tensor.external_data.add()
                entry.key = key
                entry.value = value
    onnx.save(model, path)
    for source in source_paths:
        if source.resolve() != data_path.resolve() and source.is_file():
            source.unlink()


def _clean_legacy_export_sidecars(onnx_dir: Path) -> None:
    """Remove only temporary sidecars left by failed PyTorch ONNX exports."""
    for pattern in ("onnx__*", "_layers*"):
        for sidecar in onnx_dir.glob(pattern):
            if sidecar.is_file():
                sidecar.unlink()


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


def _export_qwen(
    checkpoint_dir: Path,
    output_dir: Path,
    image_path: Path,
    inspection: dict[str, Any],
) -> None:
    """Export Qwen3.5 vision, embedding, prefill, and cached-decode graphs.

    Qwen3.5 has two distinct decoder-state kinds. Linear-attention layers
    expose convolution and recurrent state; full-attention layers expose K/V
    cache. Prefill and decode are separate graphs because linear attention
    takes a different path when recurrent state exists.
    """
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    class VisionGraph(torch.nn.Module):
        def __init__(self, qwen_model: Any) -> None:
            super().__init__()
            self.visual = qwen_model.model.visual

            # Transformers' reference helpers call grid_thw.tolist(), which
            # specializes traced exports to the fixture dimensions. These
            # tensor-only equivalents retain the grid values in ONNX.
            def interpolate(grid_thw: torch.Tensor) -> torch.Tensor:
                t, height, width = grid_thw[0, 0], grid_thw[0, 1], grid_thw[0, 2]
                side = self.visual.num_grid_per_side
                h = torch.arange(height, device=grid_thw.device, dtype=torch.float32)
                w = torch.arange(width, device=grid_thw.device, dtype=torch.float32)
                h_scale = (side - 1) / torch.clamp(height - 1, min=1)
                w_scale = (side - 1) / torch.clamp(width - 1, min=1)
                h_pos = h * h_scale
                w_pos = w * w_scale
                h0, w0 = h_pos.long(), w_pos.long()
                h1 = (h0 + 1).clamp(max=side - 1)
                w1 = (w0 + 1).clamp(max=side - 1)
                dh, dw = h_pos - h0, w_pos - w0
                base0, base1 = h0 * side, h1 * side
                indices = torch.stack((
                    (base0[:, None] + w0[None, :]).reshape(-1),
                    (base0[:, None] + w1[None, :]).reshape(-1),
                    (base1[:, None] + w0[None, :]).reshape(-1),
                    (base1[:, None] + w1[None, :]).reshape(-1),
                ))
                weights = torch.stack((
                    ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                    ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                    (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                    (dh[:, None] * dw[None, :]).reshape(-1),
                ))
                embedded = self.visual.pos_embed(indices).to(self.visual.pos_embed.weight.dtype)
                interpolated = (embedded * weights[:, :, None]).sum(dim=0)
                merge = self.visual.spatial_merge_size
                interpolated = interpolated.reshape(
                    1, height // merge, merge, width // merge, merge, -1
                ).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
                return interpolated

            def rotary_positions(grid_thw: torch.Tensor) -> torch.Tensor:
                _, height, width = grid_thw[0, 0], grid_thw[0, 1], grid_thw[0, 2]
                max_hw = torch.maximum(height, width)
                freq = self.visual.rotary_pos_emb(max_hw)
                merge = self.visual.spatial_merge_size
                mh = torch.arange(height // merge, device=grid_thw.device)
                mw = torch.arange(width // merge, device=grid_thw.device)
                intra = torch.arange(merge, device=grid_thw.device)
                rows = (mh[:, None, None, None] * merge + intra[None, None, :, None])
                cols = (mw[None, :, None, None] * merge + intra[None, None, None, :])
                rows = rows.expand(height // merge, width // merge, merge, merge).reshape(-1)
                cols = cols.expand(height // merge, width // merge, merge, merge).reshape(-1)
                coords = torch.stack((rows, cols), dim=-1)
                return freq[coords].flatten(1)

            self.visual.fast_pos_embed_interpolate = interpolate
            self.visual.rot_pos_emb = rotary_positions
            for block in self.visual.blocks:
                attention = block.attn
                def packed_single_image_attention(
                    hidden_states, cu_seqlens, rotary_pos_emb=None,
                    position_embeddings=None, _attention=attention, **kwargs,
                ):
                    sequence_length = hidden_states.shape[0]
                    query, key, value = (
                        _attention.qkv(hidden_states)
                        .reshape(sequence_length, 3, _attention.num_heads, -1)
                        .permute(1, 0, 2, 3).unbind(0)
                    )
                    cos, sin = position_embeddings
                    query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
                    query = query.transpose(0, 1).unsqueeze(0)
                    key = key.transpose(0, 1).unsqueeze(0)
                    value = value.transpose(0, 1).unsqueeze(0)
                    weights = torch.matmul(query, key.transpose(-2, -1)) * _attention.scaling
                    weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
                    output = torch.matmul(weights, value)
                    output = output.squeeze(0).transpose(0, 1).reshape(sequence_length, -1)
                    return _attention.proj(output)
                attention.forward = packed_single_image_attention

        def forward(self, pixel_values: torch.Tensor,
                    image_grid_thw: torch.Tensor) -> torch.Tensor:
            return self.visual(pixel_values, grid_thw=image_grid_thw).pooler_output

    class EmbeddingGraph(torch.nn.Module):
        def __init__(self, qwen_model: Any) -> None:
            super().__init__()
            self.embed_tokens = qwen_model.model.language_model.embed_tokens

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed_tokens(input_ids)

    class DecoderGraph(torch.nn.Module):
        def __init__(self, qwen_model: Any, decode: bool) -> None:
            super().__init__()
            self.language_model = qwen_model.model.language_model
            self.lm_head = qwen_model.lm_head
            self.config = self.language_model.config
            self.layer_types = list(self.config.layer_types)
            self.decode = decode

        def forward(self, inputs_embeds: torch.Tensor,
                    position_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    *states: torch.Tensor) -> tuple[torch.Tensor, ...]:
            cache = DynamicCache(config=self.config)
            if self.decode:
                cursor = 0
                for index, kind in enumerate(self.layer_types):
                    layer = cache.layers[index]
                    if kind == "linear_attention":
                        if self.decode:
                            layer.conv_states = states[cursor]
                            layer.recurrent_states = states[cursor + 1]
                            layer.is_conv_states_initialized = True
                            layer.is_recurrent_states_initialized = True
                            layer.has_previous_state = True
                        cursor += 2
                    else:
                        layer.keys = states[cursor]
                        layer.values = states[cursor + 1]
                        layer.is_initialized = True
                        cursor += 2
            # Transformers' cache layer mutates state buffers with copy_(),
            # which the legacy ONNX exporter cannot lower. Reassigning the
            # tensor preserves the same recurrence and exposes state outputs.
            for index, kind in enumerate(self.layer_types):
                if kind == "linear_attention":
                    layer = cache.layers[index]
                    layer.update_conv_state = (
                        lambda value, _layer=layer, **kwargs: setattr(_layer, "conv_states", value) or value
                    )
                    layer.update_recurrent_state = (
                        lambda value, _layer=layer, **kwargs: setattr(_layer, "recurrent_states", value) or value
                    )
                    if self.decode:
                        attention = self.language_model.layers[index].linear_attn
                        def functional_conv_update(hidden_states, conv_state, weight, bias=None,
                                                   activation=None, _layer=layer):
                            state_length = conv_state.shape[-1]
                            sequence_length = hidden_states.shape[-1]
                            joined = torch.cat((conv_state, hidden_states), dim=-1).to(weight.dtype)
                            _layer.conv_states = joined[:, :, -state_length:]
                            convolved = torch.nn.functional.conv1d(
                                joined, weight.unsqueeze(1), bias, padding=0,
                                groups=hidden_states.shape[1],
                            )
                            return torch.nn.functional.silu(
                                convolved[:, :, -sequence_length:]
                            ).to(hidden_states.dtype)
                        attention.causal_conv1d_update = functional_conv_update
            hidden_states = inputs_embeds
            position_embeddings = self.language_model.rotary_emb(
                hidden_states, position_ids
            )
            past_length = cache.get_seq_length()
            current_length = inputs_embeds.shape[1]
            key_positions = torch.arange(
                past_length + current_length, device=inputs_embeds.device
            )
            query_positions = torch.arange(
                past_length, past_length + current_length,
                device=inputs_embeds.device,
            )
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            allowed = allowed.unsqueeze(0).unsqueeze(0)
            allowed = allowed & attention_mask[:, None, None, :].bool()
            causal_mask = torch.zeros(
                allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device
            ).masked_fill(~allowed, torch.finfo(hidden_states.dtype).min)
            for index, layer in enumerate(self.language_model.layers):
                layer_mask = None if self.layer_types[index] == "linear_attention" else causal_mask
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    past_key_values=cache,
                )
            hidden_states = self.language_model.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            flattened: list[torch.Tensor] = [logits]
            for index, kind in enumerate(self.layer_types):
                layer = cache.layers[index]
                if kind == "linear_attention":
                    flattened.extend((layer.conv_states, layer.recurrent_states))
                else:
                    flattened.extend((layer.keys, layer.values))
            return tuple(flattened)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    _clean_legacy_export_sidecars(onnx_dir)
    graph_paths = {
        "vision": onnx_dir / "vision_encoder.onnx",
        "embedding": onnx_dir / "embed_tokens.onnx",
        "prefill": onnx_dir / "decoder_prefill.onnx",
        "decode": onnx_dir / "decoder_decode.onnx",
    }
    for path in graph_paths.values():
        _remove_previous_graph(path)

    processor = AutoProcessor.from_pretrained(
        str(checkpoint_dir), local_files_only=True
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(checkpoint_dir), torch_dtype=torch.float32, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="eager",
    ).eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    attention_mask = inputs["attention_mask"]
    mm_token_type_ids = inputs["mm_token_type_ids"]
    layer_types = list(model.config.text_config.layer_types)
    decode_state_inputs: list[torch.Tensor] = []
    with torch.inference_mode():
        inputs_embeds = model.get_input_embeddings()(input_ids)
        positions = model.model.compute_3d_position_ids(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            image_grid_thw=grid, attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        vision_reference = model.model.get_image_features(
            pixel_values, image_grid_thw=grid, return_dict=True
        ).pooler_output[0].unsqueeze(0)
        reference = model(
            **inputs, use_cache=True, logits_to_keep=1, return_dict=True
        )
        decode_attention = torch.ones((1, input_ids.shape[1] + 1), dtype=torch.long)
        decode_positions = model.model.compute_3d_position_ids(
            input_ids=None,
            inputs_embeds=model.get_input_embeddings()(input_ids[:, -1:]),
            attention_mask=decode_attention,
            past_key_values=reference.past_key_values,
        )[:, :, -1:]
        decode_state_inputs = []
        for index, kind in enumerate(layer_types):
            layer = reference.past_key_values.layers[index]
            if kind == "linear_attention":
                decode_state_inputs.extend((layer.conv_states, layer.recurrent_states))
            else:
                decode_state_inputs.extend((layer.keys, layer.values))
    image_token_id = int(model.config.image_token_id)
    if int((input_ids == image_token_id).sum()) != vision_reference.shape[1]:
        raise RuntimeError("Qwen3.5 processor image-token count differs from vision features")

    vision = VisionGraph(model).eval()
    embedding = EmbeddingGraph(model).eval()
    prefill = DecoderGraph(model, decode=False).eval()
    decode = DecoderGraph(model, decode=True).eval()
    prefill_outputs = ["logits"]
    decode_state_names: list[str] = []
    state_dynamic_axes: dict[str, dict[int, str]] = {}
    for i, kind in enumerate(layer_types):
        if kind == "linear_attention":
            names = (f"conv_state.{i}", f"recurrent_state.{i}")
            conv_dim = model.config.text_config.linear_num_key_heads * (
                2 * model.config.text_config.linear_key_head_dim
                + model.config.text_config.linear_value_head_dim
            )
            decode_state_inputs.extend((
                torch.zeros((1, conv_dim, model.config.text_config.linear_conv_kernel_dim)),
                torch.zeros((1,
                    model.config.text_config.linear_num_value_heads,
                    model.config.text_config.linear_key_head_dim,
                    model.config.text_config.linear_value_head_dim)),
            ))
            state_dynamic_axes[names[0]] = {0: "batch_size"}
            state_dynamic_axes[names[1]] = {0: "batch_size"}
        else:
            names = (f"past_key.{i}", f"past_value.{i}")
            decode_state_inputs.extend((
                torch.zeros((1, model.config.text_config.num_key_value_heads, 0,
                             model.config.text_config.head_dim)),
                torch.zeros((1, model.config.text_config.num_key_value_heads, 0,
                             model.config.text_config.head_dim)),
            ))
            state_dynamic_axes[names[0]] = {0: "batch_size", 2: "past_sequence_length"}
            state_dynamic_axes[names[1]] = {0: "batch_size", 2: "past_sequence_length"}
        decode_state_names.extend(names)
        prefill_outputs.extend((f"state.{i}.0", f"state.{i}.1"))
    decode_outputs = ["logits", *[f"state.{i}.{j}"
                                    for i in range(len(layer_types))
                                    for j in range(2)]]
    with torch.inference_mode():
        prefill_reference = prefill(inputs_embeds, positions, attention_mask)
    decode_state_inputs = [state.detach().clone() for state in decode_state_inputs]
    next_id = reference.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_embedding = model.get_input_embeddings()(next_id)
    with torch.no_grad():
        decode_reference = decode(
            next_embedding, decode_positions, decode_attention, *decode_state_inputs
        )

    options = {"opset_version": EXPORT_OPSET, "dynamo": False,
               "external_data": True, "do_constant_folding": True}
    torch.onnx.export(
        vision, (pixel_values, grid), str(graph_paths["vision"]),
        input_names=["pixel_values", "image_grid_thw"],
        output_names=["image_features"],
        dynamic_axes={"pixel_values": {0: "num_patches"},
                      "image_grid_thw": {0: "num_images"},
                      "image_features": {0: "num_image_features"}}, **options,
    )
    torch.onnx.export(
        embedding, (input_ids,), str(graph_paths["embedding"]),
        input_names=["input_ids"], output_names=["inputs_embeds"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                      "inputs_embeds": {0: "batch_size", 1: "sequence_length"}}, **options,
    )
    common_dynamic = {
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        "position_ids": {1: "batch_size", 2: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    prefill_dynamic = dict(common_dynamic)
    decode_dynamic = dict(common_dynamic)
    decode_dynamic.update(state_dynamic_axes)
    for i, kind in enumerate(layer_types):
        if kind == "linear_attention":
            prefill_dynamic[f"state.{i}.0"] = {0: "batch_size"}
            prefill_dynamic[f"state.{i}.1"] = {0: "batch_size"}
            decode_dynamic[f"state.{i}.0"] = {0: "batch_size"}
            decode_dynamic[f"state.{i}.1"] = {0: "batch_size"}
        else:
            prefill_dynamic[f"state.{i}.0"] = {0: "batch_size", 2: "total_sequence_length"}
            prefill_dynamic[f"state.{i}.1"] = {0: "batch_size", 2: "total_sequence_length"}
            decode_dynamic[f"state.{i}.0"] = {0: "batch_size", 2: "total_sequence_length"}
            decode_dynamic[f"state.{i}.1"] = {0: "batch_size", 2: "total_sequence_length"}
    torch.onnx.export(
        prefill, (inputs_embeds, positions, attention_mask), str(graph_paths["prefill"]),
        input_names=["inputs_embeds", "position_ids", "attention_mask"],
        output_names=prefill_outputs, dynamic_axes=prefill_dynamic, **options,
    )
    torch.onnx.export(
        decode,
        (model.get_input_embeddings()(input_ids[:, -1:]), decode_positions,
         decode_attention, *decode_state_inputs),
        str(graph_paths["decode"]),
        input_names=["inputs_embeds", "position_ids", "attention_mask", *decode_state_names],
        output_names=decode_outputs, dynamic_axes=decode_dynamic, **options,
    )
    for path in graph_paths.values():
        _consolidate_external_data(path)

    import onnxruntime as ort
    sessions = {name: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                for name, path in graph_paths.items()}
    vision_feed = {"pixel_values": pixel_values.numpy()}
    if any(value.name == "image_grid_thw" for value in sessions["vision"].get_inputs()):
        vision_feed["image_grid_thw"] = grid.numpy()
    ort_features = sessions["vision"].run(["image_features"], vision_feed)[0]
    np.testing.assert_allclose(ort_features, vision_reference[0].numpy(), rtol=2e-3, atol=2e-3)
    alternate_inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image.resize((700, 280))}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        alternate_reference = model.model.get_image_features(
            alternate_inputs["pixel_values"],
            image_grid_thw=alternate_inputs["image_grid_thw"],
            return_dict=True,
        ).pooler_output[0].unsqueeze(0)
    alternate_feed = {
        "pixel_values": alternate_inputs["pixel_values"].numpy(),
        "image_grid_thw": alternate_inputs["image_grid_thw"].numpy(),
    }
    ort_alternate = sessions["vision"].run(["image_features"], alternate_feed)[0]
    np.testing.assert_allclose(ort_alternate, alternate_reference[0].numpy(),
                               rtol=2e-3, atol=2e-3)
    ort_embeds = sessions["embedding"].run(["inputs_embeds"],
                                             {"input_ids": input_ids.numpy()})[0]
    np.testing.assert_allclose(ort_embeds, inputs_embeds.numpy(), rtol=1e-5, atol=1e-5)
    prefill_feed = {
        "inputs_embeds": inputs_embeds.numpy(),
        "position_ids": positions.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    ort_prefill = sessions["prefill"].run(None, prefill_feed)
    np.testing.assert_allclose(ort_prefill[0], prefill_reference[0].numpy(),
                               rtol=3e-3, atol=3e-3)
    decode_feed = {
        "inputs_embeds": next_embedding.detach().numpy(),
        "position_ids": decode_positions.numpy(),
        "attention_mask": decode_attention.numpy(),
        **{name: value.numpy() for name, value in zip(decode_state_names, decode_state_inputs)},
    }
    ort_decode = sessions["decode"].run(None, decode_feed)
    np.testing.assert_allclose(ort_decode[0], decode_reference[0].detach().numpy(),
                               rtol=3e-3, atol=3e-3)
    # Report graph signatures and probe values. The Rust runtime consumes the
    # explicit four-axis position IDs and the exported hybrid state tensors.
    graph_reports = [inspect_onnx(path) for path in graph_paths.values()]
    for report, path in zip(graph_reports, graph_paths.values()):
        report["file"] = str(path.relative_to(output_dir))
    copied_assets = _copy_checkpoint_assets(checkpoint_dir, output_dir)
    report = {
        "source_model": inspection["model_name"],
        "source_revision": inspection["revision"],
        "architecture": inspection["architecture"],
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__, "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__, "opset": EXPORT_OPSET,
        "precision": "float32",
        "processor_inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                             for name, value in inputs.items()},
        "dynamic_vision_probe": {
            "pixel_values_shape": list(alternate_inputs["pixel_values"].shape),
            "image_grid_thw": alternate_inputs["image_grid_thw"].tolist(),
            "image_features_shape": list(ort_alternate.shape),
        },
        "processor_token_ids": input_ids[0].tolist(),
        "dimensions": {"hidden_size": model.config.text_config.hidden_size,
                       "vocab_size": model.config.text_config.vocab_size,
                       "layer_types": layer_types,
                       "image_token_id": image_token_id},
        "state_contract": [
            {"layer": i, "kind": kind,
             "tensors": (["conv_state", "recurrent_state"] if kind == "linear_attention"
                         else ["key", "value"])}
            for i, kind in enumerate(layer_types)
        ],
        "parity": {"vision_features": "passed (rtol=2e-3, atol=2e-3)",
                   "vision_dynamic_grid": "passed (alternate 700x280 image; rtol=2e-3, atol=2e-3)",
                   "token_embeddings": "passed (rtol=1e-5, atol=1e-5)",
                   "prefill_logits": "passed (rtol=3e-3, atol=3e-3)",
                   "cached_decode_logits": "passed (rtol=3e-3, atol=3e-3)"},
        "graphs": graph_reports,
    }
    report["sha256"] = {}
    for path in graph_paths.values():
        rel = str(path.relative_to(output_dir))
        report["sha256"][rel] = _file_sha256(path)
        external = Path(str(path) + ".data")
        if external.is_file():
            report["sha256"][str(external.relative_to(output_dir))] = _file_sha256(external)
    for asset in copied_assets:
        report["sha256"][asset] = _file_sha256(output_dir / asset)
    (output_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nExport and parity report: {output_dir / 'export_report.json'}")


def export_onnx(
    checkpoint_dir: Path,
    output_dir: Path,
    image_path: Path,
    inspection: dict[str, Any],
) -> None:
    """Export a supported LightOnOCR full-attention checkpoint to ONNX."""
    if (
        inspection["architecture"]["inference_architecture"]
        == InferenceArchitecture.QWEN.value
    ):
        _export_qwen(checkpoint_dir, output_dir, image_path, inspection)
        return
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
