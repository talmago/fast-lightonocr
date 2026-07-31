"""Python API for Fast LightOnOCR."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ._native import OCRResult as _NativeOCRResult
from ._native import LightOnOCR as _NativeLightOnOCR
from .postprocess import PostProcessResult, Table, TableCell, post_process_text

PathLike = Union[str, Path]

_REQUIRED_JSON_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

_ONNX_FILES = {
    "default": (
        "onnx/embed_tokens.onnx",
        "onnx/embed_tokens.onnx_data*",
        "onnx/vision_encoder.onnx",
        "onnx/vision_encoder.onnx_data*",
        "onnx/decoder_model_merged.onnx",
        "onnx/decoder_model_merged.onnx_data*",
    ),
    "fp16": (
        "onnx/embed_tokens_fp16.onnx",
        "onnx/embed_tokens_fp16.onnx_data*",
        "onnx/vision_encoder_fp16.onnx",
        "onnx/vision_encoder_fp16.onnx_data*",
        "onnx/decoder_model_merged_fp16.onnx",
        "onnx/decoder_model_merged_fp16.onnx_data*",
    ),
    "q4": (
        "onnx/embed_tokens_q4.onnx",
        "onnx/embed_tokens_q4.onnx_data*",
        "onnx/vision_encoder_q4.onnx",
        "onnx/vision_encoder_q4.onnx_data*",
        "onnx/decoder_model_merged_q4.onnx",
        "onnx/decoder_model_merged_q4.onnx_data*",
    ),
}


class LightOnOCR:
    """Thin Python wrapper around the native Rust LightOnOCR engine."""

    def __init__(self, native: _NativeLightOnOCR) -> None:
        self._native = native

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: PathLike,
        revision: Optional[str] = None,
        cache_dir: Optional[PathLike] = None,
        local_files_only: bool = False,
        *,
        preset: str = "default",
        max_new_tokens: Optional[int] = None,
    ) -> "LightOnOCR":
        """Load from a local model directory or download required Hugging Face assets."""

        model_path = Path(model_id_or_path)
        if model_path.exists():
            return cls(_load_model_dir(model_path, preset, max_new_tokens))

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "LightOnOCR.from_pretrained requires the 'huggingface-hub' package"
            ) from exc

        snapshot_path = snapshot_download(
            repo_id=str(model_id_or_path),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=_allow_patterns(preset),
        )

        return cls(_load_model_dir(Path(snapshot_path), preset, max_new_tokens))

    def process(
        self,
        image_path: PathLike,
        system_prompt: Optional[str] = None,
        *,
        post_process: bool = True,
    ) -> OCRResult:
        """Run OCR on an image path."""

        return OCRResult(
            self._native.process(Path(image_path), system_prompt=system_prompt),
            post_process=post_process,
        )

    def process_file(
        self,
        image_path: PathLike,
        system_prompt: Optional[str] = None,
        *,
        post_process: bool = True,
    ) -> OCRResult:
        """Alias for :meth:`process`."""

        return self.process(
            image_path,
            system_prompt=system_prompt,
            post_process=post_process,
        )


class OCRResult:
    """Python-friendly OCR result with optional post-processing."""

    def __init__(self, native: _NativeOCRResult, *, post_process: bool = True) -> None:
        self._native = native
        self.text: str = native.text
        self.token_ids: list[int] = native.token_ids
        self.finish_reason: str = native.finish_reason
        self._postprocessed: Optional[PostProcessResult] = (
            post_process_text(self.text) if post_process else None
        )

    @property
    def clean_text(self) -> str:
        """Raw OCR text with HTML tags stripped."""

        return self._ensure_postprocessed().clean_text

    @property
    def tables(self) -> list[Table]:
        """HTML tables extracted from the OCR output."""

        return self._ensure_postprocessed().tables

    def _ensure_postprocessed(self) -> PostProcessResult:
        if self._postprocessed is None:
            self._postprocessed = post_process_text(self.text)
        return self._postprocessed

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return (
            f"OCRResult(text={self.text!r}, token_ids={len(self.token_ids)}, "
            f"finish_reason={self.finish_reason!r})"
        )


def _allow_patterns(preset: str) -> list[str]:
    try:
        onnx_files = _ONNX_FILES[preset]
    except KeyError as exc:
        raise ValueError(
            f"unknown model preset {preset!r}; expected 'default', 'fp16', or 'q4'"
        ) from exc

    return [*_REQUIRED_JSON_FILES, *onnx_files]


def _load_model_dir(
    model_dir: PathLike,
    preset: str,
    max_new_tokens: Optional[int],
) -> _NativeLightOnOCR:
    return _NativeLightOnOCR._load_model_dir(
        Path(model_dir),
        preset=preset,
        max_new_tokens=max_new_tokens,
        vision_encoder=None,
        embedding=None,
        decoder=None,
    )


__all__ = [
    "LightOnOCR",
    "OCRResult",
    "Table",
    "TableCell",
    "post_process_text",
]
