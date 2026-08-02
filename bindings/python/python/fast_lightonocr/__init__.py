"""Python API for Fast LightOnOCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, Union

from ._native import LightOnOCR as _NativeLightOnOCR
from .parser import (
    Document,
    ParserOptions,
    Table,
    TableCell,
    TableFormat,
    parse_document,
)

Preset: TypeAlias = Literal["default", "fp16", "q4"]

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


def _download_patterns(preset: Preset) -> list[str]:
    """Return the Hugging Face download patterns for a model preset.

    The returned patterns include the shared configuration files together
    with the ONNX model files required for the selected preset.

    Args:
        preset:
            Model preset to download.

    Returns:
        List of glob patterns passed to
        :func:`huggingface_hub.snapshot_download`.

    Raises:
        ValueError:
            If the preset is not supported.
    """

    try:
        onnx_files = _ONNX_FILES[preset]
    except KeyError as exc:
        supported = ", ".join(sorted(_ONNX_FILES))
        raise ValueError(
            f"Unknown model preset {preset!r}. Expected one of: {supported}."
        ) from exc

    return [
        *_REQUIRED_JSON_FILES,
        *onnx_files,
    ]


@dataclass(frozen=True)
class OCRResult:
    """Result of an OCR inference.

    Attributes:
        text:
            Raw OCR output produced by the model.

        token_ids:
            Generated token identifiers.

        finish_reason:
            Reason generation stopped.

        document:
            Parsed OCR document with structured tables.
    """

    text: str
    token_ids: list[int]
    finish_reason: str
    document: Document

    @property
    def tables(self) -> list[Table]:
        """Tables extracted from the parsed document."""

        return self.document.tables

    def __str__(self) -> str:
        """Return the rendered OCR document."""

        return str(self.document)


class LightOnOCR:
    """High-level Python interface for the Fast LightOnOCR inference engine."""

    def __init__(self, native: _NativeLightOnOCR) -> None:
        """Initialize a LightOnOCR wrapper around the native Rust engine."""

        self._native = native

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: PathLike,
        revision: str | None = None,
        cache_dir: PathLike | None = None,
        local_files_only: bool = False,
        *,
        preset: Preset = "default",
        max_new_tokens: int | None = None,
    ) -> LightOnOCR:
        """Load a pretrained LightOnOCR model.

        If ``model_id_or_path`` refers to an existing local directory, the
        model is loaded directly from disk.

        Otherwise, the required model assets are downloaded from Hugging Face
        Hub before constructing the native inference engine.

        Args:
            model_id_or_path:
                Local model directory or Hugging Face repository ID.

            revision:
                Optional Hugging Face revision.

            cache_dir:
                Optional Hugging Face cache directory.

            local_files_only:
                If ``True``, only local cached files are used.

            preset:
                ONNX model preset to load.

            max_new_tokens:
                Optional override for the model generation configuration.

        Returns:
            Loaded :class:`LightOnOCR` instance.
        """

        model_path = Path(model_id_or_path)

        if model_path.exists():
            return cls(
                _NativeLightOnOCR._load_model_dir(
                    model_path,
                    preset=preset,
                    max_new_tokens=max_new_tokens,
                    vision_encoder=None,
                    embedding=None,
                    decoder=None,
                )
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "LightOnOCR.from_pretrained requires the 'huggingface-hub' package."
            ) from exc

        snapshot_path = snapshot_download(
            repo_id=str(model_id_or_path),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=_download_patterns(preset),
        )

        return cls(
            _NativeLightOnOCR._load_model_dir(
                Path(snapshot_path),
                preset=preset,
                max_new_tokens=max_new_tokens,
                vision_encoder=None,
                embedding=None,
                decoder=None,
            )
        )

    def process(
        self,
        image_path: PathLike,
        system_prompt: str | None = None,
        *,
        table_format: TableFormat = "grid",
    ) -> OCRResult:
        """Run OCR on an image.

        Args:
            image_path:
                Path to the input image.

            system_prompt:
                Optional system prompt overriding the model default.

            table_format:
                Table rendering format passed to the document parser. Any
                format supported by ``tabulate`` may be used.

        Returns:
            OCR inference result.
        """

        native = self._native.process(
            Path(image_path),
            system_prompt=system_prompt,
        )

        return OCRResult(
            text=native.text,
            token_ids=native.token_ids,
            finish_reason=native.finish_reason,
            document=parse_document(
                native.text,
                ParserOptions(
                    table_format=table_format,
                ),
            ),
        )

    def process_file(
        self,
        image_path: PathLike,
        system_prompt: str | None = None,
        *,
        table_format: str = "grid",
    ) -> OCRResult:
        """Alias for :meth:`process`."""

        return self.process(
            image_path=image_path,
            system_prompt=system_prompt,
            table_format=table_format,
        )


__all__ = ["LightOnOCR", "OCRResult", "Table", "TableCell", "Document"]
