//! Public API for Fast LightOnOCR.
//!
//! The crate exposes the high-level [`LightOnOCR`] interface for loading
//! pretrained OCR models and processing images, together with the underlying
//! building blocks for advanced use cases.

#![warn(missing_docs)]

pub mod model;
pub mod processor;
pub mod tokenizer;
pub mod util;

pub use model::{
    EmbeddingModel, ExecutionProvider, FinishReason, GenerationConfig, ImageTensor,
    InferenceArchitecture, InferenceBackend, LightOnOCR, LightOnOCROptions, OCRResult,
    PixtralDecoder, PixtralVisionEncoder, QwenDecoder, QwenVisionEncoder, RuntimeOptions,
    StageTimings,
};

pub use processor::{ImageProcessor, ImageProcessorConfig, PixtralProcessor, QwenProcessor};

pub use tokenizer::Tokenizer;
pub use util::{Error, Result};
