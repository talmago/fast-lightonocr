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
    Decoder, EmbeddingModel, ExecutionProvider, FinishReason, GenerationConfig, ImageTensor,
    LightOnOCR, LightOnOCROptions, OCRResult, RuntimeOptions, StageTimings, VisionEncoder,
};

pub use processor::{ImageProcessor, ImageProcessorConfig, Processor};

pub use tokenizer::Tokenizer;
pub use util::{Error, Result};
