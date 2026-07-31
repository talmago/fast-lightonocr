//! Model-layer types and runtime boundaries.
//!
//! This module contains the core OCR pipeline, model configuration,
//! ONNX Runtime wrappers, and autoregressive generation.

mod attention;
pub mod config;
pub mod decoder;
mod decoder_output;
pub mod embedding;
mod embeddings;
mod image_features;
mod image_tensor;
mod logits;
pub mod pipeline;
pub mod vision;

// Shared tensor/value types
pub use attention::AttentionMask;
pub use decoder_output::DecoderOutput;
pub use embeddings::InputEmbeddings;
pub use image_features::ImageFeatures;
pub use image_tensor::ImageTensor;
pub use logits::Logits;

// Vision encoder
pub use vision::{VisionConfig, VisionEncoder};

// Token embedding
pub use embedding::{EmbeddingConfig, EmbeddingModel};

// Decoder and generation
pub use decoder::{
    Decoder, DecoderConfig, FinishReason, GenerationConfig, GenerationOutput, KvCache, LayerCache,
};

// High-level OCR pipeline
pub use pipeline::{ExecutionProvider, LightOnOCR, LightOnOCROptions, OCRResult, RuntimeOptions};
