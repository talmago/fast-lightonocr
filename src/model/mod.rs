//! Model-layer types and runtime boundaries.
//!
//! This module contains the core OCR pipeline, model configuration,
//! ONNX Runtime wrappers, and autoregressive generation.

mod attention;
pub mod config;
pub mod decoder;
pub mod embedding_model;
mod image_features;
mod image_tensor;
mod logits;
pub mod pipeline;
pub mod vision_encoder;

// Shared tensor/value types
pub use attention::AttentionMask;
pub use decoder::DecoderOutput;
pub use embedding_model::InputEmbeddings;
pub use image_features::ImageFeatures;
pub use image_tensor::ImageTensor;
pub use logits::Logits;

// Vision encoder
pub use vision_encoder::{VisionConfig, VisionEncoder};

// Token embedding
pub use embedding_model::{EmbeddingConfig, EmbeddingModel};

// Decoder and generation
pub use decoder::{
    Decoder, DecoderConfig, FinishReason, GenerationConfig, GenerationOutput, KvCache, LayerCache,
};

// High-level OCR pipeline
pub use pipeline::{ExecutionProvider, LightOnOCR, LightOnOCROptions, OCRResult, RuntimeOptions};

/// Backward-compatible alias for [`vision_encoder`].
#[deprecated(note = "use model::vision_encoder instead")]
pub mod vision {
    pub use super::vision_encoder::*;
}

/// Backward-compatible alias for [`embedding_model`].
#[deprecated(note = "use model::embedding_model instead")]
pub mod embedding {
    pub use super::embedding_model::*;
}
