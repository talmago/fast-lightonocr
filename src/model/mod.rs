//! Model-layer types and runtime boundaries.
//!
//! This module contains the core OCR pipeline, model configuration,
//! ONNX Runtime wrappers, and autoregressive generation.

pub mod decoder;
pub mod embedding_model;
mod metadata;
pub mod pipeline;
pub mod vision_encoder;

// Shared tensor/value types
pub use decoder::{AttentionMask, DecoderOutput, Logits};
pub use embedding_model::InputEmbeddings;
pub use metadata::{Activation, DataType, ModelType, RopeParameters, RopeType};
pub use vision_encoder::{ImageFeatures, ImageTensor};

// Vision encoder
pub use vision_encoder::{VisionConfig, VisionEncoder};

// Token embedding
pub use embedding_model::{EmbeddingConfig, EmbeddingModel};

// Decoder and generation
pub use decoder::{
    Decoder, DecoderConfig, FinishReason, GenerationConfig, GenerationOutput, KVCache, LayerCache,
};

// High-level OCR pipeline
pub use crate::util::{ExecutionProvider, RuntimeOptions};
pub use pipeline::{LightOnOCR, LightOnOCROptions, OCRResult, StageTimings};

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
