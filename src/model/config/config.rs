//! Types loaded from `config.json`.

use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::model::config::json::load_json_file;
use crate::model::vision_encoder::VisionConfig;

/// Top-level architecture metadata loaded from `config.json`.
///
/// Future runtime stages use this configuration for shared multimodal metadata
/// without hard-coding ONNX contract constants. Vision, embedding, and decoder
/// component-specific metadata is owned by those subsystems.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelConfig {
    /// Hugging Face architecture names supported by the exported model.
    pub architectures: Vec<String>,

    /// Numeric data type recorded in the model configuration.
    pub dtype: DataType,

    /// End-of-sequence token identifier used by the model wrapper.
    pub eos_token_id: i64,

    /// Token identifier whose embedding is replaced by image features.
    pub image_token_index: i64,

    /// Model family identifier for the top-level multimodal model.
    pub model_type: ModelType,

    /// Whether the multimodal projector includes a bias term.
    pub multimodal_projector_bias: bool,

    /// Padding token identifier used by the model wrapper.
    pub pad_token_id: i64,

    /// Activation used by the multimodal projector.
    pub projector_hidden_act: Activation,

    /// Spatial merge factor applied to vision patches.
    pub spatial_merge_size: usize,

    /// Whether token embeddings and output projection weights are tied.
    pub tie_word_embeddings: bool,

    /// Transformers version that produced the model configuration.
    pub transformers_version: Option<String>,

    /// Vision encoder metadata used internally by the vision subsystem.
    pub(crate) vision_config: VisionConfig,

    /// Hidden-state layer selected from the vision encoder output.
    pub vision_feature_layer: i32,
}

impl ModelConfig {
    /// Loads the top-level LightOnOCR model configuration from `config.json`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_json_file(path)
    }
}

/// Rotary-position-embedding parameters for text and vision components.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RopeParameters {
    /// Base theta value used when constructing rotary embeddings.
    pub rope_theta: f64,

    /// Rotary embedding variant used by the component.
    pub rope_type: RopeType,
}

/// Model family names found in LightOnOCR configuration files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ModelType {
    /// Top-level LightOnOCR multimodal model.
    #[serde(rename = "lighton_ocr")]
    LightOnOcr,

    /// Qwen3 text decoder.
    #[serde(rename = "qwen3")]
    Qwen3,

    /// Pixtral vision encoder and processor family.
    #[serde(rename = "pixtral")]
    Pixtral,
}

/// Numeric data types declared by exported model metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum DataType {
    /// Brain floating point 16-bit values.
    #[serde(rename = "bfloat16")]
    BFloat16,

    /// IEEE floating point 16-bit values.
    #[serde(rename = "float16")]
    Float16,

    /// IEEE floating point 32-bit values.
    #[serde(rename = "float32")]
    Float32,
}

/// Activation functions referenced by model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Activation {
    /// Gaussian error linear unit activation.
    #[serde(rename = "gelu")]
    Gelu,

    /// Sigmoid linear unit activation.
    #[serde(rename = "silu")]
    Silu,
}

/// Rotary embedding variants referenced by configuration files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum RopeType {
    /// Default rotary embedding implementation.
    #[serde(rename = "default")]
    Default,
}
