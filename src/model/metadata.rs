//! Shared metadata values loaded from model configuration files.

use serde::Deserialize;

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
