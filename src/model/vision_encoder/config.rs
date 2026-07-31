//! Vision encoder configuration loaded from `config.json`.

use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::model::{Activation, DataType, ModelType, RopeParameters};
use crate::util::json::load_json_file;

/// Vision encoder metadata nested inside `config.json`.
///
/// `VisionConfig` is owned by the vision subsystem because it defines the
/// ONNX vision model contract: expected input channels, patch metadata, hidden
/// size, and related architecture values used when validating image features.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct VisionConfig {
    /// Dropout configured for vision attention.
    pub attention_dropout: f32,

    /// Numeric data type recorded for vision encoder weights.
    pub dtype: DataType,

    /// Per-head vision attention dimension.
    pub head_dim: usize,

    /// Vision encoder feed-forward activation function.
    pub hidden_act: Activation,

    /// Vision hidden size used for image feature vectors.
    pub hidden_size: usize,

    /// Nominal image size recorded by the vision encoder configuration.
    pub image_size: usize,

    /// Initializer range recorded by the model configuration.
    pub initializer_range: f32,

    /// Vision feed-forward intermediate size.
    pub intermediate_size: usize,

    /// Model family identifier for the vision encoder.
    pub model_type: ModelType,

    /// Number of vision encoder attention heads.
    pub num_attention_heads: usize,

    /// Number of channels expected by the vision encoder input tensor.
    pub num_channels: usize,

    /// Number of vision transformer layers.
    pub num_hidden_layers: usize,

    /// Patch size consumed by the vision encoder.
    pub patch_size: usize,

    /// Rotary-position-embedding parameters for the vision encoder.
    pub rope_parameters: RopeParameters,
}

impl VisionConfig {
    /// Loads the vision encoder configuration from an explicit JSON file.
    ///
    /// The method reads `config.json` and extracts only the nested
    /// `vision_config` section.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_json_file::<VisionConfigEnvelope>(path).map(|envelope| envelope.vision_config)
    }
}

#[derive(Debug, Deserialize)]
struct VisionConfigEnvelope {
    vision_config: VisionConfig,
}
