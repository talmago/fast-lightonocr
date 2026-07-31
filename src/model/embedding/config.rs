//! Embedding model configuration loaded from `config.json`.

use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::model::config::json::load_json_file;
use crate::model::config::{DataType, ModelType};

/// Configuration values required by the token embedding ONNX model.
///
/// `EmbeddingConfig` is extracted from the nested `text_config` section in
/// `config.json`, but it is owned by the embedding subsystem because it defines
/// the embedding model contract: vocabulary size and output hidden size.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct EmbeddingConfig {
    /// Numeric data type recorded for the text model weights.
    pub dtype: DataType,

    /// Model family identifier for the text model.
    pub model_type: ModelType,

    /// Output embedding dimension expected from the ONNX model.
    pub hidden_size: usize,

    /// Valid vocabulary size for input token IDs.
    pub vocab_size: usize,
}

impl EmbeddingConfig {
    /// Loads embedding configuration from an explicit JSON file.
    ///
    /// The method reads `config.json` and extracts only the embedding-related
    /// subset of nested `text_config`. Shared OCR configuration remains owned
    /// by [`ModelConfig`](crate::model::config::ModelConfig).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_json_file::<EmbeddingConfigEnvelope>(path).map(|envelope| envelope.text_config)
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingConfigEnvelope {
    text_config: EmbeddingConfig,
}
