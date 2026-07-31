//! Decoder configuration loaded from `config.json`.

use std::path::Path;

use serde::{Deserialize, Deserializer};

use crate::Result;
use crate::model::config::json::load_json_file;
use crate::model::config::{Activation, DataType, ModelType, RopeParameters};

/// Default maximum number of tokens generated in a single inference request.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;

/// Generation configuration.
///
/// Most fields are loaded from Hugging Face's `generation_config.json`.
/// Runtime-only options, such as `max_new_tokens`, are initialized with
/// sensible defaults when the configuration is loaded.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GenerationConfig {
    // ---------------------------------------------------------------------
    // Runtime options
    // ---------------------------------------------------------------------
    /// Maximum number of new tokens to generate.
    ///
    /// This value is not loaded from `generation_config.json`.
    #[serde(skip)]
    pub max_new_tokens: usize,

    // ---------------------------------------------------------------------
    // Hugging Face generation_config.json
    // ---------------------------------------------------------------------
    /// Beginning-of-sequence token identifier used by generation.
    pub bos_token_id: i64,

    /// Whether the source model configuration enables sampling.
    pub do_sample: bool,

    /// End-of-sequence token identifiers that stop generation.
    #[serde(rename = "eos_token_id", deserialize_with = "deserialize_token_ids")]
    pub eos_token_ids: Vec<i64>,

    /// Padding token identifier used during generation.
    pub pad_token_id: i64,

    /// Sampling temperature from the source generation defaults.
    pub temperature: f32,

    /// Top-k sampling cutoff from the source generation defaults.
    pub top_k: u32,

    /// Top-p nucleus sampling threshold from the source generation defaults.
    pub top_p: f32,

    /// Transformers version that produced the generation configuration.
    pub transformers_version: Option<String>,

    /// Whether Hugging Face remote code was trusted for generation.
    pub trust_remote_code: bool,
}

impl GenerationConfig {
    /// Loads generation configuration from an explicit JSON file.
    ///
    /// Runtime-only options are initialized with their default values after
    /// loading the Hugging Face configuration.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut config: Self = load_json_file(path)?;
        config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
        Ok(config)
    }

    /// Sets the maximum number of new tokens to generate.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Updates the maximum number of new tokens to generate.
    pub fn set_max_new_tokens(&mut self, max_new_tokens: usize) {
        self.max_new_tokens = max_new_tokens;
    }
}

/// Decoder metadata nested inside `config.json`.
///
/// `DecoderConfig` is extracted from the nested `text_config` section because
/// it defines the ONNX decoder contract: hidden size, vocabulary size, layer
/// count, KV-cache head count, and per-head dimension.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderConfig {
    /// Whether decoder attention projections include bias terms.
    pub attention_bias: bool,

    /// Dropout configured for decoder attention.
    pub attention_dropout: f32,

    /// Beginning-of-sequence token identifier recorded for the text model.
    pub bos_token_id: Option<i64>,

    /// Numeric data type recorded for decoder weights.
    pub dtype: DataType,

    /// End-of-sequence token identifier recorded for the text model.
    pub eos_token_id: Option<i64>,

    /// Placeholder token identifier replaced by image features before decoding.
    #[serde(default)]
    pub image_token_index: i64,

    /// Per-head decoder attention dimension.
    pub head_dim: usize,

    /// Decoder feed-forward activation function.
    pub hidden_act: Activation,

    /// Decoder hidden size used by embeddings and logits input states.
    pub hidden_size: usize,

    /// Initializer range recorded by the model configuration.
    pub initializer_range: f32,

    /// Decoder feed-forward intermediate size.
    pub intermediate_size: usize,

    /// Attention implementation used by each decoder layer.
    pub layer_types: Vec<LayerType>,

    /// Maximum configured position embedding length.
    pub max_position_embeddings: usize,

    /// Number of layers using the configured maximum attention window.
    pub max_window_layers: usize,

    /// Model family identifier for the decoder.
    pub model_type: ModelType,

    /// Number of decoder attention heads.
    pub num_attention_heads: usize,

    /// Number of decoder transformer layers.
    pub num_hidden_layers: usize,

    /// Number of key/value heads stored in the KV cache.
    pub num_key_value_heads: usize,

    /// Padding token identifier recorded for the text model.
    pub pad_token_id: Option<i64>,

    /// Epsilon used by decoder RMS normalization.
    pub rms_norm_eps: f64,

    /// Rotary-position-embedding parameters for the decoder.
    pub rope_parameters: RopeParameters,

    /// Optional sliding-window attention size.
    pub sliding_window: Option<usize>,

    /// Whether token embeddings and output projection weights are tied.
    pub tie_word_embeddings: bool,

    /// Whether the decoder is configured to use KV caching.
    pub use_cache: bool,

    /// Whether query/key normalization is enabled.
    pub use_qk_norm: bool,

    /// Whether sliding-window attention is enabled.
    pub use_sliding_window: bool,

    /// Decoder vocabulary size used by the logits tensor.
    pub vocab_size: usize,
}

impl DecoderConfig {
    /// Loads decoder configuration from an explicit JSON file.
    ///
    /// The method reads `config.json` and extracts only the decoder-related
    /// subset of nested `text_config`. Shared OCR configuration remains owned
    /// by [`ModelConfig`](crate::model::config::ModelConfig).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let envelope = load_json_file::<DecoderConfigEnvelope>(path)?;
        let mut config = envelope.text_config;
        config.image_token_index = envelope.image_token_index;
        Ok(config)
    }
}

/// Decoder attention layer kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum LayerType {
    /// Full self-attention over the available sequence context.
    #[serde(rename = "full_attention")]
    FullAttention,
}

#[derive(Debug, Deserialize)]
struct DecoderConfigEnvelope {
    image_token_index: i64,
    text_config: DecoderConfig,
}

fn deserialize_token_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<i64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TokenIds {
        Single(i64),
        Multiple(Vec<i64>),
    }

    match TokenIds::deserialize(deserializer)? {
        TokenIds::Single(token_id) => Ok(vec![token_id]),
        TokenIds::Multiple(token_ids) => Ok(token_ids),
    }
}
