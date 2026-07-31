//! Types loaded from `tokenizer_config.json`.

use std::path::Path;

use serde::Deserialize;

use crate::Result;
use crate::model::config::json::load_json_file;

/// Tokenizer metadata loaded from `tokenizer_config.json`.
///
/// `HFTokenizer` owns this configuration alongside `tokenizer.json` and
/// optional `special_tokens_map.json`. Prompt construction, chat templates,
/// and image placeholders are owned by the prompt subsystem.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TokenizerConfig {
    /// Whether tokenization should add a prefix space before input text.
    pub add_prefix_space: bool,

    /// Optional beginning-of-sequence token string.
    pub bos_token: Option<String>,

    /// Whether decoded text should clean tokenization spaces.
    pub clean_up_tokenization_spaces: bool,

    /// End-of-sequence token string.
    pub eos_token: String,

    /// Optional maximum tokenized length override.
    pub max_length: Option<usize>,

    /// Maximum model sequence length recorded by the tokenizer.
    pub model_max_length: usize,

    /// Optional padding multiple requested by the tokenizer.
    pub pad_to_multiple_of: Option<usize>,

    /// Padding token string.
    pub pad_token: String,

    /// Padding token type identifier.
    pub pad_token_type_id: u32,

    /// Side on which tokenizer padding is applied.
    pub padding_side: PaddingSide,

    /// Whether special tokens should be split during tokenization.
    pub split_special_tokens: bool,

    /// Optional unknown-token string.
    pub unk_token: Option<String>,
}

impl TokenizerConfig {
    /// Loads tokenizer configuration from an explicit JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        load_json_file(path)
    }
}

/// Tokenizer padding side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum PaddingSide {
    /// Add padding before token content.
    #[serde(rename = "left")]
    Left,

    /// Add padding after token content.
    #[serde(rename = "right")]
    Right,
}
