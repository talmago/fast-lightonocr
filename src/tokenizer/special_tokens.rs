//! Special-token metadata loaded from tokenizer assets.

use serde::Deserialize;

/// Optional metadata loaded from `special_tokens_map.json`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct SpecialTokensMap {
    /// Optional beginning-of-sequence token entry.
    pub bos_token: Option<SpecialToken>,

    /// Optional end-of-sequence token entry.
    pub eos_token: Option<SpecialToken>,

    /// Optional padding token entry.
    pub pad_token: Option<SpecialToken>,

    /// Optional unknown-token entry.
    pub unk_token: Option<SpecialToken>,

    /// Additional special tokens recorded by Hugging Face assets.
    #[serde(default)]
    pub additional_special_tokens: Vec<SpecialToken>,
}

/// A special token entry represented either as a string or as a Hugging Face
/// added-token object.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum SpecialToken {
    /// Plain special token string.
    Text(String),

    /// Hugging Face added-token object.
    AddedToken {
        /// Token text content.
        content: String,
    },
}

impl SpecialToken {
    /// Returns the textual content of the special token entry.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(token) => token,
            Self::AddedToken { content } => content,
        }
    }
}

/// Resolved special token identifiers from `tokenizer.json` and tokenizer
/// configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    /// Optional beginning-of-sequence token identifier.
    pub bos_token_id: Option<i64>,

    /// End-of-sequence token identifier.
    pub eos_token_id: i64,

    /// Padding token identifier.
    pub pad_token_id: i64,

    /// Optional unknown-token identifier.
    pub unk_token_id: Option<i64>,
}
