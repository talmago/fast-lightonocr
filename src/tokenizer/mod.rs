//! Hugging Face-compatible tokenizer support.
//!
//! The tokenizer owns the assets required for text tokenization:
//! - `tokenizer.json`
//! - `tokenizer_config.json`
//! - optional `special_tokens_map.json`

mod config;
mod special_tokens;

use std::fs::File;
use std::path::Path;

use tokenizers::Tokenizer as HFTokenizer;

use crate::Error;
use crate::Result;

pub use config::{PaddingSide, TokenizerConfig};
pub use special_tokens::{SpecialToken, SpecialTokenIds, SpecialTokensMap};

/// Hugging Face tokenizer wrapper.
///
/// This type owns the tokenizer runtime together with its accompanying
/// configuration and special-token metadata.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: HFTokenizer,
    config: TokenizerConfig,
    special_tokens_map: Option<SpecialTokensMap>,
    special_token_ids: SpecialTokenIds,
}

impl Tokenizer {
    /// Loads tokenizer assets.
    pub fn from_files(
        tokenizer_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        special_tokens_map_path: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer_path = tokenizer_path.as_ref();
        if !tokenizer_path.is_file() {
            return Err(Error::MissingTokenizerAsset {
                path: tokenizer_path.to_path_buf(),
            });
        }

        let inner = HFTokenizer::from_file(tokenizer_path).map_err(|source| {
            Error::TokenizerInitialization {
                path: tokenizer_path.to_path_buf(),
                source,
            }
        })?;

        let config = TokenizerConfig::from_file(config_path)?;

        let special_tokens_map = match special_tokens_map_path {
            Some(path) if path.exists() => {
                let file = File::open(path).map_err(|source| Error::ReadTokenizerAsset {
                    path: path.to_path_buf(),
                    source,
                })?;

                let value: serde_json::Value = serde_json::from_reader(file).map_err(|source| {
                    Error::MalformedTokenizerJson {
                        path: path.to_path_buf(),
                        source,
                    }
                })?;

                let special_tokens_map = serde_json::from_value(value).map_err(|source| {
                    Error::InvalidTokenizerJson {
                        path: path.to_path_buf(),
                        source,
                    }
                })?;

                Some(special_tokens_map)
            }
            _ => None,
        };

        let special_token_ids =
            Self::resolve_special_token_ids(&inner, &config, special_tokens_map.as_ref())?;

        Ok(Self {
            inner,
            config,
            special_tokens_map,
            special_token_ids,
        })
    }

    /// Returns the tokenizer configuration.
    #[must_use]
    pub fn config(&self) -> &TokenizerConfig {
        &self.config
    }

    /// Returns the optional special-token map.
    #[must_use]
    pub fn special_tokens_map(&self) -> Option<&SpecialTokensMap> {
        self.special_tokens_map.as_ref()
    }

    /// Returns tokenizer-defined special token IDs.
    #[must_use]
    pub fn special_token_ids(&self) -> SpecialTokenIds {
        self.special_token_ids
    }

    /// Encodes text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|source| Error::TokenizerEncoding { source })?;

        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    /// Decodes token IDs.
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        self.decode_inner(ids, false)
    }

    /// Decodes token IDs while skipping special tokens.
    pub fn decode_skip_special_tokens(&self, ids: &[i64]) -> Result<String> {
        self.decode_inner(ids, true)
    }

    /// Decodes a batch of token ID sequences.
    pub fn decode_batch(&self, batch: &[Vec<i64>]) -> Result<Vec<String>> {
        self.decode_batch_inner(batch, false)
    }

    /// Decodes a batch of token ID sequences while skipping special tokens.
    pub fn decode_batch_skip_special_tokens(&self, batch: &[Vec<i64>]) -> Result<Vec<String>> {
        self.decode_batch_inner(batch, true)
    }

    /// Encodes a token into its vocabulary ID.
    pub fn token_to_id(&self, token: &str) -> Result<i64> {
        self.inner
            .token_to_id(token)
            .map(i64::from)
            .ok_or_else(|| Error::MissingSpecialTokenId {
                name: "token",
                token: token.to_owned(),
            })
    }

    /// Returns the token string for a vocabulary ID.
    #[must_use]
    pub fn id_to_token(&self, id: i64) -> Option<String> {
        u32::try_from(id)
            .ok()
            .and_then(|id| self.inner.id_to_token(id))
    }

    fn decode_inner(&self, ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids = Self::token_ids_to_u32(ids)?;

        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|source| Error::TokenizerDecoding { source })
    }

    fn decode_batch_inner(
        &self,
        batch: &[Vec<i64>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let ids = batch
            .iter()
            .map(|ids| Self::token_ids_to_u32(ids))
            .collect::<Result<Vec<_>>>()?;

        let id_slices = ids.iter().map(Vec::as_slice).collect::<Vec<_>>();

        self.inner
            .decode_batch(&id_slices, skip_special_tokens)
            .map_err(|source| Error::TokenizerDecoding { source })
    }

    fn token_ids_to_u32(ids: &[i64]) -> Result<Vec<u32>> {
        ids.iter()
            .map(|&id| u32::try_from(id).map_err(|_| Error::InvalidTokenId { token_id: id }))
            .collect()
    }

    fn resolve_special_token_ids(
        tokenizer: &HFTokenizer,
        config: &TokenizerConfig,
        map: Option<&SpecialTokensMap>,
    ) -> Result<SpecialTokenIds> {
        let bos = map
            .and_then(|m| m.bos_token.as_ref().map(SpecialToken::as_str))
            .or(config.bos_token.as_deref());

        let eos = map
            .and_then(|m| m.eos_token.as_ref().map(SpecialToken::as_str))
            .unwrap_or(&config.eos_token);

        let pad = map
            .and_then(|m| m.pad_token.as_ref().map(SpecialToken::as_str))
            .unwrap_or(&config.pad_token);

        let unk = map
            .and_then(|m| m.unk_token.as_ref().map(SpecialToken::as_str))
            .or(config.unk_token.as_deref());

        Ok(SpecialTokenIds {
            bos_token_id: Self::resolve_optional_token_id(tokenizer, "bos_token", bos)?,
            eos_token_id: Self::resolve_required_token_id(tokenizer, "eos_token", eos)?,
            pad_token_id: Self::resolve_required_token_id(tokenizer, "pad_token", pad)?,
            unk_token_id: Self::resolve_optional_token_id(tokenizer, "unk_token", unk)?,
        })
    }

    fn resolve_required_token_id(
        tokenizer: &HFTokenizer,
        name: &'static str,
        token: &str,
    ) -> Result<i64> {
        tokenizer
            .token_to_id(token)
            .map(i64::from)
            .ok_or_else(|| Error::MissingSpecialTokenId {
                name,
                token: token.to_owned(),
            })
    }

    fn resolve_optional_token_id(
        tokenizer: &HFTokenizer,
        name: &'static str,
        token: Option<&str>,
    ) -> Result<Option<i64>> {
        token
            .map(|token| Self::resolve_required_token_id(tokenizer, name, token))
            .transpose()
    }
}
