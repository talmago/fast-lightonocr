//! Decoder, autoregressive generation, attention masks, logits, and KV-cache.
//!
//! This module owns the decoder-specific subset of `text_config` from
//! `config.json`, the [`KVCache`] representation, the ONNX Runtime wrapper,
//! and the autoregressive generation engine built on top of the decoder.

mod attention;
mod config;
#[cfg(feature = "cuda")]
mod cuda_backend;
#[allow(clippy::module_inception)]
mod decoder;
mod generation;
mod kv_cache;
mod logits;
mod output;

pub use attention::AttentionMask;
pub use config::{DecoderConfig, GenerationConfig, LayerType};
pub use decoder::Decoder;
pub use generation::{FinishReason, GenerationOutput};
pub use kv_cache::{KVCache, LayerCache};
pub use logits::Logits;
pub use output::DecoderOutput;
