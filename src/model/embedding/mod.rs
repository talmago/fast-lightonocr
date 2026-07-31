//! Token embedding model configuration and ONNX Runtime wrapper.
//!
//! This module owns the embedding-related subset of `text_config` from
//! `config.json` and the ONNX session that converts model-ready token ID
//! sequences into
//! [`InputEmbeddings`](crate::model::InputEmbeddings).

mod config;
#[allow(clippy::module_inception)]
mod embedding;

pub use config::EmbeddingConfig;
pub use embedding::EmbeddingModel;
