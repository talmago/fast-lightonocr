//! Strongly typed configuration loading for LightOnOCR model assets.
//!
//! The public entry point is [`ModelConfig`], which loads shared Hugging Face
//! model metadata from `config.json`.

#[allow(clippy::module_inception)]
mod config;
pub(crate) mod json;

pub use config::{Activation, DataType, ModelConfig, ModelType, RopeParameters, RopeType};
