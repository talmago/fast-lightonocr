//! Vision encoder configuration and ONNX Runtime wrapper.
//!
//! This module owns the `vision_config` section from `config.json` and the
//! ONNX session that converts [`ImageTensor`](crate::model::ImageTensor)
//! into [`ImageFeatures`](crate::model::ImageFeatures).

mod config;
#[allow(clippy::module_inception)]
mod vision;

pub use config::VisionConfig;
pub use vision::VisionEncoder;
