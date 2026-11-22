//! Vision encoder configuration, input/output tensors, and ONNX Runtime wrapper.
//!
//! This module owns the `vision_config` section from `config.json` and the
//! ONNX session that converts [`ImageTensor`](crate::model::ImageTensor)
//! into [`ImageFeatures`](crate::model::ImageFeatures).

mod config;
mod encoder;
mod input;
mod output;
mod qwen;

pub use config::VisionConfig;
pub use encoder::PixtralVisionEncoder;
pub use input::ImageTensor;
pub use output::ImageFeatures;
pub use qwen::QwenVisionEncoder;
