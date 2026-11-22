//! Multimodal input processing for LightOnOCR.
//!
//! This module exports two public processors:
//!
//! - [`PixtralProcessor`] for LightOnOCR-2 / LightOnOCR-3-1B NCHW inputs;
//! - [`QwenProcessor`] for LightOnOCR-3-0.8B packed-patch inputs.
//!
//! [`crate::LightOnOCR`] selects one from `config.json` and stores it on
//! [`crate::InferenceBackend`].

mod config;
mod grid;
mod image;
mod output;
mod pixtral;
mod qwen;
mod text;

use std::path::Path;

use crate::Result;
use crate::tokenizer::Tokenizer;

pub use config::{DataFormat, ImageProcessorConfig, ImageSize, ProcessorConfig, ResampleFilter};
pub use grid::VisionGrid;
pub use image::ImageProcessor;
pub use output::ProcessorOutput;
pub use pixtral::PixtralProcessor;
pub use qwen::{QwenProcessor, QwenProcessorOutput};
pub use text::{Message, MessageContent, MessageRole, TextProcessor};

pub(crate) fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_files(
        dir.join("tokenizer.json"),
        dir.join("tokenizer_config.json"),
        Some(&dir.join("special_tokens_map.json")),
    )
}
