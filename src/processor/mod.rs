//! Multimodal input processing for LightOnOCR.
//!
//! The processor subsystem owns `processor_config.json` and converts text and
//! images into model-ready inputs for OCR inference.

mod config;
mod grid;
mod image;
mod output;
mod text;

use ::image::DynamicImage;
use std::path::Path;

use crate::model::AttentionMask;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

pub use config::{DataFormat, ImageProcessorConfig, ImageSize, ProcessorConfig, ResampleFilter};
pub use grid::VisionGrid;
pub use image::ImageProcessor;
pub use output::ProcessorOutput;
pub use text::{Message, MessageContent, MessageRole, TextProcessor};

/// High-level multimodal processor.
///
/// The processor orchestrates text and image preprocessing to produce
/// model-ready inputs for inference.
#[derive(Debug, Clone)]
pub struct Processor {
    config: ProcessorConfig,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
}

impl Processor {
    /// Creates a processor.
    #[must_use]
    pub fn new(
        config: ProcessorConfig,
        tokenizer: Tokenizer,
        image_processor: ImageProcessor,
    ) -> Self {
        let text_processor = TextProcessor::new(
            tokenizer,
            config.image_token(),
            config.image_break_token(),
            config.image_end_token(),
        );

        Self {
            config,
            image_processor,
            text_processor,
        }
    }

    /// Loads a processor from a Hugging Face model directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();

        let tokenizer = Tokenizer::from_files(
            dir.join("tokenizer.json"),
            dir.join("tokenizer_config.json"),
            Some(&dir.join("special_tokens_map.json")),
        )?;

        let config = ProcessorConfig::from_file(dir.join("processor_config.json"))?;

        let image_processor = ImageProcessor::new(config.image_processor().clone())?;

        Ok(Self::new(config, tokenizer, image_processor))
    }

    /// Returns the processor configuration.
    #[must_use]
    pub fn config(&self) -> &ProcessorConfig {
        &self.config
    }

    /// Returns the image processor.
    #[must_use]
    pub fn image_processor(&self) -> &ImageProcessor {
        &self.image_processor
    }

    /// Returns the text processor.
    #[must_use]
    pub fn text_processor(&self) -> &TextProcessor {
        &self.text_processor
    }

    /// Processes a multimodal conversation into model-ready inputs.
    pub fn process(
        &self,
        messages: &[Message],
        images: &[DynamicImage],
    ) -> Result<ProcessorOutput> {
        let image = images.first().ok_or_else(|| Error::ImageProcessing {
            reason: "at least one image is required".to_owned(),
        })?;

        let pixel_values = self.image_processor.process_image(image)?;

        let vision_grid = VisionGrid::from_image_size(
            pixel_values.width(),
            pixel_values.height(),
            self.config.patch_size(),
            self.config.spatial_merge_size(),
        )?;

        let input_ids = self.text_processor.process(messages, vision_grid)?;

        let attention_mask = AttentionMask::ones(input_ids.len());

        Ok(ProcessorOutput {
            input_ids,
            attention_mask,
            pixel_values,
            vision_grid,
        })
    }
}
