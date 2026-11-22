//! Pixtral NCHW image processor, chat template, and row-wise image tokens.

use std::path::Path;

use ::image::DynamicImage;

use crate::model::AttentionMask;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

use super::{ImageProcessor, Message, ProcessorConfig, ProcessorOutput, TextProcessor, VisionGrid};

/// Pixtral multimodal processor used by LightOnOCR-2 and LightOnOCR-3-1B.
///
/// The processor converts a chat turn plus a page image into:
///
/// - an NCHW [`crate::ImageTensor`];
/// - token IDs with a rectangular `<|image_pad|>` grid, row breaks, and
///   `<|vision_end|>`;
/// - a matching attention mask.
#[derive(Debug, Clone)]
pub struct PixtralProcessor {
    config: ProcessorConfig,
    image_processor: ImageProcessor,
    text_processor: TextProcessor,
}

impl PixtralProcessor {
    /// Creates a processor from parsed Pixtral configuration and a tokenizer.
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

    /// Loads Pixtral processor assets from a Hugging Face model directory.
    ///
    /// Expects `processor_config.json`, `tokenizer.json`, and
    /// `tokenizer_config.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tokenizer = super::load_tokenizer(dir)?;
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

    /// Returns the tokenizer used by the text processor.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        self.text_processor.tokenizer()
    }

    /// Processes a multimodal conversation into Pixtral model-ready inputs.
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
        let input_ids = self.text_processor.process(messages)?;
        let input_ids = self
            .text_processor
            .expand_image_placeholders(&input_ids, vision_grid)?;
        let attention_mask = AttentionMask::ones(input_ids.len());

        Ok(ProcessorOutput {
            input_ids,
            attention_mask,
            pixel_values,
            vision_grid,
        })
    }
}
