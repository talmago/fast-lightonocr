//! Types loaded from `processor_config.json`.

use serde::{Deserialize, Deserializer};
use std::fs;
use std::path::Path;

use crate::{Error, Result};

/// Metadata loaded from `processor_config.json`.
///
/// This configuration describes the Hugging Face processor used by the model.
/// It contains the image preprocessing configuration together with processor
/// metadata required to prepare model inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessorConfig {
    /// Vision patch size in pixels.
    pub patch_size: usize,

    /// Spatial merge factor applied to the patch grid before producing vision
    /// features.
    ///
    /// For example, a 34×34 patch grid with a merge size of 2 becomes a
    /// 17×17 feature grid.
    pub spatial_merge_size: usize,

    /// Token inserted for each vision feature.
    pub image_token: String,

    /// Token separating adjacent rows of vision features.
    pub image_break_token: String,

    /// Token terminating the vision feature sequence.
    pub image_end_token: String,

    /// Image preprocessing configuration.
    pub image_processor: ImageProcessorConfig,
}

impl<'de> Deserialize<'de> for ProcessorConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawProcessorConfig::deserialize(deserializer)?;
        let raw_image_processor = raw.image_processor;

        let spatial_merge_size = raw
            .spatial_merge_size
            .or(raw_image_processor.spatial_merge_size)
            .ok_or_else(|| serde::de::Error::missing_field("spatial_merge_size"))?;
        let image_token = raw
            .image_token
            .or(raw_image_processor.image_token)
            .ok_or_else(|| serde::de::Error::missing_field("image_token"))?;
        let image_break_token = raw
            .image_break_token
            .or(raw_image_processor.image_break_token)
            .ok_or_else(|| serde::de::Error::missing_field("image_break_token"))?;
        let image_end_token = raw
            .image_end_token
            .or(raw_image_processor.image_end_token)
            .ok_or_else(|| serde::de::Error::missing_field("image_end_token"))?;

        let image_processor = raw_image_processor.config;
        let patch_size = raw.patch_size.unwrap_or(image_processor.patch_size);

        Ok(Self {
            patch_size,
            spatial_merge_size,
            image_token,
            image_break_token,
            image_end_token,
            image_processor,
        })
    }
}

#[derive(Debug, Deserialize)]
struct RawProcessorConfig {
    patch_size: Option<usize>,
    spatial_merge_size: Option<usize>,
    image_token: Option<String>,
    image_break_token: Option<String>,
    image_end_token: Option<String>,
    image_processor: RawImageProcessorConfig,
}

#[derive(Debug, Deserialize)]
struct RawImageProcessorConfig {
    #[serde(flatten)]
    config: ImageProcessorConfig,
    spatial_merge_size: Option<usize>,
    image_token: Option<String>,
    image_break_token: Option<String>,
    image_end_token: Option<String>,
}

impl ProcessorConfig {
    /// Loads a processor configuration from a JSON file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let contents = fs::read_to_string(&path).map_err(|source| {
            if source.kind() == std::io::ErrorKind::NotFound {
                Error::MissingProcessorAsset { path: path.clone() }
            } else {
                Error::ReadProcessorAsset {
                    path: path.clone(),
                    source,
                }
            }
        })?;

        Ok(
            serde_json::from_str(&contents).map_err(|source| match source.classify() {
                serde_json::error::Category::Syntax | serde_json::error::Category::Eof => {
                    Error::MalformedProcessorJson {
                        path: path.clone(),
                        source,
                    }
                }
                serde_json::error::Category::Data | serde_json::error::Category::Io => {
                    Error::InvalidProcessorJson {
                        path: path.clone(),
                        source,
                    }
                }
            })?,
        )
    }

    #[must_use]
    /// Returns the nested image preprocessing configuration.
    pub fn image_processor(&self) -> &ImageProcessorConfig {
        &self.image_processor
    }

    /// Returns the vision encoder patch size.
    #[must_use]
    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Returns the spatial merge factor applied to the patch grid.
    #[must_use]
    pub fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }

    /// Returns the placeholder token representing a single vision feature.
    #[must_use]
    pub fn image_token(&self) -> &str {
        &self.image_token
    }

    /// Returns the token separating rows of vision features.
    #[must_use]
    pub fn image_break_token(&self) -> &str {
        &self.image_break_token
    }

    /// Returns the token marking the end of the vision feature sequence.
    #[must_use]
    pub fn image_end_token(&self) -> &str {
        &self.image_end_token
    }
}

/// Image preprocessing settings nested inside `processor_config.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ImageProcessorConfig {
    /// Tensor channel layout emitted by preprocessing.
    pub data_format: DataFormat,

    /// Whether missing size dimensions default to a square resize target.
    pub default_to_square: bool,

    /// Whether the image processor converts input images to RGB.
    pub do_convert_rgb: bool,

    /// Whether normalized channel values are produced.
    pub do_normalize: bool,

    /// Whether raw pixel values are rescaled before normalization.
    pub do_rescale: bool,

    /// Whether input images are resized before patch extraction.
    pub do_resize: bool,

    /// Per-channel mean values used during normalization.
    pub image_mean: Vec<f32>,

    /// Per-channel standard deviation values used during normalization.
    pub image_std: Vec<f32>,

    /// Patch size used by the image processor.
    pub patch_size: usize,

    /// Resampling filter used during image resizing.
    pub resample: ResampleFilter,

    /// Multiplicative factor applied when rescaling pixel values.
    pub rescale_factor: f32,

    /// Target image sizing policy.
    pub size: ImageSize,
}

impl ImageProcessorConfig {
    /// Returns the emitted tensor channel layout.
    #[must_use]
    pub fn data_format(&self) -> DataFormat {
        self.data_format
    }

    /// Returns whether missing image size dimensions default to a square.
    #[must_use]
    pub fn default_to_square(&self) -> bool {
        self.default_to_square
    }

    /// Returns whether input images are converted to RGB.
    #[must_use]
    pub fn do_convert_rgb(&self) -> bool {
        self.do_convert_rgb
    }

    /// Returns whether channel normalization is applied.
    #[must_use]
    pub fn do_normalize(&self) -> bool {
        self.do_normalize
    }

    /// Returns whether raw pixel values are rescaled.
    #[must_use]
    pub fn do_rescale(&self) -> bool {
        self.do_rescale
    }

    /// Returns whether images are resized before tensor conversion.
    #[must_use]
    pub fn do_resize(&self) -> bool {
        self.do_resize
    }

    /// Returns the per-channel normalization means.
    #[must_use]
    pub fn image_mean(&self) -> &[f32] {
        &self.image_mean
    }

    /// Returns the per-channel normalization standard deviations.
    #[must_use]
    pub fn image_std(&self) -> &[f32] {
        &self.image_std
    }

    /// Returns the configured resize filter.
    #[must_use]
    pub fn resample(&self) -> ResampleFilter {
        self.resample
    }

    /// Returns the multiplicative pixel rescale factor.
    #[must_use]
    pub fn rescale_factor(&self) -> f32 {
        self.rescale_factor
    }

    /// Returns the target image sizing policy.
    #[must_use]
    pub fn size(&self) -> &ImageSize {
        &self.size
    }
}

/// Target image sizing policy for the image processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct ImageSize {
    /// Longest image edge after resizing.
    pub longest_edge: usize,
}

/// Tensor channel layout emitted by the image processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum DataFormat {
    /// Channels-first tensor layout, `(channels, height, width)`.
    #[serde(rename = "channels_first")]
    ChannelsFirst,

    /// Channels-last tensor layout, `(height, width, channels)`.
    #[serde(rename = "channels_last")]
    ChannelsLast,
}

/// Image resize filters encoded by Hugging Face processor configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleFilter {
    /// Bicubic resize filter, encoded as `3` in processor configuration.
    Bicubic,
}

impl<'de> Deserialize<'de> for ResampleFilter {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        match value {
            3 => Ok(Self::Bicubic),
            other => Err(serde::de::Error::custom(format!(
                "unsupported resample filter value {other}"
            ))),
        }
    }
}
