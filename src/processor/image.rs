//! Image preprocessing implementation.

use std::path::{Path, PathBuf};

use crate::model::ImageTensor;
use crate::processor::{DataFormat, ImageProcessorConfig, ResampleFilter};
use crate::{Error, Result};
use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};

const CHANNELS: usize = 3;

/// Image processor for LightOnOCR vision inputs.
///
/// `ImageProcessor` loads `processor_config.json` from a model directory and
/// applies the Hugging Face Pixtral image preprocessing steps adapted to the
/// LightOnOCR merged-patch contract: image loading, RGB conversion,
/// aspect-preserving resize with merge-aligned output dimensions, rescaling,
/// normalization, batch padding, and NCHW tensor conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessor {
    config: ImageProcessorConfig,
}

impl ImageProcessor {
    /// Creates an image processor from a parsed processor configuration.
    pub fn new(config: ImageProcessorConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self { config })
    }

    /// Returns the image processor configuration.
    pub fn config(&self) -> &ImageProcessorConfig {
        &self.config
    }

    /// Loads an image from disk and preprocesses it into `ImageTensor`.
    pub fn process_path(&self, path: impl AsRef<Path>) -> Result<ImageTensor> {
        let image = image::open(path.as_ref()).map_err(|source| Error::ImageLoad {
            path: PathBuf::from(path.as_ref()),
            source,
        })?;
        self.process_image(&image)
    }

    /// Preprocesses one in-memory image into a single-item `ImageTensor` batch.
    pub fn process_image(&self, image: &DynamicImage) -> Result<ImageTensor> {
        self.process_images(std::slice::from_ref(image))
    }

    /// Preprocesses an image batch into padded `ImageTensor`.
    ///
    /// Every image is resized independently to a merge-aligned shape. The
    /// batch is then padded on the bottom and right with zeros after
    /// normalization so that all samples share one `(height, width)`.
    pub fn process_images(&self, images: &[DynamicImage]) -> Result<ImageTensor> {
        if images.is_empty() {
            return Err(Error::ImageProcessing {
                reason: "at least one image is required".to_owned(),
            });
        }

        let mut resized = Vec::with_capacity(images.len());
        let mut max_height = 0;
        let mut max_width = 0;

        for image in images {
            let rgb = self.resize_image(&image.to_rgb8())?;
            max_height = max_height.max(rgb.height() as usize);
            max_width = max_width.max(rgb.width() as usize);
            resized.push(rgb);
        }

        let mut data = vec![0.0; images.len() * CHANNELS * max_height * max_width];
        for (batch_index, image) in resized.iter().enumerate() {
            write_nchw_normalized(
                &mut data,
                batch_index,
                max_height,
                max_width,
                image,
                &self.config,
            );
        }

        ImageTensor::new(data, images.len(), CHANNELS, max_height, max_width)
    }

    fn resize_image(&self, image: &RgbImage) -> Result<RgbImage> {
        let (target_height, target_width) = resize_output_size(
            image.height() as usize,
            image.width() as usize,
            &self.config,
        );

        if !self.config.do_resize
            && target_height == image.height() as usize
            && target_width == image.width() as usize
        {
            return Ok(image.clone());
        }

        Ok(image::imageops::resize(
            image,
            target_width as u32,
            target_height as u32,
            filter_type(self.config.resample),
        ))
    }
}

fn write_nchw_normalized(
    data: &mut [f32],
    batch_index: usize,
    max_height: usize,
    max_width: usize,
    image: &RgbImage,
    config: &ImageProcessorConfig,
) {
    for (x, y, pixel) in image.enumerate_pixels() {
        for channel in 0..CHANNELS {
            let mut value = f32::from(pixel[channel]);
            if config.do_rescale {
                value *= config.rescale_factor;
            }
            if config.do_normalize {
                value = (value - config.image_mean[channel]) / config.image_std[channel];
            }

            let offset = (((batch_index * CHANNELS + channel) * max_height + y as usize)
                * max_width)
                + x as usize;
            data[offset] = value;
        }
    }
}

fn validate_config(config: &ImageProcessorConfig) -> Result<()> {
    if config.patch_size == 0 {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "patch_size must be greater than zero".to_owned(),
        });
    }

    if config.spatial_merge_size == 0 {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "spatial_merge_size must be greater than zero".to_owned(),
        });
    }

    if config
        .patch_size
        .checked_mul(config.spatial_merge_size)
        .is_none()
    {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "patch_size * spatial_merge_size is too large".to_owned(),
        });
    }

    if config.size.longest_edge == 0 {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "longest_edge must be greater than zero".to_owned(),
        });
    }

    if config.data_format != DataFormat::ChannelsFirst {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "only channels_first processor output is supported".to_owned(),
        });
    }

    if config.image_mean.len() != CHANNELS {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "image_mean must contain three channel values".to_owned(),
        });
    }

    if config.image_std.len() != CHANNELS {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "image_std must contain three channel values".to_owned(),
        });
    }

    if config.image_std.contains(&0.0) {
        return Err(Error::UnsupportedProcessorConfiguration {
            reason: "image_std values must be non-zero".to_owned(),
        });
    }

    Ok(())
}

fn resize_output_size(
    height: usize,
    width: usize,
    config: &ImageProcessorConfig,
) -> (usize, usize) {
    if !config.do_resize {
        return (height, width);
    }

    let longest_edge = config.size.longest_edge as f64;
    let mut resized_height = height;
    let mut resized_width = width;
    let ratio = (height as f64 / longest_edge).max(width as f64 / longest_edge);
    if ratio > 1.0 {
        resized_height = ((height as f64) / ratio).floor().max(1.0) as usize;
        resized_width = ((width as f64) / ratio).floor().max(1.0) as usize;
    }

    let patch_stride = patch_grid_stride(config);
    (
        ceil_to_multiple(resized_height, patch_stride),
        ceil_to_multiple(resized_width, patch_stride),
    )
}

fn patch_grid_stride(config: &ImageProcessorConfig) -> usize {
    config
        .patch_size
        .checked_mul(config.spatial_merge_size)
        .expect("validated non-overflowing patch grid stride")
}

fn ceil_to_multiple(value: usize, factor: usize) -> usize {
    value.div_ceil(factor) * factor
}

fn filter_type(resample: ResampleFilter) -> FilterType {
    match resample {
        ResampleFilter::Bicubic => FilterType::CatmullRom,
    }
}
