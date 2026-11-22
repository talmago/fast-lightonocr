//! Qwen packed-patch image processor, OCR chat prompt, and multimodal positions.
//!
//! Resize limits (2,048 px longest edge, 5 MP, 65,536 minimum pixels) are the
//! official LightOnOCR-3-0.8B OCR settings, not the very large
//! `size.longest_edge` values in some `processor_config.json` files.

use std::path::Path;

use image::imageops::FilterType;
use image::{DynamicImage, RgbImage};

use crate::model::AttentionMask;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

/// Vision patch size in pixels for Qwen3.5 OCR checkpoints.
pub const PATCH_SIZE: usize = 16;
/// Temporal copies of each spatial patch (Qwen2-VL packing convention).
pub const TEMPORAL_PATCH_SIZE: usize = 2;
/// Spatial merge factor applied after the patch grid.
pub const MERGE_SIZE: usize = 2;
/// Longest edge used for LightOnOCR-3-0.8B / 4B OCR.
pub const MAX_EDGE: usize = 2048;
/// Maximum resized pixel area for those OCR checkpoints.
pub const MAX_PIXELS: usize = 5_000_000;
/// Minimum resized pixel area.
pub const MIN_PIXELS: usize = 65_536;

const RESIZE_FACTOR: usize = PATCH_SIZE * MERGE_SIZE;
const IMAGE_PAD_TOKEN: &str = "<|image_pad|>";
const VISION_START_TOKEN: &str = "<|vision_start|>";
const VISION_END_TOKEN: &str = "<|vision_end|>";

/// OCR chat template with thinking disabled (empty `<think>` block).
///
/// `{user}` is optional user-turn text such as `grounding`. The image is a
/// single `<|vision_start|><|image_pad|><|vision_end|>` group that the
/// processor later expands to one token per merged patch.
const OCR_PROMPT_PREFIX: &str = "<|im_start|>user\n";
const OCR_PROMPT_VISION: &str = "<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n";
const OCR_PROMPT_ASSISTANT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Packed-patch processor output consumed by [`crate::QwenVisionEncoder`].
#[derive(Debug, Clone)]
pub struct QwenProcessorOutput {
    /// Prompt token IDs with `<|image_pad|>` expanded to `num_image_features`.
    pub input_ids: Vec<i64>,
    /// Attention mask matching `input_ids`.
    pub attention_mask: AttentionMask,
    /// Flattened packed patches, shape `[num_patches, patch_width]`.
    pub pixel_values: Vec<f32>,
    /// Number of spatial patches (`grid_h * grid_w`).
    pub num_patches: usize,
    /// Number of merged vision features after `MERGE_SIZE`.
    pub num_image_features: usize,
    /// Packed patch width: `3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE`.
    pub patch_width: usize,
    /// Temporal/height/width patch grid, ONNX `image_grid_thw` of shape `[1, 3]`.
    pub grid_thw: [i64; 3],
    /// Multimodal position IDs in contiguous `(axis, sequence)` order, length
    /// `3 * input_ids.len()`, later reshaped to `[3, 1, sequence]`.
    pub position_ids: Vec<i64>,
    /// `max(position_ids) + 1 - sequence_length`, added to the decode step
    /// index so generated tokens continue the multimodal RoPE timeline.
    pub rope_delta: i64,
    /// Tokenizer id for `<|image_pad|>`, used when merging vision features.
    pub image_token_id: i64,
}

/// Qwen processor for the single-image OCR request path.
///
/// Owns only a tokenizer. Image packing is performed in this module rather
/// than by [`super::ImageProcessor`].
#[derive(Debug, Clone)]
pub struct QwenProcessor {
    tokenizer: Tokenizer,
}

impl QwenProcessor {
    /// Creates a processor from an already loaded tokenizer.
    #[must_use]
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    /// Loads tokenizer files from a Hugging Face model directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(super::load_tokenizer(dir.as_ref())?))
    }

    /// Returns the tokenizer used to render the OCR prompt.
    #[must_use]
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Packs `image` and optionally places `user_text` on the user turn.
    ///
    /// [`crate::LightOnOCR::process`] still names this argument `system_prompt`
    /// for API compatibility. For Qwen it is user-turn text (for example
    /// `grounding`).
    pub fn process(
        &self,
        image: &DynamicImage,
        user_text: Option<&str>,
    ) -> Result<QwenProcessorOutput> {
        let (rgb, height, width) = resize_for_qwen(image)?;
        let grid_height = height / PATCH_SIZE;
        let grid_width = width / PATCH_SIZE;
        let num_patches = grid_height * grid_width;
        let num_image_features = (grid_height / MERGE_SIZE) * (grid_width / MERGE_SIZE);
        let pixel_values = pack_image_patches(&rgb, grid_height, grid_width);
        let patch_width = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;

        let image_token_id = self.tokenizer.token_to_id(IMAGE_PAD_TOKEN)?;
        let vision_start = self.tokenizer.token_to_id(VISION_START_TOKEN)?;
        let vision_end = self.tokenizer.token_to_id(VISION_END_TOKEN)?;
        let prompt = render_ocr_prompt(user_text);
        let tokenized = self.tokenizer.encode(&prompt)?;
        let input_ids = expand_image_pad(&tokenized, image_token_id, num_image_features)?;
        let position_ids = multimodal_position_ids(
            &input_ids,
            image_token_id,
            grid_height,
            grid_width,
            num_image_features,
        )?;
        let max_position = position_ids.iter().copied().max().unwrap_or(0);
        let rope_delta = max_position + 1 - input_ids.len() as i64;
        let attention_mask = AttentionMask::ones(input_ids.len());

        if !input_ids.contains(&vision_start) || !input_ids.contains(&vision_end) {
            return Err(Error::ImageProcessing {
                reason: "Qwen tokenizer is missing vision boundary markers".to_owned(),
            });
        }

        Ok(QwenProcessorOutput {
            input_ids,
            attention_mask,
            pixel_values,
            num_patches,
            num_image_features,
            patch_width,
            grid_thw: [1, grid_height as i64, grid_width as i64],
            position_ids,
            rope_delta,
            image_token_id,
        })
    }
}

fn render_ocr_prompt(user_text: Option<&str>) -> String {
    let mut prompt = String::from(OCR_PROMPT_PREFIX);
    if let Some(text) = user_text {
        prompt.push_str(text.trim());
    }
    prompt.push_str(OCR_PROMPT_VISION);
    prompt.push_str(OCR_PROMPT_ASSISTANT);
    prompt
}

fn expand_image_pad(
    tokenized: &[i64],
    image_token_id: i64,
    num_image_features: usize,
) -> Result<Vec<i64>> {
    let mut input_ids = Vec::with_capacity(tokenized.len() + num_image_features.saturating_sub(1));
    let mut placeholder_count = 0;
    for &token_id in tokenized {
        if token_id == image_token_id {
            placeholder_count += 1;
            input_ids.extend(std::iter::repeat_n(image_token_id, num_image_features));
        } else {
            input_ids.push(token_id);
        }
    }
    if placeholder_count != 1 {
        return Err(Error::ImageProcessing {
            reason: format!("Qwen template produced {placeholder_count} image placeholders"),
        });
    }
    Ok(input_ids)
}

fn multimodal_position_ids(
    input_ids: &[i64],
    image_token_id: i64,
    grid_height: usize,
    grid_width: usize,
    num_image_features: usize,
) -> Result<Vec<i64>> {
    let mut position_ids = vec![0_i64; input_ids.len() * 3];
    let mut current_position = 0_i64;
    let mut index = 0;
    while index < input_ids.len() {
        if input_ids[index] == image_token_id {
            let mut end = index + 1;
            while end < input_ids.len() && input_ids[end] == image_token_id {
                end += 1;
            }
            if end - index != num_image_features {
                return Err(Error::ImageProcessing {
                    reason: "Qwen image token run does not match the patch grid".to_owned(),
                });
            }
            let merged_height = grid_height / MERGE_SIZE;
            let merged_width = grid_width / MERGE_SIZE;
            for row in 0..merged_height {
                for column in 0..merged_width {
                    let token = index + row * merged_width + column;
                    position_ids[token] = current_position;
                    position_ids[input_ids.len() + token] = current_position + row as i64;
                    position_ids[2 * input_ids.len() + token] = current_position + column as i64;
                }
            }
            current_position += merged_height.max(merged_width) as i64;
            index = end;
        } else {
            for axis in 0..3 {
                position_ids[axis * input_ids.len() + index] = current_position;
            }
            current_position += 1;
            index += 1;
        }
    }
    Ok(position_ids)
}

fn resize_for_qwen(image: &DynamicImage) -> Result<(RgbImage, usize, usize)> {
    let original = image.to_rgb8();
    let (width, height) = (original.width() as usize, original.height() as usize);
    if width == 0 || height == 0 {
        return Err(Error::ImageProcessing {
            reason: "image dimensions must be non-zero".to_owned(),
        });
    }
    let aspect = width.max(height) as f64 / width.min(height) as f64;
    if aspect > 200.0 {
        return Err(Error::ImageProcessing {
            reason: format!("Qwen image aspect ratio {aspect:.2} exceeds 200"),
        });
    }

    let mut target_h = round_to_factor(height, RESIZE_FACTOR);
    let mut target_w = round_to_factor(width, RESIZE_FACTOR);
    let scale = (target_h as f64 / MAX_EDGE as f64)
        .max(target_w as f64 / MAX_EDGE as f64)
        .max(((target_h * target_w) as f64 / MAX_PIXELS as f64).sqrt());
    if scale > 1.0 {
        let beta = ((height * width) as f64 / (MAX_PIXELS.min(MAX_EDGE * MAX_EDGE) as f64))
            .sqrt()
            .max(height as f64 / MAX_EDGE as f64)
            .max(width as f64 / MAX_EDGE as f64);
        target_h =
            ((height as f64 / beta / RESIZE_FACTOR as f64).floor() as usize).max(1) * RESIZE_FACTOR;
        target_w =
            ((width as f64 / beta / RESIZE_FACTOR as f64).floor() as usize).max(1) * RESIZE_FACTOR;
    } else if target_h * target_w < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (height * width) as f64).sqrt();
        target_h = ((height as f64 * beta / RESIZE_FACTOR as f64).ceil() as usize) * RESIZE_FACTOR;
        target_w = ((width as f64 * beta / RESIZE_FACTOR as f64).ceil() as usize) * RESIZE_FACTOR;
    }
    let rgb = image::imageops::resize(
        &original,
        target_w as u32,
        target_h as u32,
        FilterType::CatmullRom,
    );
    Ok((rgb, target_h, target_w))
}

fn round_to_factor(value: usize, factor: usize) -> usize {
    (((value as f64 / factor as f64).round() as usize).max(1)) * factor
}

fn pack_image_patches(rgb: &RgbImage, grid_height: usize, grid_width: usize) -> Vec<f32> {
    let merged_height = grid_height / MERGE_SIZE;
    let merged_width = grid_width / MERGE_SIZE;
    let feature_size = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
    let mut output = Vec::with_capacity(grid_height * grid_width * feature_size);
    for block_y in 0..merged_height {
        for block_x in 0..merged_width {
            for merge_y in 0..MERGE_SIZE {
                for merge_x in 0..MERGE_SIZE {
                    let patch_y = block_y * MERGE_SIZE + merge_y;
                    let patch_x = block_x * MERGE_SIZE + merge_x;
                    for channel in 0..3 {
                        for _time in 0..TEMPORAL_PATCH_SIZE {
                            for y in 0..PATCH_SIZE {
                                for x in 0..PATCH_SIZE {
                                    let pixel = rgb.get_pixel(
                                        (patch_x * PATCH_SIZE + x) as u32,
                                        (patch_y * PATCH_SIZE + y) as u32,
                                    );
                                    let value = f32::from(pixel[channel]) / 255.0;
                                    output.push((value - 0.5) / 0.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_a_single_image_placeholder() {
        let tokenized = [1, 99, 2];
        let expanded = expand_image_pad(&tokenized, 99, 4).unwrap();
        assert_eq!(expanded, vec![1, 99, 99, 99, 99, 2]);
    }

    #[test]
    fn assigns_3_axis_positions_for_text_and_image() {
        let image_token = 9;
        // one text token, 4 image tokens (2x2 merged from 4x4 patches), one text token
        let input_ids = vec![1, 9, 9, 9, 9, 2];
        let positions = multimodal_position_ids(&input_ids, image_token, 4, 4, 4).unwrap();
        assert_eq!(positions.len(), 18);
        // text token 0 is position 0 on all axes
        assert_eq!(positions[0], 0);
        assert_eq!(positions[6], 0);
        assert_eq!(positions[12], 0);
        // first image token continues at position 1; height/width start at 1+row/col
        assert_eq!(positions[1], 1);
        assert_eq!(positions[7], 1);
        assert_eq!(positions[13], 1);
        // last text token is after max(merged_h, merged_w) = 2
        assert_eq!(positions[5], 3);
        assert_eq!(positions[11], 3);
        assert_eq!(positions[17], 3);
    }

    #[test]
    fn render_ocr_prompt_inserts_optional_user_text() {
        let prompt = render_ocr_prompt(Some("grounding"));
        assert!(prompt.starts_with("<|im_start|>user\ngrounding"));
        assert!(prompt.contains("<|vision_start|><|image_pad|><|vision_end|>"));
        assert!(prompt.contains("<think>\n\n</think>"));
    }
}
