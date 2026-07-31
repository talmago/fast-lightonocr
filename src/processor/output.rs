//! Outputs produced by multimodal input processing.

use crate::model::{AttentionMask, ImageTensor};
use crate::processor::VisionGrid;

/// Inputs prepared for model inference.
///
/// This is the complete output of the preprocessing pipeline and serves as the
/// input to the model inference pipeline.
#[must_use]
#[derive(Debug, Clone)]
pub struct ProcessorOutput {
    /// Token IDs with image placeholders expanded into vision tokens.
    pub input_ids: Vec<i64>,

    /// Attention mask corresponding to `input_ids`.
    pub attention_mask: AttentionMask,

    /// Preprocessed image tensor consumed by the vision encoder.
    pub pixel_values: ImageTensor,

    /// Spatial layout of the encoded vision features.
    pub vision_grid: VisionGrid,
}
