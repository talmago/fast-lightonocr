//! Image feature tensor.

/// Image features produced by the vision encoder.
///
/// `ImageFeatures` stores model-ready `float32` values with shape
/// `(batch_size, num_merged_patches, hidden_size)`. The vision encoder owns
/// construction of this type so ONNX Runtime tensors do not leak into the rest
/// of the OCR pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageFeatures {
    data: Vec<f32>,
    batch_size: usize,
    num_merged_patches: usize,
    hidden_size: usize,
}

impl ImageFeatures {
    /// Creates typed image features from raw contiguous feature data.
    pub fn new(
        data: Vec<f32>,
        batch_size: usize,
        num_merged_patches: usize,
        hidden_size: usize,
    ) -> crate::Result<Self> {
        let expected = batch_size
            .checked_mul(num_merged_patches)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| crate::Error::InvalidVisionOutput {
                reason: "image feature shape is too large".to_owned(),
            })?;
        if data.len() != expected {
            return Err(crate::Error::InvalidVisionOutput {
                reason: format!(
                    "image feature data length {} does not match shape {:?}",
                    data.len(),
                    (batch_size, num_merged_patches, hidden_size)
                ),
            });
        }

        Ok(Self {
            data,
            batch_size,
            num_merged_patches,
            hidden_size,
        })
    }

    /// Returns the feature values as a contiguous read-only slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns the feature tensor shape as
    /// `(batch_size, num_merged_patches, hidden_size)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.batch_size, self.num_merged_patches, self.hidden_size)
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the dynamic number of merged vision patches.
    pub fn num_merged_patches(&self) -> usize {
        self.num_merged_patches
    }

    /// Returns the vision hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns a single feature value from the tensor.
    pub fn get(&self, batch: usize, patch: usize, hidden: usize) -> Option<f32> {
        if batch >= self.batch_size
            || patch >= self.num_merged_patches
            || hidden >= self.hidden_size
        {
            return None;
        }

        let offset = (batch * self.num_merged_patches + patch) * self.hidden_size + hidden;
        self.data.get(offset).copied()
    }

    /// Overwrites this tensor in place from contiguous feature data.
    ///
    /// Reuses the existing allocation when the new shape fits in the current
    /// capacity; otherwise the buffer grows as needed.
    pub fn copy_from_slice(
        &mut self,
        data: &[f32],
        batch_size: usize,
        num_merged_patches: usize,
        hidden_size: usize,
    ) -> crate::Result<()> {
        let expected = batch_size
            .checked_mul(num_merged_patches)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| crate::Error::InvalidVisionOutput {
                reason: "image feature shape is too large".to_owned(),
            })?;
        if data.len() != expected {
            return Err(crate::Error::InvalidVisionOutput {
                reason: format!(
                    "image feature data length {} does not match shape {:?}",
                    data.len(),
                    (batch_size, num_merged_patches, hidden_size)
                ),
            });
        }

        self.data.clear();
        self.data.extend_from_slice(data);
        self.batch_size = batch_size;
        self.num_merged_patches = num_merged_patches;
        self.hidden_size = hidden_size;
        Ok(())
    }
}
