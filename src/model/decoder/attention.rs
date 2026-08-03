//! Decoder attention-mask tensor.

/// Attention mask consumed by the decoder.
///
/// Expected final contract: `int64` with shape
/// `(batch_size, total_sequence_length)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AttentionMask {
    mask: Vec<i64>,
}

impl AttentionMask {
    /// Creates a typed attention mask from model-ready `int64` values.
    pub fn new(mask: Vec<i64>) -> Self {
        Self { mask }
    }

    /// Creates an attention mask with every position marked as visible.
    pub fn ones(length: usize) -> Self {
        Self {
            mask: vec![1; length],
        }
    }

    /// Reserves room for additional mask values.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.mask.reserve(additional);
    }

    /// Marks every existing position as visible.
    pub(crate) fn fill_visible(&mut self) {
        self.mask.fill(1);
    }

    /// Appends one visible position to the mask.
    pub(crate) fn push_visible(&mut self) {
        self.mask.push(1);
    }

    /// Returns the attention mask as a read-only slice.
    pub fn as_slice(&self) -> &[i64] {
        &self.mask
    }

    /// Returns the number of entries in the mask.
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Returns whether the mask contains no entries.
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }
}

impl From<Vec<i64>> for AttentionMask {
    fn from(mask: Vec<i64>) -> Self {
        Self::new(mask)
    }
}
