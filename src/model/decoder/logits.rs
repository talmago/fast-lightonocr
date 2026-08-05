//! Decoder logits tensor.

/// Decoder logits produced during generation.
///
/// `Logits` stores model-ready `float32` values with shape
/// `(batch_size, sequence_length, vocabulary_size)`. The decoder owns
/// construction of this type so ONNX Runtime tensors do not leak into the
/// generation engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Logits {
    data: Vec<f32>,
    batch_size: usize,
    sequence_length: usize,
    vocab_size: usize,
}

impl Logits {
    /// Creates typed decoder logits from raw contiguous logits data.
    pub fn new(
        data: Vec<f32>,
        batch_size: usize,
        sequence_length: usize,
        vocab_size: usize,
    ) -> crate::Result<Self> {
        let expected = batch_size
            .checked_mul(sequence_length)
            .and_then(|value| value.checked_mul(vocab_size))
            .ok_or_else(|| crate::Error::InvalidDecoderOutput {
                reason: "logits shape is too large".to_owned(),
            })?;
        if data.len() != expected {
            return Err(crate::Error::InvalidDecoderOutput {
                reason: format!(
                    "logits data length {} does not match shape {:?}",
                    data.len(),
                    (batch_size, sequence_length, vocab_size)
                ),
            });
        }

        Ok(Self {
            data,
            batch_size,
            sequence_length,
            vocab_size,
        })
    }

    /// Returns the logits as a contiguous read-only slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns the tensor shape as `(batch_size, sequence_length, vocabulary_size)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.batch_size, self.sequence_length, self.vocab_size)
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the sequence length.
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns a single logit value from the tensor.
    pub fn get(&self, batch: usize, sequence: usize, token: usize) -> Option<f32> {
        if batch >= self.batch_size || sequence >= self.sequence_length || token >= self.vocab_size
        {
            return None;
        }

        let offset = (batch * self.sequence_length + sequence) * self.vocab_size + token;
        self.data.get(offset).copied()
    }

    /// Consumes the logits and returns the underlying buffer for reuse.
    pub(crate) fn into_data(self) -> Vec<f32> {
        self.data
    }
}
