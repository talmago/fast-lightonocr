//! Token and multimodal embedding tensors.

/// Token or multimodal embeddings consumed by the decoder.
///
/// `InputEmbeddings` stores model-ready `float32` values with shape
/// `(batch_size, sequence_length, hidden_size)`. The embedding model owns
/// construction of this type so ONNX Runtime tensors do not leak into the
/// generation pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InputEmbeddings {
    data: Vec<f32>,
    batch_size: usize,
    sequence_length: usize,
    hidden_size: usize,
}

impl InputEmbeddings {
    /// Creates typed input embeddings from raw contiguous embedding data.
    pub fn new(
        data: Vec<f32>,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> crate::Result<Self> {
        let expected = batch_size
            .checked_mul(sequence_length)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| crate::Error::InvalidEmbeddingOutput {
                reason: "input embedding shape is too large".to_owned(),
            })?;
        if data.len() != expected {
            return Err(crate::Error::InvalidEmbeddingOutput {
                reason: format!(
                    "input embedding data length {} does not match shape {:?}",
                    data.len(),
                    (batch_size, sequence_length, hidden_size)
                ),
            });
        }

        Ok(Self {
            data,
            batch_size,
            sequence_length,
            hidden_size,
        })
    }

    /// Returns the embedding values as a contiguous read-only slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns the tensor shape as `(batch_size, sequence_length, hidden_size)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.batch_size, self.sequence_length, self.hidden_size)
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the sequence length.
    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    /// Returns the text hidden size.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns a single embedding value from the tensor.
    pub fn get(&self, batch: usize, sequence: usize, hidden: usize) -> Option<f32> {
        if batch >= self.batch_size
            || sequence >= self.sequence_length
            || hidden >= self.hidden_size
        {
            return None;
        }

        let offset = (batch * self.sequence_length + sequence) * self.hidden_size + hidden;
        self.data.get(offset).copied()
    }
}
