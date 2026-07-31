//! Preprocessed image tensor.

/// Preprocessed image tensor accepted by the vision encoder.
///
/// `ImageTensor` stores model-ready `float32` values in NCHW layout with shape
/// `(batch_size, 3, height, width)`. The image processor owns construction of
/// this type so raw image and tensor details do not leak into runtime wrappers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageTensor {
    data: Vec<f32>,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
}

impl ImageTensor {
    /// Creates an image tensor with the given NCHW shape.
    ///
    /// Returns an error if the tensor dimensions are invalid or the number of
    /// elements does not match the specified shape.
    pub fn new(
        data: Vec<f32>,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> crate::Result<Self> {
        let expected = batch_size
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| crate::Error::ImageProcessing {
                reason: "pixel value shape is too large".to_owned(),
            })?;
        if data.len() != expected {
            return Err(crate::Error::ImageProcessing {
                reason: format!(
                    "pixel value data length {} does not match shape {:?}",
                    data.len(),
                    (batch_size, channels, height, width)
                ),
            });
        }

        Ok(Self {
            data,
            batch_size,
            channels,
            height,
            width,
        })
    }

    /// Returns the pixel values as a contiguous NCHW slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns the tensor shape as `(batch_size, channels, height, width)`.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.batch_size, self.channels, self.height, self.width)
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the number of image channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the preprocessed image height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns the preprocessed image width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns one pixel value from the NCHW tensor.
    pub fn get(&self, batch: usize, channel: usize, y: usize, x: usize) -> Option<f32> {
        if batch >= self.batch_size
            || channel >= self.channels
            || y >= self.height
            || x >= self.width
        {
            return None;
        }

        let offset = (((batch * self.channels + channel) * self.height + y) * self.width) + x;
        self.data.get(offset).copied()
    }
}
