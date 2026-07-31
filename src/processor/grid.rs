//! Vision feature grid computation.
//!
//! The image processor resizes every image to a patch-aligned resolution before
//! the vision encoder divides it into square patches. Neighboring patches are
//! then merged according to the configured spatial merge factor, producing the
//! final two-dimensional grid of image features consumed by the language model.

use crate::{Error, Result};

/// Spatial layout of the vision encoder output.
///
/// Each grid cell corresponds to a single projected image embedding that
/// replaces one `<|image_pad|>` token in the language model prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionGrid {
    /// Number of feature columns.
    pub width: usize,

    /// Number of feature rows.
    pub height: usize,
}

impl VisionGrid {
    /// Computes the vision feature grid from processed image dimensions.
    ///
    /// The supplied image dimensions must already be resized and aligned to the
    /// vision encoder patch size.
    pub fn from_image_size(
        width: usize,
        height: usize,
        patch_size: usize,
        spatial_merge_size: usize,
    ) -> Result<Self> {
        if width % patch_size != 0 || height % patch_size != 0 {
            return Err(Error::ImageProcessing {
                reason: format!(
                    "image dimensions ({width}x{height}) are not divisible by patch size ({patch_size})",
                ),
            });
        }

        let patch_width = width / patch_size;
        let patch_height = height / patch_size;

        if patch_width % spatial_merge_size != 0 || patch_height % spatial_merge_size != 0 {
            return Err(Error::ImageProcessing {
                reason: format!(
                    "patch grid ({patch_width}x{patch_height}) is not divisible by spatial merge size ({spatial_merge_size})",
                ),
            });
        }

        Ok(Self {
            width: patch_width / spatial_merge_size,
            height: patch_height / spatial_merge_size,
        })
    }

    /// Returns the total number of vision features.
    #[must_use]
    pub fn feature_count(self) -> usize {
        self.width * self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATCH_SIZE: usize = 14;
    const SPATIAL_MERGE_SIZE: usize = 2;

    #[test]
    fn feature_count() {
        let grid = VisionGrid {
            width: 3,
            height: 5,
        };

        assert_eq!(grid.feature_count(), 15);
    }

    #[test]
    fn computes_square_grid() {
        let grid = VisionGrid::from_image_size(476, 476, PATCH_SIZE, SPATIAL_MERGE_SIZE).unwrap();

        assert_eq!(grid.width, 17);
        assert_eq!(grid.height, 17);
        assert_eq!(grid.feature_count(), 289);
    }

    #[test]
    fn computes_non_square_grid() {
        let grid = VisionGrid::from_image_size(476, 448, PATCH_SIZE, SPATIAL_MERGE_SIZE).unwrap();

        assert_eq!(grid.width, 17);
        assert_eq!(grid.height, 16);
        assert_eq!(grid.feature_count(), 272);
    }

    #[test]
    fn computes_single_feature_grid() {
        let grid = VisionGrid::from_image_size(28, 28, PATCH_SIZE, SPATIAL_MERGE_SIZE).unwrap();

        assert_eq!(grid.width, 1);
        assert_eq!(grid.height, 1);
        assert_eq!(grid.feature_count(), 1);
    }

    #[test]
    fn rejects_width_not_divisible_by_patch_size() {
        let error =
            VisionGrid::from_image_size(475, 476, PATCH_SIZE, SPATIAL_MERGE_SIZE).unwrap_err();

        assert!(matches!(error, Error::ImageProcessing { .. }));
    }

    #[test]
    fn rejects_height_not_divisible_by_patch_size() {
        let error =
            VisionGrid::from_image_size(476, 475, PATCH_SIZE, SPATIAL_MERGE_SIZE).unwrap_err();

        assert!(matches!(error, Error::ImageProcessing { .. }));
    }

    #[test]
    fn rejects_patch_grid_not_divisible_by_merge_size() {
        // 462 / 14 = 33 patches.
        // A 33×33 patch grid cannot be evenly merged into 2×2 groups.
        let error =
            VisionGrid::from_image_size(462, 462, PATCH_SIZE, SPATIAL_MERGE_SIZE).unwrap_err();

        assert!(matches!(error, Error::ImageProcessing { .. }));
    }
}
