//! ONNX Runtime wrapper for the LightOnOCR vision encoder.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::{TensorElementType, TensorRef, ValueType};

use crate::{Error, Result};

use super::{ImageFeatures, ImageTensor, VisionConfig};

const VISION_INPUT_NAME: &str = "pixel_values";
const VISION_OUTPUT_NAME: &str = "image_features";

/// ONNX Runtime wrapper for the exported LightOnOCR vision encoder.
///
/// `VisionEncoder` owns exactly one ONNX Runtime session. It accepts typed
/// [`ImageTensor`] in NCHW layout and returns typed [`ImageFeatures`], keeping
/// ONNX Runtime values private to this module.
#[derive(Debug)]
pub struct VisionEncoder {
    session: Session,
    config: VisionConfig,
    model_path: PathBuf,
}

impl VisionEncoder {
    /// Loads a vision encoder from an explicit ONNX model path and config.
    ///
    /// This is useful for tests and tools that already resolved the model
    /// asset path and parsed configuration.
    pub fn from_model_path(model_path: impl AsRef<Path>, config: VisionConfig) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.is_file() {
            return Err(Error::MissingVisionModel { path: model_path });
        }

        crate::util::onnxruntime::ensure_compatible()?;

        let session = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(&model_path))
            .map_err(|source| Error::VisionModelLoad {
                path: model_path.clone(),
                source,
            })?;
        validate_session_contract(&session, &config)?;

        Ok(Self {
            session,
            config,
            model_path,
        })
    }

    /// Returns the loaded vision encoder configuration.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    /// Returns the ONNX model path backing this encoder.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Executes the vision encoder for a preprocessed image tensor.
    ///
    /// The input must follow the documented vision contract:
    /// `float32` NCHW values with shape `(batch_size, 3, height, width)`.
    /// The returned image features have shape
    /// `(batch_size, num_merged_patches, hidden_size)`.
    pub fn encode(&mut self, image_tensor: &ImageTensor) -> Result<ImageFeatures> {
        validate_image_tensor(image_tensor, &self.config)?;

        let (batch_size, channels, height, width) = image_tensor.shape();
        let input_shape = [
            batch_size as i64,
            channels as i64,
            height as i64,
            width as i64,
        ];
        let input = TensorRef::from_array_view((input_shape, image_tensor.as_slice()))
            .map_err(|source| Error::VisionTensorCreation { source })?;

        let mut outputs = self
            .session
            .run(ort::inputs! { VISION_INPUT_NAME => input })
            .map_err(|source| Error::VisionInference { source })?;
        let output =
            outputs
                .remove(VISION_OUTPUT_NAME)
                .ok_or_else(|| Error::InvalidVisionOutput {
                    reason: format!("missing `{VISION_OUTPUT_NAME}` output"),
                })?;
        let (shape, data) =
            output
                .try_extract_tensor::<f32>()
                .map_err(|source| Error::InvalidVisionOutput {
                    reason: format!(
                        "failed to extract `{VISION_OUTPUT_NAME}` as float32: {source}"
                    ),
                })?;
        let shape = shape.as_ref();
        validate_image_feature_shape(shape, &self.config)?;

        ImageFeatures::new(
            data.to_vec(),
            usize::try_from(shape[0]).expect("validated non-negative batch size"),
            usize::try_from(shape[1]).expect("validated non-negative patch count"),
            usize::try_from(shape[2]).expect("validated non-negative hidden size"),
        )
    }
}

fn validate_session_contract(session: &Session, config: &VisionConfig) -> Result<()> {
    let inputs = session.inputs();
    if inputs.len() != 1 {
        return Err(Error::InvalidVisionModel {
            reason: format!("expected 1 input, found {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    if input.name() != VISION_INPUT_NAME {
        return Err(Error::InvalidVisionModel {
            reason: format!(
                "expected input `{VISION_INPUT_NAME}`, found `{}`",
                input.name()
            ),
        });
    }
    validate_tensor_metadata(
        input.dtype(),
        VISION_INPUT_NAME,
        TensorElementType::Float32,
        4,
        &[(1, config.num_channels)],
    )?;

    let outputs = session.outputs();
    if outputs.len() != 1 {
        return Err(Error::InvalidVisionModel {
            reason: format!("expected 1 output, found {}", outputs.len()),
        });
    }

    let output = &outputs[0];
    if output.name() != VISION_OUTPUT_NAME {
        return Err(Error::InvalidVisionModel {
            reason: format!(
                "expected output `{VISION_OUTPUT_NAME}`, found `{}`",
                output.name()
            ),
        });
    }
    validate_tensor_metadata(
        output.dtype(),
        VISION_OUTPUT_NAME,
        TensorElementType::Float32,
        3,
        &[(2, config.hidden_size)],
    )
}

fn validate_tensor_metadata(
    value_type: &ValueType,
    name: &str,
    expected_type: TensorElementType,
    expected_rank: usize,
    static_dims: &[(usize, usize)],
) -> Result<()> {
    let ValueType::Tensor { ty, shape, .. } = value_type else {
        return Err(Error::InvalidVisionModel {
            reason: format!("`{name}` is not a tensor"),
        });
    };
    if *ty != expected_type {
        return Err(Error::InvalidVisionModel {
            reason: format!("`{name}` has type {ty}, expected {expected_type}"),
        });
    }
    if shape.len() != expected_rank {
        return Err(Error::InvalidVisionModel {
            reason: format!(
                "`{name}` has rank {}, expected {expected_rank}",
                shape.len()
            ),
        });
    }
    for &(axis, expected) in static_dims {
        let actual = shape[axis];
        if actual >= 0 && actual as usize != expected {
            return Err(Error::InvalidVisionModel {
                reason: format!("`{name}` dimension {axis} is {actual}, expected {expected}"),
            });
        }
    }

    Ok(())
}

fn validate_image_tensor(pixel_values: &ImageTensor, config: &VisionConfig) -> Result<()> {
    let (batch_size, channels, height, width) = pixel_values.shape();
    if batch_size == 0 {
        return Err(Error::InvalidVisionInput {
            reason: "batch size must be greater than zero".to_owned(),
        });
    }
    if channels != config.num_channels {
        return Err(Error::InvalidVisionInput {
            reason: format!(
                "expected {} channels, found {channels}",
                config.num_channels
            ),
        });
    }
    if height == 0 || width == 0 {
        return Err(Error::InvalidVisionInput {
            reason: format!("height and width must be greater than zero, found {height}x{width}"),
        });
    }

    Ok(())
}

fn validate_image_feature_shape(shape: &[i64], config: &VisionConfig) -> Result<()> {
    if shape.len() != 3 {
        return Err(Error::InvalidVisionOutput {
            reason: format!(
                "`{VISION_OUTPUT_NAME}` has rank {}, expected 3",
                shape.len()
            ),
        });
    }
    if shape.iter().any(|&dim| dim < 0) {
        return Err(Error::InvalidVisionOutput {
            reason: format!("`{VISION_OUTPUT_NAME}` has negative shape {shape:?}"),
        });
    }
    if shape[2] as usize != config.hidden_size {
        return Err(Error::InvalidVisionOutput {
            reason: format!(
                "`{VISION_OUTPUT_NAME}` hidden size is {}, expected {}",
                shape[2], config.hidden_size
            ),
        });
    }

    Ok(())
}
