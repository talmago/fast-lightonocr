//! ONNX Runtime wrapper for the Qwen packed-patch vision graph.
//!
//! The exported graph takes `pixel_values` float32 `[num_patches, patch_width]`
//! and `image_grid_thw` int64 `[1, 3]`, and returns `image_features` float32
//! `[num_image_features, hidden_size]` already projected into decoder space.

use std::path::Path;

use ort::session::Session;
use ort::value::{TensorElementType, TensorRef, ValueType};

use crate::processor::QwenProcessorOutput;
use crate::util::RuntimeOptions;
use crate::{Error, Result};

use super::ImageFeatures;

/// ONNX Runtime wrapper for the Qwen vision encoder.
#[derive(Debug)]
pub struct QwenVisionEncoder {
    session: Session,
    hidden_size: usize,
}

impl QwenVisionEncoder {
    /// Loads a Qwen vision encoder and checks the two-input packed-patch contract.
    pub fn from_model_path(
        model_path: impl AsRef<Path>,
        hidden_size: usize,
        runtime: &RuntimeOptions,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.is_file() {
            return Err(Error::MissingVisionModel { path: model_path });
        }
        let session = crate::util::onnxruntime::session_builder(runtime)?
            .commit_from_file(&model_path)
            .map_err(|source| Error::VisionModelLoad {
                path: model_path.clone(),
                source,
            })?;
        let inputs = session.inputs();
        if inputs.len() != 2
            || inputs[0].name() != "pixel_values"
            || inputs[1].name() != "image_grid_thw"
        {
            return Err(Error::InvalidVisionModel {
                reason: format!(
                    "Qwen vision inputs do not match exported contract: {:?}",
                    inputs.iter().map(|input| input.name()).collect::<Vec<_>>()
                ),
            });
        }
        let outputs = session.outputs();
        if outputs.len() != 1 || outputs[0].name() != "image_features" {
            return Err(Error::InvalidVisionModel {
                reason: "Qwen vision graph must expose only `image_features`".to_owned(),
            });
        }
        validate_tensor(
            inputs[0].dtype(),
            TensorElementType::Float32,
            2,
            "pixel_values",
        )?;
        validate_tensor(
            inputs[1].dtype(),
            TensorElementType::Int64,
            2,
            "image_grid_thw",
        )?;
        validate_tensor(
            outputs[0].dtype(),
            TensorElementType::Float32,
            2,
            "image_features",
        )?;
        Ok(Self {
            session,
            hidden_size,
        })
    }

    /// Encodes packed patches from [`QwenProcessorOutput`] into decoder features.
    pub fn encode(&mut self, input: &QwenProcessorOutput) -> Result<ImageFeatures> {
        let pixel_shape = [input.num_patches as i64, input.patch_width as i64];
        let pixels = TensorRef::from_array_view((pixel_shape, input.pixel_values.as_slice()))
            .map_err(|source| Error::VisionTensorCreation { source })?;
        let grid = TensorRef::from_array_view(([1_i64, 3_i64], input.grid_thw.as_slice()))
            .map_err(|source| Error::VisionTensorCreation { source })?;
        let mut outputs = self
            .session
            .run(ort::inputs! {
                "pixel_values" => pixels,
                "image_grid_thw" => grid,
            })
            .map_err(|source| Error::VisionInference { source })?;
        let value = outputs
            .remove("image_features")
            .ok_or_else(|| Error::InvalidVisionOutput {
                reason: "Qwen graph did not return `image_features`".to_owned(),
            })?;
        let (shape, data) =
            value
                .try_extract_tensor::<f32>()
                .map_err(|source| Error::InvalidVisionOutput {
                    reason: format!("could not read Qwen image features as float32: {source}"),
                })?;
        let shape = shape.as_ref();
        if shape.len() != 2
            || shape[0] < 0
            || shape[1] != self.hidden_size as i64
            || shape[0] as usize != input.num_image_features
        {
            return Err(Error::InvalidVisionOutput {
                reason: format!(
                    "Qwen vision output shape {shape:?} does not match {} features of width {}",
                    input.num_image_features, self.hidden_size
                ),
            });
        }
        ImageFeatures::new(data.to_vec(), 1, input.num_image_features, self.hidden_size)
    }
}

fn validate_tensor(
    value_type: &ValueType,
    expected_type: TensorElementType,
    expected_rank: usize,
    name: &str,
) -> Result<()> {
    let ValueType::Tensor { ty, shape, .. } = value_type else {
        return Err(Error::InvalidVisionModel {
            reason: format!("`{name}` is not a tensor"),
        });
    };
    if *ty != expected_type || shape.len() != expected_rank {
        return Err(Error::InvalidVisionModel {
            reason: format!(
                "`{name}` has type {ty} and rank {}, expected {expected_type} rank {expected_rank}",
                shape.len()
            ),
        });
    }
    Ok(())
}
