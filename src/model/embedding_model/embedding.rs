//! ONNX Runtime wrapper for the LightOnOCR token embedding model.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::{TensorElementType, TensorRef, ValueType};

use crate::model::InputEmbeddings;
use crate::profiling::{self, Stage};
use crate::{Error, Result};

use super::EmbeddingConfig;

const EMBEDDING_INPUT_NAME: &str = "input_ids";
const EMBEDDING_OUTPUT_NAME: &str = "inputs_embeds";

/// ONNX Runtime wrapper for the exported LightOnOCR embedding model.
///
/// `EmbeddingModel` owns exactly one ONNX Runtime session. It accepts typed
/// [`InputIds`] and returns typed [`InputEmbeddings`], keeping ONNX Runtime
/// values private to this module.
#[derive(Debug)]
pub struct EmbeddingModel {
    session: Session,
    config: EmbeddingConfig,
    model_path: PathBuf,
}

impl EmbeddingModel {
    /// Loads an embedding model from an explicit ONNX model path and config.
    ///
    /// This is useful for tests and tools that already resolved the model
    /// asset path and parsed configuration.
    pub fn from_model_path(model_path: impl AsRef<Path>, config: EmbeddingConfig) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        if !model_path.is_file() {
            return Err(Error::MissingEmbeddingModel { path: model_path });
        }

        crate::util::onnxruntime::ensure_compatible()?;

        let session = Session::builder()
            .and_then(|mut builder| builder.commit_from_file(&model_path))
            .map_err(|source| Error::EmbeddingModelLoad {
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

    /// Returns the loaded embedding configuration.
    pub fn config(&self) -> &EmbeddingConfig {
        &self.config
    }

    /// Returns the ONNX model path backing this embedding model.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Converts model-ready token IDs into decoder input embeddings.
    ///
    /// The token IDs are represented as one batch with shape
    /// `(1, sequence_length)`. The returned embeddings have shape
    /// `(1, sequence_length, hidden_size)`.
    pub fn embed(&mut self, token_ids: &[i64]) -> Result<InputEmbeddings> {
        validate_token_ids(token_ids, &self.config)?;

        let input_shape = [1_i64, token_ids.len() as i64];
        let input = TensorRef::from_array_view((input_shape, token_ids))
            .map_err(|source| Error::EmbeddingTensorCreation { source })?;

        let mut outputs = {
            let _timer = profiling::start(Stage::EmbeddingOnnx);
            self.session
                .run(ort::inputs! { EMBEDDING_INPUT_NAME => input })
                .map_err(|source| Error::EmbeddingInference { source })?
        };

        let output =
            outputs
                .remove(EMBEDDING_OUTPUT_NAME)
                .ok_or_else(|| Error::InvalidEmbeddingOutput {
                    reason: format!("missing `{EMBEDDING_OUTPUT_NAME}` output"),
                })?;

        let (shape, data) =
            output
                .try_extract_tensor::<f32>()
                .map_err(|source| Error::InvalidEmbeddingOutput {
                    reason: format!(
                        "failed to extract `{EMBEDDING_OUTPUT_NAME}` as float32: {source}"
                    ),
                })?;

        let shape = shape.as_ref();
        validate_embedding_shape(shape, token_ids.len(), &self.config)?;

        InputEmbeddings::new(
            data.to_vec(),
            usize::try_from(shape[0]).expect("validated non-negative batch size"),
            usize::try_from(shape[1]).expect("validated non-negative sequence length"),
            usize::try_from(shape[2]).expect("validated non-negative hidden size"),
        )
    }
}

fn validate_session_contract(session: &Session, config: &EmbeddingConfig) -> Result<()> {
    let inputs = session.inputs();
    if inputs.len() != 1 {
        return Err(Error::InvalidEmbeddingModel {
            reason: format!("expected 1 input, found {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    if input.name() != EMBEDDING_INPUT_NAME {
        return Err(Error::InvalidEmbeddingModel {
            reason: format!(
                "expected input `{EMBEDDING_INPUT_NAME}`, found `{}`",
                input.name()
            ),
        });
    }
    validate_tensor_metadata(
        input.dtype(),
        EMBEDDING_INPUT_NAME,
        TensorElementType::Int64,
        2,
        &[],
    )?;

    let outputs = session.outputs();
    if outputs.len() != 1 {
        return Err(Error::InvalidEmbeddingModel {
            reason: format!("expected 1 output, found {}", outputs.len()),
        });
    }

    let output = &outputs[0];
    if output.name() != EMBEDDING_OUTPUT_NAME {
        return Err(Error::InvalidEmbeddingModel {
            reason: format!(
                "expected output `{EMBEDDING_OUTPUT_NAME}`, found `{}`",
                output.name()
            ),
        });
    }
    validate_tensor_metadata(
        output.dtype(),
        EMBEDDING_OUTPUT_NAME,
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
        return Err(Error::InvalidEmbeddingModel {
            reason: format!("`{name}` is not a tensor"),
        });
    };
    if *ty != expected_type {
        return Err(Error::InvalidEmbeddingModel {
            reason: format!("`{name}` has type {ty}, expected {expected_type}"),
        });
    }
    if shape.len() != expected_rank {
        return Err(Error::InvalidEmbeddingModel {
            reason: format!(
                "`{name}` has rank {}, expected {expected_rank}",
                shape.len()
            ),
        });
    }
    for &(axis, expected) in static_dims {
        let actual = shape[axis];
        if actual >= 0 && actual as usize != expected {
            return Err(Error::InvalidEmbeddingModel {
                reason: format!("`{name}` dimension {axis} is {actual}, expected {expected}"),
            });
        }
    }

    Ok(())
}

fn validate_token_ids(token_ids: &[i64], config: &EmbeddingConfig) -> Result<()> {
    if token_ids.is_empty() {
        return Err(Error::InvalidEmbeddingInput {
            reason: "token_ids must contain at least one token".to_owned(),
        });
    }

    for (index, &token_id) in token_ids.iter().enumerate() {
        if token_id < 0 {
            return Err(Error::InvalidEmbeddingInput {
                reason: format!("token_ids[{index}] is negative: {token_id}"),
            });
        }

        if token_id as usize >= config.vocab_size {
            return Err(Error::InvalidEmbeddingInput {
                reason: format!(
                    "token_ids[{index}]={token_id} exceeds vocabulary size {}",
                    config.vocab_size
                ),
            });
        }
    }

    Ok(())
}

fn validate_embedding_shape(
    shape: &[i64],
    sequence_length: usize,
    config: &EmbeddingConfig,
) -> Result<()> {
    if shape.len() != 3 {
        return Err(Error::InvalidEmbeddingOutput {
            reason: format!(
                "`{EMBEDDING_OUTPUT_NAME}` has rank {}, expected 3",
                shape.len()
            ),
        });
    }
    if shape.iter().any(|&dim| dim < 0) {
        return Err(Error::InvalidEmbeddingOutput {
            reason: format!("`{EMBEDDING_OUTPUT_NAME}` has negative shape {shape:?}"),
        });
    }
    if shape[0] != 1 {
        return Err(Error::InvalidEmbeddingOutput {
            reason: format!(
                "`{EMBEDDING_OUTPUT_NAME}` batch size is {}, expected 1",
                shape[0]
            ),
        });
    }
    if shape[1] as usize != sequence_length {
        return Err(Error::InvalidEmbeddingOutput {
            reason: format!(
                "`{EMBEDDING_OUTPUT_NAME}` sequence length is {}, expected {sequence_length}",
                shape[1]
            ),
        });
    }
    if shape[2] as usize != config.hidden_size {
        return Err(Error::InvalidEmbeddingOutput {
            reason: format!(
                "`{EMBEDDING_OUTPUT_NAME}` hidden size is {}, expected {}",
                shape[2], config.hidden_size
            ),
        });
    }

    Ok(())
}
