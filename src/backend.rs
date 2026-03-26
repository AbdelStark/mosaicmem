//! Backend selection, tensor payload schemas, and bridge interfaces for
//! synthetic and real model integrations.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
    #[default]
    Synthetic,
    Real,
}

impl BackendMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Synthetic => "[synthetic]",
            Self::Real => "[real]",
        }
    }
}

impl fmt::Display for BackendMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AblationConfig {
    pub enable_memory: bool,
    pub enable_prope: bool,
    pub enable_warped_rope: bool,
    pub enable_warped_latent: bool,
    pub memory_gate_override: Option<f32>,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            enable_memory: true,
            enable_prope: true,
            enable_warped_rope: true,
            enable_warped_latent: true,
            memory_gate_override: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorDType {
    F32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorPayload {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: TensorDType,
}

impl TensorPayload {
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            dtype: TensorDType::F32,
        }
    }

    pub fn from_bool(data: Vec<bool>, shape: Vec<usize>) -> Self {
        Self {
            data: data
                .into_iter()
                .map(|value| if value { 1.0 } else { 0.0 })
                .collect(),
            shape,
            dtype: TensorDType::Bool,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn validate(&self) -> Result<(), BackendError> {
        let expected = self.element_count();
        if expected != self.data.len() {
            return Err(BackendError::TensorShapeMismatch {
                expected,
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendRequest {
    HealthCheck,
    Depth {
        frame: TensorPayload,
    },
    VaeEncode {
        frames: TensorPayload,
    },
    VaeDecode {
        latent: TensorPayload,
    },
    Denoise {
        latent: TensorPayload,
        timestep: f32,
        text_embedding: Vec<Vec<f32>>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BackendResponse {
    Healthy { backend: BackendMode },
    Tensor(TensorPayload),
    Error { message: String },
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("real backend requested but the `real-backend` Cargo feature is disabled")]
    RealBackendFeatureDisabled,
    #[error("checkpoint not found for real backend{path_suffix}")]
    CheckpointNotFound {
        path: Option<PathBuf>,
        path_suffix: String,
    },
    #[error("tensor payload shape mismatch: expected {expected} elements, got {actual}")]
    TensorShapeMismatch { expected: usize, actual: usize },
    #[error("backend sidecar timed out after {timeout_ms} ms")]
    SidecarTimeout { timeout_ms: u64 },
    #[error("backend sidecar protocol error: {0}")]
    SidecarProtocol(String),
    #[error("backend sidecar I/O error: {0}")]
    SidecarIo(String),
}

impl BackendError {
    pub fn checkpoint_not_found(path: Option<&Path>) -> Self {
        let owned = path.map(Path::to_path_buf);
        let path_suffix = owned
            .as_ref()
            .map(|p| format!(": {}", p.display()))
            .unwrap_or_default();
        Self::CheckpointNotFound {
            path: owned,
            path_suffix,
        }
    }
}

pub fn validate_backend_configuration(
    mode: BackendMode,
    checkpoint_path: Option<&Path>,
) -> Result<(), BackendError> {
    match mode {
        BackendMode::Synthetic => Ok(()),
        BackendMode::Real => {
            let Some(path) = checkpoint_path else {
                return Err(BackendError::checkpoint_not_found(None));
            };
            if !path.exists() {
                return Err(BackendError::checkpoint_not_found(Some(path)));
            }
            if !cfg!(feature = "real-backend") {
                return Err(BackendError::RealBackendFeatureDisabled);
            }
            Ok(())
        }
    }
}

pub trait BackendBridge: Send + Sync {
    fn health_check(&self) -> Result<(), BackendError>;
    fn infer_depth(&self, frame: &TensorPayload) -> Result<TensorPayload, BackendError>;
    fn infer_vae_encode(&self, frames: &TensorPayload) -> Result<TensorPayload, BackendError>;
    fn infer_vae_decode(&self, latent: &TensorPayload) -> Result<TensorPayload, BackendError>;
    fn infer_denoise(
        &self,
        latent: &TensorPayload,
        timestep: f32,
        text_embedding: &[Vec<f32>],
    ) -> Result<TensorPayload, BackendError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SyntheticBridge;

impl SyntheticBridge {
    fn passthrough(payload: &TensorPayload) -> Result<TensorPayload, BackendError> {
        payload.validate()?;
        Ok(payload.clone())
    }
}

impl BackendBridge for SyntheticBridge {
    fn health_check(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn infer_depth(&self, frame: &TensorPayload) -> Result<TensorPayload, BackendError> {
        frame.validate()?;
        if frame.shape.len() < 2 {
            return Self::passthrough(frame);
        }

        let height = frame.shape[frame.shape.len() - 2];
        let width = frame.shape[frame.shape.len() - 1];
        let channels = if height == 0 || width == 0 {
            1
        } else {
            frame.data.len() / (height * width)
        }
        .max(1);

        let plane = height.saturating_mul(width);
        let mut depth = vec![0.0f32; plane];
        for (pixel, depth_value) in depth.iter_mut().enumerate().take(plane) {
            let mut sum = 0.0;
            for channel in 0..channels {
                let idx = channel * plane + pixel;
                sum += frame.data.get(idx).copied().unwrap_or(0.0);
            }
            *depth_value = (sum / channels as f32).abs() + 1.0;
        }

        Ok(TensorPayload::from_f32(depth, vec![1, height, width]))
    }

    fn infer_vae_encode(&self, frames: &TensorPayload) -> Result<TensorPayload, BackendError> {
        Self::passthrough(frames)
    }

    fn infer_vae_decode(&self, latent: &TensorPayload) -> Result<TensorPayload, BackendError> {
        Self::passthrough(latent)
    }

    fn infer_denoise(
        &self,
        latent: &TensorPayload,
        timestep: f32,
        text_embedding: &[Vec<f32>],
    ) -> Result<TensorPayload, BackendError> {
        latent.validate()?;
        let text_signal = if text_embedding.is_empty() {
            0.0
        } else {
            let flat_count = text_embedding.iter().map(Vec::len).sum::<usize>().max(1);
            text_embedding
                .iter()
                .flat_map(|token| token.iter().copied())
                .sum::<f32>()
                / flat_count as f32
        };
        let scale = (1.0 - timestep.clamp(0.0, 1.0)) * 0.15 + text_signal.tanh() * 0.05;
        Ok(TensorPayload::from_f32(
            latent.data.iter().map(|value| value * scale).collect(),
            latent.shape.clone(),
        ))
    }
}

pub fn create_backend_bridge(
    mode: BackendMode,
    checkpoint_path: Option<&Path>,
) -> Result<Box<dyn BackendBridge>, BackendError> {
    match mode {
        BackendMode::Synthetic => Ok(Box::new(SyntheticBridge)),
        BackendMode::Real => {
            validate_backend_configuration(mode, checkpoint_path)?;
            #[cfg(feature = "real-backend")]
            {
                let checkpoint = checkpoint_path
                    .expect("checkpoint path already validated")
                    .to_path_buf();
                Ok(Box::new(PythonSidecarBridge::new(
                    checkpoint,
                    "python3".to_string(),
                    30_000,
                )))
            }
            #[cfg(not(feature = "real-backend"))]
            {
                Err(BackendError::RealBackendFeatureDisabled)
            }
        }
    }
}

#[cfg(feature = "real-backend")]
#[derive(Debug, Clone)]
pub struct PythonSidecarBridge {
    checkpoint_path: PathBuf,
    python_executable: String,
    timeout_ms: u64,
}

#[cfg(feature = "real-backend")]
impl PythonSidecarBridge {
    pub fn new(checkpoint_path: PathBuf, python_executable: String, timeout_ms: u64) -> Self {
        Self {
            checkpoint_path,
            python_executable,
            timeout_ms,
        }
    }

    fn request(&self, request: &BackendRequest) -> Result<BackendResponse, BackendError> {
        use std::io::Write;
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};

        let script = "import json, sys; req = json.loads(sys.stdin.read()); print(json.dumps({'kind': 'error', 'message': 'python sidecar stub not configured'}))";
        let mut child = Command::new(&self.python_executable)
            .arg("-c")
            .arg(script)
            .env("MOSAICMEM_CHECKPOINT", &self.checkpoint_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|err| BackendError::SidecarIo(err.to_string()))?;

        if let Some(stdin) = child.stdin.as_mut() {
            let payload = serde_json::to_vec(request)
                .map_err(|err| BackendError::SidecarProtocol(err.to_string()))?;
            stdin
                .write_all(&payload)
                .map_err(|err| BackendError::SidecarIo(err.to_string()))?;
        }

        let deadline = Instant::now() + Duration::from_millis(self.timeout_ms.max(1));
        loop {
            if Instant::now() > deadline {
                let _ = child.kill();
                return Err(BackendError::SidecarTimeout {
                    timeout_ms: self.timeout_ms,
                });
            }

            match child.try_wait() {
                Ok(Some(_)) => {
                    let output = child
                        .wait_with_output()
                        .map_err(|err| BackendError::SidecarIo(err.to_string()))?;
                    let response: BackendResponse = serde_json::from_slice(&output.stdout)
                        .map_err(|err| BackendError::SidecarProtocol(err.to_string()))?;
                    return Ok(response);
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(err) => return Err(BackendError::SidecarIo(err.to_string())),
            }
        }
    }

    fn tensor_response(&self, request: BackendRequest) -> Result<TensorPayload, BackendError> {
        match self.request(&request)? {
            BackendResponse::Tensor(tensor) => Ok(tensor),
            BackendResponse::Healthy { .. } => Err(BackendError::SidecarProtocol(
                "expected tensor response, received health check".to_string(),
            )),
            BackendResponse::Error { message } => Err(BackendError::SidecarProtocol(message)),
        }
    }
}

#[cfg(feature = "real-backend")]
impl BackendBridge for PythonSidecarBridge {
    fn health_check(&self) -> Result<(), BackendError> {
        match self.request(&BackendRequest::HealthCheck)? {
            BackendResponse::Healthy { .. } => Ok(()),
            BackendResponse::Tensor(_) => Err(BackendError::SidecarProtocol(
                "expected health response, received tensor".to_string(),
            )),
            BackendResponse::Error { message } => Err(BackendError::SidecarProtocol(message)),
        }
    }

    fn infer_depth(&self, frame: &TensorPayload) -> Result<TensorPayload, BackendError> {
        self.tensor_response(BackendRequest::Depth {
            frame: frame.clone(),
        })
    }

    fn infer_vae_encode(&self, frames: &TensorPayload) -> Result<TensorPayload, BackendError> {
        self.tensor_response(BackendRequest::VaeEncode {
            frames: frames.clone(),
        })
    }

    fn infer_vae_decode(&self, latent: &TensorPayload) -> Result<TensorPayload, BackendError> {
        self.tensor_response(BackendRequest::VaeDecode {
            latent: latent.clone(),
        })
    }

    fn infer_denoise(
        &self,
        latent: &TensorPayload,
        timestep: f32,
        text_embedding: &[Vec<f32>],
    ) -> Result<TensorPayload, BackendError> {
        self.tensor_response(BackendRequest::Denoise {
            latent: latent.clone(),
            timestep,
            text_embedding: text_embedding.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_mode_labels() {
        assert_eq!(BackendMode::Synthetic.label(), "[synthetic]");
        assert_eq!(BackendMode::Real.label(), "[real]");
    }

    #[test]
    fn test_real_backend_requires_checkpoint() {
        let result = validate_backend_configuration(BackendMode::Real, None);
        assert!(matches!(
            result,
            Err(BackendError::CheckpointNotFound { .. })
        ));
    }

    #[test]
    fn test_tensor_payload_validation() {
        let payload = TensorPayload::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        assert!(payload.validate().is_ok());
    }

    #[test]
    fn test_synthetic_bridge_health_check() {
        let bridge = SyntheticBridge;
        assert!(bridge.health_check().is_ok());
    }
}
