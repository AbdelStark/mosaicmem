//! Backend selection and ablation controls for the synthetic scaffold and
//! future real-checkpoint integrations.

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

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("real backend requested but the `real-backend` Cargo feature is disabled")]
    RealBackendFeatureDisabled,
    #[error("checkpoint not found for real backend{path_suffix}")]
    CheckpointNotFound {
        path: Option<PathBuf>,
        path_suffix: String,
    },
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
}
