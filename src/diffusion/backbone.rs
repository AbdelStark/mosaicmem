use thiserror::Error;

#[derive(Error, Debug)]
pub enum BackboneError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },
}

/// Conditioning inputs for the diffusion backbone.
#[derive(Debug, Clone)]
pub struct DiffusionCondition {
    /// Text embedding tokens [num_tokens, dim].
    pub text_embedding: Vec<Vec<f32>>,
    /// Memory cross-attention tokens [num_mem_tokens, dim] (from MosaicMem).
    pub memory_tokens: Option<Vec<Vec<f32>>>,
    /// Memory coverage mask.
    pub memory_mask: Option<Vec<bool>>,
    /// PRoPE camera rotations per frame.
    pub camera_rotations: Option<Vec<Vec<[f32; 2]>>>,
    /// Timestep for the denoising step.
    pub timestep: f32,
}

/// Trait for the diffusion transformer (DiT) backbone.
/// Implementations can wrap Tract ONNX inference or other backends.
pub trait DiffusionBackbone: Send + Sync {
    /// Run a single denoising step.
    ///
    /// # Arguments
    /// * `noisy_latent` - The noisy latent tensor [B, C, T, H, W] flattened
    /// * `shape` - [B, C, T, H, W]
    /// * `condition` - Conditioning inputs (text, memory, camera)
    ///
    /// # Returns
    /// Predicted noise or velocity [B, C, T, H, W] flattened
    fn denoise_step(
        &self,
        noisy_latent: &[f32],
        shape: &[usize; 5],
        condition: &DiffusionCondition,
    ) -> Result<Vec<f32>, BackboneError>;
}

/// A synthetic diffusion backbone for testing.
/// Predicts noise as a scaled version of the input (trivial, for pipeline testing).
pub struct SyntheticBackbone {
    pub noise_scale: f32,
}

impl SyntheticBackbone {
    pub fn new(noise_scale: f32) -> Self {
        Self { noise_scale }
    }
}

impl DiffusionBackbone for SyntheticBackbone {
    fn denoise_step(
        &self,
        noisy_latent: &[f32],
        _shape: &[usize; 5],
        condition: &DiffusionCondition,
    ) -> Result<Vec<f32>, BackboneError> {
        // Simple: predict noise as scaled input weighted by timestep
        let scale = self.noise_scale * condition.timestep;
        Ok(noisy_latent.iter().map(|x| x * scale).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_backbone() {
        let backbone = SyntheticBackbone::new(0.5);
        let latent = vec![1.0f32; 2 * 4 * 2 * 8 * 8];
        let shape = [2, 4, 2, 8, 8];
        let condition = DiffusionCondition {
            text_embedding: vec![vec![0.0; 64]; 10],
            memory_tokens: None,
            memory_mask: None,
            camera_rotations: None,
            timestep: 0.8,
        };

        let result = backbone.denoise_step(&latent, &shape, &condition).unwrap();
        assert_eq!(result.len(), latent.len());
        assert!((result[0] - 0.4).abs() < 1e-5); // 1.0 * 0.5 * 0.8
    }
}
