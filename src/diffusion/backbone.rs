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
    /// Rasterized memory latent in [C, T, H, W] layout for spatial guidance.
    pub memory_latent: Option<Vec<f32>>,
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
/// Predicts noise from a deterministic denoising target that blends local
/// structure with text, memory, and camera conditioning.
pub struct SyntheticBackbone {
    pub noise_scale: f32,
}

impl SyntheticBackbone {
    pub fn new(noise_scale: f32) -> Self {
        Self { noise_scale }
    }

    fn summarize_text(condition: &DiffusionCondition) -> f32 {
        if condition.text_embedding.is_empty() {
            return 0.0;
        }
        let flattened: Vec<f32> = condition
            .text_embedding
            .iter()
            .flat_map(|token| token.iter().map(|v| v.tanh()))
            .collect();
        if flattened.is_empty() {
            return 0.0;
        }
        flattened.iter().sum::<f32>() / flattened.len() as f32
    }

    fn summarize_memory(condition: &DiffusionCondition) -> f32 {
        let Some(tokens) = &condition.memory_tokens else {
            return 0.0;
        };
        if tokens.is_empty() {
            return 0.0;
        }
        let flattened: Vec<f32> = tokens
            .iter()
            .flat_map(|token| token.iter().map(|v| v.tanh()))
            .collect();
        if flattened.is_empty() {
            return 0.0;
        }
        flattened.iter().sum::<f32>() / flattened.len() as f32
    }

    fn summarize_camera(condition: &DiffusionCondition) -> f32 {
        let Some(rotations) = &condition.camera_rotations else {
            return 0.0;
        };
        if rotations.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for frame in rotations {
            for pair in frame {
                sum += pair[0] - pair[1];
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    fn summarize_mask(condition: &DiffusionCondition) -> f32 {
        let Some(mask) = &condition.memory_mask else {
            return 0.0;
        };
        if mask.is_empty() {
            0.0
        } else {
            mask.iter().filter(|&&v| v).count() as f32 / mask.len() as f32
        }
    }

    fn approx_alpha_bar(timestep: f32) -> f32 {
        let progress = timestep.clamp(0.0, 1.0);
        let alpha = (1.0 - progress).powi(2);
        alpha.clamp(0.02, 0.995)
    }

    fn index(shape: &[usize; 5], coords: (usize, usize, usize, usize, usize)) -> usize {
        let (b, c, t, h, w) = coords;
        ((((b * shape[1] + c) * shape[2] + t) * shape[3] + h) * shape[4]) + w
    }

    fn local_average(
        latent: &[f32],
        shape: &[usize; 5],
        coords: (usize, usize, usize, usize, usize),
    ) -> f32 {
        let (b, c, t, h, w) = coords;
        let mut sum = 0.0;
        let mut count = 0usize;
        let t_start = t.saturating_sub(1);
        let t_end = usize::min(t + 1, shape[2].saturating_sub(1));
        let h_start = h.saturating_sub(1);
        let h_end = usize::min(h + 1, shape[3].saturating_sub(1));
        let w_start = w.saturating_sub(1);
        let w_end = usize::min(w + 1, shape[4].saturating_sub(1));
        for dt in t_start..=t_end {
            for dh in h_start..=h_end {
                for dw in w_start..=w_end {
                    let idx = Self::index(shape, (b, c, dt, dh, dw));
                    sum += latent[idx];
                    count += 1;
                }
            }
        }
        if count == 0 {
            latent[Self::index(shape, (b, c, t, h, w))]
        } else {
            sum / count as f32
        }
    }

    fn target_latent(
        &self,
        noisy_latent: &[f32],
        shape: &[usize; 5],
        condition: &DiffusionCondition,
    ) -> Vec<f32> {
        let [b, c, t, h, w] = *shape;
        let text_sig = Self::summarize_text(condition);
        let memory_sig = Self::summarize_memory(condition);
        let camera_sig = Self::summarize_camera(condition);
        let mask_sig = Self::summarize_mask(condition);
        let timestep = condition.timestep.clamp(0.0, 1.0);
        let condition_strength = self.noise_scale * (0.08 + 0.17 * timestep);
        let global_bias = 0.03 * text_sig + 0.07 * memory_sig + 0.03 * camera_sig + 0.03 * mask_sig;
        let spatial_memory = condition
            .memory_latent
            .as_ref()
            .filter(|memory| memory.len() == noisy_latent.len());

        let mut target = vec![0.0f32; noisy_latent.len()];
        let denom_t = t.max(1) as f32;
        let denom_h = h.max(1) as f32;
        let denom_w = w.max(1) as f32;

        for batch in 0..b {
            for channel in 0..c {
                for ti in 0..t {
                    let t_phase = ti as f32 / denom_t;
                    for yi in 0..h {
                        let y_phase = yi as f32 / denom_h;
                        for xi in 0..w {
                            let idx = Self::index(shape, (batch, channel, ti, yi, xi));
                            let local = Self::local_average(
                                noisy_latent,
                                shape,
                                (batch, channel, ti, yi, xi),
                            );
                            let spatial_wave =
                                ((t_phase + y_phase + y_phase + xi as f32 / denom_w)
                                    * std::f32::consts::TAU)
                                    .sin();
                            let channel_bias =
                                ((channel + 1) as f32 * 0.13 + text_sig * 0.7 + camera_sig * 0.3)
                                    .sin();
                            let memory_anchor =
                                spatial_memory.map(|memory| memory[idx]).unwrap_or(local);
                            let anchor = if spatial_memory.is_some() {
                                0.15 * noisy_latent[idx] + 0.10 * local + 0.75 * memory_anchor
                            } else {
                                0.65 * noisy_latent[idx] + 0.35 * local
                            };
                            let cond_delta = condition_strength
                                * (spatial_wave * 0.5 + channel_bias * 0.3 + global_bias);
                            target[idx] = (anchor + cond_delta).clamp(-2.0, 2.0);
                        }
                    }
                }
            }
        }

        target
    }
}

impl DiffusionBackbone for SyntheticBackbone {
    fn denoise_step(
        &self,
        noisy_latent: &[f32],
        shape: &[usize; 5],
        condition: &DiffusionCondition,
    ) -> Result<Vec<f32>, BackboneError> {
        let target = self.target_latent(noisy_latent, shape, condition);
        let alpha_bar = Self::approx_alpha_bar(condition.timestep);
        let sqrt_alpha = alpha_bar.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar).sqrt().max(1e-4);
        let correction = self.noise_scale.max(1e-3);

        Ok(noisy_latent
            .iter()
            .zip(target.iter())
            .map(|(x, tgt)| ((x - sqrt_alpha * tgt) / sqrt_one_minus_alpha) * correction)
            .collect())
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
            memory_latent: None,
            memory_mask: None,
            camera_rotations: None,
            timestep: 0.8,
        };

        let result = backbone.denoise_step(&latent, &shape, &condition).unwrap();
        assert_eq!(result.len(), latent.len());
        assert!(result[0].is_finite());
    }

    #[test]
    fn test_condition_changes_output() {
        let backbone = SyntheticBackbone::new(0.7);
        let latent = vec![0.2f32; 3 * 2 * 4 * 4];
        let shape = [1, 3, 2, 4, 4];

        let base = DiffusionCondition {
            text_embedding: vec![vec![0.0; 8]; 2],
            memory_tokens: None,
            memory_latent: None,
            memory_mask: None,
            camera_rotations: None,
            timestep: 0.5,
        };
        let memory = DiffusionCondition {
            text_embedding: vec![vec![0.0; 8]; 2],
            memory_tokens: Some(vec![vec![0.8; 8], vec![0.2; 8]]),
            memory_latent: None,
            memory_mask: Some(vec![true, false, true, true]),
            camera_rotations: Some(vec![vec![[0.9, 0.1]; 4]]),
            timestep: 0.5,
        };

        let base_out = backbone.denoise_step(&latent, &shape, &base).unwrap();
        let memory_out = backbone.denoise_step(&latent, &shape, &memory).unwrap();
        assert_ne!(base_out, memory_out);
    }

    #[test]
    fn test_timestep_modulates_strength() {
        let backbone = SyntheticBackbone::new(0.9);
        let latent = vec![0.6f32; 3 * 3 * 3];
        let shape = [1, 3, 1, 3, 3];
        let condition_early = DiffusionCondition {
            text_embedding: vec![vec![0.1; 4]],
            memory_tokens: Some(vec![vec![0.3; 4]]),
            memory_latent: None,
            memory_mask: None,
            camera_rotations: None,
            timestep: 0.9,
        };
        let condition_late = DiffusionCondition {
            timestep: 0.1,
            ..condition_early.clone()
        };

        let early = backbone
            .denoise_step(&latent, &shape, &condition_early)
            .unwrap();
        let late = backbone
            .denoise_step(&latent, &shape, &condition_late)
            .unwrap();
        assert_ne!(early, late);
        assert!(
            early
                .iter()
                .zip(late.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    #[test]
    fn test_memory_latent_changes_spatial_output() {
        let backbone = SyntheticBackbone::new(1.0);
        let latent = vec![0.0f32; 2 * 2 * 2];
        let shape = [1, 2, 1, 2, 2];
        let mut memory_latent = vec![0.0f32; latent.len()];
        memory_latent[0] = 1.0;
        memory_latent[shape[2] * shape[3] * shape[4]] = 0.5;

        let with_memory = DiffusionCondition {
            text_embedding: vec![],
            memory_tokens: Some(vec![vec![1.0; 2]]),
            memory_latent: Some(memory_latent),
            memory_mask: Some(vec![true; 4]),
            camera_rotations: None,
            timestep: 0.5,
        };
        let without_memory = DiffusionCondition {
            text_embedding: vec![],
            memory_tokens: None,
            memory_latent: None,
            memory_mask: None,
            camera_rotations: None,
            timestep: 0.5,
        };

        let output_with = backbone
            .denoise_step(&latent, &shape, &with_memory)
            .unwrap();
        let output_without = backbone
            .denoise_step(&latent, &shape, &without_memory)
            .unwrap();

        assert_ne!(output_with, output_without);
        assert!(output_with[0].abs() > output_without[0].abs());
    }
}
