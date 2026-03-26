use crate::attention::memory_cross::MemoryContext;
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

/// Conditioning inputs that remain independent of memory state.
#[derive(Debug, Clone)]
pub struct DiffusionCondition {
    pub text_embedding: Vec<Vec<f32>>,
    pub timestep: f32,
}

pub trait DiffusionBackbone: Send + Sync {
    fn denoise_step(
        &self,
        noisy_latent: &[f32],
        shape: &[usize; 5],
        condition: &DiffusionCondition,
        memory_context: Option<&MemoryContext>,
    ) -> Result<Vec<f32>, BackboneError>;
}

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
            .flat_map(|token| token.iter().map(|value| value.tanh()))
            .collect();
        if flattened.is_empty() {
            0.0
        } else {
            flattened.iter().sum::<f32>() / flattened.len() as f32
        }
    }

    fn summarize_memory(context: Option<&MemoryContext>) -> f32 {
        let Some(context) = context else {
            return 0.0;
        };
        if context.effective_memory_gate() <= 0.0 {
            return 0.0;
        }
        let tokens = context.active_tokens();
        if tokens.is_empty() {
            return 0.0;
        }
        let flattened: Vec<f32> = tokens
            .iter()
            .flat_map(|token| token.iter().map(|value| value.tanh()))
            .collect();
        if flattened.is_empty() {
            0.0
        } else {
            (flattened.iter().sum::<f32>() / flattened.len() as f32)
                * context.effective_memory_gate()
        }
    }

    fn summarize_positions(context: Option<&MemoryContext>) -> f32 {
        let Some(context) = context else {
            return 0.0;
        };
        if context.effective_memory_gate() <= 0.0 {
            return 0.0;
        }
        let positions = context.active_positions();
        if positions.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for position in positions {
            sum += position[0] * 0.02 + position[1] * 0.02 + position[2] * 0.1;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f32) * context.effective_memory_gate()
        }
    }

    fn summarize_mask(context: Option<&MemoryContext>) -> f32 {
        let Some(context) = context else {
            return 0.0;
        };
        if context.coverage_mask.is_empty() || context.effective_memory_gate() <= 0.0 {
            return 0.0;
        }
        let covered = context.coverage_mask.iter().filter(|&&value| value).count() as f32;
        (covered / context.coverage_mask.len() as f32) * context.effective_memory_gate()
    }

    fn summarize_warp(context: Option<&MemoryContext>) -> f32 {
        let Some(context) = context else {
            return 0.0;
        };
        context.warp_valid_ratio() * context.effective_memory_gate()
    }

    fn approx_alpha_bar(timestep: f32) -> f32 {
        let progress = timestep.clamp(0.0, 1.0);
        let alpha = (1.0 - progress).powi(2);
        alpha.clamp(0.02, 0.995)
    }

    fn index(shape: &[usize; 5], coords: (usize, usize, usize, usize, usize)) -> usize {
        let (batch, channel, frame, height, width) = coords;
        ((((batch * shape[1] + channel) * shape[2] + frame) * shape[3] + height) * shape[4]) + width
    }

    fn local_average(
        latent: &[f32],
        shape: &[usize; 5],
        coords: (usize, usize, usize, usize, usize),
    ) -> f32 {
        let (batch, channel, frame, height, width) = coords;
        let mut sum = 0.0;
        let mut count = 0usize;
        let frame_start = frame.saturating_sub(1);
        let frame_end = usize::min(frame + 1, shape[2].saturating_sub(1));
        let height_start = height.saturating_sub(1);
        let height_end = usize::min(height + 1, shape[3].saturating_sub(1));
        let width_start = width.saturating_sub(1);
        let width_end = usize::min(width + 1, shape[4].saturating_sub(1));

        for dt in frame_start..=frame_end {
            for dy in height_start..=height_end {
                for dx in width_start..=width_end {
                    let idx = Self::index(shape, (batch, channel, dt, dy, dx));
                    sum += latent[idx];
                    count += 1;
                }
            }
        }

        if count == 0 {
            latent[Self::index(shape, (batch, channel, frame, height, width))]
        } else {
            sum / count as f32
        }
    }

    fn target_latent(
        &self,
        noisy_latent: &[f32],
        shape: &[usize; 5],
        condition: &DiffusionCondition,
        memory_context: Option<&MemoryContext>,
    ) -> Vec<f32> {
        let [b, c, t, h, w] = *shape;
        let text_sig = Self::summarize_text(condition);
        let memory_sig = Self::summarize_memory(memory_context);
        let position_sig = Self::summarize_positions(memory_context);
        let mask_sig = Self::summarize_mask(memory_context);
        let warp_sig = Self::summarize_warp(memory_context);
        let timestep = condition.timestep.clamp(0.0, 1.0);
        let condition_strength = self.noise_scale * (0.08 + 0.17 * timestep);
        let global_bias = 0.03 * text_sig
            + 0.07 * memory_sig
            + 0.03 * position_sig
            + 0.03 * mask_sig
            + 0.02 * warp_sig;
        let spatial_memory = memory_context
            .filter(|context| context.effective_memory_gate() > 0.0)
            .and_then(MemoryContext::canvas_cthw)
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
                                ((channel + 1) as f32 * 0.13 + text_sig * 0.7 + position_sig * 0.3)
                                    .sin();
                            let memory_anchor = spatial_memory
                                .as_ref()
                                .map(|memory| memory[idx])
                                .unwrap_or(local);
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
        memory_context: Option<&MemoryContext>,
    ) -> Result<Vec<f32>, BackboneError> {
        let target = self.target_latent(noisy_latent, shape, condition, memory_context);
        let alpha_bar = Self::approx_alpha_bar(condition.timestep);
        let sqrt_alpha = alpha_bar.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar).sqrt().max(1e-4);
        let correction = self.noise_scale.max(1e-3);

        Ok(noisy_latent
            .iter()
            .zip(target.iter())
            .map(|(value, target)| {
                ((value - sqrt_alpha * target) / sqrt_one_minus_alpha) * correction
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::memory_cross::MemoryFrameContext;
    use crate::attention::prope::{PRoPE, PRoPEOperator};
    use crate::attention::warped_latent::WarpGrid;
    use crate::backend::AblationConfig;
    use crate::camera::{CameraIntrinsics, CameraPose};
    use crate::memory::mosaic::{LatentCanvas, MosaicFrame};

    fn test_memory_context() -> MemoryContext {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let transform = PRoPE::new(8, 1, 1)
            .compute_projective_transform(
                &CameraPose::identity(0.0),
                &pose,
                &intrinsics,
                &intrinsics,
            )
            .unwrap();

        MemoryContext {
            frames: vec![MemoryFrameContext {
                mosaic: MosaicFrame {
                    target_pose: pose,
                    patches: Vec::new(),
                    coverage_mask: vec![vec![true; 2]; 2],
                    width: 32,
                    height: 32,
                },
                warp_grids: vec![WarpGrid {
                    target_coords: vec![(0.0, 0.0), (1.0, 1.0)],
                    valid_mask: vec![true, false],
                    source_shape: (1, 2),
                }],
                prope_transform: Some(transform),
                base_tokens: vec![vec![0.8; 8], vec![0.2; 8]],
                warped_value_tokens: vec![vec![0.5; 8], vec![0.1; 8]],
                base_positions: vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                warped_positions: vec![[0.5, 0.0, 0.2], [1.5, 1.0, 0.4]],
            }],
            rasterized_canvas: Some(LatentCanvas {
                data: vec![1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                frames: 1,
                height: 2,
                width: 2,
                channels: 2,
            }),
            coverage_mask: vec![true, false, true, true],
            ablation: AblationConfig::default(),
        }
    }

    #[test]
    fn test_synthetic_backbone() {
        let backbone = SyntheticBackbone::new(0.5);
        let latent = vec![1.0f32; 2 * 4 * 2 * 8 * 8];
        let shape = [2, 4, 2, 8, 8];
        let condition = DiffusionCondition {
            text_embedding: vec![vec![0.0; 64]; 10],
            timestep: 0.8,
        };

        let result = backbone
            .denoise_step(&latent, &shape, &condition, None)
            .unwrap();
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
            timestep: 0.5,
        };
        let memory_context = test_memory_context();

        let base_out = backbone.denoise_step(&latent, &shape, &base, None).unwrap();
        let memory_out = backbone
            .denoise_step(&latent, &shape, &base, Some(&memory_context))
            .unwrap();
        assert_ne!(base_out, memory_out);
    }

    #[test]
    fn test_timestep_modulates_strength() {
        let backbone = SyntheticBackbone::new(0.9);
        let latent = vec![0.6f32; 3 * 3 * 3];
        let shape = [1, 3, 1, 3, 3];
        let condition_early = DiffusionCondition {
            text_embedding: vec![vec![0.1; 4]],
            timestep: 0.9,
        };
        let condition_late = DiffusionCondition {
            timestep: 0.1,
            ..condition_early.clone()
        };

        let early = backbone
            .denoise_step(&latent, &shape, &condition_early, None)
            .unwrap();
        let late = backbone
            .denoise_step(&latent, &shape, &condition_late, None)
            .unwrap();
        assert_ne!(early, late);
        assert!(
            early
                .iter()
                .zip(late.iter())
                .any(|(left, right)| (left - right).abs() > 1e-6)
        );
    }

    #[test]
    fn test_memory_context_changes_spatial_output() {
        let backbone = SyntheticBackbone::new(1.0);
        let latent = vec![0.0f32; 2 * 2 * 2];
        let shape = [1, 2, 1, 2, 2];
        let condition = DiffusionCondition {
            text_embedding: vec![],
            timestep: 0.5,
        };
        let with_memory = backbone
            .denoise_step(&latent, &shape, &condition, Some(&test_memory_context()))
            .unwrap();
        let without_memory = backbone
            .denoise_step(&latent, &shape, &condition, None)
            .unwrap();

        assert_ne!(with_memory, without_memory);
        assert!(with_memory[0].abs() > without_memory[0].abs());
    }

    #[test]
    fn test_memory_gate_override_zero_matches_none() {
        let backbone = SyntheticBackbone::new(0.8);
        let latent = vec![0.1f32; 16];
        let shape = [1, 2, 2, 2, 2];
        let condition = DiffusionCondition {
            text_embedding: vec![vec![0.2; 4]],
            timestep: 0.5,
        };
        let mut context = test_memory_context();
        context.ablation.memory_gate_override = Some(0.0);

        let gated = backbone
            .denoise_step(&latent, &shape, &condition, Some(&context))
            .unwrap();
        let none = backbone
            .denoise_step(&latent, &shape, &condition, None)
            .unwrap();
        assert_eq!(gated, none);
    }
}
