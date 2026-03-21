use crate::camera::CameraTrajectory;
use crate::diffusion::backbone::DiffusionBackbone;
use crate::diffusion::scheduler::NoiseScheduler;
use crate::diffusion::vae::VAE;
use crate::geometry::depth::DepthEstimator;
use crate::pipeline::config::PipelineConfig;
use crate::pipeline::inference::{InferencePipeline, PipelineError, PipelineStats};
use tracing::{info, warn};

/// Callback for receiving generated frames.
pub type FrameCallback = Box<dyn FnMut(usize, &[f32], &[usize; 5]) + Send>;

/// Autoregressive pipeline: chains multiple generation windows for long-horizon video.
///
/// Per window:
/// 1. Get camera poses for this window
/// 2. Retrieve memory mosaic from store
/// 3. Run DiT denoising loop
/// 4. Decode via VAE
/// 5. Update memory with new keyframes
/// 6. Yield frames
pub struct AutoregressivePipeline {
    pub pipeline: InferencePipeline,
    pub config: PipelineConfig,
}

impl AutoregressivePipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let pipeline = InferencePipeline::new(config.clone());
        Self { pipeline, config }
    }

    /// Generate a full video from a camera trajectory.
    ///
    /// # Arguments
    /// * `trajectory` - Full camera trajectory
    /// * `text_embedding` - Text conditioning tokens
    /// * `backbone` - Diffusion backbone
    /// * `scheduler` - Noise scheduler
    /// * `vae` - VAE for encode/decode
    /// * `depth_estimator` - Depth estimation model
    /// * `callback` - Optional callback for each generated window
    ///
    /// # Returns
    /// All generated frames concatenated + stats
    pub fn generate(
        &mut self,
        trajectory: &CameraTrajectory,
        text_embedding: &[Vec<f32>],
        backbone: &dyn DiffusionBackbone,
        scheduler: &dyn NoiseScheduler,
        vae: &dyn VAE,
        depth_estimator: &dyn DepthEstimator,
        mut callback: Option<FrameCallback>,
    ) -> Result<(Vec<f32>, Vec<[usize; 5]>), PipelineError> {
        let total_frames = trajectory.len();
        let window_size = self.config.window_size;
        let overlap = self.config.window_overlap;
        let stride = window_size.saturating_sub(overlap).max(1);

        info!(
            "Starting autoregressive generation: {} frames, window={}, overlap={}, stride={}",
            total_frames, window_size, overlap, stride
        );

        let mut all_frames = Vec::new();
        let mut all_shapes = Vec::new();
        let mut window_start = 0;
        let mut window_idx = 0;

        while window_start < total_frames {
            let window_end = (window_start + window_size).min(total_frames);
            let window_poses = &trajectory.poses[window_start..window_end];

            info!(
                "Window {}: frames {}-{} ({} frames), memory: {}",
                window_idx,
                window_start,
                window_end - 1,
                window_poses.len(),
                self.pipeline.stats()
            );

            // Generate this window
            let (frames, shape) = self.pipeline.generate_window(
                window_poses,
                text_embedding,
                backbone,
                scheduler,
                vae,
            )?;

            // Update memory with generated frames
            if let Err(e) =
                self.pipeline
                    .update_memory(&frames, &shape, window_poses, depth_estimator, vae)
            {
                warn!("Memory update failed for window {}: {}", window_idx, e);
            }

            // Invoke callback
            if let Some(ref mut cb) = callback {
                cb(window_idx, &frames, &shape);
            }

            all_frames.extend_from_slice(&frames);
            all_shapes.push(shape);

            window_start += stride;
            window_idx += 1;
        }

        info!(
            "Generation complete: {} windows, final memory: {}",
            window_idx,
            self.pipeline.stats()
        );

        Ok((all_frames, all_shapes))
    }

    /// Get current pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        self.pipeline.stats()
    }

    /// Reset the memory store (start fresh).
    pub fn reset_memory(&mut self) {
        self.pipeline = InferencePipeline::new(self.config.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraPose;
    use crate::diffusion::backbone::SyntheticBackbone;
    use crate::diffusion::scheduler::DDPMScheduler;
    use crate::diffusion::vae::SyntheticVAE;
    use crate::geometry::depth::SyntheticDepthEstimator;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_autoregressive_pipeline() {
        let mut config = PipelineConfig::default();
        config.num_inference_steps = 2;
        config.width = 64;
        config.height = 64;
        config.window_size = 4;
        config.window_overlap = 1;

        let mut pipeline = AutoregressivePipeline::new(config);

        let trajectory = CameraTrajectory::new(
            (0..8)
                .map(|i| {
                    CameraPose::from_translation_rotation(
                        i as f64 * 0.1,
                        Vector3::new(i as f32 * 0.5, 0.0, 0.0),
                        UnitQuaternion::identity(),
                    )
                })
                .collect(),
        );

        let backbone = SyntheticBackbone::new(0.1);
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        let vae = SyntheticVAE::new(8, 4, 16);
        let depth = SyntheticDepthEstimator::new(5.0, 1.0);
        let text_emb = vec![vec![0.0f32; 64]; 10];

        let result = pipeline.generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            None,
        );

        assert!(result.is_ok());
        let (frames, shapes) = result.unwrap();
        assert!(!frames.is_empty());
        assert!(!shapes.is_empty());
    }
}
