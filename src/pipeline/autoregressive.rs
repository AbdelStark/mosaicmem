use crate::backend::{BackendError, validate_backend_configuration};
use crate::camera::{CameraIntrinsics, CameraTrajectory};
use crate::diffusion::backbone::DiffusionBackbone;
use crate::diffusion::scheduler::NoiseScheduler;
use crate::diffusion::vae::VAE;
use crate::geometry::depth::DepthEstimator;
use crate::pipeline::config::PipelineConfig;
use crate::pipeline::inference::{InferencePipeline, PipelineError, PipelineStats};
use tracing::{debug, info, warn};

/// Callback for receiving generated frames.
pub type FrameCallback = Box<dyn FnMut(usize, &[f32], &[usize; 5]) + Send>;

/// Autoregressive pipeline: chains multiple generation windows for long-horizon video.
///
/// Per window:
/// 1. Get camera poses for this window
/// 2. Retrieve memory mosaic from store
/// 3. Run DiT denoising loop
/// 4. Decode via VAE
/// 5. Blend overlap frames with previous window
/// 6. Update memory with new keyframes
/// 7. Yield frames
pub struct AutoregressivePipeline {
    pub pipeline: InferencePipeline,
    pub config: PipelineConfig,
}

/// Linear blend of two frame buffers in the overlap region.
///
/// For the overlap region, each pixel is: `alpha * new + (1 - alpha) * prev`
/// where alpha ramps from 0 to 1 across the overlap frames.
fn blend_overlap(
    prev_frames: &[f32],
    prev_shape: &[usize; 5],
    new_frames: &[f32],
    new_shape: &[usize; 5],
    overlap_frames: usize,
) -> Vec<f32> {
    let [_b, c, t_new, h, w] = *new_shape;
    let [_, _, t_prev, _, _] = *prev_shape;
    let plane = h * w;

    if overlap_frames == 0 || t_prev == 0 {
        return new_frames.to_vec();
    }

    let overlap = overlap_frames.min(t_prev).min(t_new);
    let mut result = new_frames.to_vec();

    // The overlap region is the first `overlap` frames of new_frames,
    // corresponding to the last `overlap` frames of prev_frames.
    for oi in 0..overlap {
        // Alpha ramps from 0 (use prev) to 1 (use new) across overlap
        let alpha = (oi as f32 + 1.0) / (overlap as f32 + 1.0);

        let prev_frame_idx = t_prev - overlap + oi;
        for channel in 0..c {
            let prev_start = channel * t_prev * plane + prev_frame_idx * plane;
            let new_start = channel * t_new * plane + oi * plane;

            for pixel in 0..plane {
                let prev_idx = prev_start + pixel;
                let new_idx = new_start + pixel;
                if prev_idx < prev_frames.len() && new_idx < result.len() {
                    result[new_idx] =
                        alpha * new_frames[new_idx] + (1.0 - alpha) * prev_frames[prev_idx];
                }
            }
        }
    }

    result
}

impl AutoregressivePipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self::try_new(config).expect("invalid backend configuration for autoregressive pipeline")
    }

    pub fn try_new(config: PipelineConfig) -> Result<Self, BackendError> {
        validate_backend_configuration(config.backend_mode, config.checkpoint_path.as_deref())?;
        let pipeline = InferencePipeline::new(config.clone());
        Ok(Self { pipeline, config })
    }

    /// Generate a full video from a camera trajectory.
    ///
    /// Overlap frames between windows are linearly blended for smooth transitions.
    /// Returned frame buffers remain window-aligned, so overlap is still present
    /// in the concatenated output even after blending.
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
    /// All generated frames concatenated + per-window shapes
    #[allow(clippy::too_many_arguments)]
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
        let mut prev_frames: Option<Vec<f32>> = None;
        let mut prev_shape: Option<[usize; 5]> = None;

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
            let intrinsics =
                CameraIntrinsics::default_for_resolution(self.config.width, self.config.height);
            let (frames, shape) = self.pipeline.generate_window_with_intrinsics(
                window_poses,
                &intrinsics,
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

            // Blend overlap with previous window and emit only non-overlapping frames
            let blended = if let (Some(pf), Some(ps)) = (&prev_frames, &prev_shape) {
                let blended = blend_overlap(pf, ps, &frames, &shape, overlap);
                debug!("Window {}: blended {} overlap frames", window_idx, overlap);
                blended
            } else {
                frames.clone()
            };

            // Invoke callback with blended frames
            if let Some(ref mut cb) = callback {
                cb(window_idx, &blended, &shape);
            }

            all_frames.extend_from_slice(&blended);
            all_shapes.push(shape);

            prev_frames = Some(frames);
            prev_shape = Some(shape);

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
        let config = PipelineConfig {
            num_inference_steps: 2,
            width: 64,
            height: 64,
            window_size: 4,
            window_overlap: 1,
            ..Default::default()
        };

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

    #[test]
    fn test_blend_overlap_basic() {
        // prev: 4 frames, new: 4 frames, 2 overlap
        let shape_prev = [1, 1, 4, 2, 2]; // B=1, C=1, T=4, H=2, W=2
        let shape_new = [1, 1, 4, 2, 2];
        let frame_size = 4; // C*H*W = 1*2*2
        let prev = vec![1.0f32; 4 * frame_size]; // 4 frames of all 1.0
        let new = vec![0.0f32; 4 * frame_size]; // 4 frames of all 0.0

        let blended = blend_overlap(&prev, &shape_prev, &new, &shape_new, 2);

        // Frame 0 (first overlap): alpha = 1/3, so ~0.33*0 + 0.67*1 ≈ 0.67
        let frame0_val = blended[0];
        assert!(
            (frame0_val - 0.6667).abs() < 0.01,
            "First overlap frame should be ~0.67, got {}",
            frame0_val
        );

        // Frame 1 (second overlap): alpha = 2/3, so ~0.67*0 + 0.33*1 ≈ 0.33
        let frame1_val = blended[frame_size];
        assert!(
            (frame1_val - 0.3333).abs() < 0.01,
            "Second overlap frame should be ~0.33, got {}",
            frame1_val
        );

        // Frame 2 (no overlap): should be 0.0 (pure new)
        let frame2_val = blended[2 * frame_size];
        assert!(
            (frame2_val - 0.0).abs() < 1e-6,
            "Non-overlap frame should be unchanged: {}",
            frame2_val
        );
    }

    #[test]
    fn test_blend_overlap_zero() {
        let shape = [1, 1, 2, 2, 2];
        let prev = vec![1.0f32; 8];
        let new = vec![0.5f32; 8];

        let blended = blend_overlap(&prev, &shape, &new, &shape, 0);
        assert_eq!(blended, new, "Zero overlap should return new unchanged");
    }

    #[test]
    fn test_blend_overlap_identity() {
        // When prev and new are the same, blending should return the same values
        let shape = [1, 1, 4, 2, 2];
        let data = vec![0.5f32; 16];

        let blended = blend_overlap(&data, &shape, &data, &shape, 2);
        for (a, b) in blended.iter().zip(data.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Blending identical data should preserve values"
            );
        }
    }

    #[test]
    fn test_blend_overlap_preserves_channel_layout() {
        let shape = [1, 3, 2, 1, 1];
        let prev = vec![
            1.0, 2.0, // red frames
            3.0, 4.0, // green frames
            5.0, 6.0, // blue frames
        ];
        let new = vec![
            10.0, 20.0, // red frames
            30.0, 40.0, // green frames
            50.0, 60.0, // blue frames
        ];

        let blended = blend_overlap(&prev, &shape, &new, &shape, 1);
        let alpha = 0.5;
        assert_eq!(
            blended,
            vec![
                alpha * 10.0 + (1.0 - alpha) * 2.0,
                20.0,
                alpha * 30.0 + (1.0 - alpha) * 4.0,
                40.0,
                alpha * 50.0 + (1.0 - alpha) * 6.0,
                60.0,
            ]
        );
    }
}
