use crate::attention::memory_cross::MemoryCrossAttention;
use crate::attention::prope::PRoPE;
use crate::attention::warped_latent::warp_patch_latent;
use crate::camera::{CameraIntrinsics, CameraPose, CameraTrajectory};
use crate::diffusion::backbone::{DiffusionBackbone, DiffusionCondition};
use crate::diffusion::scheduler::NoiseScheduler;
use crate::diffusion::vae::VAE;
use crate::geometry::depth::DepthEstimator;
use crate::geometry::fusion::StreamingFusion;
use crate::memory::retrieval::MemoryRetriever;
use crate::memory::store::{MemoryConfig, MosaicMemoryStore};
use crate::pipeline::config::PipelineConfig;
use rand::{Rng, SeedableRng, rngs::StdRng};
use thiserror::Error;
use tracing::debug;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Backbone error: {0}")]
    Backbone(String),
    #[error("VAE error: {0}")]
    VAE(String),
    #[error("Depth error: {0}")]
    Depth(String),
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Single-window inference pipeline.
/// Generates one window of frames given context and memory.
///
/// Integrates all MosaicMem components:
/// - PRoPE camera conditioning for projective positional encoding
/// - Warped latent feature-space alignment for retrieved patches
/// - Temporal decay for recency-weighted retrieval
/// - Diversity-aware patch selection for spatial coverage
/// - Coverage-guided conditioning for inpainting guidance
/// - Adaptive keyframe selection based on camera motion
pub struct InferencePipeline {
    pub config: PipelineConfig,
    pub memory_store: MosaicMemoryStore,
    pub fusion: StreamingFusion,
    pub retriever: MemoryRetriever,
    pub cross_attention: MemoryCrossAttention,
    /// PRoPE for camera-dependent positional encoding.
    pub prope: PRoPE,
    rng: StdRng,
}

/// Extract a single planar `[C, H, W]` frame from a flattened `[B, C, T, H, W]` tensor.
pub fn extract_frame_planar(
    data: &[f32],
    shape: &[usize; 5],
    frame_idx: usize,
) -> Option<Vec<f32>> {
    let [_batch, channels, frames, height, width] = *shape;
    if channels == 0 || frames == 0 || height == 0 || width == 0 || frame_idx >= frames {
        return None;
    }

    let plane = height * width;
    let channel_stride = frames * plane;
    let mut frame = vec![0.0f32; channels * plane];

    for channel in 0..channels {
        let src_start = channel * channel_stride + frame_idx * plane;
        let src_end = src_start + plane;
        let dst_start = channel * plane;
        let dst_end = dst_start + plane;
        frame[dst_start..dst_end].copy_from_slice(data.get(src_start..src_end)?);
    }

    Some(frame)
}

/// Convert a planar `[C, H, W]` frame into interleaved `RGBRGB...` bytes.
pub fn planar_frame_to_rgb8_interleaved(
    frame: &[f32],
    channels: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let plane = width.saturating_mul(height);
    let mut bytes = vec![0u8; plane.saturating_mul(3)];

    if plane == 0 || channels == 0 {
        return bytes;
    }

    for pixel in 0..plane {
        for channel in 0..3 {
            let src_channel = channel.min(channels - 1);
            let src_idx = src_channel * plane + pixel;
            let value = frame.get(src_idx).copied().unwrap_or(0.0);
            bytes[pixel * 3 + channel] = (value.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }

    bytes
}

impl InferencePipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let memory_config = MemoryConfig {
            max_patches: config.max_memory_patches,
            top_k: config.retrieval_top_k,
            patch_size: config.spatial_downsample as u32,
            latent_patch_size: 2,
            temporal_decay_half_life: config.temporal_decay_half_life,
            ..Default::default()
        };

        // Use diversity-aware retriever when configured
        let retriever = if config.diversity_radius > 0.0 {
            MemoryRetriever::with_diversity(config.diversity_radius, config.diversity_penalty)
        } else {
            MemoryRetriever::new()
        };

        let head_dim = config.hidden_dim / config.num_heads;

        Self {
            fusion: StreamingFusion::new(config.voxel_size),
            memory_store: MosaicMemoryStore::new(memory_config),
            retriever,
            cross_attention: MemoryCrossAttention::new_seeded(
                config.hidden_dim,
                config.num_heads,
                config.seed ^ 0xC0A5_5A1C,
            ),
            prope: PRoPE::new(head_dim, config.num_heads, config.temporal_compression),
            rng: StdRng::seed_from_u64(config.seed),
            config,
        }
    }

    /// Generate a single window of video frames.
    ///
    /// # Arguments
    /// * `poses` - Camera poses for this window
    /// * `text_embedding` - Text conditioning
    /// * `backbone` - Diffusion backbone for denoising
    /// * `scheduler` - Noise scheduler
    /// * `vae` - VAE for encode/decode
    ///
    /// # Returns
    /// Generated frames as flattened pixel data + shape [B, C, T, H, W]
    pub fn generate_window(
        &mut self,
        poses: &[CameraPose],
        text_embedding: &[Vec<f32>],
        backbone: &dyn DiffusionBackbone,
        scheduler: &dyn NoiseScheduler,
        vae: &dyn VAE,
    ) -> Result<(Vec<f32>, [usize; 5]), PipelineError> {
        let lat_h = self.config.latent_height();
        let lat_w = self.config.latent_width();
        let lat_t = self.config.latent_frames();
        let lat_c = self.config.latent_channels;
        let shape = [1, lat_c, lat_t, lat_h, lat_w];
        let latent_size = lat_c * lat_t * lat_h * lat_w;

        // Initialize with random noise
        let mut latent = self.init_noise(latent_size);

        if poses.is_empty() {
            return Err(PipelineError::Config("No poses provided".to_string()));
        }

        let intrinsics =
            CameraIntrinsics::default_for_resolution(self.config.width, self.config.height);

        // Retrieve memory with temporal decay applied
        let query_time = Some(poses[0].timestamp);
        let mut mosaic =
            self.retriever
                .retrieve_at_time(&self.memory_store, &poses[0], &intrinsics, query_time);

        // Apply warped latent feature-space alignment to retrieved patches
        if self.config.enable_warped_latent && !mosaic.patches.is_empty() {
            self.apply_warped_latent(&mut mosaic, &poses[0], &intrinsics);
        }

        let memory_canvas = if mosaic.patches.is_empty() {
            None
        } else {
            Some(mosaic.compose_latent_canvas(lat_h, lat_w, lat_c))
        };

        // Compose patch tokens plus a rasterized latent canvas for conditioning.
        let memory_tokens = self.compose_memory_tokens(&mosaic, memory_canvas.as_ref());
        let has_memory = !memory_tokens.is_empty();
        let memory_latent = memory_canvas.as_ref().map(|canvas| canvas.to_cthw(lat_t));

        // Compute PRoPE camera rotations for this window's poses
        let camera_rotations = if !poses.is_empty() {
            Some(self.prope.compute_rotations(poses, &intrinsics))
        } else {
            None
        };

        // Flatten coverage mask for backbone conditioning
        let coverage_mask = if has_memory {
            Some(self.flatten_coverage_mask(&mosaic.coverage_mask))
        } else {
            None
        };

        debug!(
            "Window generation: {} patches, coverage={:.1}%, PRoPE frames={}, warped_latent={}",
            mosaic.patches.len(),
            mosaic.coverage_ratio() * 100.0,
            poses.len(),
            self.config.enable_warped_latent,
        );

        // Denoising loop
        let timesteps = scheduler.inference_timesteps(self.config.num_inference_steps);
        for &t in &timesteps {
            let timestep_f = t as f32 / scheduler.num_timesteps() as f32;

            let condition = DiffusionCondition {
                text_embedding: text_embedding.to_vec(),
                memory_tokens: if has_memory {
                    Some(memory_tokens.clone())
                } else {
                    None
                },
                memory_latent: memory_latent.clone(),
                memory_mask: coverage_mask.clone(),
                camera_rotations: camera_rotations.clone(),
                timestep: timestep_f,
            };

            let predicted_noise = backbone
                .denoise_step(&latent, &shape, &condition)
                .map_err(|e| PipelineError::Backbone(e.to_string()))?;

            // Apply memory cross-attention to modulate the denoised latent.
            // The latent is reshaped into spatial tokens, cross-attended with
            // memory patches, then reshaped back.
            let denoised = scheduler.step(&predicted_noise, &latent, t);

            if has_memory {
                // Reshape denoised latent [C*T*H*W] into tokens [T*H*W, C]
                let num_spatial = lat_t * lat_h * lat_w;
                let mut tokens: Vec<Vec<f32>> = Vec::with_capacity(num_spatial);
                for i in 0..num_spatial {
                    let mut token = Vec::with_capacity(lat_c);
                    for c in 0..lat_c {
                        let idx = c * num_spatial + i;
                        if idx < denoised.len() {
                            token.push(denoised[idx]);
                        }
                    }
                    // Pad to hidden_dim if needed
                    token.resize(self.config.hidden_dim, 0.0);
                    tokens.push(token);
                }

                // Run memory cross-attention
                let attn_output =
                    self.cross_attention
                        .forward(&tokens, &mosaic, &poses[0], &intrinsics);

                // Add residual: latent = denoised + cross_attention_output
                // Only add back the latent channels (first lat_c dims of each token)
                latent = denoised.clone();
                for (i, attn_token) in attn_output.iter().enumerate().take(num_spatial) {
                    for c in 0..lat_c {
                        let idx = c * num_spatial + i;
                        if idx < latent.len() && c < attn_token.len() {
                            latent[idx] += attn_token[c];
                        }
                    }
                }
            } else {
                latent = denoised;
            }
        }

        // Decode latent to pixel space
        let lat_shape = [1, lat_c, lat_t, lat_h, lat_w];
        let (frames, frame_shape) = vae
            .decode(&latent, &lat_shape)
            .map_err(|e| PipelineError::VAE(e.to_string()))?;

        Ok((frames, frame_shape))
    }

    /// Apply warped latent feature-space alignment to retrieved patches.
    ///
    /// For each patch, computes a homography from its source camera view to the
    /// target view, then warps the latent features using bilinear interpolation.
    /// This aligns memory patches geometrically before attention.
    fn apply_warped_latent(
        &self,
        mosaic: &mut crate::memory::mosaic::MosaicFrame,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) {
        for patch in &mut mosaic.patches {
            let source_pose = patch.patch.source_pose.clone();
            let channels = if patch.patch.latent_height * patch.patch.latent_width > 0 {
                patch.patch.latent.len() / (patch.patch.latent_height * patch.patch.latent_width)
            } else {
                continue;
            };

            if channels == 0 || patch.patch.latent_height < 2 || patch.patch.latent_width < 2 {
                continue;
            }

            let warped = warp_patch_latent(
                &patch.patch.latent,
                patch.patch.latent_height,
                patch.patch.latent_width,
                channels,
                &source_pose,
                target_pose,
                intrinsics,
                patch.patch.source_depth,
            );

            // Only use warped latent if it has non-zero content (valid warp)
            let has_content = warped.iter().any(|&v| v.abs() > 1e-10);
            if has_content {
                patch.patch.latent = warped;
            }
        }
    }

    /// Flatten the 2D coverage mask into a 1D boolean vector for backbone conditioning.
    fn flatten_coverage_mask(&self, coverage: &[Vec<bool>]) -> Vec<bool> {
        coverage
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Update memory with newly generated frames.
    ///
    /// Uses adaptive keyframe selection when configured: selects frames based on
    /// camera motion magnitude rather than fixed interval sampling.
    pub fn update_memory(
        &mut self,
        frames: &[f32],
        frame_shape: &[usize; 5],
        poses: &[CameraPose],
        depth_estimator: &dyn DepthEstimator,
        vae: &dyn VAE,
    ) -> Result<(), PipelineError> {
        let [_b, _c, t, h, w] = *frame_shape;
        let intrinsics = CameraIntrinsics::default_for_resolution(w as u32, h as u32);

        // Encode frames to latent space
        let (latents, lat_shape) = vae
            .encode(frames, frame_shape)
            .map_err(|e| PipelineError::VAE(e.to_string()))?;

        let lat_h = lat_shape[3];
        let lat_w = lat_shape[4];
        let lat_c = lat_shape[1];

        // Select keyframes: adaptive (motion-based) or fixed interval
        let keyframe_indices = self.select_keyframes(poses, t);

        for i in keyframe_indices {
            if i >= t || i >= poses.len() {
                continue;
            }
            let pose = &poses[i];

            // Estimate depth for this frame
            let frame_data = extract_frame_planar(frames, frame_shape, i)
                .map(|frame| planar_frame_to_rgb8_interleaved(&frame, 3, w, h))
                .unwrap_or_else(|| vec![128u8; h * w * 3]);

            let depth_map = depth_estimator
                .estimate_depth(&frame_data, w as u32, h as u32)
                .map_err(|e| PipelineError::Depth(e.to_string()))?;

            let lat_frame_idx = i / self.config.config_temporal_downsample();
            let Some(lat_slice) = self.extract_latent_frame(&latents, &lat_shape, lat_frame_idx)
            else {
                continue;
            };

            // Insert into memory store
            self.memory_store.insert_keyframe(
                i,
                pose.timestamp,
                &lat_slice,
                lat_h,
                lat_w,
                lat_c,
                &depth_map,
                &intrinsics,
                pose,
            );

            // Also update the streaming fusion point cloud
            self.fusion
                .add_keyframe(
                    &frame_data,
                    w as u32,
                    h as u32,
                    &intrinsics,
                    pose,
                    depth_estimator,
                )
                .map_err(|e| PipelineError::Depth(e.to_string()))?;
        }

        Ok(())
    }

    /// Select keyframe indices based on configuration.
    ///
    /// When `adaptive_keyframes` is enabled, uses camera motion magnitude
    /// (translation + rotation) to select frames with significant viewpoint change.
    /// Falls back to fixed interval when disabled or when motion is below threshold.
    fn select_keyframes(&self, poses: &[CameraPose], max_frames: usize) -> Vec<usize> {
        let num_poses = poses.len().min(max_frames);
        if num_poses == 0 {
            return vec![];
        }

        if self.config.adaptive_keyframes && num_poses > 1 {
            let traj = CameraTrajectory::new(poses[..num_poses].to_vec());
            let mut keyframes = traj.select_keyframes(
                self.config.keyframe_translation_threshold,
                self.config.keyframe_angular_threshold,
            );

            // Ensure minimum keyframe density: at least every keyframe_interval frames
            let max_gap = self.config.keyframe_interval;
            let mut filled = Vec::new();
            let mut last_kf = 0;
            for &kf in &keyframes {
                // Fill gaps larger than max_gap with uniform samples
                while last_kf + max_gap < kf {
                    last_kf += max_gap;
                    filled.push(last_kf);
                }
                filled.push(kf);
                last_kf = kf;
            }
            keyframes = filled;
            keyframes.sort_unstable();
            keyframes.dedup();

            debug!(
                "Adaptive keyframe selection: {} keyframes from {} poses",
                keyframes.len(),
                num_poses
            );
            keyframes
        } else {
            (0..num_poses)
                .step_by(self.config.keyframe_interval.max(1))
                .collect()
        }
    }

    /// Initialize random noise for the latent.
    fn init_noise(&mut self, size: usize) -> Vec<f32> {
        (0..size)
            .map(|_| self.rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect()
    }

    /// Extract one latent frame from a flattened [B, C, T, H, W] tensor.
    fn extract_latent_frame(
        &self,
        latents: &[f32],
        shape: &[usize; 5],
        frame_idx: usize,
    ) -> Option<Vec<f32>> {
        let [_b, channels, latent_frames, height, width] = *shape;
        if channels == 0 || latent_frames == 0 || height == 0 || width == 0 {
            return None;
        }
        if frame_idx >= latent_frames {
            return None;
        }

        let mut frame = vec![0.0f32; height * width * channels];
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    let src_idx = (((c * latent_frames + frame_idx) * height + y) * width) + x;
                    let dst_idx = ((y * width + x) * channels) + c;
                    let &value = latents.get(src_idx)?;
                    frame[dst_idx] = value;
                }
            }
        }

        Some(frame)
    }

    /// Combine patch tokens with a rasterized latent canvas for backbone conditioning.
    fn compose_memory_tokens(
        &self,
        mosaic: &crate::memory::mosaic::MosaicFrame,
        memory_canvas: Option<&crate::memory::mosaic::LatentCanvas>,
    ) -> Vec<Vec<f32>> {
        let (mut tokens, _) = mosaic.compose_tokens();
        if let Some(canvas) = memory_canvas {
            tokens.extend(canvas.to_tokens());
        }
        tokens
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            num_patches: self.memory_store.num_patches(),
            num_points: self.fusion.num_points(),
            num_keyframes: self.fusion.num_keyframes,
            total_tokens: self.memory_store.total_tokens(),
        }
    }
}

impl PipelineConfig {
    fn config_temporal_downsample(&self) -> usize {
        self.temporal_downsample
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub num_patches: usize,
    pub num_points: usize,
    pub num_keyframes: usize,
    pub total_tokens: usize,
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Patches: {}, Points: {}, Keyframes: {}, Tokens: {}",
            self.num_patches, self.num_points, self.num_keyframes, self.total_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diffusion::backbone::SyntheticBackbone;
    use crate::diffusion::scheduler::DDPMScheduler;
    use crate::diffusion::vae::SyntheticVAE;
    use crate::geometry::depth::SyntheticDepthEstimator;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_inference_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = InferencePipeline::new(config);
        assert_eq!(pipeline.memory_store.num_patches(), 0);
    }

    #[test]
    fn test_generate_window() {
        let config = PipelineConfig {
            num_inference_steps: 2,
            width: 64,
            height: 64,
            ..Default::default()
        };

        let mut pipeline = InferencePipeline::new(config);
        let backbone = SyntheticBackbone::new(0.1);
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        let vae = SyntheticVAE::new(8, 4, 16);

        let poses = vec![CameraPose::identity(0.0)];
        let text_emb = vec![vec![0.0f32; 64]; 10];

        let result = pipeline.generate_window(&poses, &text_emb, &backbone, &scheduler, &vae);
        assert!(result.is_ok());

        let (frames, shape) = result.unwrap();
        assert_eq!(shape[0], 1); // batch
        assert_eq!(shape[1], 3); // RGB
        assert!(!frames.is_empty());
    }

    #[test]
    fn test_pipeline_uses_diversity_retriever() {
        let config = PipelineConfig {
            diversity_radius: 15.0,
            diversity_penalty: 0.3,
            ..Default::default()
        };

        let pipeline = InferencePipeline::new(config);
        assert_eq!(pipeline.retriever.diversity_radius, 15.0);
        assert_eq!(pipeline.retriever.diversity_penalty, 0.3);
    }

    #[test]
    fn test_pipeline_uses_temporal_decay() {
        let config = PipelineConfig {
            temporal_decay_half_life: 2.0,
            ..Default::default()
        };

        let pipeline = InferencePipeline::new(config);
        assert!(
            (pipeline.memory_store.config.temporal_decay_half_life - 2.0).abs() < 1e-10,
            "Memory store should have temporal decay configured"
        );
    }

    #[test]
    fn test_pipeline_has_prope() {
        let config = PipelineConfig::default();
        let pipeline = InferencePipeline::new(config.clone());
        assert_eq!(pipeline.prope.num_heads, config.num_heads);
        let expected_head_dim = config.hidden_dim / config.num_heads;
        assert_eq!(pipeline.prope.head_dim, expected_head_dim);
    }

    #[test]
    fn test_adaptive_keyframe_selection() {
        let config = PipelineConfig {
            adaptive_keyframes: true,
            keyframe_translation_threshold: 0.3,
            keyframe_angular_threshold: 0.2,
            keyframe_interval: 4,
            ..Default::default()
        };

        let pipeline = InferencePipeline::new(config);

        // Create poses with varying motion
        let poses: Vec<CameraPose> = (0..8)
            .map(|i| {
                CameraPose::from_translation_rotation(
                    i as f64 * 0.1,
                    Vector3::new(i as f32 * 0.2, 0.0, 0.0),
                    UnitQuaternion::identity(),
                )
            })
            .collect();

        let kf = pipeline.select_keyframes(&poses, 8);
        // Should always include frame 0
        assert!(kf.contains(&0), "Keyframes should include frame 0");
        // Should have selected at least one more keyframe
        assert!(
            kf.len() >= 2,
            "Should have at least 2 keyframes, got {}",
            kf.len()
        );
    }

    #[test]
    fn test_fixed_keyframe_selection() {
        let config = PipelineConfig {
            adaptive_keyframes: false,
            keyframe_interval: 3,
            ..Default::default()
        };

        let pipeline = InferencePipeline::new(config);
        let poses: Vec<CameraPose> = (0..9)
            .map(|i| CameraPose::identity(i as f64 * 0.1))
            .collect();

        let kf = pipeline.select_keyframes(&poses, 9);
        assert_eq!(kf, vec![0, 3, 6]);
    }

    #[test]
    fn test_generate_with_memory_uses_prope_and_coverage() {
        let config = PipelineConfig {
            num_inference_steps: 2,
            width: 64,
            height: 64,
            temporal_decay_half_life: 5.0,
            diversity_radius: 10.0,
            ..Default::default()
        };

        let mut pipeline = InferencePipeline::new(config);

        // Populate memory with a keyframe
        let depth = SyntheticDepthEstimator::new(5.0, 1.0);
        let vae = SyntheticVAE::new(8, 4, 16);
        let intrinsics = CameraIntrinsics::default_for_resolution(64, 64);
        let pose0 = CameraPose::identity(0.0);
        let depth_map = depth
            .estimate_depth(&vec![128u8; 64 * 64 * 3], 64, 64)
            .unwrap();
        let latents = vec![0.5f32; 8 * 8 * 16];
        pipeline.memory_store.insert_keyframe(
            0,
            0.0,
            &latents,
            8,
            8,
            16,
            &depth_map,
            &intrinsics,
            &pose0,
        );

        // Generate with memory present
        let backbone = SyntheticBackbone::new(0.1);
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        let poses = vec![CameraPose::from_translation_rotation(
            1.0,
            Vector3::new(0.5, 0.0, 0.0),
            UnitQuaternion::identity(),
        )];
        let text_emb = vec![vec![0.0f32; 64]; 10];

        let result = pipeline.generate_window(&poses, &text_emb, &backbone, &scheduler, &vae);
        assert!(result.is_ok(), "Generation with memory should succeed");
    }

    #[test]
    fn test_flatten_coverage_mask() {
        let config = PipelineConfig::default();
        let pipeline = InferencePipeline::new(config);
        let mask = vec![vec![true, false, true], vec![false, true, false]];
        let flat = pipeline.flatten_coverage_mask(&mask);
        assert_eq!(flat, vec![true, false, true, false, true, false]);
    }

    #[test]
    fn test_extract_latent_frame_reorders_bc_thw_to_hwc() {
        let pipeline = InferencePipeline::new(PipelineConfig::default());
        let shape = [1, 2, 3, 2, 2];
        let latents: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|v| v as f32)
            .collect();

        let frame = pipeline.extract_latent_frame(&latents, &shape, 1).unwrap();
        assert_eq!(frame.len(), 2 * 2 * 2);
        assert_eq!(frame, vec![4.0, 16.0, 5.0, 17.0, 6.0, 18.0, 7.0, 19.0]);
    }

    #[test]
    fn test_extract_frame_planar_reorders_cthw() {
        let shape = [1, 3, 2, 2, 2];
        let frames: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|value| value as f32)
            .collect();

        let frame = extract_frame_planar(&frames, &shape, 1).unwrap();
        assert_eq!(
            frame,
            vec![
                4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0
            ]
        );
    }

    #[test]
    fn test_planar_frame_to_rgb8_interleaved() {
        let frame = vec![
            1.0, 0.0, 0.0, 1.0, // red plane
            0.0, 1.0, 0.0, 1.0, // green plane
            0.0, 0.0, 1.0, 1.0, // blue plane
        ];

        let bytes = planar_frame_to_rgb8_interleaved(&frame, 3, 2, 2);
        assert_eq!(bytes, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,]);
    }

    #[test]
    fn test_compose_memory_tokens_includes_canvas() {
        use crate::camera::CameraPose;
        use crate::memory::mosaic::MosaicFrame;
        use crate::memory::store::{Patch3D, RetrievedPatch};
        use nalgebra::{Point2, Point3};

        let pipeline = InferencePipeline::new(PipelineConfig::default());
        let mosaic = MosaicFrame {
            target_pose: CameraPose::identity(0.0),
            patches: vec![RetrievedPatch {
                patch: Patch3D {
                    id: 0,
                    center: Point3::new(0.0, 0.0, 5.0),
                    source_pose: CameraPose::identity(0.0),
                    source_frame: 0,
                    source_timestamp: 0.0,
                    source_depth: 5.0,
                    source_rect: [0.0, 0.0, 16.0, 16.0],
                    latent: vec![1.0; 2 * 2 * 4],
                    latent_height: 2,
                    latent_width: 2,
                },
                target_position: Point2::new(32.0, 32.0),
                target_depth: 5.0,
                visibility_score: 1.0,
            }],
            coverage_mask: vec![vec![true; 4]; 4],
            width: 64,
            height: 64,
        };

        let canvas = mosaic.compose_latent_canvas(4, 4, 4);
        let tokens = pipeline.compose_memory_tokens(&mosaic, Some(&canvas));
        assert!(tokens.len() > mosaic.compose_tokens().0.len());
        assert_eq!(tokens[0].len(), 4);
    }
}
