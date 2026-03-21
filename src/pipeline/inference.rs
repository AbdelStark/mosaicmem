use crate::attention::memory_cross::MemoryCrossAttention;
use crate::camera::{CameraIntrinsics, CameraPose};
use crate::diffusion::backbone::{DiffusionBackbone, DiffusionCondition};
use crate::diffusion::scheduler::NoiseScheduler;
use crate::diffusion::vae::VAE;
use crate::geometry::depth::DepthEstimator;
use crate::geometry::fusion::StreamingFusion;
use crate::memory::retrieval::MemoryRetriever;
use crate::memory::store::{MemoryConfig, MosaicMemoryStore};
use crate::pipeline::config::PipelineConfig;
use thiserror::Error;

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
pub struct InferencePipeline {
    pub config: PipelineConfig,
    pub memory_store: MosaicMemoryStore,
    pub fusion: StreamingFusion,
    pub retriever: MemoryRetriever,
    pub cross_attention: MemoryCrossAttention,
}

impl InferencePipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let memory_config = MemoryConfig {
            max_patches: config.max_memory_patches,
            top_k: config.retrieval_top_k,
            patch_size: config.spatial_downsample as u32,
            latent_patch_size: 2,
            ..Default::default()
        };

        Self {
            fusion: StreamingFusion::new(config.voxel_size),
            memory_store: MosaicMemoryStore::new(memory_config),
            retriever: MemoryRetriever::new(),
            cross_attention: MemoryCrossAttention::new(config.hidden_dim, config.num_heads),
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

        // Retrieve memory for the first pose (representative for the window)
        let intrinsics =
            CameraIntrinsics::default_for_resolution(self.config.width, self.config.height);

        let mosaic = if !poses.is_empty() {
            self.retriever
                .retrieve(&self.memory_store, &poses[0], &intrinsics)
        } else {
            return Err(PipelineError::Config("No poses provided".to_string()));
        };

        // Compose memory tokens for backbone conditioning
        let (memory_tokens, _) = mosaic.compose_tokens();
        let has_memory = !memory_tokens.is_empty();

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
                memory_mask: None,
                camera_rotations: None,
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

    /// Update memory with newly generated frames.
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

        // Add keyframes to memory
        for (i, pose) in poses
            .iter()
            .enumerate()
            .step_by(self.config.keyframe_interval)
        {
            if i >= t {
                break;
            }

            // Estimate depth for this frame
            let frame_pixels = h * w * 3;
            let frame_start = i * frame_pixels;
            let frame_end = frame_start + frame_pixels;
            let frame_data: Vec<u8> = if frame_end <= frames.len() {
                frames[frame_start..frame_end]
                    .iter()
                    .map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8)
                    .collect()
            } else {
                vec![128u8; frame_pixels]
            };

            let depth_map = depth_estimator
                .estimate_depth(&frame_data, w as u32, h as u32)
                .map_err(|e| PipelineError::Depth(e.to_string()))?;

            // Get latent slice for this frame
            let lat_frame_size = lat_h * lat_w * lat_c;
            let lat_frame_idx = i / self.config.config_temporal_downsample();
            let lat_start = lat_frame_idx * lat_frame_size;
            let lat_end = (lat_start + lat_frame_size).min(latents.len());
            let lat_slice = if lat_start < latents.len() {
                &latents[lat_start..lat_end]
            } else {
                continue;
            };

            // Insert into memory store
            self.memory_store.insert_keyframe(
                i,
                pose.timestamp,
                lat_slice,
                lat_h,
                lat_w,
                lat_c,
                &depth_map,
                &intrinsics,
                pose,
            );

            // Also update the streaming fusion point cloud
            let _ = self.fusion.add_keyframe(
                &frame_data,
                w as u32,
                h as u32,
                &intrinsics,
                pose,
                depth_estimator,
            );
        }

        Ok(())
    }

    /// Initialize random noise for the latent.
    fn init_noise(&self, size: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
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

    #[test]
    fn test_inference_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = InferencePipeline::new(config);
        assert_eq!(pipeline.memory_store.num_patches(), 0);
    }

    #[test]
    fn test_generate_window() {
        let mut config = PipelineConfig::default();
        config.num_inference_steps = 2; // Fast for testing
        config.width = 64;
        config.height = 64;

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
    }
}
