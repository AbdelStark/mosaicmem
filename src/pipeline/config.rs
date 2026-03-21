use serde::{Deserialize, Serialize};

/// Configuration for the MosaicMem inference pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of denoising steps per window.
    pub num_inference_steps: usize,
    /// Guidance scale for classifier-free guidance.
    pub guidance_scale: f32,
    /// Number of frames per generation window.
    pub window_size: usize,
    /// Number of overlap frames between windows (for smooth transitions).
    pub window_overlap: usize,
    /// Keyframe selection interval (every Kth frame).
    pub keyframe_interval: usize,
    /// Maximum total memory patches.
    pub max_memory_patches: usize,
    /// Top-K patches to retrieve per frame.
    pub retrieval_top_k: usize,
    /// Video resolution.
    pub width: u32,
    pub height: u32,
    /// Latent space dimensions.
    pub latent_channels: usize,
    pub spatial_downsample: usize,
    pub temporal_downsample: usize,
    /// Attention configuration.
    pub hidden_dim: usize,
    pub num_heads: usize,
    /// Voxel size for point cloud fusion.
    pub voxel_size: f32,
    /// Depth estimation base depth (for synthetic estimator).
    pub depth_base: f32,
    /// Random seed.
    pub seed: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 7.5,
            window_size: 16,
            window_overlap: 4,
            keyframe_interval: 4,
            max_memory_patches: 10000,
            retrieval_top_k: 64,
            width: 256,
            height: 256,
            latent_channels: 16,
            spatial_downsample: 8,
            temporal_downsample: 4,
            hidden_dim: 64,
            num_heads: 4,
            voxel_size: 0.05,
            depth_base: 5.0,
            seed: 42,
        }
    }
}

impl PipelineConfig {
    /// Compute latent dimensions from video dimensions.
    pub fn latent_height(&self) -> usize {
        self.height as usize / self.spatial_downsample
    }

    pub fn latent_width(&self) -> usize {
        self.width as usize / self.spatial_downsample
    }

    pub fn latent_frames(&self) -> usize {
        (self.window_size / self.temporal_downsample).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_dimensions() {
        let config = PipelineConfig::default();
        assert_eq!(config.latent_height(), 32); // 256 / 8
        assert_eq!(config.latent_width(), 32); // 256 / 8
        assert_eq!(config.latent_frames(), 4); // 16 / 4
    }

    #[test]
    fn test_custom_config_dimensions() {
        let config = PipelineConfig {
            width: 512,
            height: 256,
            spatial_downsample: 8,
            window_size: 32,
            temporal_downsample: 4,
            ..Default::default()
        };
        assert_eq!(config.latent_height(), 32); // 256 / 8
        assert_eq!(config.latent_width(), 64); // 512 / 8
        assert_eq!(config.latent_frames(), 8); // 32 / 4
    }

    #[test]
    fn test_latent_frames_min_one() {
        let config = PipelineConfig {
            window_size: 1,
            temporal_downsample: 4,
            ..Default::default()
        };
        // 1 / 4 = 0, clamped to 1
        assert_eq!(config.latent_frames(), 1);
    }

    #[test]
    fn test_config_serialization() {
        let config = PipelineConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.width, config.width);
        assert_eq!(deserialized.seed, config.seed);
    }
}
