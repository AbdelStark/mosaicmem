use thiserror::Error;

#[derive(Error, Debug)]
pub enum DepthError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Invalid input dimensions: expected {expected}, got {got}")]
    InvalidDimensions { expected: String, got: String },
}

/// Trait for monocular depth estimation.
/// Implementations can use Tract/ONNX or other backends.
pub trait DepthEstimator: Send + Sync {
    /// Estimate depth from an RGB image.
    ///
    /// # Arguments
    /// * `image` - HxW RGB image as nested vectors
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// HxW depth map (metric depth in world units).
    fn estimate_depth(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<f32>>, DepthError>;
}

/// A stub depth estimator that generates synthetic depth maps for testing.
/// Produces a simple planar depth map at a configurable distance.
pub struct SyntheticDepthEstimator {
    pub base_depth: f32,
    pub noise_scale: f32,
}

impl SyntheticDepthEstimator {
    pub fn new(base_depth: f32, noise_scale: f32) -> Self {
        Self {
            base_depth,
            noise_scale,
        }
    }
}

impl DepthEstimator for SyntheticDepthEstimator {
    fn estimate_depth(
        &self,
        _image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<f32>>, DepthError> {
        let mut depth_map = Vec::with_capacity(height as usize);
        for v in 0..height {
            let mut row = Vec::with_capacity(width as usize);
            for u in 0..width {
                // Simple gradient depth: closer at center, farther at edges
                let cx = (u as f32 - width as f32 / 2.0) / width as f32;
                let cy = (v as f32 - height as f32 / 2.0) / height as f32;
                let dist_from_center = (cx * cx + cy * cy).sqrt();
                let depth = self.base_depth + dist_from_center * self.noise_scale;
                row.push(depth);
            }
            depth_map.push(row);
        }
        Ok(depth_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_depth() {
        let estimator = SyntheticDepthEstimator::new(5.0, 2.0);
        let depth = estimator.estimate_depth(&[0u8; 300], 10, 10).unwrap();
        assert_eq!(depth.len(), 10);
        assert_eq!(depth[0].len(), 10);
        // Center should be close to base_depth
        assert!((depth[5][5] - 5.0).abs() < 1.0);
    }
}
