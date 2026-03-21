use thiserror::Error;

#[derive(Error, Debug)]
pub enum VAEError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Encoding failed: {0}")]
    EncodeFailed(String),
    #[error("Decoding failed: {0}")]
    DecodeFailed(String),
}

/// Trait for VAE (Variational Autoencoder) encode/decode.
/// Implementations wrap Tract ONNX inference or other backends.
pub trait VAE: Send + Sync {
    /// Encode video frames to latent space.
    ///
    /// # Arguments
    /// * `frames` - Raw pixel data [B, C, T, H, W] flattened
    /// * `shape` - [B, C, T, H, W]
    ///
    /// # Returns
    /// Latent representation [B, C_lat, T_lat, H_lat, W_lat] flattened + shape
    fn encode(
        &self,
        frames: &[f32],
        shape: &[usize; 5],
    ) -> Result<(Vec<f32>, [usize; 5]), VAEError>;

    /// Decode latent representation back to video frames.
    ///
    /// # Arguments
    /// * `latent` - Latent tensor [B, C_lat, T_lat, H_lat, W_lat] flattened
    /// * `shape` - [B, C_lat, T_lat, H_lat, W_lat]
    ///
    /// # Returns
    /// Decoded frames [B, C, T, H, W] flattened + shape
    fn decode(
        &self,
        latent: &[f32],
        shape: &[usize; 5],
    ) -> Result<(Vec<f32>, [usize; 5]), VAEError>;

    /// Get the spatial downsampling factor.
    fn spatial_downsample(&self) -> usize;

    /// Get the temporal downsampling factor.
    fn temporal_downsample(&self) -> usize;

    /// Get the number of latent channels.
    fn latent_channels(&self) -> usize;
}

/// A synthetic VAE for testing that simply downsamples/upsamples.
pub struct SyntheticVAE {
    pub spatial_factor: usize,
    pub temporal_factor: usize,
    pub latent_channels: usize,
}

impl SyntheticVAE {
    pub fn new(spatial_factor: usize, temporal_factor: usize, latent_channels: usize) -> Self {
        Self {
            spatial_factor,
            temporal_factor,
            latent_channels,
        }
    }
}

impl VAE for SyntheticVAE {
    fn encode(
        &self,
        _frames: &[f32],
        shape: &[usize; 5],
    ) -> Result<(Vec<f32>, [usize; 5]), VAEError> {
        let [b, _c, t, h, w] = *shape;
        let lat_t = (t / self.temporal_factor).max(1);
        let lat_h = h / self.spatial_factor;
        let lat_w = w / self.spatial_factor;
        let lat_shape = [b, self.latent_channels, lat_t, lat_h, lat_w];
        let lat_size = b * self.latent_channels * lat_t * lat_h * lat_w;

        // Simple average pooling simulation
        let latent = vec![0.5f32; lat_size];
        Ok((latent, lat_shape))
    }

    fn decode(
        &self,
        _latent: &[f32],
        shape: &[usize; 5],
    ) -> Result<(Vec<f32>, [usize; 5]), VAEError> {
        let [b, _c_lat, t_lat, h_lat, w_lat] = *shape;
        let t = t_lat * self.temporal_factor;
        let h = h_lat * self.spatial_factor;
        let w = w_lat * self.spatial_factor;
        let out_shape = [b, 3, t, h, w]; // RGB output
        let out_size = b * 3 * t * h * w;

        // Simple nearest-neighbor upsample simulation
        let frames = vec![0.5f32; out_size];
        Ok((frames, out_shape))
    }

    fn spatial_downsample(&self) -> usize {
        self.spatial_factor
    }

    fn temporal_downsample(&self) -> usize {
        self.temporal_factor
    }

    fn latent_channels(&self) -> usize {
        self.latent_channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_vae_encode_decode() {
        let vae = SyntheticVAE::new(8, 4, 16);
        let shape = [1, 3, 16, 256, 256];
        let frames = vec![0.5f32; 1 * 3 * 16 * 256 * 256];

        let (latent, lat_shape) = vae.encode(&frames, &shape).unwrap();
        assert_eq!(lat_shape, [1, 16, 4, 32, 32]);

        let (decoded, dec_shape) = vae.decode(&latent, &lat_shape).unwrap();
        assert_eq!(dec_shape, [1, 3, 16, 256, 256]);
    }
}
