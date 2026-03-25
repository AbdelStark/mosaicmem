use thiserror::Error;

#[derive(Error, Debug)]
pub enum VAEError {
    #[error("Model not loaded: {0}")]
    ModelNotLoaded(String),
    #[error("Encoding failed: {0}")]
    EncodeFailed(String),
    #[error("Decoding failed: {0}")]
    DecodeFailed(String),
    #[error("Invalid dimensions: expected {expected}, got {got}")]
    InvalidDimensions { expected: String, got: String },
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
        frames: &[f32],
        shape: &[usize; 5],
    ) -> Result<(Vec<f32>, [usize; 5]), VAEError> {
        let [b, c, t, h, w] = *shape;
        let temporal_factor = self.temporal_factor.max(1);
        let spatial_factor = self.spatial_factor.max(1);
        if self.latent_channels == 0 || c == 0 || t == 0 || h == 0 || w == 0 {
            return Err(VAEError::EncodeFailed(
                "shape dimensions must be non-zero".to_string(),
            ));
        }
        if frames.len() != b * c * t * h * w {
            return Err(VAEError::InvalidDimensions {
                expected: format!("{} elements", b * c * t * h * w),
                got: format!("{} elements", frames.len()),
            });
        }

        let lat_t = (t / temporal_factor).max(1);
        let lat_h = (h / spatial_factor).max(1);
        let lat_w = (w / spatial_factor).max(1);
        let lat_shape = [b, self.latent_channels, lat_t, lat_h, lat_w];

        let mut latent = vec![0.0f32; b * self.latent_channels * lat_t * lat_h * lat_w];

        for batch in 0..b {
            for lt in 0..lat_t {
                let t_start = (lt * temporal_factor).min(t.saturating_sub(1));
                let t_end = ((lt + 1) * temporal_factor).min(t).max(t_start + 1);
                let t_count = (t_end - t_start) as f32;

                for ly in 0..lat_h {
                    let y_start = ly * spatial_factor;
                    let y_end = ((ly + 1) * spatial_factor).min(h).max(y_start + 1);
                    let y_count = (y_end - y_start) as f32;

                    for lx in 0..lat_w {
                        let x_start = lx * spatial_factor;
                        let x_end = ((lx + 1) * spatial_factor).min(w).max(x_start + 1);
                        let x_count = (x_end - x_start) as f32;
                        let sample_count = t_count * y_count * x_count;

                        let mut pooled = vec![0.0f32; c];
                        for src_t in t_start..t_end {
                            for src_y in y_start..y_end {
                                for src_x in x_start..x_end {
                                    let base = (((batch * c) * t + src_t) * h + src_y) * w + src_x;
                                    for ch in 0..c {
                                        pooled[ch] += frames[base + ch * t * h * w];
                                    }
                                }
                            }
                        }

                        for ch in &mut pooled {
                            *ch /= sample_count;
                        }

                        let luminance = if c == 0 {
                            0.0
                        } else {
                            pooled.iter().sum::<f32>() / c as f32
                        };
                        let spatial_phase = (ly as f32 / lat_h as f32) + (lx as f32 / lat_w as f32);
                        let temporal_phase = lt as f32 / lat_t as f32;

                        let out_base = (((batch * self.latent_channels) * lat_t + lt) * lat_h + ly)
                            * lat_w
                            + lx;
                        for out_ch in 0..self.latent_channels {
                            let src_ch = out_ch % c;
                            let channel_weight = if out_ch < c {
                                1.0
                            } else {
                                0.75 + 0.15 * ((out_ch / c) as f32 % 3.0)
                            };
                            let phase = ((out_ch + 1) as f32 * 0.37
                                + spatial_phase * 1.91
                                + temporal_phase * 2.27)
                                .sin()
                                * 0.05;
                            let value = if out_ch < c {
                                pooled[src_ch]
                            } else {
                                luminance * channel_weight + phase
                            };
                            latent[out_base + out_ch * lat_t * lat_h * lat_w] = value;
                        }
                    }
                }
            }
        }
        Ok((latent, lat_shape))
    }

    fn decode(
        &self,
        latent: &[f32],
        shape: &[usize; 5],
    ) -> Result<(Vec<f32>, [usize; 5]), VAEError> {
        let [b, c_lat, t_lat, h_lat, w_lat] = *shape;
        let temporal_factor = self.temporal_factor.max(1);
        let spatial_factor = self.spatial_factor.max(1);
        if c_lat == 0 || t_lat == 0 || h_lat == 0 || w_lat == 0 {
            return Err(VAEError::DecodeFailed(
                "latent dimensions must be non-zero".to_string(),
            ));
        }
        if latent.len() != b * c_lat * t_lat * h_lat * w_lat {
            return Err(VAEError::InvalidDimensions {
                expected: format!("{} elements", b * c_lat * t_lat * h_lat * w_lat),
                got: format!("{} elements", latent.len()),
            });
        }

        let t = t_lat * temporal_factor;
        let h = h_lat * spatial_factor;
        let w = w_lat * spatial_factor;
        let out_shape = [b, 3, t, h, w]; // RGB output
        let out_size = b * 3 * t * h * w;
        let mut frames = vec![0.0f32; out_size];

        for batch in 0..b {
            for out_t in 0..t {
                let lt = out_t / temporal_factor;
                let lt_next = (lt + 1).min(t_lat.saturating_sub(1));
                let t_alpha = if temporal_factor > 1 {
                    (out_t % temporal_factor) as f32 / temporal_factor as f32
                } else {
                    0.0
                };

                for out_y in 0..h {
                    let ly = out_y / spatial_factor;
                    let ly_next = (ly + 1).min(h_lat.saturating_sub(1));
                    let y_alpha = if spatial_factor > 1 {
                        (out_y % spatial_factor) as f32 / spatial_factor as f32
                    } else {
                        0.0
                    };

                    for out_x in 0..w {
                        let lx = out_x / spatial_factor;
                        let lx_next = (lx + 1).min(w_lat.saturating_sub(1));
                        let x_alpha = if spatial_factor > 1 {
                            (out_x % spatial_factor) as f32 / spatial_factor as f32
                        } else {
                            0.0
                        };

                        let mut sample = vec![0.0f32; c_lat];
                        for (corner_t, wt) in [(lt, 1.0 - t_alpha), (lt_next, t_alpha)] {
                            for (corner_y, wy) in [(ly, 1.0 - y_alpha), (ly_next, y_alpha)] {
                                for (corner_x, wx) in [(lx, 1.0 - x_alpha), (lx_next, x_alpha)] {
                                    let weight = wt * wy * wx;
                                    if weight == 0.0 {
                                        continue;
                                    }
                                    let latent_base =
                                        (((batch * c_lat) * t_lat + corner_t) * h_lat + corner_y)
                                            * w_lat
                                            + corner_x;
                                    for ch in 0..c_lat {
                                        sample[ch] += latent
                                            [latent_base + ch * t_lat * h_lat * w_lat]
                                            * weight;
                                    }
                                }
                            }
                        }

                        let aux_mean = if c_lat > 3 {
                            sample[3..].iter().sum::<f32>() / (c_lat - 3) as f32
                        } else {
                            0.0
                        };
                        let channel_samples = [
                            if c_lat > 0 { sample[0] } else { 0.0 },
                            if c_lat > 1 { sample[1] } else { sample[0] },
                            if c_lat > 2 { sample[2] } else { sample[0] },
                        ];

                        let out_base = (((batch * 3) * t + out_t) * h + out_y) * w + out_x;
                        for ch in 0..3 {
                            let mut value = if c_lat >= 3 {
                                0.8 * channel_samples[ch] + 0.2 * aux_mean
                            } else {
                                channel_samples[ch % c_lat]
                            };
                            if c_lat > 0 {
                                value += 0.05 * sample[ch % c_lat];
                            }
                            frames[out_base + ch * t * h * w] = value.clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }

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

    fn build_patterned_frames(width: usize, height: usize, timesteps: usize) -> Vec<f32> {
        let mut frames = vec![0.0f32; 3 * timesteps * height * width];
        for t in 0..timesteps {
            for y in 0..height {
                for x in 0..width {
                    let idx = t * height * width + y * width + x;
                    let bright = if x >= width / 2 { 1.0 } else { 0.0 };
                    frames[idx] = bright;
                    frames[idx + timesteps * height * width] = y as f32 / height.max(1) as f32;
                    frames[idx + 2 * timesteps * height * width] =
                        t as f32 / timesteps.max(1) as f32;
                }
            }
        }
        frames
    }

    fn latent_index(
        timesteps: usize,
        height: usize,
        width: usize,
        ch: usize,
        t: usize,
        y: usize,
        x: usize,
    ) -> usize {
        ch * timesteps * height * width + t * height * width + y * width + x
    }

    #[test]
    fn test_synthetic_vae_encode_decode() {
        let vae = SyntheticVAE::new(8, 4, 16);
        let shape = [1, 3, 16, 256, 256];
        let frames = vec![0.5f32; 3 * 16 * 256 * 256];

        let (latent, lat_shape) = vae.encode(&frames, &shape).unwrap();
        assert_eq!(lat_shape, [1, 16, 4, 32, 32]);

        let (_decoded, dec_shape) = vae.decode(&latent, &lat_shape).unwrap();
        assert_eq!(dec_shape, [1, 3, 16, 256, 256]);
    }

    #[test]
    fn test_encode_preserves_spatial_structure() {
        let vae = SyntheticVAE::new(2, 2, 8);
        let shape = [1, 3, 4, 4, 4];
        let frames = build_patterned_frames(4, 4, 4);

        let (latent, lat_shape) = vae.encode(&frames, &shape).unwrap();
        assert_eq!(lat_shape, [1, 8, 2, 2, 2]);

        let left = latent[latent_index(lat_shape[2], lat_shape[3], lat_shape[4], 0, 0, 0, 0)];
        let right = latent[latent_index(lat_shape[2], lat_shape[3], lat_shape[4], 0, 0, 0, 1)];
        assert!(
            right > left,
            "right half should stay brighter after pooling"
        );
    }

    #[test]
    fn test_decode_uses_latent_content() {
        let vae = SyntheticVAE::new(2, 2, 6);
        let lat_shape = [1, 6, 2, 2, 2];
        let mut latent = vec![0.0f32; 6 * 2 * 2 * 2];
        latent[latent_index(lat_shape[2], lat_shape[3], lat_shape[4], 0, 0, 0, 0)] = 1.0;
        latent[latent_index(lat_shape[2], lat_shape[3], lat_shape[4], 1, 0, 0, 0)] = 0.2;
        latent[latent_index(lat_shape[2], lat_shape[3], lat_shape[4], 2, 0, 0, 0)] = 0.1;

        let (decoded, out_shape) = vae.decode(&latent, &lat_shape).unwrap();
        assert_eq!(out_shape, [1, 3, 4, 4, 4]);
        assert!(
            decoded[0] > decoded[decoded.len() / 3] && decoded[0] > decoded[2 * decoded.len() / 3],
            "red channel should dominate the first voxel"
        );
    }

    #[test]
    fn test_roundtrip_retains_block_pattern() {
        let vae = SyntheticVAE::new(2, 2, 9);
        let shape = [1, 3, 4, 4, 4];
        let frames = build_patterned_frames(4, 4, 4);

        let (latent, lat_shape) = vae.encode(&frames, &shape).unwrap();
        let (decoded, decoded_shape) = vae.decode(&latent, &lat_shape).unwrap();
        assert_eq!(decoded_shape, shape);

        let mut left_sum = 0.0f32;
        let mut right_sum = 0.0f32;
        let mut left_count = 0usize;
        let mut right_count = 0usize;
        let plane = decoded_shape[3] * decoded_shape[4];
        let stride = decoded_shape[2] * plane;
        for ch in 0..decoded_shape[1] {
            for t in 0..decoded_shape[2] {
                for y in 0..decoded_shape[3] {
                    for x in 0..decoded_shape[4] {
                        let idx = ch * stride + t * plane + y * decoded_shape[4] + x;
                        if x < decoded_shape[4] / 2 {
                            left_sum += decoded[idx];
                            left_count += 1;
                        } else {
                            right_sum += decoded[idx];
                            right_count += 1;
                        }
                    }
                }
            }
        }
        let left_mean = left_sum / left_count as f32;
        let right_mean = right_sum / right_count as f32;
        assert!(right_mean > left_mean);
    }
}
