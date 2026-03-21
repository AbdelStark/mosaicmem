/// Standard Rotary Position Embedding (RoPE) implementation.
///
/// RoPE encodes position information by rotating query/key vectors
/// using position-dependent rotation matrices.

/// Compute RoPE frequency tensor for a given dimension and max sequence length.
///
/// Returns a Vec of (cos, sin) pairs for each position and dimension pair.
pub struct RoPE {
    /// Embedding dimension (must be even).
    pub dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Base frequency (default: 10000.0).
    pub base: f32,
    /// Precomputed (cos, sin) table: [max_seq_len][dim/2][2].
    pub freqs: Vec<Vec<[f32; 2]>>,
}

impl RoPE {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(dim % 2 == 0, "RoPE dimension must be even");
        let half_dim = dim / 2;

        // Compute frequency bands
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();

        // Compute (cos, sin) for each position
        let mut freqs = Vec::with_capacity(max_seq_len);
        for pos in 0..max_seq_len {
            let mut pos_freqs = Vec::with_capacity(half_dim);
            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                pos_freqs.push([angle.cos(), angle.sin()]);
            }
            freqs.push(pos_freqs);
        }

        Self {
            dim,
            max_seq_len,
            base,
            freqs,
        }
    }

    /// Apply RoPE rotation to a vector at a given position.
    ///
    /// Input: `x` has shape [dim], position index `pos`.
    /// Rotates pairs (x[2i], x[2i+1]) by the position-dependent angle.
    pub fn rotate(&self, x: &[f32], pos: usize) -> Vec<f32> {
        assert_eq!(x.len(), self.dim);
        let pos = pos.min(self.max_seq_len - 1);
        let half_dim = self.dim / 2;
        let mut output = vec![0.0f32; self.dim];

        for i in 0..half_dim {
            let [cos_val, sin_val] = self.freqs[pos][i];
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            output[2 * i] = x0 * cos_val - x1 * sin_val;
            output[2 * i + 1] = x0 * sin_val + x1 * cos_val;
        }

        output
    }

    /// Apply RoPE to a batch of vectors.
    /// `vectors`: [num_tokens][dim], `positions`: [num_tokens].
    pub fn rotate_batch(&self, vectors: &[Vec<f32>], positions: &[usize]) -> Vec<Vec<f32>> {
        assert_eq!(vectors.len(), positions.len());
        vectors
            .iter()
            .zip(positions.iter())
            .map(|(v, &pos)| self.rotate(v, pos))
            .collect()
    }
}

/// Compute 2D RoPE positions for a spatial grid.
/// Returns flattened position indices for a grid of (height, width).
pub fn grid_positions_2d(height: usize, width: usize) -> Vec<[usize; 2]> {
    let mut positions = Vec::with_capacity(height * width);
    for y in 0..height {
        for x in 0..width {
            positions.push([y, x]);
        }
    }
    positions
}

/// Compute 3D RoPE positions for a spatiotemporal grid.
/// Returns flattened position indices for a grid of (time, height, width).
pub fn grid_positions_3d(time: usize, height: usize, width: usize) -> Vec<[usize; 3]> {
    let mut positions = Vec::with_capacity(time * height * width);
    for t in 0..time {
        for y in 0..height {
            for x in 0..width {
                positions.push([t, y, x]);
            }
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_basic() {
        let rope = RoPE::new(8, 100, 10000.0);
        let x = vec![1.0; 8];
        let rotated = rope.rotate(&x, 0);
        // At position 0, cos=1, sin=0, so output should equal input
        for (a, b) in x.iter().zip(rotated.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_different_positions() {
        let rope = RoPE::new(8, 100, 10000.0);
        let x = vec![1.0; 8];
        let r0 = rope.rotate(&x, 0);
        let r1 = rope.rotate(&x, 1);
        // Different positions should produce different outputs
        assert!(r0 != r1);
    }

    #[test]
    fn test_grid_positions() {
        let pos = grid_positions_2d(3, 4);
        assert_eq!(pos.len(), 12);
        assert_eq!(pos[0], [0, 0]);
        assert_eq!(pos[11], [2, 3]);
    }
}
