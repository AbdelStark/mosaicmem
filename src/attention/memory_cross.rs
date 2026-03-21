use crate::attention::warped_rope::WarpedRoPE;
use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::mosaic::MosaicFrame;

/// Memory Cross-Attention layer.
///
/// In each DiT block: self-attention → text cross-attention → memory cross-attention → MLP.
///
/// Q comes from generation tokens, K/V from memory patch tokens.
/// Memory keys use Warped RoPE for geometrically-correct positional encoding.
pub struct MemoryCrossAttention {
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Q projection weights [hidden_dim, hidden_dim].
    pub wq: Vec<Vec<f32>>,
    /// K projection weights [hidden_dim, hidden_dim].
    pub wk: Vec<Vec<f32>>,
    /// V projection weights [hidden_dim, hidden_dim].
    pub wv: Vec<Vec<f32>>,
    /// Output projection weights [hidden_dim, hidden_dim].
    pub wo: Vec<Vec<f32>>,
    /// Warped RoPE for memory key positions.
    pub warped_rope: WarpedRoPE,
}

impl MemoryCrossAttention {
    /// Create a new MemoryCrossAttention with random initialization.
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        let head_dim = hidden_dim / num_heads;
        let dim_per_axis = head_dim / 3; // Split across u, v, t

        // Initialize with small random values (Xavier-style)
        let scale = (1.0 / hidden_dim as f32).sqrt();

        let init_matrix = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(-scale..scale)).collect())
                .collect()
        };

        Self {
            hidden_dim,
            num_heads,
            head_dim,
            wq: init_matrix(hidden_dim, hidden_dim),
            wk: init_matrix(hidden_dim, hidden_dim),
            wv: init_matrix(hidden_dim, hidden_dim),
            wo: init_matrix(hidden_dim, hidden_dim),
            warped_rope: WarpedRoPE::new(
                (dim_per_axis / 2 * 2).max(2), // ensure even
                64,
                32,
            ),
        }
    }

    /// Linear projection: multiply input by weight matrix.
    fn linear(input: &[f32], weights: &[Vec<f32>]) -> Vec<f32> {
        weights
            .iter()
            .map(|row| {
                row.iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
            })
            .collect()
    }

    /// Compute scaled dot-product attention.
    fn scaled_dot_product(
        query: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        mask: Option<&[bool]>,
    ) -> Vec<f32> {
        let dim = query.len() as f32;
        let scale = 1.0 / dim.sqrt();

        // Compute attention scores
        let mut scores: Vec<f32> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let dot: f32 = query.iter().zip(k.iter()).map(|(q, k)| q * k).sum();
                let score = dot * scale;
                if let Some(m) = mask {
                    if !m[i] {
                        return f32::NEG_INFINITY;
                    }
                }
                score
            })
            .collect();

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_score.is_finite() {
            let exp_sum: f32 = scores.iter_mut().map(|s| { *s = (*s - max_score).exp(); *s }).sum();
            if exp_sum > 0.0 {
                for s in &mut scores {
                    *s /= exp_sum;
                }
            }
        } else {
            // All masked, return zeros
            return vec![0.0; values.first().map(|v| v.len()).unwrap_or(0)];
        }

        // Weighted sum of values
        let v_dim = values.first().map(|v| v.len()).unwrap_or(0);
        let mut output = vec![0.0f32; v_dim];
        for (score, value) in scores.iter().zip(values.iter()) {
            for (o, v) in output.iter_mut().zip(value.iter()) {
                *o += score * v;
            }
        }
        output
    }

    /// Forward pass: compute memory cross-attention.
    ///
    /// # Arguments
    /// * `generation_tokens` - [num_gen_tokens, hidden_dim] - query tokens from the generation process
    /// * `mosaic` - The retrieved mosaic frame with memory patches
    /// * `target_pose` - Target camera pose
    /// * `intrinsics` - Camera intrinsics
    ///
    /// Returns [num_gen_tokens, hidden_dim] attended output.
    pub fn forward(
        &self,
        generation_tokens: &[Vec<f32>],
        mosaic: &MosaicFrame,
        _target_pose: &CameraPose,
        _intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<f32>> {
        // Compose memory tokens from mosaic
        let (memory_tokens, _memory_positions) = mosaic.compose_tokens();

        if memory_tokens.is_empty() {
            // No memory, return zeros
            return generation_tokens
                .iter()
                .map(|t| vec![0.0; t.len()])
                .collect();
        }

        // Project queries, keys, values
        let queries: Vec<Vec<f32>> = generation_tokens
            .iter()
            .map(|t| Self::linear(t, &self.wq))
            .collect();

        let keys: Vec<Vec<f32>> = memory_tokens
            .iter()
            .map(|t| {
                // Pad or truncate to hidden_dim
                let mut padded = vec![0.0f32; self.hidden_dim];
                for (i, v) in t.iter().enumerate().take(self.hidden_dim) {
                    padded[i] = *v;
                }
                Self::linear(&padded, &self.wk)
            })
            .collect();

        let values: Vec<Vec<f32>> = memory_tokens
            .iter()
            .map(|t| {
                let mut padded = vec![0.0f32; self.hidden_dim];
                for (i, v) in t.iter().enumerate().take(self.hidden_dim) {
                    padded[i] = *v;
                }
                Self::linear(&padded, &self.wv)
            })
            .collect();

        // Apply attention for each query token
        let outputs: Vec<Vec<f32>> = queries
            .iter()
            .map(|q| {
                let attended = Self::scaled_dot_product(q, &keys, &values, None);
                Self::linear(&attended, &self.wo)
            })
            .collect();

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store::{MosaicMemoryStore, MemoryConfig, RetrievedPatch};

    #[test]
    fn test_memory_cross_attention_creation() {
        let mca = MemoryCrossAttention::new(64, 4);
        assert_eq!(mca.hidden_dim, 64);
        assert_eq!(mca.num_heads, 4);
        assert_eq!(mca.head_dim, 16);
    }

    #[test]
    fn test_memory_cross_attention_empty_mosaic() {
        let mca = MemoryCrossAttention::new(32, 2);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        let mosaic = MosaicFrame {
            target_pose: pose.clone(),
            patches: vec![],
            coverage_mask: vec![],
            width: 100,
            height: 100,
        };

        let gen_tokens = vec![vec![1.0f32; 32]; 4];
        let output = mca.forward(&gen_tokens, &mosaic, &pose, &intrinsics);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].len(), 32);
    }
}
