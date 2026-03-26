use crate::attention::warped_rope::WarpedRoPE;
use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::mosaic::MosaicFrame;
use crate::memory::store::RetrievedPatch;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Memory Cross-Attention layer with multi-head attention and WarpedRoPE.
///
/// In each DiT block: self-attention → text cross-attention → memory cross-attention → MLP.
///
/// Q comes from generation tokens (with spatial grid positions).
/// K/V come from memory patch tokens (with WarpedRoPE-based geometric positions).
///
/// Multi-head: Q, K, V are split into `num_heads` heads of dimension `head_dim`,
/// attention is computed independently per head, then concatenated and projected.
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
    /// Learnable gate scalar per head (initialized to 0 for residual-friendly init).
    pub gate: Vec<f32>,
    /// Warped RoPE for memory key positions.
    pub warped_rope: WarpedRoPE,
}

impl MemoryCrossAttention {
    /// Create a new MemoryCrossAttention with Xavier initialization.
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        Self::new_seeded(hidden_dim, num_heads, 0)
    }

    /// Create a new MemoryCrossAttention with deterministic initialization.
    pub fn new_seeded(hidden_dim: usize, num_heads: usize, seed: u64) -> Self {
        assert!(num_heads > 0, "num_heads must be greater than zero");
        assert!(
            hidden_dim.is_multiple_of(num_heads),
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;
        let dim_per_axis = (head_dim / 3 / 2 * 2).max(2); // ensure even, split across u,v,t

        // Xavier-style initialization scale
        let scale = (1.0 / hidden_dim as f32).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);

        let mut init_matrix = |rows: usize, cols: usize| -> Vec<Vec<f32>> {
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
            // Gate initialized to small positive values (residual-friendly, allows gradual
            // contribution from memory cross-attention during early training)
            gate: vec![0.1; num_heads],
            warped_rope: WarpedRoPE::new(dim_per_axis, 64, 32),
        }
    }

    /// Linear projection: multiply input vector by weight matrix.
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

    /// Pad or truncate a token to match hidden_dim.
    fn pad_to_dim(&self, token: &[f32]) -> Vec<f32> {
        let mut padded = vec![0.0f32; self.hidden_dim];
        let copy_len = token.len().min(self.hidden_dim);
        padded[..copy_len].copy_from_slice(&token[..copy_len]);
        padded
    }

    /// Compute scaled dot-product attention for a single head.
    ///
    /// query: [head_dim], keys: [num_kv, head_dim], values: [num_kv, head_dim]
    /// Returns attended vector [head_dim].
    fn scaled_dot_product_single(
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
                let dot: f32 = query.iter().zip(k.iter()).map(|(q, kv)| q * kv).sum();
                let score = dot * scale;
                if let Some(m) = mask
                    && i < m.len()
                    && !m[i]
                {
                    return f32::NEG_INFINITY;
                }
                score
            })
            .collect();

        // Softmax with max-subtraction trick for numerical stability
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_score.is_finite() {
            let exp_sum: f32 = scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max_score).exp();
                    *s
                })
                .sum();
            if exp_sum > 0.0 {
                for s in &mut scores {
                    *s /= exp_sum;
                }
            }
        } else {
            // All masked — return zeros
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

    /// Split a projected vector [hidden_dim] into per-head chunks [num_heads][head_dim].
    fn split_heads(&self, projected: &[f32]) -> Vec<Vec<f32>> {
        (0..self.num_heads)
            .map(|h| {
                let start = h * self.head_dim;
                let end = start + self.head_dim;
                projected[start..end].to_vec()
            })
            .collect()
    }

    /// Concatenate per-head outputs [num_heads][head_dim] back to [hidden_dim].
    fn concat_heads(&self, head_outputs: &[Vec<f32>]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.hidden_dim);
        for head_out in head_outputs {
            result.extend_from_slice(head_out);
        }
        result
    }

    /// Compute warped positions for memory keys using the retrieved patches.
    fn compute_memory_positions(
        &self,
        patches: &[RetrievedPatch],
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<[usize; 3]> {
        self.warped_rope
            .compute_warped_positions(patches, target_pose, intrinsics)
    }

    /// Apply WarpedRoPE to a set of per-head key vectors.
    ///
    /// For each head, applies the u/v/t rotations from WarpedRoPE to the
    /// first 3*dim_per_axis dimensions of each key vector.
    fn apply_warped_rope_to_keys(&self, head_keys: &mut [Vec<Vec<f32>>], positions: &[[usize; 3]]) {
        let dim_per_axis = self.warped_rope.rope_u.dim;
        let rope_dim = 3 * dim_per_axis;

        for head_k in head_keys.iter_mut() {
            if head_k.is_empty() || head_k[0].len() < rope_dim {
                continue;
            }
            // Extract the RoPE-applicable portion, rotate, write back
            let rope_vecs: Vec<Vec<f32>> = head_k.iter().map(|k| k[..rope_dim].to_vec()).collect();
            let rotated = self.warped_rope.rotate(&rope_vecs, positions);
            for (k, rot) in head_k.iter_mut().zip(rotated.iter()) {
                k[..rope_dim].copy_from_slice(rot);
            }
        }
    }

    /// Forward pass: compute memory cross-attention with multi-head and WarpedRoPE.
    ///
    /// # Arguments
    /// * `generation_tokens` - [num_gen_tokens, hidden_dim] query tokens from generation
    /// * `mosaic` - Retrieved mosaic frame with memory patches
    /// * `target_pose` - Target camera pose (used for WarpedRoPE position computation)
    /// * `intrinsics` - Camera intrinsics (used for WarpedRoPE position computation)
    ///
    /// # Returns
    /// [num_gen_tokens, hidden_dim] — gated cross-attention output (additive residual).
    pub fn forward(
        &self,
        generation_tokens: &[Vec<f32>],
        mosaic: &MosaicFrame,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<f32>> {
        let num_gen = generation_tokens.len();

        // If no memory patches, return zeros (the gate will handle residual connection
        // at the caller level: output = x + gate * cross_attn(x, memory))
        if mosaic.patches.is_empty() {
            return vec![vec![0.0; self.hidden_dim]; num_gen];
        }

        // Compose memory tokens from mosaic
        let (memory_tokens, _memory_positions) = mosaic.compose_tokens();

        if memory_tokens.is_empty() {
            return vec![vec![0.0; self.hidden_dim]; num_gen];
        }

        // Project queries from generation tokens
        let projected_q: Vec<Vec<f32>> = generation_tokens
            .iter()
            .map(|t| Self::linear(t, &self.wq))
            .collect();

        // Project keys and values from memory tokens (pad to hidden_dim if needed)
        let projected_k: Vec<Vec<f32>> = memory_tokens
            .iter()
            .map(|t| {
                let padded = self.pad_to_dim(t);
                Self::linear(&padded, &self.wk)
            })
            .collect();

        let projected_v: Vec<Vec<f32>> = memory_tokens
            .iter()
            .map(|t| {
                let padded = self.pad_to_dim(t);
                Self::linear(&padded, &self.wv)
            })
            .collect();

        // Split into heads
        let q_heads: Vec<Vec<Vec<f32>>> = projected_q.iter().map(|q| self.split_heads(q)).collect(); // [num_gen][num_heads][head_dim]

        let mut k_heads: Vec<Vec<Vec<f32>>> = (0..self.num_heads)
            .map(|h| {
                projected_k
                    .iter()
                    .map(|k| {
                        let start = h * self.head_dim;
                        let end = start + self.head_dim;
                        k[start..end].to_vec()
                    })
                    .collect()
            })
            .collect(); // [num_heads][num_mem][head_dim]

        let v_heads: Vec<Vec<Vec<f32>>> = (0..self.num_heads)
            .map(|h| {
                projected_v
                    .iter()
                    .map(|v| {
                        let start = h * self.head_dim;
                        let end = start + self.head_dim;
                        v[start..end].to_vec()
                    })
                    .collect()
            })
            .collect(); // [num_heads][num_mem][head_dim]

        // Apply WarpedRoPE to memory keys — geometric positional encoding
        let warped_positions =
            self.compute_memory_positions(&mosaic.patches, target_pose, intrinsics);
        // We may have more tokens than patches (multiple tokens per patch),
        // so expand positions to match the token count.
        let expanded_positions =
            self.expand_positions_to_tokens(&mosaic.patches, &warped_positions);
        self.apply_warped_rope_to_keys(&mut k_heads, &expanded_positions);

        // Compute multi-head attention
        let mut outputs = vec![vec![0.0f32; self.hidden_dim]; num_gen];

        for (token_idx, q_per_head) in q_heads.iter().enumerate() {
            let mut head_outputs = Vec::with_capacity(self.num_heads);

            for (head_idx, q_head) in q_per_head.iter().enumerate() {
                let attended = Self::scaled_dot_product_single(
                    q_head,
                    &k_heads[head_idx],
                    &v_heads[head_idx],
                    None,
                );
                // Apply per-head gate
                let gated: Vec<f32> = attended.iter().map(|v| v * self.gate[head_idx]).collect();
                head_outputs.push(gated);
            }

            // Concatenate heads and project output
            let concat = self.concat_heads(&head_outputs);
            let projected = Self::linear(&concat, &self.wo);
            outputs[token_idx] = projected;
        }

        outputs
    }

    /// Expand per-patch positions to per-token positions.
    ///
    /// Each patch may produce multiple tokens (latent_height * latent_width).
    /// This replicates the patch's warped position for each of its tokens.
    fn expand_positions_to_tokens(
        &self,
        patches: &[RetrievedPatch],
        positions: &[[usize; 3]],
    ) -> Vec<[usize; 3]> {
        let mut expanded = Vec::new();
        for (patch, &pos) in patches.iter().zip(positions.iter()) {
            let num_tokens = if patch.patch.latent_height * patch.patch.latent_width > 0 {
                patch.patch.latent_height * patch.patch.latent_width
            } else {
                0
            };
            // Verify the latent data supports this many tokens
            let channels = if num_tokens > 0 {
                patch.patch.latent.len() / num_tokens
            } else {
                continue;
            };
            let valid_tokens = if channels > 0 {
                patch.patch.latent.len() / channels
            } else {
                0
            };
            for _ in 0..valid_tokens {
                expanded.push(pos);
            }
        }
        expanded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraIntrinsics;
    use crate::memory::store::{Patch3D, RetrievedPatch};
    use nalgebra::{Point2, Point3};

    #[test]
    fn test_memory_cross_attention_creation() {
        let mca = MemoryCrossAttention::new(64, 4);
        assert_eq!(mca.hidden_dim, 64);
        assert_eq!(mca.num_heads, 4);
        assert_eq!(mca.head_dim, 16);
        assert_eq!(mca.gate.len(), 4);
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
        // Empty mosaic should return zeros
        for token in &output {
            for &v in token {
                assert!((v - 0.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_memory_cross_attention_with_patches() {
        let mca = MemoryCrossAttention::new(32, 2);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        // Create a mosaic with actual patches
        let patch = Patch3D {
            id: 0,
            center: Point3::new(0.0, 0.0, 5.0),
            source_pose: CameraPose::identity(0.0),
            source_frame: 0,
            source_timestamp: 0.0,
            source_depth: 5.0,
            source_rect: [10.0, 10.0, 16.0, 16.0],
            latent: vec![0.5f32; 2 * 2 * 8], // 2x2 patch, 8 channels
            latent_height: 2,
            latent_width: 2,
            token_coords: vec![(18.0, 18.0); 4],
            depth_tile: Some(vec![5.0; 4]),
            source_intrinsics: CameraIntrinsics::default(),
            normal_estimate: None,
            latent_shape: (8, 2, 2),
        };

        let retrieved = RetrievedPatch {
            patch,
            target_position: Point2::new(50.0, 50.0),
            target_depth: 5.0,
            visibility_score: 0.9,
        };

        let mosaic = MosaicFrame {
            target_pose: pose.clone(),
            patches: vec![retrieved],
            coverage_mask: vec![vec![true; 12]; 12],
            width: 100,
            height: 100,
        };

        let gen_tokens = vec![vec![1.0f32; 32]; 4];
        let output = mca.forward(&gen_tokens, &mosaic, &pose, &intrinsics);
        assert_eq!(output.len(), 4);
        assert_eq!(output[0].len(), 32);

        // With patches, output should not be all zeros
        let has_nonzero = output.iter().any(|t| t.iter().any(|&v| v.abs() > 1e-10));
        assert!(
            has_nonzero,
            "Cross-attention output should be non-zero when patches exist"
        );
    }

    #[test]
    fn test_multi_head_split_concat() {
        let mca = MemoryCrossAttention::new(64, 4);
        let v = vec![1.0f32; 64];
        let heads = mca.split_heads(&v);
        assert_eq!(heads.len(), 4);
        assert_eq!(heads[0].len(), 16);
        let reconcat = mca.concat_heads(&heads);
        assert_eq!(reconcat.len(), 64);
        for (a, b) in v.iter().zip(reconcat.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gate_scaling() {
        let mut mca = MemoryCrossAttention::new(32, 2);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        let patch = Patch3D {
            id: 0,
            center: Point3::new(0.0, 0.0, 5.0),
            source_pose: CameraPose::identity(0.0),
            source_frame: 0,
            source_timestamp: 0.0,
            source_depth: 5.0,
            source_rect: [10.0, 10.0, 16.0, 16.0],
            latent: vec![0.5f32; 2 * 2 * 8],
            latent_height: 2,
            latent_width: 2,
            token_coords: vec![(18.0, 18.0); 4],
            depth_tile: Some(vec![5.0; 4]),
            source_intrinsics: CameraIntrinsics::default(),
            normal_estimate: None,
            latent_shape: (8, 2, 2),
        };
        let retrieved = RetrievedPatch {
            patch,
            target_position: Point2::new(50.0, 50.0),
            target_depth: 5.0,
            visibility_score: 0.9,
        };
        let mosaic = MosaicFrame {
            target_pose: pose.clone(),
            patches: vec![retrieved],
            coverage_mask: vec![vec![true; 12]; 12],
            width: 100,
            height: 100,
        };

        let gen_tokens = vec![vec![1.0f32; 32]; 2];

        // Get output with default gate (0.1)
        let output_default = mca.forward(&gen_tokens, &mosaic, &pose, &intrinsics);

        // Set gate to zero — output should be all zeros
        mca.gate = vec![0.0; 2];
        let output_zero_gate = mca.forward(&gen_tokens, &mosaic, &pose, &intrinsics);

        for token in &output_zero_gate {
            for &v in token {
                assert!(
                    v.abs() < 1e-6,
                    "With zero gate, output should be zero, got {}",
                    v
                );
            }
        }

        // Default gate output should differ from zero gate
        let default_has_nonzero = output_default
            .iter()
            .any(|t| t.iter().any(|&v| v.abs() > 1e-10));
        assert!(
            default_has_nonzero,
            "Default gate output should be non-zero"
        );
    }

    #[test]
    fn test_warped_rope_positions_in_attention() {
        let mca = MemoryCrossAttention::new(32, 2);

        // Create patches at different 3D positions
        let patches: Vec<RetrievedPatch> = (0..3)
            .map(|i| RetrievedPatch {
                patch: Patch3D {
                    id: i as u64,
                    center: Point3::new(i as f32, 0.0, 5.0),
                    source_pose: CameraPose::identity(0.0),
                    source_frame: 0,
                    source_timestamp: i as f64 * 0.5,
                    source_depth: 5.0 + i as f32,
                    source_rect: [10.0, 10.0, 16.0, 16.0],
                    latent: vec![0.5f32; 2 * 2 * 8],
                    latent_height: 2,
                    latent_width: 2,
                    token_coords: vec![(18.0, 18.0); 4],
                    depth_tile: Some(vec![5.0; 4]),
                    source_intrinsics: CameraIntrinsics::default(),
                    normal_estimate: None,
                    latent_shape: (8, 2, 2),
                },
                target_position: Point2::new(30.0 + i as f32 * 20.0, 50.0),
                target_depth: 5.0 + i as f32,
                visibility_score: 0.8,
            })
            .collect();

        let pose = CameraPose::identity(0.0);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);

        let positions = mca.compute_memory_positions(&patches, &pose, &intrinsics);
        assert_eq!(positions.len(), 3);

        // Patches at different spatial positions should get different warped positions
        assert_ne!(positions[0], positions[1]);
    }
}
