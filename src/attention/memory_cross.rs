use crate::attention::prope::{PRoPE, PRoPEOperator, ProjectiveTransform};
use crate::attention::warped_latent::WarpGrid;
use crate::attention::warped_rope::WarpedRoPE;
use crate::backend::AblationConfig;
use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::mosaic::{LatentCanvas, MosaicFrame};
use crate::memory::store::RetrievedPatch;
use crate::tensor::{TensorLayout, TensorView};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Debug, Clone)]
pub struct MemoryFrameContext {
    pub mosaic: MosaicFrame,
    pub warp_grids: Vec<WarpGrid>,
    pub prope_transform: Option<ProjectiveTransform>,
    pub base_tokens: Vec<Vec<f32>>,
    pub warped_value_tokens: Vec<Vec<f32>>,
    pub base_positions: Vec<[f32; 3]>,
    pub warped_positions: Vec<[f32; 3]>,
}

impl MemoryFrameContext {
    pub fn has_memory(&self) -> bool {
        !self.base_tokens.is_empty()
    }

    fn value_tokens(&self, use_warped_latent: bool) -> &[Vec<f32>] {
        if use_warped_latent && !self.warped_value_tokens.is_empty() {
            &self.warped_value_tokens
        } else {
            &self.base_tokens
        }
    }

    fn key_positions(&self, use_warped_rope: bool) -> &[[f32; 3]] {
        if use_warped_rope && !self.warped_positions.is_empty() {
            &self.warped_positions
        } else {
            &self.base_positions
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryContext {
    pub frames: Vec<MemoryFrameContext>,
    pub rasterized_canvas: Option<LatentCanvas>,
    pub coverage_mask: Vec<bool>,
    pub ablation: AblationConfig,
}

impl MemoryContext {
    pub fn empty(ablation: AblationConfig) -> Self {
        Self {
            frames: Vec::new(),
            rasterized_canvas: None,
            coverage_mask: Vec::new(),
            ablation,
        }
    }

    pub fn has_memory(&self) -> bool {
        self.effective_memory_gate() > 0.0 && self.frames.iter().any(MemoryFrameContext::has_memory)
    }

    pub fn effective_memory_gate(&self) -> f32 {
        if !self.ablation.enable_memory {
            return 0.0;
        }
        self.ablation
            .memory_gate_override
            .unwrap_or(1.0)
            .clamp(0.0, 1.0)
    }

    pub fn canvas_cthw(&self) -> Option<Vec<f32>> {
        self.rasterized_canvas.as_ref().map(LatentCanvas::to_cthw)
    }

    pub fn active_tokens(&self) -> Vec<Vec<f32>> {
        let use_warped_latent = self.ablation.enable_warped_latent;
        self.frames
            .iter()
            .flat_map(|frame| frame.value_tokens(use_warped_latent).iter().cloned())
            .collect()
    }

    pub fn active_positions(&self) -> Vec<[f32; 3]> {
        let use_warped_rope = self.ablation.enable_warped_rope;
        self.frames
            .iter()
            .flat_map(|frame| frame.key_positions(use_warped_rope).iter().copied())
            .collect()
    }

    pub fn warp_valid_ratio(&self) -> f32 {
        let mut total = 0.0;
        let mut count = 0usize;
        for frame in &self.frames {
            for grid in &frame.warp_grids {
                total += grid.valid_ratio();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }
}

/// Memory Cross-Attention layer with multi-head attention, PRoPE, and WarpedRoPE.
pub struct MemoryCrossAttention {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub wq: Vec<Vec<f32>>,
    pub wk: Vec<Vec<f32>>,
    pub wv: Vec<Vec<f32>>,
    pub wo: Vec<Vec<f32>>,
    pub gate: Vec<f32>,
    pub warped_rope: WarpedRoPE,
}

type HeadBatch = Vec<Vec<Vec<f32>>>;

impl MemoryCrossAttention {
    pub fn new(hidden_dim: usize, num_heads: usize) -> Self {
        Self::new_seeded(hidden_dim, num_heads, 0)
    }

    pub fn new_seeded(hidden_dim: usize, num_heads: usize, seed: u64) -> Self {
        assert!(num_heads > 0, "num_heads must be greater than zero");
        assert!(
            hidden_dim.is_multiple_of(num_heads),
            "hidden_dim must be divisible by num_heads"
        );
        let head_dim = hidden_dim / num_heads;
        let dim_per_axis = (head_dim / 3 / 2 * 2).max(2);
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
            gate: vec![0.1; num_heads],
            warped_rope: WarpedRoPE::new(dim_per_axis, 64, 32),
        }
    }

    fn linear(input: &[f32], weights: &[Vec<f32>]) -> Vec<f32> {
        weights
            .iter()
            .map(|row| {
                row.iter()
                    .zip(input.iter())
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
            })
            .collect()
    }

    fn pad_to_dim(&self, token: &[f32]) -> Vec<f32> {
        let mut padded = vec![0.0f32; self.hidden_dim];
        let copy_len = token.len().min(self.hidden_dim);
        padded[..copy_len].copy_from_slice(&token[..copy_len]);
        padded
    }

    fn scaled_dot_product_single(
        query: &[f32],
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        mask: Option<&[bool]>,
    ) -> Vec<f32> {
        let dim = query.len() as f32;
        let scale = 1.0 / dim.sqrt();
        let mut scores: Vec<f32> = keys
            .iter()
            .enumerate()
            .map(|(idx, key)| {
                let dot = query
                    .iter()
                    .zip(key.iter())
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                if let Some(mask) = mask
                    && idx < mask.len()
                    && !mask[idx]
                {
                    f32::NEG_INFINITY
                } else {
                    dot * scale
                }
            })
            .collect();

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max_score.is_finite() {
            return vec![0.0; values.first().map(Vec::len).unwrap_or(0)];
        }

        let exp_sum: f32 = scores
            .iter_mut()
            .map(|score| {
                *score = (*score - max_score).exp();
                *score
            })
            .sum();

        if exp_sum <= 0.0 {
            return vec![0.0; values.first().map(Vec::len).unwrap_or(0)];
        }

        for score in &mut scores {
            *score /= exp_sum;
        }

        let mut output = vec![0.0f32; values.first().map(Vec::len).unwrap_or(0)];
        for (score, value) in scores.iter().zip(values.iter()) {
            for (out, component) in output.iter_mut().zip(value.iter()) {
                *out += score * component;
            }
        }
        output
    }

    fn split_heads(&self, projected: &[f32]) -> Vec<Vec<f32>> {
        (0..self.num_heads)
            .map(|head| {
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                projected[start..end].to_vec()
            })
            .collect()
    }

    fn concat_heads(&self, head_outputs: &[Vec<f32>]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.hidden_dim);
        for head_output in head_outputs {
            result.extend_from_slice(head_output);
        }
        result
    }

    fn compute_memory_positions(
        &self,
        patches: &[RetrievedPatch],
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<[f32; 3]> {
        self.warped_rope
            .compute_warped_positions(patches, target_pose, intrinsics)
    }

    fn apply_warped_rope_to_keys(&self, head_keys: &mut [Vec<Vec<f32>>], positions: &[[f32; 3]]) {
        let dim_per_axis = self.warped_rope.rope_u.dim;
        let rope_dim = 3 * dim_per_axis;

        for head_key in head_keys.iter_mut() {
            if head_key.is_empty() || head_key[0].len() < rope_dim {
                continue;
            }
            let rope_vectors: Vec<Vec<f32>> = head_key
                .iter()
                .map(|key| key[..rope_dim].to_vec())
                .collect();
            let rotated = self.warped_rope.rotate(&rope_vectors, positions);
            for (key, rotated_key) in head_key.iter_mut().zip(rotated.iter()) {
                key[..rope_dim].copy_from_slice(rotated_key);
            }
        }
    }

    fn project_keys_and_values(&self, tokens: &[Vec<f32>]) -> (HeadBatch, HeadBatch) {
        let projected_k: Vec<Vec<f32>> = tokens
            .iter()
            .map(|token| {
                let padded = self.pad_to_dim(token);
                Self::linear(&padded, &self.wk)
            })
            .collect();
        let projected_v: Vec<Vec<f32>> = tokens
            .iter()
            .map(|token| {
                let padded = self.pad_to_dim(token);
                Self::linear(&padded, &self.wv)
            })
            .collect();

        let key_heads = (0..self.num_heads)
            .map(|head| {
                projected_k
                    .iter()
                    .map(|key| {
                        let start = head * self.head_dim;
                        let end = start + self.head_dim;
                        key[start..end].to_vec()
                    })
                    .collect()
            })
            .collect();
        let value_heads = (0..self.num_heads)
            .map(|head| {
                projected_v
                    .iter()
                    .map(|value| {
                        let start = head * self.head_dim;
                        let end = start + self.head_dim;
                        value[start..end].to_vec()
                    })
                    .collect()
            })
            .collect();
        (key_heads, value_heads)
    }

    fn apply_prope_to_head(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        transform: &ProjectiveTransform,
    ) -> Option<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut query_tensor = TensorView::from_shape_vec(
            &[1, self.head_dim],
            query.to_vec(),
            TensorLayout::Flat(vec![1, self.head_dim]),
        )
        .ok()?;
        let key_values: Vec<f32> = keys.iter().flat_map(|key| key.iter().copied()).collect();
        let mut key_tensor = TensorView::from_shape_vec(
            &[keys.len(), self.head_dim],
            key_values,
            TensorLayout::Flat(vec![keys.len(), self.head_dim]),
        )
        .ok()?;

        let operator = PRoPE::new(self.head_dim, 1, 1);
        operator
            .apply_to_attention(&mut query_tensor, &mut key_tensor, transform)
            .ok()?;

        let rotated_query = query_tensor.data().as_slice_memory_order()?.to_vec();
        let rotated_keys_slice = key_tensor.data().as_slice_memory_order()?;
        let rotated_keys = rotated_keys_slice
            .chunks_exact(self.head_dim)
            .map(|chunk| chunk.to_vec())
            .collect();
        Some((rotated_query, rotated_keys))
    }

    pub fn forward(
        &self,
        generation_tokens: &[Vec<f32>],
        mosaic: &MosaicFrame,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<f32>> {
        let (base_tokens, raw_positions) = mosaic.compose_tokens();
        let base_positions: Vec<[f32; 3]> = raw_positions
            .iter()
            .map(|position| {
                [
                    position[0] / intrinsics.width as f32
                        * self.warped_rope.spatial_resolution as f32,
                    position[1] / intrinsics.height as f32
                        * self.warped_rope.spatial_resolution as f32,
                    0.0,
                ]
            })
            .collect();
        let warped_positions =
            self.compute_memory_positions(&mosaic.patches, target_pose, intrinsics);
        let frame = MemoryFrameContext {
            mosaic: mosaic.clone(),
            warp_grids: Vec::new(),
            prope_transform: None,
            base_tokens: base_tokens.clone(),
            warped_value_tokens: base_tokens,
            base_positions,
            warped_positions,
        };
        let context = MemoryContext {
            frames: vec![frame],
            rasterized_canvas: None,
            coverage_mask: mosaic
                .coverage_mask
                .iter()
                .flat_map(|row| row.iter().copied())
                .collect(),
            ablation: AblationConfig::default(),
        };
        self.forward_with_context(
            generation_tokens,
            &context,
            &vec![0; generation_tokens.len()],
        )
    }

    pub fn forward_with_context(
        &self,
        generation_tokens: &[Vec<f32>],
        context: &MemoryContext,
        query_frame_indices: &[usize],
    ) -> Vec<Vec<f32>> {
        let num_gen = generation_tokens.len();
        if num_gen == 0
            || query_frame_indices.len() != num_gen
            || !context.has_memory()
            || context.effective_memory_gate() <= 0.0
        {
            return vec![vec![0.0; self.hidden_dim]; num_gen];
        }

        let projected_q: Vec<Vec<f32>> = generation_tokens
            .iter()
            .map(|token| Self::linear(token, &self.wq))
            .collect();
        let q_heads: Vec<Vec<Vec<f32>>> = projected_q
            .iter()
            .map(|query| self.split_heads(query))
            .collect();

        let use_warped_latent = context.ablation.enable_warped_latent;
        let use_warped_rope = context.ablation.enable_warped_rope;
        let use_prope = context.ablation.enable_prope;
        let mut per_frame_cache = Vec::with_capacity(context.frames.len());

        for frame in &context.frames {
            let tokens = frame.value_tokens(use_warped_latent);
            if tokens.is_empty() {
                per_frame_cache.push(None);
                continue;
            }

            let (mut key_heads, value_heads) = self.project_keys_and_values(tokens);
            let positions = frame.key_positions(use_warped_rope);
            if positions.len() == tokens.len() {
                self.apply_warped_rope_to_keys(&mut key_heads, positions);
            }
            per_frame_cache.push(Some((key_heads, value_heads)));
        }

        let mut outputs = vec![vec![0.0f32; self.hidden_dim]; num_gen];
        for (token_idx, q_per_head) in q_heads.iter().enumerate() {
            let frame_idx =
                query_frame_indices[token_idx].min(context.frames.len().saturating_sub(1));
            let Some((frame_keys, frame_values)) = per_frame_cache
                .get(frame_idx)
                .and_then(|entry| entry.as_ref())
            else {
                continue;
            };
            let frame = &context.frames[frame_idx];

            let mut head_outputs = Vec::with_capacity(self.num_heads);
            for (head_idx, q_head) in q_per_head.iter().enumerate() {
                let mut q_work = q_head.clone();
                let mut k_work = frame_keys[head_idx].clone();
                if use_prope
                    && let Some(transform) = frame.prope_transform.as_ref()
                    && let Some((rotated_q, rotated_k)) =
                        self.apply_prope_to_head(&q_work, &k_work, transform)
                {
                    q_work = rotated_q;
                    k_work = rotated_k;
                }

                let attended = Self::scaled_dot_product_single(
                    &q_work,
                    &k_work,
                    &frame_values[head_idx],
                    None,
                );
                let gate = context
                    .ablation
                    .memory_gate_override
                    .unwrap_or(self.gate[head_idx])
                    .clamp(0.0, 1.0);
                let gated: Vec<f32> = attended.iter().map(|value| value * gate).collect();
                head_outputs.push(gated);
            }

            let concat = self.concat_heads(&head_outputs);
            outputs[token_idx] = Self::linear(&concat, &self.wo);
        }

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::CameraIntrinsics;
    use crate::memory::store::Patch3D;
    use nalgebra::{Point2, Point3};

    fn test_patch(id: u64, timestamp: f64, x: f32) -> RetrievedPatch {
        RetrievedPatch {
            patch: Patch3D {
                id,
                center: Point3::new(x, 0.0, 5.0),
                source_pose: CameraPose::identity(timestamp),
                source_frame: id as usize,
                source_timestamp: timestamp,
                source_depth: 5.0,
                source_rect: [10.0, 10.0, 16.0, 16.0],
                latent: vec![0.5f32; 2 * 2 * 8],
                latent_height: 2,
                latent_width: 2,
                token_coords: vec![(18.0 + x * 4.0, 18.0); 4],
                depth_tile: Some(vec![5.0; 4]),
                source_intrinsics: CameraIntrinsics::default(),
                normal_estimate: None,
                latent_shape: (8, 2, 2),
            },
            target_position: Point2::new(50.0 + x * 10.0, 50.0),
            projected_footprint: vec![Point2::new(50.0 + x * 10.0, 50.0); 4],
            target_depth: 5.0,
            visibility_score: 0.9,
        }
    }

    fn test_context(ablation: AblationConfig) -> MemoryContext {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let mosaic = MosaicFrame {
            target_pose: pose.clone(),
            patches: vec![test_patch(0, 0.0, 0.0), test_patch(1, 1.0, 1.0)],
            coverage_mask: vec![vec![true; 12]; 12],
            width: 100,
            height: 100,
        };
        let base_tokens = mosaic.compose_tokens().0;
        let base_positions = mosaic
            .compose_tokens()
            .1
            .iter()
            .map(|position| [position[0], position[1], 0.0])
            .collect();
        let mca = MemoryCrossAttention::new(32, 2);
        let warped_positions = mca.compute_memory_positions(&mosaic.patches, &pose, &intrinsics);
        let frame = MemoryFrameContext {
            mosaic,
            warp_grids: vec![WarpGrid {
                target_coords: vec![(0.0, 0.0); 4],
                valid_mask: vec![true; 4],
                source_shape: (2, 2),
            }],
            prope_transform: Some(
                PRoPE::new(16, 2, 1)
                    .compute_projective_transform(
                        &CameraPose::identity(0.0),
                        &pose,
                        &intrinsics,
                        &intrinsics,
                    )
                    .unwrap(),
            ),
            base_tokens: base_tokens.clone(),
            warped_value_tokens: base_tokens
                .iter()
                .map(|token| token.iter().map(|value| value * 1.25).collect())
                .collect(),
            base_positions,
            warped_positions,
        };

        MemoryContext {
            frames: vec![frame],
            rasterized_canvas: Some(LatentCanvas::empty(1, 2, 2, 8)),
            coverage_mask: vec![true; 4],
            ablation,
        }
    }

    #[test]
    fn test_memory_cross_attention_creation() {
        let mca = MemoryCrossAttention::new(64, 4);
        assert_eq!(mca.hidden_dim, 64);
        assert_eq!(mca.num_heads, 4);
        assert_eq!(mca.head_dim, 16);
        assert_eq!(mca.gate.len(), 4);
    }

    #[test]
    fn test_memory_cross_attention_empty_context() {
        let mca = MemoryCrossAttention::new(32, 2);
        let queries = vec![vec![1.0f32; 32]; 4];
        let output = mca.forward_with_context(
            &queries,
            &MemoryContext::empty(AblationConfig::default()),
            &[0; 4],
        );
        assert_eq!(output.len(), 4);
        assert!(
            output
                .iter()
                .all(|token| token.iter().all(|value| value.abs() < 1e-6))
        );
    }

    #[test]
    fn test_memory_cross_attention_with_context() {
        let mca = MemoryCrossAttention::new(32, 2);
        let context = test_context(AblationConfig::default());
        let queries = vec![vec![1.0f32; 32]; 4];
        let output = mca.forward_with_context(&queries, &context, &[0; 4]);
        assert_eq!(output.len(), 4);
        assert!(
            output
                .iter()
                .any(|token| token.iter().any(|value| value.abs() > 1e-10))
        );
    }

    #[test]
    fn test_gate_override_zero_disables_attention() {
        let mca = MemoryCrossAttention::new(32, 2);
        let context = test_context(AblationConfig {
            memory_gate_override: Some(0.0),
            ..AblationConfig::default()
        });
        let queries = vec![vec![1.0f32; 32]; 2];
        let output = mca.forward_with_context(&queries, &context, &[0; 2]);
        assert!(
            output
                .iter()
                .all(|token| token.iter().all(|value| value.abs() < 1e-6))
        );
    }
}
