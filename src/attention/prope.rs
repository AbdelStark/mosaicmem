use crate::camera::{CameraIntrinsics, CameraPose};
use crate::tensor::{TensorError, TensorView};
use nalgebra::{Matrix3, SMatrix};
use thiserror::Error;

pub type ProjectionMatrix = SMatrix<f64, 3, 4>;
pub type ProjectionPseudoInverse = SMatrix<f64, 4, 3>;

#[derive(Debug, Clone)]
pub struct ProjectiveTransform {
    pub source_projection: ProjectionMatrix,
    pub target_projection: ProjectionMatrix,
    pub relative: Matrix3<f64>,
}

impl ProjectiveTransform {
    pub fn projection_matrix(pose: &CameraPose, intrinsics: &CameraIntrinsics) -> ProjectionMatrix {
        let mut extrinsic = ProjectionMatrix::zeros();
        let rotation = pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .matrix()
            .map(f64::from);
        let translation = pose.world_to_camera.translation.vector.map(f64::from);

        extrinsic.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
        extrinsic.column_mut(3).copy_from(&translation);

        intrinsics.matrix() * extrinsic
    }

    pub fn from_cameras(
        source_pose: &CameraPose,
        target_pose: &CameraPose,
        source_intrinsics: &CameraIntrinsics,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<Self, PRoPEError> {
        let source_projection = Self::projection_matrix(source_pose, source_intrinsics);
        let target_projection = Self::projection_matrix(target_pose, target_intrinsics);
        let source_pinv = pseudo_inverse(&source_projection)?;
        let relative = target_projection * source_pinv;

        Ok(Self {
            source_projection,
            target_projection,
            relative,
        })
    }

    pub fn to_rope_params(&self, half_dim: usize) -> Vec<[f32; 2]> {
        const PARAM_ORDER: [(usize, usize); 9] = [
            (0, 0),
            (0, 2),
            (2, 0),
            (1, 1),
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 2),
        ];

        let mut params = Vec::with_capacity(half_dim);
        for dim in 0..half_dim {
            let (row, col) = PARAM_ORDER[dim % PARAM_ORDER.len()];
            let baseline = if row == col { 1.0 } else { 0.0 };
            let scale = 1.0 + (dim / PARAM_ORDER.len()) as f64 * 0.25;
            let angle = ((self.relative[(row, col)] - baseline) * scale).atan();
            params.push([angle.cos() as f32, angle.sin() as f32]);
        }
        params
    }
}

#[derive(Debug, Error)]
pub enum PRoPEError {
    #[error("projection matrix is rank deficient and cannot be pseudo-inverted")]
    SingularProjection,
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("attention tensor shape mismatch: expected last dim {expected}, got {got}")]
    AttentionShape { expected: usize, got: usize },
    #[error("attention tensors must be contiguous in memory order")]
    NonContiguousTensor,
}

pub trait PRoPEOperator {
    fn compute_projective_transform(
        &self,
        source_pose: &CameraPose,
        target_pose: &CameraPose,
        source_intrinsics: &CameraIntrinsics,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<ProjectiveTransform, PRoPEError>;

    fn apply_to_attention(
        &self,
        queries: &mut TensorView,
        keys: &mut TensorView,
        transform: &ProjectiveTransform,
    ) -> Result<(), PRoPEError>;
}

/// PRoPE: Projective Rotary Position Embedding.
///
/// Encodes relative camera geometry as camera-dependent linear transformations
/// applied to Q/K rotations. Unlike additive positional encodings (like Plücker
/// coordinates), PRoPE uses multiplicative rotations derived from the camera's
/// projection geometry.
///
/// Reference: Li et al. (2025) - PRoPE camera conditioning for video generation.
pub struct PRoPE {
    /// Embedding dimension per head (must be even).
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Temporal compression factor (e.g., 4 frames → 1 latent frame).
    pub temporal_compression: usize,
}

impl PRoPE {
    pub fn new(head_dim: usize, num_heads: usize, temporal_compression: usize) -> Self {
        assert!(
            head_dim.is_multiple_of(2),
            "Head dim must be even for PRoPE"
        );
        Self {
            head_dim,
            num_heads,
            temporal_compression,
        }
    }

    /// Compute per-latent-frame rotation parameters from camera poses and intrinsics.
    ///
    /// When temporal compression is active, all original-frame poses belonging to the
    /// same latent slice are aggregated into a single rotary parameter set.
    pub fn compute_rotations(
        &self,
        poses: &[CameraPose],
        intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<[f32; 2]>> {
        if poses.is_empty() {
            return Vec::new();
        }

        let half_dim = self.head_dim / 2;
        let chunk_size = self.temporal_compression.max(1);
        let anchor_pose = &poses[0];

        poses
            .chunks(chunk_size)
            .map(|chunk| {
                let per_subframe: Vec<Vec<[f32; 2]>> = chunk
                    .iter()
                    .map(|pose| {
                        self.compute_projective_transform(anchor_pose, pose, intrinsics, intrinsics)
                            .map(|transform| transform.to_rope_params(half_dim))
                            .unwrap_or_else(|_| vec![[1.0, 0.0]; half_dim])
                    })
                    .collect();
                aggregate_rope_params(&per_subframe, half_dim)
            })
            .collect()
    }

    /// Apply PRoPE rotations to query or key vectors.
    ///
    /// # Arguments
    /// * `qk` - Query or key vectors [num_tokens, head_dim]
    /// * `rotations` - Per-frame rotation params from `compute_rotations`
    /// * `frame_indices` - Frame index for each token (handles temporal compression)
    ///
    /// Returns rotated vectors.
    pub fn apply(
        &self,
        qk: &[Vec<f32>],
        rotations: &[Vec<[f32; 2]>],
        frame_indices: &[usize],
    ) -> Vec<Vec<f32>> {
        assert_eq!(qk.len(), frame_indices.len());
        let identity = vec![[1.0, 0.0]; self.head_dim / 2];

        qk.iter()
            .zip(frame_indices.iter())
            .map(|(vector, &frame_idx)| {
                let compressed_idx =
                    (frame_idx / self.temporal_compression).min(rotations.len().saturating_sub(1));
                let params = rotations.get(compressed_idx).unwrap_or(&identity);
                let mut output = vector.clone();
                rotate_pairs(&mut output, params);
                output
            })
            .collect()
    }
}

impl PRoPEOperator for PRoPE {
    fn compute_projective_transform(
        &self,
        source_pose: &CameraPose,
        target_pose: &CameraPose,
        source_intrinsics: &CameraIntrinsics,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<ProjectiveTransform, PRoPEError> {
        ProjectiveTransform::from_cameras(
            source_pose,
            target_pose,
            source_intrinsics,
            target_intrinsics,
        )
    }

    fn apply_to_attention(
        &self,
        queries: &mut TensorView,
        keys: &mut TensorView,
        transform: &ProjectiveTransform,
    ) -> Result<(), PRoPEError> {
        let query_last_dim = queries.shape().last().copied().unwrap_or(0);
        if query_last_dim != self.head_dim {
            return Err(PRoPEError::AttentionShape {
                expected: self.head_dim,
                got: query_last_dim,
            });
        }

        let key_last_dim = keys.shape().last().copied().unwrap_or(0);
        if key_last_dim != self.head_dim {
            return Err(PRoPEError::AttentionShape {
                expected: self.head_dim,
                got: key_last_dim,
            });
        }

        let params = transform.to_rope_params(self.head_dim / 2);
        let inverse_params: Vec<[f32; 2]> = params.iter().map(|[c, s]| [*c, -*s]).collect();

        let query_data = queries
            .data_mut()
            .as_slice_memory_order_mut()
            .ok_or(PRoPEError::NonContiguousTensor)?;
        for chunk in query_data.chunks_exact_mut(self.head_dim) {
            rotate_pairs(chunk, &params);
        }

        let key_data = keys
            .data_mut()
            .as_slice_memory_order_mut()
            .ok_or(PRoPEError::NonContiguousTensor)?;
        for chunk in key_data.chunks_exact_mut(self.head_dim) {
            rotate_pairs(chunk, &inverse_params);
        }

        Ok(())
    }
}

fn pseudo_inverse(projection: &ProjectionMatrix) -> Result<ProjectionPseudoInverse, PRoPEError> {
    let gram = projection * projection.transpose();
    let inv = gram.try_inverse().ok_or(PRoPEError::SingularProjection)?;
    Ok(projection.transpose() * inv)
}

fn aggregate_rope_params(per_subframe: &[Vec<[f32; 2]>], half_dim: usize) -> Vec<[f32; 2]> {
    if per_subframe.is_empty() {
        return vec![[1.0, 0.0]; half_dim];
    }

    (0..half_dim)
        .map(|dim| {
            let mut sum_cos = 0.0f32;
            let mut sum_sin = 0.0f32;

            for frame in per_subframe {
                let [cos_val, sin_val] = frame.get(dim).copied().unwrap_or([1.0, 0.0]);
                sum_cos += cos_val;
                sum_sin += sin_val;
            }

            let norm = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt();
            if norm <= 1e-6 {
                [1.0, 0.0]
            } else {
                [sum_cos / norm, sum_sin / norm]
            }
        })
        .collect()
}

fn rotate_pairs(values: &mut [f32], params: &[[f32; 2]]) {
    for (pair, [cos_val, sin_val]) in values.chunks_exact_mut(2).zip(params.iter().copied()) {
        let x0 = pair[0];
        let x1 = pair[1];
        pair[0] = x0 * cos_val - x1 * sin_val;
        pair[1] = x0 * sin_val + x1 * cos_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorLayout;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_prope_creation() {
        let prope = PRoPE::new(64, 8, 4);
        assert_eq!(prope.head_dim, 64);
        assert_eq!(prope.num_heads, 8);
    }

    #[test]
    fn test_prope_compute_and_apply() {
        let prope = PRoPE::new(8, 1, 1);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);

        let poses = vec![
            CameraPose::identity(0.0),
            CameraPose::from_translation_rotation(
                1.0,
                Vector3::new(1.0, 0.0, 0.0),
                UnitQuaternion::from_euler_angles(0.0, 0.3, 0.0),
            ),
        ];

        let rotations = prope.compute_rotations(&poses, &intrinsics);
        assert_eq!(rotations.len(), 2);

        let qk = vec![vec![1.0f32; 8]; 4];
        let frame_indices = vec![0, 0, 1, 1];
        let rotated = prope.apply(&qk, &rotations, &frame_indices);
        assert_eq!(rotated.len(), 4);
        assert_eq!(rotated[0].len(), 8);
        assert_ne!(rotated[2], qk[2]);
    }

    #[test]
    fn test_apply_to_attention_preserves_shape() {
        let prope = PRoPE::new(8, 1, 1);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let source_pose = CameraPose::identity(0.0);
        let target_pose = CameraPose::from_translation_rotation(
            1.0,
            Vector3::new(0.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.2, 0.0),
        );
        let transform = prope
            .compute_projective_transform(&source_pose, &target_pose, &intrinsics, &intrinsics)
            .unwrap();

        let mut queries =
            TensorView::from_shape_vec(&[2, 8], vec![1.0; 16], TensorLayout::Flat(vec![2, 8]))
                .unwrap();
        let mut keys =
            TensorView::from_shape_vec(&[2, 8], vec![0.5; 16], TensorLayout::Flat(vec![2, 8]))
                .unwrap();

        prope
            .apply_to_attention(&mut queries, &mut keys, &transform)
            .unwrap();

        assert_eq!(queries.shape(), &[2, 8]);
        assert_eq!(keys.shape(), &[2, 8]);
    }
}
