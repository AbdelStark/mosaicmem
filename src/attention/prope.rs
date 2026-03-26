use crate::camera::{CameraIntrinsics, CameraPose};

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

    /// Compute per-frame rotation parameters from camera poses and intrinsics.
    ///
    /// For each frame, computes a rotation encoding derived from the camera's
    /// projection matrix. Returns rotation angles per head dimension.
    pub fn compute_rotations(
        &self,
        poses: &[CameraPose],
        intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<[f32; 2]>> {
        let k = intrinsics.matrix_f32();
        let half_dim = self.head_dim / 2;

        poses
            .iter()
            .map(|pose| {
                // Extract rotation component from the camera matrix
                let r = pose.world_to_camera.rotation.to_rotation_matrix();
                let t = pose.world_to_camera.translation.vector;

                // Compute projection-derived angles
                // Use the camera matrix elements to derive rotation angles
                let p = k * r.matrix();

                let mut rotations = Vec::with_capacity(half_dim);
                for d in 0..half_dim {
                    // Map projection matrix elements to rotation angles
                    // Use different rows/cols for different dimension pairs
                    let row = d % 3;
                    let col = (d / 3) % 3;
                    let angle = p[(row, col)].atan2(t[d % 3] + 1.0);
                    rotations.push([angle.cos(), angle.sin()]);
                }
                rotations
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
        let half_dim = self.head_dim / 2;

        qk.iter()
            .zip(frame_indices.iter())
            .map(|(v, &frame_idx)| {
                // Map to the compressed frame index
                let compressed_idx =
                    (frame_idx / self.temporal_compression).min(rotations.len().saturating_sub(1));
                let rot = &rotations[compressed_idx];

                let mut output = v.clone();
                for d in 0..half_dim.min(rot.len()) {
                    let [cos_val, sin_val] = rot[d];
                    let x0 = v[2 * d];
                    let x1 = v[2 * d + 1];
                    output[2 * d] = x0 * cos_val - x1 * sin_val;
                    output[2 * d + 1] = x0 * sin_val + x1 * cos_val;
                }
                output
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;
    use nalgebra::Vector3;

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
                UnitQuaternion::identity(),
            ),
        ];

        let rotations = prope.compute_rotations(&poses, &intrinsics);
        assert_eq!(rotations.len(), 2);

        let qk = vec![vec![1.0f32; 8]; 4];
        let frame_indices = vec![0, 0, 1, 1];

        let rotated = prope.apply(&qk, &rotations, &frame_indices);
        assert_eq!(rotated.len(), 4);
        assert_eq!(rotated[0].len(), 8);
    }
}
