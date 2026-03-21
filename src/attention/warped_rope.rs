use crate::attention::rope::RoPE;
use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::store::RetrievedPatch;

/// Warped RoPE: geometric positional encoding that uses reprojected 3D coordinates.
///
/// Instead of using standard grid positions, Warped RoPE:
/// 1. Takes each memory patch's 3D center
/// 2. Reprojects it into the target camera's 2D coordinates
/// 3. Computes temporal offset (t_source - t_current)
/// 4. Uses the warped (u, v, t) as RoPE positions
///
/// This ensures patches from different viewpoints/times appear at geometrically
/// correct positions in the attention computation.
pub struct WarpedRoPE {
    /// RoPE for spatial u dimension.
    pub rope_u: RoPE,
    /// RoPE for spatial v dimension.
    pub rope_v: RoPE,
    /// RoPE for temporal dimension.
    pub rope_t: RoPE,
    /// Spatial resolution for quantizing continuous coordinates.
    pub spatial_resolution: usize,
    /// Temporal resolution for quantizing time offsets.
    pub temporal_resolution: usize,
}

impl WarpedRoPE {
    pub fn new(dim_per_axis: usize, spatial_resolution: usize, temporal_resolution: usize) -> Self {
        Self {
            rope_u: RoPE::new(dim_per_axis, spatial_resolution, 10000.0),
            rope_v: RoPE::new(dim_per_axis, spatial_resolution, 10000.0),
            rope_t: RoPE::new(dim_per_axis, temporal_resolution, 10000.0),
            spatial_resolution,
            temporal_resolution,
        }
    }

    /// Compute warped positions for retrieved patches relative to target view.
    ///
    /// Returns quantized (u, v, t) positions for each patch.
    pub fn compute_warped_positions(
        &self,
        patches: &[RetrievedPatch],
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<[usize; 3]> {
        patches
            .iter()
            .map(|patch| {
                // Use the already-computed target position
                let u = (patch.target_position.x / intrinsics.width as f32
                    * self.spatial_resolution as f32) as usize;
                let v = (patch.target_position.y / intrinsics.height as f32
                    * self.spatial_resolution as f32) as usize;

                // Temporal offset
                let dt = (patch.patch.source_timestamp - target_pose.timestamp).abs();
                let t = (dt * self.temporal_resolution as f64 / 10.0) as usize; // 10 seconds max

                [
                    u.min(self.spatial_resolution - 1),
                    v.min(self.spatial_resolution - 1),
                    t.min(self.temporal_resolution - 1),
                ]
            })
            .collect()
    }

    /// Apply warped RoPE rotation to query/key vectors.
    ///
    /// `vectors`: [N, dim] where dim = 3 * dim_per_axis
    /// `positions`: [N, 3] warped (u, v, t) positions
    ///
    /// Returns rotated vectors.
    pub fn rotate(&self, vectors: &[Vec<f32>], positions: &[[usize; 3]]) -> Vec<Vec<f32>> {
        assert_eq!(vectors.len(), positions.len());
        let dim_per_axis = self.rope_u.dim;

        vectors
            .iter()
            .zip(positions.iter())
            .map(|(v, pos)| {
                let total_dim = v.len();
                assert!(
                    total_dim >= 3 * dim_per_axis,
                    "Vector dim must be >= 3 * dim_per_axis"
                );

                let mut output = v.clone();

                // Rotate u dimension
                let u_part = &v[..dim_per_axis];
                let rotated_u = self.rope_u.rotate(u_part, pos[0]);
                output[..dim_per_axis].copy_from_slice(&rotated_u);

                // Rotate v dimension
                let v_part = &v[dim_per_axis..2 * dim_per_axis];
                let rotated_v = self.rope_v.rotate(v_part, pos[1]);
                output[dim_per_axis..2 * dim_per_axis].copy_from_slice(&rotated_v);

                // Rotate t dimension
                let t_part = &v[2 * dim_per_axis..3 * dim_per_axis];
                let rotated_t = self.rope_t.rotate(t_part, pos[2]);
                output[2 * dim_per_axis..3 * dim_per_axis].copy_from_slice(&rotated_t);

                output
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warped_rope_creation() {
        let wrope = WarpedRoPE::new(8, 64, 32);
        assert_eq!(wrope.rope_u.dim, 8);
        assert_eq!(wrope.spatial_resolution, 64);
        assert_eq!(wrope.temporal_resolution, 32);
    }

    #[test]
    fn test_warped_rope_rotate() {
        let wrope = WarpedRoPE::new(8, 64, 32);
        let vectors = vec![vec![1.0f32; 24]; 4]; // 3 * 8 = 24 dim
        let positions = vec![[10, 20, 5], [15, 25, 3], [30, 10, 0], [5, 5, 1]];

        let rotated = wrope.rotate(&vectors, &positions);
        assert_eq!(rotated.len(), 4);
        assert_eq!(rotated[0].len(), 24);
    }
}
