use crate::attention::rope::RoPE;
use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::store::RetrievedPatch;
use nalgebra::Point2;

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
    /// Temporal normalization constant for continuous offsets.
    pub temporal_half_life: f32,
}

impl WarpedRoPE {
    pub fn new(dim_per_axis: usize, spatial_resolution: usize, temporal_resolution: usize) -> Self {
        Self {
            rope_u: RoPE::new(dim_per_axis, spatial_resolution, 10000.0),
            rope_v: RoPE::new(dim_per_axis, spatial_resolution, 10000.0),
            rope_t: RoPE::new(dim_per_axis, temporal_resolution, 10000.0),
            spatial_resolution,
            temporal_resolution,
            temporal_half_life: 1.0,
        }
    }

    /// Compute warped positions for retrieved patches relative to target view.
    ///
    /// Returns dense fractional (u, v, t) positions for each patch token.
    pub fn compute_warped_positions(
        &self,
        patches: &[RetrievedPatch],
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<[f32; 3]> {
        patches
            .iter()
            .flat_map(|patch| {
                let token_coords = if patch.patch.token_coords.is_empty() {
                    vec![(
                        patch.patch.source_rect[0] + patch.patch.source_rect[2] / 2.0,
                        patch.patch.source_rect[1] + patch.patch.source_rect[3] / 2.0,
                    )]
                } else {
                    patch.patch.token_coords.clone()
                };
                let depth_tile = patch
                    .patch
                    .depth_tile
                    .clone()
                    .unwrap_or_else(|| vec![patch.patch.source_depth; token_coords.len()]);
                let c2w = patch.patch.source_pose.camera_to_world();
                let temporal = ((patch.patch.source_timestamp - target_pose.timestamp) as f32
                    / self.temporal_half_life.max(1e-6))
                    * self.temporal_resolution as f32;

                token_coords
                    .into_iter()
                    .enumerate()
                    .map(|(idx, (u, v))| {
                        let depth = depth_tile
                            .get(idx)
                            .copied()
                            .unwrap_or(patch.patch.source_depth);
                        let fallback = patch
                            .projected_footprint
                            .get(idx)
                            .copied()
                            .unwrap_or(patch.target_position);

                        let point = if depth.is_finite() && depth > 0.0 {
                            let source_cam = patch
                                .patch
                                .source_intrinsics
                                .unproject(&Point2::new(u, v), depth);
                            let world = c2w.transform_point(&source_cam);
                            let target_cam = target_pose.transform_point(&world);
                            if target_cam.z > 0.0 {
                                intrinsics.project(&target_cam).unwrap_or(fallback)
                            } else {
                                fallback
                            }
                        } else {
                            fallback
                        };

                        [
                            point.x / intrinsics.width as f32 * self.spatial_resolution as f32,
                            point.y / intrinsics.height as f32 * self.spatial_resolution as f32,
                            temporal,
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Apply warped RoPE rotation to query/key vectors.
    ///
    /// `vectors`: [N, dim] where dim = 3 * dim_per_axis
    /// `positions`: [N, 3] warped (u, v, t) positions
    ///
    /// Returns rotated vectors.
    pub fn rotate(&self, vectors: &[Vec<f32>], positions: &[[f32; 3]]) -> Vec<Vec<f32>> {
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
                let rotated_u = rotate_fractional(&self.rope_u, u_part, pos[0]);
                output[..dim_per_axis].copy_from_slice(&rotated_u);

                // Rotate v dimension
                let v_part = &v[dim_per_axis..2 * dim_per_axis];
                let rotated_v = rotate_fractional(&self.rope_v, v_part, pos[1]);
                output[dim_per_axis..2 * dim_per_axis].copy_from_slice(&rotated_v);

                // Rotate t dimension
                let t_part = &v[2 * dim_per_axis..3 * dim_per_axis];
                let rotated_t = rotate_fractional(&self.rope_t, t_part, pos[2]);
                output[2 * dim_per_axis..3 * dim_per_axis].copy_from_slice(&rotated_t);

                output
            })
            .collect()
    }
}

fn rotate_fractional(rope: &RoPE, values: &[f32], position: f32) -> Vec<f32> {
    let half_dim = rope.dim / 2;
    let mut output = vec![0.0f32; rope.dim];

    for i in 0..half_dim {
        let inv_freq = 1.0 / rope.base.powf(2.0 * i as f32 / rope.dim as f32);
        let angle = position * inv_freq;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        let x0 = values[2 * i];
        let x1 = values[2 * i + 1];
        output[2 * i] = x0 * cos_val - x1 * sin_val;
        output[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }

    output
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
        let positions = vec![
            [10.0, 20.0, 5.0],
            [15.5, 25.0, 3.0],
            [30.0, 10.0, 0.0],
            [5.25, 5.0, 1.0],
        ];

        let rotated = wrope.rotate(&vectors, &positions);
        assert_eq!(rotated.len(), 4);
        assert_eq!(rotated[0].len(), 24);
        assert_ne!(rotated[0], rotated[1]);
    }
}
