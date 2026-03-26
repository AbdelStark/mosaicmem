use crate::camera::CameraPose;
use crate::memory::store::RetrievedPatch;

/// A mosaic frame: the assembled set of retrieved patches for a target viewpoint.
#[derive(Debug, Clone)]
pub struct MosaicFrame {
    /// Target camera pose for which this mosaic was assembled.
    pub target_pose: CameraPose,
    /// Retrieved and aligned patches, sorted by depth.
    pub patches: Vec<RetrievedPatch>,
    /// Coverage mask: which regions have memory (at 1/8 resolution).
    pub coverage_mask: Vec<Vec<bool>>,
    /// Target frame dimensions.
    pub width: u32,
    pub height: u32,
}

/// Rasterized latent canvas assembled from retrieved patches.
#[derive(Debug, Clone)]
pub struct LatentCanvas {
    pub data: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

impl LatentCanvas {
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn to_tokens(&self) -> Vec<Vec<f32>> {
        let token_count = self.height * self.width;
        let mut tokens = Vec::with_capacity(token_count);
        for idx in 0..token_count {
            let start = idx * self.channels;
            let end = start + self.channels;
            tokens.push(self.data[start..end].to_vec());
        }
        tokens
    }

    /// Convert the rasterized HWC canvas into flattened [C, T, H, W].
    ///
    /// The same spatial memory is replicated across `temporal_frames` latent
    /// timesteps, which is sufficient for the synthetic inference path.
    pub fn to_cthw(&self, temporal_frames: usize) -> Vec<f32> {
        if temporal_frames == 0 || self.channels == 0 || self.height == 0 || self.width == 0 {
            return vec![];
        }

        let frame_size = self.height * self.width;
        let mut latent = vec![0.0f32; self.channels * temporal_frames * frame_size];

        for c in 0..self.channels {
            for t in 0..temporal_frames {
                for y in 0..self.height {
                    for x in 0..self.width {
                        let src_idx = (y * self.width + x) * self.channels + c;
                        let dst_idx =
                            ((c * temporal_frames + t) * self.height + y) * self.width + x;
                        latent[dst_idx] = self.data[src_idx];
                    }
                }
            }
        }

        latent
    }
}

impl MosaicFrame {
    /// Get the number of patches in this mosaic.
    pub fn num_patches(&self) -> usize {
        self.patches.len()
    }

    /// Check if the mosaic has any memory coverage.
    pub fn has_coverage(&self) -> bool {
        self.coverage_mask.iter().any(|row| row.iter().any(|&v| v))
    }

    /// Compute coverage ratio (0..1).
    pub fn coverage_ratio(&self) -> f32 {
        let total: usize = self.coverage_mask.iter().map(|row| row.len()).sum();
        if total == 0 {
            return 0.0;
        }
        let covered: usize = self
            .coverage_mask
            .iter()
            .map(|row| row.iter().filter(|&&v| v).count())
            .sum();
        covered as f32 / total as f32
    }

    /// Flatten all patch latents into a single token sequence for attention.
    /// Returns (tokens, positions) where tokens is [N, C] and positions is [N, 3].
    pub fn compose_tokens(&self) -> (Vec<Vec<f32>>, Vec<[f32; 3]>) {
        let mut tokens = Vec::new();
        let mut positions = Vec::new();

        for patch in &self.patches {
            let channels = if patch.patch.latent_height * patch.patch.latent_width > 0 {
                patch.patch.latent.len() / (patch.patch.latent_height * patch.patch.latent_width)
            } else {
                continue;
            };

            for i in 0..(patch.patch.latent_height * patch.patch.latent_width) {
                let start = i * channels;
                let end = start + channels;
                if end <= patch.patch.latent.len() {
                    tokens.push(patch.patch.latent[start..end].to_vec());
                    positions.push([
                        patch.target_position.x,
                        patch.target_position.y,
                        patch.target_depth,
                    ]);
                }
            }
        }

        (tokens, positions)
    }

    /// Rasterize memory patches into a latent canvas aligned to the target view.
    ///
    /// Each patch token is splatted into the target latent grid using bilinear
    /// weights around the patch's projected center. Contributions are weighted
    /// by patch visibility and normalized per canvas cell.
    pub fn compose_latent_canvas(
        &self,
        latent_height: usize,
        latent_width: usize,
        channels: usize,
    ) -> LatentCanvas {
        let cell_count = latent_height.saturating_mul(latent_width);
        let mut accum = vec![0.0f32; cell_count.saturating_mul(channels)];
        let mut weights = vec![0.0f32; cell_count];

        if latent_height == 0
            || latent_width == 0
            || channels == 0
            || self.width == 0
            || self.height == 0
        {
            return LatentCanvas {
                data: accum,
                height: latent_height,
                width: latent_width,
                channels,
            };
        }

        for patch in &self.patches {
            let patch_tokens = patch.patch.latent_height * patch.patch.latent_width;
            if patch_tokens == 0 || patch.patch.latent.len() < patch_tokens * channels {
                continue;
            }

            let patch_weight = patch.visibility_score.max(0.0).max(1e-6);
            let center_x = patch.target_position.x / self.width as f32 * latent_width as f32;
            let center_y = patch.target_position.y / self.height as f32 * latent_height as f32;
            let half_w = patch.patch.latent_width as f32 / 2.0;
            let half_h = patch.patch.latent_height as f32 / 2.0;

            for py in 0..patch.patch.latent_height {
                for px in 0..patch.patch.latent_width {
                    let token_idx = py * patch.patch.latent_width + px;
                    let token_start = token_idx * channels;
                    let token = &patch.patch.latent[token_start..token_start + channels];

                    let local_x = center_x + px as f32 + 0.5 - half_w;
                    let local_y = center_y + py as f32 + 0.5 - half_h;

                    let x0 = local_x.floor();
                    let y0 = local_y.floor();
                    let fx = local_x - x0;
                    let fy = local_y - y0;

                    let weights_2d = [
                        ((1.0 - fx) * (1.0 - fy), x0 as isize, y0 as isize),
                        ((1.0 - fx) * fy, x0 as isize, y0 as isize + 1),
                        (fx * (1.0 - fy), x0 as isize + 1, y0 as isize),
                        (fx * fy, x0 as isize + 1, y0 as isize + 1),
                    ];

                    for (bilinear_weight, cx, cy) in weights_2d {
                        if bilinear_weight <= 0.0 {
                            continue;
                        }
                        if cx < 0 || cy < 0 {
                            continue;
                        }
                        let cx = cx as usize;
                        let cy = cy as usize;
                        if cx >= latent_width || cy >= latent_height {
                            continue;
                        }

                        let cell_idx = cy * latent_width + cx;
                        let cell_weight = patch_weight * bilinear_weight;
                        weights[cell_idx] += cell_weight;

                        let dst_start = cell_idx * channels;
                        for c in 0..channels {
                            accum[dst_start + c] += token[c] * cell_weight;
                        }
                    }
                }
            }
        }

        for (cell, weight) in weights.iter().copied().enumerate().take(cell_count) {
            if weight > 0.0 {
                let start = cell * channels;
                let end = start + channels;
                for value in &mut accum[start..end] {
                    *value /= weight;
                }
            }
        }

        LatentCanvas {
            data: accum,
            height: latent_height,
            width: latent_width,
            channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store::Patch3D;
    use nalgebra::{Point2, Point3};

    fn make_mosaic(num_patches: usize) -> MosaicFrame {
        let patches: Vec<RetrievedPatch> = (0..num_patches)
            .map(|i| RetrievedPatch {
                patch: Patch3D {
                    id: i as u64,
                    center: Point3::new(i as f32, 0.0, 5.0),
                    source_pose: CameraPose::identity(0.0),
                    source_frame: 0,
                    source_timestamp: 0.0,
                    source_depth: 5.0,
                    source_rect: [0.0, 0.0, 16.0, 16.0],
                    latent: vec![0.5f32; 2 * 2 * 4],
                    latent_height: 2,
                    latent_width: 2,
                    token_coords: vec![(8.0, 8.0); 4],
                    depth_tile: Some(vec![5.0; 4]),
                    source_intrinsics: crate::camera::CameraIntrinsics::default(),
                    normal_estimate: None,
                    latent_shape: (4, 2, 2),
                },
                target_position: Point2::new(50.0, 50.0),
                target_depth: 5.0,
                visibility_score: 0.9,
            })
            .collect();

        MosaicFrame {
            target_pose: CameraPose::identity(0.0),
            patches,
            coverage_mask: vec![vec![true; 12]; 12],
            width: 100,
            height: 100,
        }
    }

    #[test]
    fn test_mosaic_frame_basics() {
        let mosaic = make_mosaic(3);
        assert_eq!(mosaic.num_patches(), 3);
        assert!(mosaic.has_coverage());
        assert!(mosaic.coverage_ratio() > 0.0);
    }

    #[test]
    fn test_mosaic_empty() {
        let mosaic = MosaicFrame {
            target_pose: CameraPose::identity(0.0),
            patches: vec![],
            coverage_mask: vec![vec![false; 10]; 10],
            width: 100,
            height: 100,
        };
        assert_eq!(mosaic.num_patches(), 0);
        assert!(!mosaic.has_coverage());
        assert_eq!(mosaic.coverage_ratio(), 0.0);
    }

    #[test]
    fn test_compose_tokens() {
        let mosaic = make_mosaic(2);
        let (tokens, positions) = mosaic.compose_tokens();
        // Each patch has 2x2=4 tokens
        assert_eq!(tokens.len(), 8);
        assert_eq!(positions.len(), 8);
        // Each token should have 4 channels
        assert_eq!(tokens[0].len(), 4);
    }

    #[test]
    fn test_compose_latent_canvas_places_tokens() {
        let mosaic = make_mosaic(1);
        let canvas = mosaic.compose_latent_canvas(8, 8, 4);
        assert_eq!(canvas.height, 8);
        assert_eq!(canvas.width, 8);
        assert_eq!(canvas.channels, 4);
        assert!(canvas.data.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_compose_latent_canvas_accumulates_overlap() {
        let mut mosaic = make_mosaic(2);
        mosaic.patches[1].visibility_score = 2.0;
        let canvas = mosaic.compose_latent_canvas(8, 8, 4);
        let mid = (4 * 8 + 4) * 4;
        let sum: f32 = canvas.data[mid..mid + 4].iter().sum();
        assert!(sum > 0.0);
    }

    #[test]
    fn test_latent_canvas_to_cthw_replicates_across_time() {
        let mosaic = make_mosaic(1);
        let canvas = mosaic.compose_latent_canvas(4, 4, 4);
        let latent = canvas.to_cthw(2);
        let frame_size = 4 * 4;

        assert_eq!(latent.len(), 4 * 2 * frame_size);
        for c in 0..4 {
            let t0 = c * 2 * frame_size;
            let t1 = t0 + frame_size;
            assert_eq!(&latent[t0..t0 + frame_size], &latent[t1..t1 + frame_size],);
        }
    }

    #[test]
    fn test_coverage_ratio_partial() {
        let mosaic = MosaicFrame {
            target_pose: CameraPose::identity(0.0),
            patches: vec![],
            coverage_mask: vec![
                vec![true, false, false, false],
                vec![false, false, false, false],
            ],
            width: 100,
            height: 100,
        };
        let ratio = mosaic.coverage_ratio();
        assert!((ratio - 0.125).abs() < 1e-5); // 1/8
    }
}
