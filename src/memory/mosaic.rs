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
                    source_frame: 0,
                    source_timestamp: 0.0,
                    source_rect: [0.0, 0.0, 16.0, 16.0],
                    latent: vec![0.5f32; 2 * 2 * 4],
                    latent_height: 2,
                    latent_width: 2,
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
