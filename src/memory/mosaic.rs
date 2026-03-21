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
