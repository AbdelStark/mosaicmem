use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::mosaic::MosaicFrame;
use crate::memory::store::{MosaicMemoryStore, RetrievedPatch};

/// High-level memory retrieval that produces MosaicFrames
/// from the memory store given a target camera pose.
///
/// Supports diversity-aware retrieval: when `diversity_radius` > 0,
/// patches too close to already-selected patches in the target 2D view
/// are penalized to encourage spatial coverage diversity.
pub struct MemoryRetriever {
    /// Minimum visibility score threshold.
    pub min_visibility: f32,
    /// Whether to sort patches by depth (front-to-back).
    pub depth_sort: bool,
    /// Spatial diversity radius in target 2D pixels. Patches within this
    /// radius of an already-selected patch have their score penalized.
    /// Set to 0.0 to disable diversity filtering.
    pub diversity_radius: f32,
    /// Penalty factor for patches within the diversity radius (0..1).
    /// A value of 0.5 means the score is halved for each nearby selected patch.
    pub diversity_penalty: f32,
}

impl MemoryRetriever {
    pub fn new() -> Self {
        Self {
            min_visibility: 0.01,
            depth_sort: true,
            diversity_radius: 0.0,
            diversity_penalty: 0.5,
        }
    }

    /// Create a retriever with diversity filtering enabled.
    pub fn with_diversity(diversity_radius: f32, diversity_penalty: f32) -> Self {
        Self {
            min_visibility: 0.01,
            depth_sort: true,
            diversity_radius,
            diversity_penalty,
        }
    }

    /// Retrieve a MosaicFrame for a target camera pose.
    pub fn retrieve(
        &self,
        store: &MosaicMemoryStore,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> MosaicFrame {
        let mut patches = store.retrieve(target_pose, intrinsics);

        // Filter by minimum visibility
        patches.retain(|p| p.visibility_score >= self.min_visibility);

        // Apply diversity-aware selection if enabled
        if self.diversity_radius > 0.0 && patches.len() > 1 {
            patches = self.diversity_select(patches);
        }

        // Sort by depth (front-to-back) for proper occlusion
        if self.depth_sort {
            patches.sort_by(|a, b| {
                a.target_depth
                    .partial_cmp(&b.target_depth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Build coverage mask
        let coverage = self.compute_coverage(&patches, intrinsics);

        MosaicFrame {
            target_pose: target_pose.clone(),
            patches,
            coverage_mask: coverage,
            width: intrinsics.width,
            height: intrinsics.height,
        }
    }

    /// Diversity-aware greedy selection.
    ///
    /// Iteratively selects the highest-scoring patch, then penalizes
    /// remaining patches that are spatially close in the target 2D view.
    /// This encourages broader spatial coverage rather than clustering
    /// many patches in the same image region.
    fn diversity_select(&self, mut candidates: Vec<RetrievedPatch>) -> Vec<RetrievedPatch> {
        // Sort by visibility score descending
        candidates.sort_by(|a, b| {
            b.visibility_score
                .partial_cmp(&a.visibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_patches = candidates.len();
        let mut selected: Vec<RetrievedPatch> = Vec::with_capacity(max_patches);
        let mut scores: Vec<f32> = candidates.iter().map(|p| p.visibility_score).collect();
        let mut used = vec![false; candidates.len()];
        let r2 = self.diversity_radius * self.diversity_radius;

        for _ in 0..max_patches {
            // Find the best remaining candidate
            let best_idx = scores
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let Some(idx) = best_idx else { break };

            if scores[idx] < self.min_visibility {
                break;
            }

            used[idx] = true;
            let selected_pos = candidates[idx].target_position;

            // Clone and use the original score (before any penalties from this round)
            let mut patch = candidates[idx].clone();
            patch.visibility_score = scores[idx];
            selected.push(patch);

            // Penalize nearby unselected patches
            for (j, candidate) in candidates.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let dx = candidate.target_position.x - selected_pos.x;
                let dy = candidate.target_position.y - selected_pos.y;
                if dx * dx + dy * dy < r2 {
                    scores[j] *= self.diversity_penalty;
                }
            }
        }

        selected
    }

    /// Compute a coverage mask indicating which regions have memory patches.
    /// Returns a boolean grid at 1/8 resolution.
    fn compute_coverage(
        &self,
        patches: &[RetrievedPatch],
        intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<bool>> {
        let grid_h = (intrinsics.height / 8).max(1) as usize;
        let grid_w = (intrinsics.width / 8).max(1) as usize;
        let mut coverage = vec![vec![false; grid_w]; grid_h];

        for patch in patches {
            let gx = ((patch.target_position.x / intrinsics.width as f32) * grid_w as f32) as usize;
            let gy =
                ((patch.target_position.y / intrinsics.height as f32) * grid_h as f32) as usize;
            if gx < grid_w && gy < grid_h {
                coverage[gy][gx] = true;
            }
        }

        coverage
    }
}

impl Default for MemoryRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store::{MemoryConfig, Patch3D};
    use nalgebra::{Point2, Point3};

    #[test]
    fn test_retriever_empty_store() {
        let store = MosaicMemoryStore::new(MemoryConfig::default());
        let retriever = MemoryRetriever::new();
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        let frame = retriever.retrieve(&store, &pose, &intrinsics);
        assert!(frame.patches.is_empty());
    }

    #[test]
    fn test_retriever_with_patches() {
        let config = MemoryConfig {
            max_patches: 100,
            top_k: 10,
            ..Default::default()
        };
        let mut store = MosaicMemoryStore::new(config);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let depth_map = vec![vec![5.0f32; 10]; 10];
        let latents = vec![0.5f32; 4 * 4 * 4];
        store.insert_keyframe(0, 0.0, &latents, 4, 4, 4, &depth_map, &intrinsics, &pose);

        let retriever = MemoryRetriever::new();
        let frame = retriever.retrieve(&store, &pose, &intrinsics);
        assert!(!frame.patches.is_empty());
        assert!(frame.has_coverage());
    }

    #[test]
    fn test_diversity_select_spreads_patches() {
        let retriever = MemoryRetriever::with_diversity(20.0, 0.1);

        // Create patches clustered at (50, 50) and one at (90, 90)
        let make_patch = |id, x, y, score| RetrievedPatch {
            patch: Patch3D {
                id,
                center: Point3::new(0.0, 0.0, 5.0),
                source_frame: 0,
                source_timestamp: 0.0,
                source_rect: [0.0, 0.0, 16.0, 16.0],
                latent: vec![0.5; 4],
                latent_height: 1,
                latent_width: 1,
            },
            target_position: Point2::new(x, y),
            target_depth: 5.0,
            visibility_score: score,
        };

        let candidates = vec![
            make_patch(0, 50.0, 50.0, 0.9),
            make_patch(1, 52.0, 51.0, 0.85), // very close to patch 0
            make_patch(2, 53.0, 49.0, 0.80), // very close to patch 0
            make_patch(3, 90.0, 90.0, 0.7),  // far away
        ];

        let selected = retriever.diversity_select(candidates);

        // The first selected should be patch 0 (highest score)
        assert_eq!(selected[0].patch.id, 0);

        // The diverse outlier (patch 3) should be selected before the
        // clustered patches 1/2 which get penalized
        let id_order: Vec<u64> = selected.iter().map(|p| p.patch.id).collect();
        let pos_3 = id_order.iter().position(|&id| id == 3).unwrap();
        let pos_1 = id_order.iter().position(|&id| id == 1).unwrap();
        assert!(
            pos_3 < pos_1,
            "Diverse patch should be selected before clustered ones: {:?}",
            id_order
        );
    }

    #[test]
    fn test_diversity_disabled_by_default() {
        let retriever = MemoryRetriever::new();
        assert_eq!(retriever.diversity_radius, 0.0);
    }

    #[test]
    fn test_coverage_mask_resolution() {
        let retriever = MemoryRetriever::new();
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 80, 80);
        let patches = vec![];
        let coverage = retriever.compute_coverage(&patches, &intrinsics);
        // 80/8 = 10
        assert_eq!(coverage.len(), 10);
        assert_eq!(coverage[0].len(), 10);
    }
}
