use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::mosaic::MosaicFrame;
use crate::memory::store::{MosaicMemoryStore, RetrievedPatch};

/// High-level memory retrieval that produces MosaicFrames
/// from the memory store given a target camera pose.
pub struct MemoryRetriever {
    /// Minimum visibility score threshold.
    pub min_visibility: f32,
    /// Whether to sort patches by depth (front-to-back).
    pub depth_sort: bool,
}

impl MemoryRetriever {
    pub fn new() -> Self {
        Self {
            min_visibility: 0.01,
            depth_sort: true,
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
            let gy = ((patch.target_position.y / intrinsics.height as f32) * grid_h as f32) as usize;
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
    use crate::memory::store::MemoryConfig;

    #[test]
    fn test_retriever_empty_store() {
        let store = MosaicMemoryStore::new(MemoryConfig::default());
        let retriever = MemoryRetriever::new();
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        let frame = retriever.retrieve(&store, &pose, &intrinsics);
        assert!(frame.patches.is_empty());
    }
}
