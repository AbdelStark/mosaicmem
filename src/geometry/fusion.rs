use crate::camera::{CameraIntrinsics, CameraPose};
use crate::geometry::depth::{DepthError, DepthEstimator};
use crate::geometry::pointcloud::PointCloud3D;
use crate::geometry::projection::unproject_depth_map;
use kiddo::{KdTree, SquaredEuclidean};
use thiserror::Error;

/// Errors that can occur during streaming fusion.
#[derive(Error, Debug)]
pub enum FusionError {
    #[error("Depth estimation failed during fusion: {0}")]
    DepthFailed(#[from] DepthError),
}

/// Streaming point cloud fusion: incrementally builds a global 3D point cloud
/// from incoming frames with depth estimation.
pub struct StreamingFusion {
    /// The accumulated global point cloud.
    pub global_cloud: PointCloud3D,
    /// KD-tree for spatial queries on the global cloud.
    pub kdtree: Option<KdTree<f32, 3>>,
    /// Voxel size for deduplication.
    pub voxel_size: f32,
    /// Number of keyframes fused so far.
    pub num_keyframes: usize,
}

impl StreamingFusion {
    pub fn new(voxel_size: f32) -> Self {
        Self {
            global_cloud: PointCloud3D::new(),
            kdtree: None,
            voxel_size,
            num_keyframes: 0,
        }
    }

    /// Add a new keyframe: estimate depth, unproject to 3D, merge into global cloud.
    pub fn add_keyframe(
        &mut self,
        image: &[u8],
        width: u32,
        height: u32,
        intrinsics: &CameraIntrinsics,
        pose: &CameraPose,
        depth_estimator: &dyn DepthEstimator,
    ) -> Result<usize, FusionError> {
        // Estimate depth
        let depth_map = depth_estimator.estimate_depth(image, width, height)?;

        // Unproject to 3D
        let frame_cloud = unproject_depth_map(&depth_map, None, intrinsics, pose);

        // Voxel downsample the frame cloud
        let frame_cloud = frame_cloud.voxel_downsample(self.voxel_size);

        let num_new_points = frame_cloud.len();

        // Merge into global cloud
        self.global_cloud.merge(&frame_cloud);

        // Rebuild KD-tree
        self.rebuild_kdtree();

        self.num_keyframes += 1;

        Ok(num_new_points)
    }

    /// Add a pre-computed point cloud directly (skip depth estimation).
    pub fn add_cloud(&mut self, cloud: &PointCloud3D) {
        let downsampled = cloud.voxel_downsample(self.voxel_size);
        self.global_cloud.merge(&downsampled);
        self.rebuild_kdtree();
        self.num_keyframes += 1;
    }

    /// Rebuild the KD-tree index from the current global cloud.
    ///
    /// Handles the edge case where too many points share identical coordinates
    /// on one axis (kiddo bucket size limitation). In this case, a coarser
    /// voxel downsample is applied to reduce duplication before rebuilding.
    pub fn rebuild_kdtree(&mut self) {
        if self.global_cloud.is_empty() {
            self.kdtree = None;
            return;
        }

        match self.try_build_kdtree() {
            Some(tree) => self.kdtree = Some(tree),
            None => {
                // Too many points with same position — downsample more aggressively
                tracing::debug!(
                    "KD-tree rebuild: deduplicating {} points with coarser voxel",
                    self.global_cloud.len()
                );
                self.global_cloud = self.global_cloud.voxel_downsample(self.voxel_size * 2.0);
                self.kdtree = self.try_build_kdtree();
            }
        }
    }

    /// Attempt to build a KD-tree, returning None if kiddo panics due to
    /// too many coincident points.
    fn try_build_kdtree(&self) -> Option<KdTree<f32, 3>> {
        use std::panic;
        let points: Vec<([f32; 3], u64)> = self
            .global_cloud
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| ([p.position.x, p.position.y, p.position.z], i as u64))
            .collect();

        panic::catch_unwind(|| {
            let mut tree = KdTree::new();
            for (pos, id) in &points {
                tree.add(pos, *id);
            }
            tree
        })
        .ok()
    }

    /// Query nearest neighbors within a radius.
    pub fn query_radius(&self, center: &[f32; 3], radius: f32) -> Vec<(u64, f32)> {
        match &self.kdtree {
            Some(tree) => {
                let results = tree.within_unsorted::<SquaredEuclidean>(center, radius * radius);
                results
                    .iter()
                    .map(|n| (n.item, n.distance.sqrt()))
                    .collect()
            }
            None => vec![],
        }
    }

    /// Query K nearest neighbors.
    pub fn query_knn(&self, center: &[f32; 3], k: usize) -> Vec<(u64, f32)> {
        match &self.kdtree {
            Some(tree) => {
                let results = tree.nearest_n::<SquaredEuclidean>(center, k);
                results
                    .iter()
                    .map(|n| (n.item, n.distance.sqrt()))
                    .collect()
            }
            None => vec![],
        }
    }

    /// Downsample the global cloud to reduce memory usage.
    pub fn downsample_global(&mut self, voxel_size: f32) {
        self.global_cloud = self.global_cloud.voxel_downsample(voxel_size);
        self.rebuild_kdtree();
    }

    /// Get total number of points in the global cloud.
    pub fn num_points(&self) -> usize {
        self.global_cloud.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::depth::SyntheticDepthEstimator;
    #[test]
    fn test_streaming_fusion() {
        let mut fusion = StreamingFusion::new(0.1);
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);

        let image = vec![128u8; 100 * 100 * 3];
        let n = fusion
            .add_keyframe(&image, 100, 100, &intrinsics, &pose, &depth_estimator)
            .unwrap();

        assert!(n > 0);
        assert!(fusion.num_points() > 0);
        assert!(fusion.kdtree.is_some());
    }

    #[test]
    fn test_knn_query() {
        let mut fusion = StreamingFusion::new(0.5);
        let mut cloud = PointCloud3D::new();
        // Distribute points in 3D to avoid kiddo bucket splitting issues
        for i in 0..50 {
            let angle = i as f32 * 0.3;
            cloud.add_point(
                nalgebra::Point3::new(
                    angle.cos() * i as f32,
                    angle.sin() * i as f32,
                    i as f32 * 0.5,
                ),
                [0; 3],
            );
        }
        fusion.add_cloud(&cloud);

        let neighbors = fusion.query_knn(&[5.0, 3.0, 2.0], 3);
        assert_eq!(neighbors.len(), 3);
    }
}
