use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A single 3D point with color and optional normal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point3DColored {
    pub position: Point3<f32>,
    pub color: [u8; 3],
    pub normal: Option<Vector3<f32>>,
}

/// A collection of 3D points with colors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloud3D {
    pub points: Vec<Point3DColored>,
}

impl PointCloud3D {
    pub fn new() -> Self {
        Self { points: vec![] }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    pub fn from_points(points: Vec<Point3DColored>) -> Self {
        Self { points }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn add_point(&mut self, position: Point3<f32>, color: [u8; 3]) {
        self.points.push(Point3DColored {
            position,
            color,
            normal: None,
        });
    }

    /// Merge another point cloud into this one.
    pub fn merge(&mut self, other: &PointCloud3D) {
        self.points.extend_from_slice(&other.points);
    }

    /// Get all positions as a flat array for KD-tree construction.
    pub fn positions(&self) -> Vec<[f32; 3]> {
        self.points
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z])
            .collect()
    }

    /// Compute the bounding box (min, max) of the point cloud.
    pub fn bounding_box(&self) -> Option<(Point3<f32>, Point3<f32>)> {
        if self.points.is_empty() {
            return None;
        }
        let mut min = self.points[0].position;
        let mut max = self.points[0].position;
        for p in &self.points[1..] {
            min.x = min.x.min(p.position.x);
            min.y = min.y.min(p.position.y);
            min.z = min.z.min(p.position.z);
            max.x = max.x.max(p.position.x);
            max.y = max.y.max(p.position.y);
            max.z = max.z.max(p.position.z);
        }
        Some((min, max))
    }

    /// Voxel grid downsampling: keep one point per voxel cell.
    pub fn voxel_downsample(&self, voxel_size: f32) -> PointCloud3D {
        use std::collections::HashMap;
        let mut voxels: HashMap<(i64, i64, i64), Point3DColored> = HashMap::new();

        for point in &self.points {
            let key = (
                (point.position.x / voxel_size).floor() as i64,
                (point.position.y / voxel_size).floor() as i64,
                (point.position.z / voxel_size).floor() as i64,
            );
            voxels.entry(key).or_insert(*point);
        }

        PointCloud3D::from_points(voxels.into_values().collect())
    }

    /// Compute centroid of the point cloud.
    pub fn centroid(&self) -> Option<Point3<f32>> {
        if self.points.is_empty() {
            return None;
        }
        let sum: Vector3<f32> = self
            .points
            .par_iter()
            .map(|p| p.position.coords)
            .reduce(Vector3::zeros, |a, b| a + b);
        let n = self.points.len() as f32;
        Some(Point3::from(sum / n))
    }

    /// Filter points within a sphere of given center and radius.
    pub fn filter_sphere(&self, center: &Point3<f32>, radius: f32) -> PointCloud3D {
        let r2 = radius * radius;
        let points: Vec<_> = self
            .points
            .par_iter()
            .filter(|p| (p.position - center).norm_squared() <= r2)
            .cloned()
            .collect();
        PointCloud3D::from_points(points)
    }
}

impl Default for PointCloud3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_cloud_basics() {
        let mut cloud = PointCloud3D::new();
        cloud.add_point(Point3::new(1.0, 2.0, 3.0), [255, 0, 0]);
        cloud.add_point(Point3::new(4.0, 5.0, 6.0), [0, 255, 0]);
        assert_eq!(cloud.len(), 2);
    }

    #[test]
    fn test_bounding_box() {
        let mut cloud = PointCloud3D::new();
        cloud.add_point(Point3::new(1.0, 2.0, 3.0), [0; 3]);
        cloud.add_point(Point3::new(4.0, 5.0, 6.0), [0; 3]);
        let (min, max) = cloud.bounding_box().unwrap();
        assert_eq!(min, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(max, Point3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_voxel_downsample() {
        let mut cloud = PointCloud3D::new();
        // Add many points in the same voxel
        for i in 0..10 {
            cloud.add_point(Point3::new(0.01 * i as f32, 0.0, 0.0), [0; 3]);
        }
        let downsampled = cloud.voxel_downsample(1.0);
        assert_eq!(downsampled.len(), 1);
    }

    #[test]
    fn test_merge() {
        let mut cloud1 = PointCloud3D::new();
        cloud1.add_point(Point3::new(0.0, 0.0, 0.0), [0; 3]);
        let mut cloud2 = PointCloud3D::new();
        cloud2.add_point(Point3::new(1.0, 1.0, 1.0), [0; 3]);
        cloud1.merge(&cloud2);
        assert_eq!(cloud1.len(), 2);
    }
}
