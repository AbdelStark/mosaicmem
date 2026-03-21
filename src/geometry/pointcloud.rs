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

    /// Export the point cloud to PLY format (ASCII).
    pub fn export_ply(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);

        // PLY header
        writeln!(f, "ply")?;
        writeln!(f, "format ascii 1.0")?;
        writeln!(f, "element vertex {}", self.points.len())?;
        writeln!(f, "property float x")?;
        writeln!(f, "property float y")?;
        writeln!(f, "property float z")?;
        writeln!(f, "property uchar red")?;
        writeln!(f, "property uchar green")?;
        writeln!(f, "property uchar blue")?;
        if self.points.iter().any(|p| p.normal.is_some()) {
            writeln!(f, "property float nx")?;
            writeln!(f, "property float ny")?;
            writeln!(f, "property float nz")?;
        }
        writeln!(f, "end_header")?;

        let has_normals = self.points.iter().any(|p| p.normal.is_some());
        for p in &self.points {
            if has_normals {
                let n = p.normal.unwrap_or(nalgebra::Vector3::zeros());
                writeln!(
                    f,
                    "{} {} {} {} {} {} {} {} {}",
                    p.position.x,
                    p.position.y,
                    p.position.z,
                    p.color[0],
                    p.color[1],
                    p.color[2],
                    n.x,
                    n.y,
                    n.z
                )?;
            } else {
                writeln!(
                    f,
                    "{} {} {} {} {} {}",
                    p.position.x, p.position.y, p.position.z, p.color[0], p.color[1], p.color[2]
                )?;
            }
        }

        Ok(())
    }

    /// Import a point cloud from a PLY file (ASCII format).
    pub fn import_ply(path: &std::path::Path) -> Result<Self, std::io::Error> {
        use std::io::BufRead;
        let file = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut lines = file.lines();

        // Parse header
        let mut num_vertices: usize = 0;
        let mut has_normals = false;
        let mut in_header = true;

        while in_header {
            let line = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "Unexpected end of header")
            })??;
            let trimmed = line.trim();
            if trimmed.starts_with("element vertex") {
                if let Some(count_str) = trimmed.split_whitespace().nth(2) {
                    num_vertices = count_str.parse().map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid vertex count")
                    })?;
                }
            } else if trimmed == "property float nx" {
                has_normals = true;
            } else if trimmed == "end_header" {
                in_header = false;
            }
        }

        // Parse vertices
        let mut points = Vec::with_capacity(num_vertices);
        for line_result in lines.take(num_vertices) {
            let line = line_result?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                continue;
            }

            let parse_f32 = |s: &str| -> Result<f32, std::io::Error> {
                s.parse()
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Bad float"))
            };
            let parse_u8 = |s: &str| -> Result<u8, std::io::Error> {
                s.parse()
                    .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Bad u8"))
            };

            let position = nalgebra::Point3::new(
                parse_f32(parts[0])?,
                parse_f32(parts[1])?,
                parse_f32(parts[2])?,
            );
            let color = [
                parse_u8(parts[3])?,
                parse_u8(parts[4])?,
                parse_u8(parts[5])?,
            ];

            let normal = if has_normals && parts.len() >= 9 {
                Some(nalgebra::Vector3::new(
                    parse_f32(parts[6])?,
                    parse_f32(parts[7])?,
                    parse_f32(parts[8])?,
                ))
            } else {
                None
            };

            points.push(Point3DColored {
                position,
                color,
                normal,
            });
        }

        Ok(Self { points })
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

    #[test]
    fn test_ply_export_import_roundtrip() {
        let mut cloud = PointCloud3D::new();
        cloud.add_point(Point3::new(1.0, 2.0, 3.0), [255, 0, 0]);
        cloud.add_point(Point3::new(4.0, 5.0, 6.0), [0, 255, 0]);
        cloud.add_point(Point3::new(7.0, 8.0, 9.0), [0, 0, 255]);

        let dir = std::env::temp_dir().join("mosaicmem_test_ply");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.ply");

        cloud.export_ply(&path).unwrap();

        let loaded = PointCloud3D::import_ply(&path).unwrap();
        assert_eq!(loaded.len(), 3);

        // Check values roundtrip
        for (orig, loaded_p) in cloud.points.iter().zip(loaded.points.iter()) {
            assert!((orig.position.x - loaded_p.position.x).abs() < 1e-5);
            assert!((orig.position.y - loaded_p.position.y).abs() < 1e-5);
            assert!((orig.position.z - loaded_p.position.z).abs() < 1e-5);
            assert_eq!(orig.color, loaded_p.color);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_ply_empty_cloud() {
        let cloud = PointCloud3D::new();
        let dir = std::env::temp_dir().join("mosaicmem_test_ply_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.ply");

        cloud.export_ply(&path).unwrap();
        let loaded = PointCloud3D::import_ply(&path).unwrap();
        assert_eq!(loaded.len(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_centroid() {
        let mut cloud = PointCloud3D::new();
        cloud.add_point(Point3::new(0.0, 0.0, 0.0), [0; 3]);
        cloud.add_point(Point3::new(2.0, 4.0, 6.0), [0; 3]);
        let centroid = cloud.centroid().unwrap();
        assert!((centroid.x - 1.0).abs() < 1e-5);
        assert!((centroid.y - 2.0).abs() < 1e-5);
        assert!((centroid.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_filter_sphere() {
        let mut cloud = PointCloud3D::new();
        cloud.add_point(Point3::new(0.0, 0.0, 0.0), [0; 3]);
        cloud.add_point(Point3::new(10.0, 0.0, 0.0), [0; 3]);
        let filtered = cloud.filter_sphere(&Point3::origin(), 5.0);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_empty_cloud_operations() {
        let cloud = PointCloud3D::new();
        assert!(cloud.is_empty());
        assert!(cloud.bounding_box().is_none());
        assert!(cloud.centroid().is_none());
        assert_eq!(cloud.positions().len(), 0);
        let filtered = cloud.filter_sphere(&Point3::origin(), 10.0);
        assert!(filtered.is_empty());
        let downsampled = cloud.voxel_downsample(1.0);
        assert!(downsampled.is_empty());
    }

    #[test]
    fn test_default() {
        let cloud = PointCloud3D::default();
        assert!(cloud.is_empty());
    }
}
