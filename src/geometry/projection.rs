use crate::camera::{CameraIntrinsics, CameraPose};
use crate::geometry::pointcloud::{Point3DColored, PointCloud3D};
use nalgebra::{Point2, Point3};

/// Unproject a depth map into a 3D point cloud in world coordinates.
///
/// # Arguments
/// * `depth_map` - HxW depth values
/// * `image` - HxW RGB image (optional, for coloring points)
/// * `intrinsics` - Camera intrinsic parameters
/// * `pose` - Camera pose (world-to-camera transform)
///
/// # Returns
/// A point cloud in world coordinates.
pub fn unproject_depth_map(
    depth_map: &[Vec<f32>],
    image: Option<&[Vec<[u8; 3]>]>,
    intrinsics: &CameraIntrinsics,
    pose: &CameraPose,
) -> PointCloud3D {
    let height = depth_map.len();
    if height == 0 {
        return PointCloud3D::new();
    }
    let width = depth_map[0].len();
    let c2w = pose.camera_to_world();

    let mut cloud = PointCloud3D::with_capacity(height * width);

    for (v, depth_row) in depth_map.iter().enumerate().take(height) {
        for (u, &depth) in depth_row.iter().enumerate().take(width) {
            if depth <= 0.0 || !depth.is_finite() {
                continue;
            }

            let pixel = Point2::new(u as f32, v as f32);
            let cam_point = intrinsics.unproject(&pixel, depth);
            let world_point = c2w.transform_point(&cam_point);

            let color = image
                .and_then(|img| img.get(v).and_then(|row| row.get(u)))
                .copied()
                .unwrap_or([128, 128, 128]);

            cloud.points.push(Point3DColored {
                position: world_point,
                color,
                normal: None,
            });
        }
    }

    cloud
}

/// Project 3D world points into a camera view, returning (pixel, depth) pairs
/// for points that are visible.
pub fn project_points(
    points: &[Point3<f32>],
    intrinsics: &CameraIntrinsics,
    pose: &CameraPose,
) -> Vec<(usize, Point2<f32>, f32)> {
    let mut results = Vec::new();

    for (i, world_point) in points.iter().enumerate() {
        let cam_point = pose.transform_point(world_point);
        if cam_point.z <= 0.0 {
            continue;
        }
        if let Some(pixel) = intrinsics.project(&cam_point)
            && intrinsics.is_in_bounds(&pixel)
        {
            results.push((i, pixel, cam_point.z));
        }
    }

    results
}

/// Compute visibility of 3D points from a given camera pose using frustum culling.
/// Returns indices of visible points.
pub fn frustum_cull(
    points: &[Point3<f32>],
    intrinsics: &CameraIntrinsics,
    pose: &CameraPose,
    near: f32,
    far: f32,
) -> Vec<usize> {
    let mut visible = Vec::new();

    for (i, world_point) in points.iter().enumerate() {
        let cam_point = pose.transform_point(world_point);
        if cam_point.z < near || cam_point.z > far {
            continue;
        }
        if let Some(pixel) = intrinsics.project(&cam_point)
            && intrinsics.is_in_bounds(&pixel)
        {
            visible.push(i);
        }
    }

    visible
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unproject_depth_map() {
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        // Create a simple 3x3 depth map
        let depth_map = vec![vec![5.0; 3]; 3];

        let cloud = unproject_depth_map(&depth_map, None, &intrinsics, &pose);
        assert_eq!(cloud.len(), 9);
    }

    #[test]
    fn test_project_points() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let pose = CameraPose::identity(0.0);

        let points = vec![
            Point3::new(0.0, 0.0, 5.0),  // in front
            Point3::new(0.0, 0.0, -5.0), // behind
        ];

        let results = project_points(&points, &intrinsics, &pose);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // first point visible
    }

    #[test]
    fn test_frustum_cull() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let pose = CameraPose::identity(0.0);

        let points = vec![
            Point3::new(0.0, 0.0, 5.0),   // visible
            Point3::new(0.0, 0.0, 0.05),  // too close
            Point3::new(0.0, 0.0, 200.0), // too far
            Point3::new(0.0, 0.0, -1.0),  // behind
        ];

        let visible = frustum_cull(&points, &intrinsics, &pose, 0.1, 100.0);
        assert_eq!(visible, vec![0]);
    }
}
