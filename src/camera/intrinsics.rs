use nalgebra::{Matrix3, Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Camera intrinsic parameters (pinhole model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f32,
    /// Focal length in y (pixels).
    pub fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl CameraIntrinsics {
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, width: u32, height: u32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Build 3x3 intrinsic matrix K.
    pub fn matrix(&self) -> Matrix3<f32> {
        Matrix3::new(
            self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        )
    }

    /// Build inverse intrinsic matrix K^{-1}.
    pub fn inverse_matrix(&self) -> Matrix3<f32> {
        let inv_fx = 1.0 / self.fx;
        let inv_fy = 1.0 / self.fy;
        Matrix3::new(
            inv_fx,
            0.0,
            -self.cx * inv_fx,
            0.0,
            inv_fy,
            -self.cy * inv_fy,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Project a 3D point (in camera coords) to 2D pixel coordinates.
    pub fn project(&self, point: &Point3<f32>) -> Option<Point2<f32>> {
        if point.z <= 0.0 {
            return None;
        }
        let u = self.fx * point.x / point.z + self.cx;
        let v = self.fy * point.y / point.z + self.cy;
        Some(Point2::new(u, v))
    }

    /// Unproject a 2D pixel coordinate + depth to a 3D point in camera frame.
    pub fn unproject(&self, pixel: &Point2<f32>, depth: f32) -> Point3<f32> {
        let x = (pixel.x - self.cx) * depth / self.fx;
        let y = (pixel.y - self.cy) * depth / self.fy;
        Point3::new(x, y, depth)
    }

    /// Check if a pixel coordinate is within image bounds.
    pub fn is_in_bounds(&self, pixel: &Point2<f32>) -> bool {
        pixel.x >= 0.0
            && pixel.x < self.width as f32
            && pixel.y >= 0.0
            && pixel.y < self.height as f32
    }

    /// Create default intrinsics for a given resolution (assumes reasonable FOV).
    pub fn default_for_resolution(width: u32, height: u32) -> Self {
        let fx = width as f32;
        let fy = width as f32;
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        Self::new(fx, fy, cx, cy, width, height)
    }

    /// Compute normalized coordinates from pixel.
    pub fn normalize(&self, pixel: &Point2<f32>) -> Vector3<f32> {
        let k_inv = self.inverse_matrix();
        k_inv * Vector3::new(pixel.x, pixel.y, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_project_unproject_roundtrip() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let point = Point3::new(1.0, 2.0, 5.0);
        let pixel = intrinsics.project(&point).unwrap();
        let recovered = intrinsics.unproject(&pixel, point.z);
        assert_relative_eq!(recovered.x, point.x, epsilon = 1e-5);
        assert_relative_eq!(recovered.y, point.y, epsilon = 1e-5);
        assert_relative_eq!(recovered.z, point.z, epsilon = 1e-5);
    }

    #[test]
    fn test_behind_camera_returns_none() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let point = Point3::new(1.0, 2.0, -1.0);
        assert!(intrinsics.project(&point).is_none());
    }

    #[test]
    fn test_in_bounds() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        assert!(intrinsics.is_in_bounds(&Point2::new(100.0, 100.0)));
        assert!(!intrinsics.is_in_bounds(&Point2::new(-1.0, 100.0)));
        assert!(!intrinsics.is_in_bounds(&Point2::new(100.0, 480.0)));
    }
}
