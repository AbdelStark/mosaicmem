//! Camera intrinsics with f64-first projection math plus f32 convenience helpers
//! for the current synthetic geometry stack.

use nalgebra::{Matrix3, Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Camera intrinsic parameters (pinhole model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    /// Focal length in x (pixels).
    pub fx: f64,
    /// Focal length in y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl CameraIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
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
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    pub fn matrix_f32(&self) -> Matrix3<f32> {
        self.matrix().cast::<f32>()
    }

    /// Build inverse intrinsic matrix K^{-1}.
    pub fn inverse_matrix(&self) -> Matrix3<f64> {
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

    pub fn inverse_matrix_f32(&self) -> Matrix3<f32> {
        self.inverse_matrix().cast::<f32>()
    }

    pub fn project_f64(&self, point: &Point3<f64>) -> Option<Point2<f64>> {
        if point.z <= 0.0 {
            return None;
        }
        let u = self.fx * point.x / point.z + self.cx;
        let v = self.fy * point.y / point.z + self.cy;
        Some(Point2::new(u, v))
    }

    /// Project a 3D point (in camera coords) to 2D pixel coordinates.
    pub fn project(&self, point: &Point3<f32>) -> Option<Point2<f32>> {
        self.project_f64(&point.cast::<f64>())
            .map(|pixel| pixel.cast::<f32>())
    }

    pub fn unproject_f64(&self, pixel: &Point2<f64>, depth: f64) -> Point3<f64> {
        let x = (pixel.x - self.cx) * depth / self.fx;
        let y = (pixel.y - self.cy) * depth / self.fy;
        Point3::new(x, y, depth)
    }

    /// Unproject a 2D pixel coordinate + depth to a 3D point in camera frame.
    pub fn unproject(&self, pixel: &Point2<f32>, depth: f32) -> Point3<f32> {
        self.unproject_f64(&pixel.cast::<f64>(), f64::from(depth))
            .cast::<f32>()
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
        let fx = f64::from(width);
        let fy = f64::from(width);
        let cx = f64::from(width) / 2.0;
        let cy = f64::from(height) / 2.0;
        Self::new(fx, fy, cx, cy, width, height)
    }

    /// Compute normalized coordinates from pixel.
    pub fn normalize(&self, pixel: &Point2<f64>) -> Vector3<f64> {
        let k_inv = self.inverse_matrix();
        k_inv * Vector3::new(pixel.x, pixel.y, 1.0)
    }
}

impl Default for CameraIntrinsics {
    fn default() -> Self {
        Self::new(1.0, 1.0, 0.0, 0.0, 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_project_unproject_roundtrip() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let point = Point3::new(1.0_f64, 2.0, 5.0);
        let pixel = intrinsics.project_f64(&point).unwrap();
        let recovered = intrinsics.unproject_f64(&pixel, point.z);
        assert_relative_eq!(recovered.x, point.x, epsilon = 1e-5);
        assert_relative_eq!(recovered.y, point.y, epsilon = 1e-5);
        assert_relative_eq!(recovered.z, point.z, epsilon = 1e-5);
    }

    #[test]
    fn test_behind_camera_returns_none() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        let point = Point3::new(1.0_f64, 2.0, -1.0);
        assert!(intrinsics.project_f64(&point).is_none());
    }

    #[test]
    fn test_in_bounds() {
        let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
        assert!(intrinsics.is_in_bounds(&Point2::new(100.0, 100.0)));
        assert!(!intrinsics.is_in_bounds(&Point2::new(-1.0, 100.0)));
        assert!(!intrinsics.is_in_bounds(&Point2::new(100.0, 480.0)));
    }
}
