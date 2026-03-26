use approx::assert_relative_eq;
use mosaicmem::camera::CameraIntrinsics;
use nalgebra::Point3;

#[test]
fn test_camera_intrinsics_project_unproject_roundtrip() {
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
    let point = Point3::new(1.25_f64, -0.75, 4.5);

    let pixel = intrinsics.project_f64(&point).unwrap();
    let recovered = intrinsics.unproject_f64(&pixel, point.z);

    assert_relative_eq!(recovered.x, point.x, epsilon = 1e-9);
    assert_relative_eq!(recovered.y, point.y, epsilon = 1e-9);
    assert_relative_eq!(recovered.z, point.z, epsilon = 1e-9);
}
