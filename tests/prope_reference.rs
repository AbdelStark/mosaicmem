use mosaicmem::attention::prope::{PRoPE, PRoPEOperator, ProjectiveTransform};
use mosaicmem::camera::{CameraIntrinsics, CameraPose};
use mosaicmem::tensor::{TensorLayout, TensorView};
use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};

fn assert_matrix_close(actual: &Matrix3<f64>, expected: &Matrix3<f64>, epsilon: f64) {
    for row in 0..3 {
        for col in 0..3 {
            assert!(
                (actual[(row, col)] - expected[(row, col)]).abs() <= epsilon,
                "matrix mismatch at ({row}, {col}): actual={}, expected={}",
                actual[(row, col)],
                expected[(row, col)],
            );
        }
    }
}

#[test]
fn test_projection_matrix_construction() {
    let intrinsics = CameraIntrinsics::new(200.0, 150.0, 20.0, 10.0, 64, 64);
    let pose = CameraPose::from_translation_rotation(
        0.0,
        Vector3::new(1.0, 2.0, 3.0),
        UnitQuaternion::identity(),
    );

    let projection = ProjectiveTransform::projection_matrix(&pose, &intrinsics);
    let expected = nalgebra::SMatrix::<f64, 3, 4>::from_row_slice(&[
        200.0, 0.0, 20.0, 260.0, 0.0, 150.0, 10.0, 330.0, 0.0, 0.0, 1.0, 3.0,
    ]);

    for row in 0..3 {
        for col in 0..4 {
            assert!((projection[(row, col)] - expected[(row, col)]).abs() <= 1e-9);
        }
    }
}

#[test]
fn test_relative_projective_transform_matches_reference() {
    let prope = PRoPE::new(8, 1, 1);
    let intrinsics = CameraIntrinsics::new(1.0, 1.0, 0.0, 0.0, 64, 64);
    let source_pose = CameraPose::identity(0.0);
    let target_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::zeros(),
        UnitQuaternion::from_euler_angles(0.0, std::f32::consts::FRAC_PI_2, 0.0),
    );

    let transform = prope
        .compute_projective_transform(&source_pose, &target_pose, &intrinsics, &intrinsics)
        .unwrap();
    let expected = Matrix3::new(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0);

    assert_matrix_close(&transform.relative, &expected, 1e-6);
}

#[test]
fn test_relative_projective_transform_identity() {
    let prope = PRoPE::new(8, 1, 1);
    let intrinsics = CameraIntrinsics::new(120.0, 110.0, 32.0, 24.0, 64, 64);
    let pose = CameraPose::look_at(
        0.0,
        &Point3::new(0.0, 0.0, 0.0),
        &Point3::new(0.0, 0.0, 5.0),
        &Vector3::y(),
    );

    let transform = prope
        .compute_projective_transform(&pose, &pose, &intrinsics, &intrinsics)
        .unwrap();
    assert_matrix_close(&transform.relative, &Matrix3::identity(), 1e-6);
}

#[test]
fn test_prope_attention_sensitivity() {
    let prope = PRoPE::new(8, 1, 1);
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64);
    let source_pose = CameraPose::identity(0.0);
    let target_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::zeros(),
        UnitQuaternion::from_euler_angles(0.0, 0.35, 0.0),
    );

    let identity_transform = prope
        .compute_projective_transform(&source_pose, &source_pose, &intrinsics, &intrinsics)
        .unwrap();
    let rotated_transform = prope
        .compute_projective_transform(&source_pose, &target_pose, &intrinsics, &intrinsics)
        .unwrap();

    let mut identity_queries = TensorView::from_shape_vec(
        &[2, 8],
        (0..16).map(|value| value as f32).collect(),
        TensorLayout::Flat(vec![2, 8]),
    )
    .unwrap();
    let mut identity_keys = TensorView::from_shape_vec(
        &[2, 8],
        (0..16).map(|value| value as f32 * 0.5).collect(),
        TensorLayout::Flat(vec![2, 8]),
    )
    .unwrap();
    let mut rotated_queries = identity_queries.clone();
    let mut rotated_keys = identity_keys.clone();

    prope
        .apply_to_attention(
            &mut identity_queries,
            &mut identity_keys,
            &identity_transform,
        )
        .unwrap();
    prope
        .apply_to_attention(&mut rotated_queries, &mut rotated_keys, &rotated_transform)
        .unwrap();

    assert_ne!(identity_queries.data(), rotated_queries.data());
    assert_ne!(identity_keys.data(), rotated_keys.data());
}
