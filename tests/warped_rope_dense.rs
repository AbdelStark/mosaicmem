use mosaicmem::attention::WarpedRoPE;
use mosaicmem::camera::{CameraIntrinsics, CameraPose};
use mosaicmem::memory::store::{Patch3D, RetrievedPatch};
use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};

fn make_patch(timestamp: f64) -> RetrievedPatch {
    let token_coords = vec![
        (10.5, 10.5),
        (14.5, 10.5),
        (18.5, 10.5),
        (22.5, 10.5),
        (10.5, 14.5),
        (14.5, 14.5),
        (18.5, 14.5),
        (22.5, 14.5),
        (10.5, 18.5),
        (14.5, 18.5),
        (18.5, 18.5),
        (22.5, 18.5),
        (10.5, 22.5),
        (14.5, 22.5),
        (18.5, 22.5),
        (22.5, 22.5),
    ];

    RetrievedPatch {
        patch: Patch3D {
            id: 0,
            center: Point3::new(0.0, 0.0, 5.0),
            source_pose: CameraPose::identity(timestamp),
            source_frame: 0,
            source_timestamp: timestamp,
            source_depth: 5.0,
            source_rect: [8.0, 8.0, 16.0, 16.0],
            latent: vec![1.0; 16 * 4],
            latent_height: 4,
            latent_width: 4,
            token_coords: token_coords.clone(),
            depth_tile: Some(vec![5.0; token_coords.len()]),
            source_intrinsics: CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64),
            normal_estimate: None,
            latent_shape: (4, 4, 4),
        },
        target_position: Point2::new(16.5, 16.5),
        projected_footprint: token_coords
            .iter()
            .map(|(u, v)| Point2::new(*u, *v))
            .collect(),
        target_depth: 5.0,
        visibility_score: 1.0,
    }
}

#[test]
fn test_dense_reprojection_golden() {
    let wrope = WarpedRoPE::new(8, 64, 32);
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64);
    let target_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::new(0.25, 0.0, 0.0),
        UnitQuaternion::identity(),
    );

    let positions = wrope.compute_warped_positions(&[make_patch(0.0)], &target_pose, &intrinsics);
    assert_eq!(positions.len(), 16);

    for (idx, position) in positions.iter().enumerate() {
        let source_u = [10.5, 14.5, 18.5, 22.5][idx % 4];
        let source_v = [10.5, 14.5, 18.5, 22.5][idx / 4];
        assert!((position[0] - (source_u + 5.0)).abs() < 1e-4);
        assert!((position[1] - source_v).abs() < 1e-4);
        assert!((position[2] + 32.0).abs() < 1e-4);
    }
}

#[test]
fn test_intra_patch_positions_are_distinct() {
    let wrope = WarpedRoPE::new(8, 64, 32);
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64);
    let target_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::new(0.0, 0.0, 0.0),
        UnitQuaternion::from_euler_angles(0.0, 0.35, 0.0),
    );

    let positions = wrope.compute_warped_positions(&[make_patch(0.0)], &target_pose, &intrinsics);
    assert_eq!(positions.len(), 16);

    for i in 1..positions.len() {
        let dx = (positions[i][0] - positions[i - 1][0]).abs();
        let dy = (positions[i][1] - positions[i - 1][1]).abs();
        assert!(dx > 1e-4 || dy > 1e-4);
    }
}

#[test]
fn test_fractional_coordinate_stability() {
    let wrope = WarpedRoPE::new(8, 64, 32);
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64);
    let patch = make_patch(0.0);

    let base_pose = CameraPose::identity(1.0);
    let shifted_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::new(0.025, 0.0, 0.0),
        UnitQuaternion::identity(),
    );

    let base =
        wrope.compute_warped_positions(std::slice::from_ref(&patch), &base_pose, &intrinsics);
    let shifted = wrope.compute_warped_positions(&[patch], &shifted_pose, &intrinsics);

    for (base_pos, shifted_pos) in base.iter().zip(shifted.iter()) {
        assert!(((shifted_pos[0] - base_pos[0]) - 0.5).abs() < 0.05);
        assert!((shifted_pos[1] - base_pos[1]).abs() < 1e-4);
    }
}
