use mosaicmem::attention::{DenseWarpOperator, WarpOperator};
use mosaicmem::camera::{CameraIntrinsics, CameraPose};
use mosaicmem::memory::store::Patch3D;
use mosaicmem::tensor::{TensorLayout, TensorView};
use nalgebra::{Point3, UnitQuaternion, Vector3};

fn dense_patch(depth_tile: Vec<f32>) -> Patch3D {
    let token_coords = vec![
        (10.0, 10.0),
        (14.0, 10.0),
        (18.0, 10.0),
        (22.0, 10.0),
        (10.0, 14.0),
        (14.0, 14.0),
        (18.0, 14.0),
        (22.0, 14.0),
        (10.0, 18.0),
        (14.0, 18.0),
        (18.0, 18.0),
        (22.0, 18.0),
        (10.0, 22.0),
        (14.0, 22.0),
        (18.0, 22.0),
        (22.0, 22.0),
    ];

    Patch3D {
        id: 0,
        center: Point3::new(0.0, 0.0, 5.0),
        source_pose: CameraPose::identity(0.0),
        source_frame: 0,
        source_timestamp: 0.0,
        source_depth: 5.0,
        source_rect: [8.0, 8.0, 16.0, 16.0],
        latent: (0..32).map(|value| value as f32).collect(),
        latent_height: 4,
        latent_width: 4,
        token_coords,
        depth_tile: Some(depth_tile),
        source_intrinsics: CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64),
        normal_estimate: None,
        latent_shape: (2, 4, 4),
    }
}

#[test]
fn test_dense_warp_geometric_golden() {
    let operator = DenseWarpOperator;
    let depth_tile = vec![
        5.0, 5.5, 6.0, 6.5, 5.0, 5.5, 6.0, 6.5, 5.0, 5.5, 6.0, 6.5, 5.0, 5.5, 6.0, 6.5,
    ];
    let patch = dense_patch(depth_tile.clone());
    let target_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::new(0.25, 0.0, 0.0),
        UnitQuaternion::identity(),
    );

    let grid = operator
        .compute_warp_grid(&patch, &target_pose, &patch.source_intrinsics)
        .unwrap();

    for (idx, &(u, v)) in grid.target_coords.iter().enumerate() {
        let source_u = patch.token_coords[idx].0;
        let source_v = patch.token_coords[idx].1;
        let expected_u = source_u + 25.0 / depth_tile[idx];
        assert!((u - expected_u).abs() < 1e-4, "u mismatch at token {idx}");
        assert!((v - source_v).abs() < 1e-4, "v mismatch at token {idx}");
    }
}

#[test]
fn test_dense_warp_identity() {
    let operator = DenseWarpOperator;
    let patch = dense_patch(vec![5.0; 16]);
    let source = TensorView::from_shape_vec(
        &[4, 4, 2],
        patch.latent.clone(),
        TensorLayout::Flat(vec![4, 4, 2]),
    )
    .unwrap();

    let grid = operator
        .compute_warp_grid(&patch, &CameraPose::identity(0.0), &patch.source_intrinsics)
        .unwrap();
    let (warped, valid_mask) = operator.apply_warp(&source, &grid).unwrap();

    let warped_values = warped.data().as_slice_memory_order().unwrap();
    let source_values = source.data().as_slice_memory_order().unwrap();
    for (warped_value, source_value) in warped_values.iter().zip(source_values.iter()) {
        assert!((warped_value - source_value).abs() < 1e-5);
    }
    assert!(
        valid_mask
            .data()
            .iter()
            .all(|value| (*value - 1.0).abs() < 1e-6)
    );
}

#[test]
fn test_invalid_samples_are_masked() {
    let operator = DenseWarpOperator;
    let depth_tile = vec![
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0,
    ];
    let patch = dense_patch(depth_tile);
    let source = TensorView::from_shape_vec(
        &[4, 4, 2],
        patch.latent.clone(),
        TensorLayout::Flat(vec![4, 4, 2]),
    )
    .unwrap();
    let target_pose = CameraPose::from_translation_rotation(
        1.0,
        Vector3::new(0.0, 0.0, -6.0),
        UnitQuaternion::identity(),
    );

    let grid = operator
        .compute_warp_grid(&patch, &target_pose, &patch.source_intrinsics)
        .unwrap();
    assert!(grid.valid_mask.iter().any(|&valid| !valid));
    assert!(grid.valid_mask.iter().any(|&valid| valid));

    let (warped, valid_mask) = operator.apply_warp(&source, &grid).unwrap();
    let mask_values = valid_mask.data().as_slice_memory_order().unwrap();
    assert!(mask_values.iter().any(|&value| value < 0.5));
    assert!(mask_values.iter().any(|&value| value > 0.5));
    assert!(
        warped
            .data()
            .as_slice_memory_order()
            .unwrap()
            .iter()
            .any(|&value| value.abs() > 1e-6)
    );
}
