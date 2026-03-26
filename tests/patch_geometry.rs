use mosaicmem::camera::{CameraIntrinsics, CameraPose};
use mosaicmem::memory::{Patch3D, PatchGeometryError};
use nalgebra::Point3;

fn make_patch() -> Patch3D {
    Patch3D {
        id: 7,
        center: Point3::new(0.0, 0.0, 5.0),
        source_pose: CameraPose::identity(0.0),
        source_frame: 0,
        source_timestamp: 0.0,
        source_depth: 5.0,
        source_rect: [0.0, 0.0, 16.0, 16.0],
        latent: vec![0.5; 2 * 2 * 2],
        latent_height: 2,
        latent_width: 2,
        token_coords: vec![(4.0, 4.0), (12.0, 4.0), (4.0, 12.0), (12.0, 12.0)],
        depth_tile: Some(vec![5.0; 4]),
        source_intrinsics: CameraIntrinsics::new(100.0, 100.0, 8.0, 8.0, 16, 16),
        normal_estimate: None,
        latent_shape: (2, 2, 2),
    }
}

#[test]
fn test_patch_metadata_invariants() {
    let patch = make_patch();
    assert_eq!(patch.token_coords.len(), patch.latent_shape.1 * patch.latent_shape.2);
    assert!(patch.validate_geometry().is_ok());

    let mut wrong_coords = make_patch();
    wrong_coords.token_coords.pop();
    assert!(matches!(
        wrong_coords.validate_geometry(),
        Err(PatchGeometryError::TokenCoordsLen { .. })
    ));

    let mut wrong_depth = make_patch();
    wrong_depth.depth_tile = Some(vec![5.0; 3]);
    assert!(matches!(
        wrong_depth.validate_geometry(),
        Err(PatchGeometryError::DepthTileLen { .. })
    ));
}

#[test]
fn test_patch_metadata_backward_compatible_deserialization() {
    let legacy_json = r#"{
        "id": 1,
        "center": [0.0, 0.0, 5.0],
        "source_pose": {
            "timestamp": 0.0,
            "world_to_camera": {
                "tx": 0.0,
                "ty": 0.0,
                "tz": 0.0,
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0
            }
        },
        "source_frame": 0,
        "source_timestamp": 0.0,
        "source_depth": 5.0,
        "source_rect": [0.0, 0.0, 16.0, 16.0],
        "latent": [0.0, 1.0, 2.0, 3.0],
        "latent_height": 1,
        "latent_width": 1
    }"#;

    let patch: Patch3D = serde_json::from_str(legacy_json).unwrap();
    assert!(patch.token_coords.is_empty());
    assert!(patch.depth_tile.is_none());
    assert_eq!(patch.resolved_latent_shape(), (4, 1, 1));
    assert!(patch.validate_geometry().is_ok());
}
