use mosaicmem::camera::{CameraIntrinsics, CameraPose};
use mosaicmem::memory::retrieval::MemoryRetriever;
use mosaicmem::memory::store::{MemoryConfig, MosaicMemoryStore};
use nalgebra::{Point3, UnitQuaternion, Vector3};

fn make_pose(timestamp: f64, translation: Vector3<f32>, yaw_radians: f32) -> CameraPose {
    CameraPose::from_translation_rotation(
        timestamp,
        translation,
        UnitQuaternion::from_euler_angles(0.0, yaw_radians, 0.0),
    )
}

fn build_store() -> (MosaicMemoryStore, CameraIntrinsics) {
    let mut store = MosaicMemoryStore::new(MemoryConfig {
        max_patches: 128,
        top_k: 8,
        patch_size: 16,
        latent_patch_size: 2,
        ..Default::default()
    });
    let intrinsics = CameraIntrinsics::new(120.0, 120.0, 32.0, 32.0, 64, 64);
    let depth_map = vec![vec![5.0f32; 64]; 64];
    let latents = vec![0.5f32; 4 * 4 * 4];

    let forward_pose = CameraPose::identity(0.0);
    let left_pose = make_pose(1.0, Vector3::new(-1.5, 0.0, 0.0), 0.6);
    let right_pose = make_pose(2.0, Vector3::new(1.5, 0.0, 0.0), -0.6);

    store.insert_keyframe(
        0,
        0.0,
        &latents,
        4,
        4,
        4,
        &depth_map,
        &intrinsics,
        &forward_pose,
    );
    store.insert_keyframe(
        1,
        1.0,
        &latents,
        4,
        4,
        4,
        &depth_map,
        &intrinsics,
        &left_pose,
    );
    store.insert_keyframe(
        2,
        2.0,
        &latents,
        4,
        4,
        4,
        &depth_map,
        &intrinsics,
        &right_pose,
    );

    (store, intrinsics)
}

#[test]
fn test_frame_retrieval_varies_across_poses() {
    let (store, intrinsics) = build_store();
    let retriever = MemoryRetriever::with_diversity(8.0, 0.5);
    let poses = vec![
        CameraPose::look_at(
            3.0,
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 5.0),
            &Vector3::y_axis().into_inner(),
        ),
        CameraPose::look_at(
            4.0,
            &Point3::new(-2.0, 0.0, 0.0),
            &Point3::new(-2.5, 0.0, 5.0),
            &Vector3::y_axis().into_inner(),
        ),
        CameraPose::look_at(
            5.0,
            &Point3::new(2.0, 0.0, 0.0),
            &Point3::new(2.5, 0.0, 5.0),
            &Vector3::y_axis().into_inner(),
        ),
    ];

    let results = retriever
        .retrieve_window(&store, &poses, &intrinsics)
        .unwrap();
    assert_eq!(results.len(), 3);

    let ids: Vec<Vec<u64>> = results
        .iter()
        .map(|frame| frame.patches.iter().map(|p| p.patch.id).collect())
        .collect();

    assert_ne!(ids[0], ids[1]);
    assert_ne!(ids[1], ids[2]);
}

#[test]
fn test_frame_retrieval_coverage_varies_with_pan() {
    let (store, intrinsics) = build_store();
    let retriever = MemoryRetriever::new();
    let poses = vec![
        CameraPose::look_at(
            1.0,
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(0.0, 0.0, 5.0),
            &Vector3::y_axis().into_inner(),
        ),
        CameraPose::look_at(
            2.0,
            &Point3::new(0.0, 0.0, 0.0),
            &Point3::new(2.0, 0.0, 5.0),
            &Vector3::y_axis().into_inner(),
        ),
    ];

    let results = retriever
        .retrieve_window(&store, &poses, &intrinsics)
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_ne!(results[0].coverage_mask, results[1].coverage_mask);
}

#[test]
fn test_frame_retrieval_edge_cases() {
    let empty_store = MosaicMemoryStore::new(MemoryConfig::default());
    let retriever = MemoryRetriever::new();
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64);
    let pose = CameraPose::identity(0.0);

    let empty_results = retriever
        .retrieve_window(&empty_store, std::slice::from_ref(&pose), &intrinsics)
        .unwrap();
    assert_eq!(empty_results.len(), 1);
    assert!(empty_results[0].patches.is_empty());

    let (mut store, intrinsics) = build_store();
    store.config.top_k = 1;

    let single = retriever
        .retrieve_window(&store, std::slice::from_ref(&pose), &intrinsics)
        .unwrap();
    assert_eq!(single.len(), 1);
    assert!(single[0].patches.len() <= 1);

    let duplicate = retriever
        .retrieve_window(&store, &[pose.clone(), pose], &intrinsics)
        .unwrap();
    let first_ids: Vec<u64> = duplicate[0].patches.iter().map(|p| p.patch.id).collect();
    let second_ids: Vec<u64> = duplicate[1].patches.iter().map(|p| p.patch.id).collect();
    assert_eq!(first_ids, second_ids);
}
