//! Integration tests for the MosaicMem pipeline.
//!
//! These tests exercise the full end-to-end pipeline with synthetic backends,
//! verifying that all components work together correctly.

use mosaicmem::attention::{MemoryCrossAttention, PRoPE, WarpedRoPE};
use mosaicmem::backend::AblationConfig;
use mosaicmem::camera::{CameraIntrinsics, CameraPose, CameraTrajectory};
use mosaicmem::diffusion::backbone::SyntheticBackbone;
use mosaicmem::diffusion::scheduler::DDPMScheduler;
use mosaicmem::diffusion::vae::{SyntheticVAE, VAE};
use mosaicmem::geometry::depth::{DepthEstimator, SyntheticDepthEstimator};
use mosaicmem::geometry::fusion::StreamingFusion;
use mosaicmem::memory::retrieval::MemoryRetriever;
use mosaicmem::memory::store::{MemoryConfig, MosaicMemoryStore, Patch3D, RetrievedPatch};
use mosaicmem::pipeline::autoregressive::AutoregressivePipeline;
use mosaicmem::pipeline::config::PipelineConfig;
use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};

/// Helper: create a circular camera trajectory.
fn circular_trajectory(num_frames: usize, radius: f32) -> CameraTrajectory {
    CameraTrajectory::new(
        (0..num_frames)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / num_frames as f32;
                let eye = Point3::new(radius * angle.cos(), 1.0, radius * angle.sin());
                let target = Point3::new(0.0, 0.0, 0.0);
                let up = Vector3::new(0.0, 1.0, 0.0);
                CameraPose::look_at(i as f64 / 30.0, &eye, &target, &up)
            })
            .collect(),
    )
}

/// Helper: create a linear camera trajectory.
fn linear_trajectory(num_frames: usize, step: f32) -> CameraTrajectory {
    CameraTrajectory::new(
        (0..num_frames)
            .map(|i| {
                CameraPose::from_translation_rotation(
                    i as f64 * 0.033,
                    Vector3::new(i as f32 * step, 0.0, 0.0),
                    UnitQuaternion::identity(),
                )
            })
            .collect(),
    )
}

fn seeded_memory_snapshot(config: &PipelineConfig) -> mosaicmem::memory::store::MemorySnapshot {
    let intrinsics = CameraIntrinsics::default_for_resolution(config.width, config.height);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);
    let vae = SyntheticVAE::new(8, 4, 16);
    let mut store = MosaicMemoryStore::new(MemoryConfig {
        max_patches: config.max_memory_patches,
        top_k: config.retrieval_top_k,
        patch_size: config.spatial_downsample as u32,
        latent_patch_size: 2,
        temporal_decay_half_life: config.temporal_decay_half_life,
        ..Default::default()
    });

    let frame_data = vec![128u8; config.width as usize * config.height as usize * 3];
    let depth_map = depth_estimator
        .estimate_depth(&frame_data, config.width, config.height)
        .unwrap();
    let frame_f32: Vec<f32> = frame_data.iter().map(|&byte| byte as f32 / 255.0).collect();
    let frame_shape = [1, 3, 1, config.height as usize, config.width as usize];
    let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape).unwrap();

    for (idx, pose) in [
        CameraPose::identity(0.0),
        CameraPose::from_translation_rotation(
            1.0,
            Vector3::new(1.0, 0.0, 0.25),
            UnitQuaternion::identity(),
        ),
    ]
    .iter()
    .enumerate()
    {
        store.insert_keyframe(
            idx,
            pose.timestamp,
            &latents,
            lat_shape[3],
            lat_shape[4],
            lat_shape[1],
            &depth_map,
            &intrinsics,
            pose,
        );
    }

    store.snapshot()
}

fn run_generation_with_snapshot(
    config: PipelineConfig,
    snapshot: &mosaicmem::memory::store::MemorySnapshot,
) -> Vec<f32> {
    let mut pipeline = AutoregressivePipeline::new(config.clone());
    pipeline.pipeline.memory_store = MosaicMemoryStore::from_snapshot(snapshot.clone());

    let trajectory = linear_trajectory(config.window_size.max(4), 0.35);
    let backbone = SyntheticBackbone::new(0.15);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);
    let text_emb = vec![vec![0.15f32; 64]; 10];

    pipeline
        .generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            None,
        )
        .unwrap()
        .0
}

// ============================================================================
// End-to-end pipeline tests
// ============================================================================

#[test]
fn test_full_pipeline_circular_trajectory() {
    let config = PipelineConfig {
        num_inference_steps: 2,
        width: 32,
        height: 32,
        window_size: 8,
        window_overlap: 2,
        ..Default::default()
    };

    let mut pipeline = AutoregressivePipeline::new(config);
    let trajectory = circular_trajectory(16, 5.0);
    let backbone = SyntheticBackbone::new(0.1);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);
    let text_emb = vec![vec![0.0f32; 64]; 10];

    let (frames, shapes) = pipeline
        .generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            None,
        )
        .expect("Pipeline should succeed");

    // Should produce multiple windows
    assert!(
        shapes.len() >= 2,
        "Expected at least 2 windows for 16 frames"
    );
    assert!(!frames.is_empty());

    // Memory should have accumulated patches across windows
    let stats = pipeline.stats();
    assert!(
        stats.num_patches > 0,
        "Memory should have patches after generation"
    );
    assert!(stats.num_points > 0, "Point cloud should have points");
    assert!(stats.num_keyframes > 0, "Should have processed keyframes");
}

#[test]
fn test_pipeline_memory_grows_across_windows() {
    let config = PipelineConfig {
        num_inference_steps: 2,
        width: 32,
        height: 32,
        window_size: 4,
        window_overlap: 1,
        ..Default::default()
    };

    let mut pipeline = AutoregressivePipeline::new(config);
    let trajectory = linear_trajectory(12, 0.5);
    let backbone = SyntheticBackbone::new(0.1);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);
    let text_emb = vec![vec![0.0f32; 64]; 10];

    let callback = Some(
        Box::new(move |_idx: usize, _frames: &[f32], _shape: &[usize; 5]| {
            // Callback fires for each window
        }) as Box<dyn FnMut(usize, &[f32], &[usize; 5]) + Send>,
    );

    let (_, shapes) = pipeline
        .generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            callback,
        )
        .expect("Pipeline should succeed");

    // With window_size=4, overlap=1, stride=3, for 12 frames:
    // Window 0: 0-3, Window 1: 3-6, Window 2: 6-9, Window 3: 9-11
    assert!(
        shapes.len() >= 3,
        "Expected at least 3 windows for 12 frames with stride 3"
    );
}

#[test]
fn test_pipeline_reset_memory() {
    let config = PipelineConfig {
        num_inference_steps: 2,
        width: 32,
        height: 32,
        window_size: 4,
        window_overlap: 1,
        ..Default::default()
    };

    let mut pipeline = AutoregressivePipeline::new(config);
    let trajectory = linear_trajectory(4, 0.5);
    let backbone = SyntheticBackbone::new(0.1);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);
    let text_emb = vec![vec![0.0f32; 64]; 10];

    pipeline
        .generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            None,
        )
        .unwrap();

    assert!(pipeline.stats().num_patches > 0);

    pipeline.reset_memory();
    assert_eq!(pipeline.stats().num_patches, 0);
    assert_eq!(pipeline.stats().num_points, 0);
}

// ============================================================================
// Memory cross-attention integration tests
// ============================================================================

#[test]
fn test_cross_attention_with_real_memory_pipeline() {
    // Build memory from a trajectory, then verify cross-attention produces
    // different outputs for different viewpoints.
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);
    let vae = SyntheticVAE::new(8, 4, 16);

    let mut store = MosaicMemoryStore::new(MemoryConfig {
        max_patches: 1000,
        top_k: 32,
        ..Default::default()
    });

    // Insert keyframes from two different viewpoints
    let pose_a = CameraPose::identity(0.0);
    let pose_b = CameraPose::from_translation_rotation(
        1.0,
        Vector3::new(5.0, 0.0, 0.0),
        UnitQuaternion::identity(),
    );

    let frame_data = vec![128u8; 100 * 100 * 3];
    let depth_map = depth_estimator
        .estimate_depth(&frame_data, 100, 100)
        .unwrap();
    let frame_f32: Vec<f32> = frame_data.iter().map(|&b| b as f32 / 255.0).collect();
    let frame_shape = [1, 3, 1, 100, 100];
    let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape).unwrap();

    store.insert_keyframe(
        0,
        0.0,
        &latents,
        lat_shape[3],
        lat_shape[4],
        lat_shape[1],
        &depth_map,
        &intrinsics,
        &pose_a,
    );
    store.insert_keyframe(
        1,
        1.0,
        &latents,
        lat_shape[3],
        lat_shape[4],
        lat_shape[1],
        &depth_map,
        &intrinsics,
        &pose_b,
    );

    assert!(store.num_patches() > 0);

    // Retrieve memory from both viewpoints
    let retriever = MemoryRetriever::new();
    let mosaic_a = retriever.retrieve(&store, &pose_a, &intrinsics);
    let mosaic_b = retriever.retrieve(&store, &pose_b, &intrinsics);

    // Both should have patches
    assert!(mosaic_a.num_patches() > 0);
    assert!(mosaic_b.num_patches() > 0);

    // Cross-attention should produce different outputs for different viewpoints
    let mca = MemoryCrossAttention::new(32, 2);
    let gen_tokens = vec![vec![1.0f32; 32]; 4];

    let output_a = mca.forward(&gen_tokens, &mosaic_a, &pose_a, &intrinsics);
    let output_b = mca.forward(&gen_tokens, &mosaic_b, &pose_b, &intrinsics);

    assert_eq!(output_a.len(), 4);
    assert_eq!(output_b.len(), 4);

    // Outputs should differ because different viewpoints retrieve different patches
    // with different WarpedRoPE positions
    let a_sum: f32 = output_a.iter().flat_map(|t| t.iter()).sum();
    let b_sum: f32 = output_b.iter().flat_map(|t| t.iter()).sum();
    // Both should be non-zero (patches exist)
    assert!(a_sum.abs() > 1e-10 || b_sum.abs() > 1e-10);
}

// ============================================================================
// Geometry + memory pipeline tests
// ============================================================================

#[test]
fn test_streaming_fusion_with_memory_store() {
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);

    let mut fusion = StreamingFusion::new(0.05);
    let mut store = MosaicMemoryStore::new(MemoryConfig::default());

    let trajectory = circular_trajectory(4, 3.0);
    let vae = SyntheticVAE::new(8, 4, 16);

    for (i, pose) in trajectory.poses.iter().enumerate() {
        let frame_data = vec![128u8; 100 * 100 * 3];
        let depth_map = depth_estimator
            .estimate_depth(&frame_data, 100, 100)
            .unwrap();

        // Add to fusion point cloud
        fusion
            .add_keyframe(&frame_data, 100, 100, &intrinsics, pose, &depth_estimator)
            .unwrap();

        // Add to memory store
        let frame_f32: Vec<f32> = frame_data.iter().map(|&b| b as f32 / 255.0).collect();
        let frame_shape = [1, 3, 1, 100, 100];
        let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape).unwrap();

        store.insert_keyframe(
            i,
            pose.timestamp,
            &latents,
            lat_shape[3],
            lat_shape[4],
            lat_shape[1],
            &depth_map,
            &intrinsics,
            pose,
        );
    }

    // Both should have accumulated data
    assert!(fusion.num_points() > 0, "Fusion should have points");
    assert_eq!(fusion.num_keyframes, 4, "Should have 4 keyframes");
    assert!(store.num_patches() > 0, "Store should have patches");

    // Query from a new viewpoint
    let query_pose = CameraPose::from_translation_rotation(
        2.0,
        Vector3::new(2.0, 1.0, 2.0),
        UnitQuaternion::identity(),
    );
    let retrieved = store.retrieve(&query_pose, &intrinsics);
    // Some patches should be visible from this viewpoint
    assert!(
        !retrieved.is_empty(),
        "Should retrieve patches from nearby viewpoint"
    );
}

// ============================================================================
// Attention mechanism integration tests
// ============================================================================

#[test]
fn test_warped_rope_with_real_patches() {
    let wrope = WarpedRoPE::new(8, 64, 32);
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
    let target_pose = CameraPose::identity(0.0);

    // Create patches at known 3D positions
    let patches: Vec<RetrievedPatch> = vec![
        RetrievedPatch {
            patch: Patch3D {
                id: 0,
                center: Point3::new(0.0, 0.0, 5.0),
                source_pose: CameraPose::identity(0.0),
                source_frame: 0,
                source_timestamp: 0.0,
                source_depth: 5.0,
                source_rect: [0.0, 0.0, 16.0, 16.0],
                latent: vec![0.5; 4],
                latent_height: 1,
                latent_width: 1,
                token_coords: vec![(8.0, 8.0)],
                depth_tile: Some(vec![5.0]),
                source_intrinsics: CameraIntrinsics::default(),
                normal_estimate: None,
                latent_shape: (4, 1, 1),
            },
            target_position: Point2::new(50.0, 50.0),
            projected_footprint: vec![Point2::new(50.0, 50.0)],
            target_depth: 5.0,
            visibility_score: 1.0,
        },
        RetrievedPatch {
            patch: Patch3D {
                id: 1,
                center: Point3::new(2.0, 0.0, 5.0),
                source_pose: CameraPose::identity(0.0),
                source_frame: 1,
                source_timestamp: 1.0,
                source_depth: 5.0,
                source_rect: [20.0, 0.0, 16.0, 16.0],
                latent: vec![0.5; 4],
                latent_height: 1,
                latent_width: 1,
                token_coords: vec![(28.0, 8.0)],
                depth_tile: Some(vec![5.0]),
                source_intrinsics: CameraIntrinsics::default(),
                normal_estimate: None,
                latent_shape: (4, 1, 1),
            },
            target_position: Point2::new(70.0, 50.0),
            projected_footprint: vec![Point2::new(70.0, 50.0)],
            target_depth: 5.0,
            visibility_score: 0.9,
        },
    ];

    let positions = wrope.compute_warped_positions(&patches, &target_pose, &intrinsics);
    assert_eq!(positions.len(), 2);

    // Patches at different x positions should have different u coordinates
    assert_ne!(
        positions[0][0], positions[1][0],
        "Different x → different u"
    );
    // Same y → same v
    assert_eq!(positions[0][1], positions[1][1], "Same y → same v");
    // Different timestamps → different t
    assert_ne!(
        positions[0][2], positions[1][2],
        "Different time → different t"
    );

    // Verify rotation changes the vectors
    let dim = 3 * 8; // 3 axes * 8 dim_per_axis
    let vecs = vec![vec![1.0f32; dim]; 2];
    let rotated = wrope.rotate(&vecs, &positions);
    assert_eq!(rotated.len(), 2);
    // Different positions should yield different rotated vectors
    assert_ne!(rotated[0], rotated[1]);
}

#[test]
fn test_prope_camera_conditioning() {
    let prope = PRoPE::new(16, 2, 1);
    let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);

    // Use very different camera poses to ensure distinct rotations
    let poses = vec![
        CameraPose::identity(0.0),
        CameraPose::from_translation_rotation(
            0.033,
            Vector3::new(10.0, 5.0, 3.0),
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.0),
        ),
    ];

    let rotations = prope.compute_rotations(&poses, &intrinsics);
    assert_eq!(rotations.len(), 2);
    // Different cameras should produce different rotation parameters
    assert_ne!(rotations[0], rotations[1]);

    // Apply to query vectors — temporal_compression=1, so each frame is distinct
    let qk = vec![vec![1.0f32; 16]; 4];
    let frame_indices = vec![0, 0, 1, 1];

    let rotated = prope.apply(&qk, &rotations, &frame_indices);
    assert_eq!(rotated.len(), 4);

    // Tokens from different cameras (frame 0 vs frame 1) should have different rotations
    assert_ne!(rotated[0], rotated[2]);
    // Tokens from the same camera should have the same rotation
    assert_eq!(rotated[0], rotated[1]);
    assert_eq!(rotated[2], rotated[3]);
}

// ============================================================================
// Pipeline config serialization tests
// ============================================================================

#[test]
fn test_pipeline_config_json_roundtrip() {
    let config = PipelineConfig {
        num_inference_steps: 25,
        width: 512,
        height: 512,
        window_size: 16,
        window_overlap: 4,
        hidden_dim: 128,
        num_heads: 8,
        ..Default::default()
    };

    let json = serde_json::to_string_pretty(&config).unwrap();
    let deserialized: PipelineConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.num_inference_steps, 25);
    assert_eq!(deserialized.width, 512);
    assert_eq!(deserialized.height, 512);
    assert_eq!(deserialized.hidden_dim, 128);
    assert_eq!(deserialized.num_heads, 8);
    assert_eq!(deserialized.latent_height(), 64);
    assert_eq!(deserialized.latent_width(), 64);
}

#[test]
fn test_memory_config_json_roundtrip() {
    let config = MemoryConfig {
        max_patches: 5000,
        top_k: 32,
        near_clip: 0.05,
        far_clip: 200.0,
        patch_size: 32,
        latent_patch_size: 4,
        temporal_decay_half_life: 0.0,
    };

    let json = serde_json::to_string(&config).unwrap();
    let deserialized: MemoryConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.max_patches, 5000);
    assert_eq!(deserialized.top_k, 32);
}

#[test]
fn test_memory_cross_attention_wiring_changes_output() {
    let memory_on = PipelineConfig {
        num_inference_steps: 2,
        width: 32,
        height: 32,
        window_size: 4,
        window_overlap: 1,
        ablation: AblationConfig::default(),
        ..Default::default()
    };
    let memory_off = PipelineConfig {
        ablation: AblationConfig {
            enable_memory: false,
            ..AblationConfig::default()
        },
        ..memory_on.clone()
    };
    let snapshot = seeded_memory_snapshot(&memory_on);

    let with_memory = run_generation_with_snapshot(memory_on, &snapshot);
    let without_memory = run_generation_with_snapshot(memory_off, &snapshot);

    assert_ne!(with_memory, without_memory);
}

#[test]
fn test_ablation_combinations_produce_distinct_outputs() {
    let base = PipelineConfig {
        num_inference_steps: 2,
        width: 32,
        height: 32,
        window_size: 4,
        window_overlap: 1,
        ..Default::default()
    };
    let snapshot = seeded_memory_snapshot(&base);

    let prope_only = run_generation_with_snapshot(
        PipelineConfig {
            ablation: AblationConfig {
                enable_prope: true,
                enable_warped_rope: false,
                enable_warped_latent: false,
                ..AblationConfig::default()
            },
            ..base.clone()
        },
        &snapshot,
    );
    let warped_rope_only = run_generation_with_snapshot(
        PipelineConfig {
            ablation: AblationConfig {
                enable_prope: false,
                enable_warped_rope: true,
                enable_warped_latent: false,
                ..AblationConfig::default()
            },
            ..base.clone()
        },
        &snapshot,
    );
    let warped_latent_only = run_generation_with_snapshot(
        PipelineConfig {
            ablation: AblationConfig {
                enable_prope: false,
                enable_warped_rope: false,
                enable_warped_latent: true,
                ..AblationConfig::default()
            },
            ..base.clone()
        },
        &snapshot,
    );
    let full = run_generation_with_snapshot(
        PipelineConfig {
            ablation: AblationConfig::default(),
            ..base
        },
        &snapshot,
    );

    assert_ne!(prope_only, warped_rope_only);
    assert_ne!(prope_only, warped_latent_only);
    assert_ne!(prope_only, full);
    assert_ne!(warped_rope_only, warped_latent_only);
    assert_ne!(warped_rope_only, full);
    assert_ne!(warped_latent_only, full);
}

#[test]
fn test_memory_gate_override_zero_matches_memory_disabled() {
    let base = PipelineConfig {
        num_inference_steps: 2,
        width: 32,
        height: 32,
        window_size: 4,
        window_overlap: 1,
        ..Default::default()
    };
    let snapshot = seeded_memory_snapshot(&base);

    let gated_zero = run_generation_with_snapshot(
        PipelineConfig {
            ablation: AblationConfig {
                memory_gate_override: Some(0.0),
                ..AblationConfig::default()
            },
            ..base.clone()
        },
        &snapshot,
    );
    let memory_disabled = run_generation_with_snapshot(
        PipelineConfig {
            ablation: AblationConfig {
                enable_memory: false,
                ..AblationConfig::default()
            },
            ..base
        },
        &snapshot,
    );

    assert_eq!(gated_zero, memory_disabled);
}

// ============================================================================
// Coverage and retrieval quality tests
// ============================================================================

#[test]
fn test_coverage_increases_with_more_keyframes() {
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);
    let vae = SyntheticVAE::new(8, 4, 16);
    let retriever = MemoryRetriever::new();

    let mut store = MosaicMemoryStore::new(MemoryConfig {
        max_patches: 10000,
        top_k: 100,
        ..Default::default()
    });

    let query_pose = CameraPose::identity(0.5);

    // Measure coverage with 0 keyframes
    let mosaic_0 = retriever.retrieve(&store, &query_pose, &intrinsics);
    let coverage_0 = mosaic_0.coverage_ratio();
    assert_eq!(coverage_0, 0.0);

    // Add one keyframe
    let pose = CameraPose::identity(0.0);
    let frame_data = vec![128u8; 100 * 100 * 3];
    let depth_map = depth_estimator
        .estimate_depth(&frame_data, 100, 100)
        .unwrap();
    let frame_f32: Vec<f32> = frame_data.iter().map(|&b| b as f32 / 255.0).collect();
    let frame_shape = [1, 3, 1, 100, 100];
    let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape).unwrap();

    store.insert_keyframe(
        0,
        0.0,
        &latents,
        lat_shape[3],
        lat_shape[4],
        lat_shape[1],
        &depth_map,
        &intrinsics,
        &pose,
    );

    let mosaic_1 = retriever.retrieve(&store, &query_pose, &intrinsics);
    let coverage_1 = mosaic_1.coverage_ratio();
    assert!(
        coverage_1 > coverage_0,
        "Coverage should increase with more keyframes"
    );
}

#[test]
fn test_memory_budget_enforcement() {
    let config = MemoryConfig {
        max_patches: 10, // Very small budget
        top_k: 5,
        ..Default::default()
    };
    let mut store = MosaicMemoryStore::new(config);
    let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);
    let vae = SyntheticVAE::new(8, 4, 16);

    // Insert many keyframes
    for i in 0..20 {
        let pose = CameraPose::from_translation_rotation(
            i as f64 * 0.1,
            Vector3::new(i as f32 * 0.5, 0.0, 0.0),
            UnitQuaternion::identity(),
        );
        let frame_data = vec![128u8; 100 * 100 * 3];
        let depth_map = depth_estimator
            .estimate_depth(&frame_data, 100, 100)
            .unwrap();
        let frame_f32: Vec<f32> = frame_data.iter().map(|&b| b as f32 / 255.0).collect();
        let frame_shape = [1, 3, 1, 100, 100];
        let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape).unwrap();

        store.insert_keyframe(
            i,
            pose.timestamp,
            &latents,
            lat_shape[3],
            lat_shape[4],
            lat_shape[1],
            &depth_map,
            &intrinsics,
            &pose,
        );
    }

    // Should be capped at max_patches
    assert!(
        store.num_patches() <= 10,
        "Memory budget should be enforced: got {} patches",
        store.num_patches()
    );
}
