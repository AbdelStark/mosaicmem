//! Runs real MosaicMem pipeline operations and collects results for the TUI.

use nalgebra::{Point3, Vector3};
use std::time::Instant;

use crate::camera::{CameraIntrinsics, CameraPose, CameraTrajectory};
use crate::diffusion::backbone::SyntheticBackbone;
use crate::diffusion::scheduler::DDPMScheduler;
use crate::diffusion::vae::{SyntheticVAE, VAE};
use crate::geometry::depth::{DepthEstimator, SyntheticDepthEstimator};
use crate::geometry::fusion::StreamingFusion;
use crate::memory::manipulation;
use crate::memory::retrieval::MemoryRetriever;
use crate::memory::store::{MemoryConfig, MosaicMemoryStore};
use crate::pipeline::autoregressive::AutoregressivePipeline;
use crate::pipeline::config::PipelineConfig;

// ── Results ──────────────────────────────────────────────────

pub struct DemoResult {
    pub cam_positions: Vec<[f32; 3]>,
    pub path_length: f32,
    pub num_windows: usize,
    pub num_frames: usize,
    pub total_values: usize,
    pub num_patches: usize,
    pub num_points: usize,
    pub num_keyframes: usize,
    pub total_tokens: usize,
    pub cloud_xz: Vec<(f64, f64)>,
    pub flip_patches: usize,
    pub erase_patches: usize,
    pub translate_patches: usize,
    pub elapsed_ms: f64,
}

pub struct CoverageFrame {
    pub coverage: f32,
    pub patches: usize,
}

pub struct CoverageResult {
    pub frames: Vec<CoverageFrame>,
    pub num_patches: usize,
    pub num_points: usize,
    pub total_tokens: usize,
    pub bbox_size: [f32; 3],
    pub elapsed_ms: f64,
}

pub struct BenchIteration {
    pub duration_ms: f64,
    pub num_patches: usize,
    pub num_points: usize,
}

pub struct BenchResult {
    pub iterations: Vec<BenchIteration>,
    pub avg_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub fps: f64,
    pub per_frame_ms: f64,
    pub num_frames: usize,
}

pub struct OpsResult {
    pub original_patches: usize,
    pub flip_patches: usize,
    pub erase_patches: usize,
    pub translate_patches: usize,
    pub splice_patches: usize,
    pub elapsed_ms: f64,
}

// ── Shared trajectory builder ────────────────────────────────

fn build_trajectory(num_frames: usize) -> CameraTrajectory {
    CameraTrajectory::new(
        (0..num_frames)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / num_frames as f32;
                let radius = 5.0;
                let eye = Point3::new(radius * angle.cos(), 1.0, radius * angle.sin());
                let target = Point3::new(0.0, 0.0, 0.0);
                let up = Vector3::new(0.0, 1.0, 0.0);
                CameraPose::look_at(i as f64 / 30.0, &eye, &target, &up)
            })
            .collect(),
    )
}

fn make_config(num_frames: usize, width: u32, height: u32, steps: usize) -> PipelineConfig {
    PipelineConfig {
        num_inference_steps: steps,
        width,
        height,
        window_size: num_frames.min(16),
        window_overlap: 2,
        ..Default::default()
    }
}

fn make_components() -> (
    SyntheticBackbone,
    DDPMScheduler,
    SyntheticVAE,
    SyntheticDepthEstimator,
    Vec<Vec<f32>>,
) {
    (
        SyntheticBackbone::new(0.1),
        DDPMScheduler::linear(1000, 1e-4, 0.02),
        SyntheticVAE::new(8, 4, 16),
        SyntheticDepthEstimator::new(5.0, 1.0),
        vec![vec![0.0f32; 64]; 10],
    )
}

// ── Demo ─────────────────────────────────────────────────────

pub fn run_demo() -> DemoResult {
    let num_frames = 32;
    let (width, height, steps) = (64, 64, 5);
    let start = Instant::now();

    let trajectory = build_trajectory(num_frames);
    let cam_positions: Vec<[f32; 3]> = trajectory
        .poses
        .iter()
        .map(|p| [p.position().x, p.position().y, p.position().z])
        .collect();
    let path_length = trajectory.path_length();

    let config = make_config(num_frames, width, height, steps);
    let (backbone, scheduler, vae, depth, text_emb) = make_components();

    let mut pipeline = AutoregressivePipeline::new(config);
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
        .expect("pipeline generate");

    let stats = pipeline.stats();

    // Extract point cloud XZ positions
    let cloud_xz: Vec<(f64, f64)> = pipeline
        .pipeline
        .fusion
        .global_cloud
        .points
        .iter()
        .map(|p| (p.position.x as f64, p.position.z as f64))
        .collect();

    // Memory manipulation
    let store = &pipeline.pipeline.memory_store;
    let flipped = manipulation::flip_vertical(store);
    let erased = manipulation::erase_region(store, &Point3::origin(), 2.0);
    let translated = manipulation::translate(store, &Vector3::new(10.0, 0.0, 0.0));

    DemoResult {
        cam_positions,
        path_length,
        num_windows: shapes.len(),
        num_frames,
        total_values: frames.len(),
        num_patches: stats.num_patches,
        num_points: stats.num_points,
        num_keyframes: stats.num_keyframes,
        total_tokens: stats.total_tokens,
        cloud_xz,
        flip_patches: flipped.len(),
        erase_patches: erased.len(),
        translate_patches: translated.len(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

// ── Coverage ─────────────────────────────────────────────────

pub fn run_coverage() -> CoverageResult {
    let num_frames = 32;
    let (width, height) = (64u32, 64u32);
    let start = Instant::now();

    let trajectory = build_trajectory(num_frames);
    let intrinsics = CameraIntrinsics::default_for_resolution(width, height);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);
    let vae = SyntheticVAE::new(8, 4, 16);

    let mut fusion = StreamingFusion::new(0.05);
    let mut store = MosaicMemoryStore::new(MemoryConfig {
        max_patches: 10000,
        top_k: 64,
        ..Default::default()
    });

    // Process keyframes
    let keyframes = trajectory.select_keyframes(0.5, 0.1);
    for &ki in &keyframes {
        if ki >= trajectory.len() {
            continue;
        }
        let pose = &trajectory.poses[ki];
        let frame_data = vec![128u8; (width * height * 3) as usize];
        let depth_map = depth_estimator
            .estimate_depth(&frame_data, width, height)
            .expect("depth");
        let _ = fusion.add_keyframe(
            &frame_data,
            width,
            height,
            &intrinsics,
            pose,
            &depth_estimator,
        );

        let frame_f32: Vec<f32> = frame_data.iter().map(|&b| b as f32 / 255.0).collect();
        let frame_shape = [1, 3, 1, height as usize, width as usize];
        let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape).expect("encode");
        store.insert_keyframe(
            ki,
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

    // Per-frame coverage
    let retriever = MemoryRetriever::new();
    let frames: Vec<CoverageFrame> = trajectory
        .poses
        .iter()
        .map(|pose| {
            let mosaic = retriever.retrieve(&store, pose, &intrinsics);
            CoverageFrame {
                coverage: mosaic.coverage_ratio(),
                patches: mosaic.num_patches(),
            }
        })
        .collect();

    // Bounding box
    let mut bbox_size = [0.0f32; 3];
    if !store.patches.is_empty() {
        let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &store.patches {
            let c = [p.center.x, p.center.y, p.center.z];
            for i in 0..3 {
                min[i] = min[i].min(c[i]);
                max[i] = max[i].max(c[i]);
            }
        }
        for i in 0..3 {
            bbox_size[i] = max[i] - min[i];
        }
    }

    CoverageResult {
        frames,
        num_patches: store.num_patches(),
        num_points: fusion.num_points(),
        total_tokens: store.total_tokens(),
        bbox_size,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

// ── Benchmark ────────────────────────────────────────────────

pub fn run_bench() -> BenchResult {
    let num_frames = 32;
    let (width, height, steps, iters) = (64, 64, 5, 5);

    let trajectory = build_trajectory(num_frames);
    let (backbone, scheduler, vae, depth, text_emb) = make_components();
    let config = make_config(num_frames, width, height, steps);

    let mut iterations = Vec::with_capacity(iters);

    for _ in 0..iters {
        let mut pipeline = AutoregressivePipeline::new(config.clone());
        let start = Instant::now();
        let _ = pipeline.generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            None,
        );
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let stats = pipeline.stats();
        iterations.push(BenchIteration {
            duration_ms: ms,
            num_patches: stats.num_patches,
            num_points: stats.num_points,
        });
    }

    let total_ms: f64 = iterations.iter().map(|i| i.duration_ms).sum();
    let avg_ms = total_ms / iters as f64;
    let min_ms = iterations
        .iter()
        .map(|i| i.duration_ms)
        .fold(f64::INFINITY, f64::min);
    let max_ms = iterations
        .iter()
        .map(|i| i.duration_ms)
        .fold(f64::NEG_INFINITY, f64::max);

    BenchResult {
        iterations,
        avg_ms,
        min_ms,
        max_ms,
        fps: num_frames as f64 / (avg_ms / 1000.0),
        per_frame_ms: avg_ms / num_frames as f64,
        num_frames,
    }
}

// ── Memory Operations ────────────────────────────────────────

pub fn run_ops() -> OpsResult {
    let start = Instant::now();
    let num_frames = 32;
    let (width, height, steps) = (64, 64, 5);

    let trajectory = build_trajectory(num_frames);
    let config = make_config(num_frames, width, height, steps);
    let (backbone, scheduler, vae, depth, text_emb) = make_components();

    let mut pipeline = AutoregressivePipeline::new(config);
    let _ = pipeline.generate(
        &trajectory,
        &text_emb,
        &backbone,
        &scheduler,
        &vae,
        &depth,
        None,
    );

    let store = &pipeline.pipeline.memory_store;
    let original = store.num_patches();

    let flipped = manipulation::flip_vertical(store);
    let erased = manipulation::erase_region(store, &Point3::origin(), 2.0);
    let translated = manipulation::translate(store, &Vector3::new(10.0, 0.0, 0.0));

    // Splice: combine two copies of the store
    let spliced = manipulation::splice_horizontal(store, store, 10.0);

    OpsResult {
        original_patches: original,
        flip_patches: flipped.len(),
        erase_patches: erased.len(),
        translate_patches: translated.len(),
        splice_patches: spliced.len(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}
