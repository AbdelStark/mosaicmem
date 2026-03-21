use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use mosaicmem_rs::camera::{CameraPose, CameraTrajectory};
use mosaicmem_rs::diffusion::backbone::SyntheticBackbone;
use mosaicmem_rs::diffusion::scheduler::DDPMScheduler;
use mosaicmem_rs::diffusion::vae::SyntheticVAE;
use mosaicmem_rs::geometry::depth::SyntheticDepthEstimator;
use mosaicmem_rs::memory::manipulation;
use mosaicmem_rs::memory::store::{MemoryConfig, MosaicMemoryStore};
use mosaicmem_rs::pipeline::autoregressive::AutoregressivePipeline;
use mosaicmem_rs::pipeline::config::PipelineConfig;

#[derive(Parser)]
#[command(
    name = "mosaicmem-rs",
    about = "MosaicMem: Hybrid Spatial Memory for Video World Models",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate video frames using the MosaicMem pipeline.
    Generate {
        /// Camera trajectory JSON file.
        #[arg(short, long)]
        trajectory: PathBuf,

        /// Text prompt for generation.
        #[arg(short, long, default_value = "a scene")]
        prompt: String,

        /// Output directory for generated frames.
        #[arg(short, long, default_value = "output")]
        output: PathBuf,

        /// Video width.
        #[arg(long, default_value_t = 256)]
        width: u32,

        /// Video height.
        #[arg(long, default_value_t = 256)]
        height: u32,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 50)]
        steps: usize,

        /// Frames per generation window.
        #[arg(long, default_value_t = 16)]
        window_size: usize,

        /// Overlap frames between windows.
        #[arg(long, default_value_t = 4)]
        window_overlap: usize,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },

    /// Visualize memory statistics and point cloud info.
    Visualize {
        /// Camera trajectory JSON file.
        #[arg(short, long)]
        trajectory: PathBuf,
    },

    /// Splice two memory stores together.
    Splice {
        /// First memory trajectory.
        #[arg(long)]
        trajectory_a: PathBuf,

        /// Second memory trajectory.
        #[arg(long)]
        trajectory_b: PathBuf,

        /// Layout: "horizontal" or "vertical".
        #[arg(long, default_value = "horizontal")]
        layout: String,

        /// Spatial offset between scenes.
        #[arg(long, default_value_t = 10.0)]
        offset: f32,
    },

    /// Inspect a trajectory and show memory/geometry analysis.
    Inspect {
        /// Camera trajectory JSON file.
        #[arg(short, long)]
        trajectory: PathBuf,

        /// Video width (for projection calculations).
        #[arg(long, default_value_t = 256)]
        width: u32,

        /// Video height (for projection calculations).
        #[arg(long, default_value_t = 256)]
        height: u32,

        /// Show per-frame memory coverage analysis.
        #[arg(long)]
        coverage: bool,
    },

    /// Dump or load pipeline configuration as JSON.
    ShowConfig {
        /// Optional JSON config file to load. If omitted, prints default config.
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Run a demo with synthetic data (no ONNX models required).
    Demo {
        /// Number of frames to generate.
        #[arg(long, default_value_t = 32)]
        num_frames: usize,

        /// Video width.
        #[arg(long, default_value_t = 64)]
        width: u32,

        /// Video height.
        #[arg(long, default_value_t = 64)]
        height: u32,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 5)]
        steps: usize,
    },

    /// Export a point cloud from a trajectory to PLY format.
    ExportPly {
        /// Camera trajectory JSON file.
        #[arg(short, long)]
        trajectory: PathBuf,

        /// Output PLY file path.
        #[arg(short, long)]
        output: PathBuf,

        /// Video width (for depth estimation).
        #[arg(long, default_value_t = 100)]
        width: u32,

        /// Video height (for depth estimation).
        #[arg(long, default_value_t = 100)]
        height: u32,

        /// Voxel size for downsampling.
        #[arg(long, default_value_t = 0.05)]
        voxel_size: f32,
    },

    /// Run a performance benchmark of the pipeline.
    Bench {
        /// Number of frames.
        #[arg(long, default_value_t = 32)]
        num_frames: usize,

        /// Video width.
        #[arg(long, default_value_t = 64)]
        width: u32,

        /// Video height.
        #[arg(long, default_value_t = 64)]
        height: u32,

        /// Number of denoising steps.
        #[arg(long, default_value_t = 5)]
        steps: usize,

        /// Number of benchmark iterations.
        #[arg(long, default_value_t = 3)]
        iterations: usize,
    },
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("mosaicmem_rs=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            trajectory,
            prompt,
            output,
            width,
            height,
            steps,
            window_size,
            window_overlap,
            seed,
        } => {
            if let Err(e) = cmd_generate(&GenerateArgs {
                trajectory_path: &trajectory,
                prompt: &prompt,
                output_dir: &output,
                width,
                height,
                steps,
                window_size,
                window_overlap,
                seed,
            }) {
                error!("Generation failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Visualize { trajectory } => {
            if let Err(e) = cmd_visualize(&trajectory) {
                error!("Visualization failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Splice {
            trajectory_a,
            trajectory_b,
            layout,
            offset,
        } => {
            if let Err(e) = cmd_splice(&trajectory_a, &trajectory_b, &layout, offset) {
                error!("Splice failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Inspect {
            trajectory,
            width,
            height,
            coverage,
        } => {
            if let Err(e) = cmd_inspect(&trajectory, width, height, coverage) {
                error!("Inspect failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::ShowConfig { config } => {
            if let Err(e) = cmd_show_config(config.as_deref()) {
                error!("Config failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Demo {
            num_frames,
            width,
            height,
            steps,
        } => {
            if let Err(e) = cmd_demo(num_frames, width, height, steps) {
                error!("Demo failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::ExportPly {
            trajectory,
            output,
            width,
            height,
            voxel_size,
        } => {
            if let Err(e) = cmd_export_ply(&trajectory, &output, width, height, voxel_size) {
                error!("PLY export failed: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Bench {
            num_frames,
            width,
            height,
            steps,
            iterations,
        } => {
            if let Err(e) = cmd_bench(num_frames, width, height, steps, iterations) {
                error!("Benchmark failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

struct GenerateArgs<'a> {
    trajectory_path: &'a Path,
    prompt: &'a str,
    output_dir: &'a Path,
    width: u32,
    height: u32,
    steps: usize,
    window_size: usize,
    window_overlap: usize,
    seed: u64,
}

fn cmd_generate(args: &GenerateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let GenerateArgs {
        trajectory_path,
        prompt,
        output_dir,
        width,
        height,
        steps,
        window_size,
        window_overlap,
        seed,
    } = args;
    info!("Loading trajectory from {:?}", trajectory_path);
    let trajectory = CameraTrajectory::load_json(trajectory_path)?;
    info!("Loaded {} poses", trajectory.len());

    std::fs::create_dir_all(output_dir)?;

    let config = PipelineConfig {
        num_inference_steps: *steps,
        width: *width,
        height: *height,
        window_size: *window_size,
        window_overlap: *window_overlap,
        seed: *seed,
        ..Default::default()
    };

    info!("Initializing pipeline (using synthetic models for MVP)");
    let mut pipeline = AutoregressivePipeline::new(config);

    let backbone = SyntheticBackbone::new(0.1);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);

    // Simple text embedding (placeholder)
    let text_emb = vec![vec![0.0f32; 64]; 10];

    info!("Generating with prompt: \"{}\"", prompt);
    let output_dir_clone = output_dir.to_path_buf();
    let (frames, shapes) = pipeline.generate(
        &trajectory,
        &text_emb,
        &backbone,
        &scheduler,
        &vae,
        &depth,
        Some(Box::new(move |window_idx, _frames, shape| {
            info!(
                "Window {} complete: [{}, {}, {}, {}, {}]",
                window_idx, shape[0], shape[1], shape[2], shape[3], shape[4]
            );
        })),
    )?;

    info!(
        "Generation complete: {} total values, {} windows",
        frames.len(),
        shapes.len()
    );
    info!("Final stats: {}", pipeline.stats());

    // Write frames as PNG images
    let mut frame_offset = 0;
    for (win_idx, shape) in shapes.iter().enumerate() {
        let [_b, c, t, h, w] = *shape;
        let frame_size = c * h * w;
        for f_idx in 0..t {
            let start = frame_offset + f_idx * frame_size;
            let end = start + frame_size;
            if end > frames.len() {
                break;
            }
            let frame_data = &frames[start..end];
            write_frame_png(
                frame_data,
                w as u32,
                h as u32,
                c,
                &output_dir_clone.join(format!("frame_{:04}_{:02}.png", win_idx, f_idx)),
            )?;
        }
        frame_offset += t * frame_size;
    }
    info!("Frames written to {:?}", output_dir_clone);

    // Save memory store
    let memory_path = output_dir_clone.join("memory.json");
    pipeline
        .pipeline
        .memory_store
        .save_json(&memory_path)
        .map_err(|e| format!("Failed to save memory: {}", e))?;
    info!("Memory saved to {:?}", memory_path);

    Ok(())
}

fn cmd_visualize(trajectory_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let trajectory = CameraTrajectory::load_json(trajectory_path)?;
    info!("Trajectory: {} poses", trajectory.len());
    info!("Duration: {:.2}s", trajectory.duration());
    info!("Path length: {:.2} units", trajectory.path_length());

    let keyframes = trajectory.select_keyframes(0.5, 0.1);
    info!("Keyframes (motion-based): {}", keyframes.len());

    for (i, pose) in trajectory.poses.iter().enumerate() {
        let pos = pose.position();
        println!(
            "Frame {:4}: t={:.3}s  pos=({:.2}, {:.2}, {:.2})",
            i, pose.timestamp, pos.x, pos.y, pos.z
        );
    }

    Ok(())
}

fn cmd_splice(
    trajectory_a: &Path,
    trajectory_b: &Path,
    layout: &str,
    offset: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading trajectories...");
    let traj_a = CameraTrajectory::load_json(trajectory_a)?;
    let traj_b = CameraTrajectory::load_json(trajectory_b)?;

    // Build memory stores from trajectories
    let config = MemoryConfig::default();
    let store_a = MosaicMemoryStore::new(config.clone());
    let store_b = MosaicMemoryStore::new(config);

    let result = match layout {
        "horizontal" => manipulation::splice_horizontal(&store_a, &store_b, offset),
        "vertical" => {
            let flipped = manipulation::flip_vertical(&store_b);
            let mut combined = store_a.patches.clone();
            combined.extend(flipped);
            combined
        }
        _ => return Err(format!("Unknown layout: {}", layout).into()),
    };

    info!(
        "Splice complete: {} + {} = {} patches",
        traj_a.len(),
        traj_b.len(),
        result.len()
    );

    Ok(())
}

fn cmd_demo(
    num_frames: usize,
    width: u32,
    height: u32,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== MosaicMem Demo (Synthetic Models) ===");
    info!(
        "Frames: {}, Resolution: {}x{}, Steps: {}",
        num_frames, width, height, steps
    );

    // Create a circular camera trajectory
    use nalgebra::{Point3, Vector3};
    let trajectory = CameraTrajectory::new(
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
    );

    info!("Created circular trajectory: {} poses", trajectory.len());
    info!("Path length: {:.2} units", trajectory.path_length());

    let config = PipelineConfig {
        num_inference_steps: steps,
        width,
        height,
        window_size: num_frames.min(16),
        window_overlap: 2,
        ..Default::default()
    };

    let mut pipeline = AutoregressivePipeline::new(config);
    let backbone = SyntheticBackbone::new(0.1);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);
    let text_emb = vec![vec![0.0f32; 64]; 10];

    info!("Running autoregressive generation...");
    let (frames, shapes) = pipeline.generate(
        &trajectory,
        &text_emb,
        &backbone,
        &scheduler,
        &vae,
        &depth,
        None,
    )?;

    info!("=== Demo Results ===");
    info!("Generated {} windows", shapes.len());
    info!("Total frame data: {} values", frames.len());
    info!("Final memory: {}", pipeline.stats());

    // Show memory manipulation
    info!("--- Memory Manipulation Demo ---");
    let store = &pipeline.pipeline.memory_store;
    let flipped = manipulation::flip_vertical(store);
    info!("Vertical flip: {} patches", flipped.len());

    let erased = manipulation::erase_region(store, &Point3::origin(), 2.0);
    info!("Erased center region: {} patches remaining", erased.len());

    let translated = manipulation::translate(store, &Vector3::new(10.0, 0.0, 0.0));
    info!("Translated: {} patches", translated.len());

    // Write output artifacts
    let demo_dir = std::path::PathBuf::from("demo_output");
    std::fs::create_dir_all(&demo_dir)?;

    // Save point cloud
    let ply_path = demo_dir.join("scene.ply");
    pipeline
        .pipeline
        .fusion
        .global_cloud
        .export_ply(&ply_path)?;
    info!(
        "Point cloud exported: {:?} ({} points)",
        ply_path,
        pipeline.pipeline.fusion.num_points()
    );

    // Save memory store
    let mem_path = demo_dir.join("memory.json");
    pipeline.pipeline.memory_store.save_json(&mem_path)?;
    info!(
        "Memory saved: {:?} ({} patches)",
        mem_path,
        store.num_patches()
    );

    // Save trajectory
    let traj_path = demo_dir.join("trajectory.json");
    trajectory.save_json(&traj_path)?;
    info!("Trajectory saved: {:?}", traj_path);

    // Save sample frames
    let mut frame_offset = 0;
    for (win_idx, shape) in shapes.iter().enumerate() {
        let [_b, c, t, h, w] = *shape;
        let frame_size = c * h * w;
        for f_idx in 0..t {
            let start = frame_offset + f_idx * frame_size;
            let end = start + frame_size;
            if end > frames.len() {
                break;
            }
            write_frame_png(
                &frames[start..end],
                w as u32,
                h as u32,
                c,
                &demo_dir.join(format!("frame_{:04}_{:02}.png", win_idx, f_idx)),
            )?;
        }
        frame_offset += t * frame_size;
    }
    info!("Frames written to {:?}/", demo_dir);

    info!("=== Demo Complete ===");
    Ok(())
}

fn cmd_inspect(
    trajectory_path: &Path,
    width: u32,
    height: u32,
    show_coverage: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use mosaicmem_rs::camera::CameraIntrinsics;
    use mosaicmem_rs::diffusion::vae::VAE;
    use mosaicmem_rs::geometry::depth::DepthEstimator;
    use mosaicmem_rs::geometry::fusion::StreamingFusion;
    use mosaicmem_rs::memory::retrieval::MemoryRetriever;

    info!("=== MosaicMem Trajectory Inspector ===");
    let trajectory = CameraTrajectory::load_json(trajectory_path)?;
    info!("Trajectory: {} poses", trajectory.len());
    info!("Duration: {:.2}s", trajectory.duration());
    info!("Path length: {:.2} units", trajectory.path_length());

    // Keyframe analysis
    let keyframes_motion = trajectory.select_keyframes(0.5, 0.1);
    let keyframes_uniform = trajectory.select_keyframes(0.0, 0.0);
    info!(
        "Keyframes: {} (motion-based), {} (uniform)",
        keyframes_motion.len(),
        keyframes_uniform.len()
    );

    // Build a synthetic memory store to analyze spatial coverage
    let intrinsics = CameraIntrinsics::default_for_resolution(width, height);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);

    let mut fusion = StreamingFusion::new(0.05);
    let memory_config = MemoryConfig {
        max_patches: 10000,
        top_k: 64,
        ..Default::default()
    };
    let mut store = MosaicMemoryStore::new(memory_config);
    let vae = SyntheticVAE::new(8, 4, 16);

    // Process keyframes to build memory
    info!("--- Building synthetic memory from trajectory ---");
    for &ki in &keyframes_motion {
        if ki >= trajectory.len() {
            continue;
        }
        let pose = &trajectory.poses[ki];

        // Generate synthetic frame data
        let frame_pixels = (width * height * 3) as usize;
        let frame_data = vec![128u8; frame_pixels];

        let depth_map = depth_estimator.estimate_depth(&frame_data, width, height)?;

        // Add to fusion
        let _ = fusion.add_keyframe(
            &frame_data,
            width,
            height,
            &intrinsics,
            pose,
            &depth_estimator,
        );

        // Encode to latent and insert into memory
        let frame_f32: Vec<f32> = frame_data.iter().map(|&b| b as f32 / 255.0).collect();
        let frame_shape = [1, 3, 1, height as usize, width as usize];
        let (latents, lat_shape) = vae.encode(&frame_f32, &frame_shape)?;
        let lat_h = lat_shape[3];
        let lat_w = lat_shape[4];
        let lat_c = lat_shape[1];

        store.insert_keyframe(
            ki,
            pose.timestamp,
            &latents,
            lat_h,
            lat_w,
            lat_c,
            &depth_map,
            &intrinsics,
            pose,
        );
    }

    info!("--- Memory Statistics ---");
    info!("Stored patches: {}", store.num_patches());
    info!("Total tokens: {}", store.total_tokens());
    info!("Point cloud size: {} points", fusion.num_points());
    info!("Keyframes processed: {}", fusion.num_keyframes);

    // Spatial extent analysis
    if !store.patches.is_empty() {
        let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
        let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
        for p in &store.patches {
            min_x = min_x.min(p.center.x);
            min_y = min_y.min(p.center.y);
            min_z = min_z.min(p.center.z);
            max_x = max_x.max(p.center.x);
            max_y = max_y.max(p.center.y);
            max_z = max_z.max(p.center.z);
        }
        info!(
            "Spatial extent: ({:.2}, {:.2}, {:.2}) to ({:.2}, {:.2}, {:.2})",
            min_x, min_y, min_z, max_x, max_y, max_z
        );
        info!(
            "Bounding box size: {:.2} x {:.2} x {:.2}",
            max_x - min_x,
            max_y - min_y,
            max_z - min_z
        );
    }

    // Per-frame coverage analysis
    if show_coverage {
        info!("--- Per-Frame Coverage Analysis ---");
        let retriever = MemoryRetriever::new();
        for (i, pose) in trajectory.poses.iter().enumerate() {
            let mosaic = retriever.retrieve(&store, pose, &intrinsics);
            let coverage = mosaic.coverage_ratio();
            let bar_len = (coverage * 40.0) as usize;
            let bar: String = "#".repeat(bar_len) + &".".repeat(40 - bar_len);
            println!(
                "Frame {:4}: coverage={:.1}% patches={:3} [{}]",
                i,
                coverage * 100.0,
                mosaic.num_patches(),
                bar
            );
        }
    }

    info!("=== Inspection Complete ===");
    Ok(())
}

/// Write a single frame (C*H*W float data) as a PNG image.
fn write_frame_png(
    data: &[f32],
    width: u32,
    height: u32,
    channels: usize,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let pixels = (width * height) as usize;
    let mut rgb_buf = vec![0u8; pixels * 3];
    for i in 0..pixels {
        for c in 0..3.min(channels) {
            let idx = c * pixels + i;
            let val = if idx < data.len() {
                (data[idx].clamp(0.0, 1.0) * 255.0) as u8
            } else {
                128
            };
            rgb_buf[i * 3 + c] = val;
        }
    }
    image::save_buffer(path, &rgb_buf, width, height, image::ColorType::Rgb8)?;
    Ok(())
}

fn cmd_show_config(config_path: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let config = if let Some(path) = config_path {
        info!("Loading config from {:?}", path);
        let contents = std::fs::read_to_string(path)?;
        let config: PipelineConfig = serde_json::from_str(&contents)?;
        config
    } else {
        PipelineConfig::default()
    };

    let json = serde_json::to_string_pretty(&config)?;
    println!("{json}");

    // Also show derived dimensions
    println!("\n# Derived dimensions:");
    println!(
        "#   Latent size: {}x{}",
        config.latent_height(),
        config.latent_width()
    );
    println!("#   Latent frames: {}", config.latent_frames());
    println!(
        "#   Latent elements per window: {}",
        config.latent_channels
            * config.latent_frames()
            * config.latent_height()
            * config.latent_width()
    );

    Ok(())
}

fn cmd_export_ply(
    trajectory_path: &Path,
    output_path: &Path,
    width: u32,
    height: u32,
    voxel_size: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use mosaicmem_rs::camera::CameraIntrinsics;
    use mosaicmem_rs::geometry::fusion::StreamingFusion;

    info!("=== PLY Export ===");
    let trajectory = CameraTrajectory::load_json(trajectory_path)?;
    info!("Trajectory: {} poses", trajectory.len());

    let intrinsics = CameraIntrinsics::default_for_resolution(width, height);
    let depth_estimator = SyntheticDepthEstimator::new(5.0, 1.0);
    let mut fusion = StreamingFusion::new(voxel_size);

    let keyframes = trajectory.select_keyframes(0.5, 0.1);
    info!("Processing {} keyframes...", keyframes.len());

    for &ki in &keyframes {
        if ki >= trajectory.len() {
            continue;
        }
        let pose = &trajectory.poses[ki];
        let frame_data = vec![128u8; (width * height * 3) as usize];
        let _ = fusion.add_keyframe(
            &frame_data,
            width,
            height,
            &intrinsics,
            pose,
            &depth_estimator,
        );
    }

    info!(
        "Point cloud: {} points, {} keyframes",
        fusion.num_points(),
        fusion.num_keyframes
    );

    fusion.global_cloud.export_ply(output_path)?;
    info!("Exported PLY to {:?}", output_path);

    Ok(())
}

fn cmd_bench(
    num_frames: usize,
    width: u32,
    height: u32,
    steps: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    info!("=== MosaicMem Pipeline Benchmark ===");
    info!(
        "Frames: {}, Resolution: {}x{}, Steps: {}, Iterations: {}",
        num_frames, width, height, steps, iterations
    );

    let config = PipelineConfig {
        num_inference_steps: steps,
        width,
        height,
        window_size: num_frames.min(16),
        window_overlap: 2,
        ..Default::default()
    };

    let backbone = SyntheticBackbone::new(0.1);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(8, 4, 16);
    let depth = SyntheticDepthEstimator::new(5.0, 1.0);
    let text_emb = vec![vec![0.0f32; 64]; 10];

    use nalgebra::{Point3, Vector3};
    let trajectory = CameraTrajectory::new(
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
    );

    let mut durations = Vec::with_capacity(iterations);

    for iter in 0..iterations {
        let mut pipeline = AutoregressivePipeline::new(config.clone());
        let start = Instant::now();

        let (frames, shapes) = pipeline.generate(
            &trajectory,
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
            &depth,
            None,
        )?;

        let elapsed = start.elapsed();
        durations.push(elapsed);

        info!(
            "Iteration {}: {:.2}ms ({} windows, {} values, {})",
            iter + 1,
            elapsed.as_secs_f64() * 1000.0,
            shapes.len(),
            frames.len(),
            pipeline.stats()
        );
    }

    let total_ms: f64 = durations.iter().map(|d| d.as_secs_f64() * 1000.0).sum();
    let avg_ms = total_ms / iterations as f64;
    let min_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::INFINITY, f64::min);
    let max_ms = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let fps = num_frames as f64 / (avg_ms / 1000.0);

    info!("=== Benchmark Results ===");
    info!("Average: {:.2}ms", avg_ms);
    info!("Min:     {:.2}ms", min_ms);
    info!("Max:     {:.2}ms", max_ms);
    info!("Throughput: {:.1} frames/sec", fps);
    info!("Per-frame: {:.2}ms", avg_ms / num_frames as f64);
    info!(
        "Per-step:  {:.2}ms",
        avg_ms / (num_frames as f64 * steps as f64)
    );

    Ok(())
}
