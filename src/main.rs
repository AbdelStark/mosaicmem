use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};
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
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("mosaicmem_rs=info".parse().unwrap()))
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
            if let Err(e) = cmd_generate(
                &trajectory,
                &prompt,
                &output,
                width,
                height,
                steps,
                window_size,
                window_overlap,
                seed,
            ) {
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
    }
}

fn cmd_generate(
    trajectory_path: &PathBuf,
    prompt: &str,
    output_dir: &PathBuf,
    width: u32,
    height: u32,
    steps: usize,
    window_size: usize,
    window_overlap: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Loading trajectory from {:?}", trajectory_path);
    let trajectory = CameraTrajectory::load_json(trajectory_path)?;
    info!("Loaded {} poses", trajectory.len());

    std::fs::create_dir_all(output_dir)?;

    let config = PipelineConfig {
        num_inference_steps: steps,
        width,
        height,
        window_size,
        window_overlap,
        seed,
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

    Ok(())
}

fn cmd_visualize(trajectory_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
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
    trajectory_a: &PathBuf,
    trajectory_b: &PathBuf,
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
    info!("Frames: {}, Resolution: {}x{}, Steps: {}", num_frames, width, height, steps);

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

    info!("=== Demo Complete ===");
    Ok(())
}
