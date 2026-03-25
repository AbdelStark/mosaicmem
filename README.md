<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/AbdelStark/mosaicmem/main/.github/assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/AbdelStark/mosaicmem/main/.github/assets/logo-light.svg">
    <img alt="MosaicMem" width="460">
  </picture>
</p>

<h3 align="center">Spatial memory for camera-controlled video generation</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2603.17117"><img src="https://img.shields.io/badge/arXiv-2603.17117-b31b1b.svg" alt="arXiv"></a>
  <a href="https://mosaicmem.github.io/mosaicmem/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://www.youtube.com/watch?v=K3Q9kf8t08I"><img src="https://img.shields.io/badge/Demo-YouTube-red" alt="Demo"></a>
  <a href="https://crates.io/crates/mosaicmem-rs"><img src="https://img.shields.io/crates/v/mosaicmem-rs.svg" alt="crates.io"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</p>

---

Video world models break when the camera moves too far, revisits old areas, or tries to maintain scene structure over long rollouts. **MosaicMem** fixes this with explicit spatial memory -- a geometry-aware memory stack that lifts observed patches into 3D, retrieves them at novel viewpoints, and injects them back into the generation loop.

This Rust implementation (`mosaicmem-rs`) provides the complete memory-side pipeline: streaming 3D reconstruction, patch-level spatial storage, view-conditioned retrieval, geometric alignment via Warped RoPE / Warped Latent, and autoregressive generation plumbing. Ships with deterministic synthetic backends so the **full pipeline runs end-to-end without external model weights**.

## How it works

```
Keyframe ──> Depth ──> Lift to 3D ──> Memory Store
                                           |
Target Pose ──> Query Memory ──> Retrieve + Align ──> Condition Diffusion ──> Frame
                                     |            |
                                Warped RoPE   Warped Latent
```

1. **Depth estimation** -- extract per-pixel depth from keyframes
2. **3D lifting** -- unproject patches into a shared world-space point cloud
3. **Spatial storage** -- index patches in a kd-tree for fast nearest-neighbor lookup
4. **View-conditioned retrieval** -- given a target camera pose, find the most relevant stored patches
5. **Geometric alignment** -- apply Warped RoPE (attention-level) and Warped Latent (feature-level) to align retrieved context with the target view
6. **Conditioned generation** -- inject aligned memory into a diffusion denoising loop to produce geometry-consistent frames

## Architecture

```
mosaicmem-rs/
  src/
    attention/       # PRoPE, Warped RoPE, Warped Latent, memory cross-attention
    camera/          # Intrinsics, poses, trajectory I/O
    diffusion/       # Backbone, DDPM scheduler, VAE (synthetic stubs)
    geometry/        # Depth estimation, point cloud fusion, projection
    memory/          # Mosaic memory store, retrieval, spatial manipulation
    pipeline/        # Autoregressive generation, config, inference loop
  tests/
    integration.rs           # 120+ assertions across all modules
    meaningful_end_to_end.rs # Memory-conditioned revisit consistency test
```

| Module | Purpose | Key types |
|--------|---------|-----------|
| `attention` | Position encoding & cross-attention | `WarpedRoPE`, `WarpedLatent`, `PRoPE`, `MemoryCrossAttention` |
| `camera` | Camera model & trajectories | `CameraPose`, `CameraIntrinsics`, `CameraTrajectory` |
| `diffusion` | Generation backbone (synthetic) | `SyntheticBackbone`, `DDPMScheduler`, `SyntheticVAE` |
| `geometry` | 3D reconstruction primitives | `DepthEstimator`, `PointCloud`, `StreamingFusion` |
| `memory` | Spatial memory system | `MosaicMemoryStore`, `MemoryRetrieval`, `SpatialManipulation` |
| `pipeline` | End-to-end generation | `AutoregressivePipeline`, `PipelineConfig` |

## Quick start

### Prerequisites

- Rust 1.85+ (edition 2024)
- No GPU or external weights required

### Run

```bash
git clone https://github.com/AbdelStark/mosaicmem.git
cd mosaicmem
cargo test           # 122 tests, ~0.3s
cargo run -- demo --num-frames 16 --width 64 --height 64 --steps 5
```

### Use as a library

```toml
# Cargo.toml
[dependencies]
mosaicmem-rs = { git = "https://github.com/AbdelStark/mosaicmem.git" }
```

```rust
use mosaicmem_rs::camera::{CameraPose, CameraTrajectory};
use mosaicmem_rs::geometry::depth::SyntheticDepthEstimator;
use mosaicmem_rs::memory::store::{MemoryConfig, MosaicMemoryStore};
use mosaicmem_rs::pipeline::autoregressive::AutoregressivePipeline;
use mosaicmem_rs::pipeline::config::PipelineConfig;

// Build a camera trajectory
let trajectory = CameraTrajectory::circle(num_frames, radius, height);

// Configure and run the pipeline
let config = PipelineConfig {
    width: 256,
    height: 256,
    steps: 50,
    window_size: 16,
    ..Default::default()
};
let mut pipeline = AutoregressivePipeline::new(config);
let frames = pipeline.generate(&trajectory, "a living room scene")?;
```

## CLI reference

```
mosaicmem-rs <COMMAND>

Commands:
  generate     Generate video frames from a camera trajectory
  demo         Run with synthetic data (no models required)
  inspect      Show memory/geometry statistics for a trajectory
  visualize    Display memory store diagnostics
  splice       Merge two memory stores with spatial layout
  export-ply   Export reconstructed point cloud to PLY
  show-config  Dump or load pipeline configuration as JSON
  bench        Run pipeline performance benchmark
```

### Examples

```bash
# Generate frames from a trajectory file
cargo run -- generate --trajectory trajectory.json --output out/ --width 256 --height 256

# Inspect memory coverage across a trajectory
cargo run -- inspect --trajectory trajectory.json --coverage

# Export the reconstructed 3D scene
cargo run -- export-ply --trajectory trajectory.json --output scene.ply

# Benchmark throughput
cargo run -- bench --num-frames 64 --width 128 --height 128 --iterations 5

# Splice two scenes side-by-side
cargo run -- splice --trajectory-a scene1.json --trajectory-b scene2.json --layout horizontal
```

## Key features

- **Streaming 3D fusion** -- incrementally builds a point cloud as new keyframes arrive, no batch reconstruction needed
- **kd-tree spatial index** -- O(log n) nearest-neighbor retrieval over millions of stored patches via [kiddo](https://crates.io/crates/kiddo)
- **Warped RoPE** -- applies geometric-aware rotary position encoding so attention respects 3D spatial relationships, not just token order
- **Warped Latent** -- feature-level alignment that reprojects retrieved patches into the target view's latent space
- **PRoPE** -- progressive rotary position encoding with temporal decay for long-horizon consistency
- **Autoregressive windowing** -- generates arbitrarily long videos with overlapping windows and memory carryover
- **Adaptive keyframe selection** -- automatically selects keyframes based on camera motion magnitude
- **Memory manipulation** -- splice, transform, and compose spatial memory stores for scene editing
- **Parallel computation** -- leverages [rayon](https://crates.io/crates/rayon) for multi-core depth estimation, projection, and fusion
- **Deterministic synthetic backends** -- full pipeline testable without GPU or model weights
- **Zero unsafe code** -- pure safe Rust throughout

## Testing

```bash
cargo test                    # All 122 tests
cargo test --test integration # Integration suite
cargo test -- --nocapture     # With stdout output
```

The test suite covers:

- Camera pose composition and trajectory generation
- Depth estimation and 3D point cloud construction
- Memory store insertion, retrieval, and spatial queries
- Warped RoPE / Warped Latent geometric alignment correctness
- Full pipeline end-to-end generation with memory conditioning
- **Revisit consistency**: generated frames at revisited viewpoints are closer to the originally observed scene than unconditioned generation

## Performance

The synthetic pipeline (no neural network inference) on a single core:

| Resolution | Frames | Steps | Time |
|-----------|--------|-------|------|
| 64x64 | 32 | 5 | ~0.2s |
| 128x128 | 32 | 5 | ~0.8s |
| 256x256 | 16 | 50 | ~4s |

Memory store scales to millions of patches with sub-millisecond retrieval via kd-tree spatial indexing.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
cargo fmt           # Format
cargo clippy        # Lint
cargo test          # Must pass
```

## Citation

```bibtex
@article{mosaicmem2026,
  title   = {MosaicMem: Hybrid Spatial Memory for Controllable Video World Models},
  author  = {Wei Yu and Runjia Qian and Yumeng Li and Liquan Wang and Songheng Yin and
             Sri Siddarth Chakaravarthy P and Dennis Anthony and Yang Ye and Yidi Li and
             Weiwei Wan and Animesh Garg},
  journal = {arXiv preprint arXiv:2603.17117},
  year    = {2026}
}
```

## License

[MIT](LICENSE)
