# mosaicmem

`mosaicmem` is a Rust library and CLI that simulates the memory side of a MosaicMem-style video generation pipeline: camera trajectories, synthetic depth, 3D fusion, patch memory retrieval, geometric alignment, and synthetic diffusion/VAE inference.

## Why This Exists

Long-horizon video generation breaks down when the camera revisits old viewpoints or moves far from the initial context. This repository isolates the spatial-memory side of that problem so the core mechanics can be tested end to end without external model weights.

## Who It Is For

- Rust developers exploring video world-model architecture
- researchers who want a deterministic synthetic testbed for spatial memory ideas
- contributors who need a small, inspectable codebase before integrating real model backends

## Current Status

As of 2026-03-25, this project is **alpha**.

- Suitable for: local experimentation, synthetic end-to-end tests, architecture exploration, CLI demos
- Not yet suitable for: production inference, checkpoint-backed generation, stable public API guarantees
- Known limitations:
  - synthetic backends only; no ONNX or checkpoint loading
  - CLI scene-editing paths are demo-oriented
  - no operational monitoring stack beyond local logs
  - `cargo audit` still reports an unmaintained transitive `paste` dependency through `nalgebra`
- Breaking changes in the next release: likely, especially in the public Rust API and CLI semantics

## Quick Start

Prerequisites:

- stable Rust with edition 2024 support

Verified on this checkout:

```bash
cargo test
cargo run -- show-config
cargo run -- demo --num-frames 4 --width 32 --height 32 --steps 1
```

The demo writes synthetic artifacts to `demo_output/`, including sample PNGs, a point cloud, a saved trajectory, and a serialized memory store.

## CLI

```text
mosaicmem <COMMAND>

Commands:
  generate     Generate synthetic video windows for a trajectory and write PNGs
  visualize    Print trajectory statistics
  splice       Build synthetic stores from two trajectories and splice their patches
  inspect      Build a synthetic store and print memory / coverage statistics
  show-config  Print default or loaded pipeline configuration as JSON
  demo         Run the verified synthetic end-to-end demo
  export-ply   Export a synthetic fused point cloud for a trajectory
  bench        Benchmark the synthetic pipeline
  tui          Launch the interactive showcase
```

Examples:

```bash
cargo run -- demo --num-frames 16 --width 64 --height 64 --steps 5
cargo run -- show-config
cargo run -- --help
```

## Library Usage

```toml
[dependencies]
mosaicmem = { git = "https://github.com/AbdelStark/mosaicmem.git" }
```

```rust
use mosaicmem::camera::CameraPose;
use mosaicmem::camera::CameraTrajectory;
use mosaicmem::diffusion::backbone::SyntheticBackbone;
use mosaicmem::diffusion::scheduler::DDPMScheduler;
use mosaicmem::diffusion::vae::SyntheticVAE;
use mosaicmem::geometry::depth::SyntheticDepthEstimator;
use mosaicmem::pipeline::autoregressive::AutoregressivePipeline;
use mosaicmem::pipeline::config::PipelineConfig;

let config = PipelineConfig {
    width: 64,
    height: 64,
    window_size: 8,
    num_inference_steps: 3,
    ..Default::default()
};

let trajectory = CameraTrajectory::new(vec![CameraPose::identity(0.0)]);
let backbone = SyntheticBackbone::new(0.1);
let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
let vae = SyntheticVAE::new(8, 4, 16);
let depth = SyntheticDepthEstimator::new(5.0, 1.0);
let text_embedding = vec![vec![0.0f32; 64]; 10];

let mut pipeline = AutoregressivePipeline::new(config);
let _ = pipeline.generate(
    &trajectory,
    &text_embedding,
    &backbone,
    &scheduler,
    &vae,
    &depth,
    None,
);
```

## How It Works

```text
Keyframe -> Depth -> 3D Lift -> Memory Store
                                   |
Target Pose -> Retrieve -> Align -> Condition Synthetic Diffusion -> Frame
```

Core flow:

1. A trajectory provides camera poses.
2. Synthetic depth lifts keyframes into 3D.
3. Patch latents are stored with world-space centers and source provenance.
4. Retrieval selects visible patches for a target pose.
5. Warped RoPE and warped latent alignment condition synthetic denoising.
6. Generated windows feed back into memory for autoregressive rollouts.

## Repository Map

| Path | Responsibility |
|---|---|
| `src/camera` | poses, intrinsics, trajectory I/O, keyframe selection |
| `src/geometry` | synthetic depth, projection, point clouds, fusion |
| `src/memory` | patch storage, retrieval, mosaics, manipulation |
| `src/attention` | RoPE, PRoPE, warped geometry alignment, memory cross-attention |
| `src/diffusion` | synthetic scheduler, backbone, VAE |
| `src/pipeline` | single-window inference and autoregressive rollout |
| `src/tui` | interactive showcase |
| `tests` | integration coverage and end-to-end regressions |

## Quality Baseline

Verified locally on 2026-03-25:

- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`

At this point the repository contains 120 passing tests across unit, integration, and end-to-end coverage.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and [AGENTS.md](AGENTS.md) for agent-specific repository context.

Typical local loop:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Roadmap

### Milestone 1: API and Config Hardening

- Exit criteria:
  - explicit config validation for public constructors and CLI boundaries
  - fewer boxed errors in the CLI path
  - clearer guarantees around tensor layouts and output semantics

### Milestone 2: Real Backend Integration

- Exit criteria:
  - depth and diffusion traits backed by at least one non-synthetic implementation
  - hermetic tests for shape mapping and model I/O
  - clear failure modes when external weights are missing or incompatible

### Milestone 3: Release Candidate

- Exit criteria:
  - stable documented public API surface
  - CI, changelog, contributing docs, and agent context kept current
  - stronger operability story for non-demo CLI usage

## Help

- Open an issue in this repository for bugs, questions, or missing docs
- Check [CHANGELOG.md](CHANGELOG.md) for user-visible changes

## Citation

If you use the project as a reference implementation, cite the MosaicMem paper:

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
