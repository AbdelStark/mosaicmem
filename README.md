# mosaicmem-rs

Spatial memory for camera-controlled video generation in Rust.

`mosaicmem-rs` implements the memory side of MosaicMem: streaming 3D reconstruction, patch-level spatial memory, view-conditioned retrieval, geometric alignment, and autoregressive generation plumbing. The repo also ships deterministic synthetic backends, so the full pipeline runs end to end without external model weights.

Paper: [arXiv:2603.17117](https://arxiv.org/abs/2603.17117)  
Project: [mosaicmem.github.io](https://mosaicmem.github.io/mosaicmem/)  
Demo: [YouTube](https://www.youtube.com/watch?v=K3Q9kf8t08I)

## Why This Exists

Video world models break when the camera moves too far, revisits an old area, or tries to hold onto scene structure over long rollouts. MosaicMem fixes that with explicit spatial memory:

- estimate depth from keyframes
- lift patches into 3D
- store them in a searchable memory
- retrieve the right patches for a target pose
- align them with Warped RoPE and Warped Latent
- inject them into a diffusion-style generation loop

The result is a geometry-aware memory stack you can test, inspect, and extend from Rust.

## What Ships Here

| Area | Included |
|------|----------|
| `camera` | camera intrinsics, SE(3) poses, trajectories |
| `geometry` | synthetic depth backend, unprojection, projection, frustum culling, streaming fusion |
| `memory` | 3D patch store, retrieval, mosaic composition, spatial edits |
| `attention` | Warped RoPE, Warped Latent, PRoPE, memory cross-attention |
| `pipeline` | single-window inference and autoregressive rollout |
| `diffusion` | backend traits plus deterministic synthetic VAE and backbone |
| `cli` | demo, generate, inspect, visualize, splice, export, benchmark |

## What Does Not Ship Here

- pretrained diffusion weights
- pretrained VAE weights
- pretrained depth models
- training code

Real backends plug in behind traits in `diffusion` and `geometry::depth`.

## Quick Start

```bash
git clone https://github.com/AbdelStark/mosaicmem.git
cd mosaicmem
cargo test
cargo run -- demo --num-frames 16 --width 64 --height 64 --steps 5
```

Use the CLI:

```bash
cargo run -- --help
cargo run -- generate --trajectory trajectory.json --output out
cargo run -- inspect --trajectory trajectory.json --coverage
cargo run -- export-ply --trajectory trajectory.json --output scene.ply
```

Use it as a library:

```toml
[dependencies]
mosaicmem-rs = { git = "https://github.com/AbdelStark/mosaicmem.git" }
```

## Repository Shape

```text
src/
  attention/   memory alignment and memory cross-attention
  camera/      camera models, poses, trajectories
  diffusion/   backend traits and synthetic implementations
  geometry/    depth, projection, point clouds, fusion
  memory/      patch store, retrieval, mosaic composition, editing
  pipeline/    inference and autoregressive rollout
tests/
  integration.rs
  meaningful_end_to_end.rs
```

## Current Status

This repository is useful today if you want to:

- study the memory architecture in code
- run the full pipeline without external weights
- swap in your own depth, VAE, or diffusion backends
- test retrieval, alignment, and revisitation behavior in isolation

The synthetic path is not a substitute for a trained video model. It is there to make the system testable, debuggable, and reproducible.

## Quality Bar

The repo is kept green with:

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

The test suite covers unit, integration, and end-to-end revisitation behavior.

## License

MIT

## Citation

```bibtex
@article{mosaicmem2026,
  title={MosaicMem: Hybrid Spatial Memory for Controllable Video World Models},
  author={Wei Yu and Runjia Qian and Yumeng Li and Liquan Wang and Songheng Yin and
          Sri Siddarth Chakaravarthy P and Dennis Anthony and Yang Ye and Yidi Li and
          Weiwei Wan and Animesh Garg},
  journal={arXiv preprint arXiv:2603.17117},
  year={2026}
}
```
