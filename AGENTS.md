# AGENTS.md

## Project Identity

`mosaicmem` is a Rust library and CLI that simulates the memory side of a MosaicMem-style video generation pipeline: camera trajectories, synthetic depth, 3D fusion, patch memory retrieval, geometric alignment, and synthetic diffusion/VAE inference.

This repository is currently an alpha-quality research/demo codebase. It is suitable for local experimentation, architectural exploration, and synthetic end-to-end tests. It is not yet a real model-serving stack and does not load external checkpoints.

## Architecture Map

- `src/camera`: poses, intrinsics, trajectory I/O and keyframe selection.
- `src/geometry`: synthetic depth, projection, point clouds, streaming fusion.
- `src/memory`: patch storage, retrieval, mosaics, spatial manipulation.
- `src/attention`: RoPE, PRoPE, warped RoPE, warped latent, memory cross-attention.
- `src/diffusion`: synthetic scheduler, backbone, VAE.
- `src/pipeline`: single-window inference and autoregressive rollout.
- `src/main.rs`: CLI for demo, inspect, export, benchmark, and synthetic scene utilities.
- `src/tui`: interactive showcase built on `ratatui`.
- `tests/`: integration coverage and one meaningful end-to-end regression.

## Tech Stack

- Rust edition 2024
- Crates: `clap`, `image`, `kiddo`, `nalgebra`, `rand`, `ratatui`, `rayon`, `serde`, `serde_json`, `thiserror`, `tracing`
- Synthetic backends only: `SyntheticDepthEstimator`, `SyntheticBackbone`, `SyntheticVAE`

## Verified Commands

Run these from the repository root.

- `cargo fmt --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`
- `cargo run -- --help`
- `cargo run -- show-config`
- `cargo run -- demo --num-frames 4 --width 32 --height 32 --steps 1`

## Conventions

- Keep public errors typed. Avoid introducing new `Box<dyn Error>` paths inside library code.
- Synthetic tensor layout is flattened `[B, C, T, H, W]` unless a function explicitly converts it.
- Per-frame extraction must respect channel-major layout. Do not slice decoded tensors as if frames were interleaved.
- Use deterministic randomness for synthetic pipeline behavior when config carries a seed.
- Prefer adding regression tests for every correctness fix, especially around tensor layout and memory behavior.
- Keep README and CLI behavior aligned. Do not claim commands or test counts that have not been verified locally.

## Critical Constraints

- Do not replace synthetic components with placeholder “real model” claims unless actual model-loading code exists.
- Do not reintroduce the old `[B, C, T, H, W]` frame-layout bug in memory updates, PNG export, or overlap blending.
- Do not ignore `StreamingFusion::add_keyframe` failures in library or CLI paths that claim to produce analysis artifacts.
- Do not assume the `splice` command works on serialized memory stores; today it rebuilds synthetic stores from trajectory files.

## Gotchas

- `generate_window` returns decoded frames in flattened `[B, C, T, H, W]` order.
- `MosaicMemoryStore::insert_keyframe` expects per-frame latent data in planar token-major layout, not raw `[B, C, T, H, W]`.
- The TUI is showcase code, not an operational control plane.
- The project keeps generated demo assets in `demo_output/` and CLI assets in `output/`; both should stay ignored.

## Current State

As of 2026-03-25, the repository has a green `cargo test` and strict `cargo clippy` pass on the local machine. Known limitations:

- Synthetic models only; no ONNX or checkpoint loading.
- No stability guarantee for public APIs yet.
- CLI scene-editing commands remain demo-oriented rather than production-grade content tools.
- `cargo audit` still reports an unmaintained transitive `paste` dependency through `nalgebra`.
