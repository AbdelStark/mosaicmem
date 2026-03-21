# mosaicmem-rs: Rust Implementation of MosaicMem

## Specification v0.1.0

---

## Overview

`mosaicmem-rs` is a Rust implementation of the core components of MosaicMem (Yu, Qian, Li et al., 2026), a hybrid spatial memory system for video world models. The system lifts video patches into 3D for reliable localization and retrieval, then injects them via attention-based conditioning to enable long-horizon, camera-controlled video generation with persistent spatial consistency.

This implementation uses **Burn** as the primary deep learning framework and **Tract** for ONNX model inference, consistent with our Rust ML project ecosystem.

## What We Implement (Scope)

MosaicMem has two aspects: (a) the memory system (geometry, retrieval, alignment) and (b) the diffusion backbone (Wan 2.2 DiT). We implement **(a) completely in Rust** and provide **(b) as trait-based integration points** for existing diffusion inference via ONNX/Tract.

### In scope:
1. **Streaming 3D reconstruction** — depth estimation + point cloud fusion
2. **Mosaic memory store** — 3D patch storage and spatial indexing
3. **Point-to-frame retrieval** — camera-pose-based spatial query
4. **Warped RoPE** — geometric positional encoding with temporal coordinates
5. **Warped Latent** — latent feature space warping for alignment
6. **PRoPE camera conditioning** — projective positional encoding
7. **Memory cross-attention** — inject memory patches into transformer attention
8. **Chained autoregressive pipeline** — multi-window generation loop
9. **Patch-and-compose interface** — assemble mosaic from retrieved patches

### Out of scope (use via Tract/ONNX):
- Full Wan 2.2 DiT inference (use pretrained ONNX model)
- Full VAE encoder/decoder (use pretrained ONNX model)
- Training (inference-only for this implementation)

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       mosaicmem-rs                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  geometry/                                          │     │
│  │  ├── depth.rs      (monocular depth via Tract)     │     │
│  │  ├── pointcloud.rs (3D point cloud operations)     │     │
│  │  ├── camera.rs     (camera intrinsics/extrinsics)  │     │
│  │  ├── projection.rs (project/unproject operations)   │     │
│  │  └── fusion.rs     (streaming point cloud fusion)  │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  memory/                                            │     │
│  │  ├── store.rs      (3D patch storage + indexing)   │     │
│  │  ├── retrieval.rs  (point-to-frame spatial query)  │     │
│  │  ├── mosaic.rs     (patch-and-compose assembly)    │     │
│  │  └── manipulation.rs (splice, edit, transform)     │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  attention/                                         │     │
│  │  ├── warped_rope.rs  (Warped RoPE alignment)       │     │
│  │  ├── warped_latent.rs (Warped Latent alignment)    │     │
│  │  ├── prope.rs        (PRoPE camera conditioning)   │     │
│  │  ├── memory_cross.rs (memory cross-attention)      │     │
│  │  └── rope.rs         (standard RoPE utilities)     │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  camera/                                            │     │
│  │  ├── pose.rs       (SE(3) camera poses)            │     │
│  │  ├── trajectory.rs (camera trajectory management)  │     │
│  │  └── intrinsics.rs (focal length, principal point) │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  pipeline/                                          │     │
│  │  ├── inference.rs  (full generation pipeline)      │     │
│  │  ├── autoregressive.rs (chained multi-window)      │     │
│  │  └── config.rs     (pipeline configuration)        │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │  diffusion/                                         │     │
│  │  ├── backbone.rs   (DiT trait + Tract wrapper)     │     │
│  │  ├── scheduler.rs  (noise scheduler abstraction)   │     │
│  │  └── vae.rs        (VAE encode/decode via Tract)   │     │
│  └────────────────────────────────────────────────────┘     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Dependencies

```toml
[dependencies]
burn = { version = "0.16", features = ["ndarray", "wgpu"] }
burn-ndarray = "0.16"
tract-onnx = "0.21"          # ONNX inference for depth/DiT/VAE
nalgebra = "0.33"             # Linear algebra (SE3, projections)
ndarray = "0.16"              # N-dimensional arrays
image = "0.25"                # Image I/O
rayon = "1.10"                # Parallel point cloud operations
kiddo = "4"                   # KD-tree for spatial queries
serde = { version = "1", features = ["derive"] }
clap = { version = "4", features = ["derive"] }
ratatui = "0.29"              # TUI visualization
crossterm = "0.28"
tracing = "0.1"
thiserror = "2.0"
```

## Key Design: Burn for Neural Ops, nalgebra for Geometry

- **Burn** — attention layers, RoPE computation, cross-attention, tensor operations
- **Tract** — ONNX model inference (depth estimator, DiT backbone, VAE)
- **nalgebra** — 3D geometry (camera matrices, SE(3) transforms, projections, point clouds)
- **kiddo** — KD-tree for efficient spatial queries in point cloud retrieval

## RFC Index

| RFC | Title | Status |
|-----|-------|--------|
| RFC-001 | Core Types (Camera, PointCloud, Patch, Mosaic) | Draft |
| RFC-002 | Geometry Pipeline (Depth, Unprojection, Fusion) | Draft |
| RFC-003 | Mosaic Memory Store (3D Patch Storage + Retrieval) | Draft |
| RFC-004 | Warped RoPE (Geometric Positional Encoding) | Draft |
| RFC-005 | Warped Latent (Feature Space Alignment) | Draft |
| RFC-006 | PRoPE Camera Conditioning | Draft |
| RFC-007 | Memory Cross-Attention (Inject into DiT) | Draft |
| RFC-008 | Autoregressive Pipeline (Chained Generation) | Draft |

## CLI Design

```bash
# Single-view generation with memory
mosaicmem-rs generate \
  --model wan2.2.onnx \
  --depth-model metric3d.onnx \
  --trajectory trajectory.json \
  --prompt "A medieval village with a wolf walking through" \
  --output output_video.mp4

# Memory visualization (TUI)
mosaicmem-rs visualize \
  --pointcloud scene.ply \
  --trajectory trajectory.json

# Memory manipulation
mosaicmem-rs splice \
  --memory-a scene_medieval.mem \
  --memory-b scene_modern.mem \
  --layout horizontal \
  --output merged.mem
```
