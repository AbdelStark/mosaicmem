# mosaicmem-rs Implementation Plan

## Phase 0: Foundation (Week 1)
- Core types: CameraPose (SE3), CameraIntrinsics, PointCloud3D, Patch3D, MosaicFrame
- nalgebra integration for 3D math (rotation, projection, unprojection)
- Camera trajectory loading (JSON format: list of timestamped SE3 poses)
- Basic point cloud operations (create, merge, render from viewpoint)

## Phase 1: Geometry Pipeline (Week 2)
- Monocular depth estimation via Tract (load ONNX depth model)
- Depth map → 3D unprojection (using camera intrinsics)
- Streaming point cloud fusion (incremental, per-keyframe)
- Point cloud spatial indexing via KD-tree (kiddo crate)
- Visibility testing: which 3D points are visible from a given camera pose

## Phase 2: Mosaic Memory Store (Week 3)
- Patch3D: link 3D points to source frame + 2D patch location + latent features
- MosaicMemoryStore: insert, query, delete patches
- Point-to-Frame Retrieval: given target camera pose → retrieve visible patches
- Patch spatial alignment: reproject retrieved patches to target view coordinates
- Memory manipulation: splice, translate, flip (Inception-style compositing)

## Phase 3: Attention Mechanisms (Week 4-5)
- Standard RoPE implementation in Burn
- Warped RoPE: extend RoPE with reprojected 3D coordinates + temporal indices
- Warped Latent: direct feature-space warping of retrieved patch latents
- PRoPE: projective positional encoding from camera geometry
- Memory cross-attention: K/V from memory patches, Q from generation tokens
- Integrate with Burn's tensor operations for GPU acceleration

## Phase 4: Diffusion Integration (Week 6)
- DiT backbone trait (abstract over Tract ONNX inference)
- VAE encode/decode via Tract
- Noise scheduler (DDPM/DDIM/Flow matching abstraction)
- Inject memory cross-attention into DiT forward pass
- Conditioning assembly: text + memory patches + camera (PRoPE)

## Phase 5: Autoregressive Pipeline (Week 7)
- Single-window generation: input context frames → output next frames
- Chained autoregressive: window N output → window N+1 input
- Memory update loop: add newly generated frames to point cloud
- Long-horizon generation: 2+ minute rollouts
- Keyframe selection strategy (which frames to add to memory)

## Phase 6: Polish & Release (Week 8)
- CLI with clap (generate, visualize, splice commands)
- TUI visualization: point cloud viewer, memory patch overlay, camera trajectory
- Benchmarks: memory retrieval latency, full pipeline throughput
- Documentation, examples, README
- crates.io publish, GitHub release

## Framework Choices

### Burn vs Tract
- **Burn**: All custom neural network layers (attention, RoPE, cross-attention). Supports GPU via wgpu backend.
- **Tract**: Pre-trained ONNX model inference (depth estimator, DiT backbone, VAE). CPU-optimized, can handle large models.

### nalgebra for Geometry
All 3D operations use nalgebra:
- `Isometry3<f32>` for SE(3) camera poses
- `Matrix4<f32>` for projection matrices
- `Point3<f32>` for 3D points
- `Rotation3<f32>` for camera rotations

### kiddo for Spatial Queries
KD-tree for O(log n) nearest-neighbor queries on the 3D point cloud. Critical for efficient patch retrieval — querying which stored patches are visible from a target viewpoint.
