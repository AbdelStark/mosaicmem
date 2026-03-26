# Data Model: Paper Gap Closure

**Feature Branch**: `001-paper-gap-closure`
**Date**: 2026-03-26

## Core Entities

### TensorView

Typed wrapper over `ndarray::ArrayD<f32>` with explicit layout.

| Field | Type | Description |
| --- | --- | --- |
| `data` | `ArrayD<f32>` | Underlying n-dimensional array |
| `layout` | `TensorLayout` | Semantic layout enum |

**Layout enum variants**:
- `BCTHW` — batch, channels, time, height, width (video)
- `BCHW` — batch, channels, height, width (image/latent)
- `CTHW` — single-sample video
- `CHW` — single-sample image
- `THW` — temporal-spatial (coverage masks)
- `HW` — spatial (single mask/depth)
- `Flat(Vec<usize>)` — arbitrary shape with explicit dims

**Validation**: Shape MUST match layout semantics at construction.
Example: `BCTHW` requires `ndim == 5`.

**Operations**:
- `frame(t) -> TensorView<CHW>` — extract temporal slice
- `latent_slice(t) -> TensorView<CHW>` — extract latent timestep
- `spatial_shape() -> (H, W)` — last two dims regardless of layout

### PatchMetadata (expanded)

Existing fields preserved; new fields added for dense geometry.

| Field | Type | New? | Description |
| --- | --- | --- | --- |
| `id` | `u64` | No | Unique patch identifier |
| `center_3d` | `Point3<f32>` | No | World-space patch center |
| `depth` | `f32` | No | Scalar depth at center |
| `source_pose` | `CameraPose` | No | Camera that captured this patch |
| `source_rect` | `(u32,u32,u32,u32)` | No | `(x, y, w, h)` in source frame |
| `timestamp` | `f64` | No | Capture time |
| `latent` | `Vec<f32>` | No | Flattened latent tile |
| `token_coords` | `Vec<(f32, f32)>` | **Yes** | Per-token `(u, v)` in source frame |
| `depth_tile` | `Option<Vec<f32>>` | **Yes** | Per-token depth values |
| `source_intrinsics` | `CameraIntrinsics` | **Yes** | Source camera K matrix |
| `normal_estimate` | `Option<Vector3<f32>>` | **Yes** | Local surface normal |
| `latent_shape` | `(usize, usize, usize)` | **Yes** | `(C, H_l, W_l)` of latent tile |

**Invariants**:
- `token_coords.len() == latent_shape.1 * latent_shape.2`
- `depth_tile.map(|d| d.len()) == Some(token_coords.len())` when present
- `latent.len() == latent_shape.0 * latent_shape.1 * latent_shape.2`

### CameraIntrinsics

Camera intrinsic parameters needed for projection/unprojection.

| Field | Type | Description |
| --- | --- | --- |
| `fx` | `f64` | Focal length x |
| `fy` | `f64` | Focal length y |
| `cx` | `f64` | Principal point x |
| `cy` | `f64` | Principal point y |
| `width` | `u32` | Image width in pixels |
| `height` | `u32` | Image height in pixels |

**Operations**:
- `matrix() -> Matrix3<f64>` — 3x3 K matrix
- `inverse_matrix() -> Matrix3<f64>` — K^{-1}
- `project(Point3) -> Option<(f64, f64)>` — 3D to 2D (None if behind)
- `unproject(u, v, depth) -> Point3` — 2D + depth to 3D

### ProjectiveTransform

Relative camera-to-camera projective transform for PRoPE.

| Field | Type | Description |
| --- | --- | --- |
| `matrix` | `Matrix4<f64>` | 4x4 projective transform M_{ij} |
| `source_pose` | `CameraPose` | Camera i |
| `target_pose` | `CameraPose` | Camera j |

**Construction**: `from_cameras(cam_i, cam_j, intrinsics_i, intrinsics_j)`

**Decomposition**: `to_rope_params() -> (Vec<f64>, Vec<f64>)` — extract
rotation parameters for multiplicative Q/K transform.

### WarpGrid

Dense source-to-target coordinate mapping for Warped Latent.

| Field | Type | Description |
| --- | --- | --- |
| `target_coords` | `Vec<(f32, f32)>` | Per-token target `(u, v)` |
| `valid_mask` | `Vec<bool>` | Per-token validity (false if behind camera) |
| `source_shape` | `(usize, usize)` | `(H, W)` of source patch grid |

**Invariants**:
- `target_coords.len() == source_shape.0 * source_shape.1`
- `valid_mask.len() == target_coords.len()`

**Operations**:
- `sample_bilinear(source_latent, out) -> TensorView<CHW>` — apply warp
- `valid_ratio() -> f32` — fraction of valid samples

### BackendMode

Runtime backend selection.

| Variant | Description |
| --- | --- |
| `Synthetic` | Deterministic synthetic backends (no model deps) |
| `Real` | Real model checkpoints via Python sidecar |

Stored in `PipelineConfig::backend_mode`.

### AblationConfig

Per-operator ablation toggles for memory conditioning.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_memory` | `bool` | `true` | Master toggle |
| `enable_prope` | `bool` | `true` | PRoPE conditioning |
| `enable_warped_rope` | `bool` | `true` | Warped RoPE alignment |
| `enable_warped_latent` | `bool` | `true` | Warped Latent warping |
| `memory_gate_override` | `Option<f32>` | `None` | Force gate value |

## Entity Relationships

```
PipelineConfig
  ├── BackendMode
  ├── AblationConfig
  └── MemoryConfig

MosaicMemoryStore
  └── Vec<PatchMetadata>
       ├── CameraPose (source_pose)
       ├── CameraIntrinsics (source_intrinsics)
       └── latent tile data

RetrievalResult (per-frame)
  ├── Vec<PatchMetadata>
  ├── Vec<WarpGrid> (one per patch, target-view specific)
  └── TensorView<HW> (coverage mask)

ProjectiveTransform
  ├── CameraPose (source)
  ├── CameraPose (target)
  └── CameraIntrinsics (both)
```

## State Transitions

### Patch Lifecycle

```
Created (from keyframe + depth + VAE encode)
  → Stored (inserted into MosaicMemoryStore, KD-tree rebuilt)
  → Retrieved (selected by frustum cull + scoring for a target view)
  → Aligned (WarpGrid computed, Warped Latent applied)
  → Conditioned (fed into cross-attention as K/V tokens)
  → Decayed (temporal score decreases over time)
  → Evicted (budget enforcement removes lowest-scored)
```

### Pipeline Per-Window Flow

```
1. Slice trajectory → window poses
2. Coarse retrieval (bounding frustum over all poses)
3. For each latent timestep:
   a. Fine retrieval (score candidates against this pose)
   b. Compute WarpGrids per retrieved patch
   c. Apply Warped Latent → aligned latent tiles
   d. Rasterize into per-frame memory canvas
   e. Compute per-frame coverage mask
4. Compute PRoPE for all frame pairs
5. Compute Warped RoPE for all retrieved patches
6. Run denoising loop with memory cross-attention
7. VAE decode → frames
8. Blend overlap with previous window
9. Insert new keyframes into memory store
```
