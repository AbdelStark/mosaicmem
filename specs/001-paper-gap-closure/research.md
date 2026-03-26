# Research: Paper Gap Closure

**Feature Branch**: `001-paper-gap-closure`
**Date**: 2026-03-26

## Source Documents

- Gap report: `docs/specs/mosaicmem-paper-gap-report.md`
- Realization plan: `docs/specs/mosaicmem-realization-plan.md`
- Paper: arXiv:2603.17117 (MosaicMem: Hybrid Spatial Memory for
  Controllable Video World Models)

## Research Questions and Findings

### RQ-1: Typed Tensor Strategy for Rust

**Question**: How should we implement typed tensor wrappers in Rust that
carry layout metadata without excessive runtime overhead?

**Decision**: Use newtype wrappers around `ndarray::ArrayD<f32>` (or
`ArrayD<f64>` where precision matters) with a `Layout` enum stored at
construction time. Validate shape against layout at construction, not on
every access.

**Rationale**:
- `ndarray` is already a dependency and supports dynamic-rank arrays
- Newtype wrappers add zero runtime cost after construction validation
- `nalgebra` is for fixed-size linear algebra (poses, transforms);
  `ndarray` is for variable-size tensors (frames, latents)
- Alternative: raw `Vec<f32>` with shape tuples — rejected because it
  duplicates what `ndarray` already provides and lacks stride safety

**Alternatives considered**:
- `tch-rs` (Rust PyTorch bindings): adds massive dependency, not needed
  for data representation
- Custom `Tensor<const RANK: usize>`: Rust const generics not mature
  enough for rank-polymorphic operations
- Keep `Vec<f32>`: rejected — this is exactly the problem causing layout
  bugs

### RQ-2: Dense Patch Geometry Representation

**Question**: What geometry must each patch carry to support dense warping
and per-token alignment?

**Decision**: Expand `PatchMetadata` to include:
- `center_3d: Point3<f32>` (existing)
- `token_coords: Vec<(f32, f32)>` — 2D coordinates of each token in
  source frame space (grid of `patch_size x patch_size` positions)
- `depth_tile: Option<Vec<f32>>` — per-token depth values from source
  depth map
- `source_intrinsics: CameraIntrinsics` — camera parameters at capture
  time
- `normal_estimate: Option<Vector3<f32>>` — local surface normal for
  better warping

**Rationale**:
- Per-token coordinates are needed by Warped RoPE for dense reprojection
- Per-token depth is needed by Warped Latent for dense sampling grids
- Source intrinsics are needed to unproject token coordinates into 3D
- The gap report (HF5, HF6, HF7) all trace back to insufficient patch
  geometry

**Alternatives considered**:
- Store only corners + interpolate: saves memory but loses depth
  variation within patch
- Store full 3D point cloud per patch: excessive memory for what the
  operators need
- Store depth tile as separate structure: rejected for cache locality

### RQ-3: Frame-Aware Retrieval Architecture

**Question**: How should per-frame retrieval within a window be
structured without making the hot path N times slower?

**Decision**: Implement a two-tier retrieval strategy:
1. **Coarse pass**: Retrieve a superset of candidate patches using the
   window's bounding frustum (union of all frame frustums). This is done
   once per window.
2. **Fine pass**: For each latent timestep, score and filter the
   candidate set against that timestep's specific pose. This produces
   per-frame top-K sets.

**Rationale**:
- The coarse pass amortizes KD-tree queries
- The fine pass is just scoring/filtering, not a full spatial query
- This matches the paper's intent: retrieval is view-dependent, but the
  spatial index is shared
- Temporal compression: if latent has T/4 timesteps and each maps to 4
  frames, use the middle frame's pose as representative for that slice

**Alternatives considered**:
- Full independent retrieval per frame: too slow for large memory stores
- Retrieve once and replicate (current behavior): rejected — this is
  the bug we are fixing
- Per-frame KD-tree queries with caching: unnecessary complexity when
  coarse/fine split handles it

### RQ-4: PRoPE Implementation Path

**Question**: What is the paper's actual PRoPE formulation and how
should it be implemented in Rust?

**Decision**: Implement PRoPE as described in the paper:
1. Construct full projection matrix `P_i = K_i * [R_i | t_i]` for each
   camera (3x4 matrix)
2. Compute relative projective transform `M_{ij} = P_i * P_j^{+}` where
   `P_j^{+}` is the pseudo-inverse of `P_j`
3. Decompose `M_{ij}` into rotation parameters applied as a
   multiplicative transform on Q/K vectors inside attention
4. Handle temporal compression by computing per-subframe cameras and
   aggregating within each latent timestep

**Rationale**:
- The paper explicitly defines PRoPE as a projective operator in
  attention space
- The current heuristic (arbitrary elements of `K * R` + translation)
  does not compute `P_i * P_j^{-1}` at all
- The pseudo-inverse approach handles the 3x4 non-square case

**Alternatives considered**:
- Approximate with Plucker coordinates: the paper explicitly goes beyond
  Plucker; this would be a known simplification
- Apply PRoPE as additive bias: paper uses multiplicative rotary
  transform, not additive

### RQ-5: Warped Latent Dense Reprojection

**Question**: How should dense source-to-target warping be implemented
without the single-homography shortcut?

**Decision**: Implement dense reprojection pipeline:
1. For each token `(u_s, v_s)` in source patch with depth `d_s`:
   - Unproject to 3D: `X = K_s^{-1} * [u_s, v_s, 1]^T * d_s`
   - Transform to world: `X_w = R_s^T * (X - t_s)`
   - Project to target: `x_t = K_t * (R_t * X_w + t_t)`
   - Normalize: `(u_t, v_t) = (x_t[0]/x_t[2], x_t[1]/x_t[2])`
2. Build a sampling grid of target coordinates
3. Apply bilinear interpolation on the source latent tile
4. Mark samples with `x_t[2] <= 0` as invalid (behind camera)

**Rationale**:
- This is the standard dense reprojection approach used in multi-view
  geometry
- The single-homography shortcut (current code) assumes fronto-parallel
  plane at uniform depth — incorrect for most real scenes
- Per-token depth from `depth_tile` makes this possible

**Alternatives considered**:
- Mesh-based warping (triangulate patch, render): overkill for patch-size
  tiles
- Optical flow estimation between views: adds model dependency for
  something geometry can solve exactly
- Keep homography for small patches, dense for large: arbitrary threshold;
  just use dense everywhere

### RQ-6: Real Backend Bridge Strategy

**Question**: What is the best way to bridge Rust orchestration to
Python model inference?

**Decision**: Use a gRPC or JSON-over-stdio process bridge:
- Rust spawns a Python sidecar process at startup
- Communication via stdin/stdout JSON messages (simpler) or gRPC
  (if latency matters)
- Request/response payloads carry serialized tensors as flat arrays with
  shape metadata
- Timeout and error handling in Rust; Python process is stateless per
  request

**Rationale**:
- The realization plan recommends Python sidecar (Phase 2)
- Wan 2.2, depth estimators, VAE are all Python/PyTorch native
- JSON-over-stdio is simplest to implement and debug
- gRPC can be added later if serialization overhead matters

**Alternatives considered**:
- PyO3 (Rust-Python FFI): complex GIL management, harder to debug
- ONNX Runtime in Rust: not all paper models have ONNX exports
- REST API: unnecessary networking for same-machine communication
- Shared memory: premature optimization

### RQ-7: Backend Mode and Feature Flag Design

**Question**: How should synthetic vs real mode be structured in the
codebase?

**Decision**: Use a Cargo feature flag `real-backend` combined with a
runtime `BackendMode` enum:
- Feature flag `real-backend` gates the Python bridge dependency and real
  backend implementations at compile time
- `BackendMode::Synthetic` is always available (default)
- `BackendMode::Real` requires the `real-backend` feature
- `PipelineConfig` carries a `backend_mode` field
- CLI prints `[synthetic]` or `[real]` in all output

**Rationale**:
- Cargo features prevent pulling in heavy dependencies for synthetic-only
  users
- Runtime enum allows config-file-driven selection when both are compiled
- This pattern is idiomatic Rust (see `reqwest` with `native-tls` vs
  `rustls` features)

**Alternatives considered**:
- Runtime-only selection: forces all users to compile real backend deps
- Compile-time-only selection: can't switch modes via config
- Separate binaries: fragment the user experience

## Technology Decisions Summary

| Area | Decision | Confidence |
| --- | --- | --- |
| Tensor types | `ndarray::ArrayD` newtypes | High |
| Patch geometry | Dense per-token coords + depth tile | High |
| Frame retrieval | Coarse/fine two-tier | High |
| PRoPE | Full projective transform in attention | High |
| Warped Latent | Dense unproject-transform-project pipeline | High |
| Real backend | JSON-over-stdio Python sidecar | Medium |
| Mode selection | Cargo feature flag + runtime enum | High |
