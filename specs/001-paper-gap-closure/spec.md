# Feature Specification: Paper Gap Closure

**Feature Branch**: `001-paper-gap-closure`
**Created**: 2026-03-26
**Status**: Draft
**Input**: Gap report (`docs/specs/mosaicmem-paper-gap-report.md`) and
realization plan (`docs/specs/mosaicmem-realization-plan.md`)

## User Scenarios & Testing

### User Story 1 - Honest Dual-Backend Scaffold (Priority: P1)

The repo currently uses paper terminology for synthetic approximations.
A developer using the library MUST be able to distinguish synthetic mode
(fast, deterministic, no model dependencies) from real mode (faithful
operators, real checkpoints). Backend selection MUST be explicit in config,
CLI output, and logs.

**Why this priority**: Without truthful labeling, every subsequent
improvement risks being confused with paper fidelity. This is the
foundation for all other work.

**Independent Test**: Run `cargo run -- demo` and verify CLI output
clearly labels the active backend as `synthetic`. Verify that
`PipelineConfig` exposes a `backend` field that selects between modes.

**Acceptance Scenarios**:

1. **Given** a default config, **When** user runs any CLI command,
   **Then** output includes `[synthetic]` or `[real]` backend label.
2. **Given** a config with `backend: real` but no checkpoint available,
   **Then** the system returns a typed error, not a silent fallback.
3. **Given** the README, **When** a new reader scans it, **Then** the
   first paragraph distinguishes scaffold from paper reproduction.

---

### User Story 2 - Canonical Tensor and Patch Geometry Types (Priority: P1)

Core data flows currently use `Vec<f32>` with implicit layout assumptions
scattered across the pipeline. A developer working on alignment operators
MUST have typed tensor wrappers with explicit `[B, C, T, H, W]` layout
metadata and patch metadata rich enough for dense warping (per-token
coordinates, depth tiles, source grids).

**Why this priority**: Every subsequent operator (Warped RoPE, Warped
Latent, PRoPE) depends on correct, explicit data layout. Without this,
operator correctness cannot be verified.

**Independent Test**: Existing `tests/meaningful_end_to_end.rs` continues
to pass. New round-trip layout tests verify frame extraction, latent
extraction, and patch token ordering.

**Acceptance Scenarios**:

1. **Given** a video tensor wrapper, **When** extracting frame `t`,
   **Then** the result has shape `[C, H, W]` with correct values.
2. **Given** a patch with `patch_size=16`, **When** accessing token
   coordinates, **Then** each token has a distinct `(u, v)` in source
   frame space.
3. **Given** a latent tensor and temporal compression factor 4, **When**
   mapping latent slice `i` to original frames, **Then** the mapping is
   correct and invertible.

---

### User Story 3 - Frame-Aware Retrieval and Composition (Priority: P1)

The current pipeline retrieves memory only for `poses[0]` of each window
and replicates the result across all latent timesteps. A researcher
evaluating camera revisit behavior MUST see memory retrieval and coverage
that varies correctly as the camera moves within a window.

**Why this priority**: This is the single largest architectural mismatch
with the paper (Critical Finding 3). Memory conditioning is the paper's
core contribution; getting this wrong invalidates all downstream operators.

**Independent Test**: Generate a 2-window trajectory with 90-degree camera
turn mid-window. Verify that retrieval results and coverage masks differ
across frames within the same window.

**Acceptance Scenarios**:

1. **Given** a window with 16 poses spanning a camera pan, **When**
   retrieving memory, **Then** each latent timestep gets a distinct
   retrieval set.
2. **Given** moving camera within a window, **When** computing coverage
   masks, **Then** masks differ across frames.
3. **Given** a rasterized memory canvas, **When** the camera moves,
   **Then** the canvas is NOT temporally replicated but recomputed per
   latent slice.

---

### User Story 4 - Faithful PRoPE Operator (Priority: P2)

The current PRoPE implementation computes heuristic cosine/sine pairs
that are summarized into a scalar by the synthetic backbone. A researcher
implementing the real attention path MUST have a PRoPE operator that
computes relative projective transforms `P_i * P_j^{-1}` and applies
them inside attention Q/K/V processing as described in the paper.

**Why this priority**: PRoPE is the paper's camera conditioning
mechanism. Without it, the model has no projective awareness.

**Independent Test**: Compute PRoPE for two known camera poses and
compare against a hand-computed reference. Verify that changing camera
geometry changes attention output.

**Acceptance Scenarios**:

1. **Given** two camera poses with known intrinsics/extrinsics, **When**
   computing PRoPE, **Then** the relative projective transform matches
   the paper's `P_i * P_j^{-1}` formulation.
2. **Given** PRoPE integrated into attention, **When** camera geometry
   changes, **Then** attention scores change accordingly.
3. **Given** temporal compression, **When** a latent slice maps to
   multiple original frames, **Then** PRoPE handles subframe cameras
   correctly.

---

### User Story 5 - Faithful Warped RoPE (Priority: P2)

The current Warped RoPE uses center-only quantized positions repeated
across all tokens in a patch. A researcher studying spatial alignment
MUST have dense/fractional reprojection-based warped positions that
differ per token within a patch.

**Why this priority**: Warped RoPE is one of the paper's two alignment
mechanisms. Center-only alignment cannot represent oblique views or
intra-patch geometry.

**Independent Test**: Warp a multi-token patch from an oblique view.
Verify that different tokens receive different warped positions.

**Acceptance Scenarios**:

1. **Given** a patch with N tokens, **When** computing warped positions,
   **Then** each token gets a distinct `(u, v, t)` triple.
2. **Given** subpixel camera shifts, **When** recomputing positions,
   **Then** positions change smoothly (no quantization jumps).
3. **Given** a patch viewed obliquely, **When** comparing with frontal
   view, **Then** the warped position spread reflects the geometric
   distortion.

---

### User Story 6 - Faithful Warped Latent (Priority: P2)

The current Warped Latent uses a single planar homography from one
scalar patch depth. A researcher studying feature alignment MUST have
dense reprojection-based bilinear resampling with per-location geometry
and explicit invalid-sample masking.

**Why this priority**: Warped Latent is the paper's second alignment
mechanism. The single-plane shortcut fails for non-planar patches and
strong parallax.

**Independent Test**: Warp a patch with known depth variation between two
known camera poses. Verify that the sampling grid uses dense coordinates
and that invalid regions are masked.

**Acceptance Scenarios**:

1. **Given** a patch with per-pixel depth, **When** warping to target
   view, **Then** the sampling grid uses dense reprojected coordinates.
2. **Given** a patch partially behind the target camera, **When**
   warping, **Then** invalid samples are explicitly masked, not
   zero-filled.
3. **Given** a non-planar patch, **When** comparing dense warp vs single
   homography, **Then** the dense warp produces measurably different
   (more correct) results.

---

### User Story 7 - Real Backend Interfaces (Priority: P3)

The project currently has only synthetic backends. A developer building
the real inference path MUST have trait implementations that bridge to
actual model checkpoints for depth estimation, VAE encode/decode, and
diffusion backbone inference.

**Why this priority**: Real backends are needed to validate that the
operators actually work end-to-end with real model outputs. Deferred to
P3 because the operator correctness (P1-P2) must be established first
on the synthetic path.

**Independent Test**: Load a real checkpoint and run one forward pass
through each backend. Verify output shapes match trait contracts.

**Acceptance Scenarios**:

1. **Given** a real depth checkpoint, **When** running estimation,
   **Then** output shape and value range match the `DepthEstimator` trait
   contract.
2. **Given** a real VAE checkpoint, **When** encoding then decoding,
   **Then** the round-trip error is within expected bounds.
3. **Given** no checkpoint available, **When** selecting `real` backend,
   **Then** a typed `BackendError::CheckpointNotFound` is returned.

---

### User Story 8 - Memory Cross-Attention Wiring (Priority: P3)

The existing cross-attention module has correct structure (Q/K/V
projections, multi-head, gated residual) but uses random weights and is
not exercised in the real inference path. A researcher running the full
pipeline MUST have memory cross-attention wired into the backbone with
all alignment operators active.

**Why this priority**: This is the integration milestone that connects
all preceding operator work into a functional memory-conditioned
generation path.

**Independent Test**: Run inference with memory ON vs OFF. Verify that
outputs differ and that ablation toggles (PRoPE only, Warped RoPE only,
Warped Latent only, full) produce distinct results.

**Acceptance Scenarios**:

1. **Given** a pipeline with memory, **When** generating, **Then**
   memory cross-attention is exercised (not gated to zero).
2. **Given** ablation config `prope_only=true`, **When** generating,
   **Then** only PRoPE contributes; Warped RoPE and Warped Latent are
   disabled.
3. **Given** identical trajectories with and without memory, **When**
   comparing outputs, **Then** results are measurably different.

---

### Edge Cases

- Empty memory store at the start of the first window
- Degenerate camera pose (identity rotation, zero translation)
- Single-frame window (window_size=1, no temporal dimension)
- Memory budget exhaustion mid-rollout (eviction behavior)
- Backend timeout or crash during real inference
- Patch entirely behind the target camera (100% invalid samples)
- Trajectory with duplicate consecutive poses

## Requirements

### Functional Requirements

- **FR-001**: System MUST support explicit backend selection (`synthetic`
  vs `real`) via `PipelineConfig`.
- **FR-002**: System MUST provide typed tensor wrappers with compile-time
  or runtime layout validation for `[B, C, T, H, W]` and related shapes.
- **FR-003**: Memory retrieval MUST operate per-frame (or per-latent-
  slice) within each generation window, not only for the first pose.
- **FR-004**: PRoPE MUST compute relative projective transforms and apply
  them inside attention Q/K/V processing.
- **FR-005**: Warped RoPE MUST use dense/fractional reprojected
  coordinates, producing distinct positions per token within a patch.
- **FR-006**: Warped Latent MUST use dense reprojection-based bilinear
  resampling with explicit invalid-sample masking.
- **FR-007**: Real backend trait implementations MUST bridge to external
  model checkpoints via a Python sidecar or process bridge.
- **FR-008**: Memory cross-attention MUST be wirable into the backbone
  inference path with per-operator ablation toggles.
- **FR-009**: All CLI commands MUST display the active backend mode.
- **FR-010**: Coverage masks MUST reflect actual projected patch
  footprints, not just center-point hits.

### Key Entities

- **Patch**: 3D memory unit with center, dense coordinate grid, depth
  tile, latent tensor, source pose, provenance metadata.
- **TensorView**: Typed wrapper over `Vec<f32>` / `ndarray` carrying
  explicit layout (BCTHW, BCHW, CHW, etc.).
- **BackendMode**: Enum selecting `Synthetic` or `Real` execution path.
- **ProjectiveTransform**: Camera-to-camera relative projection matrix
  used by PRoPE.
- **WarpGrid**: Dense source-to-target coordinate mapping used by Warped
  Latent, with validity mask.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All 4 critical findings from the gap report (CF1-CF4) are
  resolved or explicitly scoped behind the `real` backend feature flag.
- **SC-002**: All 4 high findings (HF5-HF8) are resolved with
  paper-faithful implementations.
- **SC-003**: Numerical reference tests exist for PRoPE, Warped RoPE,
  and Warped Latent comparing against hand-computed values.
- **SC-004**: Frame-aware retrieval produces measurably different results
  across frames in a multi-pose window (not temporally replicated).
- **SC-005**: `cargo test` passes with all new tests, `cargo clippy`
  clean.
- **SC-006**: README accurately describes current capabilities without
  implying paper reproduction where only synthetic mode exists.

## Assumptions

- Real model checkpoints (Wan 2.2, depth estimator, VAE) will be
  available as external dependencies, not shipped with this repo.
- The Python sidecar approach for real backends is acceptable for initial
  integration; pure-Rust inference is a future optimization.
- Mosaic Forcing (the paper's autoregressive method) and the evaluation
  harness are out of scope for this feature; they are separate follow-up
  features.
- The synthetic backend remains the default and MUST continue to work
  unchanged for fast local testing.
- Training/fine-tuning pipeline is out of scope; this feature covers
  inference-path operator fidelity only.
