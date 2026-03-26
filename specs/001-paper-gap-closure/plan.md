# Implementation Plan: Paper Gap Closure

**Branch**: `001-paper-gap-closure` | **Date**: 2026-03-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-paper-gap-closure/spec.md`

## Summary

Close the 13 gaps identified in `docs/specs/mosaicmem-paper-gap-report.md`
by implementing paper-faithful operators for PRoPE, Warped RoPE, Warped
Latent, frame-aware retrieval, and memory cross-attention integration.
Introduce typed tensor wrappers, dense patch geometry, explicit backend
mode selection, and per-operator ablation. Preserve the synthetic backend
as the fast-test path while adding a real backend interface behind a
Cargo feature flag.

## Technical Context

**Language/Version**: Rust (edition 2024), minimum rustc 1.75
**Primary Dependencies**: nalgebra 0.34, ndarray 0.16, kiddo 4, rayon 1.10, serde 1, clap 4, thiserror 2.0, ratatui 0.30
**Storage**: JSON serialization for memory stores, trajectories, configs
**Testing**: `cargo test` with `approx` 0.5 for numerical tolerance
**Target Platform**: macOS / Linux (development), no GPU requirement for synthetic mode
**Project Type**: Library + CLI binary
**Performance Goals**: Synthetic pipeline demo < 5s for 32 frames at 64x64; real backend latency is external
**Constraints**: No `unsafe` except feature-gated FFI; `cargo clippy -D warnings` clean
**Scale/Scope**: ~15 source files modified, ~5 new files, ~800-1200 lines new code

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
| --- | --- | --- |
| I. Paper Fidelity | Every operator maps to a paper section; deviations documented | **PASS** — spec maps each US to paper sections; research documents operator formulations |
| I. Paper Fidelity | Trait interfaces match paper's mathematical contracts | **PASS** — contracts/trait-contracts.md defines shape and behavioral contracts |
| I. Paper Fidelity | Naming uses paper terminology | **PASS** — PRoPE, Warped RoPE, Warped Latent names preserved |
| II. Technical Soundness | SE3 via unit quaternions, no Euler angles | **PASS** — existing CameraPose already uses `UnitQuaternion`; no changes needed |
| II. Technical Soundness | f64 for accumulation-sensitive paths | **PASS** — CameraIntrinsics and ProjectiveTransform use f64; patch coords use f32 (acceptable for spatial data) |
| II. Technical Soundness | Correct attention scaling | **PASS** — cross-attention contract specifies `1/sqrt(d_k)` |
| III. Code Quality | No unsafe outside FFI | **PASS** — real backend uses process bridge, not FFI; no unsafe needed |
| III. Code Quality | Result-based error handling | **PASS** — new traits return typed errors |
| III. Code Quality | Trait-driven design | **PASS** — new traits: MemoryRetriever, WarpOperator, PRoPEOperator |
| IV. Testing Standards | Numerical reference tests | **PASS** — spec requires hand-computed reference values for all operators |
| IV. Testing Standards | Round-trip invariants | **PASS** — warp identity test, encode/decode round-trip |
| IV. Testing Standards | Integration tests | **PASS** — frame-aware retrieval, full pipeline with ablation |
| ML Constraints | Typed tensor layout | **PASS** — TensorView with Layout enum |
| ML Constraints | Reproducibility | **PASS** — existing seed-based determinism preserved |
| ML Constraints | Config-driven | **PASS** — AblationConfig, BackendMode in PipelineConfig |
| Dev Workflow | Clippy clean | **PASS** — enforced as existing CI gate |

No violations. All gates pass.

## Project Structure

### Documentation (this feature)

```text
specs/001-paper-gap-closure/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research output
├── data-model.md        # Phase 1 data model
├── quickstart.md        # Phase 1 quickstart guide
├── contracts/
│   └── trait-contracts.md  # Trait interface contracts
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── camera/
│   ├── mod.rs               # existing
│   ├── pose.rs              # existing
│   ├── trajectory.rs        # existing
│   ├── keyframe.rs          # existing
│   └── intrinsics.rs        # NEW — CameraIntrinsics type
├── geometry/
│   ├── mod.rs               # existing
│   ├── depth.rs             # existing — add TensorView output
│   ├── projection.rs        # existing
│   ├── point_cloud.rs       # existing
│   └── fusion.rs            # existing
├── memory/
│   ├── mod.rs               # existing
│   ├── store.rs             # MODIFIED — expanded PatchMetadata
│   ├── retrieval.rs         # MODIFIED — frame-aware retrieval
│   ├── mosaic.rs            # MODIFIED — per-frame canvas/coverage
│   └── manipulation.rs      # existing
├── attention/
│   ├── mod.rs               # existing
│   ├── rope.rs              # existing
│   ├── warped_rope.rs       # MODIFIED — dense/fractional positions
│   ├── prope.rs             # MODIFIED — real projective transform
│   ├── warped_latent.rs     # MODIFIED — dense reprojection warp
│   └── memory_cross.rs      # MODIFIED — wiring + ablation
├── diffusion/
│   ├── mod.rs               # existing
│   ├── backbone.rs          # existing — add MemoryContext param
│   ├── scheduler.rs         # existing
│   └── vae.rs               # existing — TensorView interface
├── pipeline/
│   ├── mod.rs               # existing
│   ├── config.rs            # MODIFIED — BackendMode, AblationConfig
│   ├── inference.rs         # MODIFIED — per-frame retrieval loop
│   └── autoregressive.rs    # existing — minor: pass AblationConfig
├── tensor.rs                # NEW — TensorView, TensorLayout
├── backend.rs               # NEW — BackendMode, backend bridge trait
└── lib.rs                   # MODIFIED — re-export new modules

tests/
├── integration.rs           # MODIFIED — add frame-aware tests
├── meaningful_end_to_end.rs # MODIFIED — verify ablation toggles
├── prope_reference.rs       # NEW — numerical reference tests
├── warped_rope_dense.rs     # NEW — dense reprojection tests
├── warped_latent_dense.rs   # NEW — dense warp tests
└── frame_retrieval.rs       # NEW — per-frame retrieval tests
```

**Structure Decision**: Single project (library + binary). No new crates
or workspace members. New files are limited to `tensor.rs`, `backend.rs`,
`camera/intrinsics.rs`, and 4 test files.

## Complexity Tracking

No constitution violations to justify.

## Phase 0 Output

Research complete. See [research.md](research.md).

All NEEDS CLARIFICATION items resolved:
- Tensor strategy: `ndarray::ArrayD` newtypes (RQ-1)
- Patch geometry: dense per-token coords + depth tile (RQ-2)
- Frame retrieval: coarse/fine two-tier (RQ-3)
- PRoPE formulation: full projective transform (RQ-4)
- Warped Latent: dense unproject-transform-project (RQ-5)
- Real backend: JSON-over-stdio Python sidecar (RQ-6)
- Mode selection: Cargo feature flag + runtime enum (RQ-7)

## Phase 1 Output

Design complete. See:
- [data-model.md](data-model.md) — entity definitions and relationships
- [contracts/trait-contracts.md](contracts/trait-contracts.md) — trait
  interface contracts
- [quickstart.md](quickstart.md) — build and verification guide

### Post-Design Constitution Re-Check

All gates still pass. No new violations introduced by the design.

Key design decisions validated against constitution:
- TensorView uses `ndarray` (ML Constraint: typed layout)
- PatchMetadata expanded with dense geometry (Principle I: paper fidelity)
- New traits for retrieval, warp, PRoPE (Principle III: trait-driven)
- All trait methods return `Result` (Principle III: error handling)
- f64 for projective transforms (Principle II: numerical precision)
- AblationConfig for per-operator toggles (ML Constraint: config-driven)

## Implementation Phases (for /speckit.tasks)

### Phase 1: Foundation (blocking)

1. **TensorView module** (`src/tensor.rs`)
   - TensorLayout enum, TensorView struct, construction validation
   - Frame/slice extraction methods
   - Conversion from existing `Vec<f32>` + shape patterns

2. **CameraIntrinsics** (`src/camera/intrinsics.rs`)
   - K matrix construction, project/unproject methods
   - Serialization

3. **Expanded PatchMetadata** (`src/memory/store.rs`)
   - Add token_coords, depth_tile, source_intrinsics, normal_estimate,
     latent_shape fields
   - Backward-compatible: new fields are `Option` or have defaults
   - Update serialization

4. **BackendMode + AblationConfig** (`src/pipeline/config.rs`,
   `src/backend.rs`)
   - BackendMode enum, AblationConfig struct
   - Add to PipelineConfig with serde support
   - CLI backend label output

### Phase 2: Frame-Aware Retrieval (US3 -- P1)

5. **Coarse/fine retrieval** (`src/memory/retrieval.rs`)
   - Bounding frustum computation from window poses
   - Per-frame scoring and top-K selection
   - MemoryRetriever trait implementation

6. **Per-frame canvas and coverage** (`src/memory/mosaic.rs`)
   - Remove temporal replication in `LatentCanvas::to_cthw`
   - Per-frame rasterization with projected footprints
   - Per-frame coverage masks

7. **Pipeline integration** (`src/pipeline/inference.rs`)
   - Replace single-pose retrieval with per-latent-slice loop
   - Wire AblationConfig through inference path

### Phase 3: Faithful Operators (US4-US6 -- P2)

8. **Faithful PRoPE** (`src/attention/prope.rs`)
   - Full projection matrix P_i = K * [R|t]
   - Relative transform M_{ij} = P_i * P_j^{+}
   - Multiplicative application to Q/K in attention
   - Temporal compression subframe handling

9. **Dense Warped RoPE** (`src/attention/warped_rope.rs`)
   - Dense coordinate reprojection per token
   - Fractional position preservation
   - Per-token distinct warped positions
   - Integration with memory_cross.rs

10. **Dense Warped Latent** (`src/attention/warped_latent.rs`)
    - WarpGrid construction via dense unproject-transform-project
    - Bilinear resampling
    - Invalid sample masking
    - WarpOperator trait implementation

### Phase 4: Integration (US7-US8 -- P3)

11. **Real backend interface** (`src/backend.rs`)
    - Python sidecar bridge trait (behind `real-backend` feature)
    - Request/response types with tensor serialization
    - Typed error handling and timeouts

12. **Memory cross-attention wiring** (`src/attention/memory_cross.rs`)
    - Wire PRoPE into attention Q/K processing
    - Wire Warped RoPE into memory keys
    - Wire Warped Latent into memory values
    - AblationConfig respects per-operator toggles
    - MemoryContext type carrying all conditioning signals

### Phase 5: Testing and Polish

13. **Numerical reference tests** (`tests/prope_reference.rs`,
    `tests/warped_rope_dense.rs`, `tests/warped_latent_dense.rs`)
    - Hand-computed reference values for known camera pairs
    - Tolerance bounds using `approx`

14. **Frame retrieval tests** (`tests/frame_retrieval.rs`)
    - Multi-pose window with known retrieval drift
    - Coverage variation assertion
    - Edge cases: empty store, single frame, budget exhaustion

15. **Integration and ablation tests** (update existing test files)
    - End-to-end with ablation toggles
    - Backend label verification
    - Regression: existing tests still pass

16. **Documentation alignment**
    - Update README to distinguish synthetic/real
    - Update AGENTS.md if present
    - Verify CLI help text includes backend info
