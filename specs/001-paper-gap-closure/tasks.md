# Tasks: Paper Gap Closure

**Input**: Design documents from `specs/001-paper-gap-closure/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included. Constitution Principle IV mandates numerical reference tests, round-trip invariants, and integration tests for all operators.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing. US1 and US2 are foundational (blocking); US3-US8 can proceed after foundation completes.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths are relative to repository root

---

## Phase 1: Setup

**Purpose**: No-op for this feature. The project already has a working Cargo.toml, src/ layout, and test infrastructure. Proceed directly to foundational work.

---

## Phase 2: Foundational — US1 Backend Scaffold + US2 Data Types (Priority: P1)

**Purpose**: Establish the typed data model and backend mode selection that all subsequent user stories depend on. US1 and US2 are tightly coupled foundational work — both must complete before any operator story can begin.

**CRITICAL**: No user story work (US3-US8) can begin until this phase is complete.

### User Story 1 — Honest Dual-Backend Scaffold

**Goal**: Every CLI output and config clearly distinguishes `synthetic` vs `real` mode.

**Independent Test**: Run `cargo run -- demo` and verify `[synthetic]` label appears. Set `backend_mode: "real"` without a checkpoint and verify typed error.

- [X] T001 [P] [US1] Create BackendMode enum (Synthetic, Real) and AblationConfig struct with serde support in src/backend.rs
- [X] T002 [P] [US1] Add backend_mode and ablation fields to PipelineConfig in src/pipeline/config.rs with Default impl preserving backward compatibility
- [X] T003 [US1] Add backend label `[synthetic]`/`[real]` to all CLI command output in src/main.rs (demo, generate, inspect, bench, splice, visualize, export-ply, show-config)
- [X] T004 [US1] Add typed BackendError::CheckpointNotFound error variant in src/backend.rs and wire into pipeline startup path in src/pipeline/autoregressive.rs
- [X] T005 [US1] Update README.md first paragraph to distinguish synthetic scaffold from paper reproduction, add "Backend Modes" section

**Checkpoint**: CLI shows backend label; config roundtrips with new fields; real mode without checkpoint returns typed error.

---

### User Story 2 — Canonical Tensor and Patch Geometry Types

**Goal**: Typed tensor wrappers with layout validation; patches carry dense geometry for warping.

**Independent Test**: Existing `tests/meaningful_end_to_end.rs` passes. New round-trip tests verify frame extraction and patch token ordering.

#### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T006 [P] [US2] Write TensorView construction and layout validation tests in tests/tensor_types.rs — test BCTHW/CHW/HW construction, reject wrong ndim, verify spatial_shape()
- [X] T007 [P] [US2] Write TensorView frame extraction round-trip test in tests/tensor_types.rs — construct BCTHW tensor, extract frame(t), verify shape [C,H,W] and values
- [X] T008 [P] [US2] Write PatchMetadata invariant tests in tests/patch_geometry.rs — verify token_coords.len() == latent_shape.1 * latent_shape.2, depth_tile length consistency
- [X] T009 [P] [US2] Write CameraIntrinsics project/unproject round-trip test in tests/camera_intrinsics.rs — unproject then project a known 2D point with known depth, verify round-trip within tolerance

#### Implementation for User Story 2

- [X] T010 [P] [US2] Create TensorLayout enum and TensorView struct with construction validation in src/tensor.rs — variants: BCTHW, BCHW, CTHW, CHW, THW, HW, Flat(Vec<usize>)
- [X] T011 [P] [US2] Implement TensorView::frame(t), latent_slice(t), spatial_shape() extraction methods in src/tensor.rs
- [X] T012 [P] [US2] Create CameraIntrinsics struct with matrix(), inverse_matrix(), project(), unproject() methods in src/camera/intrinsics.rs — use f64 for all computation
- [X] T013 [US2] Add CameraIntrinsics re-export in src/camera/mod.rs and re-export TensorView and BackendMode in src/lib.rs
- [X] T014 [US2] Expand PatchMetadata in src/memory/store.rs — add token_coords: Vec<(f32,f32)>, depth_tile: Option<Vec<f32>>, source_intrinsics: CameraIntrinsics, normal_estimate: Option<Vector3<f32>>, latent_shape: (usize,usize,usize) — all new fields Option or with Default for backward compat
- [X] T015 [US2] Update PatchMetadata serialization/deserialization in src/memory/store.rs — ensure backward compat with existing JSON (serde default for new fields)
- [X] T016 [US2] Update synthetic patch creation paths in src/memory/store.rs and src/pipeline/inference.rs to populate token_coords and latent_shape from existing patch_size and source_rect

**Checkpoint**: All T006-T009 tests pass. Existing tests/meaningful_end_to_end.rs and tests/integration.rs still pass. Patches carry dense geometry.

---

## Phase 3: User Story 3 — Frame-Aware Retrieval and Composition (Priority: P1)

**Goal**: Memory retrieval varies per-frame within a window, not replicated from first pose.

**Independent Test**: Generate a multi-pose window with 90-degree camera turn. Verify retrieval sets and coverage masks differ across frames.

### Tests for User Story 3

- [ ] T017 [P] [US3] Write coarse/fine retrieval test in tests/frame_retrieval.rs — insert patches visible from different angles, retrieve for 3 distinct poses in a window, assert different top-K sets per pose
- [ ] T018 [P] [US3] Write coverage mask variation test in tests/frame_retrieval.rs — verify coverage masks differ across frames when camera pans within a window
- [ ] T019 [P] [US3] Write edge case tests in tests/frame_retrieval.rs — empty memory store returns empty results, single-frame window works, budget exhaustion mid-retrieval, duplicate consecutive poses

### Implementation for User Story 3

- [ ] T020 [US3] Implement bounding frustum computation from a set of window poses in src/memory/retrieval.rs — union of per-pose frustums for coarse candidate selection
- [ ] T021 [US3] Add FrameRetrievalResult struct and MemoryRetriever trait in src/memory/retrieval.rs per contracts/trait-contracts.md
- [ ] T022 [US3] Implement coarse/fine two-tier retrieval: coarse pass queries KD-tree once per window, fine pass scores candidates per-frame with temporal decay and diversity in src/memory/retrieval.rs
- [ ] T023 [US3] Refactor LatentCanvas in src/memory/mosaic.rs — remove temporal replication in to_cthw(), replace with per-frame rasterization using projected patch footprints
- [ ] T024 [US3] Implement per-frame coverage mask computation in src/memory/mosaic.rs — coverage reflects actual projected footprint, not just center-point hits
- [ ] T025 [US3] Refactor generate_window in src/pipeline/inference.rs — replace single-pose retrieval (poses[0]) with per-latent-slice retrieval loop, wire AblationConfig into the inference path
- [ ] T026 [US3] Update autoregressive pipeline in src/pipeline/autoregressive.rs to pass AblationConfig and CameraIntrinsics through to generate_window

**Checkpoint**: T017-T019 tests pass. Within a single window, different latent timesteps get distinct retrieval results and coverage masks. Existing E2E tests still pass.

---

## Phase 4: User Story 4 — Faithful PRoPE Operator (Priority: P2)

**Goal**: PRoPE computes P_i * P_j^{+} and applies it multiplicatively in attention Q/K.

**Independent Test**: Compute PRoPE for two known camera poses, compare against hand-computed reference values.

### Tests for User Story 4

- [ ] T027 [P] [US4] Write PRoPE projection matrix construction test in tests/prope_reference.rs — verify P = K * [R|t] against hand-computed 3x4 matrix for a known camera
- [ ] T028 [P] [US4] Write PRoPE relative transform test in tests/prope_reference.rs — verify M_{ij} = P_i * P_j^{+} for two known cameras, compare against reference
- [ ] T029 [P] [US4] Write PRoPE identity test in tests/prope_reference.rs — verify identical source/target cameras produce identity transform (within tolerance)
- [ ] T030 [P] [US4] Write PRoPE attention sensitivity test in tests/prope_reference.rs — verify changing camera geometry changes attention Q/K output

### Implementation for User Story 4

- [ ] T031 [US4] Implement ProjectiveTransform struct with from_cameras() constructor in src/attention/prope.rs — compute P_i = K_i * [R_i|t_i] (3x4), then M_{ij} = P_i * P_j^{+} using pseudo-inverse
- [ ] T032 [US4] Implement ProjectiveTransform::to_rope_params() decomposition in src/attention/prope.rs — extract rotation parameters for multiplicative Q/K transform
- [ ] T033 [US4] Implement PRoPEOperator trait: compute_projective_transform() and apply_to_attention() in src/attention/prope.rs — apply multiplicatively (rotary) to Q/K, not additively
- [ ] T034 [US4] Add temporal compression subframe handling in src/attention/prope.rs — when a latent slice maps to multiple original frames, compute per-subframe cameras and aggregate

**Checkpoint**: T027-T030 tests pass. PRoPE produces correct relative transforms. Attention output changes when cameras change.

---

## Phase 5: User Story 5 — Faithful Warped RoPE (Priority: P2)

**Goal**: Dense/fractional reprojection-based warped positions that differ per token within a patch.

**Independent Test**: Warp a multi-token patch from an oblique view, verify distinct per-token positions.

### Tests for User Story 5

- [ ] T035 [P] [US5] Write dense reprojection golden test in tests/warped_rope_dense.rs — reproject a 4x4 token grid from known source to known target, compare against hand-computed (u,v,t) triples
- [ ] T036 [P] [US5] Write intra-patch distinctness test in tests/warped_rope_dense.rs — verify all tokens in a patch get distinct warped positions (no center-only replication)
- [ ] T037 [P] [US5] Write fractional coordinate stability test in tests/warped_rope_dense.rs — apply subpixel camera shift, verify positions change smoothly (max delta proportional to shift)

### Implementation for User Story 5

- [ ] T038 [US5] Implement dense coordinate reprojection in src/attention/warped_rope.rs — for each token in patch, unproject token_coord with depth_tile to 3D, transform to world, project to target view, preserve fractional coordinates
- [ ] T039 [US5] Replace center-only quantized position computation with per-token dense positions in src/attention/warped_rope.rs — remove integer bin quantization, use fractional (u, v) directly
- [ ] T040 [US5] Implement richer temporal coordinate handling in src/attention/warped_rope.rs — replace coarse age-to-bin scaling with continuous temporal offset normalized by half-life
- [ ] T041 [US5] Update WarpedRoPE integration in src/attention/memory_cross.rs — pass token_coords and depth_tile from PatchMetadata, use CameraIntrinsics for reprojection

**Checkpoint**: T035-T037 tests pass. Each token in a patch receives a distinct warped position. Oblique views produce wider position spread.

---

## Phase 6: User Story 6 — Faithful Warped Latent (Priority: P2)

**Goal**: Dense reprojection-based bilinear resampling with per-location geometry and invalid-sample masking.

**Independent Test**: Warp a patch with known depth variation between two known cameras. Verify dense sampling grid and masked invalid regions.

### Tests for User Story 6

- [ ] T042 [P] [US6] Write dense warp geometric golden test in tests/warped_latent_dense.rs — warp a 4x4 patch between two known cameras with known depth, compare target coords against hand-computed values
- [ ] T043 [P] [US6] Write identity warp test in tests/warped_latent_dense.rs — same source/target pose produces approximately identity warp (warped latent matches source latent within tolerance)
- [ ] T044 [P] [US6] Write invalid sample masking test in tests/warped_latent_dense.rs — place a patch partially behind target camera, verify behind-camera samples are masked (valid_mask = false), not zero-filled

### Implementation for User Story 6

- [ ] T045 [US6] Implement WarpGrid struct in src/attention/warped_latent.rs — target_coords, valid_mask, source_shape per data-model.md, with valid_ratio() method
- [ ] T046 [US6] Implement dense warp grid construction in src/attention/warped_latent.rs — for each token (u_s, v_s) with depth d_s: unproject to 3D via K_s^{-1}, transform to world, project to target via K_t, mark invalid if z_t <= 0
- [ ] T047 [US6] Implement bilinear resampling in src/attention/warped_latent.rs — WarpGrid::sample_bilinear() applies the computed sampling grid to source latent tile, zero-fills invalid samples
- [ ] T048 [US6] Implement WarpOperator trait (compute_warp_grid + apply_warp) in src/attention/warped_latent.rs per contracts/trait-contracts.md
- [ ] T049 [US6] Update Warped Latent integration in src/pipeline/inference.rs — replace single-plane homography call with WarpOperator::compute_warp_grid() + apply_warp() using dense patch geometry

**Checkpoint**: T042-T044 tests pass. Dense warp uses per-token depth. Invalid regions are masked. Identity warp preserves latent.

---

## Phase 7: User Story 7 — Real Backend Interfaces (Priority: P3)

**Goal**: Trait implementations bridging to real model checkpoints behind `real-backend` feature flag.

**Independent Test**: With `real-backend` feature enabled and checkpoint available, run one forward pass through each backend. Without checkpoint, verify typed error.

### Tests for User Story 7

- [ ] T050 [P] [US7] Write backend bridge request/response schema tests in tests/backend_bridge.rs — verify serialization/deserialization of tensor payloads with shape metadata round-trips correctly
- [ ] T051 [P] [US7] Write BackendMode config serde test in tests/backend_bridge.rs — verify PipelineConfig with backend_mode: Real serializes/deserializes; verify checkpoint-not-found error when real mode selected without checkpoint path

### Implementation for User Story 7

- [ ] T052 [US7] Define BackendBridge trait in src/backend.rs — methods for health_check(), infer_depth(), infer_vae_encode(), infer_vae_decode(), infer_denoise(), all returning Result with BackendError
- [ ] T053 [US7] Implement request/response types in src/backend.rs — TensorPayload (flat data + shape + dtype), BackendRequest, BackendResponse with serde support
- [ ] T054 [US7] Implement SyntheticBridge (always available) that delegates to existing Synthetic* structs in src/backend.rs
- [ ] T055 [US7] Implement PythonSidecarBridge (behind #[cfg(feature = "real-backend")]) in src/backend.rs — spawn Python process, communicate via JSON-over-stdio, handle timeouts
- [ ] T056 [US7] Add `real-backend` feature to Cargo.toml with conditional dependencies; wire BackendBridge into pipeline startup in src/pipeline/autoregressive.rs

**Checkpoint**: T050-T051 pass. Synthetic bridge works transparently. Real bridge compiles behind feature flag. Missing checkpoint returns typed error.

---

## Phase 8: User Story 8 — Memory Cross-Attention Wiring (Priority: P3)

**Goal**: Memory cross-attention wired into backbone with all alignment operators active and per-operator ablation toggles.

**Independent Test**: Run inference with memory ON vs OFF and verify outputs differ. Toggle individual operators and verify distinct results.

### Tests for User Story 8

- [ ] T057 [P] [US8] Write memory cross-attention wiring test in tests/integration.rs — verify that with enable_memory=true, cross-attention is exercised (output differs from enable_memory=false)
- [ ] T058 [P] [US8] Write ablation toggle test in tests/integration.rs — verify each combination (prope_only, warped_rope_only, warped_latent_only, full) produces distinct outputs
- [ ] T059 [P] [US8] Write memory gate test in tests/integration.rs — verify memory_gate_override=Some(0.0) produces same output as enable_memory=false

### Implementation for User Story 8

- [ ] T060 [US8] Define MemoryContext struct in src/attention/memory_cross.rs — carries retrieved patches, WarpGrids, PRoPE transforms, warped RoPE positions, rasterized canvas, coverage masks, and AblationConfig
- [ ] T061 [US8] Wire PRoPE into attention Q/K processing in src/attention/memory_cross.rs — call PRoPEOperator::apply_to_attention() on queries and memory keys before dot-product, gated by AblationConfig::enable_prope
- [ ] T062 [US8] Wire Warped RoPE into memory key encoding in src/attention/memory_cross.rs — apply per-token warped positions to memory keys, gated by AblationConfig::enable_warped_rope
- [ ] T063 [US8] Wire Warped Latent into memory value preparation in src/attention/memory_cross.rs — apply WarpOperator results to memory values before cross-attention, gated by AblationConfig::enable_warped_latent
- [ ] T064 [US8] Update DiffusionBackbone trait signature in src/diffusion/backbone.rs — add memory_context: Option<&MemoryContext> parameter, update SyntheticBackbone to accept and use it
- [ ] T065 [US8] Wire MemoryContext assembly into generate_window in src/pipeline/inference.rs — collect all per-frame retrieval results, WarpGrids, PRoPE transforms, and pass as MemoryContext to backbone.denoise()

**Checkpoint**: T057-T059 pass. Memory conditioning measurably affects output. Each ablation toggle produces distinct results. Gate override works.

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Regression validation, documentation, and cleanup across all stories.

- [ ] T066 [P] Verify all existing tests pass — run cargo test and confirm tests/meaningful_end_to_end.rs and tests/integration.rs green with no regressions
- [ ] T067 [P] Run cargo clippy -- -D warnings and fix any new warnings across all modified files
- [ ] T068 [P] Verify no unsafe blocks outside of feature-gated FFI — grep for unsafe in src/ and confirm each has a // SAFETY: comment or is behind #[cfg(feature = "real-backend")]
- [ ] T069 [P] Add module-level //! doc comments to new files: src/tensor.rs, src/backend.rs, src/camera/intrinsics.rs — explain purpose and relationship to paper
- [ ] T070 Update CLI --help text in src/main.rs to mention backend mode selection and ablation config
- [ ] T071 Run quickstart.md validation — execute all commands from specs/001-paper-gap-closure/quickstart.md and verify they succeed

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: Skipped — already initialized
- **Phase 2 (Foundation: US1 + US2)**: No dependencies — start immediately. **BLOCKS all other stories.**
- **Phase 3 (US3)**: Depends on Phase 2 (needs CameraIntrinsics, expanded PatchMetadata, AblationConfig)
- **Phase 4 (US4)**: Depends on Phase 2 (needs CameraIntrinsics, TensorView). Independent of US3.
- **Phase 5 (US5)**: Depends on Phase 2 (needs expanded PatchMetadata, CameraIntrinsics). Independent of US3, US4.
- **Phase 6 (US6)**: Depends on Phase 2 (needs expanded PatchMetadata, CameraIntrinsics). Independent of US3-US5.
- **Phase 7 (US7)**: Depends on Phase 2 (needs BackendMode, TensorView). Independent of US3-US6.
- **Phase 8 (US8)**: Depends on US3 (frame-aware retrieval), US4 (PRoPE), US5 (Warped RoPE), US6 (Warped Latent). This is the integration milestone.
- **Phase 9 (Polish)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 (P1)**: Foundation — no dependencies on other stories
- **US2 (P1)**: Foundation — no dependencies on other stories; co-developed with US1
- **US3 (P1)**: Depends on US1+US2 foundation only. Independently testable.
- **US4 (P2)**: Depends on US1+US2 foundation only. Independently testable.
- **US5 (P2)**: Depends on US1+US2 foundation only. Independently testable.
- **US6 (P2)**: Depends on US1+US2 foundation only. Independently testable.
- **US7 (P3)**: Depends on US1+US2 foundation only. Independently testable.
- **US8 (P3)**: Depends on US3 + US4 + US5 + US6. Integration milestone.

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Types/structs before logic
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All US1 tasks T001-T002 can run in parallel (different files)
- All US2 tests T006-T009 can run in parallel (different test files)
- All US2 implementation tasks T010-T012 can run in parallel (different files)
- After Phase 2: US3, US4, US5, US6, US7 can ALL proceed in parallel (independent files and operators)
- Within each operator story: all test tasks marked [P] can run in parallel

---

## Parallel Example: After Foundation (Phase 2)

```text
# Five independent story tracks can run simultaneously:
Track A (US3): T017-T019 tests → T020-T026 implementation
Track B (US4): T027-T030 tests → T031-T034 implementation
Track C (US5): T035-T037 tests → T038-T041 implementation
Track D (US6): T042-T044 tests → T045-T049 implementation
Track E (US7): T050-T051 tests → T052-T056 implementation

# After all five tracks complete:
Track F (US8): T057-T059 tests → T060-T065 implementation

# Finally:
Track G (Polish): T066-T071
```

---

## Implementation Strategy

### MVP First (US1 + US2 + US3)

1. Complete Phase 2: Foundation (US1 + US2)
2. Complete Phase 3: Frame-Aware Retrieval (US3)
3. **STOP and VALIDATE**: The single largest paper gap (CF3: first-pose-only retrieval) is fixed
4. This is independently demoable: `cargo run -- demo` shows per-frame coverage variation

### Incremental Delivery

1. Foundation (US1+US2) -> Backend labeling + typed data model
2. Add US3 -> Frame-aware retrieval (biggest architectural fix)
3. Add US4+US5+US6 in parallel -> All three faithful operators
4. Add US7 -> Real backend bridge ready
5. Add US8 -> Full integration milestone (memory cross-attention wired)
6. Polish -> Docs, regression, cleanup

### Parallel Team Strategy

With multiple developers after Phase 2:

- Developer A: US3 (frame-aware retrieval)
- Developer B: US4 (PRoPE)
- Developer C: US5 + US6 (Warped RoPE + Warped Latent — related math)
- Developer D: US7 (backend bridge)
- All converge on US8 (integration)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable (except US8 which integrates all)
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All numerical tests use `approx` crate with explicit tolerance bounds
