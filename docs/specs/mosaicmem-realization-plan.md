# MosaicMem Realization Plan

Date: 2026-03-26

This document is the execution plan for turning `mosaicmem` from a synthetic
scaffold into:

1. a paper-faithful MosaicMem implementation
2. a reproducible evaluation stack
3. a base that can later be hardened for production use

This plan is intentionally strict.
The project should not claim success early.

## Goal

Build a real implementation of the MosaicMem paper that:

- implements the core operators faithfully
- runs with real model backends
- reproduces the paper's mechanism classes
- supports rigorous evaluation
- remains testable and maintainable

## Non-Goals For The First Wave

These should not block the initial "make it real" effort:

- fully pure-Rust model execution at all costs
- multi-tenant serving
- cloud deployment
- distributed production orchestration
- premature latency optimization before method correctness

The first priority is method correctness and reproducibility.

## Guiding Principles

1. Paper fidelity before marketing.
2. Reproduction before production hardening.
3. Keep the synthetic backend for fast tests.
4. Add a real backend path instead of corrupting the synthetic path.
5. Replace `Vec<f32>`-everywhere APIs with explicit tensor/layout types where
   correctness matters.
6. Every major operator gets:
   - a reference test
   - a regression test
   - an acceptance criterion tied to paper behavior
7. Do not claim "implemented" until the operator is used in the real inference
   path.

## Recommended Architecture Direction

### Recommended Short-Term Backend Strategy

Keep Rust as the orchestrator and systems layer, but use a real model backend
that is practical for the paper stack.

Recommended first move:

- Rust orchestration
- Python worker or sidecar for model inference and training components

Why:

- Wan 2.2, DA V3, VIPE, training code, and evaluation tooling live in a
  Python-first ecosystem
- trying to force the first real implementation into pure Rust inference will
  likely slow the project down badly
- the repo's value is in orchestration, geometry, memory, and correctness, not
  in proving ideological purity about runtime language choice

Longer term:

- keep open the option of ONNX Runtime / TensorRT / candle / custom kernels for
  optimized inference paths once correctness is established

### Backend Split

Maintain two execution modes:

- `synthetic`
  - current fast test path
  - deterministic
  - no model dependencies
- `real`
  - actual checkpoints and operator fidelity
  - slower but meaningful

This split avoids breaking the testability benefits of the current repo.

## Workstreams

There are two main workstreams:

- Track A: paper-faithful reproduction
- Track B: production hardening

Track A comes first.
Track B should start only after Track A reaches a minimally truthful state.

## Phase Plan

## Phase 0: Truthfulness And Scaffolding

Objective:

- make the repo honest about its current state
- prepare the codebase for a clean synthetic/real split

Deliverables:

- document synthetic vs real modes explicitly
- add feature flags or backend enums for runtime selection
- rename or annotate placeholder operators where needed
- make README and CLI behavior align with actual capabilities

Code areas:

- `README.md`
- `src/main.rs`
- `src/pipeline/config.rs`
- backend trait surfaces in `src/diffusion`, `src/geometry`, `src/pipeline`

Acceptance criteria:

- no public-facing doc implies paper reproduction unless the real path is used
- CLI clearly indicates whether the active backend is synthetic or real
- synthetic and real backends can coexist without branching chaos

Why this phase first:

- if the repo keeps overclaiming during the rebuild, confusion accumulates

## Phase 1: Canonical Tensor, Patch, And Geometry Types

Objective:

- remove ambiguity from the core data model before wiring real operators

Current problem:

- too much logic moves raw flattened vectors around without strong layout types
- patch geometry is too coarse for faithful alignment

Deliverables:

- typed tensor wrappers for:
  - video frames
  - latent tensors
  - token grids
  - masks
- explicit layout metadata in types rather than comments
- patch metadata expanded to support:
  - dense per-token or per-sample coordinates
  - source coordinate grids
  - optional depth tiles or disparity tiles
  - visibility / provenance metadata needed for alignment

Code areas:

- `src/diffusion/vae.rs`
- `src/pipeline/inference.rs`
- `src/memory/store.rs`
- `src/memory/mosaic.rs`
- `src/attention/*`

Acceptance criteria:

- frame extraction and latent-frame extraction no longer rely on ad hoc index
  math scattered across the pipeline
- patches contain enough geometry to support dense warping
- layout regressions are guarded by dedicated tests

Required tests:

- roundtrip layout tests for frame and latent extraction
- patch token ordering tests
- temporal compression mapping tests
- property tests for shape/index safety

## Phase 2: Real Backend Interfaces

Objective:

- add real implementations behind existing traits or new backend-specific traits

Deliverables:

- real depth estimation backend
- real VAE backend
- real diffusion backbone backend
- backend configuration and checkpoint loading
- versioned model config and checkpoint manifest handling

Recommended initial implementation strategy:

- Python sidecar with RPC or local process bridge
- rust structs for request/response payloads
- strict typed errors and timeouts

Why not pure Rust first:

- the paper ecosystem is Python-native
- model correctness matters more than runtime purity at this stage

Acceptance criteria:

- repo can run one real forward pass through depth, VAE, and backbone with
  actual checkpoints
- failures are surfaced as typed errors, not silent fallbacks
- backend selection is explicit in config and logs

Required tests:

- smoke tests gated by environment / model availability
- schema tests for backend payloads
- shape contract tests against real backend outputs

## Phase 3: Frame-Aware Retrieval And Queried-View Composition

Objective:

- fix the biggest architectural mismatch: memory must be queried for the actual
  view being generated, not just the first pose in the window

Deliverables:

- per-frame or per-latent-slice retrieval within each window
- frame-aware coverage masks
- frame-aware rasterized memory latent
- removal of naive temporal replication in `LatentCanvas::to_cthw`

Code areas:

- `src/pipeline/inference.rs`
- `src/memory/retrieval.rs`
- `src/memory/mosaic.rs`

Design requirement:

- if the model operates at latent timestep granularity, retrieval must be
  defined at that same granularity with correct temporal compression semantics

Acceptance criteria:

- memory retrieval varies correctly as the camera moves within a window
- coverage masks differ across frames when the queried view changes
- memory latent is no longer blindly replicated over time

Required tests:

- synthetic multi-pose window tests with known expected retrieval drift
- dense coverage footprint tests
- target-view composition tests under camera motion

## Phase 4: Faithful PRoPE

Objective:

- implement PRoPE as an actual attention-space projective conditioning operator

Deliverables:

- projection matrix construction per frame
- relative projective transform logic
- PRoPE operator applied inside attention Q/K/V processing
- temporal compression support consistent with the paper's subframe handling

Code areas:

- `src/attention/prope.rs`
- real backbone integration path
- any backend bridge that carries camera conditioning into the model

This phase is successful only if:

- PRoPE is used in the real inference path
- not just computed as auxiliary numbers
- not just summarized into a scalar bias

Acceptance criteria:

- reference outputs match a trusted implementation for fixed camera examples
- changing camera geometry changes attention behavior in the intended operator
- compressed latent slices handle multiple original-frame cameras correctly

Required tests:

- numeric reference tests against a Python/Numpy implementation
- attention invariance / sensitivity tests under controlled camera changes
- regression tests for temporal compression indexing

## Phase 5: Faithful Warped RoPE

Objective:

- replace center-only quantized Warped RoPE with dense/fractional geometric
  alignment

Deliverables:

- dense coordinate reprojection from source patch support to target view
- fractional coordinate preservation
- per-token or per-sample warped positions
- richer temporal coordinate handling than coarse age bins

Code areas:

- `src/attention/warped_rope.rs`
- `src/attention/memory_cross.rs`
- patch geometry storage in `src/memory/store.rs`

Acceptance criteria:

- different tokens within the same patch can receive different warped positions
- warped positions remain stable under subpixel shifts
- the operator can represent oblique views better than center-only alignment

Required tests:

- dense reprojection goldens
- fractional-coordinate regression tests
- comparison tests showing distinct intra-patch warped coordinates

## Phase 6: Faithful Warped Latent

Objective:

- replace the single-plane homography shortcut with dense reprojection-based
  feature warping

Deliverables:

- dense source-to-target sampling grid construction
- bilinear resampling based on reprojected coordinates
- visibility / invalid-sample masking
- optional occlusion reasoning hooks

Code areas:

- `src/attention/warped_latent.rs`
- patch geometry storage
- pipeline memory alignment path

Acceptance criteria:

- warping uses dense coordinates, not one patch depth
- non-planar or high-parallax cases are representable
- invalid regions are explicit, not hidden as zero-filled output

Required tests:

- geometric goldens with known camera transforms
- reprojection consistency tests
- invalid sample mask tests

## Phase 7: Real Memory Cross-Attention Integration

Objective:

- wire MosaicMem into the actual model path, not as an untrained placeholder

Deliverables:

- real memory tokens fed into the real backbone
- attention integration point consistent with the target model
- gating strategy defined and trainable
- query tokens carry the correct model semantics

Code areas:

- real backbone adapter
- attention integration path
- `src/attention/memory_cross.rs` or its replacement

Acceptance criteria:

- memory conditioning measurably affects real model outputs
- retrieval and alignment operators are active in the real inference path
- ablations can toggle:
  - no memory
  - PRoPE only
  - Warped RoPE only
  - Warped Latent only
  - full model

Required tests:

- wiring tests showing memory path is exercised end to end
- ablation harness tests for operator toggles

## Phase 8: Training-Free Real Integration Milestone

Objective:

- reproduce the paper's "training-free" style integration milestone before
  full fine-tuning

Deliverables:

- real Wan-based inference path with memory-conditioned context insertion
- qualitative revisit demos from fixed prompts and trajectories
- inspection tooling for retrieved memory patches and aligned conditioning

Acceptance criteria:

- model runs with real memory-conditioned context on real checkpoints
- revisit outputs are visually and geometrically meaningfully different from
  no-memory outputs
- output artifacts are inspectable and traceable

Why this phase matters:

- it validates wiring before committing to training effort

## Phase 9: Fine-Tuning Pipeline

Objective:

- implement the actual training path required for the paper's core results

Deliverables:

- training dataset adapters
- training configuration
- checkpoint save/load flow
- logging and experiment tracking
- ablation configuration support
- warping-strategy mixture training support

Code areas:

- new training modules and scripts
- dataset loaders
- backend integration for gradient-based training

Important note:

- the repo currently contains no training stack
- this phase is not optional if the goal is to respect the paper fully

Acceptance criteria:

- model can be fine-tuned with MosaicMem operators active
- checkpoints can be resumed and evaluated
- ablation experiments are reproducible

Required tests:

- small-scale train-step smoke test
- checkpoint roundtrip tests
- config schema tests

## Phase 10: Evaluation Harness

Objective:

- stop relying on intuition and synthetic tests; measure the same categories the
  paper measures

Deliverables:

- camera control metrics
  - RotErr
  - TransErr
- video quality metrics
  - FID
  - FVD
- retrieval consistency metrics
  - SSIM
  - PSNR
  - LPIPS
- dynamic score / motion metrics
- benchmark runner with stored artifacts and exact config capture

Code areas:

- new evaluation modules
- benchmark scripts
- metric adapters
- dataset episode definitions

Acceptance criteria:

- evaluation can run reproducibly from a pinned config
- results for each ablation are serialized and comparable
- no method claim is made without corresponding metric evidence

## Phase 11: Mosaic Forcing

Objective:

- implement the paper's autoregressive model path, not just sliding-window
  blending

Deliverables:

- causal student / teacher setup if required by the chosen implementation path
- forcing strategy implementation
- rolling context / memory update logic
- evaluation against the sliding-window baseline

Code areas:

- new AR modules
- training and inference paths
- evaluation harness extensions

Acceptance criteria:

- AR path is a distinct algorithm, not a relabeling of overlap blending
- quality and consistency can be compared against the bidirectional baseline

## Phase 12: Production Hardening

Objective:

- after method correctness is in place, make the system robust enough to serve
  or batch reliably

Deliverables:

- scalable memory indexing and update path
- cache and storage strategy for large memory banks
- backend lifecycle management
- observability
- failure handling policy
- artifact versioning
- model registry and checkpoint management

Known hardening targets already visible from the current code:

- stop rebuilding KD-trees on every insert
- stop recomputing depth and encodes unnecessarily
- make memory update failure semantics explicit
- make real prompt conditioning observable and testable

Acceptance criteria:

- system survives long runs without silent corruption
- resource usage is bounded and inspectable
- metrics and traces are attached to runs

## Cross-Cutting Engineering Work

These tasks should run alongside multiple phases.

### Data Model Cleanup

- replace ambiguous flattened buffers with typed tensor containers where the
  cost is justified
- centralize layout conversions
- add explicit shape validation at module boundaries

### Error Handling

- keep typed public errors
- separate backend failures, geometry failures, memory failures, and evaluation
  failures
- avoid warning-only failure handling in critical paths unless policy says so

### Config Versioning

- version config formats
- version memory snapshots
- version checkpoint manifests

### Artifact Provenance

- save exact config, backend versions, checkpoint ids, and git commit for all
  benchmark and evaluation runs

### Documentation Discipline

- do not let README claims outrun code
- keep synthetic and real workflows clearly separated

## Prioritized Backlog

If work starts now, the recommended order is:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9
11. Phase 10
12. Phase 11
13. Phase 12

This ordering is deliberate:

- do not start fine-tuning before operator correctness and backend wiring exist
- do not start productionization before evaluation truth exists

## First Three Concrete Milestones

## Milestone A: Honest Dual-Backend Scaffold

Scope:

- Phase 0 plus the backend split foundation from Phase 2

Definition of done:

- repo can run in `synthetic` or `real` mode
- docs are truthful
- config selects backend explicitly
- real backend interfaces exist even if feature-incomplete

## Milestone B: Geometry-Correct Memory Path

Scope:

- Phase 1, Phase 3, and the geometry prerequisites for Phase 5 and Phase 6

Definition of done:

- memory retrieval is frame-aware
- patches carry dense enough geometry for faithful alignment
- memory coverage and composition reflect actual queried views

## Milestone C: Paper-Critical Operators In Real Inference

Scope:

- Phase 4 through Phase 8

Definition of done:

- PRoPE is real
- Warped RoPE is real
- Warped Latent is real
- memory cross-attention is wired into the real model path
- training-free real demos exist

Only after Milestone C should the project even begin discussing method-level
comparison with the paper.

## Acceptance Checklist For "Paper-Faithful Enough To Evaluate"

All items below must be true before paper-comparison benchmarking:

- real model backend runs
- real depth backend runs
- real VAE backend runs
- PRoPE is in attention, not summarized outside it
- Warped RoPE is dense/fractional
- Warped Latent is dense reprojection-based
- retrieval is queried-view and frame-aware
- memory latent is not naively replicated across time
- ablation toggles are implemented
- benchmark runner captures configs and artifacts

## Risks And External Dependencies

### External Dependencies

- model checkpoints and licensing
- depth / pose backend selection
- GPU availability
- benchmark episode availability
- evaluation metric implementations and dependencies

### Main Technical Risks

- model integration complexity may force a backend bridge design that differs
  from the current pure-Rust aesthetic
- dense warping may require more patch metadata than the current store format
  can handle cheaply
- faithful PRoPE may require deeper model surgery than initially expected
- paper benchmark reproduction may be limited by missing training data or setup
  details not present in this repo

### Project Risk

The biggest project risk is pretending partial wiring equals method completion.
That is how teams get stuck in a permanent "almost real" state.

## What To Avoid

Do not do these things:

- keep expanding synthetic behavior while calling it paper progress
- bolt real checkpoints onto incorrect operators and call it done
- implement evaluation last
- rename overlap blending as autoregressive Mosaic Forcing
- let docs imply that the rebuild is further along than it is

## Recommended Immediate Next Tasks

1. Add backend mode selection and clean up docs/CLI language.
2. Introduce canonical tensor and patch-geometry types.
3. Design the real backend bridge for depth, VAE, and backbone.
4. Refactor retrieval and composition to be frame-aware inside windows.
5. Build numeric reference tests for PRoPE, Warped RoPE, and Warped Latent.

## Final Position

The rebuild is feasible.
The current repo already has a useful modular skeleton.

But making it real means doing the hard parts for real:

- model integration
- operator fidelity
- evaluation discipline
- honest milestone gating

That is the standard this plan assumes.
