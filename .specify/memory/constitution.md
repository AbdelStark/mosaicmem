<!--
  Sync Impact Report
  ==================
  Version change: N/A (initial) -> 1.0.0
  Modified principles: N/A (first ratification)
  Added sections:
    - Principle I: Paper Fidelity
    - Principle II: Technical Soundness
    - Principle III: Code Quality
    - Principle IV: Testing Standards
    - Section: ML-Specific Constraints
    - Section: Development Workflow
    - Governance
  Removed sections: None
  Templates requiring updates:
    - .specify/templates/plan-template.md — Constitution Check section
      references "[Gates determined based on constitution file]" which is
      generic placeholder text. ✅ No update needed — the plan command
      fills this dynamically from this constitution at generation time.
    - .specify/templates/spec-template.md — No constitution-specific
      references. ✅ Compatible.
    - .specify/templates/tasks-template.md — References test-first
      ordering ("Tests MUST be written and FAIL before implementation").
      ✅ Aligned with Principle IV.
    - .specify/templates/commands/*.md — No command templates found.
      ✅ N/A.
  Follow-up TODOs: None.
-->

# MosaicMem Constitution

## Core Principles

### I. Paper Fidelity

Every algorithm, data structure, and pipeline stage MUST faithfully
implement the MosaicMem paper (arXiv:2603.17117). Specifically:

- **Algorithmic correspondence**: Each module (patch memory, frustum
  retrieval, warped RoPE, PRoPE, warped latent alignment, memory
  cross-attention, autoregressive rollout) MUST map to a clearly
  identifiable section or equation in the paper.
- **Naming alignment**: Struct, function, and variable names MUST use
  terminology from the paper (e.g., "frustum culling" not "view
  filtering", "PRoPE" not "camera positional encoding").
- **Deviation documentation**: Any intentional departure from the paper
  (performance optimization, numerical stability workaround, API
  ergonomics) MUST be documented in a code comment citing the relevant
  paper section and explaining the rationale.
- **Trait boundaries for real vs. synthetic**: Synthetic backends
  (SyntheticBackbone, SyntheticVAE, SyntheticDepthEstimator) exist for
  testing only. Trait interfaces (`DiffusionBackbone`, `NoiseScheduler`,
  `VAE`, `DepthEstimator`) MUST match the mathematical contracts from
  the paper so that plugging in real inference backends requires zero
  algorithmic changes.

### II. Technical Soundness

The implementation MUST be correct from a machine learning and 3D
geometry perspective. Non-negotiable requirements:

- **SE3 geometry**: All camera pose operations MUST use proper rigid-body
  transformations (rotation via unit quaternions, composition via matrix
  multiplication). No Euler-angle shortcuts that introduce gimbal lock.
- **Numerical precision**: Floating-point operations in depth unprojection,
  homography computation, and RoPE frequency calculations MUST use f64
  where accumulation error is a concern; f32 is acceptable only for
  latent tensors and final pixel values.
- **Diffusion correctness**: Noise schedules (DDPM linear beta) MUST
  produce alpha-bar values matching the standard formulation. The
  denoising loop MUST apply noise prediction in the correct order
  (predict noise, compute x0, add noise for next step).
- **Attention mechanics**: Cross-attention MUST apply scaled dot-product
  with correct dimensionality scaling (1/sqrt(d_k)). Warped RoPE
  positions MUST be derived from 3D-to-2D reprojection, not grid
  indices.
- **Memory retrieval**: Frustum culling MUST use the actual camera
  intrinsics and extrinsics. Top-K scoring MUST incorporate temporal
  decay and diversity filtering as described in the paper.

### III. Code Quality

Rust code MUST be idiomatic, safe, and maintainable:

- **No unsafe**: `unsafe` blocks are forbidden unless required by FFI to
  external inference runtimes, and each MUST carry a `// SAFETY:`
  comment justifying correctness.
- **Error handling**: All fallible operations MUST return `Result` with
  domain-specific error types via `thiserror`. No `.unwrap()` outside
  of tests.
- **Trait-driven design**: All model-dependent components MUST be behind
  traits so backends can be swapped without modifying pipeline logic.
- **Performance**: Hot paths (retrieval, attention, rollout) MUST use
  `rayon` for data parallelism where the workload justifies it.
  Premature optimization without benchmarks is prohibited.
- **Clippy clean**: All code MUST pass `cargo clippy -- -D warnings`
  with no suppressions unless justified in a comment.
- **Documentation**: Public API items MUST have doc comments. Internal
  modules MUST have a top-level `//!` module doc explaining purpose and
  relationship to the paper.

### IV. Testing Standards

Every component MUST be tested for correctness, not just absence of
panics:

- **Numerical correctness tests**: Core algorithms (depth unprojection,
  RoPE computation, homography warping, attention scoring) MUST have
  tests that compare outputs against hand-computed or reference values
  with explicit tolerance bounds (using `approx` crate).
- **Round-trip invariants**: Operations with mathematical inverses
  (project/unproject, encode/decode, serialize/deserialize) MUST have
  round-trip tests proving `f(f_inv(x)) ~= x`.
- **Property-based edge cases**: Retrieval with empty memory, zero-length
  trajectories, degenerate camera poses (identity, pure rotation),
  single-frame windows MUST be covered.
- **Integration tests**: The full pipeline (`AutoregressivePipeline`)
  MUST have end-to-end tests that verify output shape, value ranges,
  and deterministic reproducibility with fixed seeds.
- **Regression tests**: Any bug fix MUST include a test that reproduces
  the bug and verifies the fix.
- **No test-only code in library**: Synthetic backends live in the main
  crate for user convenience, but test helpers and fixtures MUST stay
  in `tests/` or `#[cfg(test)]` modules.

## ML-Specific Constraints

Requirements that cut across multiple principles:

- **Tensor layout**: All multi-dimensional data MUST use explicit layout
  conventions documented at the type level (e.g., `[B, C, T, H, W]`).
  Layout assumptions MUST NOT be implicit.
- **Reproducibility**: Given the same seed, trajectory, and config, the
  pipeline MUST produce bit-identical outputs. All sources of
  non-determinism (thread scheduling in rayon, hash map ordering) MUST
  be controlled or documented.
- **Config-driven behavior**: All tunable parameters (top-K, decay
  half-life, diversity radius, inference steps, window size/overlap)
  MUST be exposed via `PipelineConfig` or `MemoryConfig`. No magic
  constants buried in implementation.
- **Serialization**: Memory stores and trajectories MUST be serializable
  to JSON for debugging and offline analysis. Serialized format MUST
  be stable across patch versions.

## Development Workflow

- **Commit discipline**: Each commit MUST compile (`cargo build`), pass
  lint (`cargo clippy`), and pass all tests (`cargo test`). Broken
  commits are not acceptable even on feature branches.
- **Review gates**: Changes to core algorithms (memory/, attention/,
  geometry/, diffusion/) MUST include a reference to the paper section
  being implemented or modified.
- **Benchmark before optimizing**: Performance changes MUST be backed by
  `cargo bench` results or the `bench` CLI command showing measurable
  improvement. No speculative optimization.
- **Feature flags over dead code**: Incomplete features MUST use Cargo
  feature flags, not commented-out code or TODO blocks.

## Governance

- This constitution supersedes ad-hoc practices. All code changes MUST
  be verifiable against these principles.
- **Amendments** require: (1) written proposal stating which principle
  changes and why, (2) review for impact on existing code, (3) version
  bump per semantic versioning rules below.
- **Versioning**: MAJOR for principle removals or incompatible
  redefinitions, MINOR for new principles or material expansions,
  PATCH for wording clarifications.
- **Compliance review**: At each milestone (new module, new pipeline
  stage, release), verify all four principles are satisfied.

**Version**: 1.0.0 | **Ratified**: 2026-03-26 | **Last Amended**: 2026-03-26
