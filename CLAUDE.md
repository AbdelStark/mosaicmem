<identity>
mosaicmem-rs: Rust implementation of MosaicMem — hybrid spatial memory for controllable video world models (Yu, Qian, Li et al., 2026). Inference-only pipeline combining 3D geometry with diffusion-based video generation.
</identity>

<stack>
| Layer       | Technology          | Version | Notes                                      |
|-------------|---------------------|---------|--------------------------------------------|
| Language    | Rust                | 2024 ed | Edition 2024, stable toolchain             |
| Math        | nalgebra             | 0.33    | SE(3) transforms, projections, 3D geometry |
| Arrays      | ndarray              | 0.16    | N-dimensional tensor operations            |
| Spatial     | kiddo                | 4.x     | KD-tree for nearest-neighbor queries       |
| CLI         | clap                 | 4.x     | Derive-based argument parsing              |
| Logging     | tracing              | 0.1     | Structured logging with env-filter         |
| Errors      | thiserror            | 2.0     | Derive macro for error enums               |
| Parallelism | rayon                | 1.10    | Data-parallel iterators                    |
| Serialization | serde + serde_json | 1.x     | JSON I/O for trajectories, configs         |
| Images      | image                | 0.25    | Image I/O                                  |
| RNG         | rand                 | 0.8     | Noise initialization, synthetic backends   |
| Test        | approx               | 0.5     | Floating-point comparison (dev-dep)        |
</stack>

<structure>
src/
├── main.rs             # CLI entry point (clap subcommands) [agent: modify]
├── lib.rs              # Module declarations [agent: modify]
├── camera/             # SE(3) poses, intrinsics, trajectory I/O [agent: modify]
│   ├── mod.rs
│   ├── pose.rs         # CameraPose — Isometry3 wrapper with custom serde
│   ├── intrinsics.rs   # CameraIntrinsics — pinhole projection model
│   └── trajectory.rs   # CameraTrajectory — sequence + keyframe selection
├── geometry/           # 3D reconstruction pipeline [agent: modify]
│   ├── mod.rs
│   ├── depth.rs        # DepthEstimator trait + SyntheticDepthEstimator
│   ├── pointcloud.rs   # PointCloud3D — colored point cloud + voxel downsample
│   ├── projection.rs   # unproject/project/frustum_cull
│   └── fusion.rs       # StreamingFusion — incremental KD-tree builder
├── memory/             # Spatial memory store [agent: modify]
│   ├── mod.rs
│   ├── store.rs        # MosaicMemoryStore — patch storage + KD-tree retrieval
│   ├── retrieval.rs    # MemoryRetriever — high-level mosaic frame assembly
│   ├── mosaic.rs       # MosaicFrame — compose tokens + coverage grid
│   └── manipulation.rs # splice, flip, erase, translate, scale
├── attention/          # Positional encodings + cross-attention [agent: modify]
│   ├── mod.rs
│   ├── rope.rs         # Standard RoPE with precomputed tables
│   ├── warped_rope.rs  # WarpedRoPE — 3D-aware positional encoding
│   ├── warped_latent.rs# Homography-based latent warping + bilinear sampling
│   ├── prope.rs        # PRoPE — projective camera conditioning
│   └── memory_cross.rs # MemoryCrossAttention — Q from gen, K/V from memory
├── diffusion/          # Backbone + VAE + scheduler abstractions [agent: modify]
│   ├── mod.rs
│   ├── backbone.rs     # DiffusionBackbone trait + SyntheticBackbone
│   ├── vae.rs          # VAE trait + SyntheticVAE
│   └── scheduler.rs    # NoiseScheduler trait + DDPMScheduler
└── pipeline/           # End-to-end generation [agent: modify]
    ├── mod.rs
    ├── config.rs       # PipelineConfig — all generation parameters
    ├── inference.rs    # InferencePipeline — single-window generation
    └── autoregressive.rs # AutoregressivePipeline — chained multi-window
</structure>

<commands>
| Task           | Command                           | Notes                                |
|----------------|-----------------------------------|--------------------------------------|
| Build          | `cargo build`                     | ~44s first build, ~3s incremental    |
| Build release  | `cargo build --release`           | Optimized binary                     |
| Test all       | `cargo test`                      | 46 tests, runs in <1s                |
| Test module    | `cargo test camera::`             | Filter by module path                |
| Test verbose   | `cargo test -- --nocapture`       | Show println/tracing output          |
| Check          | `cargo check`                     | Type-check without codegen (fastest) |
| Clippy         | `cargo clippy`                    | Lint with warnings                   |
| Format         | `cargo fmt`                       | Rustfmt formatting                   |
| Format check   | `cargo fmt -- --check`            | CI-style format check                |
| Run demo       | `cargo run -- demo`               | Synthetic end-to-end test            |
| Run with logs  | `RUST_LOG=debug cargo run -- demo`| Verbose tracing output               |
</commands>

<conventions>
<code_style>
  Naming: snake_case for functions/variables, PascalCase for types/traits, SCREAMING_SNAKE for constants.
  Files: snake_case.rs. One primary type per file.
  Modules: mod.rs re-exports pub types. Flat hierarchy (no deep nesting).
  Imports: Group std → external crates → internal crate modules. Use `use crate::module::Type`.
  Traits: Used for swappable backends (DepthEstimator, DiffusionBackbone, VAE, NoiseScheduler).
  Errors: Custom enums via thiserror per module (BackboneError, VAEError, PipelineError, DepthError).
</code_style>

<patterns>
  <do>
    — Use `thiserror` for all error enums. Derive `Debug, Clone` where possible.
    — Use `nalgebra` types for all 3D math: Point3, Vector3, Matrix4, Isometry3.
    — Use `kiddo::KdTree` for spatial queries. Rebuild after batch inserts.
    — Use `rayon` for data-parallel operations on large collections.
    — Write `#[cfg(test)] mod tests` in every file with 3+ unit tests.
    — Use `SyntheticX` stub implementations for testing without ONNX models.
    — Use `tracing::info!`/`debug!` for logging, never `println!` in library code.
    — Flatten tensors as `Vec<f32>` with explicit shape arrays `[B,C,T,H,W]` for trait boundaries.
    — Clamp values (0.0..1.0) before denormalization in projection/sampling code.
  </do>
  <dont>
    — Don't use `unwrap()` in library code — use `?` or return Option/Result.
    — Don't use `unsafe` — the entire codebase is safe Rust.
    — Don't use async/await — all operations are synchronous.
    — Don't use `println!` in library code — use `tracing` macros.
    — Don't store nalgebra types directly in serde — use the custom serde module in camera/pose.rs.
    — Don't allocate KD-trees per query — rebuild in batch via `rebuild_kdtree()`.
  </dont>
</patterns>

<commit_conventions>
  Format: `type(scope): description`
  Types: feat, fix, refactor, test, docs, chore
  Scopes: camera, geometry, memory, attention, diffusion, pipeline, cli
  Example: `feat(memory): add temporal decay to patch retrieval scoring`
</commit_conventions>
</conventions>

<workflows>
<new_feature>
  1. Identify which module(s) the feature touches
  2. Read relevant RFC (RFC-001 through RFC-008) for design context
  3. Implement types/traits in the appropriate module file
  4. Add `pub use` re-exports in the module's mod.rs
  5. Write `#[cfg(test)] mod tests` with at least 3 unit tests
  6. Run `cargo test` — all 46+ tests must pass
  7. Run `cargo clippy` — resolve all warnings
  8. Run `cargo fmt` — ensure consistent formatting
  9. If adding CLI functionality, update main.rs Commands enum
</new_feature>

<bug_fix>
  1. Reproduce with a failing test in the relevant module
  2. Fix the implementation
  3. Run `cargo test` — confirm the new test passes and no regressions
  4. Run `cargo clippy` and `cargo fmt`
</bug_fix>

<add_trait_backend>
  1. Read the relevant trait definition (e.g., DiffusionBackbone in diffusion/backbone.rs)
  2. Create new struct implementing the trait
  3. Follow the Synthetic* pattern for the implementation structure
  4. Add tests using the new backend
  5. Wire into pipeline if needed (pipeline/config.rs, pipeline/inference.rs)
</add_trait_backend>
</workflows>

<boundaries>
<forbidden>
  DO NOT modify under any circumstances:
  — .env, .env.* (if they exist — credentials, API keys)
  — Cargo.lock (modified only by cargo commands, never manually)
</forbidden>

<gated>
  Modify ONLY with explicit human approval:
  — Cargo.toml (dependency changes affect the entire build)
  — SPECIFICATION.md (design authority document)
  — RFC-*.md (design decisions require review)
  — IMPLEMENTATION_PLAN.md (project roadmap)
</gated>

<safety_checks>
  Before ANY destructive operation:
  1. State what you're about to do
  2. State what could go wrong
  3. Wait for confirmation
</safety_checks>
</boundaries>

<troubleshooting>
<known_issues>
  | Symptom                              | Cause                              | Fix                                    |
  |--------------------------------------|------------------------------------|----------------------------------------|
  | `edition = "2024"` build error       | Requires Rust 1.85+                | `rustup update stable`                 |
  | KD-tree panic on empty point cloud   | Query before any inserts           | Check `num_points() > 0` before query  |
  | Isometry3 serde failure              | Direct serde of nalgebra types     | Use custom serde module in pose.rs     |
  | Shape mismatch in backbone trait     | Flattened tensor vs. shape array   | Verify shape = [B,C,T,H,W] consistency |
  | NaN in attention softmax             | Large dot products overflow exp()  | Max-subtraction trick is implemented   |
  | Homography warp returns zeros        | Near-zero determinant              | Check 1e-8 guard in warped_latent.rs   |
</known_issues>

<recovery_patterns>
  1. Read the full error message — Rust compiler errors are precise
  2. `cargo check` for fast type-error feedback (no codegen)
  3. `cargo test -- --nocapture` to see tracing output during tests
  4. If spatial queries return empty, verify KD-tree was rebuilt after inserts
  5. If pipeline produces zeros, check that SyntheticBackbone noise_scale > 0
</recovery_patterns>
</troubleshooting>

<architecture_notes>
  Key design decisions:
  — Trait-based backends: DiffusionBackbone, VAE, DepthEstimator, NoiseScheduler are traits.
    Synthetic* stubs exist for testing. Real implementations will use Tract/ONNX.
  — Flattened tensors: All trait boundaries use Vec<f32> + shape arrays, not typed tensors.
    This keeps the trait interface simple and backend-agnostic.
  — Streaming fusion: Point clouds are built incrementally via StreamingFusion.
    KD-tree is rebuilt in batch, not per-insert, for performance.
  — Memory budget: MosaicMemoryStore enforces max_patches via voxel-based eviction (oldest first).
  — Coverage grid: MosaicFrame tracks patch coverage at 1/8 resolution for inpainting guidance.
  — Window overlap: AutoregressivePipeline uses configurable overlap between generation windows
    for temporal consistency.
</architecture_notes>

<skills>
  Modular skills are located in .codex/skills/ (with symlinks at .claude/skills/ and .agents/skills/).

  Available skills:
  — _index.md: Skill registry and discovery metadata
  — rust-development.md: Rust coding patterns, error handling, testing for this project
  — geometry-3d.md: 3D reconstruction, projections, point cloud operations
  — memory-system.md: Spatial memory store, retrieval, mosaic composition
  — attention-mechanisms.md: RoPE, cross-attention, positional encoding implementations
  — pipeline-integration.md: End-to-end generation pipeline, autoregressive chaining
</skills>

<memory>
<project_decisions>
  2026-03: Rust Edition 2024 — Latest language features — Edition 2021 (fewer features)
  2026-03: nalgebra over glam — Richer SE(3) support, serde integration — glam (simpler but less complete)
  2026-03: kiddo over kd-tree — Better API, maintained, typed dimensions — kd-tree crate (less maintained)
  2026-03: Trait-based backends — Swap Synthetic↔ONNX without changing pipeline — Concrete types (inflexible)
  2026-03: Flattened Vec<f32> at trait boundaries — Backend-agnostic, simple — Typed tensors (ties to specific framework)
  2026-03: thiserror for errors — Derive macro, idiomatic — anyhow (less structured for library code)
</project_decisions>
</memory>
