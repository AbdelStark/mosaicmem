---
name: debugging
description: Debugging and troubleshooting guide for mosaicmem-rs. Activate when encountering errors, test failures, panics, or unexpected behavior. Covers common failure modes across all modules.
prerequisites: cargo, RUST_LOG env var
---

# Debugging

<purpose>
Systematic approach to diagnosing and fixing issues in mosaicmem-rs. Covers compiler errors, runtime panics, test failures, and logic bugs.
</purpose>

<context>
— Structured logging via tracing (enable with RUST_LOG=debug)
— 46 unit tests across all modules
— No unsafe code — panics come from bounds checks, unwraps, or arithmetic
— Spatial queries depend on KD-tree state (must rebuild after inserts)
— Floating-point edge cases in geometry and attention code
</context>

<procedure>
Diagnostic cascade:
1. Read the FULL error message — Rust errors are precise and actionable
2. If compiler error: `cargo check` for fast feedback, fix types first
3. If test failure: `cargo test [test_name] -- --nocapture` for tracing output
4. If runtime panic: Check for `unwrap()` calls on None/Err values
5. If wrong output: Add `tracing::debug!` at key points, run with `RUST_LOG=debug`
6. If spatial query issue: Verify KD-tree rebuilt, check point cloud not empty
7. If numerical issue: Check for NaN propagation, add intermediate value logging

For test isolation:
— `cargo test camera::` — run only camera module tests
— `cargo test test_name` — run specific test
— `cargo test -- --nocapture` — show all output
</procedure>

<patterns>
<do>
  — Start with `cargo check` — fastest feedback loop
  — Use `RUST_LOG=mosaicmem_rs=debug cargo test` for verbose test output
  — Check `num_patches()` / `num_points()` before spatial queries
  — Verify shapes match at trait boundaries: [B,C,T,H,W]
  — Use `approx::assert_relative_eq!` for floating-point assertions
  — Look for off-by-one in indexing (especially height/width vs. row/col)
</do>
<dont>
  — Don't ignore compiler warnings — they often indicate real bugs
  — Don't add `.unwrap()` to debug — use `dbg!()` or tracing instead
  — Don't assume float equality — always use epsilon comparison
  — Don't debug pipeline issues at pipeline level — isolate to specific module first
</dont>
</patterns>

<troubleshooting>
| Symptom | Likely Module | Investigation Steps |
|---------|---------------|---------------------|
| Index out of bounds | geometry, attention | Check array dimensions match HxW or shape[B,C,T,H,W] |
| NaN values | attention (softmax) | Add checks before/after softmax, verify max-subtraction |
| Empty retrieval results | memory | Check store has patches, verify poses face stored content |
| Depth map all zeros | geometry (depth) | Verify DepthEstimator implementation returns valid depths |
| Wrong frame count | pipeline | Check window_size, overlap, trajectory length alignment |
| Serde error on load | camera (pose) | Ensure JSON uses {qx,qy,qz,qw,tx,ty,tz} format for Isometry3 |
| KD-tree panic | geometry, memory | Never query empty tree; check `num_points() > 0` |
| Shape mismatch | pipeline, diffusion | Print shapes at each step: backbone output → VAE input |
</troubleshooting>

<references>
— All src/**/*.rs files contain #[cfg(test)] modules with example usage
— src/main.rs cmd_demo(): End-to-end synthetic test path
— RUST_LOG=debug enables tracing output everywhere
</references>
