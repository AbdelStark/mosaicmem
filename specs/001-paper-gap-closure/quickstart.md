# Quickstart: Paper Gap Closure

## Prerequisites

- Rust 1.75+ with `cargo`
- For real backend (optional): Python 3.10+, PyTorch, model checkpoints

## Build and Test (Synthetic Mode)

```bash
# Run all tests including new paper-fidelity tests
cargo test

# Inspect the effective backend mode and ablation config
cargo run -- show-config

# Run with synthetic backend (default)
cargo run -- demo --num-frames 8 --width 32 --height 32 --steps 2
# Output will show: [synthetic] backend active
```

## Verify Frame-Aware Retrieval

```bash
# The demo writes demo_output/trajectory.json; inspect it for per-frame memory coverage
cargo run -- inspect --trajectory demo_output/trajectory.json --width 32 --height 32 --coverage
```

## Build with Real Backend (Requires Checkpoints)

```bash
# Compile with real backend support
cargo build --features real-backend

# Inspect the config shape before editing backend_mode/checkpoint_path
cargo run --features real-backend -- show-config
# Then edit a JSON config to set backend_mode: "real" and checkpoint_path accordingly.
```

## Run Operator Reference Tests

```bash
# PRoPE numerical reference tests
cargo test --test prope_reference -- --nocapture

# Warped RoPE dense reprojection tests
cargo test --test warped_rope_dense -- --nocapture

# Warped Latent dense warp tests
cargo test --test warped_latent_dense -- --nocapture

# Frame-aware retrieval tests
cargo test --test frame_retrieval -- --nocapture
```

## Ablation Configuration

```rust
use mosaicmem::AblationConfig;
use mosaicmem::pipeline::config::PipelineConfig;

let config = PipelineConfig {
    ablation: AblationConfig {
        enable_memory: true,
        enable_prope: true,
        enable_warped_rope: false,  // disable for ablation
        enable_warped_latent: true,
        memory_gate_override: None,
    },
    ..Default::default()
};
```

## Verify Constitution Compliance

```bash
# Must pass: all code compiles
cargo build

# Must pass: lint clean
cargo clippy -- -D warnings

# Must pass: all tests
cargo test

# Must pass: no unsafe outside of feature-gated FFI
rg -n "unsafe" src/
# Should return nothing (or only justified FFI blocks behind real-backend)
```
