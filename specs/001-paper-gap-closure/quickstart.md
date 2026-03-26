# Quickstart: Paper Gap Closure

## Prerequisites

- Rust 1.75+ with `cargo`
- For real backend (optional): Python 3.10+, PyTorch, model checkpoints

## Build and Test (Synthetic Mode)

```bash
# Clone and build
git clone https://github.com/AbdelStark/mosaicmem.git
cd mosaicmem
git checkout 001-paper-gap-closure

# Run all tests including new paper-fidelity tests
cargo test

# Run with synthetic backend (default)
cargo run -- demo --num-frames 16 --width 64 --height 64 --steps 5
# Output will show: [synthetic] backend active
```

## Verify Frame-Aware Retrieval

```bash
# Run the demo with a turning trajectory to see per-frame retrieval
cargo run -- demo --num-frames 32 --width 64 --height 64 \
  --trajectory-type orbit --steps 5

# Inspect memory coverage per frame
cargo run -- inspect --num-frames 16 --trajectory-type orbit
```

## Build with Real Backend (Requires Checkpoints)

```bash
# Compile with real backend support
cargo build --features real-backend

# Configure backend in pipeline config
cargo run --features real-backend -- show-config
# Edit config to set backend_mode: "real"
# Set checkpoint paths for depth, VAE, backbone

# Run with real backend
cargo run --features real-backend -- generate \
  --config my-config.json --trajectory trajectory.json
# Output will show: [real] backend active
```

## Run Operator Reference Tests

```bash
# PRoPE numerical reference tests
cargo test prope_reference -- --nocapture

# Warped RoPE dense reprojection tests
cargo test warped_rope_dense -- --nocapture

# Warped Latent dense warp tests
cargo test warped_latent_dense -- --nocapture

# Frame-aware retrieval tests
cargo test frame_aware_retrieval -- --nocapture
```

## Ablation Configuration

```rust
use mosaicmem::pipeline::config::{PipelineConfig, AblationConfig};

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
grep -rn "unsafe" src/ --include="*.rs" | grep -v "// SAFETY:"
# Should return nothing (or only justified FFI blocks behind real-backend)
```
