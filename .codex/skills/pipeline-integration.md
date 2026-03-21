---
name: pipeline-integration
description: End-to-end generation pipeline for mosaicmem-rs — single-window inference, autoregressive chaining, configuration, and memory lifecycle. Activate when working with video generation, pipeline config, window management, or connecting modules together.
prerequisites: All other modules (camera, geometry, memory, attention, diffusion)
---

# Pipeline Integration

<purpose>
The pipeline ties all modules together: geometry → memory → attention → diffusion → output frames. It manages the autoregressive loop that enables long-horizon generation.
</purpose>

<context>
— PipelineConfig: resolution, window_size, window_overlap, steps, guidance_scale, seed, memory budget
— InferencePipeline: single-window generation with memory store
— AutoregressivePipeline: chains windows with overlap for temporal consistency
— Trait backends: DiffusionBackbone, VAE, NoiseScheduler, DepthEstimator
— All backends have Synthetic* stubs for testing without ONNX models
— Flattened Vec<f32> + shape [B,C,T,H,W] at all trait boundaries
</context>

<procedure>
For a complete generation run:
1. Create PipelineConfig with desired parameters
2. Create AutoregressivePipeline from config
3. Instantiate backends (Synthetic* for testing, Tract for production)
4. Create camera trajectory (load JSON or construct programmatically)
5. Call `pipeline.generate(trajectory, text_emb, backbone, scheduler, vae, depth, callback)`
6. Receive `(frames, shapes)` — flattened pixel data + per-window shapes

Pipeline internals per window:
1. Select keyframes from trajectory window
2. Run depth estimation on previous frames
3. Update memory store with new 3D patches
4. Retrieve relevant patches for current viewpoint
5. Apply warped RoPE + warped latent conditioning
6. Run diffusion denoising loop (N steps)
7. Decode latents via VAE
8. Return pixel frames

Memory lifecycle:
— Insert: After each window, keyframes are added to memory store
— Retrieve: Before each window, relevant patches are fetched
— Budget: When max_patches exceeded, oldest patches evicted
</procedure>

<patterns>
<do>
  — Use `PipelineConfig::default()` as a starting point, override specific fields
  — Use the frame callback for streaming progress monitoring
  — Use `pipeline.stats()` to check memory usage during generation
  — Test with SyntheticBackbone + small resolution (64x64) for fast iteration
  — Use `reset_memory()` between independent generation runs
</do>
<dont>
  — Don't set window_overlap >= window_size (must be strictly less)
  — Don't skip the depth estimation step — memory store needs 3D patches
  — Don't assume frame data is in [0,1] — check VAE output range
  — Don't run production resolution in tests — use 64x64 with 5 steps
</dont>
</patterns>

<examples>
Example: Minimal autoregressive generation
```rust
let config = PipelineConfig {
    num_inference_steps: 5,
    width: 64,
    height: 64,
    window_size: 16,
    window_overlap: 4,
    ..Default::default()
};

let mut pipeline = AutoregressivePipeline::new(config);
let backbone = SyntheticBackbone::new(0.1);
let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
let vae = SyntheticVAE::new(8, 4, 16);
let depth = SyntheticDepthEstimator::new(5.0, 1.0);
let text_emb = vec![vec![0.0f32; 64]; 10];

let (frames, shapes) = pipeline.generate(
    &trajectory, &text_emb, &backbone, &scheduler, &vae, &depth, None,
)?;
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| PipelineError::Config | Invalid window_size/overlap | Ensure overlap < window_size |
| All-zero frames | Backbone noise_scale = 0 | Use SyntheticBackbone::new(0.1) |
| Memory grows unbounded | max_patches too high or budget not enforced | Check MemoryConfig |
| Shape mismatch in VAE decode | Latent dims don't match config | Verify spatial/temporal downsample factors |
</troubleshooting>

<references>
— src/pipeline/config.rs: PipelineConfig
— src/pipeline/inference.rs: InferencePipeline
— src/pipeline/autoregressive.rs: AutoregressivePipeline
— src/main.rs: CLI usage examples
— RFC-008-autoregressive-pipeline.md: Pipeline design
</references>
