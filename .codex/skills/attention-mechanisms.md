---
name: attention-mechanisms
description: Positional encoding and cross-attention mechanisms for mosaicmem-rs — RoPE, Warped RoPE, Warped Latent, PRoPE, and Memory Cross-Attention. Activate when working with attention layers, positional encodings, or memory injection into the diffusion backbone.
prerequisites: nalgebra 0.33
---

# Attention Mechanisms

<purpose>
These mechanisms inject spatial memory into the diffusion backbone. They translate 3D geometry into the attention-compatible representations that make MosaicMem work.
</purpose>

<context>
— RoPE: Standard rotary position embedding with precomputed (cos, sin) tables
— WarpedRoPE: 3D-aware RoPE using reprojected patch coordinates (u, v, temporal_offset)
— Warped Latent: Feature-space warping via planar homography + bilinear interpolation
— PRoPE: Projective positional encoding from relative camera geometry
— MemoryCrossAttention: Cross-attention where Q = generation tokens, K/V = memory patches
— All tensor data passed as Vec<f32> with explicit shape arrays
</context>

<procedure>
For adding memory conditioning to a generation step:
1. Compute warped positions: `WarpedRoPE::compute_warped_positions(patches, target_pose, intrinsics)`
2. Apply warped RoPE: `warped_rope.rotate(memory_tokens, warped_positions)`
3. Optionally warp latents: `warp_patch_latent(patch, source_pose, target_pose, intrinsics)`
4. Compute PRoPE: `prope.compute_rotations(source_poses, target_pose)`
5. Run cross-attention: `memory_cross_attn.forward(gen_tokens, memory_tokens, mask)`

Key: WarpedRoPE and Warped Latent are complementary — RoPE gives positional signal, Latent gives feature alignment.
</procedure>

<patterns>
<do>
  — Use frequency bands as powers of base (10000.0) for RoPE
  — Apply max-subtraction trick in softmax for numerical stability
  — Use `grid_positions_2d()` or `grid_positions_3d()` for uniform position grids
  — Keep attention head dimension consistent across Q, K, V
  — Mask invalid/padding positions in cross-attention
</do>
<dont>
  — Don't skip the max-subtraction in softmax — will NaN on large sequences
  — Don't assume homography is always invertible — check determinant > 1e-8
  — Don't mix up coordinate spaces: warped positions are in target view, not source view
  — Don't use bilinear sampling without bounds checking — clamp to [0, H-1] x [0, W-1]
</dont>
</patterns>

<examples>
Example: Standard RoPE rotation
```rust
use crate::attention::rope::RoPE;

let rope = RoPE::new(64, 10000.0); // dim=64, base=10000
let tokens = vec![1.0; 64]; // single token
let positions = vec![(0, 0, 0)]; // (x, y, t)
let rotated = rope.rotate_batch(&tokens, &positions, 64);
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| NaN in attention output | Softmax overflow | Verify max-subtraction is applied |
| Warped positions all zero | Patches behind target camera | Check patch visibility before warping |
| Homography returns zeros | Near-singular matrix | Guard with determinant > 1e-8 check |
| RoPE output unchanged | Position (0,0,0) gives identity rotation | Use non-zero positions |
</troubleshooting>

<references>
— src/attention/rope.rs: Standard RoPE implementation
— src/attention/warped_rope.rs: WarpedRoPE with 3D coordinates
— src/attention/warped_latent.rs: Homography warping + bilinear sampling
— src/attention/prope.rs: PRoPE camera conditioning
— src/attention/memory_cross.rs: MemoryCrossAttention layer
— RFC-004-warped-rope.md: Warped RoPE design
— RFC-005-warped-latent.md: Warped Latent design
— RFC-006-prope.md: PRoPE design
— RFC-007-memory-cross-attention.md: Cross-attention design
</references>
