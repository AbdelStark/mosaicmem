# MosaicMem: Hybrid Spatial Memory for Controllable Video World Models — Paper Digest

**Paper:** arXiv:2603.17117 (Yu*, Qian*, Li* et al., Animesh Garg lab)
**Website:** https://mosaicmem.github.io/mosaicmem/
**Code:** Coming soon (official)
**License:** TBD
**Key figure:** Animesh Garg (project lead)

---

## 1. Problem Statement

Video world models generate impressive short clips but fail at **4D consistency** (spatial + temporal):
- Camera movement causes **geometric drift** — the world shifts as the camera moves
- Revisited scenes **don't match** — returning to a location generates a different scene
- Long autoregressive rollouts **slowly collapse** — quality degrades over time
- Dynamic objects **freeze** under explicit 3D methods, or geometry drifts under implicit methods

The core tension: **explicit 3D memory** (reprojection-based) gives geometric accuracy but can't handle motion/dynamics. **Implicit latent memory** (conditioning frames) handles dynamics but accumulates camera drift.

## 2. Key Insight: Patch-and-Compose

MosaicMem's central idea: **persistent geometry comes from memory; dynamic content comes from the model.**

Instead of choosing between explicit and implicit memory, compose them: store 3D-lifted patches for scene structure, inject them as conditioning for the diffusion model, and let the model inpaint/evolve everything else.

Like a mosaic: fill in the persistent parts with retrieved patches, leave gaps for the model to complete with dynamic content.

## 3. Architecture — The Memory Forcing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    MosaicMem Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Camera Trajectory ──→ [Streaming 3D Reconstruction]             │
│    (keyframe poses       Per-keyframe depth estimation           │
│     + depth maps)        → Incremental point cloud               │
│                          → 3D patch storage                      │
│                                                                  │
│  Target Viewpoint ──→ [Point-to-Frame Retrieval]                 │
│    (camera pose)          Query point cloud for visible patches  │
│                           → Retrieve historical frame patches    │
│                           → Spatially align to target view       │
│                                                                  │
│  Retrieved Patches ──→ [Memory Alignment]                        │
│    Two complementary methods:                                    │
│    ┌─────────────────────────────────────────────┐               │
│    │ Warped RoPE:                                 │               │
│    │   Extends RoPE with geometric reprojection   │               │
│    │   Incorporates temporal coordinates          │               │
│    │   Patches from different timesteps aligned   │               │
│    │   via rotation in Q/K/V space                │               │
│    └─────────────────────────────────────────────┘               │
│    ┌─────────────────────────────────────────────┐               │
│    │ Warped Latent:                               │               │
│    │   Directly transforms retrieved memory       │               │
│    │   patches in latent feature space            │               │
│    │   Better camera motion accuracy              │               │
│    │   Slightly lower visual quality              │               │
│    └─────────────────────────────────────────────┘               │
│                                                                  │
│  [PRoPE Camera Conditioning]                                     │
│    Projective RoPE — encodes relative camera geometry            │
│    via positional encoding in attention                          │
│    Per-frame projective conditioning in Q/K/V rotations          │
│                                                                  │
│  [DiT Backbone (Wan 2.2)]                                        │
│    Memory cross-attention: retrieved patches as K/V              │
│    Text cross-attention: prompt conditioning                     │
│    Self-attention: temporal coherence                             │
│    → Denoised latent frames                                      │
│                                                                  │
│  [Training: Chained Forward Training]                            │
│    Creates larger pose variations during training                │
│    Encourages model to USE spatial memory for consistency        │
│    Rather than relying solely on temporal context                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Component Details

### 4.1 Streaming 3D Reconstruction
- Process keyframes along camera trajectory
- Per-frame monocular depth estimation (off-the-shelf, e.g., Metric3D, DPT)
- Unproject depth maps to 3D point clouds using camera intrinsics/extrinsics
- Incrementally fuse into a global point cloud
- Store patch-level associations: each 3D point links back to its source frame + patch location

### 4.2 Point-to-Frame Retrieval
Given a target camera pose:
1. Render the global point cloud from the target viewpoint
2. For each visible 3D point, retrieve the source frame and patch
3. Collect all retrieved patches → a "mosaic" of spatially aligned memory
4. Flatten patches and concatenate with the token sequence as conditioning

### 4.3 Warped RoPE
Standard RoPE encodes position as rotation in embedding space. Warped RoPE extends this:
- Each patch carries its 3D position and source timestamp
- Geometric reprojection transforms the 3D position into the target view's 2D coordinates
- These reprojected coordinates become the positional encoding for that patch
- Temporal coordinates ensure patches from different timesteps are correctly aligned
- Applied to Q/K/V rotations in the DiT's attention layers

### 4.4 Warped Latent
Complementary to Warped RoPE:
- Instead of modifying positional encoding, directly warp the latent features
- Retrieved memory patch latents are spatially transformed to align with the target view
- Uses the geometric correspondences from the point cloud
- More accurate camera motion, but sometimes lower visual quality

### 4.5 PRoPE (Projective RoPE)
Camera conditioning mechanism from Li et al. (2025):
- Encodes relative camera geometry via projective positional encoding
- Integrates into the attention computation through camera-dependent linear transforms
- Each temporally-compressed latent frame gets per-frame projective conditioning
- Enables the model to reason about cross-view geometric correspondences natively

### 4.6 Memory Cross-Attention
In the DiT backbone:
- Retrieved mosaic patches are flattened into a token sequence
- Injected as key/value in a dedicated cross-attention layer
- Query = current generation tokens
- Key/Value = memory patch tokens (with Warped RoPE positions)
- Combined with temporal memory (recent frames) and text conditioning

### 4.7 Chained Forward Training
Training strategy for long-horizon consistency:
- Chain multiple generation windows during training
- Each window's output becomes the next window's conditioning
- Creates larger effective pose variations than single-window training
- Forces the model to rely on spatial memory for geometric consistency
- Without this, the model would ignore memory and just hallucinate

## 5. Capabilities Demonstrated

| Capability | Description | Scale |
|-----------|-------------|-------|
| Long-horizon navigation | Stable autoregressive rollouts | **2+ minutes** |
| Promptable events | Insert dynamic objects while environment stays stable | Wolf, giraffe, astronaut, etc. |
| Memory manipulation | Splice memories from different scenes | Inception-style impossible spaces |
| Memory-based editing | Add/remove objects in previously seen scenes | Scene modification |
| Precise camera control | Follow user-specified camera trajectories | Sub-degree accuracy |
| Scene revisitation | Return to previously seen locations with consistency | Same scene reconstructed |

## 6. Baselines & Comparisons

| Baseline | Type | MosaicMem advantage |
|----------|------|---------------------|
| GEN3C | Explicit 3D memory | MosaicMem handles dynamic objects; GEN3C freezes them |
| Context-as-Memory | Implicit latent | MosaicMem follows camera poses accurately; implicit drifts |
| RELIC | Retrieval-based | MosaicMem avoids retrieval errors that corrupt generation |

## 7. Technical Stack
- **Diffusion backbone:** Wan 2.2 (video DiT model)
- **Depth estimation:** Off-the-shelf monocular depth (paper doesn't specify which, likely Metric3D or similar)
- **Point cloud:** Incremental fusion from depth + camera poses
- **Works without fine-tuning** on Wan 2.2 (claims in HuggingFace discussion)

## 8. Why This Paper Matters for World Models

MosaicMem directly addresses the "4D consistency" problem that is the central limitation of current video world models. It shows that:

1. **Memory is the missing piece** — not bigger models, not more data, but structured spatial memory
2. **Hybrid approaches win** — neither pure explicit nor pure implicit memory is sufficient
3. **The patch-and-compose paradigm** enables capabilities (scene editing, memory splicing) that neither approach achieves alone
4. **2-minute stable rollouts** demonstrate that long-horizon world simulation is achievable

This is directly relevant to the AMI Labs / world models thesis: world models need persistent spatial memory to be useful for planning, navigation, and embodied AI. MosaicMem provides a concrete mechanism.

## 9. Connection to Verifiable Computation

The pipeline is highly structured and decomposable:
1. Depth estimation → deterministic output from model
2. Point cloud construction → geometric computation
3. Retrieval → spatial query (deterministic given camera pose)
4. Alignment → mathematical transformation (warping)
5. Diffusion → the only stochastic component

Steps 1-4 are fully deterministic and verifiable. Step 5 (diffusion) is where ZK proof of inference applies. The structured memory provides an auditable intermediate state — you can verify what the model "remembers" at each step.
