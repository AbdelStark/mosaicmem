# mosaicmem-rs

Rust implementation of **MosaicMem: Hybrid Spatial Memory for Controllable Video World Models** (Yu*, Qian*, Li* et al., 2026).

MosaicMem bridges explicit 3D memory and implicit latent memory for video world models. It lifts video patches into 3D for reliable localization and retrieval, then injects them via the model's native attention mechanism — enabling 2-minute stable rollouts, scene revisitation, dynamic objects, and memory-based editing.

> **Paper:** [arXiv:2603.17117](https://arxiv.org/abs/2603.17117)
> **Project:** [mosaicmem.github.io](https://mosaicmem.github.io/mosaicmem/)
> **Demo:** [YouTube](https://www.youtube.com/watch?v=K3Q9kf8t08I)

Built with **Burn** (neural ops, attention) + **Tract** (ONNX inference) + **nalgebra** (3D geometry).

## The Core Idea

```
Persistent geometry → from 3D memory (explicit patches)
Dynamic content    → from the model (implicit diffusion)

Mosaic = patches stitched in target view
       + gaps left for model to inpaint/evolve
```

## Architecture

```
Camera Trajectory → [Streaming 3D Reconstruction] → Point Cloud
                                                         │
Target Viewpoint → [Point-to-Frame Retrieval] ←──────────┘
                         │
                    Retrieved Patches
                         │
              ┌──────────┴──────────┐
              │                     │
         [Warped RoPE]      [Warped Latent]
         (positional)        (feature space)
              │                     │
              └──────────┬──────────┘
                         │
                   [Memory Cross-Attention]
                   Q: generation tokens
                   K/V: memory patches
                         │
              [PRoPE Camera Conditioning]
                         │
                   [DiT Backbone]  ← Wan 2.2 via ONNX
                         │
                   Generated Frames
```

## Capabilities (from paper)

| Capability | What it means |
|-----------|---------------|
| 2-min rollouts | Stable autoregressive generation without collapse |
| Scene revisitation | Return to a location → same scene reconstructed |
| Promptable events | Insert a wolf while the environment stays stable |
| Memory splicing | Combine scenes into impossible Inception-style spaces |
| Scene editing | Add/remove objects via memory manipulation |
| Camera control | Sub-degree accuracy following trajectories |

## RFCs

| RFC | Component |
|-----|-----------|
| [001](rfcs/RFC-001-core-types.md) | Core Types (Camera, PointCloud, Patch, Mosaic) |
| [002](rfcs/RFC-002-geometry-pipeline.md) | Geometry Pipeline (Depth, Unprojection, Fusion) |
| [003](rfcs/RFC-003-mosaic-memory-store.md) | Mosaic Memory Store (3D Patch Storage + Retrieval) |
| [004](rfcs/RFC-004-warped-rope.md) | Warped RoPE (Geometric Positional Encoding) |
| [005](rfcs/RFC-005-warped-latent.md) | Warped Latent (Feature Space Alignment) |
| [006](rfcs/RFC-006-prope.md) | PRoPE Camera Conditioning |
| [007](rfcs/RFC-007-memory-cross-attention.md) | Memory Cross-Attention (Inject into DiT) |
| [008](rfcs/RFC-008-autoregressive-pipeline.md) | Autoregressive Pipeline (Chained Generation) |

## License

MIT OR Apache-2.0

## Citation

```bibtex
@article{mosaicmem2026,
  title={MosaicMem: Hybrid Spatial Memory for Controllable Video World Models},
  author={Wei Yu and Runjia Qian and Yumeng Li and Liquan Wang and Songheng Yin and 
          Sri Siddarth Chakaravarthy P and Dennis Anthony and Yang Ye and Yidi Li and 
          Weiwei Wan and Animesh Garg},
  journal={arXiv preprint arXiv:2603.17117},
  year={2026}
}
```
