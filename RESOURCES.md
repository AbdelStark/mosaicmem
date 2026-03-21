# mosaicmem-rs Resources

## Paper & Official Materials
| Resource | URL |
|----------|-----|
| Paper (arXiv) | https://arxiv.org/abs/2603.17117 |
| Project Page | https://mosaicmem.github.io/mosaicmem/ |
| Demo Video | https://www.youtube.com/watch?v=K3Q9kf8t08I |
| HuggingFace | https://huggingface.co/papers/2603.17117 |

## Authors
- **Wei Yu** (project lead) — Animesh Garg's lab
- **Runjia Qian, Yumeng Li** (core contributors)
- **Animesh Garg** — senior author, Georgia Tech / NVIDIA

## Prerequisite Papers
| Paper | Why |
|-------|-----|
| PRoPE (Li et al., 2025) | Camera conditioning mechanism used by MosaicMem |
| Wan 2.2 | DiT video diffusion backbone |
| GEN3C (Ren et al., 2025) | Explicit 3D memory baseline |
| RoPE (Su et al., 2024) | Base positional encoding that Warped RoPE extends |
| Mask2Former (Cheng et al., 2022) | Cross-attention conditioning pattern |

## Rust Ecosystem
| Crate | Version | Purpose |
|-------|---------|---------|
| burn | 0.16 | Neural network framework (attention, RoPE) |
| burn-ndarray | 0.16 | CPU backend for Burn |
| tract-onnx | 0.21 | ONNX model inference (depth, DiT, VAE) |
| nalgebra | 0.33 | 3D geometry (SE3, projections) |
| kiddo | 4.0 | KD-tree for spatial queries |
| image | 0.25 | Image I/O |
| rayon | 1.10 | Parallel computation |
| ratatui | 0.29 | TUI visualization |
| clap | 4.0 | CLI |

## Related Implementations
- **VGGT** (geometry foundation model) — similar depth/point cloud pipeline
- **ORV** (occupancy-centric world model) — similar 3D → video generation
- **VGGT-World** — temporal flow on frozen geometry features

## Depth Estimation Models (for Tract/ONNX)
- Metric3D v2: https://github.com/YvanYin/Metric3D
- DPT-Large: https://huggingface.co/Intel/dpt-large
- Depth Anything v2: https://github.com/DepthAnything/Depth-Anything-V2
