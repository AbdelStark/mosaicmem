# mosaicmem-rs

Spatial memory for camera-controlled video generation in Rust.

`mosaicmem-rs` implements the memory side of MosaicMem: streaming 3D reconstruction, patch-level spatial memory, view-conditioned retrieval, geometric alignment, and autoregressive generation plumbing. The repo also ships deterministic synthetic backends, so the full pipeline runs end to end without external model weights.

Paper: [arXiv:2603.17117](https://arxiv.org/abs/2603.17117)  
Project: [mosaicmem.github.io](https://mosaicmem.github.io/mosaicmem/)  
Demo: [YouTube](https://www.youtube.com/watch?v=K3Q9kf8t08I)


Video world models break when the camera moves too far, revisits an old area, or tries to hold onto scene structure over long rollouts. MosaicMem fixes that with explicit spatial memory:

- estimate depth from keyframes
- lift patches into 3D
- store them in a searchable memory
- retrieve the right patches for a target pose
- align them with Warped RoPE and Warped Latent
- inject them into a diffusion-style generation loop

The result is a geometry-aware memory stack you can test, inspect, and extend from Rust.

## Quick Start

```bash
git clone https://github.com/AbdelStark/mosaicmem.git
cd mosaicmem
cargo test
cargo run -- demo --num-frames 16 --width 64 --height 64 --steps 5
```

Use the CLI:

```bash
cargo run -- --help
cargo run -- generate --trajectory trajectory.json --output out
cargo run -- inspect --trajectory trajectory.json --coverage
cargo run -- export-ply --trajectory trajectory.json --output scene.ply
```

Use it as a library:

```toml
[dependencies]
mosaicmem-rs = { git = "https://github.com/AbdelStark/mosaicmem.git" }
```

## License

MIT

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
