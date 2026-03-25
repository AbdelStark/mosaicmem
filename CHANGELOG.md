# Changelog

All notable user-visible changes should be recorded here.

## Unreleased

### Fixed

- Made synthetic pipeline noise deterministic from `PipelineConfig.seed`.
- Fixed decoded frame handling so memory updates, overlap blending, and PNG export respect flattened `[B, C, T, H, W]` layout.
- Fixed the end-to-end revisit regression so the full `cargo test` suite passes again.
- Increased KD-tree bucket capacity to reduce failures on highly aligned synthetic geometry.
- Fixed `splice` to build synthetic memory stores from the provided trajectories instead of reporting success on empty stores.

### Documentation

- Rewrote the README to reflect the current alpha status, validated commands, limitations, and roadmap.
- Added `AGENTS.md` and `CONTRIBUTING.md`.
- Added CI to run formatting, clippy, and tests on pushes and pull requests.

### Repository Hygiene

- Ignored generated output and local tool cache directories that should not pollute the working tree.
- Reduced dependency audit findings by narrowing `image` features and updating `ratatui` / `nalgebra`; one `paste` warning remains through `nalgebra`.
