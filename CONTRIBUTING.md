# Contributing

## Scope

This project is an alpha-stage Rust library and CLI for a synthetic MosaicMem pipeline. Contributions should improve correctness, testability, documentation, and research ergonomics. Avoid marketing-only changes that overstate current capabilities.

## Setup

1. Install a recent stable Rust toolchain with edition 2024 support.
2. Clone the repository.
3. Run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Development Rules

- Keep library errors typed and contextual.
- Add or update tests for every behavioral change.
- Preserve tensor layout invariants: decoded tensors are flattened `[B, C, T, H, W]`.
- Keep synthetic behavior deterministic when a config seed exists.
- Update `README.md`, `CHANGELOG.md`, and `AGENTS.md` when user-facing behavior changes.

## Pull Request Checklist

- `cargo fmt --check` passes.
- `cargo clippy --all-targets --all-features -- -D warnings` passes.
- `cargo test` passes.
- New behavior is documented.
- A regression test exists for every bug fix.

## Reporting Issues

Open a GitHub issue with:

- the command or API you ran
- the exact input or config
- the observed behavior
- the expected behavior
- local environment details if relevant
