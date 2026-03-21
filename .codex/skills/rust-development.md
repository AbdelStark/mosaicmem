---
name: rust-development
description: Rust coding patterns, error handling, testing, and toolchain conventions for mosaicmem-rs. Activate when writing new Rust code, fixing compiler errors, adding tests, or resolving clippy warnings.
prerequisites: Rust 1.85+ (edition 2024), cargo
---

# Rust Development

<purpose>
Guides Rust implementation within mosaicmem-rs conventions. Covers error handling, testing patterns, trait design, and toolchain usage.
</purpose>

<context>
— Edition 2024 Rust with stable toolchain
— No unsafe code anywhere in the codebase
— No async/await — all synchronous
— thiserror for error enums, tracing for logging
— approx crate for floating-point test assertions
</context>

<procedure>
1. Write implementation following the patterns below
2. Add `#[cfg(test)] mod tests` with at least 3 unit tests
3. Run `cargo check` for fast type feedback
4. Run `cargo test` — all must pass
5. Run `cargo clippy` — resolve all warnings
6. Run `cargo fmt` — ensure formatting
</procedure>

<patterns>
<do>
  — Define error enums with thiserror:
    ```rust
    #[derive(Debug, thiserror::Error)]
    pub enum MyError {
        #[error("description: {0}")]
        Variant(String),
    }
    ```
  — Use `Result<T, MyError>` for fallible operations
  — Derive `Debug, Clone` on all public types where possible
  — Add `#[derive(serde::Serialize, serde::Deserialize)]` for types that need JSON I/O
  — Use `tracing::info!`, `debug!`, `warn!` for logging
  — Use `approx::assert_relative_eq!` for float comparisons in tests
  — Test edge cases: empty inputs, single elements, boundary values
</do>
<dont>
  — Don't use `unwrap()` in library code — use `?` operator
  — Don't use `println!` — use tracing macros
  — Don't add `unsafe` blocks
  — Don't use `String` where `&str` suffices in function parameters
  — Don't create deep module hierarchies — keep flat (one level under src/)
</dont>
</patterns>

<examples>
Example: Adding a new type with tests
```rust
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum FooError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

#[derive(Debug, Clone)]
pub struct Foo {
    pub value: f32,
}

impl Foo {
    pub fn new(value: f32) -> Result<Self, FooError> {
        if value.is_nan() {
            return Err(FooError::InvalidInput("NaN".into()));
        }
        Ok(Self { value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo_valid() {
        let f = Foo::new(1.0).unwrap();
        assert_eq!(f.value, 1.0);
    }

    #[test]
    fn test_foo_nan() {
        assert!(Foo::new(f32::NAN).is_err());
    }

    #[test]
    fn test_foo_zero() {
        let f = Foo::new(0.0).unwrap();
        assert_eq!(f.value, 0.0);
    }
}
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| `edition = "2024"` not recognized | Rust < 1.85 | `rustup update stable` |
| Clippy: `needless_return` | Explicit `return` at end of fn | Remove `return`, use expression |
| Clippy: `clone_on_copy` | Calling `.clone()` on Copy type | Remove `.clone()` |
| Test: `assert_relative_eq` not found | Missing dev-dependency | Already in Cargo.toml as `approx` |
</troubleshooting>

<references>
— src/lib.rs: Module declarations
— Cargo.toml: Dependencies and edition
</references>
