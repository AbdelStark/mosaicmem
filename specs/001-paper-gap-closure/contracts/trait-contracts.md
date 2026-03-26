# Trait Contracts: Paper Gap Closure

These are the public trait interfaces that MUST be satisfied by both
synthetic and real backends. Any implementation that passes these
contracts is valid.

## DiffusionBackbone (existing, unchanged)

```rust
pub trait DiffusionBackbone {
    fn denoise(
        &self,
        noisy_latent: &ArrayD<f32>,   // [B, C, T, H, W]
        timestep: f32,
        text_embedding: &[Vec<f32>],
        memory_context: Option<&MemoryContext>,
    ) -> Result<ArrayD<f32>, BackboneError>;  // same shape as input
}
```

**Contract**:
- Output shape MUST equal input shape
- Output values MUST be finite (no NaN/Inf)
- With `memory_context: None`, output MUST be independent of memory state

## VAE (existing, shape contract tightened)

```rust
pub trait VAE {
    fn encode(&self, frames: &TensorView) -> Result<TensorView, VaeError>;
    fn decode(&self, latent: &TensorView) -> Result<TensorView, VaeError>;

    fn downsample_factor(&self) -> u32;
    fn latent_channels(&self) -> u32;
    fn temporal_compression(&self) -> u32;
}
```

**Contract**:
- `encode([B,C,T,H,W])` -> `[B, latent_channels, T/temporal_compression, H/downsample, W/downsample]`
- `decode(latent)` -> `[B, 3, T*temporal_compression, H*downsample, W*downsample]`
- `decode(encode(x))` round-trip error < epsilon (backend-dependent)

## DepthEstimator (existing, output contract tightened)

```rust
pub trait DepthEstimator {
    fn estimate(&self, frame: &TensorView) -> Result<TensorView, DepthError>;
}
```

**Contract**:
- Input: `[C, H, W]` or `[1, H, W]`
- Output: `[1, H, W]` depth map, values > 0 (positive depth)
- Output spatial dimensions MUST match input spatial dimensions

## MemoryRetriever (new trait)

```rust
pub trait MemoryRetriever {
    fn retrieve_for_frame(
        &self,
        store: &MosaicMemoryStore,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
        config: &RetrievalConfig,
    ) -> Result<FrameRetrievalResult, RetrievalError>;
}
```

**Contract**:
- Returns at most `config.top_k` patches
- All returned patches MUST be within the target camera's frustum
- Patches MUST be scored with temporal decay and diversity filtering
- Different `target_pose` values MUST potentially produce different results

## WarpOperator (new trait)

```rust
pub trait WarpOperator {
    fn compute_warp_grid(
        &self,
        patch: &PatchMetadata,
        target_pose: &CameraPose,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<WarpGrid, WarpError>;

    fn apply_warp(
        &self,
        source_latent: &TensorView,
        grid: &WarpGrid,
    ) -> Result<(TensorView, TensorView), WarpError>;
    // Returns (warped_latent, valid_mask)
}
```

**Contract**:
- `WarpGrid` MUST have one coordinate per source token
- Invalid samples (behind camera) MUST be marked in valid_mask
- Output latent shape MUST match source latent spatial dims
- With identity transform (same source/target pose), warp MUST be
  approximately identity

## PRoPEOperator (new trait)

```rust
pub trait PRoPEOperator {
    fn compute_projective_transform(
        &self,
        source_pose: &CameraPose,
        target_pose: &CameraPose,
        source_intrinsics: &CameraIntrinsics,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<ProjectiveTransform, PRoPEError>;

    fn apply_to_attention(
        &self,
        queries: &mut TensorView,
        keys: &mut TensorView,
        transform: &ProjectiveTransform,
    ) -> Result<(), PRoPEError>;
}
```

**Contract**:
- `ProjectiveTransform` MUST represent `P_i * P_j^{+}` (relative
  projective transform)
- With identical source/target cameras, transform MUST be identity
- `apply_to_attention` MUST modify Q/K multiplicatively (rotary), not
  additively
- Q/K shapes MUST be preserved
