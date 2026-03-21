---
name: memory-system
description: Spatial memory store for mosaicmem-rs — 3D patch storage, KD-tree retrieval, mosaic frame composition, and memory manipulation (splice, edit, erase). Activate when working with patches, memory budget, retrieval scoring, coverage grids, or scene manipulation.
prerequisites: kiddo 4.x, nalgebra 0.33
---

# Memory System

<purpose>
The memory system stores 3D patches extracted from generated frames and retrieves them for new viewpoints. It's the core of MosaicMem's spatial consistency — enabling scene revisitation and long-horizon generation.
</purpose>

<context>
— Patch3D: center (Point3), source_frame, timestamp, latent features, 2D source rect
— MosaicMemoryStore: patches Vec + KD-tree index, enforces max_patches budget
— MemoryRetriever: high-level API producing MosaicFrame for a target viewpoint
— MosaicFrame: assembled patches + coverage grid (1/8 resolution)
— Manipulation: splice_horizontal, flip_vertical, erase_region, translate, scale
— MemoryConfig: max_patches, top_k, near_clip, far_clip, patch dimensions
</context>

<procedure>
1. Insert: `store.insert_keyframe(frame_idx, patches)` — adds patches to store + rebuilds KD-tree
2. Retrieve: `store.retrieve(pose, intrinsics, top_k)` → Vec<RetrievedPatch>
   — Projects all patches to target view
   — Scores by visibility (center distance + depth)
   — Returns top_k sorted by score
3. Compose: `MosaicFrame::compose_tokens(patches)` → token grid for attention
4. Budget: When max_patches exceeded, evict oldest via voxel-based strategy
5. Manipulate: Use manipulation functions for scene editing (splice, erase, etc.)
</procedure>

<patterns>
<do>
  — Check `store.num_patches() > 0` before retrieval
  — Use `MosaicFrame::has_coverage()` to check if enough patches cover the target view
  — Use `coverage_ratio()` to quantify how much of the view is covered vs. needs inpainting
  — Use `delete_frame(frame_idx)` to remove all patches from a specific source frame
  — Keep `max_patches` tuned to available memory (default is suitable for testing)
</do>
<dont>
  — Don't query retrieval with an empty store — returns empty gracefully but wastes cycles
  — Don't modify `store.patches` directly — use the public API methods
  — Don't assume patches are ordered — they're spatially indexed, not sequentially stored
  — Don't skip KD-tree rebuild after manual patch modifications
</dont>
</patterns>

<examples>
Example: Insert keyframe and retrieve for new viewpoint
```rust
use crate::memory::store::{MosaicMemoryStore, MemoryConfig, Patch3D};
use nalgebra::Point3;

let mut store = MosaicMemoryStore::new(MemoryConfig::default());

let patches = vec![Patch3D {
    center: Point3::new(1.0, 0.0, 5.0),
    source_frame: 0,
    timestamp: 0.0,
    latent: vec![0.1; 64],
    source_rect: [10, 10, 26, 26],
}];
store.insert_keyframe(0, patches);

let retrieved = store.retrieve(&target_pose, &intrinsics, 10);
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Retrieval returns empty | No patches in store or all behind camera | Check `num_patches()`, verify target pose |
| Coverage ratio always 0 | Patches don't project into target view | Check camera intrinsics and pose alignment |
| Memory budget not enforced | max_patches set too high | Lower MemoryConfig::max_patches |
| Splice produces empty result | Both stores empty | Verify stores have patches before splicing |
</troubleshooting>

<references>
— src/memory/store.rs: MosaicMemoryStore, Patch3D, MemoryConfig
— src/memory/retrieval.rs: MemoryRetriever, MosaicFrame
— src/memory/mosaic.rs: MosaicFrame composition and coverage
— src/memory/manipulation.rs: Scene manipulation operations
— RFC-003-mosaic-memory-store.md: Memory store design
</references>
