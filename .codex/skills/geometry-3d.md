---
name: geometry-3d
description: 3D geometry operations in mosaicmem-rs — camera models, projections, point clouds, depth estimation, and streaming fusion. Activate when working with camera poses, 3D points, depth maps, unprojection, or spatial queries.
prerequisites: nalgebra 0.33, kiddo 4.x
---

# 3D Geometry

<purpose>
Covers the geometry pipeline: camera models → depth estimation → unprojection → point cloud fusion → spatial queries. This is the foundation that lifts 2D video into 3D space.
</purpose>

<context>
— nalgebra types: Point3<f32>, Vector3<f32>, Matrix4<f32>, Isometry3<f32>
— CameraPose wraps Isometry3 (world-to-camera transform)
— CameraIntrinsics: pinhole model (fx, fy, cx, cy, width, height)
— PointCloud3D: Vec of Point3DColored (position, color, optional normal)
— StreamingFusion: Incremental builder with KD-tree (kiddo)
— DepthEstimator trait: abstracts depth prediction (Synthetic stub exists)
</context>

<procedure>
1. Camera setup: Create CameraPose via `look_at()` or `from_translation_rotation()`
2. Depth: Use DepthEstimator trait to get HxW depth map
3. Unproject: `unproject_depth_map(depth, intrinsics, pose)` → world-space points
4. Fuse: `StreamingFusion::add_keyframe()` incrementally builds point cloud
5. Query: `query_radius()` or `query_knn()` for spatial lookups
6. Project back: `project_points()` maps 3D → 2D for a target viewpoint

Key decision: Always rebuild KD-tree after batch inserts (`rebuild_kdtree()`), not per-insert.
</procedure>

<patterns>
<do>
  — Use `Isometry3` for rigid transforms (rotation + translation), not Matrix4
  — Use `Point3` for positions, `Vector3` for directions
  — Call `rebuild_kdtree()` after adding points to StreamingFusion
  — Clamp normalized coordinates to [0, 1] before converting to pixel coords
  — Use `frustum_cull()` before projection to skip invisible points
  — Use `voxel_downsample()` to control point cloud density
</do>
<dont>
  — Don't use Matrix4 for camera transforms — use Isometry3 (preserves SE(3) semantics)
  — Don't query KD-tree before calling `rebuild_kdtree()`
  — Don't assume depth values are positive — filter negative/NaN values
  — Don't serialize nalgebra types directly — use custom serde module in camera/pose.rs
</dont>
</patterns>

<examples>
Example: Unproject depth map to world coordinates
```rust
use crate::camera::{CameraPose, CameraIntrinsics};
use crate::geometry::projection::unproject_depth_map;

let intrinsics = CameraIntrinsics::new(500.0, 500.0, 320.0, 240.0, 640, 480);
let pose = CameraPose::identity(0.0);
let depth: Vec<Vec<f32>> = vec![vec![1.0; 640]; 480]; // HxW

let points = unproject_depth_map(&depth, &intrinsics, &pose);
// points: Vec<Point3<f32>> in world coordinates
```
</examples>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| KD-tree query returns empty | Tree not rebuilt after inserts | Call `rebuild_kdtree()` |
| Projected points outside image | Points behind camera or out of FOV | Use `frustum_cull()` first |
| NaN in unprojected points | Zero or negative depth values | Filter depth > 0 before unproject |
| Serde error on CameraPose | Direct nalgebra serialization | Use custom serde in pose.rs |
</troubleshooting>

<references>
— src/camera/pose.rs: CameraPose with SE(3) transforms
— src/camera/intrinsics.rs: Pinhole camera model
— src/geometry/projection.rs: unproject/project/frustum_cull
— src/geometry/pointcloud.rs: PointCloud3D operations
— src/geometry/fusion.rs: StreamingFusion with KD-tree
— src/geometry/depth.rs: DepthEstimator trait
— RFC-001-core-types.md: Core type definitions
— RFC-002-geometry-pipeline.md: Geometry pipeline design
</references>
