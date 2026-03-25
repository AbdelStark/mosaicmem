use crate::camera::{CameraIntrinsics, CameraPose};
use crate::geometry::projection::frustum_cull;
use kiddo::SquaredEuclidean;
use nalgebra::{Point2, Point3};
use serde::{Deserialize, Serialize};

type SpatialKdTree = kiddo::float::kdtree::KdTree<f32, u64, 3, 256, u32>;

/// A 3D patch stored in memory with provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patch3D {
    /// Unique patch ID.
    pub id: u64,
    /// 3D center position in world coordinates.
    pub center: Point3<f32>,
    /// Source camera pose for this patch.
    pub source_pose: CameraPose,
    /// Source frame index.
    pub source_frame: usize,
    /// Source frame timestamp.
    pub source_timestamp: f64,
    /// Depth of the patch center in the source camera frame.
    pub source_depth: f32,
    /// 2D bounding box in source frame (top-left x, y, width, height).
    pub source_rect: [f32; 4],
    /// Latent feature vector (from VAE encoding of the patch).
    pub latent: Vec<f32>,
    /// Patch size in the latent grid.
    pub latent_height: usize,
    pub latent_width: usize,
}

/// A retrieved patch with warped coordinates for the target view.
#[derive(Debug, Clone)]
pub struct RetrievedPatch {
    /// Reference to the source patch.
    pub patch: Patch3D,
    /// Projected 2D position in target view.
    pub target_position: Point2<f32>,
    /// Depth in target camera frame.
    pub target_depth: f32,
    /// Visibility score (0..1).
    pub visibility_score: f32,
}

/// Configuration for the memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum number of patches to store.
    pub max_patches: usize,
    /// Maximum number of patches to retrieve per query.
    pub top_k: usize,
    /// Near/far clipping for frustum culling.
    pub near_clip: f32,
    pub far_clip: f32,
    /// Patch size in pixels (source image space).
    pub patch_size: u32,
    /// Latent patch size (after VAE encoding).
    pub latent_patch_size: u32,
    /// Temporal decay half-life in seconds. Patches older than this
    /// relative to the query time have their visibility score halved.
    /// Set to 0.0 to disable temporal decay.
    pub temporal_decay_half_life: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_patches: 10000,
            top_k: 64,
            near_clip: 0.1,
            far_clip: 100.0,
            patch_size: 16,
            latent_patch_size: 2,
            temporal_decay_half_life: 0.0,
        }
    }
}

/// Serializable snapshot of a MosaicMemoryStore (without the KD-tree).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// All stored patches.
    pub patches: Vec<Patch3D>,
    /// Configuration.
    pub config: MemoryConfig,
    /// Next patch ID.
    pub next_id: u64,
}

/// The Mosaic Memory Store: 3D patch storage with spatial indexing.
pub struct MosaicMemoryStore {
    /// All stored patches.
    pub patches: Vec<Patch3D>,
    /// KD-tree for spatial queries.
    kdtree: Option<SpatialKdTree>,
    /// Configuration.
    pub config: MemoryConfig,
    /// Next patch ID.
    next_id: u64,
}

/// Parameters for inserting a keyframe into the memory store.
pub struct KeyframeParams<'a> {
    pub frame_index: usize,
    pub timestamp: f64,
    pub latents: &'a [f32],
    pub latent_h: usize,
    pub latent_w: usize,
    pub channels: usize,
    pub depth_map: &'a [Vec<f32>],
    pub intrinsics: &'a CameraIntrinsics,
    pub pose: &'a CameraPose,
}

impl MosaicMemoryStore {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            patches: Vec::new(),
            kdtree: None,
            config,
            next_id: 0,
        }
    }

    /// Save the memory store to a JSON file.
    pub fn save_json(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let snapshot = MemorySnapshot {
            patches: self.patches.clone(),
            config: self.config.clone(),
            next_id: self.next_id,
        };
        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a memory store from a JSON file.
    pub fn load_json(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        let snapshot: MemorySnapshot = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut store = Self {
            patches: snapshot.patches,
            kdtree: None,
            config: snapshot.config,
            next_id: snapshot.next_id,
        };
        store.rebuild_index();
        Ok(store)
    }

    /// Create a serializable snapshot of the current state.
    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            patches: self.patches.clone(),
            config: self.config.clone(),
            next_id: self.next_id,
        }
    }

    /// Restore from a snapshot.
    pub fn from_snapshot(snapshot: MemorySnapshot) -> Self {
        let mut store = Self {
            patches: snapshot.patches,
            kdtree: None,
            config: snapshot.config,
            next_id: snapshot.next_id,
        };
        store.rebuild_index();
        store
    }

    /// Insert a keyframe: split the frame into patches and store them.
    ///
    /// # Arguments
    /// * `params` - Keyframe parameters (frame index, timestamp, latent data, depth, camera)
    pub fn insert_keyframe_params(&mut self, params: &KeyframeParams) -> Vec<u64> {
        self.insert_keyframe(
            params.frame_index,
            params.timestamp,
            params.latents,
            params.latent_h,
            params.latent_w,
            params.channels,
            params.depth_map,
            params.intrinsics,
            params.pose,
        )
    }

    /// Insert a keyframe: split the frame into patches and store them.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_keyframe(
        &mut self,
        frame_index: usize,
        timestamp: f64,
        latents: &[f32],
        latent_h: usize,
        latent_w: usize,
        channels: usize,
        depth_map: &[Vec<f32>],
        intrinsics: &CameraIntrinsics,
        pose: &CameraPose,
    ) -> Vec<u64> {
        let patch_h = self.config.latent_patch_size as usize;
        let patch_w = self.config.latent_patch_size as usize;
        let mut patch_ids = Vec::new();

        let c2w = pose.camera_to_world();
        let img_h = depth_map.len() as f32;
        let img_w = if depth_map.is_empty() {
            0.0
        } else {
            depth_map[0].len() as f32
        };

        for py in (0..latent_h).step_by(patch_h) {
            for px in (0..latent_w).step_by(patch_w) {
                let ph = patch_h.min(latent_h - py);
                let pw = patch_w.min(latent_w - px);

                // Extract latent features for this patch
                let mut latent = Vec::with_capacity(ph * pw * channels);
                for y in py..py + ph {
                    for x in px..px + pw {
                        let offset = (y * latent_w + x) * channels;
                        if offset + channels <= latents.len() {
                            latent.extend_from_slice(&latents[offset..offset + channels]);
                        }
                    }
                }

                // Compute 3D center by unprojecting the center pixel
                let center_u = (px as f32 + pw as f32 / 2.0) / latent_w as f32 * img_w;
                let center_v = (py as f32 + ph as f32 / 2.0) / latent_h as f32 * img_h;
                let depth_v = (center_v as usize).min(depth_map.len().saturating_sub(1));
                let depth_u = if depth_map.is_empty() {
                    0
                } else {
                    (center_u as usize).min(depth_map[0].len().saturating_sub(1))
                };
                let depth = if !depth_map.is_empty() {
                    depth_map[depth_v][depth_u]
                } else {
                    1.0
                };

                let cam_point = intrinsics.unproject(&Point2::new(center_u, center_v), depth);
                let world_center = c2w.transform_point(&cam_point);

                let source_rect = [
                    px as f32 / latent_w as f32 * img_w,
                    py as f32 / latent_h as f32 * img_h,
                    pw as f32 / latent_w as f32 * img_w,
                    ph as f32 / latent_h as f32 * img_h,
                ];

                let patch = Patch3D {
                    id: self.next_id,
                    center: world_center,
                    source_pose: pose.clone(),
                    source_frame: frame_index,
                    source_timestamp: timestamp,
                    source_depth: depth,
                    source_rect,
                    latent,
                    latent_height: ph,
                    latent_width: pw,
                };

                self.patches.push(patch);
                patch_ids.push(self.next_id);
                self.next_id += 1;
            }
        }

        // Enforce memory budget
        self.enforce_budget();

        // Rebuild KD-tree
        self.rebuild_index();

        patch_ids
    }

    /// Retrieve patches visible from a target camera pose.
    pub fn retrieve(
        &self,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> Vec<RetrievedPatch> {
        self.retrieve_at_time(target_pose, intrinsics, None)
    }

    /// Retrieve patches visible from a target camera pose with optional temporal decay.
    ///
    /// When `query_time` is provided and `temporal_decay_half_life > 0`, patches
    /// older relative to the query time receive exponentially decayed scores.
    pub fn retrieve_at_time(
        &self,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
        query_time: Option<f64>,
    ) -> Vec<RetrievedPatch> {
        if self.patches.is_empty() {
            return vec![];
        }

        // Frustum cull: find patches whose 3D centers are visible
        let centers: Vec<Point3<f32>> = self.patches.iter().map(|p| p.center).collect();
        let visible_indices = frustum_cull(
            &centers,
            intrinsics,
            target_pose,
            self.config.near_clip,
            self.config.far_clip,
        );

        let half_life = self.config.temporal_decay_half_life;

        // Score and rank visible patches
        let mut candidates: Vec<RetrievedPatch> = visible_indices
            .iter()
            .filter_map(|&idx| {
                let patch = &self.patches[idx];
                let cam_point = target_pose.transform_point(&patch.center);
                let pixel = intrinsics.project(&cam_point)?;

                // Visibility score: closer and more centered patches score higher
                let center_dist = ((pixel.x - intrinsics.cx) / intrinsics.width as f32).powi(2)
                    + ((pixel.y - intrinsics.cy) / intrinsics.height as f32).powi(2);
                let depth_score = 1.0 / (1.0 + cam_point.z * 0.1);
                let mut visibility_score = (1.0 - center_dist.sqrt()).max(0.0) * depth_score;

                // Apply temporal decay if configured
                if half_life > 0.0
                    && let Some(qt) = query_time
                {
                    let age = (qt - patch.source_timestamp).max(0.0);
                    let decay = (0.5_f64).powf(age / half_life) as f32;
                    visibility_score *= decay;
                }

                Some(RetrievedPatch {
                    patch: patch.clone(),
                    target_position: pixel,
                    target_depth: cam_point.z,
                    visibility_score,
                })
            })
            .collect();

        // Sort by visibility score (descending) and take top-K
        candidates.sort_by(|a, b| {
            b.visibility_score
                .partial_cmp(&a.visibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.top_k);

        candidates
    }

    /// Delete patches from a specific frame.
    pub fn delete_frame(&mut self, frame_index: usize) {
        self.patches.retain(|p| p.source_frame != frame_index);
        self.rebuild_index();
    }

    /// Delete a specific patch by ID.
    pub fn delete_patch(&mut self, patch_id: u64) {
        self.patches.retain(|p| p.id != patch_id);
        self.rebuild_index();
    }

    /// Enforce the memory budget by removing oldest patches.
    fn enforce_budget(&mut self) {
        if self.patches.len() > self.config.max_patches {
            // Remove oldest patches first
            self.patches.sort_by(|a, b| {
                b.source_timestamp
                    .partial_cmp(&a.source_timestamp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.patches.truncate(self.config.max_patches);
        }
    }

    /// Rebuild the KD-tree index.
    fn rebuild_index(&mut self) {
        if self.patches.is_empty() {
            self.kdtree = None;
            return;
        }
        let mut tree = SpatialKdTree::new();
        for (i, patch) in self.patches.iter().enumerate() {
            let pos = [patch.center.x, patch.center.y, patch.center.z];
            tree.add(&pos, i as u64);
        }
        self.kdtree = Some(tree);
    }

    /// Query patches near a 3D position.
    pub fn query_nearest(&self, position: &Point3<f32>, k: usize) -> Vec<&Patch3D> {
        match &self.kdtree {
            Some(tree) => {
                let results =
                    tree.nearest_n::<SquaredEuclidean>(&[position.x, position.y, position.z], k);
                results
                    .iter()
                    .filter_map(|n| self.patches.get(n.item as usize))
                    .collect()
            }
            None => vec![],
        }
    }

    /// Get total number of stored patches.
    pub fn num_patches(&self) -> usize {
        self.patches.len()
    }

    /// Get total latent token count across all patches.
    pub fn total_tokens(&self) -> usize {
        self.patches
            .iter()
            .map(|p| p.latent_height * p.latent_width)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_store_with_patches() -> MosaicMemoryStore {
        let config = MemoryConfig {
            max_patches: 100,
            top_k: 10,
            ..Default::default()
        };
        let mut store = MosaicMemoryStore::new(config);

        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let depth_map = vec![vec![5.0f32; 10]; 10];
        let latents = vec![0.5f32; 4 * 4 * 4]; // 4x4 latent grid, 4 channels

        store.insert_keyframe(0, 0.0, &latents, 4, 4, 4, &depth_map, &intrinsics, &pose);
        store
    }

    #[test]
    fn test_insert_and_retrieve() {
        let store = make_store_with_patches();
        assert!(store.num_patches() > 0);

        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let query_pose = CameraPose::identity(1.0);

        let retrieved = store.retrieve(&query_pose, &intrinsics);
        assert!(!retrieved.is_empty());
    }

    #[test]
    fn test_delete_frame() {
        let mut store = make_store_with_patches();
        let initial_count = store.num_patches();
        store.delete_frame(0);
        assert_eq!(store.num_patches(), 0);
        assert!(initial_count > 0);
    }

    #[test]
    fn test_query_nearest() {
        let store = make_store_with_patches();
        let results = store.query_nearest(&Point3::new(0.0, 0.0, 5.0), 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_save_load_json_roundtrip() {
        let store = make_store_with_patches();
        let initial_count = store.num_patches();
        assert!(initial_count > 0);

        let dir = std::env::temp_dir().join("mosaicmem_test_save_load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("memory.json");

        store.save_json(&path).unwrap();

        let loaded = MosaicMemoryStore::load_json(&path).unwrap();
        assert_eq!(loaded.num_patches(), initial_count);
        assert_eq!(loaded.config.max_patches, store.config.max_patches);

        // Verify KD-tree was rebuilt
        let results = loaded.query_nearest(&Point3::new(0.0, 0.0, 5.0), 3);
        assert!(!results.is_empty());

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let store = make_store_with_patches();
        let snapshot = store.snapshot();
        let restored = MosaicMemoryStore::from_snapshot(snapshot);
        assert_eq!(restored.num_patches(), store.num_patches());
    }

    #[test]
    fn test_temporal_decay_retrieval() {
        let config = MemoryConfig {
            max_patches: 100,
            top_k: 10,
            temporal_decay_half_life: 1.0, // 1 second half-life
            ..Default::default()
        };
        let mut store = MosaicMemoryStore::new(config);

        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let depth_map = vec![vec![5.0f32; 10]; 10];
        let latents = vec![0.5f32; 4 * 4 * 4];

        // Insert at time 0
        store.insert_keyframe(0, 0.0, &latents, 4, 4, 4, &depth_map, &intrinsics, &pose);
        assert!(store.num_patches() > 0);

        // Retrieve at time 0 — no decay
        let query = CameraPose::identity(0.0);
        let at_t0 = store.retrieve_at_time(&query, &intrinsics, Some(0.0));

        // Retrieve at time 10 — significant decay
        let at_t10 = store.retrieve_at_time(&query, &intrinsics, Some(10.0));

        if !at_t0.is_empty() && !at_t10.is_empty() {
            // Scores at t=10 should be lower than at t=0
            let max_t0 = at_t0
                .iter()
                .map(|p| p.visibility_score)
                .fold(0.0f32, f32::max);
            let max_t10 = at_t10
                .iter()
                .map(|p| p.visibility_score)
                .fold(0.0f32, f32::max);
            assert!(
                max_t10 < max_t0,
                "Temporal decay should reduce scores: t0={} t10={}",
                max_t0,
                max_t10
            );
        }
    }

    #[test]
    fn test_insert_keyframe_params() {
        let config = MemoryConfig {
            max_patches: 100,
            top_k: 10,
            ..Default::default()
        };
        let mut store = MosaicMemoryStore::new(config);

        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);
        let depth_map = vec![vec![5.0f32; 10]; 10];
        let latents = vec![0.5f32; 4 * 4 * 4];

        let params = KeyframeParams {
            frame_index: 0,
            timestamp: 0.0,
            latents: &latents,
            latent_h: 4,
            latent_w: 4,
            channels: 4,
            depth_map: &depth_map,
            intrinsics: &intrinsics,
            pose: &pose,
        };

        let ids = store.insert_keyframe_params(&params);
        assert!(!ids.is_empty());
        assert!(store.num_patches() > 0);
    }

    #[test]
    fn test_insert_keyframe_records_source_provenance() {
        let store = make_store_with_patches();
        let patch = &store.patches[0];
        assert_eq!(patch.source_frame, 0);
        assert_eq!(patch.source_timestamp, 0.0);
        assert!((patch.source_depth - 5.0).abs() < 1e-5);
        assert_eq!(patch.source_pose.timestamp, 0.0);
    }
}
