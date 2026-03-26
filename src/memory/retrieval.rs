use crate::camera::{CameraIntrinsics, CameraPose};
use crate::geometry::projection::frustum_cull;
use crate::memory::mosaic::MosaicFrame;
use crate::memory::store::{MosaicMemoryStore, Patch3D, RetrievedPatch};
use nalgebra::Point2;
use std::collections::BTreeSet;
use thiserror::Error;

pub mod traits {
    use super::*;

    pub trait MemoryRetriever {
        fn retrieve_for_frame(
            &self,
            store: &MosaicMemoryStore,
            target_pose: &CameraPose,
            intrinsics: &CameraIntrinsics,
            config: &RetrievalConfig,
        ) -> Result<FrameRetrievalResult, RetrievalError>;
    }
}

#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub top_k: usize,
    pub near_clip: f32,
    pub far_clip: f32,
    pub min_visibility: f32,
    pub diversity_radius: f32,
    pub diversity_penalty: f32,
    pub temporal_decay_half_life: f64,
}

#[derive(Debug, Clone)]
pub struct FrameRetrievalResult {
    pub target_pose: CameraPose,
    pub patches: Vec<RetrievedPatch>,
    pub coverage_mask: Vec<Vec<bool>>,
    pub width: u32,
    pub height: u32,
}

impl FrameRetrievalResult {
    pub fn into_mosaic(self) -> MosaicFrame {
        MosaicFrame {
            target_pose: self.target_pose,
            patches: self.patches,
            coverage_mask: self.coverage_mask,
            width: self.width,
            height: self.height,
        }
    }
}

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("window retrieval requires at least one pose")]
    EmptyWindow,
}

/// High-level memory retrieval that produces MosaicFrames
/// from the memory store given target camera poses.
///
/// Supports diversity-aware retrieval: when `diversity_radius` > 0,
/// patches too close to already-selected patches in the target 2D view
/// are penalized to encourage spatial coverage diversity.
pub struct MemoryRetriever {
    /// Minimum visibility score threshold.
    pub min_visibility: f32,
    /// Whether to sort patches by depth (front-to-back).
    pub depth_sort: bool,
    /// Spatial diversity radius in target 2D pixels. Patches within this
    /// radius of an already-selected patch have their score penalized.
    /// Set to 0.0 to disable diversity filtering.
    pub diversity_radius: f32,
    /// Penalty factor for patches within the diversity radius (0..1).
    /// A value of 0.5 means the score is halved for each nearby selected patch.
    pub diversity_penalty: f32,
}

impl MemoryRetriever {
    pub fn new() -> Self {
        Self {
            min_visibility: 0.01,
            depth_sort: true,
            diversity_radius: 0.0,
            diversity_penalty: 0.5,
        }
    }

    /// Create a retriever with diversity filtering enabled.
    pub fn with_diversity(diversity_radius: f32, diversity_penalty: f32) -> Self {
        Self {
            min_visibility: 0.01,
            depth_sort: true,
            diversity_radius,
            diversity_penalty,
        }
    }

    fn config_for_store(&self, store: &MosaicMemoryStore) -> RetrievalConfig {
        RetrievalConfig {
            top_k: store.config.top_k,
            near_clip: store.config.near_clip,
            far_clip: store.config.far_clip,
            min_visibility: self.min_visibility,
            diversity_radius: self.diversity_radius,
            diversity_penalty: self.diversity_penalty,
            temporal_decay_half_life: store.config.temporal_decay_half_life,
        }
    }

    /// Retrieve a MosaicFrame for a target camera pose.
    pub fn retrieve(
        &self,
        store: &MosaicMemoryStore,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
    ) -> MosaicFrame {
        self.retrieve_at_time(store, target_pose, intrinsics, None)
    }

    /// Retrieve a MosaicFrame with temporal decay applied.
    pub fn retrieve_at_time(
        &self,
        store: &MosaicMemoryStore,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
        query_time: Option<f64>,
    ) -> MosaicFrame {
        let config = self.config_for_store(store);
        self.retrieve_from_candidates(store, None, target_pose, intrinsics, query_time, &config)
            .into_mosaic()
    }

    /// Retrieve a frame-aware set of results for all poses in a window.
    pub fn retrieve_window(
        &self,
        store: &MosaicMemoryStore,
        poses: &[CameraPose],
        intrinsics: &CameraIntrinsics,
    ) -> Result<Vec<FrameRetrievalResult>, RetrievalError> {
        if poses.is_empty() {
            return Err(RetrievalError::EmptyWindow);
        }

        let config = self.config_for_store(store);
        let candidates = self.bounding_frustum_candidates(store, poses, intrinsics, &config);
        Ok(poses
            .iter()
            .map(|pose| {
                self.retrieve_from_candidates(
                    store,
                    Some(&candidates),
                    pose,
                    intrinsics,
                    Some(pose.timestamp),
                    &config,
                )
            })
            .collect())
    }

    fn bounding_frustum_candidates(
        &self,
        store: &MosaicMemoryStore,
        poses: &[CameraPose],
        intrinsics: &CameraIntrinsics,
        config: &RetrievalConfig,
    ) -> Vec<usize> {
        if store.patches.is_empty() {
            return Vec::new();
        }

        let centers: Vec<_> = store.patches.iter().map(|patch| patch.center).collect();
        let mut unique = BTreeSet::new();
        for pose in poses {
            for idx in frustum_cull(
                &centers,
                intrinsics,
                pose,
                config.near_clip,
                config.far_clip,
            ) {
                unique.insert(idx);
            }
        }
        unique.into_iter().collect()
    }

    fn retrieve_from_candidates(
        &self,
        store: &MosaicMemoryStore,
        candidate_indices: Option<&[usize]>,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
        query_time: Option<f64>,
        config: &RetrievalConfig,
    ) -> FrameRetrievalResult {
        let owned_candidates;
        let indices = if let Some(indices) = candidate_indices {
            indices
        } else {
            owned_candidates = self.bounding_frustum_candidates(
                store,
                std::slice::from_ref(target_pose),
                intrinsics,
                config,
            );
            &owned_candidates
        };

        let mut patches: Vec<RetrievedPatch> = indices
            .iter()
            .filter_map(|&idx| {
                let patch = store.patches.get(idx)?;
                let cam_point = target_pose.transform_point(&patch.center);
                if cam_point.z < config.near_clip || cam_point.z > config.far_clip {
                    return None;
                }

                let footprint = project_patch_footprint(patch, target_pose, intrinsics);
                let target_position = if footprint.is_empty() {
                    let pixel = intrinsics.project(&cam_point)?;
                    if !intrinsics.is_in_bounds(&pixel) {
                        return None;
                    }
                    pixel
                } else {
                    average_point(&footprint)
                };

                let center_dist =
                    ((target_position.x - intrinsics.cx as f32) / intrinsics.width as f32).powi(2)
                        + ((target_position.y - intrinsics.cy as f32) / intrinsics.height as f32)
                            .powi(2);
                let depth_score = 1.0 / (1.0 + cam_point.z * 0.1);
                let footprint_score =
                    (footprint.len().max(1) as f32 / patch.token_count().max(1) as f32).max(0.25);
                let mut visibility_score =
                    (1.0 - center_dist.sqrt()).max(0.0) * depth_score * footprint_score;

                if config.temporal_decay_half_life > 0.0
                    && let Some(qt) = query_time
                {
                    let age = (qt - patch.source_timestamp).max(0.0);
                    let decay = (0.5_f64).powf(age / config.temporal_decay_half_life) as f32;
                    visibility_score *= decay;
                }

                if visibility_score < config.min_visibility {
                    return None;
                }

                Some(RetrievedPatch {
                    patch: patch.clone(),
                    target_position,
                    projected_footprint: if footprint.is_empty() {
                        vec![target_position]
                    } else {
                        footprint
                    },
                    target_depth: cam_point.z,
                    visibility_score,
                })
            })
            .collect();

        patches.sort_by(|a, b| {
            b.visibility_score
                .partial_cmp(&a.visibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if config.diversity_radius > 0.0 && patches.len() > 1 {
            patches =
                self.diversity_select(patches, config.diversity_radius, config.diversity_penalty);
        }

        patches.truncate(config.top_k);

        if self.depth_sort {
            patches.sort_by(|a, b| {
                a.target_depth
                    .partial_cmp(&b.target_depth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let coverage = self.compute_coverage(&patches, intrinsics);

        FrameRetrievalResult {
            target_pose: target_pose.clone(),
            patches,
            coverage_mask: coverage,
            width: intrinsics.width,
            height: intrinsics.height,
        }
    }

    /// Diversity-aware greedy selection.
    fn diversity_select(
        &self,
        mut candidates: Vec<RetrievedPatch>,
        diversity_radius: f32,
        diversity_penalty: f32,
    ) -> Vec<RetrievedPatch> {
        candidates.sort_by(|a, b| {
            b.visibility_score
                .partial_cmp(&a.visibility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let max_patches = candidates.len();
        let mut selected: Vec<RetrievedPatch> = Vec::with_capacity(max_patches);
        let mut scores: Vec<f32> = candidates.iter().map(|p| p.visibility_score).collect();
        let mut used = vec![false; candidates.len()];
        let r2 = diversity_radius * diversity_radius;

        for _ in 0..max_patches {
            let best_idx = scores
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);

            let Some(idx) = best_idx else { break };
            if scores[idx] < self.min_visibility {
                break;
            }

            used[idx] = true;
            let selected_pos = candidates[idx].target_position;
            let mut patch = candidates[idx].clone();
            patch.visibility_score = scores[idx];
            selected.push(patch);

            for (j, candidate) in candidates.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let dx = candidate.target_position.x - selected_pos.x;
                let dy = candidate.target_position.y - selected_pos.y;
                if dx * dx + dy * dy < r2 {
                    scores[j] *= diversity_penalty;
                }
            }
        }

        selected
    }

    /// Compute a coverage mask indicating which regions have memory patches.
    /// Uses the projected patch footprint rather than just the center point.
    fn compute_coverage(
        &self,
        patches: &[RetrievedPatch],
        intrinsics: &CameraIntrinsics,
    ) -> Vec<Vec<bool>> {
        let grid_h = (intrinsics.height / 8).max(1) as usize;
        let grid_w = (intrinsics.width / 8).max(1) as usize;
        let mut coverage = vec![vec![false; grid_w]; grid_h];

        for patch in patches {
            let footprint = if patch.projected_footprint.is_empty() {
                vec![patch.target_position]
            } else {
                patch.projected_footprint.clone()
            };

            let mut cells = Vec::new();
            for point in footprint {
                let gx = ((point.x / intrinsics.width as f32) * grid_w as f32).floor() as isize;
                let gy = ((point.y / intrinsics.height as f32) * grid_h as f32).floor() as isize;
                if gx >= 0 && gy >= 0 && (gx as usize) < grid_w && (gy as usize) < grid_h {
                    cells.push((gx as usize, gy as usize));
                }
            }

            if cells.is_empty() {
                continue;
            }

            let min_x = cells.iter().map(|(x, _)| *x).min().unwrap_or(0);
            let max_x = cells.iter().map(|(x, _)| *x).max().unwrap_or(min_x);
            let min_y = cells.iter().map(|(_, y)| *y).min().unwrap_or(0);
            let max_y = cells.iter().map(|(_, y)| *y).max().unwrap_or(min_y);

            for row in coverage.iter_mut().take(max_y + 1).skip(min_y) {
                for cell in row.iter_mut().take(max_x + 1).skip(min_x) {
                    *cell = true;
                }
            }
        }

        coverage
    }
}

impl traits::MemoryRetriever for MemoryRetriever {
    fn retrieve_for_frame(
        &self,
        store: &MosaicMemoryStore,
        target_pose: &CameraPose,
        intrinsics: &CameraIntrinsics,
        config: &RetrievalConfig,
    ) -> Result<FrameRetrievalResult, RetrievalError> {
        Ok(self.retrieve_from_candidates(
            store,
            None,
            target_pose,
            intrinsics,
            Some(target_pose.timestamp),
            config,
        ))
    }
}

impl Default for MemoryRetriever {
    fn default() -> Self {
        Self::new()
    }
}

fn project_patch_footprint(
    patch: &Patch3D,
    target_pose: &CameraPose,
    intrinsics: &CameraIntrinsics,
) -> Vec<Point2<f32>> {
    let token_coords = if patch.token_coords.is_empty() {
        vec![(
            patch.source_rect[0] + patch.source_rect[2] / 2.0,
            patch.source_rect[1] + patch.source_rect[3] / 2.0,
        )]
    } else {
        patch.token_coords.clone()
    };

    let depth_tile = patch
        .depth_tile
        .clone()
        .unwrap_or_else(|| vec![patch.source_depth; token_coords.len()]);
    let c2w = patch.source_pose.camera_to_world();
    let mut footprint = Vec::with_capacity(token_coords.len());

    for (idx, (u, v)) in token_coords.into_iter().enumerate() {
        let depth = depth_tile.get(idx).copied().unwrap_or(patch.source_depth);
        if !depth.is_finite() || depth <= 0.0 {
            continue;
        }

        let source_cam = patch.source_intrinsics.unproject(&Point2::new(u, v), depth);
        let world = c2w.transform_point(&source_cam);
        let target_cam = target_pose.transform_point(&world);
        if target_cam.z <= 0.0 {
            continue;
        }
        if let Some(pixel) = intrinsics.project(&target_cam)
            && intrinsics.is_in_bounds(&pixel)
        {
            footprint.push(pixel);
        }
    }

    footprint
}

fn average_point(points: &[Point2<f32>]) -> Point2<f32> {
    let len = points.len().max(1) as f32;
    let sum_x: f32 = points.iter().map(|point| point.x).sum();
    let sum_y: f32 = points.iter().map(|point| point.y).sum();
    Point2::new(sum_x / len, sum_y / len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::store::MemoryConfig;
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    #[test]
    fn test_retriever_empty_store() {
        let store = MosaicMemoryStore::new(MemoryConfig::default());
        let retriever = MemoryRetriever::new();
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let pose = CameraPose::identity(0.0);

        let frame = retriever.retrieve(&store, &pose, &intrinsics);
        assert!(frame.patches.is_empty());
    }

    #[test]
    fn test_retriever_with_patches() {
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
        store.insert_keyframe(0, 0.0, &latents, 4, 4, 4, &depth_map, &intrinsics, &pose);

        let retriever = MemoryRetriever::new();
        let frame = retriever.retrieve(&store, &pose, &intrinsics);
        assert!(!frame.patches.is_empty());
        assert!(frame.has_coverage());
    }

    #[test]
    fn test_diversity_select_spreads_patches() {
        let retriever = MemoryRetriever::with_diversity(20.0, 0.1);

        let make_patch = |id, x, y, score| RetrievedPatch {
            patch: Patch3D {
                id,
                center: Point3::new(0.0, 0.0, 5.0),
                source_pose: CameraPose::identity(0.0),
                source_frame: 0,
                source_timestamp: 0.0,
                source_depth: 5.0,
                source_rect: [0.0, 0.0, 16.0, 16.0],
                latent: vec![0.5; 4],
                latent_height: 1,
                latent_width: 1,
                token_coords: vec![(8.0, 8.0)],
                depth_tile: Some(vec![5.0]),
                source_intrinsics: CameraIntrinsics::default(),
                normal_estimate: None,
                latent_shape: (4, 1, 1),
            },
            target_position: Point2::new(x, y),
            projected_footprint: vec![Point2::new(x, y)],
            target_depth: 5.0,
            visibility_score: score,
        };

        let candidates = vec![
            make_patch(0, 50.0, 50.0, 0.9),
            make_patch(1, 52.0, 51.0, 0.85),
            make_patch(2, 53.0, 49.0, 0.80),
            make_patch(3, 90.0, 90.0, 0.7),
        ];

        let selected = retriever.diversity_select(candidates, 20.0, 0.1);
        assert_eq!(selected[0].patch.id, 0);

        let id_order: Vec<u64> = selected.iter().map(|patch| patch.patch.id).collect();
        let pos_3 = id_order.iter().position(|&id| id == 3).unwrap();
        let pos_1 = id_order.iter().position(|&id| id == 1).unwrap();
        assert!(
            pos_3 < pos_1,
            "Diverse patch should win earlier: {:?}",
            id_order
        );
    }

    #[test]
    fn test_window_retrieval_changes_with_pose() {
        let mut store = MosaicMemoryStore::new(MemoryConfig {
            max_patches: 100,
            top_k: 8,
            ..Default::default()
        });
        let intrinsics = CameraIntrinsics::new(100.0, 100.0, 50.0, 50.0, 100, 100);
        let depth_map = vec![vec![5.0f32; 16]; 16];
        let latents = vec![0.5f32; 4 * 4 * 4];
        let pose_a = CameraPose::identity(0.0);
        let pose_b = CameraPose::from_translation_rotation(
            1.0,
            Vector3::new(2.0, 0.0, 0.0),
            UnitQuaternion::from_euler_angles(0.0, 0.5, 0.0),
        );
        store.insert_keyframe(0, 0.0, &latents, 4, 4, 4, &depth_map, &intrinsics, &pose_a);
        store.insert_keyframe(1, 1.0, &latents, 4, 4, 4, &depth_map, &intrinsics, &pose_b);

        let retriever = MemoryRetriever::new();
        let results = retriever
            .retrieve_window(&store, &[pose_a.clone(), pose_b.clone()], &intrinsics)
            .unwrap();

        assert_eq!(results.len(), 2);
        let first_ids: Vec<u64> = results[0]
            .patches
            .iter()
            .map(|patch| patch.patch.id)
            .collect();
        let second_ids: Vec<u64> = results[1]
            .patches
            .iter()
            .map(|patch| patch.patch.id)
            .collect();
        assert_ne!(first_ids, second_ids);
    }
}
