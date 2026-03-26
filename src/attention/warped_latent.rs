use crate::camera::{CameraIntrinsics, CameraPose};
use crate::memory::store::Patch3D;
use crate::tensor::{TensorError, TensorLayout, TensorView};
use nalgebra::Point2;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WarpError {
    #[error("patch geometry is invalid: {0}")]
    InvalidGeometry(String),
    #[error("warp grid shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
    #[error("unsupported tensor layout for warping: {0:?}")]
    UnsupportedLayout(TensorLayout),
    #[error("tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("tensor data is not contiguous in memory order")]
    NonContiguousTensor,
}

#[derive(Debug, Clone)]
pub struct WarpGrid {
    pub target_coords: Vec<(f32, f32)>,
    pub valid_mask: Vec<bool>,
    pub source_shape: (usize, usize),
}

impl WarpGrid {
    pub fn valid_ratio(&self) -> f32 {
        if self.valid_mask.is_empty() {
            return 0.0;
        }
        self.valid_mask.iter().filter(|&&valid| valid).count() as f32 / self.valid_mask.len() as f32
    }

    fn normalized_target_coords(&self) -> Vec<Option<(f32, f32)>> {
        let (height, width) = self.source_shape;
        let valid_points: Vec<_> = self
            .target_coords
            .iter()
            .zip(self.valid_mask.iter())
            .filter_map(|(&(u, v), &valid)| valid.then_some((u, v)))
            .collect();

        if valid_points.is_empty() {
            return vec![None; self.target_coords.len()];
        }

        let min_u = valid_points
            .iter()
            .map(|(u, _)| *u)
            .fold(f32::INFINITY, f32::min);
        let max_u = valid_points
            .iter()
            .map(|(u, _)| *u)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_v = valid_points
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::INFINITY, f32::min);
        let max_v = valid_points
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        let span_u = max_u - min_u;
        let span_v = max_v - min_v;

        self.target_coords
            .iter()
            .zip(self.valid_mask.iter())
            .enumerate()
            .map(|(idx, (&(u, v), &valid))| {
                if !valid {
                    return None;
                }

                let src_x = (idx % width) as f32;
                let src_y = (idx / width) as f32;
                let norm_x = if span_u.abs() <= 1e-6 {
                    src_x
                } else {
                    ((u - min_u) / span_u) * (width.saturating_sub(1) as f32)
                };
                let norm_y = if span_v.abs() <= 1e-6 {
                    src_y
                } else {
                    ((v - min_v) / span_v) * (height.saturating_sub(1) as f32)
                };

                Some((
                    norm_x.clamp(0.0, width.saturating_sub(1) as f32),
                    norm_y.clamp(0.0, height.saturating_sub(1) as f32),
                ))
            })
            .collect()
    }

    fn validate(&self) -> Result<(), WarpError> {
        let expected = self.source_shape.0 * self.source_shape.1;
        if self.target_coords.len() != expected {
            return Err(WarpError::ShapeMismatch {
                expected,
                actual: self.target_coords.len(),
            });
        }
        if self.valid_mask.len() != expected {
            return Err(WarpError::ShapeMismatch {
                expected,
                actual: self.valid_mask.len(),
            });
        }
        Ok(())
    }
}

pub trait WarpOperator {
    fn compute_warp_grid(
        &self,
        patch: &Patch3D,
        target_pose: &CameraPose,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<WarpGrid, WarpError>;

    fn apply_warp(
        &self,
        source_latent: &TensorView,
        grid: &WarpGrid,
    ) -> Result<(TensorView, TensorView), WarpError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DenseWarpOperator;

impl DenseWarpOperator {
    fn source_hwc(
        source_latent: &TensorView,
    ) -> Result<(usize, usize, usize, Vec<f32>, TensorLayout), WarpError> {
        match source_latent.layout() {
            TensorLayout::Flat(shape) if shape.len() == 3 => {
                let data = source_latent
                    .data()
                    .as_slice_memory_order()
                    .ok_or(WarpError::NonContiguousTensor)?
                    .to_vec();
                Ok((
                    shape[0],
                    shape[1],
                    shape[2],
                    data,
                    source_latent.layout().clone(),
                ))
            }
            TensorLayout::CHW => {
                let shape = source_latent.shape();
                let channels = shape[0];
                let height = shape[1];
                let width = shape[2];
                let data = source_latent
                    .data()
                    .as_slice_memory_order()
                    .ok_or(WarpError::NonContiguousTensor)?;
                let mut hwc = vec![0.0f32; height * width * channels];
                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            let src_idx = ((c * height + y) * width) + x;
                            let dst_idx = ((y * width + x) * channels) + c;
                            hwc[dst_idx] = data[src_idx];
                        }
                    }
                }
                Ok((height, width, channels, hwc, source_latent.layout().clone()))
            }
            layout => Err(WarpError::UnsupportedLayout(layout.clone())),
        }
    }

    fn restore_layout(
        layout: &TensorLayout,
        height: usize,
        width: usize,
        channels: usize,
        hwc: Vec<f32>,
    ) -> Result<TensorView, WarpError> {
        match layout {
            TensorLayout::Flat(_) => Ok(TensorView::from_shape_vec(
                &[height, width, channels],
                hwc,
                TensorLayout::Flat(vec![height, width, channels]),
            )?),
            TensorLayout::CHW => {
                let mut chw = vec![0.0f32; height * width * channels];
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let src_idx = ((y * width + x) * channels) + c;
                            let dst_idx = ((c * height + y) * width) + x;
                            chw[dst_idx] = hwc[src_idx];
                        }
                    }
                }
                Ok(TensorView::from_shape_vec(
                    &[channels, height, width],
                    chw,
                    TensorLayout::CHW,
                )?)
            }
            layout => Err(WarpError::UnsupportedLayout(layout.clone())),
        }
    }
}

impl WarpOperator for DenseWarpOperator {
    fn compute_warp_grid(
        &self,
        patch: &Patch3D,
        target_pose: &CameraPose,
        target_intrinsics: &CameraIntrinsics,
    ) -> Result<WarpGrid, WarpError> {
        patch
            .validate_geometry()
            .map_err(|err| WarpError::InvalidGeometry(err.to_string()))?;

        let token_count = patch.token_count();
        let token_coords = if patch.token_coords.is_empty() {
            return Err(WarpError::InvalidGeometry(
                "dense warp requires per-token coordinates".to_string(),
            ));
        } else {
            patch.token_coords.clone()
        };
        let depth_tile = patch
            .depth_tile
            .clone()
            .unwrap_or_else(|| vec![patch.source_depth; token_count]);
        let c2w = patch.source_pose.camera_to_world();

        let mut target_coords = Vec::with_capacity(token_count);
        let mut valid_mask = Vec::with_capacity(token_count);

        for (idx, (u, v)) in token_coords.into_iter().enumerate() {
            let depth = depth_tile.get(idx).copied().unwrap_or(patch.source_depth);
            if !depth.is_finite() || depth <= 0.0 {
                target_coords.push((f32::NAN, f32::NAN));
                valid_mask.push(false);
                continue;
            }

            let source_cam = patch.source_intrinsics.unproject(&Point2::new(u, v), depth);
            let world = c2w.transform_point(&source_cam);
            let target_cam = target_pose.transform_point(&world);
            if target_cam.z <= 0.0 {
                target_coords.push((f32::NAN, f32::NAN));
                valid_mask.push(false);
                continue;
            }

            let pixel = target_intrinsics
                .project(&target_cam)
                .unwrap_or_else(|| Point2::new(f32::NAN, f32::NAN));
            target_coords.push((pixel.x, pixel.y));
            valid_mask.push(true);
        }

        let grid = WarpGrid {
            target_coords,
            valid_mask,
            source_shape: (patch.latent_height, patch.latent_width),
        };
        grid.validate()?;
        Ok(grid)
    }

    fn apply_warp(
        &self,
        source_latent: &TensorView,
        grid: &WarpGrid,
    ) -> Result<(TensorView, TensorView), WarpError> {
        grid.validate()?;
        let (height, width, channels, source_hwc, source_layout) = Self::source_hwc(source_latent)?;
        if (height, width) != grid.source_shape {
            return Err(WarpError::ShapeMismatch {
                expected: grid.source_shape.0 * grid.source_shape.1,
                actual: height * width,
            });
        }

        let normalized = grid.normalized_target_coords();
        let mut output = vec![0.0f32; height * width * channels];
        let mut weights = vec![0.0f32; height * width];
        let mut output_valid = vec![false; height * width];

        for (idx, target) in normalized.into_iter().enumerate() {
            let Some((gx, gy)) = target else {
                continue;
            };
            let x0 = gx.floor() as usize;
            let y0 = gy.floor() as usize;
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let fx = gx - x0 as f32;
            let fy = gy - y0 as f32;

            let bilinear_weights = [
                (x0, y0, (1.0 - fx) * (1.0 - fy)),
                (x0, y1, (1.0 - fx) * fy),
                (x1, y0, fx * (1.0 - fy)),
                (x1, y1, fx * fy),
            ];

            let src_offset = idx * channels;
            for (x, y, weight) in bilinear_weights {
                if weight <= 0.0 {
                    continue;
                }

                let dst_token = y * width + x;
                let dst_offset = dst_token * channels;
                for c in 0..channels {
                    output[dst_offset + c] += source_hwc[src_offset + c] * weight;
                }
                weights[dst_token] += weight;
                output_valid[dst_token] = true;
            }
        }

        for (token_idx, weight) in weights.iter().copied().enumerate() {
            if weight <= 1e-6 {
                continue;
            }
            let dst_offset = token_idx * channels;
            for c in 0..channels {
                output[dst_offset + c] /= weight;
            }
        }

        let mask_values: Vec<f32> = output_valid
            .iter()
            .map(|&valid| if valid { 1.0 } else { 0.0 })
            .collect();

        let warped = Self::restore_layout(&source_layout, height, width, channels, output)?;
        let valid_mask =
            TensorView::from_shape_vec(&[height, width], mask_values, TensorLayout::HW)?;
        Ok((warped, valid_mask))
    }
}

pub fn warp_latent(
    source_latent: &TensorView,
    grid: &WarpGrid,
) -> Result<(TensorView, TensorView), WarpError> {
    DenseWarpOperator.apply_warp(source_latent, grid)
}

pub fn warp_patch_latent(
    patch: &Patch3D,
    target_pose: &CameraPose,
    target_intrinsics: &CameraIntrinsics,
) -> Result<(TensorView, TensorView), WarpError> {
    let channels = patch.channels();
    let source_latent = TensorView::from_shape_vec(
        &[patch.latent_height, patch.latent_width, channels],
        patch.latent.clone(),
        TensorLayout::Flat(vec![patch.latent_height, patch.latent_width, channels]),
    )?;
    let operator = DenseWarpOperator;
    let grid = operator.compute_warp_grid(patch, target_pose, target_intrinsics)?;
    operator.apply_warp(&source_latent, &grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    fn test_patch() -> Patch3D {
        Patch3D {
            id: 0,
            center: Point3::new(0.0, 0.0, 5.0),
            source_pose: CameraPose::identity(0.0),
            source_frame: 0,
            source_timestamp: 0.0,
            source_depth: 5.0,
            source_rect: [8.0, 8.0, 16.0, 16.0],
            latent: (0..16).map(|value| value as f32).collect(),
            latent_height: 4,
            latent_width: 4,
            token_coords: vec![
                (8.0, 8.0),
                (12.0, 8.0),
                (16.0, 8.0),
                (20.0, 8.0),
                (8.0, 12.0),
                (12.0, 12.0),
                (16.0, 12.0),
                (20.0, 12.0),
                (8.0, 16.0),
                (12.0, 16.0),
                (16.0, 16.0),
                (20.0, 16.0),
                (8.0, 20.0),
                (12.0, 20.0),
                (16.0, 20.0),
                (20.0, 20.0),
            ],
            depth_tile: Some(vec![5.0; 16]),
            source_intrinsics: CameraIntrinsics::new(100.0, 100.0, 32.0, 32.0, 64, 64),
            normal_estimate: Some(Vector3::z()),
            latent_shape: (1, 4, 4),
        }
    }

    #[test]
    fn test_warp_grid_valid_ratio() {
        let grid = WarpGrid {
            target_coords: vec![(0.0, 0.0), (1.0, 1.0)],
            valid_mask: vec![true, false],
            source_shape: (1, 2),
        };
        assert!((grid.valid_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_identity_dense_warp() {
        let operator = DenseWarpOperator;
        let patch = test_patch();
        let grid = operator
            .compute_warp_grid(&patch, &CameraPose::identity(0.0), &patch.source_intrinsics)
            .unwrap();
        let source = TensorView::from_shape_vec(
            &[4, 4, 1],
            patch.latent.clone(),
            TensorLayout::Flat(vec![4, 4, 1]),
        )
        .unwrap();
        let (warped, mask) = operator.apply_warp(&source, &grid).unwrap();
        for (warped_value, source_value) in warped
            .data()
            .as_slice_memory_order()
            .unwrap()
            .iter()
            .zip(source.data().as_slice_memory_order().unwrap().iter())
        {
            assert!((warped_value - source_value).abs() < 1e-5);
        }
        assert!(mask.data().iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }
}
