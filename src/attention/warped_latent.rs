use crate::camera::{CameraIntrinsics, CameraPose};
use nalgebra::Matrix3;

/// Warped Latent: direct feature-space warping via geometric correspondence.
///
/// Computes a homography/affine transform between source and target views,
/// then warps the latent feature grid using bilinear interpolation.

/// Compute the homography between two camera views assuming a planar scene
/// at a given depth.
pub fn compute_planar_homography(
    source_pose: &CameraPose,
    target_pose: &CameraPose,
    intrinsics: &CameraIntrinsics,
    plane_depth: f32,
) -> Matrix3<f32> {
    let k = intrinsics.matrix();
    let k_inv = intrinsics.inverse_matrix();

    // Relative transformation: target * source^{-1}
    let relative = target_pose.world_to_camera * source_pose.world_to_camera.inverse();

    let r = relative.rotation.to_rotation_matrix();
    let t = relative.translation.vector;
    let n = nalgebra::Vector3::new(0.0, 0.0, 1.0); // Plane normal in source camera

    // H = K * (R + t * n^T / d) * K^{-1}
    let h = r.matrix() + t * n.transpose() / plane_depth;
    k * h * k_inv
}

/// Warp a latent feature grid from source view to target view using bilinear sampling.
///
/// # Arguments
/// * `latent` - Source latent features [H, W, C]
/// * `height` - Latent grid height
/// * `width` - Latent grid width
/// * `channels` - Number of channels
/// * `homography` - 3x3 homography matrix (target-to-source mapping)
///
/// # Returns
/// Warped latent features [H, W, C].
pub fn warp_latent(
    latent: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    homography: &Matrix3<f32>,
) -> Vec<f32> {
    let mut output = vec![0.0f32; height * width * channels];

    // For each target pixel, find the corresponding source pixel
    for ty in 0..height {
        for tx in 0..width {
            // Normalize to [0,1] range, then to pixel coords
            let target_pt = nalgebra::Vector3::new(tx as f32, ty as f32, 1.0);
            let source_pt = homography * target_pt;

            if source_pt.z.abs() < 1e-8 {
                continue;
            }

            let sx = source_pt.x / source_pt.z;
            let sy = source_pt.y / source_pt.z;

            // Bilinear interpolation
            if sx >= 0.0 && sx < (width - 1) as f32 && sy >= 0.0 && sy < (height - 1) as f32 {
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let w00 = (1.0 - fx) * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w10 = fx * (1.0 - fy);
                let w11 = fx * fy;

                let dst_offset = (ty * width + tx) * channels;
                for c in 0..channels {
                    let v00 = latent[(y0 * width + x0) * channels + c];
                    let v01 = latent[(y1 * width + x0) * channels + c];
                    let v10 = latent[(y0 * width + x1) * channels + c];
                    let v11 = latent[(y1 * width + x1) * channels + c];
                    output[dst_offset + c] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                }
            }
        }
    }

    output
}

/// High-level warp function: compute homography and apply warping.
pub fn warp_patch_latent(
    latent: &[f32],
    height: usize,
    width: usize,
    channels: usize,
    source_pose: &CameraPose,
    target_pose: &CameraPose,
    intrinsics: &CameraIntrinsics,
    depth: f32,
) -> Vec<f32> {
    let h = compute_planar_homography(source_pose, target_pose, intrinsics, depth);
    warp_latent(latent, height, width, channels, &h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_warp() {
        let h = 4;
        let w = 4;
        let c = 2;
        let latent: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();

        // Identity homography should preserve the latent (except borders)
        let identity = Matrix3::identity();
        let warped = warp_latent(&latent, h, w, c, &identity);

        // Interior points should match
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                for ch in 0..c {
                    let idx = (y * w + x) * c + ch;
                    assert!(
                        (warped[idx] - latent[idx]).abs() < 1e-5,
                        "Mismatch at ({}, {}, {})",
                        y,
                        x,
                        ch
                    );
                }
            }
        }
    }

    #[test]
    fn test_warp_output_size() {
        let h = 8;
        let w = 8;
        let c = 4;
        let latent = vec![1.0f32; h * w * c];
        let identity = Matrix3::identity();
        let warped = warp_latent(&latent, h, w, c, &identity);
        assert_eq!(warped.len(), h * w * c);
    }
}
