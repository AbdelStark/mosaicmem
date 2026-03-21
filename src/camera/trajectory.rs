use crate::camera::CameraPose;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A sequence of camera poses forming a trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraTrajectory {
    pub poses: Vec<CameraPose>,
}

impl CameraTrajectory {
    pub fn new(poses: Vec<CameraPose>) -> Self {
        Self { poses }
    }

    /// Load trajectory from a JSON file.
    pub fn load_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let trajectory: CameraTrajectory = serde_json::from_str(&data)?;
        Ok(trajectory)
    }

    /// Save trajectory to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Get the number of poses.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Check if the trajectory is empty.
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Get pose at a specific index.
    pub fn get(&self, index: usize) -> Option<&CameraPose> {
        self.poses.get(index)
    }

    /// Get poses for a time window.
    pub fn window(&self, start: usize, count: usize) -> &[CameraPose] {
        let end = (start + count).min(self.poses.len());
        &self.poses[start..end]
    }

    /// Select keyframes based on motion threshold.
    /// Returns indices of poses with sufficient translational or angular motion.
    pub fn select_keyframes(
        &self,
        translation_threshold: f32,
        angular_threshold: f32,
    ) -> Vec<usize> {
        if self.poses.is_empty() {
            return vec![];
        }
        let mut keyframes = vec![0];
        let mut last_kf = 0;

        for i in 1..self.poses.len() {
            let trans_dist = self.poses[last_kf].translation_distance(&self.poses[i]);
            let ang_dist = self.poses[last_kf].angular_distance(&self.poses[i]);

            if trans_dist >= translation_threshold || ang_dist >= angular_threshold {
                keyframes.push(i);
                last_kf = i;
            }
        }
        keyframes
    }

    /// Select every Kth frame as a keyframe.
    pub fn select_keyframes_uniform(&self, interval: usize) -> Vec<usize> {
        (0..self.poses.len()).step_by(interval.max(1)).collect()
    }

    /// Compute total path length (sum of inter-pose translations).
    pub fn path_length(&self) -> f32 {
        self.poses
            .windows(2)
            .map(|w| w[0].translation_distance(&w[1]))
            .sum()
    }

    /// Get the duration (last timestamp - first timestamp).
    pub fn duration(&self) -> f64 {
        match (self.poses.first(), self.poses.last()) {
            (Some(first), Some(last)) => last.timestamp - first.timestamp,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn make_trajectory() -> CameraTrajectory {
        let poses: Vec<CameraPose> = (0..10)
            .map(|i| {
                CameraPose::from_translation_rotation(
                    i as f64 * 0.1,
                    Vector3::new(i as f32 * 0.5, 0.0, 0.0),
                    UnitQuaternion::identity(),
                )
            })
            .collect();
        CameraTrajectory::new(poses)
    }

    #[test]
    fn test_trajectory_len() {
        let traj = make_trajectory();
        assert_eq!(traj.len(), 10);
    }

    #[test]
    fn test_uniform_keyframes() {
        let traj = make_trajectory();
        let kf = traj.select_keyframes_uniform(3);
        assert_eq!(kf, vec![0, 3, 6, 9]);
    }

    #[test]
    fn test_motion_keyframes() {
        let traj = make_trajectory();
        let kf = traj.select_keyframes(1.0, 0.5);
        // Each step is 0.5 in translation, so every 2 frames exceeds threshold of 1.0
        assert!(kf.len() >= 2);
        assert_eq!(kf[0], 0);
    }
}
