use crate::camera::CameraPose;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during trajectory I/O.
#[derive(Error, Debug)]
pub enum TrajectoryError {
    #[error("Failed to read trajectory from {path}: {source}")]
    ReadFailed {
        path: String,
        source: std::io::Error,
    },
    #[error("Failed to write trajectory to {path}: {source}")]
    WriteFailed {
        path: String,
        source: std::io::Error,
    },
    #[error("Failed to parse trajectory JSON from {path}: {source}")]
    ParseFailed {
        path: String,
        source: serde_json::Error,
    },
    #[error("Failed to serialize trajectory: {0}")]
    SerializeFailed(#[from] serde_json::Error),
}

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
    pub fn load_json(path: &Path) -> Result<Self, TrajectoryError> {
        let data = std::fs::read_to_string(path).map_err(|e| TrajectoryError::ReadFailed {
            path: path.display().to_string(),
            source: e,
        })?;
        let trajectory: CameraTrajectory =
            serde_json::from_str(&data).map_err(|e| TrajectoryError::ParseFailed {
                path: path.display().to_string(),
                source: e,
            })?;
        Ok(trajectory)
    }

    /// Save trajectory to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), TrajectoryError> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data).map_err(|e| TrajectoryError::WriteFailed {
            path: path.display().to_string(),
            source: e,
        })?;
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
    /// Returns an empty slice if `start` is beyond the trajectory length.
    pub fn window(&self, start: usize, count: usize) -> &[CameraPose] {
        let start = start.min(self.poses.len());
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

    #[test]
    fn test_save_load_json_roundtrip() {
        let traj = make_trajectory();
        let dir = std::env::temp_dir().join("mosaicmem_test_trajectory");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trajectory.json");

        traj.save_json(&path).unwrap();
        let loaded = CameraTrajectory::load_json(&path).unwrap();
        assert_eq!(loaded.len(), traj.len());
        assert!((loaded.duration() - traj.duration()).abs() < 1e-10);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_json_nonexistent_file() {
        let result = CameraTrajectory::load_json(std::path::Path::new("/nonexistent/path.json"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent/path.json"),
            "Error should contain the file path: {}",
            err
        );
    }

    #[test]
    fn test_empty_trajectory() {
        let traj = CameraTrajectory::new(vec![]);
        assert!(traj.is_empty());
        assert_eq!(traj.len(), 0);
        assert_eq!(traj.duration(), 0.0);
        assert_eq!(traj.path_length(), 0.0);
        assert_eq!(traj.select_keyframes(1.0, 1.0), Vec::<usize>::new());
        assert_eq!(traj.select_keyframes_uniform(1), Vec::<usize>::new());
        assert!(traj.get(0).is_none());
        assert!(traj.window(0, 10).is_empty());
    }
}
