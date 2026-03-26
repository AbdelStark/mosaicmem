use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// Camera pose represented as an SE(3) rigid body transformation.
/// Transforms points from world coordinates to camera coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPose {
    /// Timestamp or frame index for this pose.
    pub timestamp: f64,
    /// SE(3) transformation: world-to-camera.
    #[serde(with = "isometry_serde")]
    pub world_to_camera: Isometry3<f32>,
}

impl CameraPose {
    /// Create a new camera pose from a world-to-camera isometry.
    pub fn new(timestamp: f64, world_to_camera: Isometry3<f32>) -> Self {
        Self {
            timestamp,
            world_to_camera,
        }
    }

    /// Create an identity pose (camera at world origin).
    pub fn identity(timestamp: f64) -> Self {
        Self {
            timestamp,
            world_to_camera: Isometry3::identity(),
        }
    }

    /// Create a pose from translation and quaternion rotation.
    pub fn from_translation_rotation(
        timestamp: f64,
        translation: Vector3<f32>,
        rotation: UnitQuaternion<f32>,
    ) -> Self {
        let iso = Isometry3::from_parts(Translation3::from(translation), rotation);
        Self::new(timestamp, iso)
    }

    /// Create a pose looking at a target point from a given position.
    pub fn look_at(
        timestamp: f64,
        eye: &Point3<f32>,
        target: &Point3<f32>,
        up: &Vector3<f32>,
    ) -> Self {
        // The projection stack treats +Z as "in front of the camera", so `look_at`
        // must use the left-handed constructor to keep visible targets at positive depth.
        Self::new(timestamp, Isometry3::look_at_lh(eye, target, up))
    }

    /// Get the camera-to-world transformation.
    pub fn camera_to_world(&self) -> Isometry3<f32> {
        self.world_to_camera.inverse()
    }

    /// Get the camera position in world coordinates.
    pub fn position(&self) -> Point3<f32> {
        self.camera_to_world().translation.vector.into()
    }

    /// Get the forward direction (+Z in camera space) in world coordinates.
    pub fn forward(&self) -> Vector3<f32> {
        let c2w = self.camera_to_world();
        c2w.rotation * Vector3::new(0.0, 0.0, 1.0)
    }

    /// Get the up direction (positive Y in camera space) in world coordinates.
    pub fn up(&self) -> Vector3<f32> {
        let c2w = self.camera_to_world();
        c2w.rotation * Vector3::new(0.0, 1.0, 0.0)
    }

    /// Get the right direction (positive X in camera space) in world coordinates.
    pub fn right(&self) -> Vector3<f32> {
        let c2w = self.camera_to_world();
        c2w.rotation * Vector3::new(1.0, 0.0, 0.0)
    }

    /// Transform a world point into camera coordinates.
    pub fn transform_point(&self, world_point: &Point3<f32>) -> Point3<f32> {
        self.world_to_camera.transform_point(world_point)
    }

    /// Compute relative pose from self to other (other * self.inverse()).
    pub fn relative_to(&self, other: &CameraPose) -> Isometry3<f32> {
        other.world_to_camera * self.world_to_camera.inverse()
    }

    /// Compute translation distance to another pose.
    pub fn translation_distance(&self, other: &CameraPose) -> f32 {
        let p1 = self.position();
        let p2 = other.position();
        (p1 - p2).norm()
    }

    /// Compute angular distance (in radians) to another pose.
    pub fn angular_distance(&self, other: &CameraPose) -> f32 {
        let rel = self.relative_to(other);
        rel.rotation.angle()
    }
}

mod isometry_serde {
    use nalgebra::{Isometry3, Translation3, UnitQuaternion};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct IsometryData {
        tx: f32,
        ty: f32,
        tz: f32,
        qx: f32,
        qy: f32,
        qz: f32,
        qw: f32,
    }

    pub fn serialize<S>(iso: &Isometry3<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let t = iso.translation;
        let q = iso.rotation;
        let data = IsometryData {
            tx: t.x,
            ty: t.y,
            tz: t.z,
            qx: q.i,
            qy: q.j,
            qz: q.k,
            qw: q.w,
        };
        data.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Isometry3<f32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = IsometryData::deserialize(deserializer)?;
        let translation = Translation3::new(data.tx, data.ty, data.tz);
        let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            data.qw, data.qx, data.qy, data.qz,
        ));
        Ok(Isometry3::from_parts(translation, rotation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_pose() {
        let pose = CameraPose::identity(0.0);
        let pos = pose.position();
        assert!((pos.x.abs() + pos.y.abs() + pos.z.abs()) < 1e-5);
    }

    #[test]
    fn test_transform_roundtrip() {
        let pose = CameraPose::from_translation_rotation(
            0.0,
            Vector3::new(1.0, 2.0, 3.0),
            UnitQuaternion::identity(),
        );
        let world_pt = Point3::new(4.0, 5.0, 6.0);
        let cam_pt = pose.transform_point(&world_pt);
        let recovered = pose.camera_to_world().transform_point(&cam_pt);
        assert!((recovered - world_pt).norm() < 1e-5);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let pose = CameraPose::from_translation_rotation(
            1.5,
            Vector3::new(1.0, 2.0, 3.0),
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
        );
        let json = serde_json::to_string(&pose).unwrap();
        let recovered: CameraPose = serde_json::from_str(&json).unwrap();
        assert!((pose.position() - recovered.position()).norm() < 1e-5);
    }

    #[test]
    fn test_look_at_keeps_target_in_front() {
        let eye = Point3::new(0.0, 0.0, 0.0);
        let target = Point3::new(0.0, 0.0, 5.0);
        let pose = CameraPose::look_at(0.0, &eye, &target, &Vector3::y());

        let cam_target = pose.transform_point(&target);
        assert!(cam_target.z > 0.0);
        assert!(cam_target.x.abs() < 1e-5);
        assert!(cam_target.y.abs() < 1e-5);
        assert!((pose.forward() - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-5);
    }
}
