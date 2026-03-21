use crate::memory::store::{MosaicMemoryStore, Patch3D};
use nalgebra::{Point3, Vector3};

/// Memory manipulation operations for scene editing.
///
/// Splices two memory stores horizontally (side-by-side scenes).
/// The `offset` determines the spatial separation between the two scenes.
pub fn splice_horizontal(
    store_a: &MosaicMemoryStore,
    store_b: &MosaicMemoryStore,
    offset: f32,
) -> Vec<Patch3D> {
    let mut result = Vec::new();

    // Keep store_a patches as-is (left side)
    result.extend(store_a.patches.iter().cloned());

    // Shift store_b patches to the right
    let shift = Vector3::new(offset, 0.0, 0.0);
    for patch in &store_b.patches {
        let mut shifted = patch.clone();
        shifted.center = Point3::from(patch.center.coords + shift);
        result.push(shifted);
    }

    result
}

/// Flip a memory store vertically (Inception-style gravity inversion).
pub fn flip_vertical(store: &MosaicMemoryStore) -> Vec<Patch3D> {
    store
        .patches
        .iter()
        .map(|patch| {
            let mut flipped = patch.clone();
            flipped.center.y = -flipped.center.y;
            flipped
        })
        .collect()
}

/// Erase patches within a 3D region (sphere).
pub fn erase_region(store: &MosaicMemoryStore, center: &Point3<f32>, radius: f32) -> Vec<Patch3D> {
    let r2 = radius * radius;
    store
        .patches
        .iter()
        .filter(|p| (p.center - center).norm_squared() > r2)
        .cloned()
        .collect()
}

/// Translate all patches in a memory store by a 3D offset.
pub fn translate(store: &MosaicMemoryStore, offset: &Vector3<f32>) -> Vec<Patch3D> {
    store
        .patches
        .iter()
        .map(|patch| {
            let mut translated = patch.clone();
            translated.center = Point3::from(patch.center.coords + offset);
            translated
        })
        .collect()
}

/// Scale patches around a center point.
pub fn scale(store: &MosaicMemoryStore, center: &Point3<f32>, factor: f32) -> Vec<Patch3D> {
    store
        .patches
        .iter()
        .map(|patch| {
            let mut scaled = patch.clone();
            let relative = patch.center - center;
            scaled.center = center + relative * factor;
            scaled
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{CameraIntrinsics, CameraPose};
    use crate::memory::store::MemoryConfig;

    fn make_store() -> MosaicMemoryStore {
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
        store
    }

    #[test]
    fn test_splice_horizontal() {
        let store_a = make_store();
        let store_b = make_store();
        let result = splice_horizontal(&store_a, &store_b, 10.0);
        assert_eq!(result.len(), store_a.num_patches() + store_b.num_patches());
    }

    #[test]
    fn test_flip_vertical() {
        let store = make_store();
        let flipped = flip_vertical(&store);
        for (orig, flip) in store.patches.iter().zip(flipped.iter()) {
            assert!((orig.center.y + flip.center.y).abs() < 1e-5);
        }
    }

    #[test]
    fn test_erase_region() {
        let store = make_store();
        let initial = store.num_patches();
        let erased = erase_region(&store, &Point3::origin(), 1000.0);
        assert!(erased.len() < initial || initial == 0);
    }
}
