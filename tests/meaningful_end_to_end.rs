use mosaicmem_rs::camera::CameraPose;
use mosaicmem_rs::diffusion::backbone::SyntheticBackbone;
use mosaicmem_rs::diffusion::scheduler::DDPMScheduler;
use mosaicmem_rs::diffusion::vae::SyntheticVAE;
use mosaicmem_rs::geometry::depth::SyntheticDepthEstimator;
use mosaicmem_rs::pipeline::config::PipelineConfig;
use mosaicmem_rs::pipeline::inference::InferencePipeline;

fn structured_frame(width: usize, height: usize) -> Vec<f32> {
    let plane = width * height;
    let mut frame = vec![0.0f32; 3 * plane];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let red = if x < width / 2 && y < (height * 3) / 4 {
                0.95
            } else {
                0.05
            };
            let green = if x >= width / 2 { 0.80 } else { 0.10 };
            let blue = if y >= height / 2 && x.abs_diff(y) < width / 4 {
                0.85
            } else {
                0.05
            };

            frame[idx] = red;
            frame[plane + idx] = green;
            frame[2 * plane + idx] = blue;
        }
    }

    frame
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        / a.len().max(1) as f32
}

#[test]
fn test_memory_conditioned_revisit_is_closer_to_observed_scene() {
    let config = PipelineConfig {
        num_inference_steps: 3,
        width: 32,
        height: 32,
        window_size: 1,
        temporal_downsample: 1,
        temporal_compression: 1,
        spatial_downsample: 4,
        latent_channels: 8,
        ..Default::default()
    };

    let mut with_memory = InferencePipeline::new(config.clone());
    let mut without_memory = InferencePipeline::new(config.clone());
    with_memory.cross_attention.gate = vec![0.0; config.num_heads];
    without_memory.cross_attention.gate = vec![0.0; config.num_heads];

    let backbone = SyntheticBackbone::new(1.0);
    let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
    let vae = SyntheticVAE::new(4, 1, 8);
    let depth = SyntheticDepthEstimator::new(5.0, 0.0);
    let source_pose = CameraPose::identity(0.0);
    let revisit_pose = CameraPose::identity(1.0);
    let frame = structured_frame(32, 32);
    let frame_shape = [1, 3, 1, 32, 32];
    let text_emb = vec![vec![0.0f32; 32]; 4];

    with_memory
        .update_memory(
            &frame,
            &frame_shape,
            std::slice::from_ref(&source_pose),
            &depth,
            &vae,
        )
        .unwrap();

    let (generated_with_memory, output_shape) = with_memory
        .generate_window(
            std::slice::from_ref(&revisit_pose),
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
        )
        .unwrap();
    let (generated_without_memory, baseline_shape) = without_memory
        .generate_window(
            std::slice::from_ref(&revisit_pose),
            &text_emb,
            &backbone,
            &scheduler,
            &vae,
        )
        .unwrap();

    assert_eq!(output_shape, frame_shape);
    assert_eq!(baseline_shape, frame_shape);
    assert!(with_memory.memory_store.num_patches() > 0);

    let with_memory_mse = mse(&generated_with_memory, &frame);
    let without_memory_mse = mse(&generated_without_memory, &frame);

    assert!(
        with_memory_mse < without_memory_mse * 0.75,
        "memory-conditioned revisit should be meaningfully closer to the observed scene: with_memory={with_memory_mse}, without_memory={without_memory_mse}"
    );
}
