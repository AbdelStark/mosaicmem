/// Noise scheduler abstraction for diffusion models.
///
/// Supports DDPM-style discrete scheduling and can be extended
/// for DDIM, flow matching, etc.
///
/// Trait for noise schedulers.
pub trait NoiseScheduler: Send + Sync {
    /// Get the number of timesteps.
    fn num_timesteps(&self) -> usize;

    /// Get the noise schedule (alpha_bar values).
    fn alphas_cumprod(&self) -> &[f32];

    /// Add noise to a clean sample at a given timestep.
    fn add_noise(&self, clean: &[f32], noise: &[f32], timestep: usize) -> Vec<f32>;

    /// Remove predicted noise from a noisy sample (single step).
    fn step(&self, predicted_noise: &[f32], noisy: &[f32], timestep: usize) -> Vec<f32>;

    /// Get the sigma (noise level) at a given timestep.
    fn sigma(&self, timestep: usize) -> f32;

    /// Get the timestep schedule for inference (possibly fewer steps than training).
    fn inference_timesteps(&self, num_steps: usize) -> Vec<usize>;
}

/// DDPM (Denoising Diffusion Probabilistic Models) scheduler.
pub struct DDPMScheduler {
    /// Number of training timesteps.
    pub num_train_timesteps: usize,
    /// Cumulative product of alphas.
    pub alphas_cumprod: Vec<f32>,
    /// Beta schedule.
    pub betas: Vec<f32>,
}

impl DDPMScheduler {
    /// Create a DDPM scheduler with a linear beta schedule.
    ///
    /// # Panics
    /// Panics if `num_timesteps` is 0.
    pub fn linear(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        assert!(num_timesteps > 0, "num_timesteps must be > 0");
        let denom = (num_timesteps.max(2) - 1) as f32;
        let betas: Vec<f32> = (0..num_timesteps)
            .map(|t| beta_start + (beta_end - beta_start) * t as f32 / denom)
            .collect();

        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut cumprod = 1.0f32;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod);
        }

        Self {
            num_train_timesteps: num_timesteps,
            alphas_cumprod,
            betas,
        }
    }

    /// Create a DDPM scheduler with a cosine beta schedule.
    pub fn cosine(num_timesteps: usize) -> Self {
        let s = 0.008f32;
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        for t in 0..num_timesteps {
            let f_t = ((t as f32 / num_timesteps as f32 + s) / (1.0 + s)
                * std::f32::consts::FRAC_PI_2)
                .cos()
                .powi(2);
            let f_0 = (s / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
            alphas_cumprod.push(f_t / f_0);
        }

        let mut betas = Vec::with_capacity(num_timesteps);
        betas.push(1.0 - alphas_cumprod[0]);
        for t in 1..num_timesteps {
            let beta = 1.0 - alphas_cumprod[t] / alphas_cumprod[t - 1];
            betas.push(beta.min(0.999));
        }

        Self {
            num_train_timesteps: num_timesteps,
            alphas_cumprod,
            betas,
        }
    }
}

impl NoiseScheduler for DDPMScheduler {
    fn num_timesteps(&self) -> usize {
        self.num_train_timesteps
    }

    fn alphas_cumprod(&self) -> &[f32] {
        &self.alphas_cumprod
    }

    fn add_noise(&self, clean: &[f32], noise: &[f32], timestep: usize) -> Vec<f32> {
        let alpha_bar = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_bar.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar).sqrt();

        clean
            .iter()
            .zip(noise.iter())
            .map(|(c, n)| sqrt_alpha * c + sqrt_one_minus_alpha * n)
            .collect()
    }

    fn step(&self, predicted_noise: &[f32], noisy: &[f32], timestep: usize) -> Vec<f32> {
        let alpha_bar = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha_bar.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha_bar).sqrt();

        // x_0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        if sqrt_alpha.abs() < 1e-8 {
            return noisy.to_vec();
        }

        noisy
            .iter()
            .zip(predicted_noise.iter())
            .map(|(x, eps)| (x - sqrt_one_minus_alpha * eps) / sqrt_alpha)
            .collect()
    }

    fn sigma(&self, timestep: usize) -> f32 {
        let alpha_bar = self.alphas_cumprod[timestep];
        ((1.0 - alpha_bar) / alpha_bar).sqrt()
    }

    fn inference_timesteps(&self, num_steps: usize) -> Vec<usize> {
        if num_steps == 0 {
            return vec![];
        }
        let step_size = self.num_train_timesteps / num_steps;
        (0..num_steps)
            .map(|i| self.num_train_timesteps - 1 - i * step_size)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_schedule() {
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        assert_eq!(scheduler.num_timesteps(), 1000);
        // Alpha_bar should decrease over time
        assert!(scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[999]);
        // First should be close to 1
        assert!(scheduler.alphas_cumprod[0] > 0.99);
    }

    #[test]
    fn test_add_and_remove_noise() {
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        let clean = vec![1.0, 2.0, 3.0];
        let noise = vec![0.1, -0.2, 0.3];
        let t = 100;

        let noisy = scheduler.add_noise(&clean, &noise, t);
        let recovered = scheduler.step(&noise, &noisy, t);

        for (c, r) in clean.iter().zip(recovered.iter()) {
            assert!((c - r).abs() < 1e-4);
        }
    }

    #[test]
    fn test_inference_timesteps() {
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        let steps = scheduler.inference_timesteps(50);
        assert_eq!(steps.len(), 50);
        assert_eq!(steps[0], 999); // Start from highest
    }

    #[test]
    fn test_cosine_schedule() {
        let scheduler = DDPMScheduler::cosine(1000);
        assert_eq!(scheduler.num_timesteps(), 1000);
        assert!(scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[999]);
    }

    #[test]
    fn test_sigma_increases() {
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        // Sigma should increase with timestep (more noise at higher t)
        let sigma_low = scheduler.sigma(10);
        let sigma_high = scheduler.sigma(900);
        assert!(
            sigma_high > sigma_low,
            "Sigma should increase: {} > {}",
            sigma_high,
            sigma_low
        );
    }

    #[test]
    fn test_alphas_cumprod_bounds() {
        let scheduler = DDPMScheduler::linear(1000, 1e-4, 0.02);
        for &ac in scheduler.alphas_cumprod() {
            assert!(ac > 0.0 && ac <= 1.0, "Alpha_cumprod out of (0,1]: {}", ac);
        }
    }
}
