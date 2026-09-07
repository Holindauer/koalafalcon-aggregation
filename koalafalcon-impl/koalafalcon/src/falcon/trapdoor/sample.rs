//! Discrete Gaussian sampling for NTRU secrets `f, g`.

use rand::Rng;
use std::f64::consts::PI;

/// Sample from a continuous standard normal via Box–Muller.
fn standard_normal(rng: &mut impl Rng) -> f64 {
    // Avoid log(0).
    let u1 = rng.random::<f64>().max(f64::EPSILON);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Approximate sample from `D_{Z, 0, sigma}` by rounding a continuous Gaussian.
///
/// Falcon’s `SamplerZ` only covers `σ ≤ 1.8205`. KoalaBear moduli force much larger
/// `σ_{f,g}`, so keygen uses this general sampler. Signature sampling can later use
/// the fixed-range `SamplerZ` from the og specification.
pub fn sample_discrete_gaussian(rng: &mut impl Rng, sigma: f64) -> i64 {
    debug_assert!(sigma > 0.0);
    (standard_normal(rng) * sigma).round() as i64
}

/// Sample a coefficient of `f`/`g` with
/// `σ_{f,g} = alpha · √(q / (2n))` (Falcon Algorithm 5 / eq. (2.12)).
pub fn gen_poly(rng: &mut impl Rng, n: usize, q: u32, alpha: f64) -> Vec<i64> {
    let sigma = alpha * ((q as f64) / (2.0 * n as f64)).sqrt();
    (0..n)
        .map(|_| sample_discrete_gaussian(rng, sigma))
        .collect()
}
