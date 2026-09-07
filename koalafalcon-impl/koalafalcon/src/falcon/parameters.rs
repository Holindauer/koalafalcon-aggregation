#![allow(non_snake_case)]

use crate::algebra::KOALA_BEAR_PRIME;

/// Supported KoalaFalcon ring degrees.
pub const N512: usize = 512;
pub const N1024: usize = 1024;

/// KoalaFalcon salt len in bytes (k = 416 bits)
pub const SALT_LEN: usize = 52;

pub trait FalconParameterSet {
    const RING_DEGREE: usize;
    const LOG_RING_DEGREE: usize;
    const SAMPLER_SIGMA: f64;
    const NORM_BOUND_SQUARED: u128;
    const MAX_COEFF_ABS: u64;
    const SECURE: bool;

    fn parameters() -> KoalaFalconParameters;
}

// Secure parameter sets
pub struct SecureKoalaFalcon512;
pub struct SecureKoalaFalcon1024;

impl FalconParameterSet for SecureKoalaFalcon512 {
    const RING_DEGREE: usize = N512;
    const LOG_RING_DEGREE: usize = 9;
    const SAMPLER_SIGMA: f64 = 0.0; // derived from `s` below
    const NORM_BOUND_SQUARED: u128 = 5_901_050_624_566;
    const MAX_COEFF_ABS: u64 = ((KOALA_BEAR_PRIME - 1) / 2) as u64;
    const SECURE: bool = true;

    fn parameters() -> KoalaFalconParameters {
        KoalaFalconParameters::falcon_512()
    }
}

impl FalconParameterSet for SecureKoalaFalcon1024 {
    const RING_DEGREE: usize = N1024;
    const LOG_RING_DEGREE: usize = 10;
    const SAMPLER_SIGMA: f64 = 0.0;
    const NORM_BOUND_SQUARED: u128 = 12_182_814_192_650;
    const MAX_COEFF_ABS: u64 = ((KOALA_BEAR_PRIME - 1) / 2) as u64;
    const SECURE: bool = true;

    fn parameters() -> KoalaFalconParameters {
        KoalaFalconParameters::falcon_1024()
    }
}

/// Runtime KoalaFalcon parameters used by signing and verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KoalaFalconParameters {
    pub n: usize,
    pub q: u32,
    pub alpha: f64,
    pub k: u32,
    /// Target Gram-Schmidt norm bound: `||B_{f,g}||_GS <= alpha * sqrt(q)`.
    pub target_gs_norm: f64,
    /// Gaussian width for PreSmp.
    pub s: f64,
    /// Signature norm bound: `beta = tau * s * sqrt(2n)`.
    pub beta: f64,
    /// Exact integer bound
    pub beta_squared: u128,
    /// Maximum centered coefficient magnitude
    pub max_coeff_abs: u64,
    /// Whether this parameter set is intended for secure signing.
    pub secure: bool,
}

impl Default for KoalaFalconParameters {
    fn default() -> Self {
        Self::falcon_512()
    }
}

impl KoalaFalconParameters {
    /// KoalaFalcon-512: `lambda = 128`, `n = 512`.
    pub fn falcon_512() -> Self {
        Self::for_degree(N512, 128, SecureKoalaFalcon512::NORM_BOUND_SQUARED, true)
    }

    /// KoalaFalcon-1024: `lambda = 256`, `n = 1024`.
    pub fn falcon_1024() -> Self {
        Self::for_degree(N1024, 256, SecureKoalaFalcon1024::NORM_BOUND_SQUARED, true)
    }

    pub(crate) fn for_degree(n: usize, lambda: u32, beta_squared: u128, secure: bool) -> Self {
        let q = KOALA_BEAR_PRIME;
        let alpha = 1.17;
        let k = 416;
        let tau = 1.1;
        let q_s = 1u128 << 64;

        let target_gs_norm = alpha * (q as f64).sqrt();
        let epsilon = 1.0 / ((q_s as f64) * (lambda as f64)).sqrt();
        let s = Self::s(n, q, alpha, epsilon);
        let beta = tau * s * (2.0 * n as f64).sqrt();

        Self {
            n,
            q,
            alpha,
            k,
            target_gs_norm,
            s,
            beta,
            beta_squared,
            max_coeff_abs: ((q - 1) / 2) as u64,
            secure,
        }
    }

    /// `s = (1/pi) * sqrt(ln(4n(1 + 1/epsilon)) / 2) * alpha * sqrt(q)`
    fn s(n: usize, q: u32, alpha: f64, epsilon: f64) -> f64 {
        let n_f = n as f64;
        let smoothing = (4.0 * n_f * (1.0 + 1.0 / epsilon)).ln() / 2.0;
        (1.0 / std::f64::consts::PI) * smoothing.sqrt() * alpha * (q as f64).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secure_512_defaults() {
        let params = KoalaFalconParameters::falcon_512();
        assert!(params.secure);
        assert_eq!(params.n, 512);
        assert_eq!(
            params.beta_squared,
            SecureKoalaFalcon512::NORM_BOUND_SQUARED
        );
        println!("params: {:?}", params);
    }
}
