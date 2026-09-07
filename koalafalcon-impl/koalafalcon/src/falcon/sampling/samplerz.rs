//! Falcon discrete Gaussian sampler over `Z` (`SamplerZ`).
//!
//! Port of [falcon.py `samplerz.py`](https://github.com/tprest/falcon.py/blob/master/samplerz.py).
//! Requires `1 < sigmin ≤ sigma ≤ MAX_SIGMA`.

use crate::utils::Error;
use crate::utils::error::FourierSamplingError;
use rand::{CryptoRng, Rng};

/// Upper bound on all values of `sigma` accepted by [`sampler_z`].
pub const MAX_SIGMA: f64 = 1.8205;

const INV_2SIGMA2: f64 = 1.0 / (2.0 * MAX_SIGMA * MAX_SIGMA);
const LN2: f64 = std::f64::consts::LN_2;
const ILN2: f64 = std::f64::consts::LOG2_E;

/// Reverse CDF table for a half-Gaussian of parameter [`MAX_SIGMA`] (72-bit precision).
const RCDT: [u128; 18] = [
    3024686241123004913666,
    1564742784480091954050,
    636254429462080897535,
    199560484645026482916,
    47667343854657281903,
    8595902006365044063,
    1163297957344668388,
    117656387352093658,
    8867391802663976,
    496969357462633,
    20680885154299,
    638331848991,
    14602316184,
    247426747,
    3104126,
    28824,
    198,
    1,
];

/// FACCT polynomial coefficients for `≈ exp(-x)`.
const C: [u64; 13] = [
    0x0000_0004_7411_83A3,
    0x0000_0036_548C_FC06,
    0x0000_024F_DCBF_140A,
    0x0000_171D_939D_E045,
    0x0000_D00C_F58F_6F84,
    0x0006_8068_1CF7_96E3,
    0x002D_82D8_305B_0FEA,
    0x0111_1111_0E06_6FD0,
    0x0555_5555_5507_0F00,
    0x1555_5555_5581_FF00,
    0x4000_0000_0002_B400,
    0x7FFF_FFFF_FFFF_4800,
    0x8000_0000_0000_0000,
];

fn next_bytes<R: Rng>(rng: &mut R, n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    rng.fill_bytes(&mut buf);
    buf
}

fn basesampler(rng: &mut (impl Rng + CryptoRng)) -> i64 {
    let bytes = next_bytes(rng, 9); // 72 bits
    let mut u = 0u128;
    for (i, &b) in bytes.iter().enumerate() {
        u |= (b as u128) << (8 * i);
    }
    let mut z0 = 0i64;
    for &elt in &RCDT {
        z0 += i64::from(u < elt);
    }
    z0
}

fn approxexp(x: f64, ccs: f64) -> u64 {
    let mut y = C[0];
    let z = (x * ((1u64 << 63) as f64)) as u64;
    for &elt in &C[1..] {
        y = elt.wrapping_sub(((z as u128 * y as u128) >> 63) as u64);
    }
    let z = ((ccs * ((1u64 << 63) as f64)) as u64) << 1;
    ((z as u128 * y as u128) >> 63) as u64
}

fn berexp(x: f64, ccs: f64, rng: &mut (impl Rng + CryptoRng)) -> bool {
    let mut s = (x * ILN2) as i32;
    let r = x - (s as f64) * LN2;
    s = s.min(63);
    let z = (approxexp(r, ccs).wrapping_sub(1)) >> s;
    let mut w = 0i32;
    for i in (0..=56).rev().step_by(8) {
        let p = next_bytes(rng, 1)[0] as i32;
        w = p - (((z >> i) & 0xFF) as i32);
        if w != 0 {
            break;
        }
    }
    w < 0
}

/// Sample `z ∼ D_{Z, μ, σ}` with relative scaling `sigmin`.
///
/// Requires `1 < sigmin ≤ sigma ≤ MAX_SIGMA` (enforced by [`sample_discrete_z`]).
pub fn sampler_z(mu: f64, sigma: f64, sigmin: f64, rng: &mut (impl Rng + CryptoRng)) -> i64 {
    let s = mu.floor() as i64;
    let r = mu - s as f64;
    let dss = 1.0 / (2.0 * sigma * sigma);
    let ccs = sigmin / sigma;

    loop {
        let z0 = basesampler(rng);
        let b = (next_bytes(rng, 1)[0] & 1) as i64;
        let z = b + (2 * b - 1) * z0;
        let mut x = (z as f64 - r).powi(2) * dss;
        x -= (z0 as f64).powi(2) * INV_2SIGMA2;
        if berexp(x, ccs, rng) {
            return z + s;
        }
    }
}

/// Whether `(mu, sigma, sigmin)` is inside SamplerZ’s proven range.
pub fn sampler_z_params_ok(mu: f64, sigma: f64, sigmin: f64) -> bool {
    mu.is_finite()
        && sigma.is_finite()
        && sigmin.is_finite()
        && sigmin > 1.0
        && sigmin <= sigma
        && sigma <= MAX_SIGMA
}

/// Sample centered on `mu` with width `sigma` via rejection-based [`sampler_z`].
///
/// Returns [`FourierSamplingError::InvalidSamplerParameters`] when parameters fall
/// outside `1 < sigmin ≤ sigma ≤ MAX_SIGMA` (or any input is non-finite).
pub fn sample_discrete_z(
    mu: f64,
    sigma: f64,
    sigmin: f64,
    rng: &mut (impl Rng + CryptoRng),
) -> Result<i64, Error> {
    if !sampler_z_params_ok(mu, sigma, sigmin) {
        return Err(FourierSamplingError::InvalidSamplerParameters { sigma, sigmin }.into());
    }
    Ok(sampler_z(mu, sigma, sigmin, rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn valid_params_sample_ok() {
        let mut rng = StdRng::from_os_rng();
        let z = sample_discrete_z(0.3, 1.5, 1.2, &mut rng).expect("valid");
        assert!(z.abs() < 100);
    }

    #[test]
    fn rejects_sigma_below_sigmin() {
        let mut rng = StdRng::from_os_rng();
        let err = sample_discrete_z(0.0, 1.1, 1.5, &mut rng).unwrap_err();
        assert!(matches!(
            err,
            Error::FourierSampling(FourierSamplingError::InvalidSamplerParameters { .. })
        ));
    }

    #[test]
    fn rejects_sigma_above_max() {
        let mut rng = StdRng::from_os_rng();
        let err = sample_discrete_z(0.0, MAX_SIGMA + 0.1, 1.2, &mut rng).unwrap_err();
        assert!(matches!(
            err,
            Error::FourierSampling(FourierSamplingError::InvalidSamplerParameters { .. })
        ));
    }

    #[test]
    fn rejects_nonfinite_params() {
        let mut rng = StdRng::from_os_rng();
        for (mu, sigma, sigmin) in [
            (f64::NAN, 1.5, 1.2),
            (0.0, f64::INFINITY, 1.2),
            (0.0, 1.5, f64::NAN),
            (f64::NEG_INFINITY, 1.5, 1.2),
        ] {
            let err = sample_discrete_z(mu, sigma, sigmin, &mut rng).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::FourierSampling(FourierSamplingError::InvalidSamplerParameters { .. })
                ),
                "mu={mu} sigma={sigma} sigmin={sigmin}"
            );
        }
    }

    #[test]
    fn production_source_has_no_continuous_fallback() {
        let src = include_str!("samplerz.rs");
        let production = src
            .rsplit_once("mod tests {")
            .map(|(p, _)| p)
            .unwrap_or(src);
        assert!(
            !production.contains("Box\u{2013}Muller")
                && !production.contains("Box\u{002d}Muller")
                && !production.contains("Continuous Gaussian")
                && !production.contains("u1.ln()"),
            "continuous-Gaussian fallback must not be present"
        );
    }
}
