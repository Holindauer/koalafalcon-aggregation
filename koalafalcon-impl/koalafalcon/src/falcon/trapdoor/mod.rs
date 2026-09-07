//! Falcon NTRU trapdoor generation (`NTRUGen`): sample `f, g` and solve for `F, G, h`.
//!
//! This is the key-generation trapdoor from Falcon §3.8.2 /
//! [falcon.py `ntrugen`](https://github.com/tprest/falcon.py).
//! Fast-Fourier signature sampling (`ffSampling`) is separate and not implemented here.

#![allow(non_snake_case)]

mod ntru_solve;
mod poly;
mod sample;
mod solve;

use crate::algebra::{CyclotomicRing, Field, KOALA_BEAR_PRIME, KoalaBear, KoalaBearRing};
use crate::falcon::keys::{PrivateKey, PublicKey};
use crate::falcon::parameters::KoalaFalconParameters;
use crate::profile;
use crate::utils::TrapdoorError;
use crate::utils::ntt::negacyclic_divide;
use poly::reduce_mod_q_i64;
use rand::Rng;
use sample::gen_poly;
use solve::verifies_ntru_equation;

/// Integer NTRU trapdoor polynomials satisfying `f G − g F = q` and `h = g f^{-1} (mod q)`.
#[derive(Debug, Clone)]
pub struct Trapdoor {
    pub f: Vec<i64>,
    pub g: Vec<i64>,
    pub F: Vec<i32>,
    pub G: Vec<i32>,
    pub h: Vec<u32>,
    #[allow(dead_code)]
    pub n: usize,
    pub q: u32,
}

impl Trapdoor {
    /// NTRU relation over `Z[x]/(x^n+1)`.
    pub fn satisfies_ntru_equation(&self) -> bool {
        verifies_ntru_equation(&self.f, &self.g, &self.F, &self.G, self.q)
    }
}

/// NTRU trapdoor generator (Falcon `NTRUGen`).
#[derive(Debug, Clone)]
pub struct TpdGen {
    pub n: usize,
    pub q: u32,
    pub alpha: f64,
    pub max_attempts: usize,
}

impl TpdGen {
    pub fn new(n: usize, q: u32, alpha: f64) -> Result<Self, TrapdoorError> {
        if n == 0 || !n.is_power_of_two() {
            return Err(TrapdoorError::UnsupportedDegree(n));
        }
        Ok(Self {
            n,
            q,
            alpha,
            max_attempts: 10_000,
        })
    }

    pub fn from_params(params: &KoalaFalconParameters) -> Result<Self, TrapdoorError> {
        Self::new(params.n, params.q, params.alpha)
    }

    #[cfg(test)]
    pub fn koala_512() -> Self {
        Self::from_params(&KoalaFalconParameters::falcon_512()).expect("n=512")
    }

    #[cfg(test)]
    pub fn koala_1024() -> Self {
        Self::from_params(&KoalaFalconParameters::falcon_1024()).expect("n=1024")
    }

    /// Sample `f, g, F, G, h` (Falcon Algorithm 5).
    pub fn generate(&self, rng: &mut impl Rng) -> Result<Trapdoor, TrapdoorError> {
        profile!("trapdoor_generate");
        let gs_bound = (self.alpha * self.alpha) * (self.q as f64);

        for _ in 0..self.max_attempts {
            let f = gen_poly(rng, self.n, self.q, self.alpha);
            let g = gen_poly(rng, self.n, self.q, self.alpha);

            if solve::gs_norm_squared_i64(&f, &g, self.q) > gs_bound {
                continue;
            }

            let Some(h) = try_public_key_ntt(&f, &g) else {
                continue;
            };

            let f_i32: Option<Vec<i32>> = f.iter().map(|&x| i32::try_from(x).ok()).collect();
            let g_i32: Option<Vec<i32>> = g.iter().map(|&x| i32::try_from(x).ok()).collect();
            let (Some(f_i32), Some(g_i32)) = (f_i32, g_i32) else {
                continue;
            };

            match solve::ntru_solve_i32(&f_i32, &g_i32) {
                Ok((F, G)) => {
                    let trap = Trapdoor {
                        f,
                        g,
                        F,
                        G,
                        h,
                        n: self.n,
                        q: self.q,
                    };
                    if trap.satisfies_ntru_equation() {
                        return Ok(trap);
                    }
                }
                Err(()) => continue,
            }
        }

        Err(TrapdoorError::ExceededAttempts(self.max_attempts))
    }

    /// Generate KoalaFalcon keys for degree `N` ∈ {512, 1024}.
    pub fn generate_keys<const N: usize>(
        &self,
        rng: &mut impl Rng,
    ) -> Result<(PrivateKey<N>, PublicKey<N>), TrapdoorError> {
        profile!("generate_keys");
        if self.n != N || (N != 512 && N != 1024) {
            return Err(TrapdoorError::UnsupportedDegree(N));
        }
        let trap = self.generate(rng)?;
        Ok((
            PrivateKey {
                f: i64_to_ring(&trap.f),
                g: i64_to_ring(&trap.g),
                F: i32_to_ring(&trap.F),
                G: i32_to_ring(&trap.G),
            },
            PublicKey {
                h: u32_to_ring(&trap.h),
            },
        ))
    }
}

fn i64_to_ring<const N: usize>(poly: &[i64]) -> KoalaBearRing<N> {
    debug_assert_eq!(poly.len(), N);
    let mut coeffs = [KoalaBear::ZERO; N];
    for (dst, &src) in coeffs.iter_mut().zip(poly) {
        *dst = KoalaBear::new(reduce_mod_q_i64(src, KOALA_BEAR_PRIME));
    }
    KoalaBearRing::from_coeffs(&coeffs)
}

fn i32_to_ring<const N: usize>(poly: &[i32]) -> KoalaBearRing<N> {
    debug_assert_eq!(poly.len(), N);
    let mut coeffs = [KoalaBear::ZERO; N];
    for (dst, &src) in coeffs.iter_mut().zip(poly) {
        *dst = KoalaBear::new(reduce_mod_q_i64(i64::from(src), KOALA_BEAR_PRIME));
    }
    KoalaBearRing::from_coeffs(&coeffs)
}

fn u32_to_ring<const N: usize>(poly: &[u32]) -> KoalaBearRing<N> {
    debug_assert_eq!(poly.len(), N);
    let mut coeffs = [KoalaBear::ZERO; N];
    for (dst, &src) in coeffs.iter_mut().zip(poly) {
        *dst = KoalaBear::new(src);
    }
    KoalaBearRing::from_coeffs(&coeffs)
}

fn try_public_key_ntt(f: &[i64], g: &[i64]) -> Option<Vec<u32>> {
    let n = f.len();
    debug_assert_eq!(n, g.len());
    let f_coeffs: Vec<KoalaBear> = f
        .iter()
        .map(|&c| KoalaBear::new(reduce_mod_q_i64(c, KOALA_BEAR_PRIME)))
        .collect();
    let g_coeffs: Vec<KoalaBear> = g
        .iter()
        .map(|&c| KoalaBear::new(reduce_mod_q_i64(c, KOALA_BEAR_PRIME)))
        .collect();
    let mut h = vec![KoalaBear::ZERO; n];
    let mut scratch_g = vec![KoalaBear::ZERO; n];
    let mut scratch_f = vec![KoalaBear::ZERO; n];
    negacyclic_divide(&g_coeffs, &f_coeffs, &mut h, &mut scratch_g, &mut scratch_f)?;
    Some(h.into_iter().map(|c| c.as_canonical_u32()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn generate_512_ntru_equation() {
        let tpd = TpdGen::koala_512();
        let mut rng = StdRng::from_os_rng();
        let trap = tpd.generate(&mut rng).expect("NTRUGen n=512");
        assert_eq!(trap.n, 512);
        assert_eq!(trap.f.len(), 512);
        assert_eq!(trap.g.len(), 512);
        assert_eq!(trap.F.len(), 512);
        assert_eq!(trap.G.len(), 512);
        assert_eq!(trap.h.len(), 512);
        assert!(trap.satisfies_ntru_equation());

        let (sk, pk) = tpd
            .generate_keys::<512>(&mut StdRng::from_os_rng())
            .unwrap();
        assert!(sk.matches_public(&pk));
    }

    #[test]
    fn generate_1024_ntru_equation() {
        let tpd = TpdGen::koala_1024();
        let mut rng = StdRng::from_os_rng();
        let trap = tpd.generate(&mut rng).expect("NTRUGen n=1024");
        assert_eq!(trap.n, 1024);
        assert_eq!(trap.f.len(), 1024);
        assert!(trap.satisfies_ntru_equation());

        let (sk, pk) = tpd
            .generate_keys::<1024>(&mut StdRng::from_os_rng())
            .unwrap();
        assert!(sk.matches_public(&pk));
    }
}
#[cfg(test)]
mod diag {
    use super::*;
    use crate::falcon::trapdoor::ntru_solve;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::collections::HashMap;

    #[test]
    fn diagnose_512_failure_modes() {
        let tpd = TpdGen::koala_512();
        let mut rng = StdRng::from_os_rng();
        let gs_bound = (tpd.alpha * tpd.alpha) * (tpd.q as f64);
        let mut gs_fail = 0usize;
        let mut h_fail = 0usize;
        let mut stages: HashMap<&'static str, usize> = HashMap::new();
        let mut ok = 0usize;
        for _ in 0..400 {
            let f_i = sample::gen_poly(&mut rng, tpd.n, tpd.q, tpd.alpha);
            let g_i = sample::gen_poly(&mut rng, tpd.n, tpd.q, tpd.alpha);
            if solve::gs_norm_squared_i64(&f_i, &g_i, tpd.q) > gs_bound {
                gs_fail += 1;
                continue;
            }
            if try_public_key_ntt(&f_i, &g_i).is_none() {
                h_fail += 1;
                continue;
            }
            let f32: Vec<i32> = f_i.iter().map(|&x| x as i32).collect();
            let g32: Vec<i32> = g_i.iter().map(|&x| x as i32).collect();
            let logn = tpd.n.trailing_zeros();
            let mut F = vec![0i32; tpd.n];
            let mut G = vec![0i32; tpd.n];
            let mut tmp_u32 = vec![0u32; ntru_solve::tmp_u32_len_for_test(logn)];
            let mut tmp_fxr = vec![ntru_solve::fxr_zero(); ntru_solve::tmp_fxr_len_for_test(logn)];
            match ntru_solve::solve_fail_stage(
                logn,
                &f32,
                &g32,
                &mut F,
                &mut G,
                &mut tmp_u32,
                &mut tmp_fxr,
            ) {
                None => {
                    ok += 1;
                    break;
                }
                Some(stage) => *stages.entry(stage).or_default() += 1,
            }
        }
        println!("gs_fail={gs_fail} h_fail={h_fail} ok={ok} stages={stages:?}");
        assert!(ok > 0, "no successful NTRUSolve");
    }
}
