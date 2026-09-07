//! KoalaBear NTT utilities for multiplication in \(R_q=\mathbb F_q[X]/(X^n+1)\).
//!
//! Forward uses decimation in frequency (natural → bit-reversed); inverse uses
//! decimation in time (bit-reversed → natural) so the multiply path needs no
//! explicit bit-reversal. Twiddle / twist tables for \(n\in\{512,1024\}\) are
//! built once and cached.

use crate::Error;
use crate::algebra::{Field, KoalaBear};
use crate::utils::{batch_invert_in_place, field_from_usize};
use std::sync::OnceLock;

fn primitive_roots(n: usize) -> Result<(KoalaBear, KoalaBear), Error> {
    match n {
        512 => Ok((KoalaBear::new(860_702_919), KoalaBear::new(665_670_555))),
        1024 => Ok((KoalaBear::new(2_000_983_452), KoalaBear::new(860_702_919))),
        _ => Err(Error::UnsupportedRingDimension(n)),
    }
}

/// Precomputed tables for twiddles and offsets for NTT, INTT for negacyclic
/// convolution. For both decimation in frequency and decimation in time.
struct NegacyclicTables {
    n: usize,
    forward_freq_twiddles: Vec<KoalaBear>,
    forward_stage_offsets: Vec<usize>,
    inverse_time_twiddles: Vec<KoalaBear>,
    inverse_stage_offsets: Vec<usize>,
    twist: Vec<KoalaBear>,   // twist vector
    untwist: Vec<KoalaBear>, // inverse twist vector + 1/n scale
}

impl NegacyclicTables {
    fn build(n: usize) -> Result<Self, Error> {
        if n == 0 || !n.is_power_of_two() {
            return Err(Error::UnsupportedRingDimension(n));
        }
        let (omega_2n, omega_n) = primitive_roots(n)?;

        let mut omega_pows = vec![KoalaBear::ONE; n];
        for t in 1..n {
            omega_pows[t] = omega_pows[t - 1] * omega_n;
        }
        let omega_inv = omega_n.invert();
        let mut omega_inv_pows = vec![KoalaBear::ONE; n];
        for t in 1..n {
            omega_inv_pows[t] = omega_inv_pows[t - 1] * omega_inv;
        }

        let mut forward_freq_twiddles = Vec::new();
        let mut forward_stage_offsets = Vec::new();
        let mut m = n;
        while m >= 2 {
            forward_stage_offsets.push(forward_freq_twiddles.len());
            let h = m / 2;
            let s = n / m;
            for j in 0..h {
                forward_freq_twiddles.push(omega_pows[j * s]);
            }
            m /= 2;
        }

        let mut inverse_time_twiddles = Vec::new();
        let mut inverse_stage_offsets = Vec::new();
        let mut m = 2usize;
        while m <= n {
            inverse_stage_offsets.push(inverse_time_twiddles.len());
            let h = m / 2;
            let s = n / m;
            for j in 0..h {
                inverse_time_twiddles.push(omega_inv_pows[j * s]);
            }
            m <<= 1;
        }

        let mut twist = vec![KoalaBear::ONE; n];
        for i in 1..n {
            twist[i] = twist[i - 1] * omega_2n;
        }
        let zeta_inv = omega_2n.invert();
        let mut untwist = vec![KoalaBear::ONE; n];
        untwist[1] = zeta_inv;
        for i in 2..n {
            untwist[i] = untwist[i - 1] * zeta_inv;
        }
        let n_inv = field_from_usize::<KoalaBear>(n).invert();
        for u in untwist.iter_mut() {
            *u *= n_inv;
        }

        Ok(Self {
            n,
            forward_freq_twiddles,
            forward_stage_offsets,
            inverse_time_twiddles,
            inverse_stage_offsets,
            twist,
            untwist,
        })
    }
}

static TABLES_512: OnceLock<NegacyclicTables> = OnceLock::new();
static TABLES_1024: OnceLock<NegacyclicTables> = OnceLock::new();

fn tables(n: usize) -> Result<&'static NegacyclicTables, Error> {
    match n {
        512 => Ok(TABLES_512.get_or_init(|| NegacyclicTables::build(512).expect("512 tables"))),
        1024 => Ok(TABLES_1024.get_or_init(|| NegacyclicTables::build(1024).expect("1024 tables"))),
        _ => Err(Error::UnsupportedRingDimension(n)),
    }
}

fn forward_ntt_decimation_in_freq(values: &mut [KoalaBear], t: &NegacyclicTables) {
    debug_assert_eq!(values.len(), t.n);
    let n = t.n;
    let mut m = n;
    let mut stage = 0usize;
    while m >= 2 {
        let h = m / 2;
        let tw_base = t.forward_stage_offsets[stage];
        let mut k = 0usize;
        while k < n {
            for j in 0..h {
                let u = values[k + j];
                let v = values[k + j + h];
                values[k + j] = u + v;
                values[k + j + h] = (u - v) * t.forward_freq_twiddles[tw_base + j];
            }
            k += m;
        }
        m /= 2;
        stage += 1;
    }
}

fn inverse_ntt_decimation_in_time_unscaled(values: &mut [KoalaBear], t: &NegacyclicTables) {
    debug_assert_eq!(values.len(), t.n);
    let n = t.n;
    let mut m = 2usize;
    let mut stage = 0usize;
    while m <= n {
        let h = m / 2;
        let tw_base = t.inverse_stage_offsets[stage];
        let mut k = 0usize;
        while k < n {
            for j in 0..h {
                let u = values[k + j];
                let v = values[k + j + h] * t.inverse_time_twiddles[tw_base + j];
                values[k + j] = u + v;
                values[k + j + h] = u - v;
            }
            k += m;
        }
        m <<= 1;
        stage += 1;
    }
}

/// Twist then forward decimation in frequency
/// (input in original order, result is bit-reversed)
pub fn forward_negacyclic_ntt(values: &mut [KoalaBear]) -> Result<(), Error> {
    let t = tables(values.len())?;
    for (value, twist) in values.iter_mut().zip(t.twist.iter()) {
        *value *= *twist;
    }
    forward_ntt_decimation_in_freq(values, t);
    Ok(())
}

/// Inverse decimation in time then untwist and normalization
/// (input is bit-reversed, result is original order)
pub fn inverse_negacyclic_ntt(values: &mut [KoalaBear]) -> Result<(), Error> {
    let t = tables(values.len())?;
    inverse_ntt_decimation_in_time_unscaled(values, t);
    for (value, untwist) in values.iter_mut().zip(t.untwist.iter()) {
        *value *= *untwist;
    }
    Ok(())
}

/// Cyclotomic ring multiplication by negacyclic convolution
pub fn negacyclic_mul(
    a: &[KoalaBear],
    b: &[KoalaBear],
    out: &mut [KoalaBear],
    scratch_a: &mut [KoalaBear],
    scratch_b: &mut [KoalaBear],
) -> Result<(), Error> {
    let n = a.len();
    if n != b.len() || n != out.len() || n != scratch_a.len() || n != scratch_b.len() {
        return Err(Error::UnsupportedRingDimension(n));
    }
    let _ = tables(n)?;
    scratch_a.copy_from_slice(a);
    scratch_b.copy_from_slice(b);
    forward_negacyclic_ntt(scratch_a)?;
    forward_negacyclic_ntt(scratch_b)?;
    for i in 0..n {
        out[i] = scratch_a[i] * scratch_b[i];
    }
    inverse_negacyclic_ntt(out)?;
    Ok(())
}

/// Cyclotomic ring inversion by negacyclic convolution + batch inversion.
#[allow(dead_code)] // reachable via CyclotomicRing::maybe_invert (private algebra)
pub fn negacyclic_invert(
    a: &[KoalaBear],
    out: &mut [KoalaBear],
    scratch: &mut [KoalaBear],
) -> Option<()> {
    let n = a.len();
    if n != out.len() || n != scratch.len() {
        return None;
    }
    tables(n).ok()?;
    scratch.copy_from_slice(a);
    forward_negacyclic_ntt(scratch).ok()?;
    out.copy_from_slice(scratch);
    batch_invert_in_place(out)?;
    inverse_negacyclic_ntt(out).ok()?;
    Some(())
}

/// Cyclotomic ring division by negacyclic convolution + batch inversion
pub fn negacyclic_divide(
    numerator: &[KoalaBear],
    denominator: &[KoalaBear],
    out: &mut [KoalaBear],
    scratch_num: &mut [KoalaBear],
    scratch_den: &mut [KoalaBear],
) -> Option<()> {
    let n = numerator.len();
    if n != denominator.len() || n != out.len() || n != scratch_num.len() || n != scratch_den.len()
    {
        return None;
    }
    tables(n).ok()?;
    scratch_num.copy_from_slice(numerator);
    scratch_den.copy_from_slice(denominator);
    forward_negacyclic_ntt(scratch_num).ok()?;
    forward_negacyclic_ntt(scratch_den).ok()?;
    batch_invert_in_place(scratch_den)?;
    for i in 0..n {
        out[i] = scratch_num[i] * scratch_den[i];
    }
    inverse_negacyclic_ntt(out).ok()?;
    Some(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::algebra::KOALA_BEAR_PRIME;
    use rand::{SeedableRng, rngs::StdRng};

    fn random_poly(n: usize, rng: &mut StdRng) -> Vec<KoalaBear> {
        (0..n).map(|_| KoalaBear::random(rng)).collect()
    }

    fn naive_negacyclic_mul(a: &[KoalaBear], b: &[KoalaBear]) -> Vec<KoalaBear> {
        let n = a.len();
        let mut c = vec![KoalaBear::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                let product = a[i] * b[j];
                let index = i + j;
                if index < n {
                    c[index] += product;
                } else {
                    c[index - n] -= product;
                }
            }
        }
        c
    }

    fn assert_ntt_root_properties(n: usize) {
        let (omega_2n, omega_n) = primitive_roots(n).unwrap();
        let two_n = 2 * n;

        assert_eq!(
            omega_2n.power(n as u32),
            KoalaBear::new(KOALA_BEAR_PRIME - 1),
            "omega_2n^n should equal -1"
        );
        assert_eq!(
            omega_2n.power(two_n as u32),
            KoalaBear::ONE,
            "omega_2n^2n should equal 1"
        );
        for i in 1..(two_n - 1) {
            assert_ne!(
                omega_2n.power(i as u32),
                KoalaBear::ONE,
                "omega_2n should have primitive order 2n"
            );
        }

        assert_eq!(omega_n, omega_2n.power(2));
        assert_eq!(omega_n.power(n as u32), KoalaBear::ONE);
        for i in 1..(n - 1) {
            assert_ne!(
                omega_n.power(i as u32),
                KoalaBear::ONE,
                "omega_n should have primitive order n"
            );
        }
    }

    #[test]
    fn precomputed_ntt_root_properties() {
        assert_ntt_root_properties(512);
        assert_ntt_root_properties(1024);
    }

    #[test]
    fn unsupported_ring_dimension_error() {
        assert!(matches!(
            primitive_roots(256),
            Err(Error::UnsupportedRingDimension(256))
        ));
        assert!(matches!(tables(0), Err(Error::UnsupportedRingDimension(0))));
        assert!(matches!(tables(3), Err(Error::UnsupportedRingDimension(3))));
        assert!(matches!(
            forward_negacyclic_ntt(&mut [KoalaBear::ZERO; 8]),
            Err(Error::UnsupportedRingDimension(8))
        ));
        let a = vec![KoalaBear::ONE; 512];
        let b = vec![KoalaBear::ONE; 1024];
        let mut out = vec![KoalaBear::ZERO; 512];
        let mut sa = vec![KoalaBear::ZERO; 512];
        let mut sb = vec![KoalaBear::ZERO; 512];
        assert!(matches!(
            negacyclic_mul(&a, &b, &mut out, &mut sa, &mut sb),
            Err(Error::UnsupportedRingDimension(_))
        ));
    }

    #[test]
    fn forward_inverse_negacyclic_roundtrip() {
        let mut rng = StdRng::from_os_rng();
        for n in [512usize, 1024] {
            let original = random_poly(n, &mut rng);
            let mut x = original.clone();
            forward_negacyclic_ntt(&mut x).unwrap();
            inverse_negacyclic_ntt(&mut x).unwrap();
            assert_eq!(x, original, "n={n}");
        }
    }

    #[test]
    fn negacyclic_mul_matches_naive() {
        let mut rng = StdRng::from_os_rng();
        for n in [512usize, 1024] {
            let a = random_poly(n, &mut rng);
            let b = random_poly(n, &mut rng);
            let mut out = vec![KoalaBear::ZERO; n];
            let mut sa = vec![KoalaBear::ZERO; n];
            let mut sb = vec![KoalaBear::ZERO; n];
            negacyclic_mul(&a, &b, &mut out, &mut sa, &mut sb).unwrap();
            assert_eq!(out, naive_negacyclic_mul(&a, &b), "n={n}");
        }
    }

    #[test]
    fn invert_roundtrip_and_noninvertible() {
        let mut rng = StdRng::from_os_rng();
        for n in [512usize, 1024] {
            let a = random_poly(n, &mut rng);
            let mut inv = vec![KoalaBear::ZERO; n];
            let mut scratch = vec![KoalaBear::ZERO; n];
            negacyclic_invert(&a, &mut inv, &mut scratch).expect("invertible");
            let mut prod = vec![KoalaBear::ZERO; n];
            let mut sa = vec![KoalaBear::ZERO; n];
            let mut sb = vec![KoalaBear::ZERO; n];
            negacyclic_mul(&a, &inv, &mut prod, &mut sa, &mut sb).unwrap();
            let mut one = vec![KoalaBear::ZERO; n];
            one[0] = KoalaBear::ONE;
            assert_eq!(prod, one, "n={n}");

            let zero = vec![KoalaBear::ZERO; n];
            assert!(negacyclic_invert(&zero, &mut inv, &mut scratch).is_none());
        }
    }

    #[test]
    fn divide_matches_mul_by_inverse() {
        let mut rng = StdRng::from_os_rng();
        for n in [512usize, 1024] {
            let g = random_poly(n, &mut rng);
            let f = random_poly(n, &mut rng);
            let mut h_div = vec![KoalaBear::ZERO; n];
            let mut sn = vec![KoalaBear::ZERO; n];
            let mut sd = vec![KoalaBear::ZERO; n];
            negacyclic_divide(&g, &f, &mut h_div, &mut sn, &mut sd).expect("f invertible");

            let mut f_inv = vec![KoalaBear::ZERO; n];
            let mut scratch = vec![KoalaBear::ZERO; n];
            negacyclic_invert(&f, &mut f_inv, &mut scratch).unwrap();
            let mut h_mul = vec![KoalaBear::ZERO; n];
            let mut sa = vec![KoalaBear::ZERO; n];
            let mut sb = vec![KoalaBear::ZERO; n];
            negacyclic_mul(&g, &f_inv, &mut h_mul, &mut sa, &mut sb).unwrap();
            assert_eq!(h_div, h_mul, "n={n}");
        }
    }
}
