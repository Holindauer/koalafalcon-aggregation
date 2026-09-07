//! Centered coefficient and norm helpers for signing and verification.

use crate::algebra::{CyclotomicRing, Field, KOALA_BEAR_PRIME, KoalaBear, KoalaBearRing};

/// Exact centered representatives in \((-q/2,q/2]\).
pub(crate) fn center_poly_i64(coeffs: &[KoalaBear]) -> Vec<i64> {
    let q = KOALA_BEAR_PRIME as i64;
    let half = q / 2;
    coeffs
        .iter()
        .map(|c| {
            let mut v = c.as_canonical_u32() as i64;
            if v > half {
                v -= q;
            }
            v
        })
        .collect()
}

pub(crate) fn i64_to_ring<const N: usize>(coeffs: &[i64]) -> KoalaBearRing<N> {
    debug_assert_eq!(coeffs.len(), N);
    let mut out = [KoalaBear::ZERO; N];
    let q = KOALA_BEAR_PRIME as i64;
    for (dst, &src) in out.iter_mut().zip(coeffs.iter()) {
        let mut r = src % q;
        if r < 0 {
            r += q;
        }
        *dst = KoalaBear::new(r as u32);
    }
    KoalaBearRing::from_coeffs(&out)
}

pub(crate) fn squared_l2(v: &[i64]) -> u128 {
    v.iter()
        .map(|&x| {
            let x = x as i128;
            (x * x) as u128
        })
        .sum()
}

pub(crate) fn l1_norm(v: &[i64]) -> u128 {
    v.iter().map(|&x| x.unsigned_abs() as u128).sum()
}

pub(crate) fn linf_norm(v: &[i64]) -> u64 {
    v.iter()
        .map(|&x| x.unsigned_abs() as u64)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::KoalaBear;

    #[test]
    fn center_poly_i64_boundaries() {
        let q = KOALA_BEAR_PRIME as i64;
        let half = q / 2;
        let vals = [
            KoalaBear::new(0),
            KoalaBear::new(1),
            KoalaBear::new(half as u32),
            KoalaBear::new(half as u32 + 1),
            KoalaBear::new(KOALA_BEAR_PRIME - 1),
        ];
        let c = center_poly_i64(&vals);
        assert_eq!(c[0], 0);
        assert_eq!(c[1], 1);
        assert_eq!(c[2], half);
        assert_eq!(c[3], half + 1 - q);
        assert_eq!(c[4], -1);
    }

    #[test]
    fn i64_to_ring_roundtrip_mod_q() {
        let q = KOALA_BEAR_PRIME as i64;
        let coeffs = [0i64, 1, q - 1, -1, q + 5];
        let ring = i64_to_ring::<5>(&coeffs);
        assert_eq!(ring.coeffs()[0].as_canonical_u32(), 0);
        assert_eq!(ring.coeffs()[1].as_canonical_u32(), 1);
        assert_eq!(ring.coeffs()[2].as_canonical_u32(), KOALA_BEAR_PRIME - 1);
        assert_eq!(ring.coeffs()[3].as_canonical_u32(), KOALA_BEAR_PRIME - 1);
        assert_eq!(ring.coeffs()[4].as_canonical_u32(), 5);
    }

    #[test]
    fn squared_l2_basic() {
        assert_eq!(squared_l2(&[3, 4]), 25);
        assert_eq!(squared_l2(&[-3, 4]), 25);
    }

    #[test]
    fn l1_and_linf_basic() {
        assert_eq!(l1_norm(&[3, -4]), 7);
        assert_eq!(linf_norm(&[3, -4]), 4);
        assert_eq!(linf_norm(&[]), 0);
    }
}
