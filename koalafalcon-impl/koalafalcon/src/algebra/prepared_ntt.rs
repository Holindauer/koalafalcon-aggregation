//! Prepared fixed-left negacyclic multiplication.
//!
//! A full ring product `a * b` needs a forward NTT of both operands. When the
//! left factor is fixed (Falcon’s public `h` on verify / sampling), that
//! transform can be done once and reused: each multiply then only NTTs the
//! varying right-hand side, pointwise-muls, and inverse-NTTs.

use crate::algebra::{CyclotomicRing, KoalaBear, KoalaBearRing};
use crate::profile;
use crate::utils::Error;
use crate::utils::ntt::{forward_negacyclic_ntt, inverse_negacyclic_ntt};

/// Cached forward NTT of a fixed left ring element (bit-reversed order).
///
/// Stored on [`VerifyingKey`](crate::VerifyingKey) as `h_mul` so verify does not
/// re-transform `h` on every signature check.
#[derive(Debug, Clone)]
pub struct PreparedNegacyclicMultiplier<const N: usize> {
    left_hat_bitrev: Vec<KoalaBear>,
}

impl<const N: usize> PreparedNegacyclicMultiplier<N> {
    pub fn new(left: KoalaBearRing<N>) -> Result<Self, Error> {
        profile!("prepare_negacyclic_multiplier");
        if N != 512 && N != 1024 {
            return Err(Error::UnsupportedRingDimension(N));
        }
        let mut left_hat_bitrev = left.coeffs().to_vec();
        forward_negacyclic_ntt(&mut left_hat_bitrev)?;
        Ok(Self { left_hat_bitrev })
    }

    pub fn mul(&self, rhs: &KoalaBearRing<N>) -> KoalaBearRing<N> {
        profile!("prepared_negacyclic_mul");
        let mut x = rhs.coeffs().to_vec();
        self.mul_coeffs_in_place(&mut x);
        KoalaBearRing::from_coeffs(&x)
    }

    /// In-place `left * rhs` for length-`N` coefficient buffers (negacyclic).
    pub fn mul_coeffs_in_place(&self, rhs: &mut [KoalaBear]) {
        debug_assert_eq!(rhs.len(), N);
        forward_negacyclic_ntt(rhs).expect("cached NTT tables");
        for (coeff, hat) in rhs.iter_mut().zip(self.left_hat_bitrev.iter()) {
            *coeff *= *hat;
        }
        inverse_negacyclic_ntt(rhs).expect("cached NTT tables");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Field, KOALA_BEAR_PRIME, KoalaBear, KoalaBearRing};
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn random_ring(rng: &mut StdRng) -> KoalaBearRing<512> {
        let mut c = [KoalaBear::ZERO; 512];
        for coeff in &mut c {
            *coeff = KoalaBear::new(rng.random_range(0..KOALA_BEAR_PRIME));
        }
        KoalaBearRing::from_coeffs(&c)
    }

    #[test]
    fn prepared_scalar_matches_ring_mul() {
        let mut rng = StdRng::from_os_rng();
        let left = random_ring(&mut rng);
        let prep = PreparedNegacyclicMultiplier::new(left).unwrap();
        for _ in 0..5 {
            let rhs = random_ring(&mut rng);
            assert_eq!(prep.mul(&rhs), left * rhs);
        }
    }
}
