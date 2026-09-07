use super::{Field, koalabear_field::KoalaBear, ring::CyclotomicRing};
use crate::utils::Error;
use crate::utils::ntt::{negacyclic_invert, negacyclic_mul};
use rand::Rng;
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct KoalaBearRing<const N: usize> {
    coeffs: [KoalaBear; N],
}

impl<const N: usize> CyclotomicRing<KoalaBear> for KoalaBearRing<N> {
    const ZERO: Self = Self {
        coeffs: [KoalaBear::ZERO; N],
    };

    const ONE: Self = {
        let mut coeffs = [KoalaBear::ZERO; N];
        coeffs[0] = KoalaBear::ONE;
        Self { coeffs }
    };

    fn from_coeffs(coeffs: &[KoalaBear]) -> Self {
        debug_assert_eq!(
            coeffs.len(),
            N,
            "expected {N} coefficients, got {}",
            coeffs.len()
        );

        let mut ring_coeffs = [KoalaBear::ZERO; N];
        ring_coeffs.copy_from_slice(coeffs);
        Self {
            coeffs: ring_coeffs,
        }
    }

    fn to_coeffs(&self) -> &[KoalaBear] {
        self.coeffs()
    }

    fn random(rng: &mut impl Rng) -> Self {
        Self {
            coeffs: std::array::from_fn(|_| KoalaBear::random(rng)),
        }
    }

    fn is_invertible(&self) -> Result<(), Error> {
        match self.maybe_invert() {
            Some(_) => Ok(()),
            None => Err(Error::NotInvertible),
        }
    }

    fn maybe_invert(&self) -> Option<Self> {
        let mut out = [KoalaBear::ZERO; N];
        let mut scratch = [KoalaBear::ZERO; N];
        negacyclic_invert(self.coeffs.as_slice(), &mut out, &mut scratch)?;
        Some(Self { coeffs: out })
    }
}

impl<const N: usize> KoalaBearRing<N> {
    pub fn coeffs(&self) -> &[KoalaBear; N] {
        &self.coeffs
    }

    /// Byte length required by [`Self::from_bytes`]: `N` coefficients × [`KoalaBear::N_BYTES`].
    pub const fn required_input_bytes() -> usize {
        N * KoalaBear::N_BYTES
    }

    /// Decode a ring element from the concatenation of `N` little-endian KoalaBear encodings.
    ///
    /// Each coefficient is parsed with [`KoalaBear::from_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let expected = Self::required_input_bytes();
        if bytes.len() != expected {
            return Err(Error::InsufficientInputBytes {
                expected,
                got: bytes.len(),
            });
        }

        let mut coeffs = [KoalaBear::ZERO; N];
        for (coeff, chunk) in coeffs
            .iter_mut()
            .zip(bytes.chunks_exact(KoalaBear::N_BYTES))
        {
            *coeff = KoalaBear::from_bytes(chunk);
        }
        Ok(Self { coeffs })
    }
}

impl<const N: usize> Add for KoalaBearRing<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            coeffs: std::array::from_fn(|i| self.coeffs[i] + other.coeffs[i]),
        }
    }
}

impl<const N: usize> Sub for KoalaBearRing<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            coeffs: std::array::from_fn(|i| self.coeffs[i] - other.coeffs[i]),
        }
    }
}

impl<const N: usize> Neg for KoalaBearRing<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            coeffs: std::array::from_fn(|i| -self.coeffs[i]),
        }
    }
}

impl<const N: usize> Mul for KoalaBearRing<N> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        debug_assert!(
            N.is_power_of_two(),
            "ring multiplication requires N to be a power of two"
        );

        let mut out = [KoalaBear::ZERO; N];
        let mut scratch_a = [KoalaBear::ZERO; N];
        let mut scratch_b = [KoalaBear::ZERO; N];
        negacyclic_mul(
            self.coeffs.as_slice(),
            other.coeffs.as_slice(),
            &mut out,
            &mut scratch_a,
            &mut scratch_b,
        )
        .expect("ring multiplication requires N ∈ {512, 1024}");
        Self { coeffs: out }
    }
}

impl<const N: usize> AddAssign for KoalaBearRing<N> {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl<const N: usize> SubAssign for KoalaBearRing<N> {
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl<const N: usize> MulAssign for KoalaBearRing<N> {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl<const N: usize> Debug for KoalaBearRing<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.coeffs.iter()).finish()
    }
}

impl<const N: usize> Display for KoalaBearRing<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop, clippy::assign_op_pattern)]
    use super::*;
    use crate::algebra::Field;
    use rand::{TryRngCore, rngs::OsRng};

    const SAMPLES: usize = 3;

    fn random_ring_elements() -> [KoalaBearRing<512>; SAMPLES] {
        let mut os_rng = OsRng;
        let mut rng = os_rng.unwrap_mut();
        std::array::from_fn(|_| KoalaBearRing::<512>::random(&mut rng))
    }

    #[test]
    fn from_coeffs_roundtrip() {
        let mut coeffs = [KoalaBear::ZERO; 512];
        coeffs[0] = KoalaBear::new(1);
        coeffs[1] = KoalaBear::new(2);
        coeffs[2] = KoalaBear::new(3);
        coeffs[3] = KoalaBear::new(4);
        let element = KoalaBearRing::<512>::from_coeffs(&coeffs);
        assert_eq!(element.coeffs(), &coeffs);
    }

    #[test]
    fn zero_and_one_constants() {
        assert!(KoalaBearRing::<512>::ZERO.is_zero());
        assert_eq!(KoalaBearRing::<512>::ZERO.coeffs(), &[KoalaBear::ZERO; 512]);

        let mut expected_one = [KoalaBear::ZERO; 512];
        expected_one[0] = KoalaBear::ONE;
        assert_eq!(KoalaBearRing::<512>::ONE.coeffs(), &expected_one);
    }

    #[test]
    fn random_samples_coefficients() {
        let mut os_rng = OsRng;
        let mut rng = os_rng.unwrap_mut();
        let element = KoalaBearRing::<512>::random(&mut rng);
        assert_eq!(element.coeffs().len(), 512);
    }

    #[test]
    fn random_elements_differ() {
        let mut os_rng = OsRng;
        let mut rng = os_rng.unwrap_mut();
        let a = KoalaBearRing::<512>::random(&mut rng);
        let b = KoalaBearRing::<512>::random(&mut rng);
        assert_ne!(a, b);
    }

    #[test]
    fn additive_identity() {
        for a in random_ring_elements() {
            assert_eq!(KoalaBearRing::<512>::ZERO + a, a);
            assert_eq!(a + KoalaBearRing::<512>::ZERO, a);
        }
    }

    #[test]
    fn additive_inverse() {
        for a in random_ring_elements() {
            assert_eq!(a + (-a), KoalaBearRing::<512>::ZERO);
            assert_eq!((-a) + a, KoalaBearRing::<512>::ZERO);
        }
    }

    #[test]
    fn addition_commutative() {
        let [a, b, c] = random_ring_elements();
        for (x, y) in [(a, b), (b, c), (c, a)] {
            assert_eq!(x + y, y + x);
        }
    }

    #[test]
    fn addition_associative() {
        let [a, b, c] = random_ring_elements();
        assert_eq!((a + b) + c, a + (b + c));
    }

    #[test]
    fn subtraction_as_addition_of_negation() {
        for a in random_ring_elements() {
            for b in random_ring_elements() {
                assert_eq!(a - b, a + (-b));
            }
        }
    }

    #[test]
    fn add_assign() {
        let [a, b, _] = random_ring_elements();
        let mut assigned = a;
        assigned += b;
        assert_eq!(assigned, a + b);
    }

    #[test]
    fn sub_assign() {
        let [a, b, _] = random_ring_elements();
        let mut assigned = a;
        assigned -= b;
        assert_eq!(assigned, a - b);
    }

    #[test]
    fn multiplication_matches_naive_negacyclic() {
        let mut a_coeffs = [KoalaBear::ZERO; 512];
        let mut b_coeffs = [KoalaBear::ZERO; 512];
        a_coeffs[0] = KoalaBear::new(1);
        a_coeffs[1] = KoalaBear::new(2);
        a_coeffs[2] = KoalaBear::new(3);
        a_coeffs[3] = KoalaBear::new(4);
        b_coeffs[0] = KoalaBear::new(5);
        b_coeffs[1] = KoalaBear::new(6);
        b_coeffs[2] = KoalaBear::new(7);
        b_coeffs[3] = KoalaBear::new(8);

        let a = KoalaBearRing::<512>::from_coeffs(&a_coeffs);
        let b = KoalaBearRing::<512>::from_coeffs(&b_coeffs);
        let product = a * b;

        let mut expected = [KoalaBear::ZERO; 512];
        let ac = a.coeffs();
        let bc = b.coeffs();
        for i in 0..512 {
            for j in 0..512 {
                let term = ac[i] * bc[j];
                let index = i + j;
                if index < 512 {
                    expected[index] = expected[index] + term;
                } else {
                    expected[index - 512] = expected[index - 512] - term;
                }
            }
        }

        assert_eq!(product.coeffs(), &expected);
    }

    #[test]
    fn multiplicative_identity() {
        for a in random_ring_elements() {
            assert_eq!(KoalaBearRing::<512>::ONE * a, a);
            assert_eq!(a * KoalaBearRing::<512>::ONE, a);
        }
    }

    #[test]
    fn multiplication_matches_neg_one_skew_circulant_matrix() {
        use crate::utils::matrix::{matmul, neg_one_skew_circulant_matrix};

        let [a, b, _] = random_ring_elements();
        let product = a * b;

        let matrix = neg_one_skew_circulant_matrix(a.to_coeffs(), 512);
        let expected = matmul(&matrix, b.to_coeffs(), 512, 512, 1);

        assert_eq!(product.to_coeffs(), expected.as_slice());
    }

    #[test]
    fn ring_invert_roundtrip() {
        for a in random_ring_elements() {
            if a.coeffs().iter().all(|c| c.is_zero()) {
                continue;
            }
            let inv = a
                .maybe_invert()
                .expect("random element should be invertible w.h.p.");
            assert!(a.is_invertible().is_ok());
            assert_eq!(a * inv, KoalaBearRing::<512>::ONE);
            assert_eq!(inv * a, KoalaBearRing::<512>::ONE);
        }
        assert!(KoalaBearRing::<512>::ZERO.maybe_invert().is_none());
        assert!(matches!(
            KoalaBearRing::<512>::ZERO.is_invertible(),
            Err(Error::NotInvertible)
        ));
    }
}
