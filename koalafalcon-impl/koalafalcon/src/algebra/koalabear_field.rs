//! KoalaBear field element: \(q = 2^{31} - 2^{24} + 1\).

use super::Field;
use p3_field::{Field as P3Field, PrimeCharacteristicRing, PrimeField32};
use p3_koala_bear::KoalaBear as P3KoalaBear;
use rand::Rng;
use std::{
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub const KOALA_BEAR_PRIME: u32 = (1 << 31) - (1 << 24) + 1;

/// KoalaBear element backed by Plonky3 Montgomery arithmetic.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct KoalaBear(pub P3KoalaBear);

impl KoalaBear {
    /// Construct from a canonical residue (any `u32`; reduced into Montgomery form).
    #[inline(always)]
    pub const fn new(value: u32) -> Self {
        Self(P3KoalaBear::new(value))
    }

    /// Canonical representative in `[0, q)`.
    #[inline(always)]
    pub fn as_canonical_u32(self) -> u32 {
        self.0.as_canonical_u32()
    }
}

impl Field for KoalaBear {
    const ZERO: Self = Self(P3KoalaBear::ZERO);
    const ONE: Self = Self(P3KoalaBear::ONE);
    const N_BYTES: usize = 4;

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[inline(always)]
    fn invert(&self) -> Self {
        debug_assert!(!self.is_zero(), "Cannot compute inverse of zero");
        Self(P3Field::inverse(&self.0))
    }

    fn random(rng: &mut impl Rng) -> Self {
        Self::new(rng.random_range(0..KOALA_BEAR_PRIME))
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), Self::N_BYTES);
        let mut u32_bytes = [0u8; 4];
        u32_bytes.copy_from_slice(bytes);
        // `P3KoalaBear::new` accepts any u32 and reduces via Montgomery conversion.
        Self::new(u32::from_le_bytes(u32_bytes))
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.as_canonical_u32().to_le_bytes().to_vec()
    }
}

impl Add for KoalaBear {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self::Output {
        Self(self.0 + other.0)
    }
}

impl Mul for KoalaBear {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}

impl Sub for KoalaBear {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self::Output {
        Self(self.0 - other.0)
    }
}

impl Neg for KoalaBear {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl AddAssign for KoalaBear {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl SubAssign for KoalaBear {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for KoalaBear {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl From<u8> for KoalaBear {
    #[inline(always)]
    fn from(value: u8) -> Self {
        Self::new(value as u32)
    }
}

impl From<u32> for KoalaBear {
    #[inline(always)]
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<P3KoalaBear> for KoalaBear {
    #[inline(always)]
    fn from(value: P3KoalaBear) -> Self {
        Self(value)
    }
}

impl From<KoalaBear> for P3KoalaBear {
    #[inline(always)]
    fn from(value: KoalaBear) -> Self {
        value.0
    }
}

impl Display for KoalaBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_canonical_u32())
    }
}

impl Debug for KoalaBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KoalaBear({})", self.as_canonical_u32())
    }
}

impl Hash for KoalaBear {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_canonical_u32().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    const SAMPLES: usize = 4;

    fn random_elements() -> [KoalaBear; SAMPLES] {
        let mut rng = StdRng::from_os_rng();
        std::array::from_fn(|_| KoalaBear::random(&mut rng))
    }

    fn random_nonzero_elements() -> [KoalaBear; SAMPLES] {
        let mut rng = StdRng::from_os_rng();
        std::array::from_fn(|_| {
            loop {
                let value = KoalaBear::random(&mut rng);
                if !value.is_zero() {
                    return value;
                }
            }
        })
    }

    #[test]
    fn additive_identity() {
        for a in random_elements() {
            assert_eq!(KoalaBear::ZERO + a, a);
            assert_eq!(a + KoalaBear::ZERO, a);
        }
    }

    #[test]
    fn multiplicative_identity() {
        for a in random_elements() {
            assert_eq!(KoalaBear::ONE * a, a);
            assert_eq!(a * KoalaBear::ONE, a);
        }
    }

    #[test]
    fn additive_inverse() {
        for a in random_elements() {
            assert_eq!(a + (-a), KoalaBear::ZERO);
            assert_eq!((-a) + a, KoalaBear::ZERO);
        }
    }

    #[test]
    fn multiplicative_inverse() {
        for a in random_nonzero_elements() {
            assert_eq!(a * a.invert(), KoalaBear::ONE);
            assert_eq!(a.invert() * a, KoalaBear::ONE);
        }
    }

    #[test]
    fn addition_commutative() {
        let [a, b, c, d] = random_elements();
        for (x, y) in [(a, b), (b, c), (c, d), (d, a)] {
            assert_eq!(x + y, y + x);
        }
    }

    #[test]
    fn multiplication_commutative() {
        let [a, b, c, d] = random_elements();
        for (x, y) in [(a, b), (b, c), (c, d), (d, a)] {
            assert_eq!(x * y, y * x);
        }
    }

    #[test]
    fn addition_associative() {
        let [a, b, c, d] = random_elements();
        for (x, y, z) in [(a, b, c), (b, c, d)] {
            assert_eq!((x + y) + z, x + (y + z));
        }
    }

    #[test]
    fn multiplication_associative() {
        let [a, b, c, d] = random_elements();
        for (x, y, z) in [(a, b, c), (b, c, d)] {
            assert_eq!((x * y) * z, x * (y * z));
        }
    }

    #[test]
    fn distributivity() {
        let [a, b, c, d] = random_elements();
        for (x, y, z) in [(a, b, c), (b, c, d)] {
            assert_eq!(x * (y + z), x * y + x * z);
        }
    }

    #[test]
    fn subtraction_as_addition_of_negation() {
        for a in random_elements() {
            for b in random_elements() {
                assert_eq!(a - b, a + (-b));
            }
        }
    }

    #[test]
    fn bytes_roundtrip() {
        for a in random_elements() {
            assert_eq!(KoalaBear::from_bytes(&a.to_bytes()), a);
        }
    }
}
