//! Prime field trait used by KoalaFalcon algebra and NTT.

use rand::Rng;
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait Field:
    Sized
    + Eq
    + Copy
    + Clone
    + Debug
    + Hash
    + Send
    + Sync
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
{
    const ZERO: Self;
    const ONE: Self;
    const N_BYTES: usize;

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn invert(&self) -> Self;

    fn random(rng: &mut impl Rng) -> Self;

    fn from_bytes(bytes: &[u8]) -> Self;

    fn to_bytes(&self) -> Vec<u8>;

    /// Repeated squaring.
    fn power(&self, mut exp: u32) -> Self {
        if exp == 0 {
            return Self::ONE;
        }
        if exp == 1 {
            return *self;
        }

        let mut result = Self::ONE;
        let mut base = *self;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }

        result
    }
}
