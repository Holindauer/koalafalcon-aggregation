//! Polynomial ring arithmetic over the KoalaBear field.

use super::Field;
use crate::Error;
use rand::Rng;
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[allow(dead_code)]
pub trait CyclotomicRing<F: Field>:
    Sized
    + Eq
    + Copy
    + Clone
    + Debug
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

    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }

    fn from_coeffs(coeffs: &[F]) -> Self;

    fn to_coeffs(&self) -> &[F];

    fn random(rng: &mut impl Rng) -> Self;

    fn is_invertible(&self) -> Result<(), Error>;

    fn maybe_invert(&self) -> Option<Self>;
}
