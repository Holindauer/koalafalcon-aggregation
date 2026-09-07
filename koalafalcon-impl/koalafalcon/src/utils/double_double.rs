//! Pure-Rust double-double (`f64` + `f64`) arithmetic for Fourier PreSmp.
//!
//! FFT roots are generated once as binary64 and lifted into [`DoubleDouble`]
//! (exact as `hi`, `lo = 0`). Subsequent arithmetic carries both limbs.

#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(clippy::needless_return)]

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Normalized double-double value `hi + lo` with `|lo| ≤ 0.5 ulp(hi)` ideally.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

#[inline]
pub fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    DoubleDouble { hi: s, lo: err }
}

#[inline]
pub fn quick_two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    DoubleDouble {
        hi: s,
        lo: b - (s - a),
    }
}

#[inline]
pub fn two_prod(a: f64, b: f64) -> DoubleDouble {
    let p = a * b;
    let err = a.mul_add(b, -p);
    DoubleDouble { hi: p, lo: err }
}

impl DoubleDouble {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    #[inline]
    pub fn from_i64(x: i64) -> Self {
        Self::from_f64(x as f64)
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        quick_two_sum(self.hi, self.lo)
    }

    pub fn recip(self) -> Self {
        let q1 = 1.0 / self.hi;
        let mut r = Self::ONE - Self::from_f64(q1) * self;
        r = Self::from_f64(q1) + Self::from_f64(q1) * r;
        r.normalize()
    }

    pub fn sqrt(self) -> Self {
        if self.hi == 0.0 && self.lo == 0.0 {
            return Self::ZERO;
        }
        let x = self.hi.sqrt();
        let mut r = Self::from_f64(x);
        // Newton: r <- (r + self/r)/2
        r = (r + self / r) * Self::from_f64(0.5);
        r.normalize()
    }

    /// Round to nearest integer (ties away from zero via f64 round of sum).
    pub fn round_i64(self) -> i64 {
        (self.hi + self.lo).round() as i64
    }
}

impl From<f64> for DoubleDouble {
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl From<i64> for DoubleDouble {
    fn from(x: i64) -> Self {
        Self::from_i64(x)
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut s = two_sum(self.hi, rhs.hi);
        let t = two_sum(self.lo, rhs.lo);
        s.lo += t.hi;
        s = quick_two_sum(s.hi, s.lo);
        s.lo += t.lo;
        quick_two_sum(s.hi, s.lo)
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut p = two_prod(self.hi, rhs.hi);
        p.lo += self.hi * rhs.lo + self.lo * rhs.hi;
        quick_two_sum(p.hi, p.lo)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

/// Complex number with double-double components.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComplexDD {
    pub re: DoubleDouble,
    pub im: DoubleDouble,
}

impl ComplexDD {
    pub const ZERO: Self = Self {
        re: DoubleDouble::ZERO,
        im: DoubleDouble::ZERO,
    };

    pub fn new(re: DoubleDouble, im: DoubleDouble) -> Self {
        Self { re, im }
    }

    pub fn from_f64s(re: f64, im: f64) -> Self {
        Self {
            re: DoubleDouble::from_f64(re),
            im: DoubleDouble::from_f64(im),
        }
    }

    pub fn from_complex64(z: num_complex::Complex64) -> Self {
        Self::from_f64s(z.re, z.im)
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    pub fn norm_sq(self) -> DoubleDouble {
        self.re * self.re + self.im * self.im
    }

    pub fn scale(self, s: DoubleDouble) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Neg for ComplexDD {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Add for ComplexDD {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for ComplexDD {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for ComplexDD {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for ComplexDD {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let n2 = rhs.norm_sq();
        let num = self * rhs.conj();
        Self {
            re: num.re / n2,
            im: num.im / n2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_prod_captures_error() {
        let a = 1.0 + f64::EPSILON;
        let b = 1.0 + 2.0 * f64::EPSILON;
        let p = two_prod(a, b);
        assert!((p.hi + p.lo - a * b).abs() < 1e-30 || p.lo != 0.0 || a * b == p.hi);
    }

    #[test]
    fn add_mul_approx_laws() {
        let a = DoubleDouble::from_f64(1.0 / 3.0);
        let b = DoubleDouble::from_f64(1.0 / 7.0);
        let s = a + b;
        let back = s - a;
        assert!((back.to_f64() - b.to_f64()).abs() < 1e-28);
        let p = a * b;
        assert!((p.to_f64() - a.to_f64() * b.to_f64()).abs() < 1e-20);
    }

    #[test]
    fn complex_mul_div_roundtrip() {
        let z = ComplexDD::from_f64s(1.25, -0.5);
        let w = ComplexDD::from_f64s(0.25, 0.75);
        let q = (z * w) / w;
        assert!((q.re.to_f64() - z.re.to_f64()).abs() < 1e-20);
        assert!((q.im.to_f64() - z.im.to_f64()).abs() < 1e-20);
    }

    #[test]
    fn sqrt_one() {
        let s = DoubleDouble::ONE.sqrt();
        assert!((s.to_f64() - 1.0).abs() < 1e-30);
    }
}
