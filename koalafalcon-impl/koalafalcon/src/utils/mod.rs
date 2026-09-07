pub mod coefficients;
pub mod double_double;
pub mod error;
pub mod fft;
pub mod fft_dd;
pub mod matrix;
pub mod ntt;

pub use error::{Error, TrapdoorError};

pub(crate) use coefficients::{center_poly_i64, i64_to_ring, l1_norm, linf_norm, squared_l2};

use crate::algebra::Field;

#[inline]
pub fn bit_reverse(mut x: usize, bits: u32) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Montgomery trick batch inversion.
pub fn batch_invert_in_place<F: Field>(xs: &mut [F]) -> Option<()> {
    let n = xs.len();
    if n == 0 {
        return Some(());
    }
    for x in xs.iter() {
        if x.is_zero() {
            return None;
        }
    }
    let mut prefix = vec![F::ONE; n];
    for i in 1..n {
        prefix[i] = prefix[i - 1] * xs[i - 1];
    }
    let mut inv_all = (prefix[n - 1] * xs[n - 1]).invert();
    for i in (0..n).rev() {
        let xi = xs[i];
        xs[i] = prefix[i] * inv_all;
        inv_all *= xi;
    }
    Some(())
}

pub(crate) fn field_from_usize<F: Field>(n: usize) -> F {
    let mut acc = F::ZERO;
    for _ in 0..n {
        acc += F::ONE;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::KoalaBear;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn bit_reverse_examples() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
        assert_eq!(bit_reverse(0b100, 3), 0b001);
        assert_eq!(bit_reverse(0b101, 3), 0b101);
        assert_eq!(bit_reverse(0b110, 3), 0b011);
        assert_eq!(bit_reverse(0b111, 3), 0b111);
    }

    #[test]
    fn batch_invert_matches_pointwise() {
        let mut rng = StdRng::from_os_rng();
        let vals: Vec<KoalaBear> = (0..64)
            .map(|_| {
                loop {
                    let x = KoalaBear::random(&mut rng);
                    if !x.is_zero() {
                        return x;
                    }
                }
            })
            .collect();
        let expected: Vec<_> = vals.iter().map(|x| x.invert()).collect();
        let mut got = vals.clone();
        batch_invert_in_place(&mut got).unwrap();
        assert_eq!(got, expected);
        let mut with_zero = vals;
        with_zero[7] = KoalaBear::ZERO;
        assert!(batch_invert_in_place(&mut with_zero).is_none());
    }
}
