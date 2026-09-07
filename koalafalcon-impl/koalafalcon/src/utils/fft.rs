//! Real FFT over `R[x]/(x^n + 1)` with reusable plans and scratch.
//!
//! Transform convention (unchanged from the falcon.py port):
//! - `n` real coefficients map to `n` `Complex64` values;
//! - base case `n = 2`: `[f0, f1] → [f0+i f1, f0−i f1]`.
//!
//! Used by the NTRUGen GS filter and Fourier DD (via `falcon_roots`).

use crate::utils::Error;
use num_complex::Complex64;
use std::sync::OnceLock;

/// Reusable twiddle tables for Falcon-style split/merge FFT.
#[derive(Debug, Clone)]
pub struct FftPlan {
    n: usize,
    /// For each `m ∈ {2,4,...,n}`, `merge_twiddles[log2(m)-1][i] = roots(m)[2i]`.
    merge_twiddles: Vec<Vec<Complex64>>,
}

/// Reusable buffers for planned transforms (`O(n log n)` complex + real workspace).
#[derive(Debug, Default, Clone)]
pub struct FftScratch {
    pub real: Vec<f64>,
    pub complex: Vec<Complex64>,
}

impl FftScratch {
    pub fn reserve(&mut self, n: usize) {
        let levels = (n as u64).ilog2() as usize + 1;
        let need_r = n * levels;
        let need_c = n * levels;
        if self.real.len() < need_r {
            self.real.resize(need_r, 0.0);
        }
        if self.complex.len() < need_c {
            self.complex.resize(need_c, Complex64::new(0.0, 0.0));
        }
    }
}

/// Falcon.py / `generate_constants.sage` root order for `x^n + 1`:
/// start from `{i, −i}` and replace each `r` by `{√r, −√r}` (principal square root).
/// Sequential odd roots of unity are *not* isomorphic under this split/merge FFT.
pub fn falcon_roots(n: usize) -> Vec<Complex64> {
    debug_assert!(n >= 2 && n.is_power_of_two());
    if n == 2 {
        return vec![Complex64::new(0.0, 1.0), Complex64::new(0.0, -1.0)];
    }
    let parent = falcon_roots(n / 2);
    let mut out = Vec::with_capacity(n);
    for r in parent {
        let s = r.sqrt();
        out.push(s);
        out.push(-s);
    }
    out
}

impl FftPlan {
    pub fn new(n: usize) -> Result<Self, Error> {
        if n < 2 || !n.is_power_of_two() {
            return Err(Error::UnsupportedRingDimension(n));
        }
        let log_n = (n as u64).ilog2() as usize;
        let mut merge_twiddles = Vec::with_capacity(log_n);
        let mut m = 2usize;
        while m <= n {
            let roots_m = falcon_roots(m);
            let half = m / 2;
            let mut tw = Vec::with_capacity(half);
            for i in 0..half {
                tw.push(roots_m[2 * i]);
            }
            merge_twiddles.push(tw);
            m <<= 1;
        }
        Ok(Self { n, merge_twiddles })
    }

    fn twiddles_for_size(&self, m: usize) -> &[Complex64] {
        let idx = (m as u64).ilog2() as usize - 1;
        &self.merge_twiddles[idx]
    }

    pub fn split_fft_into(
        &self,
        values: &[Complex64],
        left: &mut [Complex64],
        right: &mut [Complex64],
    ) {
        let m = values.len();
        debug_assert_eq!(left.len(), m / 2);
        debug_assert_eq!(right.len(), m / 2);
        // Twiddles come from the plan covering this recursion size when `m <= self.n`.
        let tw = if m <= self.n && m >= 2 {
            self.twiddles_for_size(m)
        } else {
            // Ad-hoc for tests at sizes below a larger plan: rebuild locally.
            // Prefer constructing a matching plan for production sizes.
            return self.split_fft_into_fallback(values, left, right);
        };
        for i in 0..(m / 2) {
            left[i] = 0.5 * (values[2 * i] + values[2 * i + 1]);
            right[i] = 0.5 * (values[2 * i] - values[2 * i + 1]) * tw[i].conj();
        }
    }

    fn split_fft_into_fallback(
        &self,
        values: &[Complex64],
        left: &mut [Complex64],
        right: &mut [Complex64],
    ) {
        let m = values.len();
        let roots_m = falcon_roots(m);
        for i in 0..(m / 2) {
            left[i] = 0.5 * (values[2 * i] + values[2 * i + 1]);
            right[i] = 0.5 * (values[2 * i] - values[2 * i + 1]) * roots_m[2 * i].conj();
        }
    }

    pub fn merge_fft_into(&self, left: &[Complex64], right: &[Complex64], out: &mut [Complex64]) {
        let half = left.len();
        let m = 2 * half;
        debug_assert_eq!(right.len(), half);
        debug_assert_eq!(out.len(), m);
        let tw = if m <= self.n {
            self.twiddles_for_size(m)
        } else {
            return self.merge_fft_into_fallback(left, right, out);
        };
        for i in 0..half {
            let t = tw[i] * right[i];
            out[2 * i] = left[i] + t;
            out[2 * i + 1] = left[i] - t;
        }
    }

    fn merge_fft_into_fallback(
        &self,
        left: &[Complex64],
        right: &[Complex64],
        out: &mut [Complex64],
    ) {
        let half = left.len();
        let m = 2 * half;
        let roots_m = falcon_roots(m);
        for i in 0..half {
            let t = roots_m[2 * i] * right[i];
            out[2 * i] = left[i] + t;
            out[2 * i + 1] = left[i] - t;
        }
    }

    fn fft_rec(
        &self,
        f: &[f64],
        out: &mut [Complex64],
        scratch: &mut FftScratch,
        real_base: usize,
        complex_base: usize,
    ) {
        let n = f.len();
        if n == 2 {
            out[0] = Complex64::new(f[0], f[1]);
            out[1] = Complex64::new(f[0], -f[1]);
            return;
        }
        let half = n / 2;
        // Pack even/odd into scratch.real[real_base ..]
        let even_off = real_base;
        let odd_off = real_base + half;
        for i in 0..half {
            scratch.real[even_off + i] = f[2 * i];
            scratch.real[odd_off + i] = f[2 * i + 1];
        }
        let left_c = complex_base;
        let right_c = complex_base + half;
        let even: Vec<f64> = scratch.real[even_off..even_off + half].to_vec();
        let odd: Vec<f64> = scratch.real[odd_off..odd_off + half].to_vec();
        let child_real = real_base + n;
        let child_complex = complex_base + n;
        self.fft_rec_indexed(&even, left_c, half, scratch, child_real, child_complex);
        self.fft_rec_indexed(&odd, right_c, half, scratch, child_real, child_complex);
        let left = scratch.complex[left_c..left_c + half].to_vec();
        let right = scratch.complex[right_c..right_c + half].to_vec();
        self.merge_fft_into(&left, &right, out);
    }

    fn fft_rec_indexed(
        &self,
        f: &[f64],
        out_off: usize,
        n: usize,
        scratch: &mut FftScratch,
        real_base: usize,
        complex_base: usize,
    ) {
        if n == 2 {
            scratch.complex[out_off] = Complex64::new(f[0], f[1]);
            scratch.complex[out_off + 1] = Complex64::new(f[0], -f[1]);
            return;
        }
        let half = n / 2;
        let even_off = real_base;
        let odd_off = real_base + half;
        for i in 0..half {
            scratch.real[even_off + i] = f[2 * i];
            scratch.real[odd_off + i] = f[2 * i + 1];
        }
        let left_c = complex_base;
        let right_c = complex_base + half;
        let even: Vec<f64> = scratch.real[even_off..even_off + half].to_vec();
        let odd: Vec<f64> = scratch.real[odd_off..odd_off + half].to_vec();
        let child_real = real_base + n;
        let child_complex = complex_base + n;
        self.fft_rec_indexed(&even, left_c, half, scratch, child_real, child_complex);
        self.fft_rec_indexed(&odd, right_c, half, scratch, child_real, child_complex);
        let left = scratch.complex[left_c..left_c + half].to_vec();
        let right = scratch.complex[right_c..right_c + half].to_vec();
        let mut out_tmp = vec![Complex64::new(0.0, 0.0); n];
        self.merge_fft_into(&left, &right, &mut out_tmp);
        scratch.complex[out_off..out_off + n].copy_from_slice(&out_tmp);
    }

    pub fn fft_into(&self, coeffs: &[f64], out: &mut [Complex64], scratch: &mut FftScratch) {
        assert_eq!(coeffs.len(), self.n);
        assert_eq!(out.len(), self.n);
        scratch.reserve(self.n);
        self.fft_rec(coeffs, out, scratch, 0, 0);
    }

    fn ifft_rec(
        &self,
        f_fft: &[Complex64],
        out: &mut [f64],
        scratch: &mut FftScratch,
        complex_base: usize,
        real_base: usize,
    ) {
        let n = f_fft.len();
        if n == 2 {
            out[0] = f_fft[0].re;
            out[1] = f_fft[0].im;
            return;
        }
        let half = n / 2;
        let left_c = complex_base;
        let right_c = complex_base + half;
        {
            let mut l = vec![Complex64::new(0.0, 0.0); half];
            let mut r = vec![Complex64::new(0.0, 0.0); half];
            self.split_fft_into(f_fft, &mut l, &mut r);
            scratch.complex[left_c..left_c + half].copy_from_slice(&l);
            scratch.complex[right_c..right_c + half].copy_from_slice(&r);
        }
        let left = scratch.complex[left_c..left_c + half].to_vec();
        let right = scratch.complex[right_c..right_c + half].to_vec();
        let child_c = complex_base + n;
        let child_r = real_base + n;
        let mut left_r = vec![0.0; half];
        let mut right_r = vec![0.0; half];
        self.ifft_rec(&left, &mut left_r, scratch, child_c, child_r);
        self.ifft_rec(&right, &mut right_r, scratch, child_c, child_r);
        for i in 0..half {
            out[2 * i] = left_r[i];
            out[2 * i + 1] = right_r[i];
        }
    }

    pub fn ifft_into(&self, values: &[Complex64], out: &mut [f64], scratch: &mut FftScratch) {
        assert_eq!(values.len(), self.n);
        assert_eq!(out.len(), self.n);
        scratch.reserve(self.n);
        self.ifft_rec(values, out, scratch, 0, 0);
    }
}

static PLAN_512: OnceLock<FftPlan> = OnceLock::new();
static PLAN_1024: OnceLock<FftPlan> = OnceLock::new();

pub fn koala_fft_plan(n: usize) -> Result<&'static FftPlan, Error> {
    match n {
        512 => Ok(PLAN_512.get_or_init(|| FftPlan::new(512).expect("fft 512"))),
        1024 => Ok(PLAN_1024.get_or_init(|| FftPlan::new(1024).expect("fft 1024"))),
        _ => Err(Error::UnsupportedRingDimension(n)),
    }
}

/// Compatibility FFT (allocates one plan + scratch for non-cached sizes).
pub fn fft(f: &[f64]) -> Vec<Complex64> {
    let n = f.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    if let Ok(plan) = koala_fft_plan(n) {
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        let mut scratch = FftScratch::default();
        plan.fft_into(f, &mut out, &mut scratch);
        return out;
    }
    let plan = FftPlan::new(n).expect("fft plan");
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    let mut scratch = FftScratch::default();
    plan.fft_into(f, &mut out, &mut scratch);
    out
}

pub fn ifft(f_fft: &[Complex64]) -> Vec<f64> {
    let n = f_fft.len();
    debug_assert!(n.is_power_of_two() && n >= 2);
    if let Ok(plan) = koala_fft_plan(n) {
        let mut out = vec![0.0; n];
        let mut scratch = FftScratch::default();
        plan.ifft_into(f_fft, &mut out, &mut scratch);
        return out;
    }
    let plan = FftPlan::new(n).expect("fft plan");
    let mut out = vec![0.0; n];
    let mut scratch = FftScratch::default();
    plan.ifft_into(f_fft, &mut out, &mut scratch);
    out
}

pub fn mul_fft(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); a.len()];
    mul_fft_into(a, b, &mut out);
    out
}

pub fn mul_fft_into(a: &[Complex64], b: &[Complex64], out: &mut [Complex64]) {
    for ((o, x), y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x * y;
    }
}

pub fn div_fft(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); a.len()];
    div_fft_into(a, b, &mut out);
    out
}

pub fn div_fft_into(a: &[Complex64], b: &[Complex64], out: &mut [Complex64]) {
    for ((o, x), y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x / y;
    }
}

pub fn adj_fft(a: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); a.len()];
    adj_fft_into(a, &mut out);
    out
}

pub fn adj_fft_into(a: &[Complex64], out: &mut [Complex64]) {
    for (o, x) in out.iter_mut().zip(a.iter()) {
        *o = x.conj();
    }
}

pub fn mul_poly(a: &[f64], b: &[f64]) -> Vec<f64> {
    ifft(&mul_fft(&fft(a), &fft(b)))
}

pub fn div_poly(a: &[f64], b: &[f64]) -> Vec<f64> {
    ifft(&div_fft(&fft(a), &fft(b)))
}

pub fn adj_poly(a: &[f64]) -> Vec<f64> {
    ifft(&adj_fft(&fft(a)))
}

pub fn add_poly(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[cfg(test)]
mod legacy {
    #![allow(dead_code)]
    use super::falcon_roots;
    use num_complex::Complex64;

    fn roots(n: usize) -> Vec<Complex64> {
        falcon_roots(n)
    }

    fn split_coeff(f: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n2 = f.len() / 2;
        let mut f0 = Vec::with_capacity(n2);
        let mut f1 = Vec::with_capacity(n2);
        for i in 0..n2 {
            f0.push(f[2 * i]);
            f1.push(f[2 * i + 1]);
        }
        (f0, f1)
    }

    fn merge_coeff(f0: &[f64], f1: &[f64]) -> Vec<f64> {
        let n2 = f0.len();
        let mut f = vec![0.0; 2 * n2];
        for i in 0..n2 {
            f[2 * i] = f0[i];
            f[2 * i + 1] = f1[i];
        }
        f
    }

    fn split_fft(f: &[Complex64]) -> (Vec<Complex64>, Vec<Complex64>) {
        let n = f.len();
        let w = roots(n);
        let mut f0 = vec![Complex64::new(0.0, 0.0); n / 2];
        let mut f1 = vec![Complex64::new(0.0, 0.0); n / 2];
        for i in 0..(n / 2) {
            f0[i] = 0.5 * (f[2 * i] + f[2 * i + 1]);
            f1[i] = 0.5 * (f[2 * i] - f[2 * i + 1]) * w[2 * i].conj();
        }
        (f0, f1)
    }

    fn merge_fft(f0: &[Complex64], f1: &[Complex64]) -> Vec<Complex64> {
        let n = 2 * f0.len();
        let w = roots(n);
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..(n / 2) {
            let t = w[2 * i] * f1[i];
            out[2 * i] = f0[i] + t;
            out[2 * i + 1] = f0[i] - t;
        }
        out
    }

    pub fn fft(f: &[f64]) -> Vec<Complex64> {
        let n = f.len();
        if n > 2 {
            let (f0, f1) = split_coeff(f);
            merge_fft(&fft(&f0), &fft(&f1))
        } else {
            vec![Complex64::new(f[0], f[1]), Complex64::new(f[0], -f[1])]
        }
    }

    pub fn ifft(f_fft: &[Complex64]) -> Vec<f64> {
        let n = f_fft.len();
        if n > 2 {
            let (f0, f1) = split_fft(f_fft);
            merge_coeff(&ifft(&f0), &ifft(&f1))
        } else {
            vec![f_fft[0].re, f_fft[0].im]
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    fn max_abs_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn fft_roundtrip() {
        let f = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let back = ifft(&fft(&f));
        for (a, b) in f.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn planned_matches_legacy_sizes() {
        let mut rng = StdRng::from_os_rng();
        for &n in &[2usize, 4, 8, 16, 32, 64, 512] {
            let f: Vec<f64> = (0..n).map(|_| rng.random_range(-10.0..10.0)).collect();
            let legacy = legacy::fft(&f);
            let planned = fft(&f);
            assert!(
                max_abs_diff(&legacy, &planned) < 1e-9 * (n as f64).sqrt(),
                "n={n}"
            );
            let back = ifft(&planned);
            for (a, b) in f.iter().zip(back.iter()) {
                assert!((a - b).abs() < 1e-9 * (n as f64).sqrt());
            }
        }
    }

    #[test]
    fn split_merge_roundtrip() {
        let plan = FftPlan::new(16).unwrap();
        let mut rng = StdRng::from_os_rng();
        let f: Vec<f64> = (0..16).map(|_| rng.random_range(-1.0..1.0)).collect();
        let vals = fft(&f);
        let mut left = vec![Complex64::new(0.0, 0.0); 8];
        let mut right = vec![Complex64::new(0.0, 0.0); 8];
        plan.split_fft_into(&vals, &mut left, &mut right);
        let mut merged = vec![Complex64::new(0.0, 0.0); 16];
        plan.merge_fft_into(&left, &right, &mut merged);
        assert!(max_abs_diff(&vals, &merged) < 1e-10);
    }

    #[test]
    fn cached_plan_ptr_stable_for_repeated_fft() {
        let p0 = koala_fft_plan(512).unwrap() as *const _;
        let mut scratch = FftScratch::default();
        let plan = koala_fft_plan(512).unwrap();
        let f = vec![1.0; 512];
        let mut out = vec![Complex64::new(0.0, 0.0); 512];
        for _ in 0..10 {
            plan.fft_into(&f, &mut out, &mut scratch);
        }
        let p1 = koala_fft_plan(512).unwrap() as *const _;
        assert_eq!(p0, p1);
        let cap = scratch.complex.capacity();
        plan.fft_into(&f, &mut out, &mut scratch);
        assert_eq!(scratch.complex.capacity(), cap);
    }

    #[test]
    fn pointwise_into_matches() {
        let a = fft(&[1.0, 2.0, 3.0, 4.0]);
        let b = fft(&[4.0, 3.0, 2.0, 1.0]);
        let mut out = vec![Complex64::new(0.0, 0.0); 4];
        mul_fft_into(&a, &b, &mut out);
        assert!(max_abs_diff(&out, &mul_fft(&a, &b)) < 1e-15);
    }

    #[test]
    fn convolution_matches_negacyclic() {
        let mut rng = StdRng::from_os_rng();
        for &n in &[8usize, 16, 64, 512] {
            let a: Vec<f64> = (0..n).map(|_| rng.random_range(-50.0..50.0)).collect();
            let b: Vec<f64> = (0..n).map(|_| rng.random_range(-50.0..50.0)).collect();
            let mut expect = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    let k = i + j;
                    let p = a[i] * b[j];
                    if k < n {
                        expect[k] += p;
                    } else {
                        expect[k - n] -= p;
                    }
                }
            }
            let got = ifft(&mul_fft(&fft(&a), &fft(&b)));
            let err = a
                .iter()
                .zip(got.iter().zip(expect.iter()))
                .map(|(_, (g, e))| (g - e).abs())
                .fold(0.0, f64::max);
            assert!(err < 1e-6 * (n as f64), "n={n} err={err}");
        }
    }
}
