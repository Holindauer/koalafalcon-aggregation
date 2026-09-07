//! Double-double FFT plan mirroring [`crate::utils::fft::FftPlan`].
//!
//! Roots are binary64 Falcon √-tree values lifted into [`DoubleDouble`] (`lo = 0`).
//! Recursion uses [`FftScratchDd`] only — no per-level heap allocation.

#![allow(clippy::manual_memcpy, clippy::needless_range_loop)]

use crate::utils::Error;
use crate::utils::double_double::{ComplexDD, DoubleDouble};

#[derive(Debug, Clone)]
pub struct FftPlanDd {
    n: usize,
    /// For each `m ∈ {2,4,...,n}`, merge twiddles `roots(m)[2i]`.
    merge_twiddles: Vec<Vec<ComplexDD>>,
}

#[derive(Debug, Default, Clone)]
pub struct FftScratchDd {
    pub real: Vec<DoubleDouble>,
    pub complex: Vec<ComplexDD>,
}

impl FftScratchDd {
    pub fn reserve(&mut self, n: usize) {
        // Per recursion level: n reals (even/odd) + n complexes (left/right), × log₂n.
        let levels = (n as u64).ilog2() as usize + 2;
        let need = n * levels;
        if self.real.len() < need {
            self.real.resize(need, DoubleDouble::ZERO);
        }
        if self.complex.len() < need {
            self.complex.resize(need, ComplexDD::ZERO);
        }
    }
}

impl FftPlanDd {
    pub fn new(n: usize) -> Result<Self, Error> {
        if n < 2 || !n.is_power_of_two() {
            return Err(Error::UnsupportedRingDimension(n));
        }
        let log_n = (n as u64).ilog2() as usize;
        let mut merge_twiddles = Vec::with_capacity(log_n);
        let mut m = 2usize;
        while m <= n {
            let roots_m = crate::utils::fft::falcon_roots(m);
            let half = m / 2;
            let mut tw = Vec::with_capacity(half);
            for i in 0..half {
                tw.push(ComplexDD::from_complex64(roots_m[2 * i]));
            }
            merge_twiddles.push(tw);
            m <<= 1;
        }
        Ok(Self { n, merge_twiddles })
    }

    fn twiddles_for_size(&self, m: usize) -> &[ComplexDD] {
        let idx = (m as u64).ilog2() as usize - 1;
        &self.merge_twiddles[idx]
    }

    pub fn split_fft_into(
        &self,
        values: &[ComplexDD],
        left: &mut [ComplexDD],
        right: &mut [ComplexDD],
    ) {
        let m = values.len();
        debug_assert_eq!(left.len(), m / 2);
        debug_assert_eq!(right.len(), m / 2);
        let half = m / 2;
        let tw = self.twiddles_for_size(m);
        let half_dd = DoubleDouble::from_f64(0.5);
        for i in 0..half {
            let f0 = values[2 * i];
            let f1 = values[2 * i + 1];
            left[i] = (f0 + f1).scale(half_dd);
            let diff = (f0 - f1).scale(half_dd);
            right[i] = diff * tw[i].conj();
        }
    }

    #[cfg(test)]
    pub fn merge_fft_into(&self, left: &[ComplexDD], right: &[ComplexDD], out: &mut [ComplexDD]) {
        let half = left.len();
        debug_assert_eq!(right.len(), half);
        debug_assert_eq!(out.len(), 2 * half);
        let m = 2 * half;
        let tw = self.twiddles_for_size(m);
        for i in 0..half {
            let t = tw[i] * right[i];
            out[2 * i] = left[i] + t;
            out[2 * i + 1] = left[i] - t;
        }
    }

    /// Split `buf[src..src+n]` → `buf[left..]` / `buf[right..]` (each length `n/2`).
    /// Regions must not overlap the source.
    pub fn split_indexed(
        &self,
        buf: &mut [ComplexDD],
        src: usize,
        left: usize,
        right: usize,
        n: usize,
    ) {
        debug_assert!(n >= 2 && n.is_power_of_two());
        let half = n / 2;
        let tw = self.twiddles_for_size(n);
        let half_dd = DoubleDouble::from_f64(0.5);
        for i in 0..half {
            let f0 = buf[src + 2 * i];
            let f1 = buf[src + 2 * i + 1];
            buf[left + i] = (f0 + f1).scale(half_dd);
            let diff = (f0 - f1).scale(half_dd);
            buf[right + i] = diff * tw[i].conj();
        }
    }

    /// Merge `buf[left..]` / `buf[right..]` (each `half`) → `buf[out..out+2*half]`.
    pub fn merge_indexed(
        &self,
        buf: &mut [ComplexDD],
        left: usize,
        right: usize,
        out: usize,
        half: usize,
    ) {
        let m = 2 * half;
        let tw = self.twiddles_for_size(m);
        for i in 0..half {
            let t = tw[i] * buf[right + i];
            let l = buf[left + i];
            buf[out + 2 * i] = l + t;
            buf[out + 2 * i + 1] = l - t;
        }
    }

    fn merge_scratch(
        &self,
        scratch: &mut FftScratchDd,
        left_c: usize,
        right_c: usize,
        out_c: usize,
        half: usize,
    ) {
        let m = 2 * half;
        let tw = self.twiddles_for_size(m);
        for i in 0..half {
            let t = tw[i] * scratch.complex[right_c + i];
            let l = scratch.complex[left_c + i];
            scratch.complex[out_c + 2 * i] = l + t;
            scratch.complex[out_c + 2 * i + 1] = l - t;
        }
    }

    fn split_scratch(
        &self,
        scratch: &mut FftScratchDd,
        in_c: usize,
        left_c: usize,
        right_c: usize,
        half: usize,
    ) {
        let m = 2 * half;
        let tw = self.twiddles_for_size(m);
        let half_dd = DoubleDouble::from_f64(0.5);
        for i in 0..half {
            let f0 = scratch.complex[in_c + 2 * i];
            let f1 = scratch.complex[in_c + 2 * i + 1];
            scratch.complex[left_c + i] = (f0 + f1).scale(half_dd);
            let diff = (f0 - f1).scale(half_dd);
            scratch.complex[right_c + i] = diff * tw[i].conj();
        }
    }

    /// FFT of `scratch.real[in_re..in_re+n]` → `scratch.complex[out_c..out_c+n]`.
    fn fft_rec(
        &self,
        scratch: &mut FftScratchDd,
        in_re: usize,
        out_c: usize,
        n: usize,
        work_re: usize,
        work_c: usize,
    ) {
        if n == 2 {
            let a = scratch.real[in_re];
            let b = scratch.real[in_re + 1];
            scratch.complex[out_c] = ComplexDD::new(a, b);
            scratch.complex[out_c + 1] = ComplexDD::new(a, -b);
            return;
        }
        let half = n / 2;
        let even = work_re;
        let odd = work_re + half;
        for i in 0..half {
            scratch.real[even + i] = scratch.real[in_re + 2 * i];
            scratch.real[odd + i] = scratch.real[in_re + 2 * i + 1];
        }
        let left_c = work_c;
        let right_c = work_c + half;
        let next_re = work_re + n;
        let next_c = work_c + n;
        self.fft_rec(scratch, even, left_c, half, next_re, next_c);
        self.fft_rec(scratch, odd, right_c, half, next_re, next_c);
        self.merge_scratch(scratch, left_c, right_c, out_c, half);
    }

    /// IFFT of `scratch.complex[in_c..in_c+n]` → `scratch.real[out_re..out_re+n]`.
    fn ifft_rec(
        &self,
        scratch: &mut FftScratchDd,
        in_c: usize,
        out_re: usize,
        n: usize,
        work_c: usize,
        work_re: usize,
    ) {
        if n == 2 {
            scratch.real[out_re] = scratch.complex[in_c].re;
            scratch.real[out_re + 1] = scratch.complex[in_c].im;
            return;
        }
        let half = n / 2;
        let left_c = work_c;
        let right_c = work_c + half;
        self.split_scratch(scratch, in_c, left_c, right_c, half);
        let left_r = work_re;
        let right_r = work_re + half;
        let next_c = work_c + n;
        let next_re = work_re + n;
        self.ifft_rec(scratch, left_c, left_r, half, next_c, next_re);
        self.ifft_rec(scratch, right_c, right_r, half, next_c, next_re);
        for i in 0..half {
            scratch.real[out_re + 2 * i] = scratch.real[left_r + i];
            scratch.real[out_re + 2 * i + 1] = scratch.real[right_r + i];
        }
    }

    #[cfg(test)]
    pub fn fft_into(
        &self,
        coeffs: &[DoubleDouble],
        out: &mut [ComplexDD],
        scratch: &mut FftScratchDd,
    ) {
        assert_eq!(coeffs.len(), self.n);
        assert_eq!(out.len(), self.n);
        scratch.reserve(self.n);
        for i in 0..self.n {
            scratch.real[i] = coeffs[i];
        }
        // Children write into scratch.complex[0..n]; copy out at end.
        // Workspace starts after the input reals / complex out region.
        self.fft_rec(scratch, 0, 0, self.n, self.n, self.n);
        out.copy_from_slice(&scratch.complex[..self.n]);
    }

    pub fn fft_i64(&self, coeffs: &[i64], scratch: &mut FftScratchDd) -> Vec<ComplexDD> {
        scratch.reserve(self.n);
        for i in 0..self.n {
            scratch.real[i] = DoubleDouble::from_i64(coeffs[i]);
        }
        self.fft_rec(scratch, 0, 0, self.n, self.n, self.n);
        scratch.complex[..self.n].to_vec()
    }

    #[cfg(test)]
    pub fn ifft_round_i64(&self, values: &[ComplexDD], scratch: &mut FftScratchDd) -> Vec<i64> {
        let mut out = vec![0i64; self.n];
        self.ifft_round_i64_into(values, &mut out, scratch);
        out
    }

    pub fn ifft_round_i64_into(
        &self,
        values: &[ComplexDD],
        out: &mut [i64],
        scratch: &mut FftScratchDd,
    ) {
        assert_eq!(values.len(), self.n);
        assert_eq!(out.len(), self.n);
        scratch.reserve(self.n);
        scratch.complex[..self.n].copy_from_slice(values);
        self.ifft_rec(scratch, 0, 0, self.n, self.n, self.n);
        for i in 0..self.n {
            out[i] = scratch.real[i].round_i64();
        }
    }

    /// FFT `i64` coeffs into `out` (length `n`) using `scratch` (no heap beyond scratch).
    pub fn fft_i64_into(&self, coeffs: &[i64], out: &mut [ComplexDD], scratch: &mut FftScratchDd) {
        assert_eq!(coeffs.len(), self.n);
        assert_eq!(out.len(), self.n);
        scratch.reserve(self.n);
        for i in 0..self.n {
            scratch.real[i] = DoubleDouble::from_i64(coeffs[i]);
        }
        self.fft_rec(scratch, 0, 0, self.n, self.n, self.n);
        out.copy_from_slice(&scratch.complex[..self.n]);
    }
}

pub fn mul_fft_dd(a: &[ComplexDD], b: &[ComplexDD]) -> Vec<ComplexDD> {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
}

pub fn add_fft_dd(a: &[ComplexDD], b: &[ComplexDD]) -> Vec<ComplexDD> {
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

pub fn sub_fft_dd(a: &[ComplexDD], b: &[ComplexDD]) -> Vec<ComplexDD> {
    a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
}

pub fn adj_fft_dd(a: &[ComplexDD]) -> Vec<ComplexDD> {
    a.iter().map(|z| z.conj()).collect()
}

pub fn div_fft_dd(a: &[ComplexDD], b: &[ComplexDD]) -> Vec<ComplexDD> {
    a.iter().zip(b.iter()).map(|(x, y)| *x / *y).collect()
}

#[cfg(test)]
pub fn scale_fft_dd(a: &[ComplexDD], s: DoubleDouble) -> Vec<ComplexDD> {
    a.iter().map(|z| z.scale(s)).collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]
    use super::*;
    use crate::utils::fft::{FftPlan, FftScratch};

    #[test]
    fn dd_matches_f64_small() {
        let n = 16;
        let plan_f = FftPlan::new(n).unwrap();
        let plan_d = FftPlanDd::new(n).unwrap();
        let mut scratch_f = FftScratch::default();
        let mut scratch_d = FftScratchDd::default();
        let coeffs: Vec<f64> = (0..n).map(|i| (i as f64) - 7.5).collect();
        let mut out_f = vec![num_complex::Complex64::new(0.0, 0.0); n];
        plan_f.fft_into(&coeffs, &mut out_f, &mut scratch_f);
        let dd: Vec<DoubleDouble> = coeffs.iter().copied().map(DoubleDouble::from_f64).collect();
        let mut out_d = vec![ComplexDD::ZERO; n];
        plan_d.fft_into(&dd, &mut out_d, &mut scratch_d);
        for i in 0..n {
            let err = (out_d[i].re.to_f64() - out_f[i].re).abs()
                + (out_d[i].im.to_f64() - out_f[i].im).abs();
            assert!(err < 1e-9, "i={i} err={err}");
        }
    }

    #[test]
    fn dd_convolution_512() {
        let n = 512;
        let plan = FftPlanDd::new(n).unwrap();
        let mut scratch = FftScratchDd::default();
        let a: Vec<i64> = (0..n).map(|i| ((i * 17) % 200) as i64 - 100).collect();
        let b: Vec<i64> = (0..n).map(|i| ((i * 31) % 200) as i64 - 100).collect();
        let mut expect = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let p = a[i] as i128 * b[j] as i128;
                let k = i + j;
                if k < n {
                    expect[k] += p;
                } else {
                    expect[k - n] -= p;
                }
            }
        }
        let fa = plan.fft_i64(&a, &mut scratch);
        let fb = plan.fft_i64(&b, &mut scratch);
        let prod = mul_fft_dd(&fa, &fb);
        let got = plan.ifft_round_i64(&prod, &mut scratch);
        for i in 0..n {
            assert_eq!(got[i] as i128, expect[i], "i={i}");
        }
    }
}
