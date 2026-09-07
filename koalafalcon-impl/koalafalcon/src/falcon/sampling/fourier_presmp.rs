//! Fourier / double-double Fast Fourier Orthogonalization PreSmp for KoalaFalcon-512.
//!
//! Every returned sample is checked exactly in the KoalaBear ring:
//! `s1 + s2·h ≡ point`.

#![allow(non_snake_case)]
#![allow(
    clippy::missing_const_for_thread_local,
    clippy::needless_range_loop,
    clippy::only_used_in_recursion,
    clippy::too_many_arguments
)]

use super::samplerz::{MAX_SIGMA, sample_discrete_z};
use crate::algebra::{KOALA_BEAR_PRIME, KoalaBear, PreparedNegacyclicMultiplier};
use crate::utils::Error;
use crate::utils::double_double::{ComplexDD, DoubleDouble};
use crate::utils::error::FourierSamplingError;
use crate::utils::fft_dd::{
    FftPlanDd, FftScratchDd, add_fft_dd, adj_fft_dd, div_fft_dd, mul_fft_dd, sub_fft_dd,
};
use crate::utils::i64_to_ring;
use rand::{CryptoRng, Rng};
use std::cell::RefCell;

thread_local! {
    static SAMPLE_BUF: RefCell<Vec<ComplexDD>> = RefCell::new(Vec::new());
    static FFT_SCRATCH: RefCell<FftScratchDd> = RefCell::new(FftScratchDd::default());
    static V0_BUF: RefCell<Vec<i64>> = RefCell::new(Vec::new());
    static V1_BUF: RefCell<Vec<i64>> = RefCell::new(Vec::new());
    static CHECK_BUF: RefCell<Vec<KoalaBear>> = RefCell::new(Vec::new());
}

/// LDL tree with double-double nodes.
#[derive(Debug, Clone)]
pub enum FalconTreeDd {
    Node {
        l10: Vec<ComplexDD>,
        left: Box<FalconTreeDd>,
        right: Box<FalconTreeDd>,
    },
    Leaf {
        sigma: DoubleDouble,
    },
}

type GramDd = [[Vec<ComplexDD>; 2]; 2];
type BasisFft = [[Vec<ComplexDD>; 2]; 2];

/// O(n log n) Fourier sampler (double-double arithmetic).
#[derive(Debug, Clone)]
pub struct FourierPreSmp<const N: usize> {
    n: usize,
    basis_fft: BasisFft,
    /// Centered integer basis polys for exact reconstruction: `g`, `G`, `-f`, `-F`.
    #[allow(dead_code)]
    g: Vec<i64>,
    #[allow(dead_code)]
    G: Vec<i64>,
    #[allow(dead_code)]
    neg_f: Vec<i64>,
    #[allow(dead_code)]
    neg_F: Vec<i64>,
    tree: FalconTreeDd,
    #[cfg_attr(not(test), allow(dead_code))]
    plan: FftPlanDd,
    sigmin: f64,
    h_mul: PreparedNegacyclicMultiplier<N>,
}

impl<const N: usize> FourierPreSmp<N> {
    /// Build from centered integer NTRU polys and public `h` (centered).
    pub fn from_ntru_i64(
        g: &[i64],
        neg_f: &[i64],
        G: &[i64],
        neg_F: &[i64],
        h: &[i64],
        sigma: f64,
        sigmin: f64,
    ) -> Result<Self, Error> {
        let n = g.len();
        if n != N || N != 512 && N != 1024 {
            return Err(FourierSamplingError::InvalidDegree(n).into());
        }
        if neg_f.len() != n || G.len() != n || neg_F.len() != n || h.len() != n {
            return Err(Error::SigningFailed);
        }

        let plan = FftPlanDd::new(n)?;
        let mut scratch = FftScratchDd::default();
        // Falcon.py layout B₀ = [[g, −f], [G, −F]]: rows are the lattice basis,
        // and v = z · B₀ ⇒ (z₀g+z₁G, z₀(−f)+z₁(−F)). Dense PreSmp stores the
        // column-equivalent [[g,G],[−f,−F]]; Gram here MUST use Falcon rows.
        let a = plan.fft_i64(g, &mut scratch);
        let b = plan.fft_i64(neg_f, &mut scratch);
        let c = plan.fft_i64(G, &mut scratch);
        let d = plan.fft_i64(neg_F, &mut scratch);
        let basis_fft = [[a, b], [c, d]];

        let gram = gram_fft_dd(&basis_fft);
        let mut tree = ffldl_fft_dd(&gram)?;
        normalize_tree_dd(&mut tree, DoubleDouble::from_f64(sigma))?;
        validate_tree_dd(&tree, n, sigmin)?;

        let h_mul = PreparedNegacyclicMultiplier::new(i64_to_ring::<N>(h))?;

        Ok(Self {
            n,
            basis_fft,
            g: g.to_vec(),
            G: G.to_vec(),
            neg_f: neg_f.to_vec(),
            neg_F: neg_F.to_vec(),
            tree,
            plan,
            sigmin,
            h_mul,
        })
    }

    #[cfg(test)]
    pub(crate) fn tree_stored_complex_values(&self) -> usize {
        count_l10(&self.tree)
    }

    pub fn sample(
        &self,
        point: &[i64],
        rng: &mut (impl Rng + CryptoRng),
    ) -> Result<(Vec<i64>, Vec<i64>), Error> {
        if point.len() != self.n {
            return Err(Error::SigningFailed);
        }
        let n = self.n;
        let need = sampling_buf_len(n);
        let (t0, t1, z0, z1, work) = sampling_layout(n);

        SAMPLE_BUF.with(|b| {
            FFT_SCRATCH.with(|fs| {
                V0_BUF.with(|v0b| {
                    V1_BUF.with(|v1b| {
                        let mut buf = b.borrow_mut();
                        let mut fft_scratch = fs.borrow_mut();
                        let mut v0 = v0b.borrow_mut();
                        let mut v1 = v1b.borrow_mut();
                        if buf.len() < need {
                            buf.resize(need, ComplexDD::ZERO);
                        }
                        if v0.len() < n {
                            v0.resize(n, 0);
                            v1.resize(n, 0);
                        }

                        // point_fft temporarily in z0 slot, then t0/t1.
                        self.plan
                            .fft_i64_into(point, &mut buf[z0..z0 + n], &mut fft_scratch);
                        let q = DoubleDouble::from_f64(KOALA_BEAR_PRIME as f64);
                        let inv_q = q.recip();
                        let neg_inv_q = -inv_q;
                        for i in 0..n {
                            let p = buf[z0 + i];
                            buf[t0 + i] = (p * self.basis_fft[1][1][i]).scale(inv_q);
                            buf[t1 + i] = (p * self.basis_fft[0][1][i]).scale(neg_inv_q);
                        }

                        ffsampling_indexed(
                            &mut buf,
                            t0,
                            t1,
                            z0,
                            z1,
                            n,
                            work,
                            &self.tree,
                            self.sigmin,
                            rng,
                            &self.plan,
                        )?;

                        for i in 0..n {
                            let zz0 = buf[z0 + i];
                            let zz1 = buf[z1 + i];
                            buf[t0 + i] =
                                zz0 * self.basis_fft[0][0][i] + zz1 * self.basis_fft[1][0][i];
                            buf[t1 + i] =
                                zz0 * self.basis_fft[0][1][i] + zz1 * self.basis_fft[1][1][i];
                        }
                        self.plan.ifft_round_i64_into(
                            &buf[t0..t0 + n],
                            &mut v0[..n],
                            &mut fft_scratch,
                        );
                        self.plan.ifft_round_i64_into(
                            &buf[t1..t1 + n],
                            &mut v1[..n],
                            &mut fft_scratch,
                        );

                        let mut s1 = Vec::with_capacity(n);
                        let mut s2 = Vec::with_capacity(n);
                        for i in 0..n {
                            s1.push(point[i] - v0[i]);
                            s2.push(-v1[i]);
                        }
                        self.check_exact_preimage(point, &s1, &s2)?;
                        Ok((s1, s2))
                    })
                })
            })
        })
    }

    /// `v = B (z0; z1)` over \(\mathbb Z[x]/(x^n+1)\), then `s1=point-v0`, `s2=-v1`.
    #[cfg(test)]
    fn reconstruct_exact(
        &self,
        point: &[i64],
        z0: &[i64],
        z1: &[i64],
    ) -> Result<(Vec<i64>, Vec<i64>), Error> {
        let v0 = add_i128(
            &negacyclic_mul_i64(&self.g, z0),
            &negacyclic_mul_i64(&self.G, z1),
        );
        let v1 = add_i128(
            &negacyclic_mul_i64(&self.neg_f, z0),
            &negacyclic_mul_i64(&self.neg_F, z1),
        );
        let mut s1 = Vec::with_capacity(self.n);
        let mut s2 = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let s1i = (point[i] as i128) - v0[i];
            let s2i = -v1[i];
            s1.push(i64::try_from(s1i).map_err(|_| FourierSamplingError::OutputOverflow)?);
            s2.push(i64::try_from(s2i).map_err(|_| FourierSamplingError::OutputOverflow)?);
        }
        Ok((s1, s2))
    }

    fn check_exact_preimage(&self, point: &[i64], s1: &[i64], s2: &[i64]) -> Result<(), Error> {
        CHECK_BUF.with(|cb| {
            let mut coeffs = cb.borrow_mut();
            if coeffs.len() < self.n {
                coeffs.resize(self.n, KoalaBear::new(0));
            }
            let q = KOALA_BEAR_PRIME as i64;
            for i in 0..self.n {
                let mut r = s2[i] % q;
                if r < 0 {
                    r += q;
                }
                coeffs[i] = KoalaBear::new(r as u32);
            }
            self.h_mul.mul_coeffs_in_place(&mut coeffs[..self.n]);
            for i in 0..self.n {
                let mut a = s1[i] % q;
                if a < 0 {
                    a += q;
                }
                let mut c = point[i] % q;
                if c < 0 {
                    c += q;
                }
                let sum = KoalaBear::new(a as u32) + coeffs[i];
                if sum != KoalaBear::new(c as u32) {
                    return Err(FourierSamplingError::ExactPreimageFailed.into());
                }
            }
            Ok(())
        })
    }
}

/// Negacyclic product in \(\mathbb Z[x]/(x^n+1)\) with `i128` accumulators (tests).
#[cfg(test)]
fn negacyclic_mul_i64(a: &[i64], b: &[i64]) -> Vec<i128> {
    let n = a.len();
    debug_assert_eq!(b.len(), n);
    let mut out = vec![0i128; n];
    for i in 0..n {
        let ai = a[i] as i128;
        if ai == 0 {
            continue;
        }
        for j in 0..n {
            let prod = ai * b[j] as i128;
            let k = i + j;
            if k < n {
                out[k] += prod;
            } else {
                out[k - n] -= prod;
            }
        }
    }
    out
}

#[cfg(test)]
fn add_i128(a: &[i128], b: &[i128]) -> Vec<i128> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

fn gram_fft_dd(b: &BasisFft) -> GramDd {
    let deg = b[0][0].len();
    let mut g = [
        [vec![ComplexDD::ZERO; deg], vec![ComplexDD::ZERO; deg]],
        [vec![ComplexDD::ZERO; deg], vec![ComplexDD::ZERO; deg]],
    ];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let term = mul_fft_dd(&b[i][k], &adj_fft_dd(&b[j][k]));
                g[i][j] = add_fft_dd(&g[i][j], &term);
            }
        }
    }
    g
}

fn ldl_fft_dd(g: &GramDd) -> Result<(Vec<ComplexDD>, GramDd), Error> {
    let deg = g[0][0].len();
    for z in &g[0][0] {
        if !z.is_finite() {
            return Err(FourierSamplingError::NonFinite.into());
        }
        if z.norm_sq().to_f64() < 1e-30 {
            return Err(FourierSamplingError::NearZeroDenominator.into());
        }
    }
    let d00 = g[0][0].clone();
    let l10 = div_fft_dd(&g[1][0], &g[0][0]);
    let d11 = sub_fft_dd(
        &g[1][1],
        &mul_fft_dd(&mul_fft_dd(&l10, &adj_fft_dd(&l10)), &g[0][0]),
    );
    let zero = vec![ComplexDD::ZERO; deg];
    let d = [[d00, zero.clone()], [zero, d11]];
    Ok((l10, d))
}

fn split_fft_dd(plan: &FftPlanDd, values: &[ComplexDD]) -> [Vec<ComplexDD>; 2] {
    let half = values.len() / 2;
    let mut left = vec![ComplexDD::ZERO; half];
    let mut right = vec![ComplexDD::ZERO; half];
    // Plan sized for full n; split uses twiddles for `values.len()`.
    plan.split_fft_into(values, &mut left, &mut right);
    [left, right]
}

#[cfg(test)]
fn merge_fft_dd(plan: &FftPlanDd, left: &[ComplexDD], right: &[ComplexDD]) -> Vec<ComplexDD> {
    let mut out = vec![ComplexDD::ZERO; left.len() * 2];
    plan.merge_fft_into(left, right, &mut out);
    out
}

fn ffldl_fft_dd(g: &GramDd) -> Result<FalconTreeDd, Error> {
    let n = g[0][0].len();
    // Temporary plan sized to this recursion level for split/merge.
    let plan = FftPlanDd::new(n.max(2))?;
    ffldl_rec(g, &plan)
}

fn ffldl_rec(g: &GramDd, plan: &FftPlanDd) -> Result<FalconTreeDd, Error> {
    let n = g[0][0].len();
    let (l10, d) = ldl_fft_dd(g)?;
    if n > 2 {
        let child_plan = FftPlanDd::new(n)?;
        let d00 = split_fft_dd(&child_plan, &d[0][0]);
        let d11 = split_fft_dd(&child_plan, &d[1][1]);
        let g0 = [
            [d00[0].clone(), d00[1].clone()],
            [adj_fft_dd(&d00[1]), d00[0].clone()],
        ];
        let g1 = [
            [d11[0].clone(), d11[1].clone()],
            [adj_fft_dd(&d11[1]), d11[0].clone()],
        ];
        Ok(FalconTreeDd::Node {
            l10,
            left: Box::new(ffldl_rec(&g0, plan)?),
            right: Box::new(ffldl_rec(&g1, plan)?),
        })
    } else {
        Ok(FalconTreeDd::Node {
            l10,
            left: Box::new(FalconTreeDd::Leaf {
                sigma: d[0][0][0].re,
            }),
            right: Box::new(FalconTreeDd::Leaf {
                sigma: d[1][1][0].re,
            }),
        })
    }
}

fn normalize_tree_dd(tree: &mut FalconTreeDd, sigma: DoubleDouble) -> Result<(), Error> {
    match tree {
        FalconTreeDd::Node { left, right, .. } => {
            normalize_tree_dd(left, sigma)?;
            normalize_tree_dd(right, sigma)
        }
        FalconTreeDd::Leaf { sigma: leaf } => {
            let norm_sq = *leaf;
            if !norm_sq.is_finite() || norm_sq.to_f64() <= 0.0 {
                return Err(FourierSamplingError::InvalidLeafSigma(norm_sq.to_f64()).into());
            }
            *leaf = sigma / norm_sq.sqrt();
            if !leaf.is_finite() {
                return Err(FourierSamplingError::NonFinite.into());
            }
            Ok(())
        }
    }
}

fn validate_tree_dd(tree: &FalconTreeDd, expected_n: usize, sigmin: f64) -> Result<(), Error> {
    if !sigmin.is_finite() || sigmin <= 1.0 {
        return Err(FourierSamplingError::InvalidSamplerParameters {
            sigma: f64::NAN,
            sigmin,
        }
        .into());
    }
    let leaves = count_leaves(tree);
    if leaves != expected_n {
        return Err(FourierSamplingError::BadTreeShape {
            expected_leaves: expected_n,
            got_leaves: leaves,
        }
        .into());
    }
    walk_validate(tree, sigmin)
}

fn walk_validate(tree: &FalconTreeDd, sigmin: f64) -> Result<(), Error> {
    match tree {
        FalconTreeDd::Node { l10, left, right } => {
            for z in l10 {
                if !z.is_finite() {
                    return Err(FourierSamplingError::NonFinite.into());
                }
            }
            walk_validate(left, sigmin)?;
            walk_validate(right, sigmin)
        }
        FalconTreeDd::Leaf { sigma } => {
            let s = sigma.to_f64();
            if !s.is_finite() {
                return Err(FourierSamplingError::InvalidLeafSigma(s).into());
            }
            if s < sigmin {
                return Err(FourierSamplingError::LeafBelowSigmin { sigma: s, sigmin }.into());
            }
            if s > MAX_SIGMA {
                return Err(FourierSamplingError::LeafAboveSamplerMax {
                    sigma: s,
                    max_sigma: MAX_SIGMA,
                }
                .into());
            }
            Ok(())
        }
    }
}

fn count_leaves(tree: &FalconTreeDd) -> usize {
    match tree {
        FalconTreeDd::Leaf { .. } => 1,
        FalconTreeDd::Node { left, right, .. } => count_leaves(left) + count_leaves(right),
    }
}

#[cfg(test)]
fn count_l10(tree: &FalconTreeDd) -> usize {
    match tree {
        FalconTreeDd::Leaf { .. } => 0,
        FalconTreeDd::Node { l10, left, right } => l10.len() + count_l10(left) + count_l10(right),
    }
}

#[cfg(test)]
fn leaf_sigmas(tree: &FalconTreeDd, out: &mut Vec<f64>) {
    match tree {
        FalconTreeDd::Leaf { sigma } => out.push(sigma.to_f64()),
        FalconTreeDd::Node { left, right, .. } => {
            leaf_sigmas(left, out);
            leaf_sigmas(right, out);
        }
    }
}

#[cfg(test)]
fn tiny_valid_tree(sigma: f64) -> FalconTreeDd {
    FalconTreeDd::Node {
        l10: vec![ComplexDD::ZERO, ComplexDD::ZERO],
        left: Box::new(FalconTreeDd::Leaf {
            sigma: DoubleDouble::from_f64(sigma),
        }),
        right: Box::new(FalconTreeDd::Leaf {
            sigma: DoubleDouble::from_f64(sigma),
        }),
    }
}

#[cfg(test)]
fn split_as_target(plan: &FftPlanDd, t1: &[ComplexDD]) -> [Vec<ComplexDD>; 2] {
    split_fft_dd(plan, t1)
}

#[inline]
fn sampling_buf_len(n: usize) -> usize {
    let levels = (n as u64).ilog2() as usize + 2;
    // t0,t1,z0,z1 + 2n per recursion level (sequential children reuse work).
    4 * n + 2 * n * levels
}

#[inline]
fn sampling_layout(n: usize) -> (usize, usize, usize, usize, usize) {
    (0, n, 2 * n, 3 * n, 4 * n)
}

/// Index-based Falcon `ffSampling` — no per-level heap allocation.
fn ffsampling_indexed(
    buf: &mut [ComplexDD],
    t0: usize,
    t1: usize,
    z0: usize,
    z1: usize,
    n: usize,
    work: usize,
    tree: &FalconTreeDd,
    sigmin: f64,
    rng: &mut (impl Rng + CryptoRng),
    plan: &FftPlanDd,
) -> Result<(), Error> {
    match tree {
        FalconTreeDd::Node { l10, left, right } if n > 1 => {
            let half = n / 2;
            // Split t1 → (t0', t1') for the right child into work[0..n).
            let child_t0 = work;
            let child_t1 = work + half;
            let child_z0 = work + n;
            let child_z1 = work + n + half;
            let child_work = work + 2 * n;
            plan.split_indexed(buf, t1, child_t0, child_t1, n);
            ffsampling_indexed(
                buf, child_t0, child_t1, child_z0, child_z1, half, child_work, right, sigmin, rng,
                plan,
            )?;
            plan.merge_indexed(buf, child_z0, child_z1, z1, half);

            // t0b = t0 + (t1 − z1) ⊙ ℓ  → reuse work[0..n).
            let t0b = work;
            for i in 0..n {
                buf[t0b + i] = buf[t0 + i] + (buf[t1 + i] - buf[z1 + i]) * l10[i];
            }
            // Split t0b → child targets (write into work[n..2n)).
            let left_t0 = work + n;
            let left_t1 = work + n + half;
            plan.split_indexed(buf, t0b, left_t0, left_t1, n);
            let left_z0 = work + 2 * n;
            let left_z1 = work + 2 * n + half;
            let left_work = work + 3 * n;
            ffsampling_indexed(
                buf, left_t0, left_t1, left_z0, left_z1, half, left_work, left, sigmin, rng, plan,
            )?;
            plan.merge_indexed(buf, left_z0, left_z1, z0, half);
            Ok(())
        }
        FalconTreeDd::Leaf { sigma } if n == 1 => {
            let sigma_f = sigma.to_f64();
            if !sigma_f.is_finite() || sigma_f <= 0.0 {
                return Err(FourierSamplingError::LeafConversionFailed.into());
            }
            let c0 = buf[t0].re.to_f64();
            let c1 = buf[t1].re.to_f64();
            if !c0.is_finite() || !c1.is_finite() {
                return Err(FourierSamplingError::NonFinite.into());
            }
            let zz0 = sample_discrete_z(c0, sigma_f, sigmin, rng)?;
            let zz1 = sample_discrete_z(c1, sigma_f, sigmin, rng)?;
            buf[z0] = ComplexDD::new(DoubleDouble::from_i64(zz0), DoubleDouble::ZERO);
            buf[z1] = ComplexDD::new(DoubleDouble::from_i64(zz1), DoubleDouble::ZERO);
            Ok(())
        }
        _ => Err(FourierSamplingError::BadTreeShape {
            expected_leaves: n,
            got_leaves: 0,
        }
        .into()),
    }
}

/// Deterministic Fast Fourier nearest plane (Babai) in FFT domain (test helper).
#[cfg(test)]
fn ffnp_fft_dd(
    t: &[Vec<ComplexDD>; 2],
    tree: &FalconTreeDd,
    plan: &FftPlanDd,
) -> [Vec<ComplexDD>; 2] {
    let n = t[0].len();
    match tree {
        FalconTreeDd::Node { l10, left, right } if n > 1 => {
            let z1_parts = ffnp_fft_dd(&split_as_target(plan, &t[1]), right, plan);
            let z1 = merge_fft_dd(plan, &z1_parts[0], &z1_parts[1]);
            let t0b = add_fft_dd(&t[0], &mul_fft_dd(&sub_fft_dd(&t[1], &z1), l10));
            let z0_parts = ffnp_fft_dd(&split_as_target(plan, &t0b), left, plan);
            let z0 = merge_fft_dd(plan, &z0_parts[0], &z0_parts[1]);
            [z0, z1]
        }
        FalconTreeDd::Leaf { .. } if n == 1 => [
            vec![ComplexDD::new(
                DoubleDouble::from_f64(t[0][0].re.to_f64().round()),
                DoubleDouble::ZERO,
            )],
            vec![ComplexDD::new(
                DoubleDouble::from_f64(t[1][0].re.to_f64().round()),
                DoubleDouble::ZERO,
            )],
        ],
        _ => panic!("ffnp tree shape mismatch n={n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::falcon::hash::hash_to_point;
    use crate::falcon::keys::SigningKey;
    use crate::falcon::parameters::KoalaFalconParameters;
    use crate::falcon::trapdoor::TpdGen;
    use crate::utils::center_poly_i64;
    use crate::utils::fft_dd::scale_fft_dd;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn dd_fft_convolution_matches_negacyclic() {
        let mut rng = StdRng::from_os_rng();
        let n = 512usize;
        let plan = FftPlanDd::new(n).unwrap();
        let mut scratch = FftScratchDd::default();
        let a: Vec<i64> = (0..n).map(|_| rng.random_range(-100..100)).collect();
        let b: Vec<i64> = (0..n).map(|_| rng.random_range(-100..100)).collect();
        let expect = negacyclic_mul_i64(&a, &b);
        let fa = plan.fft_i64(&a, &mut scratch);
        let fb = plan.fft_i64(&b, &mut scratch);
        let prod = mul_fft_dd(&fa, &fb);
        let got = plan.ifft_round_i64(&prod, &mut scratch);
        let mut max_err = 0i128;
        for i in 0..n {
            max_err = max_err.max((got[i] as i128 - expect[i]).abs());
        }
        println!("DD FFT convolution max err={max_err}");
        assert_eq!(max_err, 0);
    }

    #[test]
    fn b_times_b_inv_recovers_point() {
        let mut rng = StdRng::from_os_rng();
        let params = KoalaFalconParameters::default();
        let tpd = TpdGen::from_params(&params).expect("tpd");
        let (trap_sk, pk) = tpd.generate_keys::<512>(&mut rng).expect("keygen");
        let signing_key = SigningKey::<512>::from_trapdoor(trap_sk, pk, &params).expect("expand");
        let fp = &signing_key.sampler;
        let c_ring = hash_to_point(&pk.h, b"salt", b"m").unwrap();
        let point = center_poly_i64(c_ring.coeffs());
        let mut scratch = FftScratchDd::default();
        let point_fft = fp.plan.fft_i64(&point, &mut scratch);
        let q = DoubleDouble::from_f64(KOALA_BEAR_PRIME as f64);
        let inv_q = q.recip();
        let t0 = scale_fft_dd(&mul_fft_dd(&point_fft, &fp.basis_fft[1][1]), inv_q);
        let t1 = scale_fft_dd(&mul_fft_dd(&point_fft, &fp.basis_fft[0][1]), -inv_q);
        // v = z·B0 with z = t: v0 = t0⊙a + t1⊙c, v1 = t0⊙b + t1⊙d
        let v0_fft = add_fft_dd(
            &mul_fft_dd(&t0, &fp.basis_fft[0][0]),
            &mul_fft_dd(&t1, &fp.basis_fft[1][0]),
        );
        let v1_fft = add_fft_dd(
            &mul_fft_dd(&t0, &fp.basis_fft[0][1]),
            &mul_fft_dd(&t1, &fp.basis_fft[1][1]),
        );
        let v0 = fp.plan.ifft_round_i64(&v0_fft, &mut scratch);
        let v1 = fp.plan.ifft_round_i64(&v1_fft, &mut scratch);
        let point_roundtrip = fp.plan.ifft_round_i64(&point_fft, &mut scratch);
        let mut max0 = 0i64;
        let mut max1 = 0i64;
        let mut max_rt = 0i64;
        let mut max_fft_diff = 0.0f64;
        for i in 0..512 {
            max0 = max0.max((v0[i] - point[i]).abs());
            max1 = max1.max(v1[i].abs());
            max_rt = max_rt.max((point_roundtrip[i] - point[i]).abs());
            let d = (v0_fft[i] - point_fft[i]).norm_sq().to_f64().sqrt();
            if d > max_fft_diff {
                max_fft_diff = d;
            }
        }
        let a = &fp.basis_fft[0][0];
        let b = &fp.basis_fft[0][1];
        let c = &fp.basis_fft[1][0];
        let d = &fp.basis_fft[1][1];
        let ad = mul_fft_dd(a, d);
        let bc = mul_fft_dd(b, c);
        let det = sub_fft_dd(&ad, &bc);
        let mut max_det_err = 0.0f64;
        for z in &det {
            let err = (z.re.to_f64() - KOALA_BEAR_PRIME as f64).abs() + z.im.to_f64().abs();
            max_det_err = max_det_err.max(err);
        }
        println!(
            "B·B⁻¹ err: max|v0-point|={max0} max|v1|={max1} fft↔ifft|point|={max_rt} max|v0_fft-point_fft|={max_fft_diff} max|det-q|={max_det_err}"
        );
        // Integer NTRU identity over Z[x]/(x^n+1).
        let f = center_poly_i64(trap_sk.f.coeffs());
        let g = center_poly_i64(trap_sk.g.coeffs());
        let F = center_poly_i64(trap_sk.F.coeffs());
        let G = center_poly_i64(trap_sk.G.coeffs());
        let fG = negacyclic_mul_i64(&f, &G);
        let gF = negacyclic_mul_i64(&g, &F);
        let mut max_ntru = 0i128;
        for i in 0..512 {
            let expect = if i == 0 { KOALA_BEAR_PRIME as i128 } else { 0 };
            max_ntru = max_ntru.max((fG[i] - gF[i] - expect).abs());
        }
        println!("integer fG-gF-q max coeff err={max_ntru}");

        assert!(max_rt <= 1, "fft roundtrip failed: {max_rt}");
        assert!(max_ntru == 0, "NTRU identity failed over Z");
        assert!(max0 <= 2, "v0 should recover point, err={max0}");
        assert!(max1 <= 2, "v1 should be ~0, err={max1}");
    }

    #[test]
    fn babai_fourier_norm_under_beta_scale() {
        let mut rng = StdRng::from_os_rng();
        let params = KoalaFalconParameters::default();
        let tpd = TpdGen::from_params(&params).expect("tpd");
        let (trap_sk, pk) = tpd.generate_keys::<512>(&mut rng).expect("keygen");
        let signing_key = SigningKey::<512>::from_trapdoor(trap_sk, pk, &params).expect("expand");
        let fp = &signing_key.sampler;
        let c_ring = hash_to_point(&pk.h, b"salt", b"m").unwrap();
        let point = center_poly_i64(c_ring.coeffs());
        let mut scratch = FftScratchDd::default();
        let point_fft = fp.plan.fft_i64(&point, &mut scratch);
        let q = DoubleDouble::from_f64(KOALA_BEAR_PRIME as f64);
        let inv_q = q.recip();
        let t0 = scale_fft_dd(&mul_fft_dd(&point_fft, &fp.basis_fft[1][1]), inv_q);
        let t1 = scale_fft_dd(&mul_fft_dd(&point_fft, &fp.basis_fft[0][1]), -inv_q);
        let z = ffnp_fft_dd(&[t0, t1], &fp.tree, &fp.plan);
        let z0 = fp.plan.ifft_round_i64(&z[0], &mut scratch);
        let z1 = fp.plan.ifft_round_i64(&z[1], &mut scratch);
        let (s1, s2) = fp.reconstruct_exact(&point, &z0, &z1).expect("recon");
        let n2: u128 = s1
            .iter()
            .chain(s2.iter())
            .map(|x| (*x as i128 * *x as i128) as u128)
            .sum();
        let beta2 = params.beta * params.beta;
        println!("Babai ‖s‖²={n2} β²={beta2}");
        // Babai can miss β, but should be within a small factor of σ√(2n), not 10⁶× over.
        assert!(
            (n2 as f64) < beta2 * 100.0,
            "Babai residual wildly large: {n2} vs β²={beta2}"
        );
    }

    #[test]
    fn validate_rejects_leaf_below_sigmin() {
        let tree = tiny_valid_tree(1.1);
        let err = validate_tree_dd(&tree, 2, 1.5).unwrap_err();
        assert!(matches!(
            err,
            Error::FourierSampling(FourierSamplingError::LeafBelowSigmin { .. })
        ));
    }

    #[test]
    fn validate_rejects_leaf_above_max_sigma() {
        let tree = tiny_valid_tree(MAX_SIGMA + 0.05);
        let err = validate_tree_dd(&tree, 2, 1.2).unwrap_err();
        assert!(matches!(
            err,
            Error::FourierSampling(FourierSamplingError::LeafAboveSamplerMax { .. })
        ));
    }

    #[test]
    fn validate_accepts_in_range_leaves() {
        let tree = tiny_valid_tree(1.5);
        validate_tree_dd(&tree, 2, 1.2).expect("in range");
    }

    #[test]
    fn production_source_has_no_leaf_clamp() {
        let src = include_str!("fourier_presmp.rs");
        let production = src
            .rsplit_once("mod tests {")
            .map(|(p, _)| p)
            .unwrap_or(src);
        assert!(
            !production.contains("max(sigmin)") && !production.contains("sigmin.min"),
            "leaf-width clamping must not be present"
        );
    }

    #[test]
    fn valid_tree_samples_exact_preimage() {
        let mut rng = StdRng::from_os_rng();
        let params = KoalaFalconParameters::default();
        let tpd = TpdGen::from_params(&params).expect("tpd");
        let (trap_sk, pk) = tpd.generate_keys::<512>(&mut rng).expect("keygen");
        let signing_key = match SigningKey::<512>::from_trapdoor(trap_sk, pk, &params) {
            Ok(sk) => sk,
            Err(e) => {
                // Expose leaf-width incompatibility rather than hiding it.
                let sigmin = {
                    let bound = params.s / params.target_gs_norm;
                    (bound * 0.995).max(1.001)
                };
                panic!(
                    "FourierPreSmp construction failed: {e:?}; \
                         params.s={}, target_gs_norm={}, sigmin≈{sigmin}, MAX_SIGMA={MAX_SIGMA}. \
                         Follow-up: raise fixed signing sigma, strengthen keygen GS rejection, \
                         or add a reviewed sampler for the required range.",
                    params.s, params.target_gs_norm
                );
            }
        };
        let c_ring = hash_to_point(&pk.h, b"salt", b"m").unwrap();
        let point = center_poly_i64(c_ring.coeffs());
        let (s1, s2) = signing_key
            .sampler
            .sample(&point, &mut rng)
            .expect("sample");
        signing_key
            .sampler
            .check_exact_preimage(&point, &s1, &s2)
            .expect("exact preimage");
    }

    #[test]
    fn report_leaf_width_range_on_keygen_expand() {
        fn report<const N: usize>(label: &str) {
            let mut rng = StdRng::from_os_rng();
            let params = if N == 512 {
                KoalaFalconParameters::falcon_512()
            } else {
                KoalaFalconParameters::falcon_1024()
            };
            let tpd = TpdGen::from_params(&params).expect("tpd");
            let (trap_sk, pk) = tpd.generate_keys::<N>(&mut rng).expect("keygen");

            let f_i = crate::utils::center_poly_i64(trap_sk.f.coeffs());
            let g_i = crate::utils::center_poly_i64(trap_sk.g.coeffs());
            let F_i = crate::utils::center_poly_i64(trap_sk.F.coeffs());
            let G_i = crate::utils::center_poly_i64(trap_sk.G.coeffs());
            let neg_f: Vec<i64> = f_i.iter().map(|&x| -x).collect();
            let neg_F: Vec<i64> = F_i.iter().map(|&x| -x).collect();
            let plan = FftPlanDd::new(N).unwrap();
            let mut scratch = FftScratchDd::default();
            let a = plan.fft_i64(&g_i, &mut scratch);
            let b = plan.fft_i64(&neg_f, &mut scratch);
            let c = plan.fft_i64(&G_i, &mut scratch);
            let d = plan.fft_i64(&neg_F, &mut scratch);
            let basis_fft = [[a, b], [c, d]];
            let gram = gram_fft_dd(&basis_fft);
            let mut tree = ffldl_fft_dd(&gram).unwrap();
            normalize_tree_dd(&mut tree, DoubleDouble::from_f64(params.s)).unwrap();
            let mut widths = Vec::new();
            leaf_sigmas(&tree, &mut widths);
            let min_w = widths.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_w = widths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sigmin = {
                let bound = params.s / params.target_gs_norm;
                (bound * 0.995).max(1.001)
            };
            println!(
                "{label} leaf widths: min={min_w} max={max_w} sigmin={sigmin} MAX_SIGMA={MAX_SIGMA} params.s={}",
                params.s
            );
            let below = widths.iter().filter(|&&w| w < sigmin).count();
            let above = widths.iter().filter(|&&w| w > MAX_SIGMA).count();
            println!(
                "{label} leaves below sigmin={below}/{} above MAX_SIGMA={above}/{}",
                widths.len(),
                widths.len()
            );

            let expand = trap_sk.expand_fourier_presmp(&pk, &params);
            if below > 0 || above > 0 {
                assert!(
                    expand.is_err(),
                    "{label}: expected construction failure when leaves out of SamplerZ range"
                );
            } else {
                expand.expect("{label}: in-range tree should construct");
            }
        }

        report::<512>("KoalaFalcon-512");
        report::<1024>("KoalaFalcon-1024");
    }
}
