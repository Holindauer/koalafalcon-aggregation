//! RNS + small-prime NTT NTRUSolve (Pornin / Falcon keygen style).
//!
//! Adapted from [rust-fn-dsa](https://github.com/pornin/rust-fn-dsa) `fn-dsa-kgen`
//! (public domain). Limb budgets and the prime table are enlarged for KoalaBear `q`.

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]

mod fxp;
mod mp31;
mod ntru;
mod poly_mp;
mod vect;
mod zint31;

pub(crate) fn tmp_u32_len_for_test(logn: u32) -> usize {
    tmp_u32_len(logn)
}

pub(crate) fn tmp_fxr_len_for_test(logn: u32) -> usize {
    tmp_fxr_len(logn)
}

pub(crate) fn fxr_zero() -> fxp::FXR {
    fxp::FXR::ZERO
}

#[cfg(test)]
pub(crate) fn solve_fail_stage(
    logn: u32,
    f: &[i32],
    g: &[i32],
    F: &mut [i32],
    G: &mut [i32],
    tmp_u32: &mut [u32],
    tmp_fxr: &mut [fxp::FXR],
) -> Option<&'static str> {
    ntru::solve_NTRU_fail_stage(logn, f, g, F, G, tmp_u32, tmp_fxr)
}

fn tmp_u32_len(logn: u32) -> usize {
    let n = 1usize << logn;
    // Scaled MOD_LARGE_BL max is 308*3=924; peak layout ~O(max_llen * n) plus saves + NTT.
    48 * n + 4 * 924 * (n / 64).max(1)
}

fn tmp_fxr_len(logn: u32) -> usize {
    let n = 1usize << logn;
    // Intermediate Babai wants rt1..rt4 (4*n). Falcon only ran that path down
    // to depth 1 (n/2), so 2.5*n_top sufficed; we also reduce at depth 0.
    4 * n
}

/// Solve `f G - g F = q` with RNS+NTT; coeffs of `f,g` must fit in `i32`.
pub(crate) fn solve(f: &[i32], g: &[i32]) -> Result<(Vec<i32>, Vec<i32>), ()> {
    let n = f.len();
    if n != g.len() || !n.is_power_of_two() || !(2..=1024).contains(&n) {
        return Err(());
    }
    let logn = n.trailing_zeros();
    let mut F = vec![0i32; n];
    let mut G = vec![0i32; n];
    let mut tmp_u32 = vec![0u32; tmp_u32_len(logn)];
    let mut tmp_fxr = vec![fxp::FXR::ZERO; tmp_fxr_len(logn)];
    if ntru::solve_NTRU(logn, f, g, &mut F, &mut G, &mut tmp_u32, &mut tmp_fxr) {
        Ok((F, G))
    } else {
        Err(())
    }
}
