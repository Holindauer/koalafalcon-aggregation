//! NTRUSolve and the Gram–Schmidt filter for NTRUGen.
//!
//! The equation `f G − g F = q` is solved with Pornin’s RNS + small-prime NTT
//! path (see `ntru_solve/`), not multiprecision Karatsuba.

#![allow(non_snake_case)]

use crate::profile;

/// GS filter without allocating multiprecision polys (keygen fast path).
pub fn gs_norm_squared_i64(f: &[i64], g: &[i64], q: u32) -> f64 {
    profile!("gs_norm_filter");
    let f_r: Vec<f64> = f.iter().map(|&c| c as f64).collect();
    let g_r: Vec<f64> = g.iter().map(|&c| c as f64).collect();
    gs_norm_squared_f64(&f_r, &g_r, q)
}

fn gs_norm_squared_f64(f_r: &[f64], g_r: &[f64], q: u32) -> f64 {
    use crate::utils::fft::{add_poly, adj_poly, div_poly, mul_poly};

    let sqnorm_fg = sqnorm_f64(&[f_r, g_r]);

    let ffgg = add_poly(
        &mul_poly(f_r, &adj_poly(f_r)),
        &mul_poly(g_r, &adj_poly(g_r)),
    );
    let ft = div_poly(&adj_poly(g_r), &ffgg);
    let gt = div_poly(&adj_poly(f_r), &ffgg);
    let sqnorm_fg_ortho = (q as f64).powi(2) * sqnorm_f64(&[&ft, &gt]);
    sqnorm_fg.max(sqnorm_fg_ortho)
}

fn sqnorm_f64(polys: &[&[f64]]) -> f64 {
    let mut res = 0.0;
    for poly in polys {
        for &c in *poly {
            res += c * c;
        }
    }
    res
}

/// Solve from centered `i32` coefficients.
pub fn ntru_solve_i32(f: &[i32], g: &[i32]) -> Result<(Vec<i32>, Vec<i32>), ()> {
    profile!("ntru_solve");
    super::ntru_solve::solve(f, g)
}

/// Verify `f G − g F = q` (constant polynomial) exactly in `Z[x]/(x^n+1)`.
pub fn verifies_ntru_equation(f: &[i64], g: &[i64], F: &[i32], G: &[i32], q: u32) -> bool {
    super::poly::verifies_ntru_equation(f, g, F, G, q)
}
