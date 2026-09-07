//! Exact integer arithmetic in `Z[x]/(x^n + 1)` for NTRU verification.

#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]

/// Negacyclic product in `Z[x]/(x^n+1)` using `i128` accumulators.
pub(crate) fn negacyclic_mul_i128(a: &[i64], b: &[i64]) -> Vec<i128> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let mut out = vec![0i128; n];
    for i in 0..n {
        for j in 0..n {
            let prod = i128::from(a[i]) * i128::from(b[j]);
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

/// Map an integer coefficient to `[0, q)`.
pub(crate) fn reduce_mod_q_i64(c: i64, q: u32) -> u32 {
    let q = i64::from(q);
    ((c % q + q) % q) as u32
}

/// Verify `f G − g F = q` exactly in `Z[x]/(x^n+1)`.
pub(crate) fn verifies_ntru_equation(f: &[i64], g: &[i64], F: &[i32], G: &[i32], q: u32) -> bool {
    let f_g = negacyclic_mul_i128(f, &i32_slice_to_i64(G));
    let g_f = negacyclic_mul_i128(g, &i32_slice_to_i64(F));
    let q = i128::from(q);
    f_g.iter().zip(g_f).enumerate().all(|(i, (left, right))| {
        let expected = if i == 0 { q } else { 0 };
        left - right == expected
    })
}

fn i32_slice_to_i64(v: &[i32]) -> Vec<i64> {
    v.iter().map(|&x| i64::from(x)).collect()
}
