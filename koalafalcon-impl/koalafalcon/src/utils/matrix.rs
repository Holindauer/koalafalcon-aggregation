//! Reference matrix helpers for negacyclic multiplication tests.

#[cfg(test)]
pub fn neg_one_skew_circulant_matrix<F: crate::algebra::Field>(v: &[F], n: usize) -> Vec<F> {
    debug_assert_eq!(v.len(), n, "expected {n} coefficients, got {}", v.len());

    let mut matrix = vec![F::ZERO; n * n];
    for k in 0..n {
        for j in 0..n {
            matrix[k * n + j] = if k >= j { v[k - j] } else { -v[k + n - j] };
        }
    }

    matrix
}

#[cfg(test)]
pub fn matmul<F: crate::algebra::Field>(
    a: &[F],
    b: &[F],
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> Vec<F> {
    let mut result = vec![F::ZERO; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                result[i * b_cols + j] += a[i * a_cols + k] * b[k * b_cols + j];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::KoalaBear;

    #[test]
    fn matmul_2x2() {
        let a = [
            KoalaBear::new(1),
            KoalaBear::new(2),
            KoalaBear::new(3),
            KoalaBear::new(4),
        ];
        let b = [
            KoalaBear::new(5),
            KoalaBear::new(6),
            KoalaBear::new(7),
            KoalaBear::new(8),
        ];
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(
            c,
            [
                KoalaBear::new(19),
                KoalaBear::new(22),
                KoalaBear::new(43),
                KoalaBear::new(50)
            ]
        );
    }

    #[test]
    fn neg_one_skew_circulant_matrix_n4() {
        let v = [
            KoalaBear::new(1),
            KoalaBear::new(2),
            KoalaBear::new(3),
            KoalaBear::new(4),
        ];
        let m = neg_one_skew_circulant_matrix(&v, 4);
        assert_eq!(
            m,
            [
                KoalaBear::new(1),
                -KoalaBear::new(4),
                -KoalaBear::new(3),
                -KoalaBear::new(2),
                KoalaBear::new(2),
                KoalaBear::new(1),
                -KoalaBear::new(4),
                -KoalaBear::new(3),
                KoalaBear::new(3),
                KoalaBear::new(2),
                KoalaBear::new(1),
                -KoalaBear::new(4),
                KoalaBear::new(4),
                KoalaBear::new(3),
                KoalaBear::new(2),
                KoalaBear::new(1),
            ]
        );
    }
}
