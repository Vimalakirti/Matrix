pub mod dense;
pub mod sparse;

pub use dense::{DenseMLPoly, DeviceDenseMLPoly};
pub use sparse::{SelectionPolynomial, SparseMLPoly};

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rayon::prelude::*;
use std::any::Any;

use crate::util::arith::{agl_mul, agl_sub, ext2_mul, ext2_sub};

const EQ_PAR_THRESHOLD: usize = 8192;

// ============================================================================
// MLPoly trait — multilinear polynomial interface
// ============================================================================

/// Multilinear polynomial trait. Both [`DenseMLPoly`] and [`SparseMLPoly`]
/// implement it so consumers can hold a `Box<dyn MLPoly>` for either backing.
pub trait MLPoly: std::fmt::Debug + Any + Send + Sync {
    fn fix_variables(&self, partial_point: &[AlmostGoldilocksField]) -> Box<dyn MLPoly>;
    fn n(&self) -> usize;
    fn len(&self) -> usize;
    fn evaluate_at_point(&self, point: &[AlmostGoldilocksField]) -> AlmostGoldilocksField;
    fn evaluate_at_point_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2;
    fn evaluations(&self) -> Vec<AlmostGoldilocksField>;
    /// Borrow the evaluation table without cloning. Returns `Some` for dense
    /// polynomials, `None` for sparse ones (no contiguous backing).
    fn try_evaluations_ref(&self) -> Option<&[AlmostGoldilocksField]> {
        None
    }
    /// Like [`try_evaluations_ref`] but panics if the underlying type doesn't
    /// support O(1) slice access (e.g., sparse polynomials).
    fn evaluations_ref(&self) -> &[AlmostGoldilocksField] {
        self.try_evaluations_ref()
            .expect("evaluations_ref not available for this polynomial type")
    }
    fn index(&self, index: usize) -> AlmostGoldilocksField;
    fn index_mut(&mut self, index: usize) -> &mut AlmostGoldilocksField;
    fn clone_box(&self) -> Box<dyn MLPoly>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn mul_by_scalar(&self, scalar: AlmostGoldilocksField) -> Box<dyn MLPoly>;
    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly>;
}

// ============================================================================
// Lagrange eq-tables
// ============================================================================

/// Compute the Lagrange-basis evaluation table
/// `eq(r, x) = Π_i (r_i · x_i + (1 − r_i)(1 − x_i))` for all
/// `x ∈ {0,1}^n`. Uses the little-endian `evals[j | half]` pattern so it
/// matches [`DenseMLPoly::fix_variables`] (variable 0 = bit 0 = LSB).
pub fn evaluate_lagrange_basis(r: &[AlmostGoldilocksField]) -> Vec<AlmostGoldilocksField> {
    let n = r.len();
    if n == 0 {
        return vec![AlmostGoldilocksField(1)];
    }
    let size = 1usize << n;
    let mut evals = vec![AlmostGoldilocksField(0); size];
    evals[0] = AlmostGoldilocksField(1);
    for i in 0..n {
        let one_minus_ri = agl_sub(AlmostGoldilocksField(1), r[i]);
        let ri = r[i];
        let half = 1usize << i;
        if half >= EQ_PAR_THRESHOLD {
            let (lo, hi) = evals[..2 * half].split_at_mut(half);
            lo.par_iter_mut().zip(hi.par_iter_mut()).for_each(|(lo_val, hi_val)| {
                *hi_val = agl_mul(*lo_val, ri);
                *lo_val = agl_mul(*lo_val, one_minus_ri);
            });
        } else {
            for j in (0..half).rev() {
                evals[j | half] = agl_mul(evals[j], ri);
                evals[j] = agl_mul(evals[j], one_minus_ri);
            }
        }
    }
    evals
}

/// Same as [`evaluate_lagrange_basis`] but with an Ext2 challenge point.
pub fn evaluate_lagrange_basis_ext2(r: &[AlmostGoldilocksExt2]) -> Vec<AlmostGoldilocksExt2> {
    let n = r.len();
    if n == 0 {
        return vec![AlmostGoldilocksExt2::one()];
    }
    let size = 1usize << n;
    let mut evals = vec![AlmostGoldilocksExt2::zero(); size];
    evals[0] = AlmostGoldilocksExt2::one();
    let one = AlmostGoldilocksExt2::one();
    for i in 0..n {
        let one_minus_ri = ext2_sub(one, r[i]);
        let ri = r[i];
        let half = 1usize << i;
        if half >= EQ_PAR_THRESHOLD {
            let (lo, hi) = evals[..2 * half].split_at_mut(half);
            lo.par_iter_mut().zip(hi.par_iter_mut()).for_each(|(lo_val, hi_val)| {
                *hi_val = ext2_mul(*lo_val, ri);
                *lo_val = ext2_mul(*lo_val, one_minus_ri);
            });
        } else {
            for j in (0..half).rev() {
                evals[j | half] = ext2_mul(evals[j], ri);
                evals[j] = ext2_mul(evals[j], one_minus_ri);
            }
        }
    }
    evals
}

// ============================================================================
// Public table polynomials
// ============================================================================

/// Range table: `table[i] = i` for `i ∈ [0, 2^num_vars)`. Used as the
/// reference column in the range-check lookup (§5.5).
pub fn range_dense(num_vars: usize) -> DenseMLPoly {
    let vec_len = 1usize << num_vars;
    let evaluations: Vec<AlmostGoldilocksField> =
        (0..vec_len).map(|i| AlmostGoldilocksField(i as u64)).collect();
    DenseMLPoly::new(num_vars, evaluations)
}

/// Two-power table: `table[i] = 2^(15 − i)` for `i ∈ [0, 16)`. Used as the
/// reference column for the two-pow lookup (ExpHelper auxiliary).
pub fn two_pow_dense() -> DenseMLPoly {
    let num_vars = 4;
    let vec_len = 1usize << num_vars;
    let evaluations: Vec<AlmostGoldilocksField> = (0..vec_len)
        .map(|i| AlmostGoldilocksField(1u64 << (15 - i)))
        .collect();
    DenseMLPoly::new(num_vars, evaluations)
}

/// Horner evaluation of a univariate polynomial `coeffs[0] + coeffs[1]·x + …`.
pub fn evaluate_univariate(
    coeffs: &[AlmostGoldilocksField],
    x: AlmostGoldilocksField,
) -> AlmostGoldilocksField {
    let mut result = AlmostGoldilocksField(0);
    for &c in coeffs.iter().rev() {
        result = result * x + c;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `evaluate_lagrange_basis` over `n = 0` returns the trivial table `[1]`.
    #[test]
    fn lagrange_basis_empty_returns_unit() {
        let evals = evaluate_lagrange_basis(&[]);
        assert_eq!(evals, vec![AlmostGoldilocksField(1)]);
    }

    /// Verify `evaluate_lagrange_basis` matches the textbook definition
    /// `eq(r, x) = Π_i (bit_i · r_i + (1 − bit_i)(1 − r_i))` for small n.
    #[test]
    fn lagrange_basis_matches_textbook() {
        let r = [
            AlmostGoldilocksField(3),
            AlmostGoldilocksField(5),
            AlmostGoldilocksField(7),
        ];
        let evals = evaluate_lagrange_basis(&r);
        assert_eq!(evals.len(), 8);
        let one = AlmostGoldilocksField(1);
        for x in 0..8usize {
            let mut want = one;
            for i in 0..3 {
                let bit = (x >> i) & 1;
                let factor = if bit == 1 { r[i] } else { agl_sub(one, r[i]) };
                want = agl_mul(want, factor);
            }
            assert_eq!(evals[x].reduce(), want.reduce(), "x = {}", x);
        }
    }

    /// Lagrange evaluations must sum to 1 over the Boolean hypercube
    /// (partition-of-unity property).
    #[test]
    fn lagrange_basis_sums_to_one() {
        let r = [
            AlmostGoldilocksField(11),
            AlmostGoldilocksField(13),
            AlmostGoldilocksField(17),
            AlmostGoldilocksField(19),
        ];
        let evals = evaluate_lagrange_basis(&r);
        let mut sum = AlmostGoldilocksField(0);
        for e in evals {
            sum = sum + e;
        }
        assert_eq!(sum.reduce(), AlmostGoldilocksField(1));
    }

    /// Ext2 Lagrange basis: same partition-of-unity check.
    #[test]
    fn lagrange_basis_ext2_sums_to_one() {
        let r = vec![
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(7), AlmostGoldilocksField(11)),
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(2), AlmostGoldilocksField(0)),
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(3), AlmostGoldilocksField(5)),
        ];
        let evals = evaluate_lagrange_basis_ext2(&r);
        let mut sum = AlmostGoldilocksExt2::zero();
        for e in evals {
            sum = sum + e;
        }
        assert!(crate::util::arith::ext2_field_eq(sum, AlmostGoldilocksExt2::one()));
    }

    #[test]
    fn lagrange_basis_at_boolean_point_is_indicator() {
        // For r ∈ {0, 1}^n, evaluate_lagrange_basis(r) is the indicator that
        // selects the single position equal to r.
        for r0 in 0..2u64 {
            for r1 in 0..2u64 {
                let r = [AlmostGoldilocksField(r0), AlmostGoldilocksField(r1)];
                let evals = evaluate_lagrange_basis(&r);
                let want_idx = (r0 as usize) | ((r1 as usize) << 1);
                for (i, e) in evals.iter().enumerate() {
                    let expected = if i == want_idx {
                        AlmostGoldilocksField(1)
                    } else {
                        AlmostGoldilocksField(0)
                    };
                    assert_eq!(e.reduce(), expected, "r = ({}, {}), i = {}", r0, r1, i);
                }
            }
        }
    }

    /// Parallel path (`half >= EQ_PAR_THRESHOLD`) must produce the same table
    /// as the sequential path. Exercise at exactly `n = log2(EQ_PAR_THRESHOLD)
    /// + 1` so the par branch fires.
    #[test]
    fn lagrange_basis_par_path_matches_sequential() {
        let n = (EQ_PAR_THRESHOLD as u64).trailing_zeros() as usize + 1;
        let r: Vec<_> = (0..n).map(|i| AlmostGoldilocksField((i as u64) * 31 + 7)).collect();
        let par_evals = evaluate_lagrange_basis(&r);
        // Recompute serially: same algorithm with a sequential loop only.
        let mut seq_evals = vec![AlmostGoldilocksField(0); 1 << n];
        seq_evals[0] = AlmostGoldilocksField(1);
        for i in 0..n {
            let one_minus_ri = agl_sub(AlmostGoldilocksField(1), r[i]);
            let ri = r[i];
            let half = 1usize << i;
            for j in (0..half).rev() {
                seq_evals[j | half] = agl_mul(seq_evals[j], ri);
                seq_evals[j] = agl_mul(seq_evals[j], one_minus_ri);
            }
        }
        for i in 0..(1 << n) {
            assert_eq!(par_evals[i].reduce(), seq_evals[i].reduce(), "idx {}", i);
        }
    }

    #[test]
    fn range_dense_layout() {
        let r = range_dense(3);
        assert_eq!(r.n, 3);
        for i in 0..8 {
            assert_eq!(r.evaluations[i], AlmostGoldilocksField(i as u64));
        }
    }

    #[test]
    fn two_pow_dense_layout() {
        let t = two_pow_dense();
        assert_eq!(t.n, 4);
        for i in 0..16 {
            assert_eq!(t.evaluations[i], AlmostGoldilocksField(1u64 << (15 - i)));
        }
    }

    #[test]
    fn evaluate_univariate_horner() {
        // p(x) = 1 + 2x + 3x^2
        let coeffs = [
            AlmostGoldilocksField(1),
            AlmostGoldilocksField(2),
            AlmostGoldilocksField(3),
        ];
        for x_raw in 0u64..10 {
            let x = AlmostGoldilocksField(x_raw);
            let want = AlmostGoldilocksField(1 + 2 * x_raw + 3 * x_raw * x_raw);
            assert_eq!(evaluate_univariate(&coeffs, x).reduce(), want);
        }
        // Empty coeffs => 0.
        assert_eq!(
            evaluate_univariate(&[], AlmostGoldilocksField(7)).reduce(),
            AlmostGoldilocksField(0)
        );
    }
}
