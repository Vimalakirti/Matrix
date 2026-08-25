pub mod dense;
pub mod sparse;

pub use dense::{DenseMLPoly, DeviceDenseMLPoly};
pub use sparse::SparseMLPoly;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use rayon::prelude::*;
use std::any::Any;

use crate::util::arith::{gl_add, gl_sub, gl_mul, gl_inv, ext2_sub, ext2_mul};
use crate::GOLDILOCKS_PRIME;

const EQ_PAR_THRESHOLD: usize = 8192;

// ============================================================================
// CryptoField trait — minimal field interface for proof system code
// ============================================================================

pub trait CryptoField:
    Clone
    + Copy
    + std::fmt::Debug
    + PartialEq
    + Send
    + Sync
    + 'static
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_u32(n: u32) -> Self;
    fn from_u64(n: u64) -> Self;
    fn to_u64(&self) -> u64;
    fn invert(&self) -> Self;
    fn to_bytes_le(&self) -> Vec<u8>;
    fn from_bytes_le(bytes: &[u8]) -> Self;
}

impl CryptoField for GoldilocksField {
    fn zero() -> Self {
        GoldilocksField(0)
    }
    fn one() -> Self {
        GoldilocksField(1)
    }
    fn from_u32(n: u32) -> Self {
        GoldilocksField(n as u64)
    }
    fn from_u64(n: u64) -> Self {
        GoldilocksField(n % GOLDILOCKS_PRIME)
    }
    fn to_u64(&self) -> u64 {
        self.0
    }
    fn invert(&self) -> Self {
        gl_inv(*self)
    }
    fn to_bytes_le(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
    fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut buf = [0u8; 8];
        let len = bytes.len().min(8);
        buf[..len].copy_from_slice(&bytes[..len]);
        GoldilocksField(u64::from_le_bytes(buf) % GOLDILOCKS_PRIME)
    }
}

// ============================================================================
// MLPoly trait — multilinear polynomial interface
// ============================================================================

pub trait MLPoly: std::fmt::Debug + Any + Send + Sync {
    fn fix_variables(&self, partial_point: &[GoldilocksField]) -> Box<dyn MLPoly>;
    fn n(&self) -> usize;
    fn len(&self) -> usize;
    fn evaluate_at_point(&self, point: &[GoldilocksField]) -> GoldilocksField;
    fn evaluate_at_point_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2;
    fn evaluations(&self) -> Vec<GoldilocksField>;
    /// Borrow the evaluation table without cloning.
    /// Returns `Some` for DenseMLPoly, `None` for SparseMLPoly.
    fn try_evaluations_ref(&self) -> Option<&[GoldilocksField]> {
        None
    }
    /// Borrow the evaluation table without cloning.
    /// Only available for DenseMLPoly; panics for SparseMLPoly.
    fn evaluations_ref(&self) -> &[GoldilocksField] {
        self.try_evaluations_ref().expect("evaluations_ref not available for this polynomial type")
    }
    fn index(&self, index: usize) -> GoldilocksField;
    fn index_mut(&mut self, index: usize) -> &mut GoldilocksField;
    fn clone_box(&self) -> Box<dyn MLPoly>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn mul_by_scalar(&self, scalar: GoldilocksField) -> Box<dyn MLPoly>;
    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly>;
}

/// Evaluate Lagrange basis polynomial eq(r, x) = Π(r_i * x_i + (1-r_i)*(1-x_i))
/// Returns the full evaluation table of size 2^n.
/// Uses little-endian convention: variable i corresponds to bit i of the index,
/// matching DenseMLPoly::fix_variables.
pub fn evaluate_lagrange_basis(r: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let n = r.len();
    if n == 0 {
        return vec![GoldilocksField(1)];
    }
    let size = 1usize << n;
    let mut evals = vec![GoldilocksField(0); size];
    evals[0] = GoldilocksField(1);
    for i in 0..n {
        let one_minus_ri = gl_sub(GoldilocksField(1), r[i]);
        let ri = r[i];
        let half = 1usize << i;
        if half >= EQ_PAR_THRESHOLD {
            let (lo, hi) = evals[..2 * half].split_at_mut(half);
            lo.par_iter_mut().zip(hi.par_iter_mut()).for_each(|(lo_val, hi_val)| {
                *hi_val = gl_mul(*lo_val, ri);
                *lo_val = gl_mul(*lo_val, one_minus_ri);
            });
        } else {
            for j in (0..half).rev() {
                evals[j | half] = gl_mul(evals[j], ri);
                evals[j] = gl_mul(evals[j], one_minus_ri);
            }
        }
    }
    evals
}

/// Evaluate Lagrange basis polynomial eq(r, x) for Ext2 challenge points.
/// Same algorithm as `evaluate_lagrange_basis` but with Ext2 arithmetic.
pub fn evaluate_lagrange_basis_ext2(r: &[GoldilocksExt2]) -> Vec<GoldilocksExt2> {
    let n = r.len();
    if n == 0 {
        return vec![GoldilocksExt2::one()];
    }
    let size = 1usize << n;
    let mut evals = vec![GoldilocksExt2::zero(); size];
    evals[0] = GoldilocksExt2::one();
    let one = GoldilocksExt2::one();
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

/// Range table polynomial: table[i] = i for i in 0..2^num_vars.
pub fn range_dense(num_vars: usize) -> DenseMLPoly {
    let vec_len = 1usize << num_vars;
    let evaluations: Vec<GoldilocksField> = (0..vec_len)
        .map(|i| GoldilocksField(i as u64))
        .collect();
    DenseMLPoly::new(num_vars, evaluations)
}

/// Two-power table polynomial: table[i] = 2^(15-i) for i in 0..16.
pub fn two_pow_dense() -> DenseMLPoly {
    assert!(4 <= 16); // always true, but documents we need 4 vars = 16 entries
    let num_vars = 4;
    let vec_len = 1usize << num_vars;
    let evaluations: Vec<GoldilocksField> = (0..vec_len)
        .map(|i| GoldilocksField(1u64 << (15 - i)))
        .collect();
    DenseMLPoly::new(num_vars, evaluations)
}

/// Evaluate a univariate polynomial given as coefficients at a point.
pub fn evaluate_univariate(coeffs: &[GoldilocksField], x: GoldilocksField) -> GoldilocksField {
    // Horner's method
    let mut result = GoldilocksField(0);
    for &c in coeffs.iter().rev() {
        result = gl_add(gl_mul(result, x), c);
    }
    result
}
