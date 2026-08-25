//! Host-side arithmetic helpers over the AlmostGoldilocks field.
//!
//! The `AlmostGoldilocksField` type already implements `Add/Sub/Mul/Neg`, so
//! the `agl_*` helpers here are thin wrappers preserved for parity with the
//! zk-torch-3 module shape (downstream call sites expect free-function names
//! like `agl_mul(a, b)`).

use almost_goldilocks_cuda::extension::{AEXT2_W, AlmostGoldilocksExt2};
use almost_goldilocks_cuda::field::{AlmostGoldilocksField, ALMOST_GOLDILOCKS_PRIME};

// ============================================================================
// Shape / size helpers (field-agnostic)
// ============================================================================

/// Compute `ceil(log2(n))`. Returns 0 for `n <= 1`.
pub fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// Compute `2^k`.
pub fn pow_2(k: usize) -> usize {
    1usize << k
}

/// Compute the number of multilinear variables needed to index a tensor of
/// the given shape. Each dimension is rounded up to the next power of two
/// (so the index space is a Boolean hypercube) and the per-dim bit counts
/// are summed.
pub fn get_n(shape: &[usize]) -> usize {
    shape.iter().map(|&s| log2_ceil(s.max(1))).sum()
}

/// Round `n` up to the next power of two (returning 1 for `n <= 1`).
pub fn next_pow(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

// ============================================================================
// Field <-> signed integer
// ============================================================================

/// Interpret a field element as a signed integer. Elements above `q/2` are
/// treated as negatives (two's-complement-style).
pub fn f_to_int(f: AlmostGoldilocksField) -> i128 {
    let v = f.reduce().0;
    if v > ALMOST_GOLDILOCKS_PRIME / 2 {
        v as i128 - ALMOST_GOLDILOCKS_PRIME as i128
    } else {
        v as i128
    }
}

/// Inverse of [`f_to_int`]: any `i128` lifts to its canonical representative
/// in `[0, q)`.
pub fn int_to_f(x: i128) -> AlmostGoldilocksField {
    if x >= 0 {
        AlmostGoldilocksField((x as u128 % ALMOST_GOLDILOCKS_PRIME as u128) as u64)
    } else {
        let mag = ((-x) as u128 % ALMOST_GOLDILOCKS_PRIME as u128) as u64;
        if mag == 0 {
            AlmostGoldilocksField(0)
        } else {
            AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - mag)
        }
    }
}

// ============================================================================
// Base-field arithmetic helpers (thin wrappers over the `Add/Sub/Mul/Neg`
// impls; preserved for zk-torch-3 API parity).
// ============================================================================

#[inline]
pub fn agl_add(a: AlmostGoldilocksField, b: AlmostGoldilocksField) -> AlmostGoldilocksField {
    a + b
}

#[inline]
pub fn agl_sub(a: AlmostGoldilocksField, b: AlmostGoldilocksField) -> AlmostGoldilocksField {
    a - b
}

#[inline]
pub fn agl_mul(a: AlmostGoldilocksField, b: AlmostGoldilocksField) -> AlmostGoldilocksField {
    a * b
}

#[inline]
pub fn agl_neg(a: AlmostGoldilocksField) -> AlmostGoldilocksField {
    -a
}

/// Compute `a^{-1}` via Fermat (`a^{q-2} mod q`). Inverting 0 returns 0 — same
/// quirky-but-handy convention as zk-torch-3.
pub fn agl_inv(a: AlmostGoldilocksField) -> AlmostGoldilocksField {
    let a = a.reduce();
    if a.0 == 0 {
        return AlmostGoldilocksField(0);
    }
    calc_pow(a, ALMOST_GOLDILOCKS_PRIME - 2)
}

/// `base^exp` via repeated squaring.
pub fn calc_pow(base: AlmostGoldilocksField, exp: u64) -> AlmostGoldilocksField {
    if exp == 0 {
        return AlmostGoldilocksField(1);
    }
    let mut result = AlmostGoldilocksField(1);
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result * b;
        }
        b = b * b;
        e >>= 1;
    }
    result
}

/// `[1, alpha, alpha^2, ..., alpha^{n-1}]`.
pub fn calc_pow_vec(alpha: AlmostGoldilocksField, n: usize) -> Vec<AlmostGoldilocksField> {
    if n == 0 {
        return Vec::new();
    }
    let mut pow = Vec::with_capacity(n);
    pow.push(AlmostGoldilocksField(1));
    let mut current = AlmostGoldilocksField(1);
    for _ in 1..n {
        current = current * alpha;
        pow.push(current);
    }
    pow
}

// ============================================================================
// Ext2 helpers
// ============================================================================

#[inline]
pub fn ext2_add(a: AlmostGoldilocksExt2, b: AlmostGoldilocksExt2) -> AlmostGoldilocksExt2 {
    a + b
}

#[inline]
pub fn ext2_sub(a: AlmostGoldilocksExt2, b: AlmostGoldilocksExt2) -> AlmostGoldilocksExt2 {
    a - b
}

#[inline]
pub fn ext2_mul(a: AlmostGoldilocksExt2, b: AlmostGoldilocksExt2) -> AlmostGoldilocksExt2 {
    a * b
}

/// Inverse of `a = c0 + c1·X` in `F_q[X]/(X^2 − W)` via norm-based formula:
/// `a^{-1} = (c0 − c1·X) / (c0^2 − W·c1^2)`.
pub fn ext2_inv(a: AlmostGoldilocksExt2) -> AlmostGoldilocksExt2 {
    let w = AlmostGoldilocksField(AEXT2_W);
    let norm = a.c0 * a.c0 - w * a.c1 * a.c1;
    let norm_inv = agl_inv(norm);
    AlmostGoldilocksExt2::new(a.c0 * norm_inv, -(a.c1 * norm_inv))
}

/// Canonicalized equality. The underlying representation may store
/// `>= q` values; this normalizes before comparing.
pub fn ext2_field_eq(a: AlmostGoldilocksExt2, b: AlmostGoldilocksExt2) -> bool {
    a.c0.reduce().0 == b.c0.reduce().0 && a.c1.reduce().0 == b.c1.reduce().0
}

pub fn calc_pow_vec_ext2(alpha: AlmostGoldilocksExt2, n: usize) -> Vec<AlmostGoldilocksExt2> {
    if n == 0 {
        return Vec::new();
    }
    let mut pow = Vec::with_capacity(n);
    pow.push(AlmostGoldilocksExt2::one());
    let mut current = AlmostGoldilocksExt2::one();
    for _ in 1..n {
        current = current * alpha;
        pow.push(current);
    }
    pow
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_ceil_matches_textbook() {
        assert_eq!(log2_ceil(0), 0);
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(4), 2);
        assert_eq!(log2_ceil(5), 3);
        assert_eq!(log2_ceil(8), 3);
        assert_eq!(log2_ceil(16), 4);
        assert_eq!(log2_ceil(17), 5);
    }

    #[test]
    fn pow_2_basic() {
        assert_eq!(pow_2(0), 1);
        assert_eq!(pow_2(5), 32);
        assert_eq!(pow_2(20), 1 << 20);
    }

    #[test]
    fn get_n_sums_per_dim_bits() {
        assert_eq!(get_n(&[]), 0);
        assert_eq!(get_n(&[1]), 0);
        assert_eq!(get_n(&[2]), 1);
        assert_eq!(get_n(&[3]), 2);
        assert_eq!(get_n(&[4, 4]), 4);
        assert_eq!(get_n(&[7, 5]), 3 + 3);
        assert_eq!(get_n(&[1, 1024, 768]), 0 + 10 + 10);
    }

    #[test]
    fn next_pow_rounds_up() {
        assert_eq!(next_pow(0), 1);
        assert_eq!(next_pow(1), 1);
        assert_eq!(next_pow(2), 2);
        assert_eq!(next_pow(3), 4);
        assert_eq!(next_pow(1024), 1024);
        assert_eq!(next_pow(1025), 2048);
    }

    #[test]
    fn f_to_int_handles_negatives() {
        assert_eq!(f_to_int(AlmostGoldilocksField(0)), 0);
        assert_eq!(f_to_int(AlmostGoldilocksField(1)), 1);
        assert_eq!(f_to_int(AlmostGoldilocksField(100)), 100);
        assert_eq!(f_to_int(AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - 1)), -1);
        assert_eq!(f_to_int(AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - 100)), -100);
    }

    #[test]
    fn f_to_int_inverts_int_to_f() {
        for x in &[0i128, 1, -1, 42, -42, 1_000_000, -1_000_000, i64::MAX as i128, i64::MIN as i128] {
            let f = int_to_f(*x);
            let back = f_to_int(f);
            // For values within (-q/2, q/2) the round-trip is exact.
            if x.abs() < (ALMOST_GOLDILOCKS_PRIME as i128) / 2 {
                assert_eq!(back, *x, "round-trip failed for {}", x);
            }
        }
    }

    #[test]
    fn agl_arith_basic() {
        let a = AlmostGoldilocksField(7);
        let b = AlmostGoldilocksField(13);
        assert_eq!(agl_add(a, b), AlmostGoldilocksField(20));
        assert_eq!(agl_sub(b, a), AlmostGoldilocksField(6));
        assert_eq!(agl_mul(a, b), AlmostGoldilocksField(91));
        assert_eq!(agl_neg(a) + a, AlmostGoldilocksField(0));
    }

    #[test]
    fn agl_inv_satisfies_a_times_inv_eq_one() {
        for v in [1u64, 2, 3, 5, 17, 1000, ALMOST_GOLDILOCKS_PRIME - 1, ALMOST_GOLDILOCKS_PRIME - 7] {
            let a = AlmostGoldilocksField(v);
            let inv = agl_inv(a);
            let prod = agl_mul(a, inv).reduce();
            assert_eq!(prod, AlmostGoldilocksField(1), "v = {}", v);
        }
        // Inverse of zero is zero (convention).
        assert_eq!(agl_inv(AlmostGoldilocksField(0)).reduce(), AlmostGoldilocksField(0));
    }

    #[test]
    fn calc_pow_matches_naive() {
        let alpha = AlmostGoldilocksField(5);
        let mut acc = AlmostGoldilocksField(1);
        for e in 0u64..32 {
            let p = calc_pow(alpha, e).reduce();
            assert_eq!(p, acc.reduce(), "alpha^{} mismatch", e);
            acc = acc * alpha;
        }
    }

    #[test]
    fn calc_pow_vec_first_n_powers() {
        let alpha = AlmostGoldilocksField(11);
        let v = calc_pow_vec(alpha, 6);
        assert_eq!(v.len(), 6);
        let mut acc = AlmostGoldilocksField(1);
        for (i, x) in v.iter().enumerate() {
            assert_eq!(x.reduce(), acc.reduce(), "calc_pow_vec[{}]", i);
            acc = acc * alpha;
        }
        // Empty case.
        assert_eq!(calc_pow_vec(alpha, 0).len(), 0);
    }

    #[test]
    fn ext2_inv_satisfies_a_times_inv_eq_one() {
        let cases = [
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(1), AlmostGoldilocksField(0)),
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(0), AlmostGoldilocksField(1)),
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(7), AlmostGoldilocksField(13)),
            AlmostGoldilocksExt2::new(AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - 3), AlmostGoldilocksField(42)),
        ];
        for a in cases {
            let inv = ext2_inv(a);
            let prod = a * inv;
            // prod should be the multiplicative identity: c0=1, c1=0 (canonically).
            assert!(ext2_field_eq(prod, AlmostGoldilocksExt2::one()), "a*inv != 1 for a = {:?}", a);
        }
    }

    #[test]
    fn ext2_field_eq_normalizes_inputs() {
        let a = AlmostGoldilocksExt2::new(AlmostGoldilocksField(5), AlmostGoldilocksField(9));
        // Same values stored with a non-canonical rep (5 + q).
        let b = AlmostGoldilocksExt2::new(
            AlmostGoldilocksField(5u64.wrapping_add(ALMOST_GOLDILOCKS_PRIME)),
            AlmostGoldilocksField(9u64.wrapping_add(ALMOST_GOLDILOCKS_PRIME)),
        );
        // Direct == compares wrapped reps so it'd reject; ext2_field_eq accepts.
        assert!(ext2_field_eq(a, b));
    }

    #[test]
    fn calc_pow_vec_ext2_first_n_powers() {
        let alpha = AlmostGoldilocksExt2::new(AlmostGoldilocksField(2), AlmostGoldilocksField(3));
        let v = calc_pow_vec_ext2(alpha, 5);
        assert_eq!(v.len(), 5);
        let mut acc = AlmostGoldilocksExt2::one();
        for (i, x) in v.iter().enumerate() {
            assert!(ext2_field_eq(*x, acc), "calc_pow_vec_ext2[{}]", i);
            acc = acc * alpha;
        }
    }
}
