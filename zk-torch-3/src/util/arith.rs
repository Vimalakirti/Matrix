use crate::GOLDILOCKS_PRIME;
use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

/// Compute ceil(log2(n)), returns 0 for n <= 1.
pub fn log2_ceil(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as usize
}

/// Compute 2^k.
pub fn pow_2(k: usize) -> usize {
    1usize << k
}

/// Calculate the number of variables needed for a given shape.
/// Each dimension is rounded up to the next power of two, then we take log2.
pub fn get_n(shape: &[usize]) -> usize {
    shape
        .iter()
        .map(|&s| log2_ceil(s.max(1)))
        .sum()
}

/// Convert a field element to a signed integer.
/// Elements > p/2 are treated as negative.
pub fn f_to_int(f: GoldilocksField) -> i128 {
    let v = f.0;
    if v > GOLDILOCKS_PRIME / 2 {
        v as i128 - GOLDILOCKS_PRIME as i128
    } else {
        v as i128
    }
}

/// Convert a signed integer to a field element.
pub fn int_to_f(x: i128) -> GoldilocksField {
    if x >= 0 {
        GoldilocksField((x as u64) % GOLDILOCKS_PRIME)
    } else {
        let neg = ((-x) as u64) % GOLDILOCKS_PRIME;
        if neg == 0 {
            GoldilocksField(0)
        } else {
            GoldilocksField(GOLDILOCKS_PRIME - neg)
        }
    }
}

/// Compute [1, alpha, alpha^2, ..., alpha^{n-1}] — vector of powers.
pub fn calc_pow_vec(alpha: GoldilocksField, n: usize) -> Vec<GoldilocksField> {
    if n == 0 {
        return Vec::new();
    }
    let mut pow = Vec::with_capacity(n);
    pow.push(GoldilocksField(1));
    let mut current = GoldilocksField(1);
    for _ in 1..n {
        current = gl_mul(current, alpha);
        pow.push(current);
    }
    pow
}

/// Compute base^exp using repeated squaring in the field.
pub fn calc_pow(base: GoldilocksField, exp: u64) -> GoldilocksField {
    if exp == 0 {
        return GoldilocksField(1);
    }
    let mut result = GoldilocksField(1);
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = gl_mul(result, b);
        }
        b = gl_mul(b, b);
        e >>= 1;
    }
    result
}

/// Round up to the next power of two.
pub fn next_pow(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    (n as u32).next_power_of_two()
}

// CPU field arithmetic (matching goldilocks-cuda conventions)

pub fn gl_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    let (sum, carry) = a.0.overflowing_add(b.0);
    let (result, carry2) = sum.overflowing_sub(GOLDILOCKS_PRIME);
    if carry || !carry2 {
        GoldilocksField(result)
    } else {
        GoldilocksField(sum)
    }
}

pub fn gl_sub(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    if a.0 >= b.0 {
        GoldilocksField(a.0 - b.0)
    } else {
        GoldilocksField(GOLDILOCKS_PRIME - (b.0 - a.0))
    }
}

pub fn gl_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
    let full = (a.0 as u128) * (b.0 as u128);
    GoldilocksField(reduce128(full))
}

pub fn gl_neg(a: GoldilocksField) -> GoldilocksField {
    if a.0 == 0 {
        GoldilocksField(0)
    } else {
        GoldilocksField(GOLDILOCKS_PRIME - a.0)
    }
}

pub fn gl_inv(a: GoldilocksField) -> GoldilocksField {
    if a.0 == 0 {
        return GoldilocksField(0);
    }
    // Fermat's little theorem: a^(p-2) mod p
    calc_pow(a, GOLDILOCKS_PRIME - 2)
}

// ============================================================================
// Ext2 host arithmetic
// ============================================================================

pub fn ext2_add(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    a + b
}

pub fn ext2_sub(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    a - b
}

pub fn ext2_mul(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    a * b
}

/// Inverse of Ext2 element using norm-based inversion:
/// (a0 + a1*X)^{-1} = (a0 - a1*X) / (a0^2 - W*a1^2)
pub fn ext2_inv(a: GoldilocksExt2) -> GoldilocksExt2 {
    let w = GoldilocksField(goldilocks_cuda::EXT2_W);
    let norm = gl_sub(gl_mul(a.c0, a.c0), gl_mul(w, gl_mul(a.c1, a.c1)));
    let norm_inv = gl_inv(norm);
    GoldilocksExt2::new(gl_mul(a.c0, norm_inv), gl_neg(gl_mul(a.c1, norm_inv)))
}

/// Compare two GoldilocksExt2 values with canonical normalization.
/// GPU may store non-canonical values (>= p), so raw == comparison can fail
/// for values that are mathematically equal.
pub fn ext2_field_eq(a: GoldilocksExt2, b: GoldilocksExt2) -> bool {
    const P: u64 = crate::GOLDILOCKS_PRIME;
    (a.c0.0 % P == b.c0.0 % P) && (a.c1.0 % P == b.c1.0 % P)
}

/// Compute [1, alpha, alpha^2, ..., alpha^{n-1}] for Ext2.
pub fn calc_pow_vec_ext2(alpha: GoldilocksExt2, n: usize) -> Vec<GoldilocksExt2> {
    if n == 0 {
        return Vec::new();
    }
    let mut pow = Vec::with_capacity(n);
    pow.push(GoldilocksExt2::one());
    let mut current = GoldilocksExt2::one();
    for _ in 1..n {
        current = ext2_mul(current, alpha);
        pow.push(current);
    }
    pow
}

fn reduce128(x: u128) -> u64 {
    // p = 2^64 - 2^32 + 1, so 2^64 ≡ 2^32 - 1 (mod p)
    // x = hi * 2^64 + lo ≡ lo + hi * (2^32 - 1) (mod p)
    let lo = x as u64;
    let hi = (x >> 64) as u64;

    // Compute hi * (2^32 - 1) = hi * 2^32 - hi
    // This can overflow u64, so use u128
    let mid = (hi as u128) * ((1u128 << 32) - 1);
    let sum = lo as u128 + mid;

    // sum fits in ~97 bits. Reduce again: sum = sum_hi * 2^64 + sum_lo
    let sum_lo = sum as u64;
    let sum_hi = (sum >> 64) as u64;

    if sum_hi == 0 {
        if sum_lo >= GOLDILOCKS_PRIME {
            sum_lo - GOLDILOCKS_PRIME
        } else {
            sum_lo
        }
    } else {
        // Second round: sum_hi is small (at most ~33 bits)
        let mid2 = (sum_hi as u128) * ((1u128 << 32) - 1);
        let sum2 = sum_lo as u128 + mid2;
        let mut result = (sum2 % (GOLDILOCKS_PRIME as u128)) as u64;
        if result >= GOLDILOCKS_PRIME {
            result -= GOLDILOCKS_PRIME;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log2_ceil() {
        assert_eq!(log2_ceil(0), 0);
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(4), 2);
        assert_eq!(log2_ceil(5), 3);
        assert_eq!(log2_ceil(8), 3);
        assert_eq!(log2_ceil(16), 4);
    }

    #[test]
    fn test_f_to_int() {
        assert_eq!(f_to_int(GoldilocksField(0)), 0);
        assert_eq!(f_to_int(GoldilocksField(1)), 1);
        assert_eq!(f_to_int(GoldilocksField(100)), 100);
        // Negative
        assert_eq!(f_to_int(GoldilocksField(GOLDILOCKS_PRIME - 1)), -1);
        assert_eq!(f_to_int(GoldilocksField(GOLDILOCKS_PRIME - 100)), -100);
    }

    #[test]
    fn test_field_arithmetic() {
        let a = GoldilocksField(7);
        let b = GoldilocksField(13);
        assert_eq!(gl_add(a, b), GoldilocksField(20));
        assert_eq!(gl_sub(b, a), GoldilocksField(6));
        assert_eq!(gl_mul(a, b), GoldilocksField(91));

        // Test inverse
        let inv_a = gl_inv(a);
        assert_eq!(gl_mul(a, inv_a), GoldilocksField(1));
    }
}
