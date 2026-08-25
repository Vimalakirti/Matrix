//! Signed bit decomposition for Ajtai commits (plan §2).
//!
//! A signed integer value `v ∈ [−2^(b-1), 2^(b-1))` decomposes into `b`
//! binary planes `f_0, …, f_{b-1}` with
//!
//!     v  =  Σ_{i=0..b-2} 2^i · f_i  −  2^(b-1) · f_{b-1}
//!
//! Concretely this is `b`-bit two's-complement, with `f_{b-1}` carrying the
//! sign. The Ajtai commit kernel sees each plane as a flat binary witness
//! packed into `u64` bitmasks (one bit per witness position, 64 bits per
//! `u64`); helpers in this module produce that packed form and broadcast a
//! short-arity plane up to `max_num_vars` width when needed.

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::util::arith::f_to_int;

// ============================================================================
// Higher-radix (base-β) decomposition — generalizes the binary scheme.
//
// Math: for value range [−R, R), pick `b` such that `β^b ≥ 2R`. Map signed v
// to unsigned u = v mod β^b ∈ [0, β^b), then write u = Σ_i β^i d_i with
// digits d_i ∈ {0..β-1}. The verifier reconstructs the value-commitment via
// the homomorphism c_v = Σ_i β^i c_{d_i}.
//
// Plane count vs binary:                        bit-plane equivalent of one
//                                               digit-plane (β = 2^k):
//   binary (β=2)   : b = ⌈log₂(2R)⌉   = 21        1
//   base-4   (k=2) :     ⌈21/2⌉       = 11        2
//   base-16  (k=4) :     ⌈21/4⌉       =  6        4
//   base-64  (k=6) :     ⌈21/6⌉       =  4        6
//
// The fold-tree LEAF count drops to `b_β`; each digit-plane is stored as
// `k = log₂β` internal binary bit-planes so the existing binary fast paths
// (selective F_u, binary round-0, TC multifold) can be reused log₂β times per
// digit with `2^k`-weighted combination — see `radix_to_bit_planes` /
// `radix_from_bit_planes`. SIS binding norm grows from 1 to β-1 (≤63 for
// β≤64), negligible against q ≈ 2⁶⁴.
// ============================================================================

/// Number of base-β digit-planes needed to represent signed values in
/// `[−2^(b-1), 2^(b-1))`, i.e. ⌈b / log₂β⌉. β must be a power of 2 ≥ 2.
pub fn digit_planes_for(b: usize, base: usize) -> usize {
    assert!(base >= 2 && base.is_power_of_two(),
        "base must be a power of 2 ≥ 2; got {}", base);
    let k = base.trailing_zeros() as usize; // log₂β
    (b + k - 1) / k
}

/// Decompose each value into `b_β = digit_planes_for(b, base)` base-β digits
/// using signed two's-complement in base β: the sign is carried by the top
/// digit being ≥ β/2. Returns `digits[i][k] = i-th digit of value k`, each
/// digit in `{0..base-1}`. Panics if any value is outside `[−2^(b-1), 2^(b-1))`.
///
/// Round-trip inverse of [`radix_reconstruct_signed`].
pub fn radix_decompose_signed(
    values: &[AlmostGoldilocksField],
    b: usize,
    base: usize,
) -> Vec<Vec<u32>> {
    assert!(base >= 2 && base.is_power_of_two(), "base must be power of 2 ≥ 2");
    assert!(b >= 1 && b <= 60, "b must be in [1, 60], got {}", b);
    let k = base.trailing_zeros() as usize;
    let b_beta = digit_planes_for(b, base);
    // β^{b_β} — the wrap-around modulus. With β = 2^k and b_β ≥ b/k, we have
    // β^{b_β} ≥ 2^b. Use the same 2^b modulus so binary (k=1) is identical.
    let modulus: i128 = 1i128 << (b_beta * k);
    let half: i128 = modulus / 2;
    let mut digits = vec![vec![0u32; values.len()]; b_beta];
    let base_u128 = base as u128;
    for (idx, &v) in values.iter().enumerate() {
        let signed = f_to_int(v);
        assert!(
            signed >= -half && signed < half,
            "value {} out of range [-{}, {}) for b={} base={} ({} digit planes)",
            signed, half, half, b, base, b_beta,
        );
        let mut u: u128 = if signed >= 0 { signed as u128 } else { (signed + modulus) as u128 };
        for i in 0..b_beta {
            digits[i][idx] = (u % base_u128) as u32;
            u /= base_u128;
        }
    }
    digits
}

/// Reconstruct signed integers from base-β digit planes:
/// `u = Σ_i β^i d_i`; `v = u − β^{b_β}` if `u ≥ β^{b_β}/2`, else `u`.
pub fn radix_reconstruct_signed(digits: &[Vec<u32>], base: usize) -> Vec<i128> {
    assert!(!digits.is_empty(), "reconstruct: need at least one plane");
    assert!(base >= 2 && base.is_power_of_two(), "base must be power of 2 ≥ 2");
    let k = base.trailing_zeros() as usize;
    let b_beta = digits.len();
    let n = digits[0].len();
    for (i, p) in digits.iter().enumerate() {
        assert_eq!(p.len(), n, "plane {} length mismatch", i);
    }
    let modulus: i128 = 1i128 << (b_beta * k);
    let half: i128 = modulus / 2;
    let base_i128 = base as i128;
    (0..n).map(|j| {
        let mut u: i128 = 0;
        let mut beta_i: i128 = 1;
        for i in 0..b_beta {
            u += (digits[i][j] as i128) * beta_i;
            beta_i *= base_i128;
        }
        if u >= half { u - modulus } else { u }
    }).collect()
}

/// Decompose each digit-plane (digits in {0..β-1}) into `log₂β` internal
/// binary bit-planes for processing through the existing binary fold-tree
/// kernels. Plane `(i, k)` carries bit `k` of digit `i`. So a base-β scheme
/// has `b_β` digit-planes, each storing `log₂β` bit-planes — total bit count
/// `b_β · log₂β ≥ b`, matching the binary view at the bit level while giving
/// the fold tree only `b_β` leaves.
pub fn radix_to_bit_planes(digits: &[Vec<u32>], base: usize) -> Vec<Vec<Vec<bool>>> {
    assert!(base >= 2 && base.is_power_of_two());
    let k = base.trailing_zeros() as usize;
    let n = if digits.is_empty() { 0 } else { digits[0].len() };
    digits.iter().map(|digit_plane| {
        (0..k).map(|bit_k| {
            (0..n).map(|j| ((digit_plane[j] >> bit_k) & 1) == 1).collect()
        }).collect()
    }).collect()
}

/// Inverse of [`radix_to_bit_planes`]: combine `log₂β` bit-planes per digit
/// into a digit-plane with values in `{0..base-1}`.
pub fn radix_from_bit_planes(bit_planes: &[Vec<Vec<bool>>], base: usize) -> Vec<Vec<u32>> {
    assert!(base >= 2 && base.is_power_of_two());
    let k = base.trailing_zeros() as usize;
    bit_planes.iter().map(|digit_bits| {
        assert_eq!(digit_bits.len(), k, "expected {} bit-planes per digit", k);
        let n = digit_bits[0].len();
        (0..n).map(|j| {
            let mut d: u32 = 0;
            for bit_k in 0..k {
                if digit_bits[bit_k][j] { d |= 1u32 << bit_k; }
            }
            d
        }).collect()
    }).collect()
}

/// Decompose each value into `b` binary planes using the signed two's-
/// complement convention. Panics if any value is outside the range
/// `[−2^(b-1), 2^(b-1))`.
///
/// Returns `planes[i][k] = bit i of value k` (so each plane has the same
/// length as the input). Linearity: the recovery
/// `Σ 2^i · planes[i] − 2^(b-1) · planes[b-1]` exactly reproduces the input
/// (modulo the modulus).
pub fn bit_decompose_signed(values: &[AlmostGoldilocksField], b: usize) -> Vec<Vec<bool>> {
    assert!(b >= 1 && b <= 127, "b must be in [1, 127], got {}", b);
    let half: i128 = 1i128 << (b - 1);
    let modulus: i128 = 1i128 << b;
    let mut planes = vec![vec![false; values.len()]; b];

    for (idx, &v) in values.iter().enumerate() {
        let signed = f_to_int(v);
        assert!(
            signed >= -half && signed < half,
            "value {} out of range [-2^{}, 2^{}) for b={} signed decomposition",
            signed,
            b - 1,
            b - 1,
            b,
        );
        // Convert to unsigned `b`-bit representation: standard two's-complement.
        let unsigned: u128 = if signed >= 0 {
            signed as u128
        } else {
            (signed + modulus) as u128
        };
        for i in 0..b {
            planes[i][idx] = (unsigned >> i) & 1 == 1;
        }
    }
    planes
}

/// Reconstruct the signed-integer representation from `b` binary planes via
/// `Σ 2^i · planes[i] − 2^(b-1) · planes[b-1]`. Round-trip inverse of
/// [`bit_decompose_signed`].
pub fn reconstruct_signed(planes: &[Vec<bool>]) -> Vec<i128> {
    assert!(!planes.is_empty(), "reconstruct: need at least one plane");
    let b = planes.len();
    let n = planes[0].len();
    for (i, p) in planes.iter().enumerate() {
        assert_eq!(p.len(), n, "plane {} length {} != plane 0 length {}", i, p.len(), n);
    }
    (0..n)
        .map(|k| {
            let mut acc: i128 = 0;
            for i in 0..(b - 1) {
                if planes[i][k] {
                    acc += 1i128 << i;
                }
            }
            if planes[b - 1][k] {
                acc -= 1i128 << (b - 1);
            }
            acc
        })
        .collect()
}

/// Pack a boolean plane into `u64` bitmasks. Position `k` of the plane maps
/// to bit `k % 64` of `u64` `k / 64`. Trailing bits past the input length
/// are zero. Caller is responsible for the length being a multiple of 64
/// (or padding into a longer plane first); we accept arbitrary lengths and
/// round up.
pub fn pack_bits(plane: &[bool]) -> Vec<u64> {
    let n_u64s = (plane.len() + 63) / 64;
    let mut packed = vec![0u64; n_u64s];
    for (idx, &bit) in plane.iter().enumerate() {
        if bit {
            packed[idx / 64] |= 1u64 << (idx % 64);
        }
    }
    packed
}

/// Broadcast a packed binary plane from arity `k` (input has `2^k` bits) up
/// to arity `max_num_vars` (output has `2^max_num_vars` bits, i.e.
/// `2^(max_num_vars - 6)` `u64`s). Repeats the input `2^(max_num_vars − k)`
/// times — matches the MLE broadcast semantics where a `k`-variate
/// polynomial is lifted to `max_num_vars` variables by ignoring the extra
/// variables.
///
/// Two regimes:
/// - `k ≥ 6` (input has ≥ 64 bits): repeat the input `u64` buffer
///   `2^(max_num_vars − k)` times.
/// - `k < 6` (input has < 64 bits, packed into the low bits of a single
///   `u64`): tile the bit pattern within a `u64` then fill the buffer.
pub fn broadcast_packed(packed: &[u64], k: usize, max_num_vars: usize) -> Vec<u64> {
    assert!(
        k <= max_num_vars,
        "broadcast: input arity {} > max_num_vars {}",
        k,
        max_num_vars,
    );
    let target_bits = 1usize << max_num_vars;
    let target_u64s = target_bits / 64;
    // At least one u64 so the kernel always sees a valid buffer.
    let target_u64s = target_u64s.max(1);

    if k == max_num_vars {
        return packed.to_vec();
    }

    if k >= 6 {
        let src_u64s = 1usize << (k - 6);
        assert_eq!(
            packed.len(),
            src_u64s,
            "broadcast: input length {} u64s != 2^(k-6) = {}",
            packed.len(),
            src_u64s,
        );
        let reps = 1usize << (max_num_vars - k);
        let mut out = Vec::with_capacity(target_u64s);
        for _ in 0..reps {
            out.extend_from_slice(packed);
        }
        debug_assert_eq!(out.len(), target_u64s);
        out
    } else {
        // Sub-u64 case: the input is the low `2^k` bits of `packed[0]`.
        let n_bits = 1usize << k;
        let mask = if n_bits == 64 { u64::MAX } else { (1u64 << n_bits) - 1 };
        let original = packed[0] & mask;
        let reps_per_u64 = 64 / n_bits;
        let mut tile: u64 = 0;
        for r in 0..reps_per_u64 {
            tile |= original << (r * n_bits);
        }
        vec![tile; target_u64s]
    }
}

/// Convenience: decompose a witness's evaluation table into `b` packed
/// bit-plane buffers, each broadcast to `max_num_vars`. Output: `b` vectors
/// of length `2^(max_num_vars − 6)` `u64`s each.
///
/// **Legacy path** — used by the broadcast-to-max-arity commit. The
/// per-arity (Option A) commit path uses [`decompose_and_pack_native`]
/// instead, which stays at the native arity `k` and lets the fold
/// tree bucket leaves accordingly.
pub fn decompose_and_pack(
    values: &[AlmostGoldilocksField],
    b: usize,
    k: usize,
    max_num_vars: usize,
) -> Vec<Vec<u64>> {
    let planes = bit_decompose_signed(values, b);
    planes
        .into_iter()
        .map(|plane| {
            let packed = pack_bits(&plane);
            broadcast_packed(&packed, k, max_num_vars)
        })
        .collect()
}

/// Bit-decompose + pack at native arity `k` (no broadcast). Output: `b`
/// vectors of length `2^max(k − 6, 0)` `u64`s each (one full word when
/// `k < 6`, since each `u64` packs 64 ring coefficients). This is the
/// production path for the per-arity commit (M_k = first 2^k columns
/// of M_max).
///
/// **Parallel fused implementation**: processes values in groups of 64,
/// extracts the `b` bits per value, and emits the `b` plane u64s for
/// that group directly — no `Vec<Vec<bool>>` intermediate. Cuts memory
/// allocation by ~64× and gives a linear rayon speedup over groups.
pub fn decompose_and_pack_native(
    values: &[AlmostGoldilocksField],
    b: usize,
    k: usize,
) -> Vec<Vec<u64>> {
    assert!(b >= 1 && b <= 127, "b must be in [1, 127], got {}", b);
    use rayon::prelude::*;

    let half: i128 = 1i128 << (b - 1);
    let modulus: i128 = 1i128 << b;
    let n = values.len();
    let needed = (1usize << k.max(6)).div_ceil(64); // u64s per plane
    let group_count = needed; // one u64 per plane per group of 64 values

    // For each group of 64 values, produce `b` u64s (one per plane).
    // Output layout: planes[plane_idx][group_idx]; we transpose at the end.
    let per_group: Vec<Vec<u64>> = (0..group_count)
        .into_par_iter()
        .map(|g| {
            let base = g * 64;
            let mut out = vec![0u64; b];
            for kk in 0..64 {
                let idx = base + kk;
                if idx >= n { break; }
                let signed = f_to_int(values[idx]);
                debug_assert!(signed >= -half && signed < half,
                    "value {} out of range [-2^{}, 2^{}) for b={}", signed, b-1, b-1, b);
                let unsigned: u128 = if signed >= 0 {
                    signed as u128
                } else {
                    (signed + modulus) as u128
                };
                let bit_pos = 1u64 << kk;
                for plane_i in 0..b {
                    if (unsigned >> plane_i) & 1 == 1 {
                        out[plane_i] |= bit_pos;
                    }
                }
            }
            out
        })
        .collect();

    // Transpose: planes[plane_i][group] = per_group[group][plane_i].
    (0..b).map(|plane_i| {
        per_group.iter().map(|g| g[plane_i]).collect()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::arith::int_to_f;

    fn agl_of(v: i128) -> AlmostGoldilocksField {
        int_to_f(v)
    }

    #[test]
    fn bit_decompose_positive_values_b8() {
        // b = 8, range [-128, 128).
        let vs: Vec<_> = [0i128, 1, 2, 7, 15, 42, 127].iter().map(|&v| agl_of(v)).collect();
        let planes = bit_decompose_signed(&vs, 8);
        let back = reconstruct_signed(&planes);
        assert_eq!(back, [0, 1, 2, 7, 15, 42, 127]);
    }

    #[test]
    fn bit_decompose_negative_values_b8() {
        let vs: Vec<_> = [-1i128, -2, -7, -42, -128].iter().map(|&v| agl_of(v)).collect();
        let planes = bit_decompose_signed(&vs, 8);
        let back = reconstruct_signed(&planes);
        assert_eq!(back, [-1, -2, -7, -42, -128]);
    }

    /// `f_{b-1}` is the sign bit: 1 iff value is negative.
    #[test]
    fn bit_decompose_sign_bit_correct() {
        let vs: Vec<_> = [0i128, 1, -1, 50, -50, 127, -128].iter().map(|&v| agl_of(v)).collect();
        let planes = bit_decompose_signed(&vs, 8);
        let sign = &planes[7];
        assert_eq!(sign, &[false, false, true, false, true, false, true]);
    }

    #[test]
    fn bit_decompose_roundtrip_b21() {
        let raw: Vec<i128> = vec![
            0,
            1,
            -1,
            (1i128 << 20) - 1,
            -(1i128 << 20),
            123_456,
            -987_654,
        ];
        let vs: Vec<_> = raw.iter().map(|&v| agl_of(v)).collect();
        let planes = bit_decompose_signed(&vs, 21);
        assert_eq!(planes.len(), 21);
        let back = reconstruct_signed(&planes);
        assert_eq!(back, raw);
    }

    /// Roundtrip across the full b=21 signed range on a randomized sample.
    #[test]
    fn bit_decompose_roundtrip_random_b21() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xB17C0DE);
        let n = 256;
        let max = 1i128 << 20;
        let raw: Vec<i128> = (0..n).map(|_| rng.gen_range(-max..max)).collect();
        let vs: Vec<_> = raw.iter().map(|&v| agl_of(v)).collect();
        let planes = bit_decompose_signed(&vs, 21);
        let back = reconstruct_signed(&planes);
        assert_eq!(back, raw);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bit_decompose_rejects_overflow() {
        let vs = vec![agl_of(128)]; // out of [-128, 128) for b=8
        let _ = bit_decompose_signed(&vs, 8);
    }

    // === Higher-radix (base-β) round-trips ===

    #[test]
    fn radix_plane_counts() {
        // b=21 (range ±2^20): plane count = ⌈21 / log₂β⌉.
        assert_eq!(digit_planes_for(21, 2), 21);
        assert_eq!(digit_planes_for(21, 4), 11);
        assert_eq!(digit_planes_for(21, 16), 6);
        assert_eq!(digit_planes_for(21, 64), 4);
        assert_eq!(digit_planes_for(8, 4), 4);
        assert_eq!(digit_planes_for(8, 16), 2);
    }

    /// At base=2 the radix decomposition is bit-exactly the binary one
    /// (modulo the bool→u32 type change). Regression guard for backward compat.
    #[test]
    fn radix_base2_matches_binary() {
        let vs: Vec<_> = [0i128, 1, -1, 42, -42, 127, -128].iter().map(|&v| agl_of(v)).collect();
        let binary = bit_decompose_signed(&vs, 8);
        let radix = radix_decompose_signed(&vs, 8, 2);
        assert_eq!(binary.len(), radix.len());
        for i in 0..binary.len() {
            for j in 0..binary[i].len() {
                assert_eq!(binary[i][j] as u32, radix[i][j], "mismatch at plane {} value {}", i, j);
            }
        }
    }

    /// Round-trip every (signed) value across all the practical bases.
    #[test]
    fn radix_roundtrip_all_bases() {
        let raw: Vec<i128> = vec![
            0, 1, -1,
            (1i128 << 20) - 1,
            -(1i128 << 20),
            123_456, -987_654, 1, -1, 1024, -1024, 65535, -65535,
        ];
        let vs: Vec<_> = raw.iter().map(|&v| agl_of(v)).collect();
        for &base in &[2usize, 4, 16, 64] {
            let digits = radix_decompose_signed(&vs, 21, base);
            assert_eq!(digits.len(), digit_planes_for(21, base),
                "plane count for base={}", base);
            // Each digit must be in {0..base-1}.
            for plane in &digits {
                for &d in plane { assert!((d as usize) < base, "digit overflow at base={}", base); }
            }
            let back = radix_reconstruct_signed(&digits, base);
            assert_eq!(back, raw, "round-trip failed at base={}", base);
        }
    }

    /// Randomized round-trip — every base must invert correctly across a
    /// uniform sample of the full b=21 signed range.
    #[test]
    fn radix_roundtrip_random() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0DEBABE);
        let n = 256;
        let max = 1i128 << 20;
        let raw: Vec<i128> = (0..n).map(|_| rng.gen_range(-max..max)).collect();
        let vs: Vec<_> = raw.iter().map(|&v| agl_of(v)).collect();
        for &base in &[2usize, 4, 16, 64] {
            let digits = radix_decompose_signed(&vs, 21, base);
            let back = radix_reconstruct_signed(&digits, base);
            assert_eq!(back, raw, "random round-trip failed at base={}", base);
        }
    }

    /// `radix_to_bit_planes` and `radix_from_bit_planes` are mutual
    /// inverses. This is the layout the fold tree will store internally:
    /// each digit-plane becomes `log₂β` packed bit-planes so the existing
    /// binary kernels can be reused with `2^k` weighting.
    #[test]
    fn radix_bit_plane_layout_roundtrip() {
        let raw: Vec<i128> = vec![0, 1, 2, 3, 15, 63, -1, -42, 127, -128];
        let vs: Vec<_> = raw.iter().map(|&v| agl_of(v)).collect();
        for &base in &[4usize, 16, 64] {
            let digits = radix_decompose_signed(&vs, 21, base);
            let bit_planes = radix_to_bit_planes(&digits, base);
            // Each digit-plane should now be log₂β bit-planes.
            let k = base.trailing_zeros() as usize;
            for dp in &bit_planes { assert_eq!(dp.len(), k); }
            let digits_back = radix_from_bit_planes(&bit_planes, base);
            assert_eq!(digits_back, digits, "bit-plane layout failed at base={}", base);
            // And combined with reconstruction we recover the original values.
            let back = radix_reconstruct_signed(&digits_back, base);
            assert_eq!(back, raw, "end-to-end failed at base={}", base);
        }
    }

    #[test]
    fn pack_bits_low_bits_first() {
        let plane = vec![true, false, true, true, false, false, false, false];
        // = 0b00001101 = 13.
        assert_eq!(pack_bits(&plane), vec![0b1101u64]);
    }

    #[test]
    fn pack_bits_handles_64_boundary() {
        let mut plane = vec![false; 128];
        plane[0] = true;   // bit 0 of u64 0
        plane[63] = true;  // bit 63 of u64 0
        plane[64] = true;  // bit 0 of u64 1
        plane[127] = true; // bit 63 of u64 1
        let packed = pack_bits(&plane);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], (1u64 << 0) | (1u64 << 63));
        assert_eq!(packed[1], (1u64 << 0) | (1u64 << 63));
    }

    #[test]
    fn broadcast_no_op_when_k_equals_max() {
        let packed = vec![0xDEADBEEFu64, 0xCAFEBABE];
        let out = broadcast_packed(&packed, 7, 7);
        assert_eq!(out, packed);
    }

    /// 64-bit input (k=6) broadcast to 256 bits (max=8) → repeats 4×.
    #[test]
    fn broadcast_k6_to_max8_repeats_4x() {
        let packed = vec![0x123456789ABCDEFu64];
        let out = broadcast_packed(&packed, 6, 8);
        assert_eq!(out.len(), 4);
        assert_eq!(out, vec![0x123456789ABCDEF; 4]);
    }

    /// 2-bit input (k=1) broadcast to 64 bits: tile the 2-bit pattern.
    /// k=1 → 2 bits in the input. Pattern `0b10` (= value bit-0 = 0, value bit-1 = 1).
    #[test]
    fn broadcast_k1_to_max6_tiles_within_u64() {
        let packed = vec![0b10u64];
        let out = broadcast_packed(&packed, 1, 6);
        assert_eq!(out.len(), 1);
        // 32 copies of `10` in 64 bits = 0xAAAAAAAAAAAAAAAA.
        assert_eq!(out[0], 0xAAAA_AAAA_AAAA_AAAA);
    }

    /// 4-bit input (k=2 → 4 bits), broadcast to 128 bits = 2 u64s.
    /// Pattern `0b1011` tiles within each u64 (16 copies) → each u64 has the
    /// same value.
    #[test]
    fn broadcast_k2_to_max7_tiles_then_repeats() {
        let packed = vec![0b1011u64];
        let out = broadcast_packed(&packed, 2, 7);
        assert_eq!(out.len(), 2);
        // 16 copies of `1011` per u64.
        let mut expected = 0u64;
        for r in 0..16 {
            expected |= 0b1011u64 << (r * 4);
        }
        assert_eq!(out[0], expected);
        assert_eq!(out[1], expected);
    }

    /// decompose_and_pack output shapes are `b × 2^(max-6)` u64s each.
    #[test]
    fn decompose_and_pack_shape() {
        let raw: Vec<i128> = vec![1, -1, 50, -50, 0, 7];
        let mut padded = raw.clone();
        // arity k = 3 → 8 entries needed; pad with zeros.
        padded.resize(8, 0);
        let vs: Vec<_> = padded.iter().map(|&v| agl_of(v)).collect();
        let b = 8;
        let k = 3;
        let max = 8;
        let planes = decompose_and_pack(&vs, b, k, max);
        assert_eq!(planes.len(), b);
        let expected_u64s = 1usize << (max - 6);
        for (i, p) in planes.iter().enumerate() {
            assert_eq!(p.len(), expected_u64s, "plane {} length", i);
        }
    }
}
