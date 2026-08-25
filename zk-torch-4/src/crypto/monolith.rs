//! Monolith permutation over the almost-Goldilocks field
//! (`q = 2^64 − 2^32 − 31 = 0xFFFFFFFEFFFFFFE1`).
//!
//! Mirrors the structure of `goldilocks-cuda-rs::cpu_monolith` (`WIDTH = 12`,
//! `NUM_BARS = 4`, `N_ROUNDS = 6`, Bars → Bricks → Concrete schedule). The
//! `bar_64` S-box and the `MDS` matrix are field-agnostic (small-integer
//! entries), so only the field arithmetic in `bricks` / `concrete` and the
//! per-round constants change for AlmostGoldilocks.
//!
//! ## Round constants
//!
//! The Goldilocks Monolith reference (`cpu_monolith.rs`) uses constants
//! derived from a Plonky3 Grain-LFSR. We cannot byte-reproduce that here
//! without the upstream script, so AGL constants are derived deterministically
//! from SHA-256 with the explicit domain tag
//! `"AGL-Monolith-RC-v1" || round.to_le_bytes() || pos.to_le_bytes() || salt.to_le_bytes()`,
//! taking the first 8 bytes (little-endian) and rejection-sampling until
//! `val < q`. Rejection probability per draw is `(2^64 − q) / 2^64 ≈ 2^{-58}`,
//! so the salt counter is effectively never used past 0; it's wired in for
//! cryptographic hygiene.
//!
//! Following the Monolith convention, the round-0 (initial Concrete) and
//! the round-`N_ROUNDS` (final Concrete) constants are all zero — only the
//! interior rounds `1..N_ROUNDS` receive nontrivial constants.

use almost_goldilocks_cuda::field::{AlmostGoldilocksField, ALMOST_GOLDILOCKS_PRIME};
use sha2::{Digest as Sha2Digest, Sha256};
use std::sync::OnceLock;

pub const WIDTH: usize = 12;
pub const NUM_BARS: usize = 4;
pub const N_ROUNDS: usize = 6;

/// Number of `concrete` invocations per permutation = `N_ROUNDS + 1`
/// (one initial wrap + one per round).
pub const NUM_CONCRETE: usize = N_ROUNDS + 1;

/// Domain-separation tag for SHA-256-based round-constant derivation.
const RC_TAG: &[u8] = b"AGL-Monolith-RC-v1";

/// MDS matrix — identical to Goldilocks-side Monolith. Small-integer entries
/// `{6, 7, 8, 9, 10, 13, 21, 22, 23, 26}` arranged circulantly. MDS over any
/// prime field with `q` greater than the largest 11×11 minor determinant —
/// trivially true for both Goldilocks and AlmostGoldilocks.
pub const MDS: [[u64; WIDTH]; WIDTH] = [
    [ 7, 23,  8, 26, 13, 10,  9,  7,  6, 22, 21,  8],
    [ 8,  7, 23,  8, 26, 13, 10,  9,  7,  6, 22, 21],
    [21,  8,  7, 23,  8, 26, 13, 10,  9,  7,  6, 22],
    [22, 21,  8,  7, 23,  8, 26, 13, 10,  9,  7,  6],
    [ 6, 22, 21,  8,  7, 23,  8, 26, 13, 10,  9,  7],
    [ 7,  6, 22, 21,  8,  7, 23,  8, 26, 13, 10,  9],
    [ 9,  7,  6, 22, 21,  8,  7, 23,  8, 26, 13, 10],
    [10,  9,  7,  6, 22, 21,  8,  7, 23,  8, 26, 13],
    [13, 10,  9,  7,  6, 22, 21,  8,  7, 23,  8, 26],
    [26, 13, 10,  9,  7,  6, 22, 21,  8,  7, 23,  8],
    [ 8, 26, 13, 10,  9,  7,  6, 22, 21,  8,  7, 23],
    [23,  8, 26, 13, 10,  9,  7,  6, 22, 21,  8,  7],
];

const Q: u64 = ALMOST_GOLDILOCKS_PRIME;

// ============================================================================
// Round-constant derivation
// ============================================================================

fn derive_one(round: u32, pos: u32) -> u64 {
    let mut salt: u32 = 0;
    loop {
        let mut h = Sha256::new();
        h.update(RC_TAG);
        h.update(round.to_le_bytes());
        h.update(pos.to_le_bytes());
        h.update(salt.to_le_bytes());
        let digest = h.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[0..8]);
        let v = u64::from_le_bytes(bytes);
        if v < Q {
            return v;
        }
        // 2^{-58} per draw; reaching salt = 1 is astronomically unlikely. The
        // explicit panic on overflow is just defense-in-depth.
        salt = salt
            .checked_add(1)
            .expect("monolith RC rejection-sampling salt overflowed u32");
    }
}

fn derive_round_constants() -> [[u64; WIDTH]; NUM_CONCRETE] {
    let mut out = [[0u64; WIDTH]; NUM_CONCRETE];
    // Monolith convention: the initial (round 0) and final (round N_ROUNDS)
    // concrete layers use zero constants. Interior rounds 1..N_ROUNDS-1 do
    // not receive derived constants either in the Goldilocks reference (see
    // `goldilocks-cuda-rs/src/cpu_monolith.rs`: ROUND_CONSTANTS[0] and [6]
    // are zero, [1..5] are nontrivial, [6] is zero — note that the
    // reference treats the *last applied* layer as the zero one).
    //
    // We match this: rounds 1..N_ROUNDS-1 receive derived constants
    // (positions 1, 2, …, N_ROUNDS-1 in the NUM_CONCRETE-length table);
    // rounds 0 and N_ROUNDS stay zero.
    for r in 1..N_ROUNDS {
        for i in 0..WIDTH {
            out[r][i] = derive_one(r as u32, i as u32);
        }
    }
    out
}

/// Returns the `[NUM_CONCRETE][WIDTH]` round-constant table for AlmostGoldilocks
/// Monolith. Lazily computed exactly once per process — subsequent calls
/// return the same `&'static` reference at no cost.
pub fn round_constants() -> &'static [[u64; WIDTH]; NUM_CONCRETE] {
    static CONSTANTS: OnceLock<[[u64; WIDTH]; NUM_CONCRETE]> = OnceLock::new();
    CONSTANTS.get_or_init(derive_round_constants)
}

// ============================================================================
// Field arithmetic helpers (host-side AGL)
// ============================================================================

#[inline]
fn agl_add(a: u64, b: u64) -> u64 {
    (AlmostGoldilocksField(a) + AlmostGoldilocksField(b)).reduce().0
}

#[inline]
fn agl_mul(a: u64, b: u64) -> u64 {
    (AlmostGoldilocksField(a) * AlmostGoldilocksField(b)).reduce().0
}

/// Reduce a 128-bit accumulator `a` to canonical `[0, q)`.
#[inline]
fn agl_reduce_u128(a: u128) -> u64 {
    let m = a % (Q as u128);
    m as u64
}

// ============================================================================
// Monolith layers
// ============================================================================

/// Byte-wise S-box (the "Bars" layer). Field-agnostic: it operates on the raw
/// `u64` representation of one state element. The output may exceed `q` but
/// the next field operation (`bricks` → `gl_mul` → reduction) canonicalizes.
#[inline]
fn bar_64(limb: u64) -> u64 {
    let limbl1 = ((!limb & 0x8080808080808080) >> 7) | ((!limb & 0x7F7F7F7F7F7F7F7F) << 1);
    let limbl2 = ((limb & 0xC0C0C0C0C0C0C0C0) >> 6) | ((limb & 0x3F3F3F3F3F3F3F3F) << 2);
    let limbl3 = ((limb & 0xE0E0E0E0E0E0E0E0) >> 5) | ((limb & 0x1F1F1F1F1F1F1F1F) << 3);
    let tmp = limb ^ (limbl1 & limbl2 & limbl3);
    ((tmp & 0x8080808080808080) >> 7) | ((tmp & 0x7F7F7F7F7F7F7F7F) << 1)
}

fn bars(state: &mut [u64; WIDTH]) {
    for i in 0..NUM_BARS {
        state[i] = bar_64(state[i]);
    }
}

/// Feistel Type-3, reverse iteration: `state[i] += state[i-1]^2` for
/// `i = WIDTH-1` down to `1`.
fn bricks(state: &mut [u64; WIDTH]) {
    for i in (1..WIDTH).rev() {
        let sq = agl_mul(state[i - 1], state[i - 1]);
        state[i] = agl_add(state[i], sq);
    }
}

/// MDS·state + round_constants[round]. All arithmetic in `F_q`.
fn concrete(state: &mut [u64; WIDTH], round: usize) {
    let rc = round_constants();
    let mut result = [0u64; WIDTH];
    for row in 0..WIDTH {
        // Each row's dot product fits in u128 with room to spare:
        // 12 × (q − 1) × max(MDS) ≤ 12 × 2^64 × 32 < 2^73 < 2^128. ✓
        let mut acc: u128 = 0;
        for col in 0..WIDTH {
            // Canonicalize state[col] first (bars may leave non-canonical reps).
            let s = AlmostGoldilocksField(state[col]).reduce().0 as u128;
            acc += s * (MDS[row][col] as u128);
        }
        acc += rc[round][row] as u128;
        result[row] = agl_reduce_u128(acc);
    }
    *state = result;
}

// ============================================================================
// Full permutation
// ============================================================================

/// Apply the Monolith permutation in-place. After return, every `state[i]` is
/// in canonical form `[0, q)`.
pub fn monolith_permute(state: &mut [u64; WIDTH]) {
    concrete(state, 0);
    for r in 1..=N_ROUNDS {
        bars(state);
        bricks(state);
        concrete(state, r);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_constants_are_deterministic_and_in_range() {
        let rc = round_constants();
        // Round 0 and N_ROUNDS are all-zero by Monolith convention.
        for i in 0..WIDTH {
            assert_eq!(rc[0][i], 0, "RC[0][{}] expected 0, got {:#x}", i, rc[0][i]);
            assert_eq!(
                rc[N_ROUNDS][i], 0,
                "RC[{}][{}] expected 0, got {:#x}",
                N_ROUNDS, i, rc[N_ROUNDS][i]
            );
        }
        // Interior rounds: every constant is in [0, q) and they're not all
        // identical (catches obvious derivation bugs).
        let mut seen = std::collections::HashSet::new();
        for r in 1..N_ROUNDS {
            for i in 0..WIDTH {
                let v = rc[r][i];
                assert!(v < Q, "RC[{}][{}] = {:#x} not in [0, q)", r, i, v);
                seen.insert(v);
            }
        }
        let interior_count = (N_ROUNDS - 1) * WIDTH;
        assert_eq!(
            seen.len(),
            interior_count,
            "expected {} distinct interior constants, found {} (collisions in SHA-256 derivation?)",
            interior_count,
            seen.len()
        );
    }

    /// Calling `round_constants()` twice returns the same `&'static` reference
    /// and the same content — verifies the `OnceLock` semantics.
    #[test]
    fn round_constants_are_lazily_cached() {
        let r1 = round_constants();
        let r2 = round_constants();
        assert!(std::ptr::eq(r1, r2));
        for r in 0..NUM_CONCRETE {
            for i in 0..WIDTH {
                assert_eq!(r1[r][i], r2[r][i]);
            }
        }
    }

    /// Independently reproduce one SHA-256-derived constant by hand to nail
    /// down the derivation format (so a future reader can re-verify any
    /// constant they care about).
    #[test]
    fn round_constant_derivation_format_is_documented() {
        // RC[1][0]: round=1, pos=0, salt=0.
        let mut h = Sha256::new();
        h.update(b"AGL-Monolith-RC-v1");
        h.update(1u32.to_le_bytes());
        h.update(0u32.to_le_bytes());
        h.update(0u32.to_le_bytes());
        let digest = h.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&digest[0..8]);
        let expected = u64::from_le_bytes(bytes);
        // We know `expected < q` (`2^{-58}` rejection probability — this value
        // is just one fixed SHA-256 output; if it ever turns out to be ≥ q the
        // test will fail loudly and we'll know the derivation needs to salt).
        assert!(
            expected < Q,
            "Demo derivation produced {:#x} ≥ q; test stub assumes salt=0 lands. \
             If you see this, the salt loop in derive_one is correct — just \
             update this test to walk through the salts.",
            expected
        );
        assert_eq!(round_constants()[1][0], expected);
    }

    #[test]
    fn bar_64_is_a_byte_permutation() {
        // `bar_64` is the Monolith "Bars" S-box: it permutes the 256 possible
        // byte values within each of the 8 bytes of the u64. Apply to all
        // single-byte inputs and confirm distinct outputs.
        let mut seen = std::collections::HashSet::new();
        for b in 0u64..=255 {
            // Place the test byte into the low byte; zero elsewhere.
            let out = bar_64(b);
            // The byte permutation operates per-byte and is independent; only
            // the low output byte is influenced by the low input byte.
            seen.insert(out & 0xFF);
        }
        assert_eq!(seen.len(), 256, "bar_64 should permute the 256 byte values");
    }

    #[test]
    fn permutation_is_deterministic() {
        let mut a = [0u64; WIDTH];
        let mut b = [0u64; WIDTH];
        for i in 0..WIDTH {
            a[i] = i as u64;
            b[i] = i as u64;
        }
        monolith_permute(&mut a);
        monolith_permute(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn permutation_outputs_are_canonical() {
        let mut state = [0u64; WIDTH];
        for i in 0..WIDTH {
            state[i] = i as u64;
        }
        monolith_permute(&mut state);
        for i in 0..WIDTH {
            assert!(
                state[i] < Q,
                "state[{}] = {:#x} not in [0, q) after permutation",
                i,
                state[i]
            );
        }
    }

    #[test]
    fn distinct_inputs_yield_distinct_outputs() {
        let mut a = [0u64; WIDTH];
        let mut b = [0u64; WIDTH];
        a[0] = 1;
        b[0] = 2;
        monolith_permute(&mut a);
        monolith_permute(&mut b);
        assert_ne!(a, b);
    }

    /// Single-element diffusion: changing one input element should affect
    /// every output element. This catches MDS / Bricks bugs that would
    /// otherwise pass the previous test.
    #[test]
    fn permutation_diffuses_single_input_change() {
        let mut a = [0u64; WIDTH];
        for i in 0..WIDTH {
            a[i] = (i as u64).wrapping_mul(0x123456789ABCDEF);
        }
        let mut b = a;
        b[0] = b[0].wrapping_add(1);
        monolith_permute(&mut a);
        monolith_permute(&mut b);
        for i in 0..WIDTH {
            assert_ne!(
                a[i], b[i],
                "output element {} did not change after perturbing input[0]",
                i
            );
        }
    }

    /// Frozen golden vector for the AGL Monolith permutation. Captured at
    /// port time by running the same code against input `[0..11]`. If this
    /// fires, **something protocol-relevant changed**: either the round
    /// constants (RC derivation logic / SHA-256 tag / rejection sampler),
    /// the MDS matrix, the permutation schedule, or the field arithmetic.
    /// All four are part of the protocol's identity; a deliberate bump
    /// requires regenerating dependent code (transcript test vectors,
    /// prover/verifier consistency, persisted offline commits).
    #[test]
    fn agl_monolith_golden_vector_input_0_to_11() {
        let mut state = [0u64; WIDTH];
        for i in 0..WIDTH {
            state[i] = i as u64;
        }
        monolith_permute(&mut state);
        const EXPECTED: [u64; WIDTH] = [
            0xed82a66d4453e6b3,
            0x752b191c23268621,
            0x590fab26245cc4e9,
            0xa63b2125e2a94a88,
            0x6ab5bea637404df2,
            0x6dbeafbbb35f0b77,
            0xfc9cdfae47529f13,
            0xe57947afed67d70f,
            0x59da37a113e7987e,
            0x95662ed82cec7bb2,
            0x7429836455d95e39,
            0x91861bd52d8c67f2,
        ];
        assert_eq!(state, EXPECTED);
    }

    /// Frozen golden values for `RC[1][0..3]`. The full table has 84 entries
    /// (7 × 12); regression coverage of the first few values is enough to
    /// detect any drift in the SHA-256 derivation since each constant is
    /// produced by an independent SHA-256 call.
    #[test]
    fn agl_monolith_round_constants_frozen_sample() {
        let rc = round_constants();
        const EXPECTED_R1: [u64; 4] = [
            0xdc5869754490c2c9,
            0x8df3ae2a26b03334,
            0x6531a374c7fc9b9b,
            0x0758fb4f5d864daf,
        ];
        for (i, &v) in EXPECTED_R1.iter().enumerate() {
            assert_eq!(rc[1][i], v, "RC[1][{}] drifted: expected {:#018x}, got {:#018x}", i, v, rc[1][i]);
        }
    }

    /// Diagnostic helper: dump the full RC table + golden output for the
    /// permutation. Useful when intentionally bumping any constant; not part
    /// of the protocol-correctness gate.
    #[test]
    #[ignore]
    fn dump_constants_and_golden() {
        let rc = round_constants();
        println!("\nROUND_CONSTANTS:");
        for r in 0..NUM_CONCRETE {
            println!("  // round {}", r);
            for v in rc[r] {
                println!("  0x{:016x},", v);
            }
        }
        let mut state = [0u64; WIDTH];
        for i in 0..WIDTH {
            state[i] = i as u64;
        }
        monolith_permute(&mut state);
        println!("\nGolden(input=[0..11]):");
        for v in &state {
            println!("  0x{:016x},", v);
        }
    }
}
