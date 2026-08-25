//! Fold-tree opening (plan §6). Contracts the leaf set of per-(edge,
//! bit-plane) Ajtai commitments down to a single `FoldInstance` whose
//! witness is small enough to ship to the verifier verbatim.
//!
//! Module layout:
//! - [`same_point_sumcheck`] (§6.1): reduce M heterogeneous-`r_i`
//!   instances to a single shared challenge `R`.
//! - [`multifold`] (§6.2): `c' = c_0 + Σ γ_i c_i`, `f' = f_0 + Σ γ_i f_i`.
//! - [`split`] (§6.3): decompose the wide i16 folded witness into 13
//!   ternary chunks and commit them with the SuperNeo `b=2, k=13` chunk
//!   scheme.
//! - [`tree`] (§6.4): group of 63 → 13 ternary chunks per level,
//!   recurse.
//! - [`verifier`] (§7): replays the tree's transcript.
//!
//! All Ext2 challenge points share the [`crate::transcript::Transcript`]
//! Fiat-Shamir state with the rest of the protocol.

use almost_goldilocks_cuda::ajtai::{
    RingChallenge, RingCommitment, TernaryChunks,
};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use serde::{Deserialize, Serialize};

pub mod multifold;
pub mod same_point_sumcheck;
pub mod split;
pub mod tree;
pub mod verifier;

pub use multifold::{MultifoldProof, prove_multifold, verify_multifold};
pub use same_point_sumcheck::{SamePointProof, prove_same_point, verify_same_point};
pub use split::{SplitProof, prove_split, verify_split_chunks_match};
pub use tree::{FoldTreeProof, prove_fold_tree, verify_fold_tree};
pub use verifier::FoldTreeError;

// ============================================================================
// FoldInstance — leaf and intermediate state in the fold tree.
// ============================================================================

/// Backing storage for a `FoldInstance`'s witness. Leaves are
/// [`FoldData::Binary`] (packed `u64`s — bit-decomposed planes from the
/// Ajtai commit step); after the first `multifold + split` the witness
/// becomes [`FoldData::Ternary`] (13 ternary chunks). The verifier
/// derives chunks homomorphically and never sees `Ternary` directly.
#[derive(Clone, Debug)]
pub enum FoldData {
    /// Packed binary witness: `vec.len() = 2^{arity - 6}` u64 elements,
    /// each packing 64 ring coefficients. Each bit is `{0, 1}`.
    Binary(Vec<u64>),
    /// Ternary chunks after a `split_witness` call. Each chunk has
    /// coefficients `∈ {−1, 0, 1}`; reconstruction:
    /// `Σ_i 2^i · (pos_i − neg_i)` recovers the wide witness.
    Ternary(TernaryChunks),
    /// Higher-radix (base-β) digit-plane witness for a single fold-tree leaf
    /// (one digit-plane = one Ajtai commit). Each digit value is in
    /// `{0..base-1}` (norm β-1), stored as `log₂β` internal binary bit-planes
    /// so the existing binary fast paths can be reused by running them
    /// `log₂β` times weighted by `2^k` (see `radix_to_bit_planes`).
    ///
    /// `bit_planes[k]` is a packed `Vec<u64>` of bit `k` across the digit-plane.
    /// `bit_planes.len() == log₂(base)`. The MLE eval of this digit-plane is
    /// `Σ_k 2^k · bin_eval(bit_planes[k])`, with an optional `negate_top_bit`
    /// flag that flips the sign of the highest bit's contribution — used to
    /// carry the signed two's-complement sign for the top digit-plane of the
    /// decomposition, mirroring the binary scheme's `-2^{b-1}·b_{b-1}` term.
    Digit {
        base: usize,
        bit_planes: Vec<Vec<u64>>,
        /// `true` only for the TOP digit-plane of a signed two's-complement
        /// decomposition (carries the sign weight). Reconstruction is then
        /// `Σ_{k<K-1} 2^k · b_k − 2^{K-1} · b_{K-1}` instead of `Σ_k 2^k b_k`.
        negate_top_bit: bool,
    },
}

impl FoldData {
    /// Bit-pack arity. Coefficients live in `2^arity` slots.
    pub fn arity_from_binary(packed_len: usize) -> usize {
        assert!(packed_len > 0, "empty binary witness");
        assert!(packed_len.is_power_of_two(), "packed_len must be a power of 2");
        // Each u64 packs 64 = 2^6 ring coefficients.
        let n_ring = packed_len;
        (n_ring.trailing_zeros() as usize) + 6
    }

    pub fn is_binary(&self) -> bool {
        matches!(self, FoldData::Binary(_))
    }

    pub fn is_ternary(&self) -> bool {
        matches!(self, FoldData::Ternary(_))
    }

    pub fn is_digit(&self) -> bool {
        matches!(self, FoldData::Digit { .. })
    }

    /// Evaluate this multilinear polynomial at the given Ext2 point.
    /// Treats the witness as 2^arity field-elements (`F_q`, then lifted to
    /// `F_q^2` for the inner product). For binary, the polynomial values
    /// at the boolean cube are `{0, 1}`; for ternary, they decompose into
    /// `Σ_i 2^i · (pos_bit − neg_bit)`.
    ///
    /// O(2^arity) time. Only used for tests and the verifier's final
    /// `f*(R*)` check.
    pub fn evaluate_at_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        let eq = crate::poly::evaluate_lagrange_basis_ext2(point);
        self.evaluate_with_eq(&eq)
    }

    /// Like [`Self::evaluate_at_ext2`] but takes a pre-built Lagrange
    /// basis table `eq = eq(point, ·)`. When many witnesses are
    /// evaluated at the SAME point (e.g. the 13 ternary split chunks
    /// at a fold-tree node, all at `shared_r`), building `eq` once and
    /// passing it here avoids the dominant cost — the `2^n`-sized eq
    /// table reconstruction — being paid per witness.
    pub fn evaluate_with_eq(&self, eq: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        let total = eq.len();
        let n = total.trailing_zeros() as usize;
        let mut acc = AlmostGoldilocksExt2::zero();
        match self {
            FoldData::Binary(packed) => {
                if n >= 6 {
                    assert_eq!(packed.len(), 1usize << (n - 6),
                        "Binary witness length {} does not match arity {}", packed.len(), n);
                } else {
                    // Sub-word arities are degenerate — only the bottom n bits
                    // of packed[0] are meaningful.
                    assert_eq!(packed.len(), 1, "sub-word binary witness must have length 1");
                }
                // Linearize as (j * 64 + k) → j-th u64, k-th bit.
                for j in 0..packed.len() {
                    let word = packed[j];
                    if word == 0 { continue; }
                    let base = j * 64;
                    for k in 0..64 {
                        if (word >> k) & 1 == 1 {
                            let idx = base + k;
                            if idx < total {
                                acc = crate::util::arith::ext2_add(acc, eq[idx]);
                            }
                        }
                    }
                }
            }
            FoldData::Ternary(chunks) => {
                let two = AlmostGoldilocksExt2::from_base(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(2),
                );
                let mut pow_two_i = AlmostGoldilocksExt2::one();
                for i in 0..chunks.k_chunks {
                    let (pos, neg) = chunks.chunk(i);
                    let mut layer = AlmostGoldilocksExt2::zero();
                    for j in 0..chunks.n_ring {
                        let pw = pos[j];
                        let nw = neg[j];
                        if pw == 0 && nw == 0 { continue; }
                        let base = j * 64;
                        for k in 0..64 {
                            let p = (pw >> k) & 1 == 1;
                            let m = (nw >> k) & 1 == 1;
                            if !p && !m { continue; }
                            let idx = base + k;
                            if idx >= total { continue; }
                            if p { layer = crate::util::arith::ext2_add(layer, eq[idx]); }
                            if m { layer = crate::util::arith::ext2_sub(layer, eq[idx]); }
                        }
                    }
                    acc = crate::util::arith::ext2_add(acc, crate::util::arith::ext2_mul(pow_two_i, layer));
                    pow_two_i = crate::util::arith::ext2_mul(pow_two_i, two);
                }
            }
            FoldData::Digit { base, bit_planes, negate_top_bit } => {
                // Digit-plane eval = Σ_k w_k · bin_eval(bit_planes[k]) where
                // w_k = 2^k normally and w_{m-1} = -2^{m-1} when this is the
                // sign-carrying top digit (mirrors binary two's-complement).
                // bit_planes.len() ≤ log₂base: equal for non-top digits, may
                // be smaller for the TOP digit when `b` isn't a multiple of K
                // (e.g., b=21 base=4 → top digit has only 1 effective bit).
                let k_bits = bit_planes.len();
                assert!(k_bits >= 1, "Digit needs ≥ 1 bit-plane");
                assert!(*base >= 2 && base.is_power_of_two(),
                    "Digit base must be power of 2 ≥ 2; got {}", base);
                let max_k = base.trailing_zeros() as usize;
                assert!(k_bits <= max_k,
                    "Digit.bit_planes.len()={} exceeds log₂(base={})={}", k_bits, base, max_k);
                let two = AlmostGoldilocksExt2::from_base(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(2),
                );
                let mut pow_two_k = AlmostGoldilocksExt2::one();
                for bk in 0..k_bits {
                    let packed = &bit_planes[bk];
                    // Inline the binary selective-add (same as FoldData::Binary)
                    // to avoid building a temporary FoldData::Binary.
                    let mut layer = AlmostGoldilocksExt2::zero();
                    if n >= 6 {
                        debug_assert_eq!(packed.len(), 1usize << (n - 6),
                            "Digit bit-plane {} length {} != arity {}", bk, packed.len(), n);
                    }
                    for j in 0..packed.len() {
                        let word = packed[j];
                        if word == 0 { continue; }
                        let base_idx = j * 64;
                        for kk in 0..64 {
                            if (word >> kk) & 1 == 1 {
                                let idx = base_idx + kk;
                                if idx < total {
                                    layer = crate::util::arith::ext2_add(layer, eq[idx]);
                                }
                            }
                        }
                    }
                    let signed_top = *negate_top_bit && bk == k_bits - 1;
                    if signed_top {
                        acc = crate::util::arith::ext2_sub(acc, crate::util::arith::ext2_mul(pow_two_k, layer));
                    } else {
                        acc = crate::util::arith::ext2_add(acc, crate::util::arith::ext2_mul(pow_two_k, layer));
                    }
                    pow_two_k = crate::util::arith::ext2_mul(pow_two_k, two);
                }
            }
        }
        acc
    }
}

/// One leaf or intermediate node of the fold tree. Carries the Ajtai
/// commitment, the witness data, and the current claim point + value.
#[derive(Clone, Debug)]
pub struct FoldInstance {
    /// The commitment `c_i ∈ R^15`. For leaves: per-bit-plane Ajtai
    /// commit. After multifold: `c' = c_0 + Σ γ_i c_i`.
    pub commitment: RingCommitment,
    /// Witness backing. See [`FoldData`].
    pub data: FoldData,
    /// Number of variables in the claim point (= `log2(witness size)`).
    /// All instances are eventually equalized to `max_num_vars` by the
    /// same-point sumcheck.
    pub arity: usize,
    /// `r_i ∈ Ext2^arity` — current evaluation point.
    pub claim_pt: Vec<AlmostGoldilocksExt2>,
    /// `y_i = f_i(r_i)` — current claimed value.
    pub claim_val: AlmostGoldilocksExt2,
}

/// Per-FoldInstance "weight" used by the same-point sumcheck. Equals
/// `2^{max_num_vars − arity}`. Materialized once when a leaf set is
/// assembled.
pub fn broadcast_weight(arity: usize, max_num_vars: usize) -> AlmostGoldilocksExt2 {
    assert!(arity <= max_num_vars, "arity {} > max_num_vars {}", arity, max_num_vars);
    let shift = max_num_vars - arity;
    AlmostGoldilocksExt2::from_base(
        almost_goldilocks_cuda::field::AlmostGoldilocksField(1u64 << shift),
    )
}

// ============================================================================
// Helper: serializable wire-form of RingCommitment / RingChallenge
// ============================================================================

/// Wire-form RingCommitment for the proof. `RingCommitment` itself isn't
/// `Serialize` in `almost-goldilocks-cuda`, so we round-trip through the
/// flat `[KAPPA][RING_DIM]` u64 array.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireCommitment {
    /// Row-major flat `[KAPPA · RING_DIM]` u64 entries.
    pub rows: Vec<u64>,
}

impl WireCommitment {
    pub fn from_ring(c: &RingCommitment) -> Self {
        use almost_goldilocks_cuda::ajtai::{KAPPA, RING_DIM};
        let mut rows = Vec::with_capacity(KAPPA * RING_DIM);
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                rows.push(c.rows[i][k]);
            }
        }
        Self { rows }
    }

    pub fn to_ring(&self) -> RingCommitment {
        use almost_goldilocks_cuda::ajtai::{KAPPA, RING_DIM};
        let mut c = RingCommitment::zero();
        assert_eq!(self.rows.len(), KAPPA * RING_DIM, "WireCommitment rows length");
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                c.rows[i][k] = self.rows[i * RING_DIM + k];
            }
        }
        c
    }
}

/// Wire-form RingChallenge for the proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireRingChallenge {
    pub coeffs: Vec<i8>,
}

impl WireRingChallenge {
    pub fn from_ring(r: &RingChallenge) -> Self {
        Self { coeffs: r.coeffs.to_vec() }
    }

    pub fn to_ring(&self) -> RingChallenge {
        assert_eq!(self.coeffs.len(), 64, "RingChallenge coeffs must be length 64");
        let mut arr = [0i8; 64];
        arr.copy_from_slice(&self.coeffs);
        RingChallenge::from_coeffs_unchecked(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    /// A Digit-plane's MLE eval must equal `Σ_k w_k · binary_eval(bit_planes[k])`
    /// where w_k = 2^k for normal digits and -2^{K-1} for the top bit when
    /// `negate_top_bit` is set. This is the math the rest of phase 2 depends on.
    #[test]
    fn digit_evaluate_matches_weighted_binary() {
        let n_ring = 2;
        let arity = 7; // 128 coefficients
        let point: Vec<_> = (0..arity)
            .map(|i| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(i as u64 * 7 + 3)))
            .collect();
        let eq = crate::poly::evaluate_lagrange_basis_ext2(&point);

        // base-4: two bit-planes per digit. Pick arbitrary bit patterns.
        let bit0: Vec<u64> = vec![0x0123456789ABCDEFu64, 0xFEDCBA9876543210u64];
        let bit1: Vec<u64> = vec![0xAAAAAAAAAAAAAAAAu64, 0x5555555555555555u64];

        // Reference: independently evaluate each as Binary, combine with 2^k.
        let v0 = FoldData::Binary(bit0.clone()).evaluate_with_eq(&eq);
        let v1 = FoldData::Binary(bit1.clone()).evaluate_with_eq(&eq);
        let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
        let unsigned_expected = crate::util::arith::ext2_add(
            v0, crate::util::arith::ext2_mul(two, v1));

        // Through Digit (unsigned, negate_top_bit=false): must match.
        let fd_unsigned = FoldData::Digit {
            base: 4,
            bit_planes: vec![bit0.clone(), bit1.clone()],
            negate_top_bit: false,
        };
        let v_unsigned = fd_unsigned.evaluate_with_eq(&eq);
        assert!(crate::util::arith::ext2_field_eq(v_unsigned, unsigned_expected),
            "Digit (unsigned) eval mismatched the weighted binary combination");

        // Through Digit (signed top bit): must equal v0 - 2·v1 (mirrors the
        // signed-two's-complement weight on the top bit of the top digit).
        let signed_expected = crate::util::arith::ext2_sub(
            v0, crate::util::arith::ext2_mul(two, v1));
        let fd_signed = FoldData::Digit {
            base: 4,
            bit_planes: vec![bit0, bit1],
            negate_top_bit: true,
        };
        let v_signed = fd_signed.evaluate_with_eq(&eq);
        assert!(crate::util::arith::ext2_field_eq(v_signed, signed_expected),
            "Digit (signed top) eval mismatched the signed weighted combination");

        let _ = n_ring;
    }

    #[test]
    fn binary_evaluate_matches_dense_mle() {
        // Build a tiny binary witness — 2 ring elements = 128 binary coefs
        // = arity 7. Set a sparse pattern and check evaluate_at_ext2
        // against a manual eq-table inner product.
        let n_ring = 2;
        let arity = (n_ring as usize).trailing_zeros() as usize + 6; // = 7
        let mut packed = vec![0u64; n_ring];
        // Set bits at flat indices 0, 65, 100.
        packed[0] |= 1u64 << 0;
        packed[1] |= 1u64 << 1; // flat index 64+1 = 65
        packed[1] |= 1u64 << 36; // flat index 64+36 = 100
        let fd = FoldData::Binary(packed);

        let point: Vec<_> = (0..arity)
            .map(|i| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(i as u64 * 13 + 5)))
            .collect();
        let v = fd.evaluate_at_ext2(&point);
        let eq = crate::poly::evaluate_lagrange_basis_ext2(&point);
        let expected = crate::util::arith::ext2_add(
            crate::util::arith::ext2_add(eq[0], eq[65]),
            eq[100],
        );
        assert!(
            crate::util::arith::ext2_field_eq(v, expected),
            "binary evaluation mismatch: got {:?} expected {:?}",
            v, expected,
        );
    }

    #[test]
    fn wire_commitment_roundtrip() {
        let mut c = RingCommitment::zero();
        c.rows[0][0] = 42;
        c.rows[14][63] = 7;
        let wire = WireCommitment::from_ring(&c);
        let back = wire.to_ring();
        assert_eq!(back.rows[0][0], 42);
        assert_eq!(back.rows[14][63], 7);
    }

    #[test]
    fn broadcast_weight_powers_of_two() {
        let w = broadcast_weight(7, 10);
        assert_eq!(w.c0.0, 8); // 2^(10-7) = 8
        assert_eq!(w.c1.0, 0);
        let w_eq = broadcast_weight(10, 10);
        assert_eq!(w_eq.c0.0, 1);
    }
}
