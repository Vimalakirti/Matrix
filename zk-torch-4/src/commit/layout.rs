//! Packing layout for the packed PCS (`ZK4_PCS=packed`).
//!
//! Every leaf witness — one `(edge, digit plane)` pair — becomes an aligned
//! power-of-two *block* inside a larger committed polynomial. A commitment is
//! therefore a concatenation of many leaves plus one hiding block, and the
//! Ajtai map being linear means it can be computed as the ring-sum of each
//! block committed against its own column window of `M_max`
//! (`ajtai::commit_wide(.., col_offset, ..)` and friends).
//!
//! ## Why pack at all
//!
//! The masked-RLC opening merges `|I|` commitments into `tau` dense responses.
//! Its parameters are driven by `B* = |I|·T_exp·(B_x−1)+1`, so the shift norm
//! grows with the *number of commitments*, while the response length grows with
//! the *ambient dimension* `D`. Concatenation trades one against the other
//! without touching witness values: coefficients stay in `{0,1}` so `B_x = 2`
//! holds, which is what folding cannot do (a fold multiplies the norm by
//! `T_exp = 128`).
//!
//! ## What the verifier recomputes
//!
//! The layout is public: it is a deterministic function of the DAG, which the
//! verifier already has. A claim on leaf `i` at point `r` becomes a claim on
//! the packed polynomial at `(block_prefix(i), r)`, where the prefix is a
//! *Boolean* point selecting the leaf's sub-cube. `eq` factorizes across that
//! prefix, so the eq-term stays supported on the leaf's own block and the link
//! sumcheck costs one pass over the packed domain rather than `t` passes.
//!
//! ## Sizing
//!
//! `D = 2^A` with the top `D_HID` slots reserved for the hiding block, leaving
//! `2^A − D_HID` of message capacity. Since that capacity is a sum of powers of
//! two, aligned blocks fill it exactly — no padding waste. `A` is chosen to
//! minimize `|I|·D·link + tau·retries·D·wide`, whose closed form is
//! `D* ≈ sqrt(N·D_hid / (tau·w))`, then rounded to a power of two and checked
//! against the measured per-arity kernel costs.

use std::collections::BTreeMap;

use almost_goldilocks_cuda::ajtai::{KAPPA, RING_DIM};

/// Ternary hiding symbols per ring coordinate.
///
/// The leftover-hash bound needs `H > kappa·d·log2(q) + 2·lambda` bits of
/// entropy. Spending one symbol per ring coordinate (as the write-up's
/// parameter table does) costs 64 coefficient *slots* per symbol; filling all
/// `RING_DIM` slots gives the same entropy in 64x fewer slots, at identical
/// `||s||_inf = 1` and identical Module-SIS accounting. The security argument
/// differs — dense coordinates need a ring regularity lemma rather than the
/// plain leftover hash lemma over `Z_q` — so this is a parameter, not a
/// constant, and `1` recovers the conservative layout.
pub const HIDING_SYMBOLS_PER_COORD: usize = 64;

/// Statistical security margin for commitment hiding, in bits.
pub const HIDING_LAMBDA: usize = 128;

/// `log2(q)` for Almost-Goldilocks, rounded up for the entropy budget.
const LOG2_Q: f64 = 64.0;

/// Hiding block size in coefficients, as a power of two.
///
/// Depends only on `(kappa, d, q, lambda)` — no model quantity appears — so it
/// does **not** shrink with the model. Shrinking `s` alongside the message
/// would silently break commitment hiding, which is why this is derived here
/// and asserted rather than passed in.
pub fn hiding_block_coeffs() -> usize {
    let need_bits = (KAPPA * RING_DIM) as f64 * LOG2_Q + 2.0 * HIDING_LAMBDA as f64;
    let bits_per_coord = HIDING_SYMBOLS_PER_COORD as f64 * 3f64.log2();
    let coords = (need_bits / bits_per_coord).ceil() as usize;
    // Round the coordinate count up to a power of two, then convert to slots.
    let coords_pow2 = coords.next_power_of_two();
    (coords_pow2 * RING_DIM).next_power_of_two()
}

/// Realized entropy of the hiding block, in bits, for the chosen size.
pub fn hiding_entropy_bits() -> f64 {
    let coords = hiding_block_coeffs() / RING_DIM;
    (coords * HIDING_SYMBOLS_PER_COORD) as f64 * 3f64.log2()
}

/// One leaf's placement inside a packed commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlacement {
    /// Index of the packed commitment this leaf lives in.
    pub commitment: usize,
    /// Offset of the block, in coefficients, from the start of the commitment.
    /// Always a multiple of the block length (aligned sub-cube).
    pub offset: usize,
    /// `log2` of the block length, i.e. the leaf's own arity.
    pub arity: usize,
}

impl BlockPlacement {
    /// Column window this block commits against, in ring elements.
    pub fn col_offset(&self) -> u64 {
        (self.offset / RING_DIM) as u64
    }

    /// Boolean prefix that selects this block inside the packed domain, most
    /// significant variable first. A claim on the leaf at point `r` becomes a
    /// claim on the packed polynomial at `(prefix, r)`.
    pub fn block_prefix(&self, ambient_arity: usize) -> Vec<bool> {
        let prefix_len = ambient_arity - self.arity;
        let index = self.offset >> self.arity;
        (0..prefix_len)
            .map(|k| (index >> (prefix_len - 1 - k)) & 1 == 1)
            .collect()
    }
}

/// A deterministic, public packing of leaves into commitments.
#[derive(Clone, Debug)]
pub struct PackLayout {
    /// `log2` of the ambient dimension of every commitment in this group.
    pub ambient_arity: usize,
    /// Hiding block size in coefficients (identical for every commitment).
    pub hiding_coeffs: usize,
    /// Message capacity per commitment, `2^ambient_arity − hiding_coeffs`.
    pub message_capacity: usize,
    /// Number of packed commitments.
    pub num_commitments: usize,
    /// Placement per leaf key, in the caller's key order.
    pub placements: BTreeMap<LeafKey, BlockPlacement>,
}

/// Identifies a leaf witness. Ordering is what makes the layout deterministic:
/// the verifier sorts the DAG's leaves by this key and replays the same packing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeafKey {
    /// Edge id in the DAG.
    pub edge: usize,
    /// Digit plane within that edge's decomposition.
    pub plane: usize,
}

/// A leaf as presented to the packer.
#[derive(Clone, Copy, Debug)]
pub struct LeafSpec {
    pub key: LeafKey,
    /// `log2` of the leaf's witness length in coefficients.
    pub arity: usize,
}

/// Error cases the packer refuses rather than silently working around.
#[derive(Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// A leaf is larger than the message capacity of one commitment.
    LeafTooLarge { arity: usize, capacity_arity: usize },
    /// The hiding block does not leave usable message capacity.
    AmbientTooSmall { ambient_arity: usize, hiding_coeffs: usize },
}

impl PackLayout {
    /// Build a layout at a fixed ambient arity.
    ///
    /// Leaves are sorted by `LeafKey` (deterministic and verifier-reproducible),
    /// then placed largest-first into aligned slots. Because every block is a
    /// power of two and the capacity `2^A − hiding` is a sum of powers of two,
    /// a descending-size first-fit fills each commitment exactly, with slack
    /// only in the final commitment.
    pub fn build(leaves: &[LeafSpec], ambient_arity: usize) -> Result<Self, LayoutError> {
        let hiding = hiding_block_coeffs();
        let ambient = 1usize << ambient_arity;
        if ambient <= hiding {
            return Err(LayoutError::AmbientTooSmall {
                ambient_arity,
                hiding_coeffs: hiding,
            });
        }
        let capacity = ambient - hiding;
        let capacity_arity = capacity.trailing_zeros() as usize;

        // Sort by (descending arity, ascending key). Descending size is what
        // makes first-fit exact for power-of-two blocks; the key tiebreak keeps
        // it reproducible.
        let mut order: Vec<LeafSpec> = leaves.to_vec();
        order.sort_by(|a, b| b.arity.cmp(&a.arity).then(a.key.cmp(&b.key)));

        for leaf in &order {
            if (1usize << leaf.arity) > capacity {
                return Err(LayoutError::LeafTooLarge {
                    arity: leaf.arity,
                    capacity_arity,
                });
            }
        }

        let mut placements = BTreeMap::new();
        // `filled[c]` is the next free coefficient offset in commitment c.
        // Blocks are placed in descending size, so `filled[c]` is always a
        // multiple of the next block's size and alignment is automatic.
        let mut filled: Vec<usize> = Vec::new();

        for leaf in &order {
            let len = 1usize << leaf.arity;
            let mut placed = false;
            for (c, used) in filled.iter_mut().enumerate() {
                if *used + len <= capacity {
                    debug_assert_eq!(*used % len, 0, "descending-size placement keeps alignment");
                    placements.insert(
                        leaf.key,
                        BlockPlacement { commitment: c, offset: *used, arity: leaf.arity },
                    );
                    *used += len;
                    placed = true;
                    break;
                }
            }
            if !placed {
                let c = filled.len();
                filled.push(len);
                placements.insert(
                    leaf.key,
                    BlockPlacement { commitment: c, offset: 0, arity: leaf.arity },
                );
            }
        }

        Ok(Self {
            ambient_arity,
            hiding_coeffs: hiding,
            message_capacity: capacity,
            num_commitments: filled.len().max(1),
            placements,
        })
    }

    /// Offset of the hiding block within a commitment: it occupies the top
    /// `hiding_coeffs` slots, above all message blocks.
    pub fn hiding_offset(&self) -> usize {
        (1usize << self.ambient_arity) - self.hiding_coeffs
    }

    /// Column window of the hiding block, in ring elements.
    pub fn hiding_col_offset(&self) -> u64 {
        (self.hiding_offset() / RING_DIM) as u64
    }

    /// `B* = |I|·T_exp·(B_x−1)+1`, the masked-RLC shift bound this layout implies.
    pub fn shift_bound(&self, t_exp: usize, b_x: usize) -> usize {
        self.num_commitments * t_exp * (b_x - 1) + 1
    }

    /// Smallest power of two strictly above `shift_bound` — the `B` that sizes
    /// the Gaussian.
    pub fn gaussian_b(&self, t_exp: usize, b_x: usize) -> u64 {
        1u64 << (usize::BITS - self.shift_bound(t_exp, b_x).leading_zeros()) as u64
    }
}

/// Choose the ambient arity that minimizes projected opening time.
///
/// `link_ns` and `wide_ns` are measured per-arity kernel costs (ns per
/// coefficient) for the link sumcheck and the wide mask commit; `retries` is
/// the expected rejection-sampling attempt count. Returns the best arity along
/// with its projected millisecond cost.
///
/// This is deliberately driven by measurement rather than the closed form,
/// because both kernels have strong per-arity efficiency effects below arity
/// ~22 that the asymptotic expression misses.
pub fn choose_ambient_arity(
    total_message_coeffs: usize,
    candidates: &[(usize, f64, f64)], // (arity, link_ns, wide_ns)
    tau: usize,
    retries: f64,
) -> Option<(usize, f64)> {
    let hiding = hiding_block_coeffs();
    let mut best: Option<(usize, f64)> = None;
    for &(arity, link_ns, wide_ns) in candidates {
        let ambient = 1usize << arity;
        if ambient <= hiding {
            continue;
        }
        let capacity = ambient - hiding;
        let num_commitments = total_message_coeffs.div_ceil(capacity);
        let link_ms = num_commitments as f64 * ambient as f64 * link_ns / 1e6;
        let mask_ms = tau as f64 * retries * ambient as f64 * wide_ns / 1e6;
        let total = link_ms + mask_ms;
        if best.map_or(true, |(_, b)| total < b) {
            best = Some((arity, total));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaves(spec: &[(usize, usize)]) -> Vec<LeafSpec> {
        let mut out = Vec::new();
        let mut edge = 0usize;
        for &(arity, count) in spec {
            for _ in 0..count {
                out.push(LeafSpec { key: LeafKey { edge, plane: 0 }, arity });
                edge += 1;
            }
        }
        out
    }

    #[test]
    fn hiding_block_is_model_independent_and_has_enough_entropy() {
        let need = (KAPPA * RING_DIM) as f64 * LOG2_Q + 2.0 * HIDING_LAMBDA as f64;
        assert!(
            hiding_entropy_bits() > need,
            "hiding entropy {} must exceed the leftover-hash requirement {}",
            hiding_entropy_bits(),
            need
        );
        assert!(hiding_block_coeffs().is_power_of_two());
    }

    #[test]
    fn placements_are_aligned_disjoint_and_within_capacity() {
        let ls = leaves(&[(20, 3), (16, 40), (12, 200), (6, 500)]);
        let layout = PackLayout::build(&ls, 22).expect("layout");

        // group placements by commitment and check disjointness + alignment
        let mut per_commit: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (_, p) in &layout.placements {
            assert_eq!(p.offset % (1usize << p.arity), 0, "block must be aligned");
            assert!(
                p.offset + (1usize << p.arity) <= layout.message_capacity,
                "block must stay inside the message capacity"
            );
            per_commit.entry(p.commitment).or_default().push((p.offset, 1usize << p.arity));
        }
        for (_, mut spans) in per_commit {
            spans.sort();
            for w in spans.windows(2) {
                assert!(w[0].0 + w[0].1 <= w[1].0, "blocks must not overlap");
            }
        }
        assert_eq!(layout.placements.len(), ls.len(), "every leaf placed exactly once");
    }

    #[test]
    fn layout_is_deterministic_under_input_permutation() {
        let ls = leaves(&[(18, 5), (14, 33), (10, 111)]);
        let mut shuffled = ls.clone();
        shuffled.reverse();
        let a = PackLayout::build(&ls, 21).expect("a");
        let b = PackLayout::build(&shuffled, 21).expect("b");
        assert_eq!(a.placements, b.placements, "verifier must reproduce the layout");
        assert_eq!(a.num_commitments, b.num_commitments);
    }

    #[test]
    fn block_prefix_selects_the_right_subcube() {
        // A block at offset 3·2^10 inside a 2^14 ambient has prefix 0011.
        let p = BlockPlacement { commitment: 0, offset: 3 << 10, arity: 10 };
        assert_eq!(p.block_prefix(14), vec![false, false, true, true]);
        // Offset 0 is the all-zero prefix.
        let p0 = BlockPlacement { commitment: 0, offset: 0, arity: 10 };
        assert_eq!(p0.block_prefix(14), vec![false, false, false, false]);
        // col_offset is the block offset in ring elements.
        assert_eq!(p.col_offset(), ((3usize << 10) / RING_DIM) as u64);
    }

    #[test]
    fn oversized_leaf_is_rejected_not_silently_split() {
        let ls = leaves(&[(24, 1)]);
        match PackLayout::build(&ls, 20) {
            Err(LayoutError::LeafTooLarge { arity, .. }) => assert_eq!(arity, 24),
            other => panic!("expected LeafTooLarge, got {:?}", other.map(|l| l.num_commitments)),
        }
    }

    #[test]
    fn ambient_below_the_hiding_floor_is_rejected() {
        let ls = leaves(&[(6, 1)]);
        let too_small = hiding_block_coeffs().trailing_zeros() as usize;
        assert!(matches!(
            PackLayout::build(&ls, too_small),
            Err(LayoutError::AmbientTooSmall { .. })
        ));
    }

    #[test]
    fn packing_is_dense_every_commitment_but_the_last() {
        // Uniform blocks tile the capacity exactly.
        let ls = leaves(&[(16, 300)]);
        let layout = PackLayout::build(&ls, 20).expect("layout");
        let per = layout.message_capacity / (1usize << 16);
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for (_, p) in &layout.placements {
            *counts.entry(p.commitment).or_default() += 1;
        }
        let full = counts.values().filter(|&&c| c == per).count();
        assert!(
            full >= layout.num_commitments - 1,
            "all but the last commitment should be full: {:?}",
            counts
        );
    }

    #[test]
    fn ambient_choice_prefers_the_measured_optimum() {
        // Measured A100 costs (kappa=42).
        let candidates = [
            (20usize, 0.342f64, 30.28f64),
            (21, 0.205, 24.99),
            (22, 0.197, 25.64),
            (23, 0.269, 25.00),
            (24, 0.241, 24.69),
        ];
        // GPT-2 12L/seq64 measured leaf population.
        let n = 23_365_000_000usize;
        let (arity, ms) = choose_ambient_arity(n, &candidates, 2, 1.0 / 0.65).expect("choice");
        assert!((20..=24).contains(&arity));
        assert!(ms > 0.0);
        // The link term dominates at this N, so the chosen arity must be one
        // whose link cost is near the minimum rather than the largest ambient.
        assert!(arity <= 23, "should not pick the largest ambient at this N");
    }
}
