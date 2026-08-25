//! Interleaved uniform-arity packing.
//!
//! [`super::layout::PackLayout`] places each leaf as a contiguous block:
//! `index = (block << leaf_arity) | leaf_index`. That is the natural layout, and
//! it is why the link has to materialize its query-weight table.
//!
//! The sumcheck binds the high index variable first, so with the block in the
//! high bits it merges blocks on round 0. After `r` rounds a live position is a
//! superposition of `2^r` blocks, so the weight there is a sum over `2^r`
//! queries — by round 16 that is 65536 queries per position. Evaluating it on
//! demand is hopeless, which forces a dense folded table at 16 bytes per witness
//! bit, and that is what caps how many commitments fit in a batch.
//!
//! Interleaving swaps the roles:
//!
//! ```text
//!     index = (leaf_index << log2(G)) | block
//! ```
//!
//! Now the leaf variables bind first and blocks stay disjoint for the whole leaf
//! phase, so every live position belongs to exactly one query and its weight is
//! `gamma_j * c_j * eq(remaining leaf point)` — computable pointwise in
//! `O(remaining vars)` with no table at all. Once the leaf variables are
//! exhausted each block has collapsed to a single scalar, and the surviving
//! domain is just the block count, which is small enough to materialize.
//!
//! The cost is that a leaf is no longer contiguous: its coefficients sit at
//! stride `G`. Commitment is unaffected because the Ajtai map is a sum over
//! positions — sparse leaves map their position list through the same index
//! formula, which is what keeps commit support-sensitive.

use crate::commit::layout::hiding_block_coeffs;

/// One leaf's placement in an interleaved commitment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// Which packed commitment.
    pub commitment: usize,
    /// Which block within it (the low `log2(G)` bits of every position).
    pub block: usize,
}

/// A uniform-arity interleaved group: every leaf in it has the same arity.
#[derive(Clone, Debug)]
pub struct InterleavedGroup {
    /// Arity shared by every leaf in this group.
    pub leaf_arity: usize,
    /// Blocks per commitment, `G`. Always a power of two.
    pub blocks: usize,
    /// `log2(G)`.
    pub block_bits: usize,
    /// `leaf_arity + block_bits`.
    pub ambient_arity: usize,
    pub num_commitments: usize,
    /// Blocks at or above this index carry hiding randomness, not leaves.
    pub hiding_block_start: usize,
    /// Placement per leaf, in the order given to [`Self::build`].
    pub slots: Vec<Slot>,
}

impl InterleavedGroup {
    /// Packed coefficient index of leaf-local coefficient `l` in `block`.
    #[inline]
    pub fn position(&self, block: usize, l: usize) -> usize {
        (l << self.block_bits) | block
    }

    /// Build a group from `n_leaves` leaves of one arity.
    ///
    /// `target_blocks` is the desired leaf blocks per commitment; it is rounded
    /// up to a power of two and grown until the hiding block fits alongside.
    /// Hiding occupies whole blocks at the top of the block index, so it never
    /// interleaves with a leaf and its weight is simply absent.
    pub fn build(leaf_arity: usize, n_leaves: usize, target_blocks: usize) -> Option<Self> {
        if n_leaves == 0 {
            return None;
        }
        let hiding = hiding_block_coeffs();
        // Hiding needs ceil(hiding / 2^leaf_arity) blocks, at least one.
        let hiding_blocks = ((hiding + (1usize << leaf_arity) - 1) >> leaf_arity).max(1);

        let mut blocks = target_blocks.max(1).next_power_of_two();
        // Leave room for the hiding blocks.
        while blocks <= hiding_blocks {
            blocks <<= 1;
        }
        let leaf_capacity = blocks - hiding_blocks;

        let block_bits = blocks.trailing_zeros() as usize;
        let num_commitments = n_leaves.div_ceil(leaf_capacity);

        let slots = (0..n_leaves)
            .map(|i| Slot {
                commitment: i / leaf_capacity,
                block: i % leaf_capacity,
            })
            .collect();

        Some(Self {
            leaf_arity,
            blocks,
            block_bits,
            ambient_arity: leaf_arity + block_bits,
            num_commitments,
            hiding_block_start: leaf_capacity,
            slots,
        })
    }

    /// Live link state per commitment, in bytes.
    ///
    /// Only the folded witness is stored: the query weights are evaluated on
    /// demand, which is the entire point of this layout. Round 0 reads the
    /// bit-packed witness and writes the half-size folded table, so the peak is
    /// `2^(A-1)` Ext2 rather than `2^A` twice over.
    pub fn state_bytes(&self) -> usize {
        (1usize << (self.ambient_arity - 1)) * 16
    }

    /// Blocks per commitment to hit a per-commitment state budget.
    pub fn blocks_for_budget(leaf_arity: usize, budget_bytes: usize) -> usize {
        // state = 2^(leaf_arity + block_bits - 1) * 16
        let mut bb = 0usize;
        while (1usize << (leaf_arity + bb)) * 8 <= budget_bytes && bb < 20 {
            bb += 1;
        }
        1usize << bb.saturating_sub(1).max(0)
    }
}

/// Split a mixed-arity leaf set into uniform-arity groups.
///
/// Returns `(arity, indices)` per group, in ascending arity so the caller's
/// group order is deterministic and the verifier reproduces it.
pub fn group_by_arity(arities: &[usize]) -> Vec<(usize, Vec<usize>)> {
    let mut by: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (i, a) in arities.iter().enumerate() {
        by.entry(*a).or_default().push(i);
    }
    by.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaving_keeps_blocks_disjoint_under_folding() {
        // The property the whole design rests on: with the block in the LOW
        // bits, binding high variables never mixes two blocks, so after any
        // number of leaf rounds a live position still belongs to one block.
        let g = InterleavedGroup::build(10, 8, 8).expect("group");
        let mut seen = std::collections::HashMap::new();
        for b in 0..g.hiding_block_start.min(8) {
            for l in 0..(1usize << g.leaf_arity) {
                let p = g.position(b, l);
                assert!(seen.insert(p, b).is_none(), "positions must be distinct");
                assert_eq!(p & (g.blocks - 1), b, "block is recoverable from the low bits");
            }
        }
        // Folding r high variables maps p -> p mod 2^(A-r); the block bits are
        // the lowest ones, so they survive every leaf round untouched.
        for r in 1..=g.leaf_arity {
            let live = 1usize << (g.ambient_arity - r);
            for (&p, &b) in seen.iter().take(64) {
                assert_eq!((p % live) & (g.blocks - 1), b);
            }
        }
    }

    #[test]
    fn hiding_occupies_whole_blocks_at_the_top() {
        for arity in [8usize, 12, 17, 20, 24] {
            let g = InterleavedGroup::build(arity, 40, 64).expect("group");
            let hiding_blocks = g.blocks - g.hiding_block_start;
            assert!(hiding_blocks >= 1);
            assert!(
                hiding_blocks * (1usize << arity) >= hiding_block_coeffs(),
                "arity {}: hiding blocks must cover the required randomness", arity
            );
            // and no leaf is placed in them
            for s in &g.slots {
                assert!(s.block < g.hiding_block_start);
            }
        }
    }

    #[test]
    fn every_leaf_gets_a_distinct_slot() {
        let g = InterleavedGroup::build(14, 300, 32).expect("group");
        let mut seen = std::collections::HashSet::new();
        for s in &g.slots {
            assert!(seen.insert((s.commitment, s.block)), "slots must be unique");
        }
        assert_eq!(seen.len(), 300);
        assert_eq!(g.ambient_arity, g.leaf_arity + g.block_bits);
    }

    #[test]
    fn state_is_half_of_the_contiguous_layout() {
        // Contiguous keeps witness AND weights as Ext2 over the full ambient;
        // interleaved keeps only the folded half-size witness.
        let g = InterleavedGroup::build(20, 64, 64).expect("group");
        let contiguous = 2 * (1usize << g.ambient_arity) * 16;
        assert_eq!(g.state_bytes() * 4, contiguous);
    }

    #[test]
    fn grouping_is_deterministic_and_total() {
        let arities = vec![10, 22, 10, 14, 22, 10];
        let groups = group_by_arity(&arities);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].0, 10);
        assert_eq!(groups[0].1, vec![0, 2, 5]);
        let total: usize = groups.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(total, arities.len());
    }
}
