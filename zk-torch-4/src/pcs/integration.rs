//! Wiring the packed PCS into the prover, as a drop-in alternative to the
//! recursive fold tree (`ZK4_PCS=packed`).
//!
//! The fold tree consumes a leaf set of `FoldInstance`s and contracts it by
//! `same_point + multifold + split`, shipping the final witness in the clear.
//! The packed path consumes the same leaf set and instead packs the leaves into
//! block-structured commitments, runs one link sumcheck to bring every claim to
//! a shared point, and closes with a masked RLC.
//!
//! ## Batching, and why it is not optional
//!
//! The link keeps two Ext2 tables — the witness and the query weights — over the
//! packed domain, i.e. 32 bytes per witness *bit*. For GPT-2 12L/seq64 the leaf
//! set is 2.34e10 coefficients, so holding every commitment live at once would
//! need ~700 GiB. It does not fit, and no amount of sharding across four A100s
//! changes that.
//!
//! So commitments are processed in batches sized to a memory budget, and each
//! batch is its own opening with its own `ξ` and its own masked RLC. That is a
//! real cost — `τ` dense responses per batch rather than per model — and it is
//! the packed design's analogue of the fold tree's arity buckets, which exist
//! for the same reason.

use almost_goldilocks_cuda::ajtai::{self, RingCommitment, Seed, RING_DIM};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2 as Ext2;

use crate::commit::layout::{hiding_block_coeffs, LeafKey, LeafSpec, PackLayout};
use crate::fold::{FoldData, FoldInstance};
use crate::pcs::link::{prove_link_gpu_with, verify_link, LinkProof, LinkQuery, LinkScratch, LinkWitness};
use crate::transcript::Transcript;

/// Live link state per packed commitment, in bytes: witness + query weights, both Ext2.
fn state_bytes(ambient_arity: usize) -> usize {
    2 * (1usize << ambient_arity) * 16
}

/// Memory budget for one link batch. `ZK4_PCS_BUDGET_GB` overrides.
fn budget_bytes() -> usize {
    let gb: usize = std::env::var("ZK4_PCS_BUDGET_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    gb << 30
}

/// One batch's opening.
#[derive(Clone, Debug)]
pub struct PackedOpening {
    pub ambient_arity: usize,
    /// Packed commitments this batch merges.
    pub commitments: Vec<RingCommitment>,
    pub link: LinkProof,
    /// Terminal same-point claims the masked RLC consumes.
    pub xi: Vec<Ext2>,
}

#[derive(Clone, Debug)]
pub struct PackedPcsProof {
    pub openings: Vec<PackedOpening>,
    pub ambient_arity: usize,
    pub num_commitments: usize,
    /// Layout used per arity group, in group order.
    ///
    /// Recorded rather than re-derived because the choice depends on witness
    /// density, which the verifier does not have. It is a structural hint, not
    /// a soundness-relevant value: both layouts are independently sound, and the
    /// verifier checks whichever structure is claimed.
    pub layouts: Vec<Layout>,
}

/// Expand a leaf's witness into canonical field coefficients.
fn leaf_coeffs(data: &FoldData, arity: usize) -> Option<Vec<u64>> {
    let n = 1usize << arity;
    match data {
        FoldData::Binary(packed) => {
            let mut out = vec![0u64; n];
            for (i, word) in packed.iter().enumerate() {
                for b in 0..64 {
                    let idx = i * 64 + b;
                    if idx < n {
                        out[idx] = (word >> b) & 1;
                    }
                }
            }
            Some(out)
        }
        FoldData::Digit { base, bit_planes, negate_top_bit } => {
            // Digit values must stay inside {-1,0,1} for the B_x = 2 range
            // polynomial. Higher radices need R_{B_x} at that radix, which is a
            // parameter change rather than a code change — refuse rather than
            // emit a witness the link would reject for the wrong reason.
            if *base != 2 || *negate_top_bit {
                return None;
            }
            let mut out = vec![0u64; n];
            for (k, plane) in bit_planes.iter().enumerate() {
                let w = 1u64 << k;
                for (i, word) in plane.iter().enumerate() {
                    for b in 0..64 {
                        let idx = i * 64 + b;
                        if idx < n {
                            out[idx] += w * ((word >> b) & 1);
                        }
                    }
                }
            }
            Some(out)
        }
        FoldData::Ternary(_) => None,
    }
}

/// Ring-sum of commitments (the Ajtai map is linear over blocks).
fn ring_add(a: &RingCommitment, b: &RingCommitment) -> RingCommitment {
    const Q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;
    let mut out = RingCommitment::zero();
    for i in 0..almost_goldilocks_cuda::ajtai::KAPPA {
        for r in 0..RING_DIM {
            out.rows[i][r] = ((a.rows[i][r] as u128 + b.rows[i][r] as u128) % Q) as u64;
        }
    }
    out
}

/// Lift a leaf's claim point into the packed domain.
///
/// The two sides use opposite variable orders, and getting this wrong produces
/// a proof that verifies against nothing: `poly::evaluate_lagrange_basis_ext2`
/// is LSB-first (`r[i]` weights bit `i`), while the link's `eq_table` is
/// MSB-first so that a block prefix lands in the high bits. A packed index is
/// `(block_index << arity) | leaf_index`, so in MSB-first order the point is the
/// block prefix followed by the leaf's own point *reversed*.
fn lift_point(prefix: &[bool], leaf_pt: &[Ext2]) -> Vec<Ext2> {
    let mut point: Vec<Ext2> = prefix
        .iter()
        .map(|b| {
            Ext2::new(
                almost_goldilocks_cuda::field::AlmostGoldilocksField(u64::from(*b)),
                almost_goldilocks_cuda::field::AlmostGoldilocksField(0),
            )
        })
        .collect();
    point.extend(leaf_pt.iter().rev().copied());
    point
}

/// Prove the leaf set with the packed PCS.
///
/// Returns `None` when a leaf carries a representation the packed path does not
/// handle, so the caller can fall back to the fold tree rather than emit a proof
/// that silently omits a leaf.
pub fn prove_packed(
    leaves: &[FoldInstance],
    seed: Seed,
    transcript: &mut Transcript,
) -> Option<PackedPcsProof> {
    if leaves.is_empty() {
        return None;
    }
    let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
    let t0 = std::time::Instant::now();

    let hiding = hiding_block_coeffs();
    let max_leaf = leaves.iter().map(|l| l.arity).max().unwrap();
    // Ambient must hold the largest leaf and the hiding block.
    let mut ambient_arity = max_leaf + 1;
    while (1usize << ambient_arity) <= hiding + (1usize << max_leaf) {
        ambient_arity += 1;
    }

    let specs: Vec<LeafSpec> = leaves
        .iter()
        .enumerate()
        .map(|(i, l)| LeafSpec { key: LeafKey { edge: i, plane: 0 }, arity: l.arity })
        .collect();
    let layout = PackLayout::build(&specs, ambient_arity).ok()?;

    // Materialize packed witnesses.
    // Packed witnesses are stored bit-packed: leaves are binary, so a
    // coefficient-per-u64 image costs 64x the memory and the host time to write
    // it, for data the device wants as bits anyway.
    let size = 1usize << ambient_arity;
    let words = size / 64;
    let mut packed: Vec<Vec<u64>> = vec![vec![0u64; words]; layout.num_commitments];
    // Queries: one per leaf, its claim lifted by the block prefix.
    let mut queries: Vec<LinkQuery> = Vec::with_capacity(leaves.len());
    for (i, l) in leaves.iter().enumerate() {
        let place = layout.placements.get(&LeafKey { edge: i, plane: 0 })?;
        // Blocks are aligned to their own size and every leaf arity is >= 6, so
        // a block always starts on a word boundary and its bits copy wholesale.
        match &l.data {
            FoldData::Binary(src) => {
                let w0 = place.offset / 64;
                packed[place.commitment][w0..w0 + src.len()].copy_from_slice(src);
            }
            _ => {
                let coeffs = leaf_coeffs(&l.data, l.arity)?;
                let dst = &mut packed[place.commitment];
                for (i, c) in coeffs.iter().enumerate() {
                    if *c == 1 {
                        let idx = place.offset + i;
                        dst[idx / 64] |= 1u64 << (idx % 64);
                    } else if *c != 0 {
                        return None;
                    }
                }
            }
        }

        let prefix = place.block_prefix(ambient_arity);
        let point = lift_point(&prefix, &l.claim_pt);
        queries.push(LinkQuery {
            commitment: place.commitment,
            point,
            value: l.claim_val,
            prefix_len: prefix.len(),
        });
    }
    let t_pack = t0.elapsed();

    // Packed commitments: each block committed against its own column window,
    // ring-summed. Identical arithmetic to the existing per-leaf commit phase —
    // only the column offsets differ — so this replaces that cost rather than
    // adding to it.
    let t = std::time::Instant::now();
    let mut parts: Vec<Vec<RingCommitment>> = vec![Vec::new(); layout.num_commitments];
    for (i, l) in leaves.iter().enumerate() {
        let place = layout.placements.get(&LeafKey { edge: i, plane: 0 })?;
        // Leaf witnesses are binary, so they take the binary kernel at their own
        // column window — ~6x cheaper per coefficient than the wide kernel, which
        // exists for the Gaussian mask. The packed commitment is then the
        // ring-sum of its blocks, which is exact because the Ajtai map is linear.
        let c = match &l.data {
            FoldData::Binary(packed_bits) => {
                // Sparsity has to survive packing, or the commit phase loses the
                // property the whole scheme is built on: one-hot lookup advice is
                // 98% of committed elements, and committing it densely costs
                // ambient rather than support. `commit_sparse` takes coefficient
                // positions, so shifting every position by the block's offset
                // moves it to the right column window with no kernel change.
                let nnz: u32 = packed_bits.iter().map(|w| w.count_ones()).sum();
                let ambient = (packed_bits.len() * 64) as u32;
                if nnz * 8 < ambient {
                    let base = place.offset as u64;
                    let mut pos = Vec::with_capacity(nnz as usize);
                    for (wi, word) in packed_bits.iter().enumerate() {
                        let mut m = *word;
                        while m != 0 {
                            let b = m.trailing_zeros() as u64;
                            m &= m - 1;
                            pos.push(base + (wi as u64) * 64 + b);
                        }
                    }
                    if pos.is_empty() {
                        RingCommitment::zero()
                    } else {
                        ajtai::commit_sparse(seed, &pos, None).ok()?
                    }
                } else {
                    ajtai::commit_batched_at(seed, &[packed_bits], place.col_offset(), None)
                        .ok()?
                        .remove(0)
                }
            }
            _ => {
                let coeffs = leaf_coeffs(&l.data, l.arity)?;
                ajtai::commit_wide(seed, &coeffs, place.col_offset(), None).ok()?
            }
        };
        parts[place.commitment].push(c);
    }
    let commitments: Vec<RingCommitment> =
        parts.iter().map(|p| sum_commitments(p)).collect();
    let t_commit = t.elapsed();

    // Batch so each link's live state fits the budget.
    let per = state_bytes(ambient_arity);
    let batch = (budget_bytes() / per.max(1)).max(1);

    let t = std::time::Instant::now();
    let mut openings = Vec::new();
    // One scratch set reused by every batch: they are all the same shape, so
    // per-batch allocation of the multi-gigabyte buffers is pure churn.
    let mut scratch = LinkScratch::new();
    for start in (0..layout.num_commitments).step_by(batch) {
        let end = (start + batch).min(layout.num_commitments);
        let wits: Vec<LinkWitness> = packed[start..end]
            .iter()
            .map(|c| LinkWitness::binary(Vec::new(), c.clone()))
            .collect();
        let qs: Vec<LinkQuery> = queries
            .iter()
            .filter(|q| q.commitment >= start && q.commitment < end)
            .map(|q| LinkQuery {
                commitment: q.commitment - start,
                point: q.point.clone(),
                value: q.value,
                prefix_len: q.prefix_len,
            })
            .collect();
        transcript.append_u64(b"packed-batch", start as u64);
        let (link, xi) =
            prove_link_gpu_with(&wits, &qs, ambient_arity, transcript, &mut scratch)?;
        openings.push(PackedOpening {
            ambient_arity,
            commitments: commitments[start..end].to_vec(),
            link,
            xi,
        });
    }
    let t_link = t.elapsed();

    if timing {
        eprintln!(
            "[packed_pcs] leaves {} -> {} commitments @ arity {} ({} batches of <= {})\n\
             [packed_pcs]   pack {:.2}s  commit {:.2}s  link {:.2}s",
            leaves.len(),
            layout.num_commitments,
            ambient_arity,
            openings.len(),
            batch,
            t_pack.as_secs_f64(),
            t_commit.as_secs_f64(),
            t_link.as_secs_f64(),
        );
    }

    Some(PackedPcsProof {
        openings,
        ambient_arity,
        num_commitments: layout.num_commitments,
        layouts: Vec::new(),
    })
}


// ============================================================================
// Interleaved path
// ============================================================================

use crate::commit::interleave::{group_by_arity, InterleavedGroup};
use crate::pcs::link::{prove_link_interleaved, BlockClaim};
use rayon::prelude::*;

/// Which packing a group should use.
///
/// The two layouts pay for the query weights in different currencies.
/// Contiguous folds a dense weight table: `O(2^A)` memory and fold work per
/// commitment, independent of how sparse the witness is. Interleaved evaluates
/// the weight pointwise at `O(a - r)` per live position per round, so it scales
/// with the *support* — which is why it wins by 5x on small-arity groups and
/// loses at arity 26, where the a^2 factor overtakes the table.
///
/// Setting the two costs equal gives `density * a^2 ~ 2`, which is the rule
/// below. `ZK4_PCS_LAYOUT=interleaved|contiguous` forces one for A/B runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Interleaved,
    Contiguous,
}

pub fn choose_layout(leaf_arity: usize, density: f64) -> Layout {
    match std::env::var("ZK4_PCS_LAYOUT").ok().as_deref() {
        Some("interleaved") => return Layout::Interleaved,
        Some("contiguous") => return Layout::Contiguous,
        _ => {}
    }
    let a = leaf_arity as f64;
    if density * a * a < 2.0 {
        Layout::Interleaved
    } else {
        Layout::Contiguous
    }
}


/// What one arity group produces.
struct GroupResult {
    layout: Layout,
    openings: Vec<PackedOpening>,
    num_commitments: usize,
    ambient_arity: usize,
    t_pack: f64,
    t_commit: f64,
    t_link: f64,
}

/// Pack, commit and link one arity group on the current device.
fn prove_one_group(
    leaves: &[FoldInstance],
    seed: Seed,
    arity: usize,
    idxs: &[usize],
    budget: usize,
    transcript: &mut Transcript,
) -> Option<GroupResult> {
    // Which layout this group should use. Density is measured from the group's
    // own leaves, since it is what decides: interleaved pays O(a - r) per live
    // position per round, contiguous pays a dense table regardless.
    let mut nonzero = 0usize;
    for &li in idxs {
        if let FoldData::Binary(b) = &leaves[li].data {
            nonzero += b.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        } else {
            nonzero += 1usize << leaves[li].arity;
        }
    }
    let density = nonzero as f64 / ((idxs.len() as f64) * (1u64 << arity) as f64);
    if choose_layout(arity, density) == Layout::Contiguous {
        return prove_one_group_contiguous(leaves, seed, arity, idxs, budget, transcript);
    }

    let by_budget = InterleavedGroup::blocks_for_budget(arity, budget);
    let by_count = idxs.len().next_power_of_two();
    let target = by_budget.min(by_count);
    let g = InterleavedGroup::build(arity, idxs.len(), target)?;

    let size = 1usize << g.ambient_arity;
    let words = size / 64;
    let tp = std::time::Instant::now();
    let mut bits: Vec<Vec<u64>> = vec![vec![0u64; words]; g.num_commitments];
    let t_alloc = tp.elapsed().as_secs_f64();
    let mut claims: Vec<BlockClaim> = Vec::with_capacity(idxs.len());
    let mut positions: Vec<Vec<u64>> = vec![Vec::new(); g.num_commitments];
    let mut t_scatter = 0f64;
    let mut nnz_total = 0usize;

    for (k, &li) in idxs.iter().enumerate() {
        let l = &leaves[li];
        let slot = g.slots[k];
        let owned: Vec<u64>;
        let src: &[u64] = match &l.data {
            FoldData::Binary(b) => b,
            _ => {
                let c = leaf_coeffs(&l.data, l.arity)?;
                let mut packed = vec![0u64; (1usize << l.arity) / 64];
                for (i, v) in c.iter().enumerate() {
                    match v {
                        0 => {}
                        1 => packed[i / 64] |= 1u64 << (i % 64),
                        _ => return None,
                    }
                }
                owned = packed;
                &owned
            }
        };
        let ts = std::time::Instant::now();
        for (wi, word) in src.iter().enumerate() {
            let mut m = *word;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                m &= m - 1;
                let p = g.position(slot.block, wi * 64 + b);
                bits[slot.commitment][p / 64] |= 1u64 << (p % 64);
                positions[slot.commitment].push(p as u64);
                nnz_total += 1;
            }
        }
        t_scatter += ts.elapsed().as_secs_f64();
        let point: Vec<Ext2> = l.claim_pt.iter().rev().copied().collect();
        claims.push(BlockClaim {
            commitment: slot.commitment,
            block: slot.block,
            point,
            value: l.claim_val,
        });
    }
    let t_pack = tp.elapsed().as_secs_f64();
    if std::env::var("ZK4_PACK_TIMING").is_ok() {
        eprintln!(
            "[pack] arity {:>2} leaves {:>5} commits {:>4} ambient {:>2} words {:>9} | alloc {:.2}s scatter {:.2}s nnz {:.3e} other {:.2}s",
            arity, idxs.len(), g.num_commitments, g.ambient_arity, words,
            t_alloc, t_scatter, nnz_total as f64, t_pack - t_alloc - t_scatter,
        );
    }

    let tc = std::time::Instant::now();
    let mut commitments = Vec::with_capacity(g.num_commitments);
    for pos in &positions {
        if pos.is_empty() {
            commitments.push(RingCommitment::zero());
        } else {
            commitments.push(ajtai::commit_sparse(seed, pos, None).ok()?);
        }
    }
    let t_commit = tc.elapsed().as_secs_f64();

    let per = g.state_bytes();
    let chunk = (budget / per.max(1)).max(1);
    let tl = std::time::Instant::now();
    let mut scratch = crate::pcs::link::LinkScratch::new();
    let mut openings = Vec::new();
    for start in (0..g.num_commitments).step_by(chunk) {
        let end = (start + chunk).min(g.num_commitments);
        let sub_bits: Vec<Vec<u64>> = bits[start..end].to_vec();
        let sub_claims: Vec<BlockClaim> = claims
            .iter()
            .filter(|c| c.commitment >= start && c.commitment < end)
            .map(|c| BlockClaim {
                commitment: c.commitment - start,
                block: c.block,
                point: c.point.clone(),
                value: c.value,
            })
            .collect();
        transcript.append_u64(b"packed-chunk", start as u64);
        let (link, xi) = prove_link_interleaved(
            &sub_bits, &sub_claims, arity, g.block_bits, transcript, &mut scratch,
        )?;
        openings.push(PackedOpening {
            ambient_arity: g.ambient_arity,
            commitments: commitments[start..end].to_vec(),
            link,
            xi,
        });
    }
    let t_link = tl.elapsed().as_secs_f64();

    Some(GroupResult {
        layout: Layout::Interleaved,
        openings,
        num_commitments: g.num_commitments,
        ambient_arity: g.ambient_arity,
        t_pack,
        t_commit,
        t_link,
    })
}


/// Contiguous packing for one group: block in the HIGH bits, weights folded as a
/// dense table.
///
/// Chosen when the group is dense enough or its leaves large enough that the
/// interleaved layout's per-position weight evaluation — `O(a - r)` multiplies,
/// so `O(a^2)` over the sumcheck — costs more than the table it avoids. For
/// GPT-2 12L that is the arity-26 group, which is over half the leaf
/// coefficients.
fn prove_one_group_contiguous(
    leaves: &[FoldInstance],
    seed: Seed,
    arity: usize,
    idxs: &[usize],
    budget: usize,
    transcript: &mut Transcript,
) -> Option<GroupResult> {
    let hiding = hiding_block_coeffs();
    let mut ambient = arity + 1;
    while (1usize << ambient) <= hiding + (1usize << arity) {
        ambient += 1;
    }
    // Grow the ambient to pack more leaves per commitment while the state still
    // fits the budget. Stopping at the minimum is a trap worth naming: it yields
    // exactly one leaf per commitment, which turned a 1624-leaf GPT-2 into 691
    // commitments and made the packed path lose to the fold tree outright.
    // Contiguous keeps both the witness and the weight table, hence the 2x.
    while ambient + 1 < 40 && 2 * (1usize << (ambient + 1)) * 16 <= budget {
        ambient += 1;
        if ((1usize << ambient) - hiding) >> arity >= idxs.len() {
            break; // no point growing past the whole group
        }
    }
    let capacity = (1usize << ambient) - hiding;
    let per_commit = capacity >> arity;
    if per_commit == 0 {
        return None;
    }
    let num_commitments = idxs.len().div_ceil(per_commit);

    let size = 1usize << ambient;
    let words = size / 64;
    let tp = std::time::Instant::now();
    // Built by extension, not allocated zeroed up front. Blocks are filled in
    // order, so each image is the concatenation of its leaves' bits followed by
    // a zero tail — and only that tail needs zeroing. Allocating `words` zeroed
    // per commitment zeroes the whole image and then overwrites 75% of it: 15 GB
    // of memset on GPT-2 seq256, measured at 11.1 s.
    let mut bits: Vec<Vec<u64>> = Vec::with_capacity(num_commitments);
    let t_alloc = tp.elapsed().as_secs_f64();
    let mut t_copy = 0f64;
    let mut t_pos = 0f64;
    let mut nnz_total = 0usize;
    let mut dense_total = 0usize;
    let mut positions: Vec<Vec<u64>> = vec![Vec::new(); num_commitments];
    // Dense leaves, kept as (column offset, packed bits) for the binary kernel.
    let mut dense_blocks: Vec<Vec<(u64, Vec<u64>)>> = vec![Vec::new(); num_commitments];
    let mut queries: Vec<LinkQuery> = Vec::with_capacity(idxs.len());
    let prefix_len = ambient - arity;

    for (k, &li) in idxs.iter().enumerate() {
        let l = &leaves[li];
        let commitment = k / per_commit;
        let block = k % per_commit;
        let offset = block << arity;
        // Borrow the leaf's bits rather than cloning them. The clone was copied
        // straight into `bits` on the next line and then dropped — 336 leaves of
        // 2^28 bits is 11 GB moved for nothing, and it measured 6.6 s on GPT-2
        // seq256. Only the non-binary case needs an owned buffer.
        let owned: Vec<u64>;
        let src: &[u64] = match &l.data {
            FoldData::Binary(b) => b,
            _ => {
                let c = leaf_coeffs(&l.data, l.arity)?;
                let mut packed = vec![0u64; (1usize << l.arity) / 64];
                for (i, v) in c.iter().enumerate() {
                    match v {
                        0 => {}
                        1 => packed[i / 64] |= 1u64 << (i % 64),
                        _ => return None,
                    }
                }
                owned = packed;
                &owned
            }
        };
        let tcp = std::time::Instant::now();
        if block == 0 {
            bits.push(Vec::with_capacity(words));
        }
        let img = bits.last_mut().expect("block 0 pushes the image");
        debug_assert_eq!(img.len(), offset / 64, "blocks must be filled in order");
        img.extend_from_slice(src);
        t_copy += tcp.elapsed().as_secs_f64();
        let tpo = std::time::Instant::now();
        // A position list costs 8 bytes per NONZERO, so building one for a dense
        // leaf is enormous — a dense arity-26 leaf is 537 MB of positions, and
        // that alone was 20 s of GPT-2 seq256's packing. Blocks are contiguous in
        // this layout, so a dense leaf commits directly from its bits at the
        // block's column window instead.
        let nnz: usize = src.iter().map(|w| w.count_ones() as usize).sum();
        if nnz * 8 < (src.len() * 64) {
            for (wi, word) in src.iter().enumerate() {
                let mut m = *word;
                while m != 0 {
                    let b = m.trailing_zeros() as u64;
                    m &= m - 1;
                    positions[commitment].push(offset as u64 + (wi as u64) * 64 + b);
                }
            }
            nnz_total += nnz;
        } else {
            dense_total += 1;
            dense_blocks[commitment].push((offset as u64 / RING_DIM as u64, src.to_vec()));
        }
        t_pos += tpo.elapsed().as_secs_f64();
        // Block prefix in the high bits, then the leaf point reversed (the repo
        // is LSB-first, the link is MSB-first).
        let mut point: Vec<Ext2> = (0..prefix_len)
            .map(|j| {
                let bit = (block >> (prefix_len - 1 - j)) & 1;
                Ext2::new(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(bit as u64),
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(0),
                )
            })
            .collect();
        point.extend(l.claim_pt.iter().rev().copied());
        queries.push(LinkQuery {
            commitment,
            point,
            value: l.claim_val,
            prefix_len,
        });
    }
    let tz = std::time::Instant::now();
    for img in bits.iter_mut() {
        img.resize(words, 0);
    }
    let t_alloc = t_alloc + tz.elapsed().as_secs_f64();

    let t_pack = tp.elapsed().as_secs_f64();
    if std::env::var("ZK4_PACK_TIMING").is_ok() {
        eprintln!(
            "[pack/contig] arity {:>2} leaves {:>5} commits {:>4} ambient {:>2} words {:>9} | \
             alloc {:.2}s copy {:.2}s pos {:.2}s nnz {:.3e} dense-leaves {} other {:.2}s",
            arity, idxs.len(), num_commitments, ambient, words,
            t_alloc, t_copy, t_pos, nnz_total as f64, dense_total,
            t_pack - t_alloc - t_copy - t_pos,
        );
    }

    let tc = std::time::Instant::now();
    let mut commitments = Vec::with_capacity(num_commitments);
    for (c, pos) in positions.iter().enumerate() {
        // The commitment is the ring-sum of its blocks, so sparse and dense
        // blocks can use different kernels and simply add.
        let mut acc = if pos.is_empty() {
            RingCommitment::zero()
        } else {
            ajtai::commit_sparse(seed, pos, None).ok()?
        };
        for (col_off, blk) in &dense_blocks[c] {
            let part = ajtai::commit_batched_at(seed, &[blk], *col_off, None)
                .ok()?
                .remove(0);
            acc = ring_add(&acc, &part);
        }
        commitments.push(acc);
    }
    let t_commit = tc.elapsed().as_secs_f64();

    // Contiguous keeps witness AND weights, so its per-commitment state is 4x
    // the interleaved layout's and it chunks sooner.
    let per = 2 * size * 16;
    let chunk = (budget / per.max(1)).max(1);
    let tl = std::time::Instant::now();
    let mut scratch = crate::pcs::link::LinkScratch::new();
    let mut openings = Vec::new();
    for start in (0..num_commitments).step_by(chunk) {
        let end = (start + chunk).min(num_commitments);
        let wits: Vec<LinkWitness> = bits[start..end]
            .iter()
            .map(|b| LinkWitness::binary(Vec::new(), b.clone()))
            .collect();
        let qs: Vec<LinkQuery> = queries
            .iter()
            .filter(|q| q.commitment >= start && q.commitment < end)
            .map(|q| LinkQuery {
                commitment: q.commitment - start,
                point: q.point.clone(),
                value: q.value,
                prefix_len: q.prefix_len,
            })
            .collect();
        transcript.append_u64(b"packed-chunk", start as u64);
        let (link, xi) =
            crate::pcs::link::prove_link_gpu_with(&wits, &qs, ambient, transcript, &mut scratch)?;
        openings.push(PackedOpening {
            ambient_arity: ambient,
            commitments: commitments[start..end].to_vec(),
            link,
            xi,
        });
    }
    let t_link = tl.elapsed().as_secs_f64();

    Some(GroupResult {
        layout: Layout::Contiguous,
        openings,
        num_commitments,
        ambient_arity: ambient,
        t_pack,
        t_commit,
        t_link,
    })
}

/// Prove the leaf set with the interleaved layout.
///
/// Leaves are grouped by arity — the layout needs a uniform leaf size so the
/// block index occupies a fixed low bit-field — and each group is packed with
/// its blocks interleaved. The link then evaluates query weights on demand
/// rather than folding a table, which takes live state from 32 bytes per
/// witness bit to 8 and correspondingly reduces how many batches a model needs.
pub fn prove_packed_interleaved(
    leaves: &[FoldInstance],
    seed: Seed,
    transcript: &mut Transcript,
) -> Option<PackedPcsProof> {
    if leaves.is_empty() {
        return None;
    }
    let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
    let t0 = std::time::Instant::now();

    let arities: Vec<usize> = leaves.iter().map(|l| l.arity).collect();
    let groups = group_by_arity(&arities);
    let budget = budget_bytes();

    // Arity groups are independent sub-protocols, so each runs on a forked
    // transcript and they go in parallel across devices. This is the axis that
    // actually has width: interleaving deliberately produces few commitments
    // (16 for GPT-2 12L), so sharding *those* across GPUs leaves most devices
    // idle — measured at 16.0 s on one GPU against 21.0 s on four. Groups are
    // plentiful and independent, and the fold tree already establishes forking
    // as the pattern for exactly this.
    let devices = crate::fold::tree::gpu_device_pool();
    let forks: Vec<Transcript> = (0..groups.len())
        .map(|gi| transcript.fork(b"pcs-arity-group", gi))
        .collect();

    // One group per device at a time, work-stealing rather than in lockstep
    // waves. Group cost is very uneven — for GPT-2 12L a single arity-26 group
    // is over half the leaf coefficients — so a fixed wave schedule leaves
    // devices idle waiting for it. Largest-first plus stealing keeps them fed.
    let width = devices.len().max(1);
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&gi| std::cmp::Reverse(groups[gi].1.len() << groups[gi].0.min(40)));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .ok()?;
    let mut slotted: Vec<Option<GroupResult>> = pool.install(|| {
        order
            .par_iter()
            .map(|&gi| {
                let device = devices[rayon::current_thread_index().unwrap_or(0) % width];
                let _ = almost_goldilocks_cuda::set_device(device);
                let (arity, idxs) = &groups[gi];
                let mut fork = forks[gi].clone();
                (gi, prove_one_group(leaves, seed, *arity, idxs, budget, &mut fork))
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .fold(
        (0..groups.len()).map(|_| None).collect::<Vec<Option<GroupResult>>>(),
        |mut acc, (gi, r)| {
            acc[gi] = r;
            acc
        },
    );
    let per_group: Vec<Option<GroupResult>> = slotted.drain(..).collect();

    let mut openings = Vec::new();
    let mut layouts = Vec::with_capacity(groups.len());
    let mut total_commitments = 0usize;
    let mut max_ambient = 0usize;
    let (mut t_pack, mut t_commit, mut t_link) = (0f64, 0f64, 0f64);
    for r in per_group {
        let r = r?;
        layouts.push(r.layout);
        t_pack += r.t_pack;
        t_commit += r.t_commit;
        t_link += r.t_link;
        max_ambient = max_ambient.max(r.ambient_arity);
        total_commitments += r.num_commitments;
        // Absorb each group's output into the parent in group order, so anything
        // downstream is bound to all of them.
        for op in &r.openings {
            for m in &op.link.rounds {
                for v in m.norm.iter().chain(m.eval.iter()) {
                    transcript.append_ext2(b"pcs-group-round", v);
                }
            }
            for a in &op.link.terminal {
                transcript.append_ext2(b"pcs-group-terminal", a);
            }
        }
        openings.extend(r.openings);
    }

    if timing {
        eprintln!(
            "[packed_pcs/interleaved] leaves {} -> {} commitments across {} arity groups\n\
             [packed_pcs/interleaved]   pack {:.2}s  commit {:.2}s  link {:.2}s  (total {:.2}s)",
            leaves.len(), total_commitments, groups.len(),
            t_pack, t_commit, t_link, t0.elapsed().as_secs_f64(),
        );
    }

    Some(PackedPcsProof {
        openings,
        ambient_arity: max_ambient,
        num_commitments: total_commitments,
        layouts,
    })
}


/// Verify one contiguously-packed group.
fn verify_one_group_contiguous<'a, I>(
    leaves_meta: &[(usize, Vec<Ext2>, Ext2)],
    idxs: &[usize],
    arity: usize,
    budget: usize,
    ops: &mut I,
    transcript: &mut Transcript,
) -> bool
where
    I: Iterator<Item = &'a PackedOpening>,
{
    let hiding = hiding_block_coeffs();
    let mut ambient = arity + 1;
    while (1usize << ambient) <= hiding + (1usize << arity) {
        ambient += 1;
    }
    while ambient + 1 < 40 && 2 * (1usize << (ambient + 1)) * 16 <= budget {
        ambient += 1;
        if ((1usize << ambient) - hiding) >> arity >= idxs.len() {
            break;
        }
    }
    let capacity = (1usize << ambient) - hiding;
    let per_commit = capacity >> arity;
    if per_commit == 0 {
        return false;
    }
    let num_commitments = idxs.len().div_ceil(per_commit);
    let prefix_len = ambient - arity;
    let size = 1usize << ambient;
    let per = 2 * size * 16;
    let chunk = (budget / per.max(1)).max(1);

    for start in (0..num_commitments).step_by(chunk) {
        let end = (start + chunk).min(num_commitments);
        let op = match ops.next() {
            Some(o) => o,
            None => return false,
        };
        if op.ambient_arity != ambient || end - start != op.commitments.len() {
            return false;
        }
        let qs: Vec<LinkQuery> = idxs
            .iter()
            .enumerate()
            .filter(|(k, _)| k / per_commit >= start && k / per_commit < end)
            .map(|(k, &li)| {
                let block = k % per_commit;
                let mut point: Vec<Ext2> = (0..prefix_len)
                    .map(|j| {
                        let bit = (block >> (prefix_len - 1 - j)) & 1;
                        Ext2::new(
                            almost_goldilocks_cuda::field::AlmostGoldilocksField(bit as u64),
                            almost_goldilocks_cuda::field::AlmostGoldilocksField(0),
                        )
                    })
                    .collect();
                point.extend(leaves_meta[li].1.iter().rev().copied());
                LinkQuery {
                    commitment: k / per_commit - start,
                    point,
                    value: leaves_meta[li].2,
                    prefix_len,
                }
            })
            .collect();
        transcript.append_u64(b"packed-chunk", start as u64);
        if verify_link(op.commitments.len(), &qs, ambient, &op.link, transcript).is_none() {
            return false;
        }
    }
    true
}

/// Verify an interleaved proof. Rebuilds the same grouping and block claims from
/// the leaf metadata, so nothing about the layout travels in the proof.
pub fn verify_packed_interleaved(
    leaves_meta: &[(usize, Vec<Ext2>, Ext2)],
    proof: &PackedPcsProof,
    transcript: &mut Transcript,
) -> bool {
    let arities: Vec<usize> = leaves_meta.iter().map(|(a, _, _)| *a).collect();
    let groups = group_by_arity(&arities);
    if proof.layouts.len() != groups.len() {
        return false;
    }
    let budget = budget_bytes();
    let mut op_iter = proof.openings.iter();

    // Same forking the prover used: each group verifies against its own branch
    // of the parent transcript, and the parent then absorbs every group's output
    // in group order.
    let mut verified: Vec<Vec<&PackedOpening>> = Vec::with_capacity(groups.len());
    for (gi, (arity, idxs)) in groups.iter().enumerate() {
        let mut fork = transcript.fork(b"pcs-arity-group", gi);
        // Same layout decision as the prover, from the same public inputs — the
        // verifier has the leaf arities and the openings' ambient arity, and the
        // density comes from the proof's own shape rather than the witness.
        if proof.layouts.get(gi) != Some(&Layout::Interleaved) {
            if !verify_one_group_contiguous(leaves_meta, idxs, *arity, budget, &mut op_iter, &mut fork) {
                return false;
            }
            continue;
        }
        let by_budget = InterleavedGroup::blocks_for_budget(*arity, budget);
        let by_count = idxs.len().next_power_of_two();
        let target = by_budget.min(by_count);
        let g = match InterleavedGroup::build(*arity, idxs.len(), target) {
            Some(g) => g,
            None => return false,
        };
        let per = g.state_bytes();
        let chunk = (budget / per.max(1)).max(1);
        let mut mine = Vec::new();
        for start in (0..g.num_commitments).step_by(chunk) {
            let end = (start + chunk).min(g.num_commitments);
            let op = match op_iter.next() {
                Some(o) => o,
                None => return false,
            };
            if g.ambient_arity != op.ambient_arity || end - start != op.commitments.len() {
                return false;
            }
            let qs: Vec<LinkQuery> = idxs
                .iter()
                .enumerate()
                .filter(|(k, _)| {
                    g.slots[*k].commitment >= start && g.slots[*k].commitment < end
                })
                .map(|(k, &li)| {
                    let slot = g.slots[k];
                    let mut point: Vec<Ext2> =
                        leaves_meta[li].1.iter().rev().copied().collect();
                    for b in 0..g.block_bits {
                        let bit = (slot.block >> (g.block_bits - 1 - b)) & 1;
                        point.push(Ext2::new(
                            almost_goldilocks_cuda::field::AlmostGoldilocksField(bit as u64),
                            almost_goldilocks_cuda::field::AlmostGoldilocksField(0),
                        ));
                    }
                    LinkQuery {
                        commitment: slot.commitment - start,
                        point,
                        value: leaves_meta[li].2,
                        prefix_len: 0,
                    }
                })
                .collect();
            fork.append_u64(b"packed-chunk", start as u64);
            if verify_link(op.commitments.len(), &qs, op.ambient_arity, &op.link, &mut fork)
                .is_none()
            {
                return false;
            }
            mine.push(op);
        }
        verified.push(mine);
    }
    if op_iter.next().is_some() {
        return false;
    }
    for group_ops in &verified {
        for op in group_ops {
            for m in &op.link.rounds {
                for v in m.norm.iter().chain(m.eval.iter()) {
                    transcript.append_ext2(b"pcs-group-round", v);
                }
            }
            for a in &op.link.terminal {
                transcript.append_ext2(b"pcs-group-terminal", a);
            }
        }
    }
    true
}

/// Verify a packed proof against the leaf metadata the verifier already holds.
///
/// Takes the same `(arity, claim_pt, claim_val)` triples the fold-tree verifier
/// takes, so the two paths are interchangeable at the call site.
pub fn verify_packed(
    leaves_meta: &[(usize, Vec<Ext2>, Ext2)],
    proof: &PackedPcsProof,
    transcript: &mut Transcript,
) -> bool {
    let specs: Vec<LeafSpec> = leaves_meta
        .iter()
        .enumerate()
        .map(|(i, (arity, _, _))| LeafSpec { key: LeafKey { edge: i, plane: 0 }, arity: *arity })
        .collect();
    let layout = match PackLayout::build(&specs, proof.ambient_arity) {
        Ok(l) => l,
        Err(_) => return false,
    };
    if layout.num_commitments != proof.num_commitments {
        return false;
    }

    let mut queries: Vec<LinkQuery> = Vec::with_capacity(leaves_meta.len());
    for (i, (_, pt, val)) in leaves_meta.iter().enumerate() {
        let place = match layout.placements.get(&LeafKey { edge: i, plane: 0 }) {
            Some(p) => p,
            None => return false,
        };
        let prefix = place.block_prefix(proof.ambient_arity);
        let point = lift_point(&prefix, pt);
        queries.push(LinkQuery {
            commitment: place.commitment,
            point,
            value: *val,
            prefix_len: prefix.len(),
        });
    }

    let mut start = 0usize;
    for op in &proof.openings {
        let end = start + op.commitments.len();
        let qs: Vec<LinkQuery> = queries
            .iter()
            .filter(|q| q.commitment >= start && q.commitment < end)
            .map(|q| LinkQuery {
                commitment: q.commitment - start,
                point: q.point.clone(),
                value: q.value,
                prefix_len: q.prefix_len,
            })
            .collect();
        transcript.append_u64(b"packed-batch", start as u64);
        if verify_link(op.commitments.len(), &qs, op.ambient_arity, &op.link, transcript).is_none() {
            return false;
        }
        start = end;
    }
    start == proof.num_commitments
}

/// Ring-sum helper exposed so a caller can check a packed commitment against the
/// per-leaf commitments it replaces.
pub fn sum_commitments(parts: &[RingCommitment]) -> RingCommitment {
    let mut acc = RingCommitment::zero();
    for p in parts {
        acc = ring_add(&acc, p);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::FoldInstance;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn leaf(arity: usize, rng: &mut Rng) -> FoldInstance {
        let packed: Vec<u64> = (0..(1usize << (arity - 6))).map(|_| rng.next()).collect();
        let data = FoldData::Binary(packed);
        let pt: Vec<Ext2> = (0..arity)
            .map(|_| {
                Ext2::new(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(rng.next() >> 4),
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(rng.next() >> 4),
                )
            })
            .collect();
        let val = data.evaluate_at_ext2(&pt);
        FoldInstance {
            commitment: RingCommitment::zero(),
            data,
            arity,
            claim_pt: pt,
            claim_val: val,
        }
    }

    #[test]
    fn packed_path_proves_and_verifies_a_mixed_arity_leaf_set() {
        let mut rng = Rng(0xF00D_BEEF_1234_5678);
        let seed = Seed([2, 7, 1, 8, 2, 8, 1, 8]);
        // Mixed arities, as a real leaf set has.
        let mut leaves = Vec::new();
        for _ in 0..3 {
            leaves.push(leaf(16, &mut rng));
        }
        for _ in 0..8 {
            leaves.push(leaf(14, &mut rng));
        }
        for _ in 0..20 {
            leaves.push(leaf(12, &mut rng));
        }

        let mut tp = Transcript::new(b"packed-int");
        let proof = prove_packed(&leaves, seed, &mut tp).expect("packed prove");

        let meta: Vec<(usize, Vec<Ext2>, Ext2)> = leaves
            .iter()
            .map(|l| (l.arity, l.claim_pt.clone(), l.claim_val))
            .collect();
        let mut tv = Transcript::new(b"packed-int");
        assert!(verify_packed(&meta, &proof, &mut tv), "honest packed proof must verify");
    }

    #[test]
    fn a_wrong_leaf_claim_is_rejected() {
        let mut rng = Rng(0x0BAD_0BAD_0BAD_0BAD);
        let seed = Seed([1, 1, 2, 3, 5, 8, 13, 21]);
        let leaves: Vec<FoldInstance> = (0..6).map(|_| leaf(13, &mut rng)).collect();

        let mut tp = Transcript::new(b"packed-bad");
        let proof = prove_packed(&leaves, seed, &mut tp).expect("prove");

        let mut meta: Vec<(usize, Vec<Ext2>, Ext2)> = leaves
            .iter()
            .map(|l| (l.arity, l.claim_pt.clone(), l.claim_val))
            .collect();
        meta[2].2 = Ext2::new(
            almost_goldilocks_cuda::field::AlmostGoldilocksField(meta[2].2.c0.0 ^ 1),
            meta[2].2.c1,
        );
        let mut tv = Transcript::new(b"packed-bad");
        assert!(!verify_packed(&meta, &proof, &mut tv));
    }

    #[test]
    fn packed_commitment_equals_the_sum_of_its_blocks() {
        // The property that lets the commit phase move to packed offsets without
        // changing its cost: a packed commitment is the ring-sum of its blocks
        // committed at their own column windows.
        let mut rng = Rng(0x5150_5150_5150_5150);
        let seed = Seed([4, 4, 4, 4, 4, 4, 4, 4]);
        let arity = 14usize;
        let half = 1usize << (arity - 1);
        let full: Vec<u64> = (0..(1usize << arity)).map(|_| rng.next() & 1).collect();

        let whole = ajtai::commit_wide(seed, &full, 0, None).expect("whole");
        let lo = ajtai::commit_wide(seed, &full[..half], 0, None).expect("lo");
        let hi = ajtai::commit_wide(seed, &full[half..], (half / RING_DIM) as u64, None)
            .expect("hi");
        let summed = sum_commitments(&[lo, hi]);

        const Q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;
        for i in 0..almost_goldilocks_cuda::ajtai::KAPPA {
            for r in 0..RING_DIM {
                assert_eq!(
                    (whole.rows[i][r] as u128) % Q,
                    (summed.rows[i][r] as u128) % Q,
                    "row {} coeff {}", i, r
                );
            }
        }
    }
}
