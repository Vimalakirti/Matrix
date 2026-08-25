//! Same-point sumcheck (plan §6.1).
//!
//! Given M heterogeneous instances `{(f_i, r_i, y_i)}` with
//! `f_i : F^{k_i} → F`, `r_i ∈ F^{k_i}`, and `y_i = f_i(r_i)`, the
//! sumcheck identity
//!
//! ```text
//! Σ_i α^i · y_i · 2^{N − k_i}
//!   = Σ_{x ∈ {0,1}^N} Σ_i α^i · eq(r_i, x_{[..k_i]}) · f_i(x_{[..k_i]})
//! ```
//!
//! (where `N = max_num_vars`) reduces all instances to a shared
//! challenge `R ∈ F^N`. After `N` rounds the prover supplies
//! `f_i(R_{[..k_i]})` for each instance; the verifier checks the
//! sumcheck against the α-power-weighted combination of those evals
//! and pins them in for the next stage (multifold) by the
//! α-randomization.
//!
//! Per-instance state is compact: we keep two `2^{k_i − rounds_done}`-sized
//! Ext2 tables for `(eq_i, f_i)` while `rounds_done ≤ k_i`. Once
//! `rounds_done > k_i` the instance becomes a single scalar `h_i(R[..k_i])`
//! that contributes a constant `α^i · h_i · 2^{N − round_idx}` to every
//! subsequent round message — no folding work is done past arity.
//!
//! The round polynomial is degree 2 (eq · f are each degree-1 multilinear
//! in the live variable), so each round message ships 3 evaluations
//! `T(0), T(1), T(2)`.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use serde::{Deserialize, Serialize};

use crate::fold::FoldInstance;
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{
    calc_pow_vec_ext2, ext2_add, ext2_field_eq, ext2_inv, ext2_mul, ext2_sub,
};

/// Same-point sumcheck proof. Includes the standard sumcheck transcript
/// plus the per-instance evaluation reveals `f_i(R[..k_i])`. The
/// verifier needs the latter to (a) check the sumcheck's final eval and
/// (b) carry concrete y'_i forward to the multifold stage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamePointProof {
    /// Standard sumcheck transcript: `N` round messages of length 3
    /// (degree-2 round polynomial), plus a final evaluation.
    pub sumcheck: SumcheckProof,
    /// `f_evals[i] = f_i(R[..k_i])` for each input instance.
    pub f_evals: Vec<AlmostGoldilocksExt2>,
}

/// Run the same-point sumcheck prover. Mutates the transcript to absorb
/// challenges and emit round messages. Returns the proof plus the shared
/// challenge `R` (length `max_num_vars`) — needed by the caller to build
/// the next-stage `FoldInstance` claim points.
pub fn prove_same_point(
    instances: &[FoldInstance],
    max_num_vars: usize,
    transcript: &mut Transcript,
) -> (SamePointProof, Vec<AlmostGoldilocksExt2>) {
    assert!(!instances.is_empty(), "same-point sumcheck needs ≥ 1 instance");
    for (i, inst) in instances.iter().enumerate() {
        assert!(
            inst.arity <= max_num_vars,
            "instance {} arity {} > max_num_vars {}",
            i, inst.arity, max_num_vars,
        );
        assert_eq!(
            inst.claim_pt.len(), inst.arity,
            "instance {} claim_pt len {} != arity {}",
            i, inst.claim_pt.len(), inst.arity,
        );
    }

    // Transcript header — bind to the structural parameters before we
    // sample α.
    transcript.append_u64(b"sp_num_var", max_num_vars as u64);
    transcript.append_u64(b"sp_num_inst", instances.len() as u64);
    for (i, inst) in instances.iter().enumerate() {
        transcript.append_u64(b"sp_arity_i", inst.arity as u64);
        for c in &inst.claim_pt { transcript.append_ext2(b"sp_r_i", c); }
        transcript.append_ext2(b"sp_y_i", &inst.claim_val);
        let _ = i;
    }

    // α-power weights for instance combination.
    let alpha = transcript.challenge_ext2(b"sp_alpha");
    let alphas = calc_pow_vec_ext2(alpha, instances.len());

    // Build per-instance live tables in parallel — at arity 22 each eq
    // table is ~64 MB and takes ~30 ms; doing 42 serially was a hidden
    // ~1.3 s cost. Parallel across leaves.
    use rayon::prelude::*;
    let t_state_init = std::time::Instant::now();
    let mut states: Vec<InstanceState> = instances.par_iter().enumerate().map(|(i, inst)| {
        InstanceState::new(inst, alphas[i], max_num_vars)
    }).collect();
    let timing_init = std::env::var("ZK4_TIMING").is_ok() && max_num_vars >= 18;
    if timing_init {
        eprintln!("[sp arity={} leaves={}] state_init={:?}",
            max_num_vars, instances.len(), t_state_init.elapsed());
    }

    // Round-by-round sumcheck. Parallelize across leaves (outer); each
    // leaf folds serially internally to avoid nested rayon overhead and
    // memory-allocator contention from concurrent large Vec allocs.
    use rayon::prelude::*;
    let timing = std::env::var("ZK4_TIMING").is_ok() && max_num_vars >= 18;
    let mut round_messages: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(max_num_vars);
    let mut challenges: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(max_num_vars);
    let mut t_msg = std::time::Duration::ZERO;
    let mut t_fold = std::time::Duration::ZERO;
    for round in 0..max_num_vars {
        let t0 = std::time::Instant::now();
        let msg = compute_round_message(&states, round, max_num_vars);
        let t1 = std::time::Instant::now();
        t_msg += t1 - t0;
        for m in &msg { transcript.append_ext2(b"sp_round_msg", m); }
        let r = transcript.challenge_ext2(b"sp_round_challenge");
        round_messages.push(msg);
        challenges.push(r);
        states.par_iter_mut().for_each(|s| s.absorb(r, round));
        let t2 = std::time::Instant::now();
        t_fold += t2 - t1;
        if timing && round <= 3 {
            eprintln!("[sp arity={} leaves={}] round {} msg={:?} fold={:?}",
                max_num_vars, instances.len(), round, t1 - t0, t2 - t1);
        }
    }
    if timing {
        eprintln!("[sp arity={} leaves={}] TOTAL msg={:?} fold={:?}",
            max_num_vars, instances.len(), t_msg, t_fold);
    }

    // Reveal per-instance f_i(R[..k_i]).
    let f_evals: Vec<AlmostGoldilocksExt2> = states.iter().map(|s| s.f_final()).collect();
    for e in &f_evals { transcript.append_ext2(b"sp_f_eval", e); }

    // Compute final_eval = Σ_i α^i · eq(r_i, R[..k_i]) · f_i(R[..k_i]).
    // (Matches what the verifier will compute against the sumcheck's
    // last round message interpolated at the last challenge.)
    let mut final_eval = AlmostGoldilocksExt2::zero();
    for (i, s) in states.iter().enumerate() {
        let eq_i = eq_eval_ext2(&instances[i].claim_pt, &challenges[..instances[i].arity]);
        final_eval = ext2_add(final_eval, ext2_mul(alphas[i], ext2_mul(eq_i, s.f_final())));
    }

    (
        SamePointProof {
            sumcheck: SumcheckProof { final_eval, round_messages },
            f_evals,
        },
        challenges,
    )
}

/// Verify a same-point sumcheck. Returns `Some(R)` on success — the
/// shared challenge the caller forwards to the multifold stage.
pub fn verify_same_point(
    instances_meta: &[(usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)],
    // (arity_i, r_i, y_i)
    max_num_vars: usize,
    proof: &SamePointProof,
    transcript: &mut Transcript,
) -> Option<Vec<AlmostGoldilocksExt2>> {
    if instances_meta.is_empty() { return None; }

    transcript.append_u64(b"sp_num_var", max_num_vars as u64);
    transcript.append_u64(b"sp_num_inst", instances_meta.len() as u64);
    for (arity, r, y) in instances_meta {
        transcript.append_u64(b"sp_arity_i", *arity as u64);
        for c in r { transcript.append_ext2(b"sp_r_i", c); }
        transcript.append_ext2(b"sp_y_i", y);
    }

    let alpha = transcript.challenge_ext2(b"sp_alpha");
    let alphas = calc_pow_vec_ext2(alpha, instances_meta.len());

    // Claimed sum = Σ_i α^i · y_i · 2^{N − arity_i}.
    let mut claimed_sum = AlmostGoldilocksExt2::zero();
    for (i, (arity, _, y)) in instances_meta.iter().enumerate() {
        let weight_int = 1u64 << (max_num_vars - arity);
        let weight = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(weight_int));
        claimed_sum = ext2_add(claimed_sum, ext2_mul(alphas[i], ext2_mul(*y, weight)));
    }

    // Replay each round.
    if proof.sumcheck.round_messages.len() != max_num_vars { return None; }
    let mut current_sum = claimed_sum;
    let mut challenges: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(max_num_vars);
    for round in 0..max_num_vars {
        let msg = &proof.sumcheck.round_messages[round];
        if msg.len() != 3 { return None; }
        let s0 = msg[0];
        let s1 = msg[1];
        let sum = ext2_add(s0, s1);
        if !ext2_field_eq(sum, current_sum) { return None; }
        for m in msg { transcript.append_ext2(b"sp_round_msg", m); }
        let r = transcript.challenge_ext2(b"sp_round_challenge");
        challenges.push(r);
        current_sum = interpolate_degree2(msg, r);
    }

    if proof.f_evals.len() != instances_meta.len() { return None; }
    for e in &proof.f_evals { transcript.append_ext2(b"sp_f_eval", e); }

    // Final eval check: current_sum = Σ_i α^i · eq(r_i, R[..k_i]) · f_eval_i
    let mut expected = AlmostGoldilocksExt2::zero();
    for (i, (arity, r_i, _)) in instances_meta.iter().enumerate() {
        let eq_i = eq_eval_ext2(r_i, &challenges[..*arity]);
        expected = ext2_add(expected, ext2_mul(alphas[i], ext2_mul(eq_i, proof.f_evals[i])));
    }
    if !ext2_field_eq(current_sum, expected) { return None; }
    if !ext2_field_eq(proof.sumcheck.final_eval, expected) { return None; }

    Some(challenges)
}

// ============================================================================
// Internals
// ============================================================================

/// Witness representation. Round 0 keeps the original small-alphabet
/// packing so we can use selective-add inner loops; once any fold
/// happens the witness becomes general Ext2.
enum FState {
    /// Packed binary bitmask: bit `i` of word `i/64` is `f[i] ∈ {0, 1}`.
    /// Length covers `2^arity` bits.
    Binary(Vec<u64>),
    /// Single-chunk ternary: `f[i] = pos_bit - neg_bit ∈ {-1, 0, +1}`.
    /// `pos` and `neg` are both `2^(arity-6)` u64s (disjoint by construction).
    Ternary { pos: Vec<u64>, neg: Vec<u64> },
    /// General Ext2 table of length `2^(arity - rounds_done)`.
    Ext2(Vec<AlmostGoldilocksExt2>),
    /// Sparse same-point state for a witness with few nonzeros relative to
    /// `2^arity` (range-check / lookup auxes: one nonzero per input row,
    /// `nnz = 2^input_n ≪ 2^arity`). We never allocate the dense `2^arity`
    /// Ext2 eq table (≈1 GB/leaf at arity 26, ≈16 GB at 30 — the host-OOM /
    /// slow path). Only `f` is stored sparsely; `eq` is the FACTORED form
    /// We carry BOTH the eq value and f per live support position and fold
    /// them together each round (LSB-first), exactly like the dense path —
    /// so eq is O(1) per entry, not an O(arity) recompute (that was
    /// O(nnz·arity²); this is O(nnz·arity)). The degree-2 term
    /// `T(2)=(2e₁−e₀)(2f₁−f₀)` needs eq at a pair's MISSING partner (f=0);
    /// for the range aux's function-graph support most pairs are incomplete,
    /// so we reconstruct the partner's eq in O(1) from the present sibling
    /// via `e₁/e₀ = c/(1−c)` (`c = claim_pt[round]`) rather than recomputing
    /// — see [`sparse_round_msg`]. f=0 entries are dropped (eq there is only
    /// ever needed as a reconstructed partner, never stored). Byte-identical
    /// to the dense `FState::Ext2` path. `InstanceState.eq` is empty here.
    Sparse {
        /// `(idx, eq_val, f_val)` at the live support, sorted ascending by
        /// `idx` in the current `2^(arity-round)` domain.
        support: Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)>,
        /// Full original claim point (length `arity`); `claim_pt[round]` is
        /// the current round's coordinate `c` (for the partner-eq ratio).
        claim_pt: Vec<AlmostGoldilocksExt2>,
        /// `Π_{k<round} eq̃(claim_pt[k], r_k)` — only used by the degenerate
        /// fallback (c=1, prob 2^-128) to recompute a missing partner's eq
        /// from the factored form; the common path uses the O(1) ratio.
        prefix: AlmostGoldilocksExt2,
        /// Number of variables already bound.
        round: usize,
    },
}

/// Per-instance folding state.
struct InstanceState {
    arity: usize,
    weight: AlmostGoldilocksExt2,
    /// `2^N` broadcast factor folded into the per-round message — set
    /// once to `2^{N − arity}` so that the per-instance contribution
    /// matches the LHS scaling.
    broadcast: AlmostGoldilocksExt2,
    /// Folded eq table at the current round. Length halves every round
    /// while `rounds_done < arity`; once `rounds_done == arity` it's a
    /// singleton (the final `eq(r_i, R[..arity])`).
    eq: Vec<AlmostGoldilocksExt2>,
    /// Folded f table — same size convention. Tagged for round-0 fast
    /// paths; transitions to `Ext2` after the first fold.
    f_state: FState,
    /// Set when round number > arity (instance no longer live).
    constant: Option<AlmostGoldilocksExt2>, // = eq[0] · f[0]
}

impl InstanceState {
    fn new(inst: &FoldInstance, alpha_i: AlmostGoldilocksExt2, max_num_vars: usize) -> Self {
        let arity = inst.arity;
        let size = 1usize << arity;

        // Sparse fast path: a binary OR single-chunk ternary witness with
        // few nonzeros relative to `2^arity` is processed WITHOUT the dense
        // `2^arity` Ext2 eq table — that table is the host-memory wall at
        // high arity (≈1 GB/leaf at 26, ≈16 GB at 30). We carry only the
        // f-support and recompute eq from its factored form. Binary leaves
        // are the level-0 range/lookup auxes; single-chunk ternary leaves
        // are the level-1+ split chunks (high chunks are near-zero, so they
        // sparsify essentially for free). Round messages are byte-identical
        // to the dense path. Gated: arity ≥ MIN and density below THRESHOLD.
        // ZK4_SPARSE_SP=0 disables.
        {
            let sparse_on = std::env::var("ZK4_SPARSE_SP").ok().as_deref() != Some("0");
            let min_arity = std::env::var("ZK4_SPARSE_SP_MIN_ARITY").ok()
                .and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
            // Worth it when the dense eq table dwarfs the support. 8× headroom
            // keeps the bookkeeping a clear win; tune via ZK4_SPARSE_SP_RATIO.
            let ratio = std::env::var("ZK4_SPARSE_SP_RATIO").ok()
                .and_then(|s| s.parse::<usize>().ok()).unwrap_or(8);
            let support = if sparse_on && arity >= min_arity && arity >= 6 {
                let words = 1usize << (arity - 6);
                match &inst.data {
                    crate::fold::FoldData::Binary(packed) => {
                        let nnz: usize = packed[..words].iter()
                            .map(|w| w.count_ones() as usize).sum();
                        if nnz.saturating_mul(ratio) < size {
                            Some(build_sparse_support_binary(packed, &inst.claim_pt, arity, nnz))
                        } else { None }
                    }
                    crate::fold::FoldData::Ternary(c) if c.k_chunks == 1 => {
                        let nnz: usize = c.pos[..words].iter()
                            .map(|w| w.count_ones() as usize).sum::<usize>()
                            + c.neg[..words].iter()
                            .map(|w| w.count_ones() as usize).sum::<usize>();
                        if nnz.saturating_mul(ratio) < size {
                            Some(build_sparse_support_ternary(c, &inst.claim_pt, arity, nnz))
                        } else { None }
                    }
                    _ => None,
                }
            } else { None };
            if let Some(support) = support {
                let shift = max_num_vars - arity;
                let broadcast = AlmostGoldilocksExt2::from_base(
                    AlmostGoldilocksField(1u64 << shift));
                return Self {
                    arity,
                    weight: alpha_i,
                    broadcast,
                    eq: Vec::new(), // sparse: eq stored per-entry in the support
                    f_state: FState::Sparse {
                        support,
                        claim_pt: inst.claim_pt.clone(),
                        prefix: AlmostGoldilocksExt2::one(),
                        round: 0,
                    },
                    constant: None,
                };
            }
        }

        let eq = evaluate_lagrange_basis_ext2(&inst.claim_pt);
        assert_eq!(eq.len(), size, "eq table size {} != 2^arity {}", eq.len(), size);
        // For arity ≥ 6 the bit-packed paths exist; smaller arities
        // fall through to the dense Ext2 path so the indexing logic
        // stays uniform.
        let f_state = if arity >= 6 {
            match &inst.data {
                crate::fold::FoldData::Binary(packed) => {
                    // Trust caller to size packed correctly (2^(arity-6) words).
                    debug_assert!(packed.len() >= 1usize << (arity - 6));
                    FState::Binary(packed.clone())
                }
                crate::fold::FoldData::Ternary(chunks) if chunks.k_chunks == 1 => {
                    debug_assert_eq!(chunks.n_ring, 1usize << (arity - 6));
                    FState::Ternary { pos: chunks.pos.clone(), neg: chunks.neg.clone() }
                }
                _ => FState::Ext2(lift_witness_to_ext2(&inst.data, arity)),
            }
        } else {
            FState::Ext2(lift_witness_to_ext2(&inst.data, arity))
        };
        let shift = max_num_vars - arity;
        let broadcast_int = 1u64 << shift;
        let broadcast = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(broadcast_int));
        Self {
            arity,
            weight: alpha_i,
            broadcast,
            eq,
            f_state,
            constant: None,
        }
    }

    /// In-place fold of the (Ext2) eq table along bit 0 with challenge r.
    /// Serial — outer parallelism handles per-leaf concurrency.
    fn fold_eq_in_place(&mut self, r: AlmostGoldilocksExt2) {
        let half = self.eq.len() / 2;
        // Write to first half; reads from positions ≥ 2j ≥ j stay valid.
        for j in 0..half {
            let a = self.eq[2 * j];
            let b = self.eq[2 * j + 1];
            self.eq[j] = ext2_add(a, ext2_mul(r, ext2_sub(b, a)));
        }
        self.eq.truncate(half);
    }

    fn absorb(&mut self, r: AlmostGoldilocksExt2, round: usize) {
        if round < self.arity {
            // Sparse path: fold eq AND f together (O(1)/entry). Set
            // `constant = eq(R)·f(R)` at the last round from the lone entry.
            if matches!(self.f_state, FState::Sparse { .. }) {
                let taken = std::mem::replace(&mut self.f_state, FState::Ext2(Vec::new()));
                if let FState::Sparse { support, claim_pt, prefix, round: sr } = taken {
                    let folded = sparse_fold(&support, &claim_pt, sr, prefix, r);
                    // Advance the eq prefix (for the degenerate fallback only).
                    let c = claim_pt[sr];
                    let one = AlmostGoldilocksExt2::one();
                    let eq_factor = ext2_add(
                        ext2_mul(ext2_sub(one, c), ext2_sub(one, r)),
                        ext2_mul(c, r));
                    let new_prefix = ext2_mul(prefix, eq_factor);
                    if round + 1 == self.arity {
                        let cval = folded.first()
                            .map(|&(_, eq, f)| ext2_mul(eq, f))
                            .unwrap_or_else(AlmostGoldilocksExt2::zero);
                        self.constant = Some(cval);
                    }
                    self.f_state = FState::Sparse {
                        support: folded, claim_pt, prefix: new_prefix, round: sr + 1,
                    };
                }
                return;
            }
            self.fold_eq_in_place(r);
            // Fold f. Round-0 small-alphabet states transition to Ext2;
            // the Ext2 path folds in place.
            match &self.f_state {
                FState::Binary(_) | FState::Ternary { .. } => {
                    // Extract the small-alphabet state, do the round-0
                    // fold which produces a fresh Ext2 vec.
                    let mut tmp = FState::Ext2(Vec::new());
                    std::mem::swap(&mut tmp, &mut self.f_state);
                    let new_f = match tmp {
                        FState::Binary(packed) => fold_round0_binary_serial(&packed, r, self.arity),
                        FState::Ternary { pos, neg } => fold_round0_ternary_single_serial(&pos, &neg, r, self.arity),
                        _ => unreachable!(),
                    };
                    self.f_state = FState::Ext2(new_f);
                }
                FState::Sparse { .. } => unreachable!("sparse f_state handled by early return above"),
                FState::Ext2(_) => {
                    if let FState::Ext2(f) = &mut self.f_state {
                        let half = f.len() / 2;
                        for j in 0..half {
                            let a = f[2 * j];
                            let b = f[2 * j + 1];
                            f[j] = ext2_add(a, ext2_mul(r, ext2_sub(b, a)));
                        }
                        f.truncate(half);
                    }
                }
            }
            if round + 1 == self.arity {
                let f = match &self.f_state { FState::Ext2(f) => f, _ => unreachable!() };
                debug_assert_eq!(self.eq.len(), 1);
                debug_assert_eq!(f.len(), 1);
                self.constant = Some(ext2_mul(self.eq[0], f[0]));
            }
        }
    }

    /// `f_i(R[..arity])` after all rounds.
    fn f_final(&self) -> AlmostGoldilocksExt2 {
        match &self.f_state {
            FState::Ext2(f) => {
                assert_eq!(f.len(), 1, "f_final called before all rounds done");
                f[0]
            }
            FState::Sparse { support, .. } => {
                debug_assert!(support.len() <= 1, "f_final: sparse support not fully folded");
                support.first().map(|&(_, _, f)| f).unwrap_or_else(AlmostGoldilocksExt2::zero)
            }
            _ => panic!("f_final called before any fold (f still in small-alphabet form)"),
        }
    }
}

/// Fold a binary-packed witness by `r`, producing an Ext2 vec of length 2^(arity-1).
/// For each pair (b0, b1), new value = b0 + r·(b1 - b0) ∈ {0, r, 1-r, 1}.
/// Serial; outer parallelism handles per-leaf concurrency.
fn fold_round0_binary_serial(packed: &[u64], r: AlmostGoldilocksExt2, arity: usize) -> Vec<AlmostGoldilocksExt2> {
    let half = 1usize << (arity - 1);
    let zero = AlmostGoldilocksExt2::zero();
    let one = AlmostGoldilocksExt2::one();
    // Indexed by (b1 << 1) | b0 — exactly `(w >> 2k) & 0b11`.
    let v: [AlmostGoldilocksExt2; 4] = [zero, ext2_sub(one, r), r, one];
    let mut out = Vec::with_capacity(half);
    'outer: for &w in packed {
        for k in 0..32 {
            let bits = ((w >> (2 * k)) & 0b11) as usize;
            out.push(v[bits]);
            if out.len() >= half { break 'outer; }
        }
    }
    out
}

/// Fold a single-chunk ternary witness by `r`. Serial.
fn fold_round0_ternary_single_serial(
    pos: &[u64],
    neg: &[u64],
    r: AlmostGoldilocksExt2,
    arity: usize,
) -> Vec<AlmostGoldilocksExt2> {
    let half = 1usize << (arity - 1);
    let zero = AlmostGoldilocksExt2::zero();
    let one = AlmostGoldilocksExt2::one();
    let neg_one = ext2_sub(zero, one);

    let mut tbl = [zero; 16];
    let dirs: [(usize, usize, AlmostGoldilocksExt2); 3] =
        [(0, 0, zero), (1, 0, one), (0, 1, neg_one)];
    for &(p0, n0, v0) in &dirs {
        for &(p1, n1, v1) in &dirs {
            let idx = (p1 << 3) | (n1 << 2) | (p0 << 1) | n0;
            tbl[idx] = ext2_add(v0, ext2_mul(r, ext2_sub(v1, v0)));
        }
    }
    assert_eq!(pos.len(), neg.len());
    let mut out = Vec::with_capacity(half);
    'outer: for wi in 0..pos.len() {
        let pw = pos[wi];
        let nw = neg[wi];
        for k in 0..32 {
            let p0 = ((pw >> (2 * k)) & 1) as usize;
            let n0 = ((nw >> (2 * k)) & 1) as usize;
            let p1 = ((pw >> (2 * k + 1)) & 1) as usize;
            let n1 = ((nw >> (2 * k + 1)) & 1) as usize;
            let idx = (p1 << 3) | (n1 << 2) | (p0 << 1) | n0;
            out.push(tbl[idx]);
            if out.len() >= half { break 'outer; }
        }
    }
    out
}

/// `eq(claim_pt, idx) = Π_k ℓ(claim_pt[k], idx_k)` (ℓ(c,0)=1−c, ℓ(c,1)=c).
/// O(arity). Used once per nonzero at build time and by the degenerate
/// fallback in [`pair_eq`].
#[inline]
pub(crate) fn sparse_eq_full(idx: u64, claim_pt: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let one = AlmostGoldilocksExt2::one();
    let mut e = one;
    for (k, &c) in claim_pt.iter().enumerate() {
        e = ext2_mul(e, if (idx >> k) & 1 == 1 { c } else { ext2_sub(one, c) });
    }
    e
}

/// `prefix · Π_b ℓ(claim_pt[round+1+b], y_b)` — the current eq value at a
/// position whose current-domain index is `2y` would be this × (1−c); at
/// `2y+1`, this × c. Only the degenerate fallback calls it (O(arity)).
#[inline]
pub(crate) fn sparse_eq_suffix(
    y: u64, claim_pt: &[AlmostGoldilocksExt2], round: usize, prefix: AlmostGoldilocksExt2,
) -> AlmostGoldilocksExt2 {
    let one = AlmostGoldilocksExt2::one();
    let mut e = prefix;
    let nbits = claim_pt.len() - round - 1;
    for b in 0..nbits {
        let c = claim_pt[round + 1 + b];
        e = ext2_mul(e, if (y >> b) & 1 == 1 { c } else { ext2_sub(one, c) });
    }
    e
}

/// Build the sparse same-point support for a binary witness: `(idx, eq, 1)`
/// at every set bit, `eq = eq(claim_pt, idx)`. Sorted ascending by `idx`.
fn build_sparse_support_binary(
    packed: &[u64], claim_pt: &[AlmostGoldilocksExt2], arity: usize, nnz: usize,
) -> Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)> {
    let one = AlmostGoldilocksExt2::one();
    let words = 1usize << (arity - 6);
    let mut out = Vec::with_capacity(nnz);
    for wi in 0..words {
        let w = packed[wi];
        if w == 0 { continue; }
        let base = (wi as u64) * 64;
        let mut bits = w;
        while bits != 0 {
            let k = bits.trailing_zeros() as u64;
            let idx = base + k;
            out.push((idx, sparse_eq_full(idx, claim_pt), one));
            bits &= bits - 1;
        }
    }
    out
}

/// Single-chunk ternary support: `(idx, eq, +1)` at pos bits, `(idx, eq, −1)`
/// at neg bits (disjoint). Sorted ascending by `idx`.
fn build_sparse_support_ternary(
    c: &almost_goldilocks_cuda::ajtai::TernaryChunks,
    claim_pt: &[AlmostGoldilocksExt2], arity: usize, nnz: usize,
) -> Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)> {
    let one = AlmostGoldilocksExt2::one();
    let neg_one = ext2_sub(AlmostGoldilocksExt2::zero(), one);
    let words = 1usize << (arity - 6);
    let mut out = Vec::with_capacity(nnz);
    for wi in 0..words {
        let pw = c.pos[wi];
        let nw = c.neg[wi];
        let both = pw | nw;
        if both == 0 { continue; }
        let base = (wi as u64) * 64;
        let mut bits = both;
        while bits != 0 {
            let k = bits.trailing_zeros();
            let idx = base + k as u64;
            let eq = sparse_eq_full(idx, claim_pt);
            if (pw >> k) & 1 == 1 { out.push((idx, eq, one)); }
            else { out.push((idx, eq, neg_one)); }
            bits &= bits - 1;
        }
    }
    out
}

/// Recover the eq values `(e0, e1)` of a current-domain pair `(2y, 2y+1)`
/// from whichever sibling(s) are present. eq is folded alongside f, so a
/// present sibling's eq is read directly. A MISSING sibling's eq is needed
/// for the degree-2 `T(2)` term even though its f=0; recover it in O(1) via
/// `e1/e0 = c/(1−c)` (the precomputed `ratio_*`). The only case the ratio
/// can't cover is a zero denominator — c=1 for a missing odd (prob 2^-128)
/// or c=0 for a missing even (only off the padding axis, also 2^-128) — and
/// there we fall back to the factored recompute via `prefix` (O(arity)).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_eq(
    e0_opt: Option<AlmostGoldilocksExt2>, e1_opt: Option<AlmostGoldilocksExt2>,
    y: u64, claim_pt: &[AlmostGoldilocksExt2], round: usize, prefix: AlmostGoldilocksExt2,
    c: AlmostGoldilocksExt2, omc: AlmostGoldilocksExt2,
    ratio_e2o: Option<AlmostGoldilocksExt2>, ratio_o2e: Option<AlmostGoldilocksExt2>,
) -> (AlmostGoldilocksExt2, AlmostGoldilocksExt2) {
    let zero = AlmostGoldilocksExt2::zero();
    match (e0_opt, e1_opt) {
        (Some(e0), Some(e1)) => (e0, e1),
        (Some(e0), None) => {
            let e1 = match ratio_e2o {
                Some(r) => ext2_mul(e0, r),
                None => ext2_mul(sparse_eq_suffix(y, claim_pt, round, prefix), c),
            };
            (e0, e1)
        }
        (None, Some(e1)) => {
            let e0 = match ratio_o2e {
                Some(r) => ext2_mul(e1, r),
                None => ext2_mul(sparse_eq_suffix(y, claim_pt, round, prefix), omc),
            };
            (e0, e1)
        }
        (None, None) => (zero, zero),
    }
}

/// Per-round `(c, 1−c, ratio_e2o, ratio_o2e)`: `ratio_e2o = c/(1−c)` (eq even
/// → odd), `ratio_o2e = (1−c)/c` (odd → even). `None` when the denominator
/// is 0 (degenerate, handled by [`pair_eq`]'s fallback).
#[inline]
pub(crate) fn round_ratios(claim_pt: &[AlmostGoldilocksExt2], round: usize)
    -> (AlmostGoldilocksExt2, AlmostGoldilocksExt2,
        Option<AlmostGoldilocksExt2>, Option<AlmostGoldilocksExt2>)
{
    let zero = AlmostGoldilocksExt2::zero();
    let c = claim_pt[round];
    let omc = ext2_sub(AlmostGoldilocksExt2::one(), c);
    let ratio_e2o = if omc != zero { Some(ext2_mul(c, ext2_inv(omc))) } else { None };
    let ratio_o2e = if c != zero { Some(ext2_mul(omc, ext2_inv(c))) } else { None };
    (c, omc, ratio_e2o, ratio_o2e)
}

/// One sparse same-point round: degree-2 message `(T0,T1,T2)`, byte-identical
/// to the dense `FState::Ext2` branch. O(live support) — eq read from the
/// stored values, missing partners reconstructed O(1) via [`pair_eq`].
fn sparse_round_msg(
    support: &[(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)],
    claim_pt: &[AlmostGoldilocksExt2], round: usize, prefix: AlmostGoldilocksExt2,
) -> (AlmostGoldilocksExt2, AlmostGoldilocksExt2, AlmostGoldilocksExt2) {
    let zero = AlmostGoldilocksExt2::zero();
    let two = ext2_add(AlmostGoldilocksExt2::one(), AlmostGoldilocksExt2::one());
    let (c, omc, ratio_e2o, ratio_o2e) = round_ratios(claim_pt, round);
    let (mut s0, mut s1, mut s2) = (zero, zero, zero);
    let mut i = 0;
    while i < support.len() {
        let y = support[i].0 >> 1;
        let (mut e0o, mut f0, mut e1o, mut f1) = (None, zero, None, zero);
        while i < support.len() && (support[i].0 >> 1) == y {
            if support[i].0 & 1 == 0 { e0o = Some(support[i].1); f0 = support[i].2; }
            else { e1o = Some(support[i].1); f1 = support[i].2; }
            i += 1;
        }
        let (e0, e1) = pair_eq(e0o, e1o, y, claim_pt, round, prefix, c, omc, ratio_e2o, ratio_o2e);
        let e2 = ext2_sub(ext2_mul(two, e1), e0);
        let f2 = ext2_sub(ext2_mul(two, f1), f0);
        s0 = ext2_add(s0, ext2_mul(e0, f0));
        s1 = ext2_add(s1, ext2_mul(e1, f1));
        s2 = ext2_add(s2, ext2_mul(e2, f2));
    }
    (s0, s1, s2)
}

/// Fold eq AND f by `r` (LSB-first): merge each `(2y,2y+1)` pair into `y`
/// with `v ← v0 + r·(v1−v0)` for both, reconstructing a missing partner's eq
/// via [`pair_eq`]. Drops f=0 results (their eq is only ever needed as a
/// reconstructed partner of a present entry), keeping the support ≤ nnz.
fn sparse_fold(
    support: &[(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)],
    claim_pt: &[AlmostGoldilocksExt2], round: usize, prefix: AlmostGoldilocksExt2,
    r: AlmostGoldilocksExt2,
) -> Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)> {
    let zero = AlmostGoldilocksExt2::zero();
    let (c, omc, ratio_e2o, ratio_o2e) = round_ratios(claim_pt, round);
    let mut out = Vec::with_capacity(support.len());
    let mut i = 0;
    while i < support.len() {
        let y = support[i].0 >> 1;
        let (mut e0o, mut f0, mut e1o, mut f1) = (None, zero, None, zero);
        while i < support.len() && (support[i].0 >> 1) == y {
            if support[i].0 & 1 == 0 { e0o = Some(support[i].1); f0 = support[i].2; }
            else { e1o = Some(support[i].1); f1 = support[i].2; }
            i += 1;
        }
        let nf = ext2_add(f0, ext2_mul(r, ext2_sub(f1, f0)));
        if nf != zero {
            let (e0, e1) = pair_eq(e0o, e1o, y, claim_pt, round, prefix, c, omc, ratio_e2o, ratio_o2e);
            let neq = ext2_add(e0, ext2_mul(r, ext2_sub(e1, e0)));
            out.push((y, neq, nf));
        }
    }
    out
}

/// Round-0 message for a binary witness — no Ext2·Ext2 in the inner loop.
/// Serial; outer parallelism distributes leaves across cores.
fn round0_msg_binary_serial(
    packed: &[u64],
    eq: &[AlmostGoldilocksExt2],
) -> (AlmostGoldilocksExt2, AlmostGoldilocksExt2, AlmostGoldilocksExt2) {
    let zero = AlmostGoldilocksExt2::zero();
    let mut acc: [AlmostGoldilocksExt2; 4] = [zero; 4];
    for (wi, &w) in packed.iter().enumerate() {
        if w == 0 { continue; }
        let eq_base = wi * 64;
        for k in 0..32 {
            let bits = (w >> (2 * k)) & 0b11;
            if bits == 0 { continue; }
            let e0 = eq[eq_base + 2 * k];
            let e1 = eq[eq_base + 2 * k + 1];
            if bits & 0b01 != 0 {
                acc[0] = ext2_add(acc[0], e0);
                acc[1] = ext2_add(acc[1], e1);
            }
            if bits & 0b10 != 0 {
                acc[2] = ext2_add(acc[2], e0);
                acc[3] = ext2_add(acc[3], e1);
            }
        }
    }
    let [e0_b0, e1_b0, e0_b1, e1_b1] = acc;
    let two = ext2_add(AlmostGoldilocksExt2::one(), AlmostGoldilocksExt2::one());
    let four = ext2_add(two, two);
    // T(0) = Σ b0·e0;  T(1) = Σ b1·e1.
    let s0 = e0_b0;
    let s1 = e1_b1;
    // T(2) = Σ (2b1 - b0)(2e1 - e0) = 4·Σ b1·e1 - 2·Σ b1·e0 - 2·Σ b0·e1 + Σ b0·e0.
    let s2 = ext2_sub(
        ext2_add(ext2_mul(four, e1_b1), e0_b0),
        ext2_add(ext2_mul(two, e0_b1), ext2_mul(two, e1_b0)),
    );
    (s0, s1, s2)
}

/// Round-0 message for a single-chunk ternary witness. Serial.
fn round0_msg_ternary_single_serial(
    pos: &[u64],
    neg: &[u64],
    eq: &[AlmostGoldilocksExt2],
) -> (AlmostGoldilocksExt2, AlmostGoldilocksExt2, AlmostGoldilocksExt2) {
    let zero = AlmostGoldilocksExt2::zero();
    assert_eq!(pos.len(), neg.len());
    let mut acc: [AlmostGoldilocksExt2; 8] = [zero; 8];
    for wi in 0..pos.len() {
        let pw = pos[wi];
        let nw = neg[wi];
        if pw == 0 && nw == 0 { continue; }
        let eq_base = wi * 64;
        for k in 0..32 {
            let p_bits = (pw >> (2 * k)) & 0b11;
            let n_bits = (nw >> (2 * k)) & 0b11;
            if p_bits == 0 && n_bits == 0 { continue; }
            let e0 = eq[eq_base + 2 * k];
            let e1 = eq[eq_base + 2 * k + 1];
            if p_bits & 0b01 != 0 { acc[0] = ext2_add(acc[0], e0); acc[1] = ext2_add(acc[1], e1); }
            if n_bits & 0b01 != 0 { acc[2] = ext2_add(acc[2], e0); acc[3] = ext2_add(acc[3], e1); }
            if p_bits & 0b10 != 0 { acc[4] = ext2_add(acc[4], e0); acc[5] = ext2_add(acc[5], e1); }
            if n_bits & 0b10 != 0 { acc[6] = ext2_add(acc[6], e0); acc[7] = ext2_add(acc[7], e1); }
        }
    }
    let [e0_p0, e1_p0, e0_n0, e1_n0, e0_p1, e1_p1, e0_n1, e1_n1] = acc;
    let two = ext2_add(AlmostGoldilocksExt2::one(), AlmostGoldilocksExt2::one());
    let four = ext2_add(two, two);
    // A = Σ v0·e0;  B = Σ v0·e1;  C = Σ v1·e0;  D = Σ v1·e1
    // where v_i = p_i - n_i.
    let a = ext2_sub(e0_p0, e0_n0);
    let b = ext2_sub(e1_p0, e1_n0);
    let c = ext2_sub(e0_p1, e0_n1);
    let d = ext2_sub(e1_p1, e1_n1);
    // T(0) = A;  T(1) = D;  T(2) = 4D - 2C - 2B + A.
    let s0 = a;
    let s1 = d;
    let s2 = ext2_sub(
        ext2_add(ext2_mul(four, d), a),
        ext2_add(ext2_mul(two, c), ext2_mul(two, b)),
    );
    (s0, s1, s2)
}

/// Build the round message `[T(0), T(1), T(2)]` at the given (0-indexed)
/// round. Sum over each live instance + the constant-mode contributions.
/// Parallelizes across leaves; each leaf's contribution is computed
/// serially (outer parallelism scales with #leaves which is typically
/// 30–63 per bucket level — comparable to host core count).
fn compute_round_message(
    states: &[InstanceState],
    round: usize,
    max_num_vars: usize,
) -> Vec<AlmostGoldilocksExt2> {
    let zero = AlmostGoldilocksExt2::zero();
    let one = AlmostGoldilocksExt2::one();
    let two = ext2_add(one, one);

    let remaining_after_c = max_num_vars - round - 1;
    let pow_remaining = AlmostGoldilocksExt2::from_base(
        AlmostGoldilocksField(1u64 << remaining_after_c),
    );

    use rayon::prelude::*;
    let t_arr = states.par_iter().map(|s| {
        if let Some(constant) = s.constant {
            let weighted = ext2_mul(s.weight, ext2_mul(pow_remaining, constant));
            [weighted, weighted, weighted]
        } else {
            // Live instance: serial-inner work.
            let (s0, s1, s2) = match &s.f_state {
                FState::Binary(packed) => {
                    debug_assert_eq!(round, 0, "Binary state should only exist at round 0");
                    round0_msg_binary_serial(packed, &s.eq)
                }
                FState::Ternary { pos, neg } => {
                    debug_assert_eq!(round, 0, "Ternary state should only exist at round 0");
                    round0_msg_ternary_single_serial(pos, neg, &s.eq)
                }
                FState::Sparse { support, claim_pt, prefix, round: sr } =>
                    sparse_round_msg(support, claim_pt, *sr, *prefix),
                FState::Ext2(f) => {
                    let half = s.eq.len() / 2;
                    let mut s0 = zero; let mut s1 = zero; let mut s2 = zero;
                    for j in 0..half {
                        let e0 = s.eq[2 * j];
                        let e1 = s.eq[2 * j + 1];
                        let f0 = f[2 * j];
                        let f1 = f[2 * j + 1];
                        let e2 = ext2_sub(ext2_mul(two, e1), e0);
                        let f2 = ext2_sub(ext2_mul(two, f1), f0);
                        s0 = ext2_add(s0, ext2_mul(e0, f0));
                        s1 = ext2_add(s1, ext2_mul(e1, f1));
                        s2 = ext2_add(s2, ext2_mul(e2, f2));
                    }
                    (s0, s1, s2)
                }
            };
            let scale = ext2_mul(s.weight, s.broadcast);
            [ext2_mul(scale, s0), ext2_mul(scale, s1), ext2_mul(scale, s2)]
        }
    }).reduce(
        || [zero, zero, zero],
        |a, b| [ext2_add(a[0], b[0]), ext2_add(a[1], b[1]), ext2_add(a[2], b[2])],
    );
    t_arr.to_vec()
}

/// Lagrange-interpolate three evals at `x ∈ {0, 1, 2}` and eval at `r`.
fn interpolate_degree2(evals: &[AlmostGoldilocksExt2], r: AlmostGoldilocksExt2) -> AlmostGoldilocksExt2 {
    debug_assert_eq!(evals.len(), 3);
    let one = AlmostGoldilocksExt2::one();
    let two = ext2_add(one, one);
    // L_0(x) = (x-1)(x-2)/((0-1)(0-2)) = (x-1)(x-2)/2
    // L_1(x) = (x)(x-2)/((1-0)(1-2)) = x(x-2)/(-1)
    // L_2(x) = (x)(x-1)/((2-0)(2-1)) = x(x-1)/2
    let inv2 = ext2_inv(two);
    let neg_one = ext2_sub(AlmostGoldilocksExt2::zero(), one);
    let xm1 = ext2_sub(r, one);
    let xm2 = ext2_sub(r, two);
    let l0 = ext2_mul(ext2_mul(xm1, xm2), inv2);
    let l1 = ext2_mul(ext2_mul(r, xm2), ext2_inv(neg_one));
    let l2 = ext2_mul(ext2_mul(r, xm1), inv2);
    ext2_add(
        ext2_add(ext2_mul(l0, evals[0]), ext2_mul(l1, evals[1])),
        ext2_mul(l2, evals[2]),
    )
}

/// `eq(a, b)` for two Ext2 vectors of the same length.
fn eq_eval_ext2(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    assert_eq!(a.len(), b.len(), "eq_eval length mismatch");
    let one = AlmostGoldilocksExt2::one();
    let mut acc = one;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let term = ext2_add(
            ext2_mul(*ai, *bi),
            ext2_mul(ext2_sub(one, *ai), ext2_sub(one, *bi)),
        );
        acc = ext2_mul(acc, term);
    }
    acc
}

/// Lift a binary or ternary witness to a dense `Vec<Ext2>` of length
/// `2^arity`. For binary: each bit becomes 0 or 1. For ternary:
/// `Σ_i 2^i · (pos_i − neg_i)`.
fn lift_witness_to_ext2(data: &crate::fold::FoldData, arity: usize) -> Vec<AlmostGoldilocksExt2> {
    let total = 1usize << arity;
    let mut out = vec![AlmostGoldilocksExt2::zero(); total];
    match data {
        crate::fold::FoldData::Binary(packed) => {
            // Parallelize over chunks of 64 output values (= one u64 of bits).
            // For arity ≥ 20 this is the dominant prep cost.
            use rayon::prelude::*;
            const CHUNK: usize = 4096;
            if packed.len() >= CHUNK {
                let chunks: Vec<Vec<AlmostGoldilocksExt2>> = packed
                    .par_chunks(CHUNK)
                    .map(|word_chunk| {
                        let mut local = vec![AlmostGoldilocksExt2::zero(); word_chunk.len() * 64];
                        for (j, &word) in word_chunk.iter().enumerate() {
                            if word == 0 { continue; }
                            let base = j * 64;
                            for k in 0..64 {
                                if (word >> k) & 1 == 1 {
                                    local[base + k] = AlmostGoldilocksExt2::one();
                                }
                            }
                        }
                        local
                    })
                    .collect();
                let mut idx = 0;
                for chunk in chunks {
                    let take = (total - idx).min(chunk.len());
                    out[idx..idx + take].copy_from_slice(&chunk[..take]);
                    idx += take;
                    if idx >= total { break; }
                }
            } else {
                for j in 0..packed.len() {
                    let word = packed[j];
                    if word == 0 { continue; }
                    let base = j * 64;
                    for k in 0..64 {
                        let idx = base + k;
                        if idx >= total { break; }
                        if (word >> k) & 1 == 1 {
                            out[idx] = AlmostGoldilocksExt2::one();
                        }
                    }
                }
            }
        }
        crate::fold::FoldData::Ternary(chunks) => {
            let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
            let mut pow_two_i = AlmostGoldilocksExt2::one();
            for ki in 0..chunks.k_chunks {
                let pos = &chunks.pos[ki * chunks.n_ring..(ki + 1) * chunks.n_ring];
                let neg = &chunks.neg[ki * chunks.n_ring..(ki + 1) * chunks.n_ring];
                for j in 0..chunks.n_ring {
                    let pw = pos[j];
                    let nw = neg[j];
                    if pw == 0 && nw == 0 { continue; }
                    let base = j * 64;
                    for k in 0..64 {
                        let idx = base + k;
                        if idx >= total { break; }
                        let p = (pw >> k) & 1 == 1;
                        let m = (nw >> k) & 1 == 1;
                        if p { out[idx] = ext2_add(out[idx], pow_two_i); }
                        if m { out[idx] = ext2_sub(out[idx], pow_two_i); }
                    }
                }
                pow_two_i = ext2_mul(pow_two_i, two);
            }
        }
        crate::fold::FoldData::Digit { bit_planes, negate_top_bit, .. } => {
            // Digit lift: out[x] = Σ_k w_k · bit_planes[k][x], where w_k = 2^k
            // and w_{m-1} = -2^{m-1} when negate_top_bit (top digit's sign).
            // Built in i64 then lifted to Ext2 (positive → field, negative
            // → field_neg). Parallelized over the output positions.
            use rayon::prelude::*;
            let k_bits = bit_planes.len();
            out.par_iter_mut().enumerate().for_each(|(x, slot)| {
                let mut v: i64 = 0;
                for bk in 0..k_bits {
                    let packed = &bit_planes[bk];
                    let word_idx = x / 64;
                    let bit_pos = x % 64;
                    if word_idx < packed.len() && (packed[word_idx] >> bit_pos) & 1 == 1 {
                        if *negate_top_bit && bk == k_bits - 1 {
                            v -= 1i64 << bk;
                        } else {
                            v += 1i64 << bk;
                        }
                    }
                }
                *slot = if v >= 0 {
                    AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v as u64))
                } else {
                    AlmostGoldilocksExt2::from_base(-AlmostGoldilocksField((-v) as u64))
                };
            });
        }
    }
    out
}

/// GPU **batched** same-point sumcheck. All leaves of a per-arity
/// fold-tree bucket share `arity = max_num_vars` (so broadcast = 1).
/// All (eq, f) Ext2 tables reside in one device buffer; one kernel
/// launch per round handles ALL leaves. Falls back to the CPU path
/// transparently on GPU OOM — caller sees a unified return type.
pub fn prove_same_point_gpu_batched(
    instances: &[FoldInstance],
    arity: usize,
    transcript: &mut Transcript,
) -> (SamePointProof, Vec<AlmostGoldilocksExt2>) {
    // Snapshot the transcript so we can restore on GPU failure and
    // re-run on CPU without leaving the transcript in a partial state.
    let snapshot = transcript.clone();
    if let Some(result) = try_prove_same_point_gpu_batched(instances, arity, transcript) {
        return result;
    }
    // GPU OOM: restore transcript to pre-attempt state and run CPU path.
    *transcript = snapshot;
    prove_same_point(instances, arity, transcript)
}

/// Device-side witness input for [`prove_same_point_gpu_batched_dev`]:
/// the group's leaf data is already assembled on the CURRENT device as
/// concat buffer(s) with leaf `i` at element offset `i · 2^(arity−6)`.
pub enum SpDevInput<'a> {
    Binary(&'a almost_goldilocks_cuda::memory::DeviceBuffer<u64>),
    Ternary(
        &'a almost_goldilocks_cuda::memory::DeviceBuffer<u64>,
        &'a almost_goldilocks_cuda::memory::DeviceBuffer<u64>,
    ),
}

/// Device-input same-point prover for the device-resident fold tree: same
/// transcript absorbs, same round loop, same (field-identical) messages as
/// [`prove_same_point_gpu_batched`], but the witness data never crosses
/// PCIe — F_u builds straight from the assembled device concat and the
/// final f_evals are recovered from it too. Falls back to the CPU prover
/// (host data still lives in `instances`) on any GPU failure.
pub fn prove_same_point_gpu_batched_dev(
    instances: &[FoldInstance],
    dev: SpDevInput<'_>,
    arity: usize,
    transcript: &mut Transcript,
) -> (
    SamePointProof,
    Vec<AlmostGoldilocksExt2>,
    Option<almost_goldilocks_cuda::memory::DeviceBuffer<AlmostGoldilocksExt2>>,
) {
    let snapshot = transcript.clone();
    if let Some(result) = try_prove_same_point_gpu_batched_dev(instances, dev, arity, transcript) {
        return result;
    }
    *transcript = snapshot;
    let (proof, r) = prove_same_point(instances, arity, transcript);
    (proof, r, None)
}

fn try_prove_same_point_gpu_batched_dev(
    instances: &[FoldInstance],
    dev: SpDevInput<'_>,
    arity: usize,
    transcript: &mut Transcript,
) -> Option<(
    SamePointProof,
    Vec<AlmostGoldilocksExt2>,
    Option<almost_goldilocks_cuda::memory::DeviceBuffer<AlmostGoldilocksExt2>>,
)> {
    use almost_goldilocks_cuda::sumcheck_prover::GpuSharedEqState;
    assert!(!instances.is_empty(), "prove_same_point_gpu_batched_dev: empty input");
    for (i, inst) in instances.iter().enumerate() {
        assert_eq!(inst.arity, arity, "instance {} arity {} != {}", i, inst.arity, arity);
        assert_eq!(inst.claim_pt.len(), arity,
            "instance {} claim_pt len {} != arity {}", i, inst.claim_pt.len(), arity);
    }

    transcript.append_u64(b"sp_num_var", arity as u64);
    transcript.append_u64(b"sp_num_inst", instances.len() as u64);
    for inst in instances {
        transcript.append_u64(b"sp_arity_i", inst.arity as u64);
        for c in &inst.claim_pt { transcript.append_ext2(b"sp_r_i", c); }
        transcript.append_ext2(b"sp_y_i", &inst.claim_val);
    }

    let alpha = transcript.challenge_ext2(b"sp_alpha");
    let alphas = calc_pow_vec_ext2(alpha, instances.len());
    let claim_pts: Vec<Vec<AlmostGoldilocksExt2>> =
        instances.iter().map(|inst| inst.claim_pt.clone()).collect();

    let mut state = SpState::SharedEq(match dev {
        SpDevInput::Binary(d_packed) =>
            GpuSharedEqState::new_binary_packed_f_dev(&claim_pts, d_packed, &alphas).ok()?,
        SpDevInput::Ternary(d_pos, d_neg) =>
            GpuSharedEqState::new_ternary_packed_dev(&claim_pts, d_pos, d_neg, &alphas).ok()?,
    });

    let mut round_messages: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(arity);
    let mut challenges: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(arity);
    for _round in 0..arity {
        let combined = state.round_message_combined(&alphas).ok()?;
        for m in &combined { transcript.append_ext2(b"sp_round_msg", m); }
        let r = transcript.challenge_ext2(b"sp_round_challenge");
        round_messages.push(combined.to_vec());
        challenges.push(r);
        state.fold(r).ok()?;
    }

    // f_i(R) straight from the device concat (no host plane re-upload).
    // The eq table at the shared challenge point is built ONCE here and
    // returned to the caller — the fold-tree group's chunk evals use the
    // same point, so they reuse it instead of running another eq dp.
    let d_pt = almost_goldilocks_cuda::memory::DeviceBuffer::<AlmostGoldilocksExt2>
        ::from_slice(&challenges).ok()?;
    let (d_a, d_b, in_a) =
        almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all_device(&d_pt, arity).ok()?;
    let d_eq = if in_a { d_a } else { d_b };
    let f_evals: Vec<AlmostGoldilocksExt2> = match dev {
        SpDevInput::Binary(d_packed) => {
            almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_with_eq_dev(
                &d_eq, arity, &[(d_packed, instances.len())]).ok()?
        }
        SpDevInput::Ternary(d_pos, d_neg) => {
            let evals = almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_with_eq_dev(
                &d_eq, arity, &[(d_pos, instances.len()), (d_neg, instances.len())]).ok()?;
            let m = instances.len();
            (0..m).map(|i| ext2_sub(evals[i], evals[m + i])).collect()
        }
    };
    for e in &f_evals { transcript.append_ext2(b"sp_f_eval", e); }

    let mut final_eval = AlmostGoldilocksExt2::zero();
    for (i, inst) in instances.iter().enumerate() {
        let eq_i = eq_eval_ext2(&inst.claim_pt, &challenges[..inst.arity]);
        final_eval = ext2_add(final_eval, ext2_mul(alphas[i], ext2_mul(eq_i, f_evals[i])));
    }

    Some((
        SamePointProof {
            sumcheck: SumcheckProof { final_eval, round_messages },
            f_evals,
        },
        challenges,
        Some(d_eq),
    ))
}

/// Either same-point GPU backend. The shared-eq variant folds eq once per
/// unique claim_pt (not per leaf); both expose the same round-message / fold
/// / final-eval API so the sumcheck driver below is layout-agnostic.
enum SpState {
    Batched(almost_goldilocks_cuda::sumcheck_prover::GpuBatchedSamePointState),
    SharedEq(almost_goldilocks_cuda::sumcheck_prover::GpuSharedEqState),
}
impl SpState {
    /// Combined degree-2 round message `[T(0),T(1),T(2)]`. Batched returns
    /// per-leaf messages combined as `Σ_i α_i·T_i`; SharedEq returns
    /// per-unique messages combined as `Σ_u T_u` (α already folded into F_u).
    fn round_message_combined(
        &mut self,
        alphas: &[AlmostGoldilocksExt2],
    ) -> almost_goldilocks_cuda::error::Result<[AlmostGoldilocksExt2; 3]> {
        let zero = AlmostGoldilocksExt2::zero();
        let mut combined = [zero; 3];
        match self {
            SpState::Batched(s) => {
                let msg = s.compute_round_messages()?;
                for (i, alpha_i) in alphas.iter().enumerate() {
                    for c in 0..3 {
                        combined[c] = ext2_add(combined[c], ext2_mul(*alpha_i, msg[i * 3 + c]));
                    }
                }
            }
            SpState::SharedEq(s) => {
                let msg = s.compute_round_messages()?;
                let nu = s.num_unique();
                for u in 0..nu {
                    for c in 0..3 {
                        combined[c] = ext2_add(combined[c], msg[u * 3 + c]);
                    }
                }
            }
        }
        Ok(combined)
    }
    fn fold(&mut self, r: AlmostGoldilocksExt2) -> almost_goldilocks_cuda::error::Result<()> {
        match self {
            SpState::Batched(s) => s.fold(r),
            SpState::SharedEq(s) => s.fold(r),
        }
    }
    fn is_shared_eq(&self) -> bool { matches!(self, SpState::SharedEq(_)) }
}

/// Expand any Digit leaves in `instances` into K virtual binary leaves (one
/// per bit-plane), so the full shared-eq binary GPU path can be used even
/// for higher-radix groups. Binary leaves pass through as a single virtual.
///
/// For each virtual leaf the α weight is the parent α scaled by 2^k (the
/// digit plane's place value), or `−2^{K-1}` when the parent Digit's top
/// plane is the signed sign-bit (`negate_top_bit`). After the sumcheck the
/// per-virtual f_evals are recombined into per-original-leaf f_evals using
/// the same weights — preserving the SamePointProof.f_evals length.
///
/// Returns: (virtual_claim_pts, virtual_packed_planes, virtual_alphas,
/// leaf_to_virtuals[(start, len)]). `leaf_to_virtuals` is indexed by the
/// original leaf index and gives the half-open range in the virtual arrays.
fn expand_digit_to_virtual(
    instances: &[FoldInstance],
    alphas: &[AlmostGoldilocksExt2],
) -> (
    Vec<Vec<AlmostGoldilocksExt2>>,
    Vec<Vec<u64>>,
    Vec<AlmostGoldilocksExt2>,
    Vec<(usize, usize)>,
) {
    let mut claim_pts: Vec<Vec<AlmostGoldilocksExt2>> = Vec::new();
    let mut packed: Vec<Vec<u64>> = Vec::new();
    let mut virt_alphas: Vec<AlmostGoldilocksExt2> = Vec::new();
    let mut map: Vec<(usize, usize)> = Vec::with_capacity(instances.len());
    for (i, inst) in instances.iter().enumerate() {
        let start = packed.len();
        match &inst.data {
            crate::fold::FoldData::Binary(v) => {
                claim_pts.push(inst.claim_pt.clone());
                packed.push(v.clone());
                virt_alphas.push(alphas[i]);
                map.push((start, 1));
            }
            crate::fold::FoldData::Digit { bit_planes, negate_top_bit, .. } => {
                let k_planes = bit_planes.len();
                for k in 0..k_planes {
                    claim_pts.push(inst.claim_pt.clone());
                    packed.push(bit_planes[k].clone());
                    let is_signed_top = *negate_top_bit && k + 1 == k_planes;
                    let w = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1u64 << k));
                    let weighted = ext2_mul(alphas[i], w);
                    let signed = if is_signed_top {
                        ext2_sub(AlmostGoldilocksExt2::zero(), weighted)
                    } else {
                        weighted
                    };
                    virt_alphas.push(signed);
                }
                map.push((start, k_planes));
            }
            _ => unreachable!("expand_digit_to_virtual: only Binary + Digit supported"),
        }
    }
    (claim_pts, packed, virt_alphas, map)
}

/// Host fallback for [`eval_binary_planes_device`]: builds eq(R) on the host
/// and selective-adds the packed plane bits. Produces IDENTICAL field values
/// to the device kernel (the cuda crate's own test asserts device == this CPU
/// reference bit-for-bit), so the resulting leaf claims — and thus the
/// transcript — are unchanged. Used when the device eq-table allocation fails
/// (GPU OOM): a large dense same-point leaf (e.g. a wide vocab argmax range
/// check) then evaluates on the host instead of aborting the whole fold tree.
/// Slower, but the eq table lives in host RAM (far larger than VRAM), and
/// vocab sharding keeps these leaves small so this rarely fires.
fn eval_binary_planes_host(
    claim_pt: &[AlmostGoldilocksExt2],
    packed_planes: &[&[u64]],
) -> Vec<AlmostGoldilocksExt2> {
    let total = 1usize << claim_pt.len();
    // Pure-host eq(R) table (no device alloc — the GPU `ext2_eq_dp_all`
    // allocates VRAM and would re-OOM). `evaluate_lagrange_basis_ext2` is the
    // canonical CPU eq builder (same convention as the GPU sumcheck path,
    // asserted == the device DP in `host_eq_matches_device_dp`).
    let eq = crate::poly::evaluate_lagrange_basis_ext2(claim_pt);
    packed_planes
        .iter()
        .map(|plane| {
            let mut acc = AlmostGoldilocksExt2::zero();
            for (wi, &w) in plane.iter().enumerate() {
                if w == 0 {
                    continue;
                }
                let base = wi * 64;
                for k in 0..64 {
                    if (w >> k) & 1 == 1 {
                        let idx = base + k;
                        if idx < total {
                            acc = acc + eq[idx];
                        }
                    }
                }
            }
            acc
        })
        .collect()
}

/// Evaluate binary planes at `claim_pt` on the GPU, falling back to the host
/// on any device failure (chiefly the eq-table OOM for a large dense leaf).
fn eval_binary_planes(
    claim_pt: &[AlmostGoldilocksExt2],
    packed_planes: &[&[u64]],
) -> Vec<AlmostGoldilocksExt2> {
    match almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_device(claim_pt, packed_planes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[same_point] shared-eq plane eval GPU failed ({:?}) — host fallback", e);
            eval_binary_planes_host(claim_pt, packed_planes)
        }
    }
}

fn try_prove_same_point_gpu_batched(
    instances: &[FoldInstance],
    arity: usize,
    transcript: &mut Transcript,
) -> Option<(SamePointProof, Vec<AlmostGoldilocksExt2>)> {
    use almost_goldilocks_cuda::sumcheck_prover::{GpuBatchedSamePointState, GpuSharedEqState};
    assert!(!instances.is_empty(), "prove_same_point_gpu_batched: empty input");
    for (i, inst) in instances.iter().enumerate() {
        assert_eq!(inst.arity, arity,
            "GPU batched same-point requires uniform arity; instance {} arity {} != {}",
            i, inst.arity, arity);
        assert_eq!(inst.claim_pt.len(), arity,
            "instance {} claim_pt len {} != arity {}", i, inst.claim_pt.len(), arity);
    }

    transcript.append_u64(b"sp_num_var", arity as u64);
    transcript.append_u64(b"sp_num_inst", instances.len() as u64);
    for inst in instances {
        transcript.append_u64(b"sp_arity_i", inst.arity as u64);
        for c in &inst.claim_pt { transcript.append_ext2(b"sp_r_i", c); }
        transcript.append_ext2(b"sp_y_i", &inst.claim_val);
    }

    let alpha = transcript.challenge_ext2(b"sp_alpha");
    let alphas = calc_pow_vec_ext2(alpha, instances.len());

    // Higher-radix expansion: if any Digit leaves are present, expand each
    // Digit into K virtual binary leaves (one per bit-plane) so the entire
    // group can use the fast shared-eq binary path. Each virtual leaf
    // shares its parent Digit's claim_pt and gets α_virtual = α_leaf · 2^k
    // (negated for the top digit's sign bit). Binary leaves pass through
    // unchanged. After the sumcheck, the virtual f_evals are combined back
    // into per-original-leaf f_evals via the same 2^k weights — so the
    // SamePointProof.f_evals length matches the original leaf count.
    let any_digit = instances.iter().any(|i| matches!(i.data, crate::fold::FoldData::Digit { .. }));
    let (virt_claim_pts, virt_packed_owned, virt_alphas, leaf_to_virtuals) =
        if any_digit { expand_digit_to_virtual(instances, &alphas) } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

    // Eq tables built on device from each leaf's claim_pt; f tables
    // lifted on device. Three witness shapes:
    //   - All-binary: per-leaf packed `Vec<u64>` → binary lift kernel
    //   - All-ternary (single chunk): per-leaf (pos, neg) → ternary lift kernel
    //   - Mixed / multi-chunk ternary: host-lift fallback (rare; only
    //     hit before any split has happened on a non-binary input).
    // Upload shrinks from ~MBs of Ext2 to ~KBs of packed bits at large arities.
    let timing = std::env::var("ZK4_TIMING").is_ok();
    let timing_sp = std::env::var("ZK4_TIMING_SP").is_ok();
    let t0 = std::time::Instant::now();
    let all_binary = instances.iter().all(|inst| matches!(inst.data, crate::fold::FoldData::Binary(_)));
    let all_ternary_single = !all_binary && instances.iter().all(|inst| match &inst.data {
        crate::fold::FoldData::Ternary(c) => c.k_chunks == 1,
        _ => false,
    });
    let claim_pts: Vec<Vec<AlmostGoldilocksExt2>> =
        instances.iter().map(|inst| inst.claim_pt.clone()).collect();
    // Shared-eq backend (default on for arity ≥ threshold): folds eq once per
    // unique claim_pt instead of once per leaf. Big win for the 21 bit-planes
    // of an edge sharing one eq. Gated by ZK4_SHARED_EQ / _MIN_ARITY.
    let sharedeq_min = std::env::var("ZK4_SHARED_EQ_MIN_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
    let use_sharedeq = std::env::var("ZK4_SHARED_EQ").as_deref() != Ok("0")
        && arity >= sharedeq_min;
    let mut state = if any_digit {
        // Route through shared-eq binary path with virtual leaves (one per
        // bit-plane of each Digit, plus the originals for Binary leaves).
        // Reuses the FAST GpuSharedEqState binary kernel — no new CUDA.
        let packed_refs: Vec<&[u64]> = virt_packed_owned.iter().map(|v| v.as_slice()).collect();
        SpState::SharedEq(GpuSharedEqState::new_binary_packed_f(
            &virt_claim_pts, &packed_refs, &virt_alphas).ok()?)
    } else if all_binary && use_sharedeq {
        let packed_refs: Vec<&[u64]> = instances.iter().map(|inst| match &inst.data {
            crate::fold::FoldData::Binary(v) => v.as_slice(),
            _ => unreachable!(),
        }).collect();
        SpState::SharedEq(GpuSharedEqState::new_binary_packed_f(&claim_pts, &packed_refs, &alphas).ok()?)
    } else if all_binary {
        let packed_refs: Vec<&[u64]> = instances.iter().map(|inst| match &inst.data {
            crate::fold::FoldData::Binary(v) => v.as_slice(),
            _ => unreachable!(),
        }).collect();
        SpState::Batched(GpuBatchedSamePointState::new_device_eq_packed_f(&claim_pts, &packed_refs).ok()?)
    } else if all_ternary_single {
        let pos_refs: Vec<&[u64]> = instances.iter().map(|inst| match &inst.data {
            crate::fold::FoldData::Ternary(c) => c.pos.as_slice(),
            _ => unreachable!(),
        }).collect();
        let neg_refs: Vec<&[u64]> = instances.iter().map(|inst| match &inst.data {
            crate::fold::FoldData::Ternary(c) => c.neg.as_slice(),
            _ => unreachable!(),
        }).collect();
        if use_sharedeq {
            SpState::SharedEq(GpuSharedEqState::new_ternary_packed(&claim_pts, &pos_refs, &neg_refs, &alphas).ok()?)
        } else {
            SpState::Batched(GpuBatchedSamePointState::new_device_eq_packed_ternary(&claim_pts, &pos_refs, &neg_refs).ok()?)
        }
    } else {
        // Multi-chunk ternary fallback: host-lift, on-device eq.
        use rayon::prelude::*;
        let fs: Vec<Vec<AlmostGoldilocksExt2>> = instances
            .par_iter()
            .map(|inst| lift_witness_to_ext2(&inst.data, arity))
            .collect();
        SpState::Batched(GpuBatchedSamePointState::new_device_eq(&claim_pts, &fs).ok()?)
    };
    let t1 = std::time::Instant::now();
    if timing {
        let kind = if all_binary { "binary" } else if all_ternary_single { "ternary_single" } else { "mixed" };
        eprintln!("[gpu_batched arity={} leaves={}] kind={} setup={:?}",
            arity, instances.len(), kind, t1 - t0);
    }

    let t_setup_end = std::time::Instant::now();
    let mut round_messages: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(arity);
    let mut challenges: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(arity);
    let mut t_msg = std::time::Duration::ZERO;
    let mut t_fold = std::time::Duration::ZERO;
    for _round in 0..arity {
        let r0 = std::time::Instant::now();
        let combined = state
            .round_message_combined(&alphas)
            .expect("GPU batched round message");
        let r1 = std::time::Instant::now();
        t_msg += r1 - r0;
        for m in &combined { transcript.append_ext2(b"sp_round_msg", m); }
        let r = transcript.challenge_ext2(b"sp_round_challenge");
        round_messages.push(combined.to_vec());
        challenges.push(r);
        state.fold(r).expect("GPU batched fold");
        let r2 = std::time::Instant::now();
        t_fold += r2 - r1;
    }
    if timing || timing_sp {
        let setup_dt = t_setup_end - t0;
        eprintln!("[gpu_sp arity={} M={}] setup={:?} sumcheck_msg={:?} sumcheck_fold={:?}",
            arity, instances.len(), setup_dt, t_msg, t_fold);
    }

    // Per-leaf f_i(R). The interleaved state folded f per leaf and reads it
    // off; the shared-eq state never materialized per-leaf f (only F_u), so
    // recover f_i(R) as the MLE of the (binary/ternary) f_i at the
    // challenges via the on-device selective-add eval (returns scalars).
    let f_evals = if state.is_shared_eq() {
        if any_digit {
            // Eval all virtual binary planes at R, then combine back into
            // per-original-leaf f_evals. Binary leaves pass through; Digit
            // leaves recombine as Σ_k 2^k · v_k (negated at the signed top).
            let packed_refs: Vec<&[u64]> = virt_packed_owned.iter().map(|v| v.as_slice()).collect();
            let virt_evals = eval_binary_planes(&challenges, &packed_refs);
            let mut leaf_evals: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(instances.len());
            for (leaf_idx, inst) in instances.iter().enumerate() {
                let (start, len) = leaf_to_virtuals[leaf_idx];
                match &inst.data {
                    crate::fold::FoldData::Binary(_) => {
                        debug_assert_eq!(len, 1);
                        leaf_evals.push(virt_evals[start]);
                    }
                    crate::fold::FoldData::Digit { negate_top_bit, .. } => {
                        let mut acc = AlmostGoldilocksExt2::zero();
                        for k in 0..len {
                            let is_signed_top = *negate_top_bit && k + 1 == len;
                            let w = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1u64 << k));
                            let term = ext2_mul(virt_evals[start + k], w);
                            acc = if is_signed_top { ext2_sub(acc, term) } else { ext2_add(acc, term) };
                        }
                        leaf_evals.push(acc);
                    }
                    _ => unreachable!("digit expansion supports only Binary + Digit leaves"),
                }
            }
            leaf_evals
        } else if all_binary {
            let packed_refs: Vec<&[u64]> = instances.iter().map(|inst| match &inst.data {
                crate::fold::FoldData::Binary(v) => v.as_slice(),
                _ => unreachable!("binary shared-eq"),
            }).collect();
            eval_binary_planes(&challenges, &packed_refs)
        } else {
            // Ternary: f_i(R) = eval(pos_i) - eval(neg_i). Eval all pos/neg
            // planes in one call ([pos_0, neg_0, pos_1, neg_1, ...]).
            let mut planes: Vec<&[u64]> = Vec::with_capacity(instances.len() * 2);
            for inst in instances {
                match &inst.data {
                    crate::fold::FoldData::Ternary(c) => {
                        planes.push(c.pos.as_slice());
                        planes.push(c.neg.as_slice());
                    }
                    _ => unreachable!("ternary shared-eq"),
                }
            }
            let evals = eval_binary_planes(&challenges, &planes);
            (0..instances.len())
                .map(|i| ext2_sub(evals[2 * i], evals[2 * i + 1]))
                .collect()
        }
    } else {
        match &state {
            SpState::Batched(s) => s.final_f_evals().expect("final_f_evals"),
            SpState::SharedEq(_) => unreachable!(),
        }
    };
    for e in &f_evals { transcript.append_ext2(b"sp_f_eval", e); }

    let mut final_eval = AlmostGoldilocksExt2::zero();
    for (i, inst) in instances.iter().enumerate() {
        let eq_i = eq_eval_ext2(&inst.claim_pt, &challenges[..inst.arity]);
        final_eval = ext2_add(final_eval, ext2_mul(alphas[i], ext2_mul(eq_i, f_evals[i])));
    }

    Some((
        SamePointProof {
            sumcheck: SumcheckProof { final_eval, round_messages },
            f_evals,
        },
        challenges,
    ))
}

/// GPU same-point sumcheck (legacy per-leaf path — kept for reference).
/// Skipped if `arity < GPU_ARITY_THRESHOLD` (small arities are CPU-faster).
pub fn prove_same_point_gpu(
    instances: &[FoldInstance],
    arity: usize,
    transcript: &mut Transcript,
) -> (SamePointProof, Vec<AlmostGoldilocksExt2>) {
    use almost_goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;

    assert!(!instances.is_empty(), "prove_same_point_gpu: empty input");
    for (i, inst) in instances.iter().enumerate() {
        assert_eq!(inst.arity, arity,
            "GPU same-point requires uniform arity; instance {} arity {} != {}",
            i, inst.arity, arity);
        assert_eq!(inst.claim_pt.len(), arity,
            "instance {} claim_pt len {} != arity {}", i, inst.claim_pt.len(), arity);
    }

    transcript.append_u64(b"sp_num_var", arity as u64);
    transcript.append_u64(b"sp_num_inst", instances.len() as u64);
    for inst in instances {
        transcript.append_u64(b"sp_arity_i", inst.arity as u64);
        for c in &inst.claim_pt { transcript.append_ext2(b"sp_r_i", c); }
        transcript.append_ext2(b"sp_y_i", &inst.claim_val);
    }

    let alpha = transcript.challenge_ext2(b"sp_alpha");
    let alphas = calc_pow_vec_ext2(alpha, instances.len());

    // Build per-leaf eq + lifted-f tables in parallel (CPU). GPU state
    // construction (which uploads to device) stays sequential to avoid
    // multi-thread contention on CUDA's default stream.
    use rayon::prelude::*;
    let prepped: Vec<(Vec<AlmostGoldilocksExt2>, Vec<AlmostGoldilocksExt2>)> = instances
        .par_iter()
        .map(|inst| {
            let eq = evaluate_lagrange_basis_ext2(&inst.claim_pt);
            let f = lift_witness_to_ext2(&inst.data, arity);
            (eq, f)
        })
        .collect();
    let mut states: Vec<GpuSumcheckStateExt2> = prepped.iter().map(|(eq, f)| {
        let refs: Vec<&[AlmostGoldilocksExt2]> = vec![eq, f];
        GpuSumcheckStateExt2::new(&refs)
            .expect("GpuSumcheckStateExt2::new failed (likely GPU OOM)")
    }).collect();
    drop(prepped); // Free host buffers; data is now on GPU.

    let mut round_messages: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(arity);
    let mut challenges: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(arity);
    for _round in 0..arity {
        // Per-leaf round message (degree-2 → 3 evals per leaf).
        let mut combined = vec![AlmostGoldilocksExt2::zero(); 3];
        for (i, st) in states.iter_mut().enumerate() {
            let r = st.compute_round_message().expect("GPU compute_round_message");
            assert_eq!(r.len(), 3, "expected degree-2 round (3 evals), got {}", r.len());
            for c in 0..3 {
                combined[c] = ext2_add(combined[c], ext2_mul(alphas[i], r[c]));
            }
        }
        for m in &combined { transcript.append_ext2(b"sp_round_msg", m); }
        let r = transcript.challenge_ext2(b"sp_round_challenge");
        round_messages.push(combined);
        challenges.push(r);
        for st in states.iter_mut() {
            st.fold(r).expect("GPU fold");
        }
    }

    // Final f_i(R) extraction: each state's final_evaluations() returns [eq_i(R), f_i(R)].
    let mut f_evals: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(instances.len());
    let mut eq_evals: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(instances.len());
    for st in &states {
        let finals = st.final_evaluations().expect("GPU final_evaluations");
        eq_evals.push(finals[0]);
        f_evals.push(finals[1]);
    }
    for e in &f_evals { transcript.append_ext2(b"sp_f_eval", e); }

    // final_eval = Σ_i α^i · eq_i(R) · f_i(R)
    let mut final_eval = AlmostGoldilocksExt2::zero();
    for i in 0..instances.len() {
        final_eval = ext2_add(final_eval,
            ext2_mul(alphas[i], ext2_mul(eq_evals[i], f_evals[i])));
    }

    (
        SamePointProof {
            sumcheck: SumcheckProof { final_eval, round_messages },
            f_evals,
        },
        challenges,
    )
}

/// Arity threshold above which the GPU path is preferred. Below this,
/// GPU launch overhead exceeds CPU compute. Empirical sweet spot for
/// AGL on A100: ~14 (2^14 = 16K Ext2 ops per round becomes meaningfully
/// HBM-bound).
pub const GPU_ARITY_THRESHOLD: usize = 14;

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::ajtai::RingCommitment;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    /// The pure-host eq builder used by `eval_binary_planes_host` must match
    /// the device DP (`ext2_eq_dp_all`) bit-for-bit, else the same-point host
    /// fallback would emit wrong leaf claims. Small arity → tiny GPU alloc, so
    /// this runs even on a contended/near-full GPU.
    #[test]
    fn host_eq_matches_device_dp() {
        if almost_goldilocks_cuda::init().is_err() {
            eprintln!("skipping host_eq_matches_device_dp: no CUDA");
            return;
        }
        for n in [1usize, 4, 10] {
            let r: Vec<AlmostGoldilocksExt2> = (0..n as u64)
                .map(|i| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(i * 7 + 11)))
                .collect();
            let host = crate::poly::evaluate_lagrange_basis_ext2(&r);
            let dev = almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all(&r)
                .expect("device eq dp");
            assert_eq!(host.len(), dev.len(), "len mismatch n={}", n);
            for i in 0..host.len() {
                assert_eq!(host[i].c0.0, dev[i].c0.0, "c0 mismatch n={} idx={}", n, i);
                assert_eq!(host[i].c1.0, dev[i].c1.0, "c1 mismatch n={} idx={}", n, i);
            }
        }
    }

    fn make_binary_instance(
        packed: Vec<u64>,
        arity: usize,
        claim_pt: Vec<AlmostGoldilocksExt2>,
    ) -> FoldInstance {
        let data = crate::fold::FoldData::Binary(packed);
        let claim_val = data.evaluate_at_ext2(&claim_pt);
        FoldInstance {
            commitment: RingCommitment::zero(),
            data,
            arity,
            claim_pt,
            claim_val,
        }
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    /// Single instance with arity = N: degenerate same-point sumcheck
    /// reduces to a single point R = R, the proof's f_eval should match
    /// the input y, and verification accepts.
    #[test]
    fn single_instance_arity_equals_max_roundtrip() {
        let n = 4;
        let pack = vec![0b1010_1100u64 ^ (0b1111_0000u64 << 8)]; // 16 bits
        let pt: Vec<_> = (0..n).map(|i| lift(i as u64 * 7 + 3)).collect();
        let inst = make_binary_instance(pack, n, pt.clone());
        let y = inst.claim_val;

        let mut t = Transcript::new(b"sp-single");
        let (proof, r) = prove_same_point(&[inst.clone()], n, &mut t);
        assert_eq!(r.len(), n);
        assert!(ext2_field_eq(proof.f_evals[0], inst.data.evaluate_at_ext2(&r[..inst.arity])),
                "f_eval should be the witness MLE at R[..arity]");

        let meta = vec![(inst.arity, pt.clone(), y)];
        let mut t_v = Transcript::new(b"sp-single");
        let r2 = verify_same_point(&meta, n, &proof, &mut t_v).expect("verify");
        assert_eq!(r, r2);
    }

    /// Two instances with different arities (k_0 = 2, k_1 = 4, N = 4).
    /// Verifier should accept and the f_evals should match witness MLEs
    /// at R[..k_i].
    #[test]
    fn two_instances_heterogeneous_arity() {
        let n = 4;
        // Instance 0: arity 2 → 4 bits packed into 1 u64.
        let pack0 = vec![0b1011u64]; // bits 0,1,3 = 1; bit 2 = 0
        let pt0: Vec<_> = (0..2).map(|i| lift(i as u64 * 5 + 1)).collect();
        // Instance 1: arity 4 → 16 bits packed into 1 u64.
        let pack1 = vec![0b0101_0011_1100_1010u64];
        let pt1: Vec<_> = (0..4).map(|i| lift(i as u64 * 11 + 2)).collect();

        let inst0 = make_binary_instance(pack0, 2, pt0.clone());
        let inst1 = make_binary_instance(pack1, 4, pt1.clone());
        let y0 = inst0.claim_val;
        let y1 = inst1.claim_val;

        let mut t = Transcript::new(b"sp-two");
        let (proof, r) = prove_same_point(&[inst0.clone(), inst1.clone()], n, &mut t);
        assert!(ext2_field_eq(proof.f_evals[0], inst0.data.evaluate_at_ext2(&r[..inst0.arity])));
        assert!(ext2_field_eq(proof.f_evals[1], inst1.data.evaluate_at_ext2(&r[..inst1.arity])));

        let meta = vec![(2, pt0, y0), (4, pt1, y1)];
        let mut t_v = Transcript::new(b"sp-two");
        let r2 = verify_same_point(&meta, n, &proof, &mut t_v).expect("verify");
        assert_eq!(r, r2);
    }

    /// Three instances exercising the constant-mode path: smallest
    /// arity terminates first, contributes constants for the remaining
    /// rounds.
    #[test]
    fn three_instances_constant_mode_after_arity() {
        let n = 5;
        let pack0 = vec![0b1010u64];                          // arity 2
        let pack1 = vec![0b1111_0011u64];                     // arity 3
        let pack2 = vec![0b0101_0101u64 | (0b1010u64 << 16)]; // arity 5: 32 bits = full u64 chunk
        // arity-5 packing needs 32 bits — same u64 works because 32 ≤ 64.

        let pt0: Vec<_> = (0..2).map(|i| lift(i as u64 + 1)).collect();
        let pt1: Vec<_> = (0..3).map(|i| lift(i as u64 * 2 + 5)).collect();
        let pt2: Vec<_> = (0..5).map(|i| lift(i as u64 * 3 + 9)).collect();

        let inst0 = make_binary_instance(pack0, 2, pt0.clone());
        let inst1 = make_binary_instance(pack1, 3, pt1.clone());
        let inst2 = make_binary_instance(pack2, 5, pt2.clone());
        let (y0, y1, y2) = (inst0.claim_val, inst1.claim_val, inst2.claim_val);

        let mut t = Transcript::new(b"sp-three");
        let (proof, r) = prove_same_point(
            &[inst0.clone(), inst1.clone(), inst2.clone()], n, &mut t,
        );

        let meta = vec![(2, pt0, y0), (3, pt1, y1), (5, pt2, y2)];
        let mut t_v = Transcript::new(b"sp-three");
        let r2 = verify_same_point(&meta, n, &proof, &mut t_v).expect("verify");
        assert_eq!(r, r2);
        for (i, e) in proof.f_evals.iter().enumerate() {
            let inst = [&inst0, &inst1, &inst2][i];
            assert!(ext2_field_eq(*e, inst.data.evaluate_at_ext2(&r[..inst.arity])),
                    "f_eval[{}] mismatch", i);
        }
    }

    /// GPU same-point sumcheck output must be bit-exactly equal to
    /// the CPU version for uniform-arity input (the per-arity-bucket
    /// invariant).
    #[test]
    fn gpu_same_point_matches_cpu_uniform_arity() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let n = 4;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
        let pt: Vec<_> = (0..n).map(|i| lift(i as u64 * 7 + 3)).collect();
        let group: Vec<FoldInstance> = (0..3).map(|_| {
            let pack = vec![rng.gen::<u64>() & ((1u64 << (1 << n)) - 1)];
            make_binary_instance(pack, n, pt.clone())
        }).collect();

        let mut t_cpu = Transcript::new(b"sp-gpu-vs-cpu");
        let (cpu_proof, cpu_r) = prove_same_point(&group, n, &mut t_cpu);
        let mut t_gpu = Transcript::new(b"sp-gpu-vs-cpu");
        let (gpu_proof, gpu_r) = prove_same_point_gpu(&group, n, &mut t_gpu);
        assert_eq!(cpu_r, gpu_r, "challenge sequence diverged");
        assert_eq!(cpu_proof.sumcheck.round_messages, gpu_proof.sumcheck.round_messages,
                   "round messages diverged");
        assert_eq!(cpu_proof.f_evals, gpu_proof.f_evals, "f_evals diverged");
        assert!(crate::util::arith::ext2_field_eq(cpu_proof.sumcheck.final_eval,
                                                  gpu_proof.sumcheck.final_eval));
    }

    /// The shared-eq GPU batched prover (factored-eq backend) must produce
    /// the same round messages, challenges, and f_evals as the CPU prover.
    /// Uses multiple unique claim points (3 leaves on each of 2 points) so
    /// the per-unique prefix-scalar logic is exercised, and enough arity
    /// that several rounds run. ZK4_SHARED_EQ_MIN_ARITY=1 forces the
    /// shared-eq path at this small test arity (tests run single-threaded).
    #[test]
    fn gpu_batched_sharedeq_factored_matches_cpu() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        std::env::set_var("ZK4_SHARED_EQ_MIN_ARITY", "1");
        std::env::set_var("ZK4_GPU_SP_MIN_ARITY", "1");
        let n = 8; // 256 elements → 4 packed u64s per leaf
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xFAC707ED);
        let pt_a: Vec<_> = (0..n).map(|i| lift(i as u64 * 7 + 3)).collect();
        let pt_b: Vec<_> = (0..n).map(|i| lift(i as u64 * 13 + 11)).collect();
        let group: Vec<FoldInstance> = (0..6).map(|i| {
            let pack: Vec<u64> = (0..4).map(|_| rng.gen::<u64>()).collect();
            let pt = if i < 3 { pt_a.clone() } else { pt_b.clone() };
            make_binary_instance(pack, n, pt)
        }).collect();

        let mut t_cpu = Transcript::new(b"sp-sharedeq-vs-cpu");
        let (cpu_proof, cpu_r) = prove_same_point(&group, n, &mut t_cpu);
        let mut t_gpu = Transcript::new(b"sp-sharedeq-vs-cpu");
        let (gpu_proof, gpu_r) = prove_same_point_gpu_batched(&group, n, &mut t_gpu);
        std::env::remove_var("ZK4_SHARED_EQ_MIN_ARITY");
        std::env::remove_var("ZK4_GPU_SP_MIN_ARITY");

        assert_eq!(cpu_r.len(), gpu_r.len());
        for (i, (a, b)) in cpu_r.iter().zip(gpu_r.iter()).enumerate() {
            assert!(ext2_field_eq(*a, *b), "challenge {} diverged", i);
        }
        assert_eq!(cpu_proof.sumcheck.round_messages.len(),
                   gpu_proof.sumcheck.round_messages.len());
        for (round, (mc, mg)) in cpu_proof.sumcheck.round_messages.iter()
            .zip(gpu_proof.sumcheck.round_messages.iter()).enumerate()
        {
            assert_eq!(mc.len(), mg.len());
            for (c, (a, b)) in mc.iter().zip(mg.iter()).enumerate() {
                assert!(ext2_field_eq(*a, *b),
                        "round {} message T({}) diverged", round, c);
            }
        }
        for (i, (a, b)) in cpu_proof.f_evals.iter().zip(gpu_proof.f_evals.iter()).enumerate() {
            assert!(ext2_field_eq(*a, *b), "f_eval {} diverged", i);
        }
        // And the GPU proof must satisfy the verifier.
        let meta: Vec<_> = group.iter()
            .map(|inst| (inst.arity, inst.claim_pt.clone(), inst.claim_val))
            .collect();
        let mut t_v = Transcript::new(b"sp-sharedeq-vs-cpu");
        verify_same_point(&meta, n, &gpu_proof, &mut t_v)
            .expect("shared-eq factored proof must verify");
    }

    /// Tamper with the proof's f_eval — verifier rejects (because
    /// final_eval check uses the α-randomized combination).
    #[test]
    fn verifier_rejects_tampered_f_eval() {
        let n = 3;
        let pack = vec![0b101u64];
        let pt: Vec<_> = (0..n).map(|i| lift(i as u64 + 1)).collect();
        let inst = make_binary_instance(pack, n, pt.clone());
        let y = inst.claim_val;

        let mut t = Transcript::new(b"sp-tamper");
        let (mut proof, _r) = prove_same_point(&[inst.clone()], n, &mut t);
        proof.f_evals[0] = ext2_add(proof.f_evals[0], AlmostGoldilocksExt2::one());

        let meta = vec![(inst.arity, pt, y)];
        let mut t_v = Transcript::new(b"sp-tamper");
        assert!(verify_same_point(&meta, n, &proof, &mut t_v).is_none(),
                "tampered f_eval should be rejected");
    }

    /// Byte-identical check with ZERO-PADDED claim points — exactly what
    /// `extend_claim_point_to` produces for real auxes (high coords = 0).
    /// Exercises the `c = claim_pt[round] = 0` ratio path (ratio_e2o = 0).
    /// Run serial: `... sparse_same_point_zero_padded -- --test-threads=1`.
    #[test]
    fn sparse_same_point_zero_padded_claim() {
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        let n = 12;
        let pad = 3; // last `pad` coords are zero (padding)
        let total = 1usize << n;
        let words = total / 64;
        let mut rng = StdRng::seed_from_u64(0x70AD);
        let mut mk = |rng: &mut StdRng| {
            // Nonzeros only in the low (n-pad) bits, mirroring real auxes whose
            // positions are zero in the padded high bits.
            let live = 1usize << (n - pad);
            let mut packed = vec![0u64; words];
            for _ in 0..25 {
                let b = rng.gen_range(0..live);
                packed[b / 64] |= 1u64 << (b % 64);
            }
            let mut pt: Vec<_> = (0..n - pad)
                .map(|_| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(rng.gen::<u64>())))
                .collect();
            pt.extend(std::iter::repeat(AlmostGoldilocksExt2::zero()).take(pad));
            make_binary_instance(packed, n, pt)
        };
        let insts: Vec<_> = (0..4).map(|_| mk(&mut rng)).collect();

        std::env::set_var("ZK4_SPARSE_SP", "0");
        std::env::remove_var("ZK4_SPARSE_SP_MIN_ARITY");
        let mut td = Transcript::new(b"sp-pad");
        let (pd, _rd) = prove_same_point(&insts, n, &mut td);
        std::env::remove_var("ZK4_SPARSE_SP");
        std::env::set_var("ZK4_SPARSE_SP_MIN_ARITY", "0");
        let mut ts = Transcript::new(b"sp-pad");
        let (ps, _rs) = prove_same_point(&insts, n, &mut ts);
        std::env::remove_var("ZK4_SPARSE_SP_MIN_ARITY");

        for (round, (md, ms)) in pd.sumcheck.round_messages.iter()
            .zip(&ps.sumcheck.round_messages).enumerate()
        {
            for (k, (a, b)) in md.iter().zip(ms).enumerate() {
                assert!(ext2_field_eq(*a, *b), "zero-pad round {} coeff {} differs", round, k);
            }
        }
        assert!(ext2_field_eq(pd.sumcheck.final_eval, ps.sumcheck.final_eval), "final_eval differs");
    }

    /// The sparse same-point path must produce a BYTE-IDENTICAL proof to the
    /// dense path on the same sparse binary instances (round messages,
    /// challenges, f_evals, final_eval). Run filtered/serial — toggles env:
    ///   cargo test --release --lib sparse_same_point_matches_dense -- --test-threads=1
    #[test]
    fn sparse_same_point_matches_dense() {
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        let n = 12; // small for test speed; threshold lowered below
        let total = 1usize << n;
        let words = total / 64;
        let mut rng = StdRng::seed_from_u64(0xABCD);
        let mut mk = |rng: &mut StdRng| {
            let mut packed = vec![0u64; words];
            for _ in 0..37 { // sparse: 37 set bits ≪ 4096
                let b = rng.gen_range(0..total);
                packed[b / 64] |= 1u64 << (b % 64);
            }
            let pt: Vec<_> = (0..n)
                .map(|_| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(rng.gen::<u64>())))
                .collect();
            make_binary_instance(packed, n, pt)
        };
        let insts: Vec<_> = (0..6).map(|_| mk(&mut rng)).collect();

        std::env::set_var("ZK4_SPARSE_SP", "0");
        std::env::remove_var("ZK4_SPARSE_SP_MIN_ARITY");
        let mut td = Transcript::new(b"sp-equiv");
        let (pd, rd) = prove_same_point(&insts, n, &mut td);

        std::env::remove_var("ZK4_SPARSE_SP");
        std::env::set_var("ZK4_SPARSE_SP_MIN_ARITY", "0");
        let mut ts = Transcript::new(b"sp-equiv");
        let (ps, rs) = prove_same_point(&insts, n, &mut ts);
        std::env::remove_var("ZK4_SPARSE_SP_MIN_ARITY");

        // Direct check: factored eq (e0 = suffix·(1−c0)) vs dense eq table.
        {
            let cp = &insts[0].claim_pt;
            let eqtab = evaluate_lagrange_basis_ext2(cp);
            let c0 = cp[0];
            let omc0 = ext2_sub(AlmostGoldilocksExt2::one(), c0);
            let mut bad = 0;
            for y in 0..(1u64 << (n - 1)) {
                let suffix = sparse_eq_suffix(y, cp, 0, AlmostGoldilocksExt2::one());
                let e0 = ext2_mul(suffix, omc0);
                let e1 = ext2_mul(suffix, c0);
                if !ext2_field_eq(e0, eqtab[(2 * y) as usize]) { bad += 1; }
                if !ext2_field_eq(e1, eqtab[(2 * y + 1) as usize]) { bad += 1; }
            }
            assert_eq!(bad, 0, "factored eq must match the dense Lagrange table");
        }
        assert_eq!(rd.len(), rs.len(), "challenge count");
        for (ri, (a, b)) in rd.iter().zip(&rs).enumerate() {
            assert!(ext2_field_eq(*a, *b), "challenge {} differs (round msgs diverged earlier)", ri);
        }
        assert_eq!(pd.sumcheck.round_messages.len(), ps.sumcheck.round_messages.len());
        for ( round, (md, ms)) in pd.sumcheck.round_messages.iter()
            .zip(&ps.sumcheck.round_messages).enumerate()
        {
            for (k, (a, b)) in md.iter().zip(ms).enumerate() {
                assert!(ext2_field_eq(*a, *b),
                    "round {} message coeff {} differs: dense vs sparse", round, k);
            }
        }
        for (i, (a, b)) in pd.f_evals.iter().zip(&ps.f_evals).enumerate() {
            assert!(ext2_field_eq(*a, *b), "f_eval {} differs", i);
        }
        assert!(ext2_field_eq(pd.sumcheck.final_eval, ps.sumcheck.final_eval), "final_eval differs");
    }

    /// Same byte-identical guarantee for single-chunk TERNARY witnesses (the
    /// level-1+ split chunks, f ∈ {−1,0,+1}). Run serial:
    ///   cargo test --release --lib sparse_same_point_ternary -- --test-threads=1
    #[test]
    fn sparse_same_point_ternary_matches_dense() {
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        use almost_goldilocks_cuda::ajtai::TernaryChunks;
        let n = 12;
        let total = 1usize << n;
        let words = total / 64;
        let mut rng = StdRng::seed_from_u64(0x5151);
        let mut mk = |rng: &mut StdRng| {
            let mut pos = vec![0u64; words];
            let mut neg = vec![0u64; words];
            for _ in 0..40 {
                let b = rng.gen_range(0..total);
                // keep pos/neg disjoint: only set if neither already set there
                let (wi, bit) = (b / 64, b % 64);
                if (pos[wi] >> bit) & 1 == 1 || (neg[wi] >> bit) & 1 == 1 { continue; }
                if rng.gen::<bool>() { pos[wi] |= 1u64 << bit; } else { neg[wi] |= 1u64 << bit; }
            }
            let chunks = TernaryChunks { n_ring: words, k_chunks: 1, pos, neg };
            let data = crate::fold::FoldData::Ternary(chunks);
            let claim_pt: Vec<_> = (0..n)
                .map(|_| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(rng.gen::<u64>())))
                .collect();
            let claim_val = data.evaluate_at_ext2(&claim_pt);
            FoldInstance { commitment: RingCommitment::zero(), data, arity: n, claim_pt, claim_val }
        };
        let insts: Vec<_> = (0..6).map(|_| mk(&mut rng)).collect();

        std::env::set_var("ZK4_SPARSE_SP", "0");
        std::env::remove_var("ZK4_SPARSE_SP_MIN_ARITY");
        let mut td = Transcript::new(b"sp-tern");
        let (pd, rd) = prove_same_point(&insts, n, &mut td);

        std::env::remove_var("ZK4_SPARSE_SP");
        std::env::set_var("ZK4_SPARSE_SP_MIN_ARITY", "0");
        std::env::set_var("ZK4_SPARSE_SP_RATIO", "1"); // force sparse even if dense-ish
        let mut ts = Transcript::new(b"sp-tern");
        let (ps, rs) = prove_same_point(&insts, n, &mut ts);
        std::env::remove_var("ZK4_SPARSE_SP_MIN_ARITY");
        std::env::remove_var("ZK4_SPARSE_SP_RATIO");

        for (ri, (a, b)) in rd.iter().zip(&rs).enumerate() {
            assert!(ext2_field_eq(*a, *b), "challenge {} differs", ri);
        }
        for (round, (md, ms)) in pd.sumcheck.round_messages.iter()
            .zip(&ps.sumcheck.round_messages).enumerate()
        {
            for (k, (a, b)) in md.iter().zip(ms).enumerate() {
                assert!(ext2_field_eq(*a, *b), "round {} coeff {} differs (ternary)", round, k);
            }
        }
        for (i, (a, b)) in pd.f_evals.iter().zip(&ps.f_evals).enumerate() {
            assert!(ext2_field_eq(*a, *b), "f_eval {} differs (ternary)", i);
        }
        assert!(ext2_field_eq(pd.sumcheck.final_eval, ps.sumcheck.final_eval), "final_eval differs");
    }
}
