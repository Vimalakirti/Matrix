//! The witness-recovering link `Π_link`.
//!
//! One tagged sumcheck that simultaneously (a) certifies every source
//! evaluation claim and (b) proves every committed coefficient is in range,
//! terminating at a single verifier-derived point `ξ` where all commitments now
//! have a claim. That common point is what the masked RLC needs: linearity can
//! only merge claims that already share an evaluation point.
//!
//! ## The identity
//!
//! ```text
//!   N(X) = eq(X, α) · Σ_e γ^tag(e) · R_Bx( w̃_e(X) )
//!   E(X) = Σ_j γ^tag(j) · eq(X, r_j) · w̃_{e_j}(X)
//!
//!   Σ_{a ∈ {0,1}^A} ( N(a) + η·E(a) )  =  η · Σ_j γ^tag(j) · y_j
//! ```
//!
//! `N` sums to zero exactly when every coefficient is in range: `R_Bx` vanishes
//! on `{-(Bx-1) … Bx-1}`, so an in-range witness makes the bracketed table
//! identically zero and its multilinear extension zero at the random `α`. `E`
//! sums to the tagged combination of claimed values by the MLE reproducing
//! identity. Disjoint tag ranges keep the two halves from cancelling each other.
//!
//! At `B_x = 2` the range polynomial is `R(u) = (u+1)·u·(u-1) = u³ - u`, so the
//! round polynomial has degree `1 + 3 = 4` — versus degree 2 for a plain
//! same-point sumcheck. That factor is the price of proving the norm bound
//! rather than exhibiting it, which is what the recursive fold tree did by
//! shipping its final witness in the clear.
//!
//! ## Why this shards by commitment
//!
//! Both halves are *sums over commitments*, so each round message decomposes as
//! `T(k) = Σ_e T_e(k)`. A device holding a disjoint subset of commitments
//! computes its own partial message, contributes 5 field elements to an
//! all-reduce, and then folds its own tables with the shared challenge. Witness
//! data never moves between devices. [`round_message_partial`] is that unit, and
//! [`shard_ranges`] partitions commitments across devices; a test pins that the
//! proof is byte-identical at any shard count.
//!
//! Packing makes the per-query work block-local: a claim on a packed leaf sits
//! at a point whose prefix is Boolean, and `eq` factorizes across that prefix,
//! so `eq(X, r_j)` is supported on the leaf's own sub-cube. The per-commitment
//! query weights are therefore accumulated once, in time proportional to the
//! total witness size rather than `t ×` the ambient dimension.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2 as Ext2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use serde::{Deserialize, Serialize};

use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, ext2_sub};

/// Evaluations of the norm half shipped per round.
///
/// Gruen's zerocheck factorization is what makes this 4 rather than 5. The norm
/// half is `eq(X,α)·g(X)`, and `eq` splits as
/// `eq(r_<i, α_<i) · eq(t, α_i) · eq(x_>i, α_>i)`. Pulling the first two factors
/// out leaves `S_i(t) = Σ_{x_>i} eq(x_>i, α_>i)·g(r_<i, t, x_>i)`, which has the
/// degree of `g` alone (3 at `B_x = 2`) instead of `1 + deg g`. The verifier
/// multiplies the degree-1 `eq(t, α_i)` back in itself.
pub const NORM_EVALS: usize = 4;
/// Evaluations of the evaluation half per round: `ω·w` is degree 2.
pub const EVAL_EVALS: usize = 3;

/// Combined round polynomial degree, for the verifier's reconstruction.
pub const ROUND_DEGREE: usize = 4;

fn ext2_zero() -> Ext2 {
    Ext2::new(AlmostGoldilocksField(0), AlmostGoldilocksField(0))
}
fn ext2_one() -> Ext2 {
    Ext2::new(AlmostGoldilocksField(1), AlmostGoldilocksField(0))
}
fn ext2_from_u64(v: u64) -> Ext2 {
    Ext2::new(AlmostGoldilocksField(v), AlmostGoldilocksField(0))
}
#[inline]
fn is_zero(v: Ext2) -> bool {
    v.c0.0 == 0 && v.c1.0 == 0
}

/// `R_{B_x=2}(u) = u³ − u`. Vanishes exactly on `{-1, 0, 1}`, which is also why
/// zero padding and the ternary hiding block pass the range check for free.
#[inline]
fn range_poly(u: Ext2) -> Ext2 {
    let u2 = ext2_mul(u, u);
    let u3 = ext2_mul(u2, u);
    ext2_sub(u3, u)
}

/// One source evaluation claim, already lifted to the packed ambient domain:
/// `point` is the full `ambient_arity`-length point, i.e. the leaf's block
/// prefix followed by its own claim point.
#[derive(Clone, Debug)]
pub struct LinkQuery {
    /// Index of the packed commitment this claim refers to.
    pub commitment: usize,
    /// Evaluation point over the packed domain, most significant variable first.
    pub point: Vec<Ext2>,
    /// Claimed value.
    pub value: Ext2,
    /// Length of the Boolean block prefix at the front of `point`.
    ///
    /// A packed leaf's claim is `(block prefix, leaf point)`, and `eq`
    /// factorizes across the prefix: since the prefix is Boolean, the factor is
    /// 1 on exactly one block and 0 everywhere else. So the query's weight is
    /// supported on `2^(ambient − prefix_len)` slots, and building it costs
    /// that rather than the full ambient dimension.
    ///
    /// This is passed explicitly rather than inferred, because a random point
    /// coordinate can be 0 or 1 by accident and guessing would silently
    /// mis-place the block. `0` means a full-domain query.
    pub prefix_len: usize,
}

impl LinkQuery {
    /// A claim over the whole packed domain (no block prefix).
    pub fn full(commitment: usize, point: Vec<Ext2>, value: Ext2) -> Self {
        Self { commitment, point, value, prefix_len: 0 }
    }

    /// Decode the block this query targets: `(offset, block_arity)` in
    /// coefficients. Panics if the declared prefix is not Boolean, since that
    /// would place the block wrongly rather than fail loudly.
    fn block(&self, ambient_arity: usize) -> (usize, usize) {
        let block_arity = ambient_arity - self.prefix_len;
        let mut index = 0usize;
        for p in &self.point[..self.prefix_len] {
            let bit = if p.c1.0 == 0 && p.c0.0 == 0 {
                0
            } else if p.c1.0 == 0 && p.c0.0 == 1 {
                1
            } else {
                panic!("block prefix coordinate is not Boolean");
            };
            index = (index << 1) | bit;
        }
        (index << block_arity, block_arity)
    }
}

/// A packed witness: `2^ambient_arity` canonical field coefficients, message
/// blocks followed by the hiding block.
#[derive(Clone, Debug)]
pub struct LinkWitness {
    pub coeffs: Vec<u64>,
    /// Bit-packed form of the same witness, when it is binary.
    ///
    /// The GPU path uploads this instead of `coeffs` and expands on device: one
    /// bit per coefficient rather than a 16-byte Ext2, which is the difference
    /// between tens of megabytes and gigabytes of PCIe traffic.
    pub bits: Option<Vec<u64>>,
}

impl LinkWitness {
    pub fn dense(coeffs: Vec<u64>) -> Self {
        Self { coeffs, bits: None }
    }
    /// Binary witness carried in both forms: `coeffs` for the CPU reference and
    /// support analysis, `bits` for the device upload.
    pub fn binary(coeffs: Vec<u64>, bits: Vec<u64>) -> Self {
        Self { coeffs, bits: Some(bits) }
    }
}

/// One round's message: the two halves are sent separately because they have
/// different degrees, so the norm half costs 4 evaluations and the evaluation
/// half 3, rather than 5 apiece for a combined polynomial.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoundMsg {
    /// `S_i(0..=3)` — the norm half with the `eq` prefix factored out.
    pub norm: Vec<Ext2>,
    /// `E_i(0..=2)` — the evaluation half, before the `η` weight.
    pub eval: Vec<Ext2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkProof {
    pub rounds: Vec<RoundMsg>,
    /// `a_e = w̃_e(ξ)` — one terminal value per commitment.
    ///
    /// These are what the masked RLC consumes. Under
    /// [`crate::commit::hiding::HidingMode::InEvaluation`] the hiding block is
    /// part of the domain, so each `a_e` is blinded by that commitment's own
    /// randomness rather than being a bare evaluation of the message.
    pub terminal: Vec<Ext2>,
}

/// Equality table `eq(·, point)` over `2^len` Boolean indices, **most
/// significant variable first**: `point[0]` corresponds to the high bit of the
/// index.
///
/// That convention is load-bearing rather than cosmetic. A packed leaf's claim
/// point is `(block prefix, leaf point)` with the prefix in the high bits, and
/// the sumcheck below binds the high index bit first — so round `k` binds
/// `point[k]` and `ξ` comes out in the same order as the query points. Building
/// the table LSB-first (the natural loop) would silently reverse `ξ` relative to
/// the query points and every honest proof would fail its final check.
fn eq_table(point: &[Ext2]) -> Vec<Ext2> {
    let mut table = vec![ext2_one(); 1usize << point.len()];
    let mut span = 1usize;
    // Iterate low-to-high index bit, which means consuming `point` in reverse.
    for &r in point.iter().rev() {
        let one_minus = ext2_sub(ext2_one(), r);
        for i in (0..span).rev() {
            let v = table[i];
            table[i] = ext2_mul(v, one_minus);
            table[span + i] = ext2_mul(v, r);
        }
        span <<= 1;
    }
    table
}

/// `eq(x, point)` at a single point pair.
fn eq_at(a: &[Ext2], b: &[Ext2]) -> Ext2 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = ext2_one();
    for (x, y) in a.iter().zip(b.iter()) {
        acc = ext2_mul(acc, eq_scalar(*x, *y));
    }
    acc
}

/// `x·y + (1−x)(1−y) = 2xy − x − y + 1`.
#[inline]
fn eq_scalar(x: Ext2, y: Ext2) -> Ext2 {
    let xy = ext2_mul(x, y);
    ext2_add(ext2_sub(ext2_add(xy, xy), ext2_add(x, y)), ext2_one())
}

/// Powers `γ^1 … γ^n`.
fn tag_powers(gamma: Ext2, n: usize) -> Vec<Ext2> {
    let mut out = Vec::with_capacity(n);
    let mut cur = gamma;
    for _ in 0..n {
        out.push(cur);
        cur = ext2_mul(cur, gamma);
    }
    out
}

/// Split `n` commitments into `shards` contiguous ranges. This is the multi-GPU
/// partition; the round message is a sum over commitments, so any partition
/// yields the same total.
pub fn shard_ranges(n: usize, shards: usize) -> Vec<std::ops::Range<usize>> {
    let shards = shards.max(1).min(n.max(1));
    let base = n / shards;
    let rem = n % shards;
    let mut out = Vec::with_capacity(shards);
    let mut start = 0;
    for s in 0..shards {
        let len = base + usize::from(s < rem);
        out.push(start..start + len);
        start += len;
    }
    out
}

/// Mutable per-commitment sumcheck state.
pub struct Tables {
    /// Folded witness table for this commitment.
    w: Vec<Ext2>,
    /// Folded query-weight table `ω_e(X) = Σ_{j: e_j = e} γ^tag(j)·eq(X, r_j)`.
    omega: Vec<Ext2>,
}

/// Partial round message over one shard of commitments.
///
/// `T(k) = Σ_{e ∈ shard} …`, so shards are summed by the caller — an all-reduce
/// of 7 field elements on a real multi-GPU run.
///
/// Three things keep the inner loop cheap:
///
/// * **Incremental interpolation.** `w(k) = w_lo + k·w_d` at consecutive integer
///   `k` is `w(k-1) + w_d`, so the evaluation points cost additions rather than
///   a multiply each.
/// * **Scalar hoisting.** `γ_e` is constant over the whole table and `η` over
///   the whole round, so both leave the inner loop entirely.
/// * **Support sensitivity.** Where a witness entry and its partner are both
///   zero, `R(0) = 0` and `ω·0 = 0`, so the entry contributes nothing to either
///   half and is skipped. Sparse lookup advice is mostly zero in the early
///   rounds, which are also the largest, and density only doubles per round.
pub fn round_message_partial(
    tables: &[Tables],
    shard: std::ops::Range<usize>,
    eq_suffix: &[Ext2],
    commit_tags: &[Ext2],
    half: usize,
    first_round: bool,
) -> RoundMsg {
    let mut norm_acc = vec![ext2_zero(); NORM_EVALS];
    let mut eval_acc = vec![ext2_zero(); EVAL_EVALS];

    for e in shard {
        let t = &tables[e];
        let mut norm_e = [ext2_zero(); NORM_EVALS];
        let mut eval_e = [ext2_zero(); EVAL_EVALS];

        for idx in 0..half {
            let w_lo = t.w[idx];
            let w_hi = t.w[half + idx];
            if is_zero(w_lo) && is_zero(w_hi) {
                continue;
            }
            let w_d = ext2_sub(w_hi, w_lo);
            let o_lo = t.omega[idx];
            let o_d = ext2_sub(t.omega[half + idx], o_lo);
            let eqs = eq_suffix[idx];

            // Interpolants at k = 0,1,2,3 by repeated addition.
            let w0 = w_lo;
            let w1 = ext2_add(w0, w_d);
            let w2 = ext2_add(w1, w_d);
            let w3 = ext2_add(w2, w_d);

            // In the first round the witness still holds committed coefficients,
            // which are in {-1,0,1}, so R vanishes at k = 0 and k = 1 and those
            // two evaluations are free. A prover holding an out-of-range witness
            // would produce a wrong message here and be rejected — which is the
            // outcome the range check exists to force anyway.
            if !first_round {
                norm_e[0] = ext2_add(norm_e[0], ext2_mul(eqs, range_poly(w0)));
                norm_e[1] = ext2_add(norm_e[1], ext2_mul(eqs, range_poly(w1)));
            }
            norm_e[2] = ext2_add(norm_e[2], ext2_mul(eqs, range_poly(w2)));
            norm_e[3] = ext2_add(norm_e[3], ext2_mul(eqs, range_poly(w3)));

            let o0 = o_lo;
            let o1 = ext2_add(o0, o_d);
            let o2 = ext2_add(o1, o_d);
            eval_e[0] = ext2_add(eval_e[0], ext2_mul(o0, w0));
            eval_e[1] = ext2_add(eval_e[1], ext2_mul(o1, w1));
            eval_e[2] = ext2_add(eval_e[2], ext2_mul(o2, w2));
        }

        let tag = commit_tags[e];
        for k in 0..NORM_EVALS {
            norm_acc[k] = ext2_add(norm_acc[k], ext2_mul(tag, norm_e[k]));
        }
        for k in 0..EVAL_EVALS {
            eval_acc[k] = ext2_add(eval_acc[k], eval_e[k]);
        }
    }
    RoundMsg { norm: norm_acc, eval: eval_acc }
}

/// Fold a table in place with the round challenge.
fn fold_in_place(table: &mut Vec<Ext2>, r: Ext2, half: usize) {
    for i in 0..half {
        let lo = table[i];
        let hi = table[half + i];
        table[i] = ext2_add(lo, ext2_mul(r, ext2_sub(hi, lo)));
    }
    table.truncate(half);
}

/// Reconstruct the combined round polynomial `T_i(t)` from a round message.
///
/// `T_i(t) = c_i · eq(t, α_i) · S_i(t) + η · E_i(t)`, where `c_i` is the running
/// `eq` prefix. Both parties use this, so the factorization can never drift
/// between them.
fn combined_at(msg: &RoundMsg, c: Ext2, alpha_i: Ext2, eta: Ext2, t: Ext2) -> Ext2 {
    let s = lagrange_eval(&msg.norm, t);
    let e = lagrange_eval(&msg.eval, t);
    let norm = ext2_mul(ext2_mul(c, eq_scalar(t, alpha_i)), s);
    ext2_add(norm, ext2_mul(eta, e))
}

/// Run the link prover.
///
/// `shards` selects how many commitment partitions the round message is summed
/// over. It is a scheduling choice only: the transcript, and therefore the
/// proof, is identical for any value.
pub fn prove_link(
    witnesses: &[LinkWitness],
    queries: &[LinkQuery],
    ambient_arity: usize,
    shards: usize,
    transcript: &mut Transcript,
) -> (LinkProof, Vec<Ext2>) {
    let n_commit = witnesses.len();
    let size = 1usize << ambient_arity;
    assert!(n_commit > 0, "link needs at least one commitment");
    for w in witnesses {
        assert_eq!(w.coeffs.len(), size, "witness must fill the ambient domain");
    }

    let (commit_tags, query_tags, alpha, _eta) =
        derive_challenges(transcript, n_commit, queries, ambient_arity);

    let mut tables: Vec<Tables> = witnesses
        .iter()
        .map(|w| Tables {
            w: w.coeffs.iter().map(|c| ext2_from_u64(*c)).collect(),
            omega: vec![ext2_zero(); size],
        })
        .collect();
    for (j, q) in queries.iter().enumerate() {
        // Block-local: build eq only over the leaf's own sub-cube and scatter it
        // at the block offset. Total cost is the packed witness size, not
        // (queries x ambient dimension).
        let (offset, _) = q.block(ambient_arity);
        let eqt = eq_table(&q.point[q.prefix_len..]);
        let tag = query_tags[j];
        let om = &mut tables[q.commitment].omega;
        for (i, v) in eqt.iter().enumerate() {
            om[offset + i] = ext2_add(om[offset + i], ext2_mul(tag, *v));
        }
    }

    let ranges = shard_ranges(n_commit, shards);
    let mut rounds = Vec::with_capacity(ambient_arity);
    let mut xi = Vec::with_capacity(ambient_arity);
    let mut half = size >> 1;

    for i in 0..ambient_arity {
        // eq(x_>i, α_>i): the suffix factor. Built fresh each round at the
        // current size, which costs the same O(half) the old full-table fold did.
        let eq_suffix = eq_table(&alpha[i + 1..]);

        let mut msg = RoundMsg {
            norm: vec![ext2_zero(); NORM_EVALS],
            eval: vec![ext2_zero(); EVAL_EVALS],
        };
        for range in &ranges {
            let part = round_message_partial(
                &tables, range.clone(), &eq_suffix, &commit_tags, half, i == 0,
            );
            for k in 0..NORM_EVALS {
                msg.norm[k] = ext2_add(msg.norm[k], part.norm[k]);
            }
            for k in 0..EVAL_EVALS {
                msg.eval[k] = ext2_add(msg.eval[k], part.eval[k]);
            }
        }
        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        xi.push(r);

        for t in tables.iter_mut() {
            fold_in_place(&mut t.w, r, half);
            fold_in_place(&mut t.omega, r, half);
        }

        rounds.push(msg);
        half >>= 1;
    }

    let terminal: Vec<Ext2> = tables.iter().map(|t| t.w[0]).collect();
    for a in &terminal {
        transcript.append_ext2(b"link-terminal", a);
    }

    (LinkProof { rounds, terminal }, xi)
}

/// Verify a link proof. Returns the terminal claims `(ξ, a_e)` on success —
/// exactly the same-point inputs the masked RLC consumes.
pub fn verify_link(
    n_commit: usize,
    queries: &[LinkQuery],
    ambient_arity: usize,
    proof: &LinkProof,
    transcript: &mut Transcript,
) -> Option<(Vec<Ext2>, Vec<Ext2>)> {
    if proof.rounds.len() != ambient_arity || proof.terminal.len() != n_commit {
        return None;
    }
    let (commit_tags, query_tags, alpha, eta) =
        derive_challenges(transcript, n_commit, queries, ambient_arity);

    // Initial claim: the norm half sums to zero on an in-range witness, so only
    // the tagged combination of claimed values survives.
    let mut claim = ext2_zero();
    for (j, q) in queries.iter().enumerate() {
        claim = ext2_add(claim, ext2_mul(query_tags[j], q.value));
    }
    claim = ext2_mul(eta, claim);

    let mut xi = Vec::with_capacity(ambient_arity);
    let mut c = ext2_one();
    for (i, msg) in proof.rounds.iter().enumerate() {
        if msg.norm.len() != NORM_EVALS || msg.eval.len() != EVAL_EVALS {
            return None;
        }
        let t0 = combined_at(msg, c, alpha[i], eta, ext2_zero());
        let t1 = combined_at(msg, c, alpha[i], eta, ext2_one());
        if !ext2_field_eq(ext2_add(t0, t1), claim) {
            return None;
        }
        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        claim = combined_at(msg, c, alpha[i], eta, r);
        c = ext2_mul(c, eq_scalar(r, alpha[i]));
        xi.push(r);
    }

    for a in &proof.terminal {
        transcript.append_ext2(b"link-terminal", a);
    }

    // Final check at ξ. Everything except a_e is public: `c` is now eq(ξ,α), and
    // ω_e(ξ) is a tagged sum of eq's over this commitment's queries — which is
    // why only a_e travels in the proof.
    let mut expect = ext2_zero();
    let mut omega_xi = vec![ext2_zero(); n_commit];
    for (j, q) in queries.iter().enumerate() {
        if q.commitment >= n_commit || q.point.len() != ambient_arity {
            return None;
        }
        let e = q.commitment;
        omega_xi[e] = ext2_add(omega_xi[e], ext2_mul(query_tags[j], eq_at(&q.point, &xi)));
    }
    for e in 0..n_commit {
        let a = proof.terminal[e];
        let norm = ext2_mul(ext2_mul(c, commit_tags[e]), range_poly(a));
        let eval = ext2_mul(eta, ext2_mul(omega_xi[e], a));
        expect = ext2_add(expect, ext2_add(norm, eval));
    }

    if !ext2_field_eq(expect, claim) {
        return None;
    }
    Some((xi, proof.terminal.clone()))
}

/// Lagrange-interpolate from values at `0..n-1` and evaluate at `r`.
fn lagrange_eval(values: &[Ext2], r: Ext2) -> Ext2 {
    let n = values.len();
    let mut acc = ext2_zero();
    for i in 0..n {
        let xi_ = ext2_from_u64(i as u64);
        let mut num = ext2_one();
        let mut den = ext2_one();
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = ext2_from_u64(j as u64);
            num = ext2_mul(num, ext2_sub(r, xj));
            den = ext2_mul(den, ext2_sub(xi_, xj));
        }
        let inv = crate::util::arith::ext2_inv(den);
        acc = ext2_add(acc, ext2_mul(values[i], ext2_mul(num, inv)));
    }
    acc
}

/// Derive the tag base, `α`, and `η`. Prover and verifier must agree exactly,
/// so this is shared rather than duplicated.
fn derive_challenges(
    transcript: &mut Transcript,
    n_commit: usize,
    queries: &[LinkQuery],
    ambient_arity: usize,
) -> (Vec<Ext2>, Vec<Ext2>, Vec<Ext2>, Ext2) {
    transcript.append_u64(b"link-commitments", n_commit as u64);
    transcript.append_u64(b"link-queries", queries.len() as u64);
    transcript.append_u64(b"link-arity", ambient_arity as u64);
    for q in queries {
        transcript.append_u64(b"q-commit", q.commitment as u64);
        transcript.append_ext2(b"q-value", &q.value);
        for p in &q.point {
            transcript.append_ext2(b"q-point", p);
        }
    }
    let gamma = transcript.challenge_ext2(b"link-gamma");
    // Disjoint tag ranges: commitments take γ^1..γ^{|I|}, queries take the
    // range above that. Overlapping tags would let the norm and evaluation
    // halves cancel each other.
    let all = tag_powers(gamma, n_commit + queries.len());
    let commit_tags = all[..n_commit].to_vec();
    let query_tags = all[n_commit..].to_vec();

    let alpha: Vec<Ext2> = (0..ambient_arity)
        .map(|_| transcript.challenge_ext2(b"link-alpha"))
        .collect();
    let eta = transcript.challenge_ext2(b"link-eta");
    (commit_tags, query_tags, alpha, eta)
}


// ============================================================================
// GPU path
// ============================================================================

use almost_goldilocks_cuda::memory::DeviceBuffer;
use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds spent in the round-message kernels by the last
/// [`prove_link_gpu`] call. Setup (query-weight construction, upload) is
/// excluded: it is O(total witness size) with block-local queries but O(t·2^A)
/// with full-domain ones, so mixing it in would measure the benchmark's query
/// shape rather than the sumcheck.
pub static LAST_ROUND_NANOS: AtomicU64 = AtomicU64::new(0);

/// Ext2 as the two canonical u64 limbs the kernels expect.
fn ext2_to_limbs(v: Ext2) -> [u64; 2] {
    [v.c0.0, v.c1.0]
}
fn limbs_to_ext2(l: &[u64]) -> Ext2 {
    Ext2::new(AlmostGoldilocksField(l[0]), AlmostGoldilocksField(l[1]))
}
fn flatten(vs: &[Ext2]) -> Vec<u64> {
    let mut out = Vec::with_capacity(vs.len() * 2);
    for v in vs {
        let l = ext2_to_limbs(*v);
        out.push(l[0]);
        out.push(l[1]);
    }
    out
}

/// Index chunks per commitment. Enough blocks to fill an A100 without making
/// the partial buffer (and its reduction) dominate at small `half`.
fn chunks_for(half: usize) -> usize {
    const BLOCK: usize = 256;
    ((half + BLOCK - 1) / BLOCK).clamp(1, 64)
}

/// Chunks for an explicit work count (sparse path), where the grid must track
/// the list length rather than the ambient domain.
fn chunks_for_work(items: usize) -> usize {
    const BLOCK: usize = 256;
    ((items + BLOCK - 1) / BLOCK).clamp(1, 64)
}


/// Reusable device buffers for the link.
///
/// The prover runs one link per batch, and each batch was allocating and
/// freeing its own multi-gigabyte witness and query-weight buffers. Since every
/// batch is the same shape, the allocator churn is pure overhead — buffers are
/// grown once here and reused.
#[derive(Default)]
pub struct LinkScratch {
    w: Option<DeviceBuffer<u64>>,
    omega: Option<DeviceBuffer<u64>>,
    bits: Option<DeviceBuffer<u64>>,
    partial: Option<DeviceBuffer<u64>>,
}

impl LinkScratch {
    pub fn new() -> Self {
        Self::default()
    }
    /// Hand out a buffer of at least `len` elements, allocating only on growth.
    fn take(slot: &mut Option<DeviceBuffer<u64>>, len: usize) -> Option<&mut DeviceBuffer<u64>> {
        let need = slot.as_ref().map_or(true, |b| b.len() < len);
        if need {
            *slot = Some(DeviceBuffer::<u64>::new(len).ok()?);
        }
        slot.as_mut()
    }
}

/// GPU link prover. Produces a proof identical to [`prove_link`] — the transcript
/// is driven from the host, and every kernel is a pure function of device state,
/// so there is no ordering freedom that could make the two diverge.
///
/// Device residency is the point: the witness and query-weight tables are
/// uploaded once and folded in place, so only 7 field elements cross the bus per
/// round. That is also exactly the multi-GPU boundary — sharding `gridDim.y`
/// across devices turns those 7 elements into the all-reduce.
pub fn prove_link_gpu(
    witnesses: &[LinkWitness],
    queries: &[LinkQuery],
    ambient_arity: usize,
    transcript: &mut Transcript,
) -> Option<(LinkProof, Vec<Ext2>)> {
    let mut scratch = LinkScratch::new();
    prove_link_gpu_with(witnesses, queries, ambient_arity, transcript, &mut scratch)
}

/// [`prove_link_gpu`] reusing caller-owned device buffers across invocations.
pub fn prove_link_gpu_with(
    witnesses: &[LinkWitness],
    queries: &[LinkQuery],
    ambient_arity: usize,
    transcript: &mut Transcript,
    scratch: &mut LinkScratch,
) -> Option<(LinkProof, Vec<Ext2>)> {
    let n_commit = witnesses.len();
    let size = 1usize << ambient_arity;
    if n_commit == 0 {
        return None;
    }
    for w in witnesses {
        match &w.bits {
            Some(b) => assert_eq!(b.len() * 64, size, "bit image must fill the ambient domain"),
            None => assert_eq!(w.coeffs.len(), size, "witness must fill the ambient domain"),
        }
    }

    let timing = std::env::var("ZK4_LINK_TIMING").ok().as_deref() == Some("1");
    let t_start = std::time::Instant::now();

    let (commit_tags, query_tags, alpha, _eta) =
        derive_challenges(transcript, n_commit, queries, ambient_arity);

    // Host-side table construction mirrors the CPU path exactly.
    let all_bits = witnesses.iter().all(|w| w.bits.is_some());
    // Query weights are seeded on the host with one element per query — the eq
    // table over zero variables, scaled by the query's tag — and expanded on
    // device. Expansion is batched by *level* rather than by query: every query
    // whose block is still larger doubles its live span in one launch, so the
    // whole construction is `max_block_arity` launches instead of one chain per
    // query. At realistic query counts that is the difference between tens of
    // launches and six figures.
    // Seeded directly on device: the seed is one Ext2 per query in an otherwise
    // zero buffer, so materializing it on the host costs a multi-gigabyte
    // allocation to write a few kilobytes of data.
    let mut q_base = Vec::with_capacity(queries.len());
    let mut q_seed: Vec<[u64; 2]> = Vec::with_capacity(queries.len());
    let mut q_suffix: Vec<&[Ext2]> = Vec::with_capacity(queries.len());
    for (j, q) in queries.iter().enumerate() {
        let (offset, block_arity) = q.block(ambient_arity);
        let base = q.commitment * size + offset;
        // Distinct queries must not share a block, or their seeds would collide
        // rather than accumulate; the packing layout guarantees disjointness.
        let l = ext2_to_limbs(query_tags[j]);
        q_seed.push(l);
        if q_base.contains(&(base as u64)) {
            // Two queries on the same block would overwrite rather than
            // accumulate in the device seed. The packing layout gives each leaf
            // its own block, so this does not arise in practice; refuse instead
            // of silently producing a wrong weight.
            return None;
        }
        q_base.push(base as u64);
        q_suffix.push(&q.point[q.prefix_len..]);
        debug_assert_eq!(block_arity, q.point.len() - q.prefix_len);
    }

    let t_omega = t_start.elapsed();
    // Round 0 runs off the bit-packed witness and writes its folded, half-size
    // result directly. That matters for more than upload bandwidth: round 0 is
    // the largest round, so materializing its input as Ext2 would set the peak
    // at 16 bytes per witness bit for a table discarded after one fold.
    let words = size / 64;
    // Raw pointers into the reused buffers: two simultaneous &mut borrows of the
    // scratch are not expressible, and the FFI boundary is unsafe regardless.
    // The buffers outlive every call below.
    let bits_ptr: *const u64 = if all_bits {
        let b = LinkScratch::take(&mut scratch.bits, n_commit * words)?;
        for (e, wit) in witnesses.iter().enumerate() {
            b.copy_from_slice_at(e * words, wit.bits.as_ref().unwrap()).ok()?;
        }
        b.as_ptr()
    } else {
        std::ptr::null()
    };
    // Only the folded half is ever allocated at Ext2 width, and it is pooled
    // across batches like the rest.
    let w_ptr: *mut u64 = if all_bits {
        LinkScratch::take(&mut scratch.w, 2 * n_commit * (size / 2))?.as_mut_ptr()
    } else {
        std::ptr::null_mut()
    };
    let (mut d_w, d_bits) = if all_bits {
        (DeviceBuffer::<u64>::new(0).ok()?, Some(()))
    } else {
        let mut w_flat: Vec<u64> = Vec::with_capacity(n_commit * size * 2);
        for wit in witnesses {
            for c in &wit.coeffs {
                w_flat.push(*c);
                w_flat.push(0);
            }
        }
        (DeviceBuffer::<u64>::from_slice(&w_flat).ok()?, None)
    };
    // Stride of the Ext2 witness buffer: full ambient on the dense path, half on
    // the bit path (round 0 already consumed the top variable).
    let w_stride = if all_bits { size / 2 } else { size };
    let (omega_ptr, omega_mut) = {
        let o = LinkScratch::take(&mut scratch.omega, 2 * n_commit * size)?;
        o.zero().ok()?;
        (o.as_ptr(), o.as_mut_ptr())
    };
    for (b, l) in q_base.iter().zip(q_seed.iter()) {
        // One Ext2 per query into an otherwise zeroed device buffer.
        let rc = unsafe {
            almost_goldilocks_cuda::ffi::cuda_memcpy_htod(
                omega_mut.add(2 * (*b) as usize) as *mut std::ffi::c_void,
                l.as_ptr() as *const std::ffi::c_void,
                2 * std::mem::size_of::<u64>(),
            )
        };
        if rc != 0 {
            return None;
        }
    }
    {
        // Level-batched expansion. eq_table consumes its point in reverse, so
        // level k uses the k-th coordinate from the end of each query's suffix.
        let max_arity = q_suffix.iter().map(|p| p.len()).max().unwrap_or(0);
        for k in 0..max_arity {
            let mut bases = Vec::new();
            let mut rs = Vec::new();
            for (i, suf) in q_suffix.iter().enumerate() {
                if suf.len() > k {
                    bases.push(q_base[i]);
                    let l = ext2_to_limbs(suf[suf.len() - 1 - k]);
                    rs.push(l[0]);
                    rs.push(l[1]);
                }
            }
            if bases.is_empty() {
                continue;
            }
            let d_bases = DeviceBuffer::<u64>::from_slice(&bases).ok()?;
            let d_rs = DeviceBuffer::<u64>::from_slice(&rs).ok()?;
            let rc = unsafe {
                almost_goldilocks_cuda::ffi::link_omega_expand_ffi(
                    omega_mut, d_bases.as_ptr(), d_rs.as_ptr(),
                    1u64 << k, bases.len() as u64,
                )
            };
            if rc != 0 {
                return None;
            }
        }
    }
    let d_tags = DeviceBuffer::<u64>::from_slice(&flatten(&commit_tags)).ok()?;
    let mut d_out = DeviceBuffer::<u64>::new(2 * (NORM_EVALS + EVAL_EVALS)).ok()?;
    let max_chunks = chunks_for(size >> 1);
    let mut d_partial =
        DeviceBuffer::<u64>::new(2 * n_commit * max_chunks * (NORM_EVALS + EVAL_EVALS)).ok()?;

    // Support lists, one per commitment per round.
    //
    // Folding never creates a nonzero outside its parents' positions, so the
    // set of pairs that can contribute at round r is determined by the initial
    // support alone — independent of the challenges. That makes the lists
    // precomputable at setup, which avoids a GPU stream-compaction per round.
    //
    // The dense kernel's zero test only retires a warp when all 32 lanes are
    // zero, and one-hot Shout advice spaces its nonzeros 2^table_commit_log
    // apart, so the dense path caps out at warp width however sparse the witness
    // is. Driving the loop from the list removes that cap.
    let t_sup = std::time::Instant::now();
    // Support is read from the bit image when there is one, so the caller never
    // has to materialize a coefficient-per-u64 copy of a binary witness.
    let supports: Vec<Vec<u32>> = witnesses
        .iter()
        .map(|w| match &w.bits {
            Some(bits) => {
                let mut out = Vec::new();
                for (wi, word) in bits.iter().enumerate() {
                    let mut m = *word;
                    while m != 0 {
                        let b = m.trailing_zeros() as usize;
                        m &= m - 1;
                        out.push((wi * 64 + b) as u32);
                    }
                }
                out
            }
            None => w
                .coeffs
                .iter()
                .enumerate()
                .filter(|(_, c)| **c != 0)
                .map(|(i, _)| i as u32)
                .collect(),
        })
        .collect();

    // Read list per round: L_r = { p mod h_r : p in support }, the pairs whose
    // contribution to the round message can be nonzero. Folding never creates a
    // nonzero outside its parents' positions, so L_r depends only on the initial
    // support and is precomputable — no GPU stream compaction per round.
    let d_sup_ms = t_sup.elapsed().as_secs_f64()*1e3;
    let t_lists = std::time::Instant::now();
    // Built incrementally: L_{r+1} = { p mod h_{r+1} : p in L_r }. Deriving each
    // round from the full support instead costs O(|support| x rounds), which at
    // 9e7 nonzeros and 27 rounds is billions of host operations — it was the
    // largest single line in the profile before this.
    let mut read_lists: Vec<(Vec<u32>, Vec<u64>, Vec<u64>)> = Vec::with_capacity(ambient_arity);
    {
        let mut seen = vec![false; (size >> 1).max(1)];
        let mut prev: Vec<Vec<u32>> = supports.clone();
        for r in 0..ambient_arity {
            let h = (size >> (r + 1)).max(1);
            let mut cat: Vec<u32> = Vec::new();
            let mut offs = Vec::with_capacity(n_commit);
            let mut lens = Vec::with_capacity(n_commit);
            let mut next: Vec<Vec<u32>> = Vec::with_capacity(n_commit);
            for sup in &prev {
                offs.push(cat.len() as u64);
                let start = cat.len();
                for &p in sup {
                    let q = (p as usize) & (h - 1);
                    if !seen[q] {
                        seen[q] = true;
                        cat.push(q as u32);
                    }
                }
                for &v in &cat[start..] {
                    seen[v as usize] = false;
                }
                lens.push((cat.len() - start) as u64);
                next.push(cat[start..].to_vec());
            }
            read_lists.push((cat, offs, lens));
            prev = next;
        }
    }

    let d_lists_ms = t_lists.elapsed().as_secs_f64()*1e3;
    // Take the sparse path only while it actually saves work: past half the
    // pairs, the extra index load costs more than the skipped arithmetic.
    let use_sparse: Vec<bool> = (0..ambient_arity)
        .map(|r| {
            let h = (size >> (r + 1)).max(1);
            // Below a few thousand pairs the round is launch-bound, and the
            // gather plus list upload cost more than the arithmetic they save.
            h >= 1 << 13
                && read_lists[r].2.iter().all(|&l| (l as usize) * 2 <= h)
        })
        .collect();

    let t_upload = t_start.elapsed();
    let mut d_eq_ms = 0f64;
    let mut d_round_ms = 0f64;
    let mut d_fold_ms = 0f64;

    // One dense eq table for the whole sumcheck, plus the per-round scalar that
    // converts a prefix of it into that round's suffix eq.
    let t_eqs = std::time::Instant::now();
    // The dense suffix-eq table is only needed by rounds that decline the sparse
    // path, which are the late, small ones (the sparse path is declined below a
    // few thousand pairs, where it is launch-bound). Sizing the table to the
    // earliest such round instead of the full ambient is the difference between
    // a 1 GiB build per batch and a negligible one.
    let dense_from = (0..ambient_arity).find(|i| !use_sparse[*i]);
    let d_eq_shared = match dense_from {
        None => DeviceBuffer::<u64>::new(0).ok()?,
        Some(i0) => {
            let m = ambient_arity - i0 - 1; // entries = 2^m
            let mut seedv = vec![0u64; 2 * (1usize << m)];
            seedv[0] = 1;
            let mut d = DeviceBuffer::<u64>::from_slice(&seedv).ok()?;
            let mut span = 1usize;
            for &r in alpha[i0 + 1..].iter().rev() {
                let l = ext2_to_limbs(r);
                let rc = unsafe {
                    almost_goldilocks_cuda::ffi::link_eq_expand_ffi(
                        d.as_mut_ptr(), span as u64, l[0], l[1],
                    )
                };
                if rc != 0 {
                    return None;
                }
                span <<= 1;
            }
            d
        }
    };
    let d_eqs_ms = t_eqs.elapsed().as_secs_f64() * 1e3;
    // Round i >= i0 reads a prefix of that table, which carries an extra factor
    // prod_{j=i0+1..i}(1 - alpha_j); the norm half is divided by it afterwards.
    let mut eq_prefix_scale = vec![ext2_one(); ambient_arity];
    if let Some(i0) = dense_from {
        let mut acc = ext2_one();
        for i in i0..ambient_arity {
            eq_prefix_scale[i] = acc;
            if i + 1 < ambient_arity {
                acc = ext2_mul(acc, ext2_sub(ext2_one(), alpha[i + 1]));
            }
        }
    }

    let mut rounds = Vec::with_capacity(ambient_arity);
    let mut xi = Vec::with_capacity(ambient_arity);
    let mut half = size >> 1;

    for i in 0..ambient_arity {
        let t_r = std::time::Instant::now();
        // eq(x_>i, alpha_>i), built on device by expanding one variable at a
        // time over the suffix of alpha.
        let suffix = &alpha[i + 1..];
        // Round 0 on the bit path uses the dense kernel, which indexes the eq
        // table by domain position; the gathered table is indexed by list
        // position. Mixing them reads out of bounds, so round 0 keeps the dense
        // eq build (one 2^(A-1) table, paid once).
        let sparse_now = use_sparse[i];
        // Seed the table with eq() over zero variables = [1], then expand one
        // variable at a time on device. Seeding from a host vector avoids a
        // device-to-device copy (a plain memcpy across device pointers is not
        // valid host memory access).
        // Dense suffix eq: derived from one table built once, not rebuilt per
        // round. `eq_table(alpha[1..])` restricted to its first 2^(A-1-i)
        // entries equals eq(x_>i, alpha_>i) scaled by prod_{j<=i}(1-alpha_j), so
        // each round is a prefix of the same buffer plus one scalar correction
        // applied to the norm half afterwards. Rebuilding cost O(A) launches per
        // round, which at A = 27 was the single largest line in the profile.
        let mut d_eq = if sparse_now {
            // Placeholder; filled by the gather below once the list is on device.
            DeviceBuffer::<u64>::new(2).ok()?
        } else {
            DeviceBuffer::<u64>::new(0).ok()?
        };
        let mut eq_scale = ext2_one();
        if !sparse_now {
            eq_scale = eq_prefix_scale[i];
        }

        d_eq_ms += t_r.elapsed().as_secs_f64() * 1e3;
        let t_r = std::time::Instant::now();

        // Size the grid to the work that actually exists. With a sparse list the
        // dense-domain grid launches hundreds of blocks that each pay a full
        // 7-slot block reduction over 256 threads for a handful of entries, and
        // that fixed cost, not the arithmetic, becomes the floor.
        let chunks = if use_sparse[i] {
            let max_len = read_lists[i].2.iter().copied().max().unwrap_or(1) as usize;
            chunks_for_work(max_len)
        } else {
            chunks_for(half)
        };
        let dev = |t: &(Vec<u32>, Vec<u64>, Vec<u64>)| {
            (
                DeviceBuffer::<u32>::from_slice(&t.0),
                DeviceBuffer::<u64>::from_slice(&t.1),
                DeviceBuffer::<u64>::from_slice(&t.2),
            )
        };
        let lists = if sparse_now { Some(dev(&read_lists[i])) } else { None };
        if let Some((Ok(dc), _, _)) = &lists {
            let total = read_lists[i].0.len() as u64;
            let mut a_flat = Vec::with_capacity(suffix.len() * 2);
            for r in suffix {
                let l = ext2_to_limbs(*r);
                a_flat.push(l[0]);
                a_flat.push(l[1]);
            }
            if a_flat.is_empty() {
                a_flat.extend_from_slice(&[0, 0]);
            }
            let d_alpha = DeviceBuffer::<u64>::from_slice(&a_flat).ok()?;
            let mut d_gather = DeviceBuffer::<u64>::new((2 * total.max(1)) as usize).ok()?;
            let rc = unsafe {
                almost_goldilocks_cuda::ffi::link_eq_gather_ffi(
                    d_alpha.as_ptr(), suffix.len() as u64, dc.as_ptr(),
                    d_gather.as_mut_ptr(), total,
                )
            };
            if rc != 0 {
                return None;
            }
            d_eq = d_gather;
        }
        let rc = if i == 0 && d_bits.is_some() {
            let b = d_bits.as_ref().unwrap();
            match &lists {
                Some((Ok(dc), Ok(doff), Ok(dlen))) => unsafe {
                    almost_goldilocks_cuda::ffi::link_round0_bits_sparse_ffi(
                        bits_ptr, omega_ptr, d_eq.as_ptr(), d_tags.as_ptr(),
                        dc.as_ptr(), doff.as_ptr(), dlen.as_ptr(),
                        size as u64, half as u64, n_commit as u64,
                        d_partial.as_mut_ptr(), chunks as u64, d_out.as_mut_ptr(),
                    )
                },
                _ => unsafe {
                    almost_goldilocks_cuda::ffi::link_round0_bits_ffi(
                        bits_ptr, omega_ptr, d_eq_shared.as_ptr(), d_tags.as_ptr(),
                        size as u64, half as u64, n_commit as u64,
                        d_partial.as_mut_ptr(), chunks_for(half) as u64, d_out.as_mut_ptr(),
                    )
                },
            }
        } else {
            match &lists {
                Some((Ok(dc), Ok(doff), Ok(dlen))) => unsafe {
                    almost_goldilocks_cuda::ffi::link_round_message_sparse_ffi(
                        if all_bits { w_ptr as *const u64 } else { d_w.as_ptr() }, omega_ptr, d_eq.as_ptr(), d_tags.as_ptr(),
                        dc.as_ptr(), doff.as_ptr(), dlen.as_ptr(),
                        w_stride as u64, size as u64, half as u64, n_commit as u64,
                        0, d_partial.as_mut_ptr(), chunks as u64,
                        d_out.as_mut_ptr(),
                    )
                },
                _ => unsafe {
                    almost_goldilocks_cuda::ffi::link_round_message_ffi(
                        if all_bits { w_ptr as *const u64 } else { d_w.as_ptr() }, omega_ptr, d_eq_shared.as_ptr(), d_tags.as_ptr(),
                        w_stride as u64, size as u64, half as u64, n_commit as u64,
                        0, d_partial.as_mut_ptr(), chunks as u64,
                        d_out.as_mut_ptr(),
                    )
                },
            }
        };
        if rc != 0 {
            return None;
        }
        let raw = d_out.to_vec().ok()?;
        // Undo the shared table's prefix factor on the norm half.
        let inv = crate::util::arith::ext2_inv(eq_scale);
        let msg = RoundMsg {
            norm: (0..NORM_EVALS)
                .map(|k| ext2_mul(limbs_to_ext2(&raw[2 * k..]), inv))
                .collect(),
            eval: (0..EVAL_EVALS)
                .map(|k| limbs_to_ext2(&raw[2 * (NORM_EVALS + k)..]))
                .collect(),
        };

        d_round_ms += t_r.elapsed().as_secs_f64() * 1e3;

        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        xi.push(r);

        let t_r = std::time::Instant::now();
        let l = ext2_to_limbs(r);
        // The fold stays dense even when the message is sparse. `omega` is an eq
        // table, so the message at round r+1 reads it at both `idx` and
        // `idx + h`, and only one of those is guaranteed to be in round r's
        // read list. Propagating that requirement backwards doubles the needed
        // set each round, which degenerates to the full domain — so a sparse
        // fold buys nothing and silently corrupts `omega`. The fold is ~2 muls
        // per entry against the message's ~11, so this keeps most of the win.
        let rc = if i == 0 && d_bits.is_some() {
            let b = d_bits.as_ref().unwrap();
            unsafe {
                almost_goldilocks_cuda::ffi::link_fold0_bits_ffi(
                    bits_ptr, if all_bits { w_ptr } else { d_w.as_mut_ptr() }, omega_mut,
                    size as u64, half as u64, n_commit as u64, l[0], l[1],
                )
            }
        } else {
            unsafe {
                almost_goldilocks_cuda::ffi::link_fold_ffi(
                    if all_bits { w_ptr } else { d_w.as_mut_ptr() }, omega_mut, w_stride as u64,
                    size as u64, half as u64, n_commit as u64, l[0], l[1],
                )
            }
        };
        if rc != 0 {
            return None;
        }

        d_fold_ms += t_r.elapsed().as_secs_f64() * 1e3;
        rounds.push(msg);
        half >>= 1;
    }

    LAST_ROUND_NANOS.store((d_round_ms * 1e6) as u64, Ordering::Relaxed);
    if timing {
        let coeffs = (n_commit * size) as f64;
        eprintln!(
            "[link_gpu] omega(host) {:.1}ms  upload {:.1}ms  sup {:.1}ms  lists {:.1}ms  eqshared {:.1}ms  eq {:.1}ms  round {:.1}ms  fold {:.1}ms  | round-only {:.4} ns/coeff",
            t_omega.as_secs_f64() * 1e3,
            (t_upload - t_omega).as_secs_f64() * 1e3,
            d_sup_ms, d_lists_ms, d_eqs_ms,
            d_eq_ms, d_round_ms, d_fold_ms,
            d_round_ms * 1e6 / coeffs,
        );
    }

    // Only the first element of each commitment's region is needed; downloading
    // the whole buffer to read n_commit values would move gigabytes.
    let mut terminal = Vec::with_capacity(n_commit);
    for e in 0..n_commit {
        let mut limbs = [0u64; 2];
        if all_bits {
            let rc = unsafe {
                almost_goldilocks_cuda::ffi::cuda_memcpy_dtoh(
                    limbs.as_mut_ptr() as *mut std::ffi::c_void,
                    w_ptr.add(2 * e * w_stride) as *const std::ffi::c_void,
                    2 * std::mem::size_of::<u64>(),
                )
            };
            if rc != 0 {
                return None;
            }
        } else {
            d_w.copy_to_slice_at(2 * e * w_stride, &mut limbs).ok()?;
        }
        terminal.push(limbs_to_ext2(&limbs));
    }
    for a in &terminal {
        transcript.append_ext2(b"link-terminal", a);
    }

    Some((LinkProof { rounds, terminal }, xi))
}



// ============================================================================
// Interleaved-layout prover (query weights evaluated on demand)
// ============================================================================

/// A leaf's claim as the interleaved prover wants it: which block it occupies,
/// its own point, and its value.
#[derive(Clone, Debug)]
pub struct BlockClaim {
    pub commitment: usize,
    pub block: usize,
    /// Leaf point, MSB-first, `leaf_arity` entries.
    pub point: Vec<Ext2>,
    pub value: Ext2,
}

/// Prove the link over an interleaved packed witness.
///
/// The witness is bit-packed, one entry per commitment, `2^ambient` bits with
/// the block index in the low `block_bits`. Nothing here materializes a query
/// weight table: during the leaf rounds each live position belongs to exactly
/// one block, so its weight is `scale[block] * eq(remaining leaf index, point)`
/// and the kernel evaluates it. Once the leaf variables are exhausted the
/// surviving domain is just the blocks, small enough to fold densely.
pub fn prove_link_interleaved(
    bits: &[Vec<u64>],
    claims: &[BlockClaim],
    leaf_arity: usize,
    block_bits: usize,
    transcript: &mut Transcript,
    scratch: &mut LinkScratch,
) -> Option<(LinkProof, Vec<Ext2>)> {
    let n_commit = bits.len();
    let ambient = leaf_arity + block_bits;
    let size = 1usize << ambient;
    let blocks = 1usize << block_bits;
    if n_commit == 0 {
        return None;
    }

    // Lift each block claim to a packed query so the challenge derivation — and
    // therefore the transcript — matches the generic path exactly.
    let queries: Vec<LinkQuery> = claims
        .iter()
        .map(|c| {
            let mut point = c.point.clone();
            for k in 0..block_bits {
                let bit = (c.block >> (block_bits - 1 - k)) & 1;
                point.push(ext2_from_u64(bit as u64));
            }
            LinkQuery {
                commitment: c.commitment,
                point,
                value: c.value,
                prefix_len: 0,
            }
        })
        .collect();
    let (commit_tags, query_tags, alpha, _eta) =
        derive_challenges(transcript, n_commit, &queries, ambient);

    // Per (commitment, block): the leaf point and the running scale.
    let mut pts = vec![ext2_zero(); n_commit * blocks * leaf_arity.max(1)];
    let mut scale = vec![ext2_zero(); n_commit * blocks];
    for (j, c) in claims.iter().enumerate() {
        let base = (c.commitment * blocks + c.block) * leaf_arity.max(1);
        for (k, p) in c.point.iter().enumerate() {
            pts[base + k] = *p;
        }
        scale[c.commitment * blocks + c.block] = query_tags[j];
    }

    let words = size / 64;
    let bits_ptr = {
        let b = LinkScratch::take(&mut scratch.bits, n_commit * words)?;
        for (e, w) in bits.iter().enumerate() {
            b.copy_from_slice_at(e * words, w).ok()?;
        }
        b.as_ptr()
    };
    // One half-size witness buffer, folded in place. Round 0 reads the bits and
    // writes here; every later round reads and writes the same buffer, which is
    // safe because each thread consumes exactly the two entries it overwrites.
    // This is the whole memory win: 8 bytes per witness bit, against 32 for the
    // contiguous layout's witness-plus-weights.
    let w_buf = LinkScratch::take(&mut scratch.w, 2 * n_commit * (size / 2))?.as_mut_ptr();
    let d_tags = DeviceBuffer::<u64>::from_slice(&flatten(&commit_tags)).ok()?;
    let mut d_out = DeviceBuffer::<u64>::new(2 * (NORM_EVALS + EVAL_EVALS)).ok()?;
    let mut d_partial =
        DeviceBuffer::<u64>::new(2 * n_commit * 64 * (NORM_EVALS + EVAL_EVALS)).ok()?;
    let d_pts = DeviceBuffer::<u64>::from_slice(&flatten(&pts)).ok()?;

    // Support lists, built incrementally as in the generic path.
    let supports: Vec<Vec<u32>> = bits
        .iter()
        .map(|b| {
            let mut out = Vec::new();
            for (wi, word) in b.iter().enumerate() {
                let mut m = *word;
                while m != 0 {
                    let t = m.trailing_zeros() as usize;
                    m &= m - 1;
                    out.push((wi * 64 + t) as u32);
                }
            }
            out
        })
        .collect();

    let mut rounds = Vec::with_capacity(ambient);
    let mut xi = Vec::with_capacity(ambient);
    let mut half = size >> 1;
    let mut prev: Vec<Vec<u32>> = supports;
    let mut seen = vec![false; (size >> 1).max(1)];
    let mut cur_scale = scale.clone();
    let mut src_is_bits = true;
    // Both witness buffers keep a fixed stride of size/2 — the largest live
    // domain any round produces. Shrinking the stride with the domain would
    // overlap adjacent commitments' regions.
    let w_stride_fixed = size / 2;

    for i in 0..leaf_arity {
        // Round list.
        let mut cat: Vec<u32> = Vec::new();
        let mut offs = Vec::with_capacity(n_commit);
        let mut lens = Vec::with_capacity(n_commit);
        let mut next: Vec<Vec<u32>> = Vec::with_capacity(n_commit);
        for sup in &prev {
            offs.push(cat.len() as u64);
            let start = cat.len();
            for &p in sup {
                let q = (p as usize) & (half - 1);
                if !seen[q] {
                    seen[q] = true;
                    cat.push(q as u32);
                }
            }
            for &v in &cat[start..] {
                seen[v as usize] = false;
            }
            lens.push((cat.len() - start) as u64);
            next.push(cat[start..].to_vec());
        }

        let d_list = DeviceBuffer::<u32>::from_slice(&cat).ok()?;
        let d_off = DeviceBuffer::<u64>::from_slice(&offs).ok()?;
        let d_len = DeviceBuffer::<u64>::from_slice(&lens).ok()?;
        let d_scale = DeviceBuffer::<u64>::from_slice(&flatten(&cur_scale)).ok()?;

        // eq(x_>i, alpha_>i) gathered at the listed positions.
        let mut a_flat = Vec::with_capacity((ambient - i - 1) * 2);
        for r in &alpha[i + 1..] {
            let l = ext2_to_limbs(*r);
            a_flat.push(l[0]);
            a_flat.push(l[1]);
        }
        if a_flat.is_empty() {
            a_flat.extend_from_slice(&[0, 0]);
        }
        let d_alpha = DeviceBuffer::<u64>::from_slice(&a_flat).ok()?;
        let mut d_eq = DeviceBuffer::<u64>::new(2 * cat.len().max(1)).ok()?;
        let rc = unsafe {
            almost_goldilocks_cuda::ffi::link_eq_gather_ffi(
                d_alpha.as_ptr(), (ambient - i - 1) as u64, d_list.as_ptr(),
                d_eq.as_mut_ptr(), cat.len() as u64,
            )
        };
        if rc != 0 {
            return None;
        }

        let max_len = lens.iter().copied().max().unwrap_or(1) as usize;
        let chunks = chunks_for_work(max_len);
        let rc = unsafe {
            almost_goldilocks_cuda::ffi::link_round_interleaved_ffi(
                bits_ptr,
                if src_is_bits { std::ptr::null() } else { w_buf as *const u64 },
                d_pts.as_ptr(), d_scale.as_ptr(), d_eq.as_ptr(), d_tags.as_ptr(),
                d_list.as_ptr(), d_off.as_ptr(), d_len.as_ptr(),
                w_stride_fixed as u64, (size / 64) as u64, half as u64, n_commit as u64,
                (blocks - 1) as u32, block_bits as i32, leaf_arity as i32,
                i as i32, i32::from(src_is_bits),
                d_partial.as_mut_ptr(), chunks as u64, d_out.as_mut_ptr(),
            )
        };
        if rc != 0 {
            return None;
        }
        let raw = d_out.to_vec().ok()?;
        let msg = RoundMsg {
            norm: (0..NORM_EVALS).map(|k| limbs_to_ext2(&raw[2 * k..])).collect(),
            eval: (0..EVAL_EVALS)
                .map(|k| limbs_to_ext2(&raw[2 * (NORM_EVALS + k)..]))
                .collect(),
        };
        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        xi.push(r);

        // Fold the witness; the weights need no fold, only a scale update.
        let l = ext2_to_limbs(r);
        let rc = unsafe {
            almost_goldilocks_cuda::ffi::link_fold_w_ffi(
                bits_ptr,
                if src_is_bits { std::ptr::null() } else { w_buf as *const u64 },
                w_buf, w_stride_fixed as u64, (size / 64) as u64,
                w_stride_fixed as u64, half as u64, n_commit as u64,
                i32::from(src_is_bits), l[0], l[1],
            )
        };
        if rc != 0 {
            return None;
        }
        for e in 0..n_commit {
            for b in 0..blocks {
                let idx = e * blocks + b;
                let p = pts[idx * leaf_arity.max(1) + i];
                cur_scale[idx] = ext2_mul(cur_scale[idx], eq_scalar(r, p));
            }
        }

        rounds.push(msg);
        prev = next;
        src_is_bits = false;
        half >>= 1;
    }

    // Block phase: the witness has collapsed to one value per block and the
    // weight is exactly `scale`, so both are small dense arrays from here.
    let mut w_host = vec![0u64; 2 * n_commit * blocks];
    for e in 0..n_commit {
        let rc = unsafe {
            almost_goldilocks_cuda::ffi::cuda_memcpy_dtoh(
                w_host[2 * e * blocks..].as_mut_ptr() as *mut std::ffi::c_void,
                (w_buf as *const u64).add(2 * e * w_stride_fixed) as *const std::ffi::c_void,
                2 * blocks * std::mem::size_of::<u64>(),
            )
        };
        if rc != 0 {
            return None;
        }
    }
    let mut w_small: Vec<Vec<Ext2>> = (0..n_commit)
        .map(|e| {
            (0..blocks)
                .map(|b| limbs_to_ext2(&w_host[2 * (e * blocks + b)..]))
                .collect()
        })
        .collect();
    let mut o_small: Vec<Vec<Ext2>> = (0..n_commit)
        .map(|e| (0..blocks).map(|b| cur_scale[e * blocks + b]).collect())
        .collect();

    let mut hb = blocks >> 1;
    for i in leaf_arity..ambient {
        let eq_suffix = eq_table(&alpha[i + 1..]);
        let mut norm = vec![ext2_zero(); NORM_EVALS];
        let mut eval = vec![ext2_zero(); EVAL_EVALS];
        for e in 0..n_commit {
            let mut ne = [ext2_zero(); NORM_EVALS];
            for idx in 0..hb {
                let w_lo = w_small[e][idx];
                let w_d = ext2_sub(w_small[e][hb + idx], w_lo);
                let o_lo = o_small[e][idx];
                let o_d = ext2_sub(o_small[e][hb + idx], o_lo);
                let eqs = eq_suffix[idx];
                let w0 = w_lo;
                let w1 = ext2_add(w0, w_d);
                let w2 = ext2_add(w1, w_d);
                let w3 = ext2_add(w2, w_d);
                ne[0] = ext2_add(ne[0], ext2_mul(eqs, range_poly(w0)));
                ne[1] = ext2_add(ne[1], ext2_mul(eqs, range_poly(w1)));
                ne[2] = ext2_add(ne[2], ext2_mul(eqs, range_poly(w2)));
                ne[3] = ext2_add(ne[3], ext2_mul(eqs, range_poly(w3)));
                let o0 = o_lo;
                let o1 = ext2_add(o0, o_d);
                let o2 = ext2_add(o1, o_d);
                eval[0] = ext2_add(eval[0], ext2_mul(o0, w0));
                eval[1] = ext2_add(eval[1], ext2_mul(o1, w1));
                eval[2] = ext2_add(eval[2], ext2_mul(o2, w2));
            }
            for k in 0..NORM_EVALS {
                norm[k] = ext2_add(norm[k], ext2_mul(commit_tags[e], ne[k]));
            }
        }
        let msg = RoundMsg { norm, eval };
        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        xi.push(r);
        for e in 0..n_commit {
            for idx in 0..hb {
                let wl = w_small[e][idx];
                w_small[e][idx] = ext2_add(wl, ext2_mul(r, ext2_sub(w_small[e][hb + idx], wl)));
                let ol = o_small[e][idx];
                o_small[e][idx] = ext2_add(ol, ext2_mul(r, ext2_sub(o_small[e][hb + idx], ol)));
            }
            w_small[e].truncate(hb.max(1));
            o_small[e].truncate(hb.max(1));
        }
        rounds.push(msg);
        hb = (hb >> 1).max(if i + 1 < ambient { 1 } else { 0 });
    }

    let terminal: Vec<Ext2> = (0..n_commit).map(|e| w_small[e][0]).collect();
    for a in &terminal {
        transcript.append_ext2(b"link-terminal", a);
    }
    Some((LinkProof { rounds, terminal }, xi))
}


// ============================================================================
// Multi-GPU interleaved prover
// ============================================================================

/// Per-device state for one shard of the commitments.
///
/// The round message is a sum over commitments, so a device owns a disjoint
/// subset outright: its witness is uploaded once, folded in place, and never
/// moves. Only the 7-element partial message crosses between devices each
/// round — about 112 bytes against gigabytes of resident state.
struct Shard {
    device: i32,
    start: usize,
    len: usize,
    bits: DeviceBuffer<u64>,
    w: DeviceBuffer<u64>,
    pts: DeviceBuffer<u64>,
    tags: DeviceBuffer<u64>,
    partial: DeviceBuffer<u64>,
    out: DeviceBuffer<u64>,
    /// Support lists for this shard's commitments, one entry per round.
    prev: Vec<Vec<u32>>,
}

/// Split `n` commitments across `devices`, largest-first so the remainder lands
/// on the earliest devices rather than leaving one device idle.
fn shard_split(n: usize, devices: usize) -> Vec<(usize, usize)> {
    let d = devices.max(1).min(n.max(1));
    let base = n / d;
    let rem = n % d;
    let mut out = Vec::with_capacity(d);
    let mut start = 0;
    for i in 0..d {
        let len = base + usize::from(i < rem);
        if len > 0 {
            out.push((start, len));
        }
        start += len;
    }
    out
}

/// Multi-GPU interleaved link prover.
///
/// Produces the same proof as [`prove_link_interleaved`] for any device count:
/// the transcript is driven from the host and the per-device partials are summed
/// in a fixed order, so sharding is a scheduling choice, not a protocol one.
pub fn prove_link_interleaved_multi(
    bits: &[Vec<u64>],
    claims: &[BlockClaim],
    leaf_arity: usize,
    block_bits: usize,
    devices: &[i32],
    transcript: &mut Transcript,
) -> Option<(LinkProof, Vec<Ext2>)> {
    use rayon::prelude::*;

    let n_commit = bits.len();
    let ambient = leaf_arity + block_bits;
    let size = 1usize << ambient;
    let blocks = 1usize << block_bits;
    if n_commit == 0 || devices.is_empty() {
        return None;
    }

    let queries: Vec<LinkQuery> = claims
        .iter()
        .map(|c| {
            let mut point = c.point.clone();
            for k in 0..block_bits {
                let bit = (c.block >> (block_bits - 1 - k)) & 1;
                point.push(ext2_from_u64(bit as u64));
            }
            LinkQuery {
                commitment: c.commitment,
                point,
                value: c.value,
                prefix_len: 0,
            }
        })
        .collect();
    let (commit_tags, query_tags, alpha, _eta) =
        derive_challenges(transcript, n_commit, &queries, ambient);

    let mut pts = vec![ext2_zero(); n_commit * blocks * leaf_arity.max(1)];
    let mut scale = vec![ext2_zero(); n_commit * blocks];
    for (j, c) in claims.iter().enumerate() {
        let base = (c.commitment * blocks + c.block) * leaf_arity.max(1);
        for (k, p) in c.point.iter().enumerate() {
            pts[base + k] = *p;
        }
        scale[c.commitment * blocks + c.block] = query_tags[j];
    }

    let words = size / 64;
    let splits = shard_split(n_commit, devices.len());

    // Allocate and upload each shard on its own device.
    let mut shards: Vec<Shard> = splits
        .iter()
        .enumerate()
        .map(|(i, &(start, len))| {
            let device = devices[i % devices.len()];
            almost_goldilocks_cuda::set_device(device).ok()?;
            let mut b = DeviceBuffer::<u64>::new(len * words).ok()?;
            for e in 0..len {
                b.copy_from_slice_at(e * words, &bits[start + e]).ok()?;
            }
            let w = DeviceBuffer::<u64>::new(2 * len * (size / 2)).ok()?;
            let pt_slice = &pts[start * blocks * leaf_arity.max(1)
                ..(start + len) * blocks * leaf_arity.max(1)];
            let p = DeviceBuffer::<u64>::from_slice(&flatten(pt_slice)).ok()?;
            let t = DeviceBuffer::<u64>::from_slice(&flatten(&commit_tags[start..start + len]))
                .ok()?;
            let partial =
                DeviceBuffer::<u64>::new(2 * len * 64 * (NORM_EVALS + EVAL_EVALS)).ok()?;
            let out = DeviceBuffer::<u64>::new(2 * (NORM_EVALS + EVAL_EVALS)).ok()?;
            let prev: Vec<Vec<u32>> = (0..len)
                .map(|e| {
                    let mut v = Vec::new();
                    for (wi, word) in bits[start + e].iter().enumerate() {
                        let mut m = *word;
                        while m != 0 {
                            let z = m.trailing_zeros() as usize;
                            m &= m - 1;
                            v.push((wi * 64 + z) as u32);
                        }
                    }
                    v
                })
                .collect();
            Some(Shard { device, start, len, bits: b, w, pts: p, tags: t, partial, out, prev })
        })
        .collect::<Option<Vec<_>>>()?;

    let mut rounds = Vec::with_capacity(ambient);
    let mut xi = Vec::with_capacity(ambient);
    let mut half = size >> 1;
    let mut cur_scale = scale.clone();
    let mut src_is_bits = true;
    let w_stride_fixed = size / 2;

    for i in 0..leaf_arity {
        let alpha_suffix: Vec<u64> = {
            let mut v = Vec::new();
            for r in &alpha[i + 1..] {
                let l = ext2_to_limbs(*r);
                v.push(l[0]);
                v.push(l[1]);
            }
            if v.is_empty() {
                v.extend_from_slice(&[0, 0]);
            }
            v
        };

        // Each device computes its own partial message over its own commitments.
        let partials: Vec<Option<(Vec<Ext2>, Vec<Vec<u32>>)>> = shards
            .par_iter_mut()
            .map(|sh| {
                almost_goldilocks_cuda::set_device(sh.device).ok()?;
                let mut cat: Vec<u32> = Vec::new();
                let mut offs = Vec::with_capacity(sh.len);
                let mut lens = Vec::with_capacity(sh.len);
                let mut next: Vec<Vec<u32>> = Vec::with_capacity(sh.len);
                let mut seen = vec![false; half.max(1)];
                for sup in &sh.prev {
                    offs.push(cat.len() as u64);
                    let st = cat.len();
                    for &p in sup {
                        let q = (p as usize) & (half - 1);
                        if !seen[q] {
                            seen[q] = true;
                            cat.push(q as u32);
                        }
                    }
                    for &v in &cat[st..] {
                        seen[v as usize] = false;
                    }
                    lens.push((cat.len() - st) as u64);
                    next.push(cat[st..].to_vec());
                }

                let d_list = DeviceBuffer::<u32>::from_slice(&cat).ok()?;
                let d_off = DeviceBuffer::<u64>::from_slice(&offs).ok()?;
                let d_len = DeviceBuffer::<u64>::from_slice(&lens).ok()?;
                let sc = &cur_scale[sh.start * blocks..(sh.start + sh.len) * blocks];
                let d_scale = DeviceBuffer::<u64>::from_slice(&flatten(sc)).ok()?;
                let d_alpha = DeviceBuffer::<u64>::from_slice(&alpha_suffix).ok()?;
                let mut d_eq = DeviceBuffer::<u64>::new(2 * cat.len().max(1)).ok()?;
                let rc = unsafe {
                    almost_goldilocks_cuda::ffi::link_eq_gather_ffi(
                        d_alpha.as_ptr(), (ambient - i - 1) as u64, d_list.as_ptr(),
                        d_eq.as_mut_ptr(), cat.len() as u64,
                    )
                };
                if rc != 0 {
                    return None;
                }
                let max_len = lens.iter().copied().max().unwrap_or(1) as usize;
                let chunks = chunks_for_work(max_len);
                let rc = unsafe {
                    almost_goldilocks_cuda::ffi::link_round_interleaved_ffi(
                        sh.bits.as_ptr(),
                        if src_is_bits { std::ptr::null() } else { sh.w.as_ptr() },
                        sh.pts.as_ptr(), d_scale.as_ptr(), d_eq.as_ptr(), sh.tags.as_ptr(),
                        d_list.as_ptr(), d_off.as_ptr(), d_len.as_ptr(),
                        w_stride_fixed as u64, words as u64, half as u64, sh.len as u64,
                        (blocks - 1) as u32, block_bits as i32, leaf_arity as i32,
                        i as i32, i32::from(src_is_bits),
                        sh.partial.as_mut_ptr(), chunks as u64, sh.out.as_mut_ptr(),
                    )
                };
                if rc != 0 {
                    return None;
                }
                let raw = sh.out.to_vec().ok()?;
                let vals: Vec<Ext2> = (0..NORM_EVALS + EVAL_EVALS)
                    .map(|k| limbs_to_ext2(&raw[2 * k..]))
                    .collect();
                Some((vals, next))
            })
            .collect();

        // Fixed-order sum: the proof must not depend on device count.
        let mut acc = vec![ext2_zero(); NORM_EVALS + EVAL_EVALS];
        for p in &partials {
            let (vals, _) = p.as_ref()?;
            for (a, v) in acc.iter_mut().zip(vals.iter()) {
                *a = ext2_add(*a, *v);
            }
        }
        for (sh, p) in shards.iter_mut().zip(partials.into_iter()) {
            sh.prev = p?.1;
        }

        let msg = RoundMsg {
            norm: acc[..NORM_EVALS].to_vec(),
            eval: acc[NORM_EVALS..].to_vec(),
        };
        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        xi.push(r);

        let l = ext2_to_limbs(r);
        let ok: Option<Vec<()>> = shards
            .par_iter_mut()
            .map(|sh| {
                almost_goldilocks_cuda::set_device(sh.device).ok()?;
                let rc = unsafe {
                    almost_goldilocks_cuda::ffi::link_fold_w_ffi(
                        sh.bits.as_ptr(),
                        if src_is_bits { std::ptr::null() } else { sh.w.as_ptr() },
                        sh.w.as_mut_ptr(), w_stride_fixed as u64, words as u64,
                        w_stride_fixed as u64, half as u64, sh.len as u64,
                        i32::from(src_is_bits), l[0], l[1],
                    )
                };
                if rc != 0 {
                    return None;
                }
                Some(())
            })
            .collect();
        ok?;

        for e in 0..n_commit {
            for b in 0..blocks {
                let idx = e * blocks + b;
                let p = pts[idx * leaf_arity.max(1) + i];
                cur_scale[idx] = ext2_mul(cur_scale[idx], eq_scalar(r, p));
            }
        }

        rounds.push(msg);
        src_is_bits = false;
        half >>= 1;
    }

    // Block phase on the host: the surviving domain is one value per block.
    let mut w_small: Vec<Vec<Ext2>> = vec![Vec::new(); n_commit];
    for sh in &shards {
        almost_goldilocks_cuda::set_device(sh.device).ok()?;
        let mut host = vec![0u64; 2 * sh.len * blocks];
        for e in 0..sh.len {
            sh.w
                .copy_to_slice_at(2 * e * w_stride_fixed, &mut host[2 * e * blocks..2 * (e + 1) * blocks])
                .ok()?;
        }
        for e in 0..sh.len {
            w_small[sh.start + e] = (0..blocks)
                .map(|b| limbs_to_ext2(&host[2 * (e * blocks + b)..]))
                .collect();
        }
    }
    let mut o_small: Vec<Vec<Ext2>> = (0..n_commit)
        .map(|e| (0..blocks).map(|b| cur_scale[e * blocks + b]).collect())
        .collect();

    let (block_rounds, term) = block_phase(
        &mut w_small, &mut o_small, &commit_tags, &alpha, leaf_arity, ambient, blocks, transcript,
    );
    rounds.extend(block_rounds.0);
    xi.extend(block_rounds.1);

    for a in &term {
        transcript.append_ext2(b"link-terminal", a);
    }
    Some((LinkProof { rounds, terminal: term }, xi))
}

/// The rounds after the leaf variables are exhausted, shared by the single- and
/// multi-device provers. The domain here is one value per block, so it runs on
/// the host: shipping it to a GPU would cost more than the arithmetic.
#[allow(clippy::too_many_arguments)]
fn block_phase(
    w_small: &mut [Vec<Ext2>],
    o_small: &mut [Vec<Ext2>],
    commit_tags: &[Ext2],
    alpha: &[Ext2],
    leaf_arity: usize,
    ambient: usize,
    blocks: usize,
    transcript: &mut Transcript,
) -> ((Vec<RoundMsg>, Vec<Ext2>), Vec<Ext2>) {
    let n_commit = w_small.len();
    let mut rounds = Vec::new();
    let mut xi = Vec::new();
    let mut hb = blocks >> 1;
    for i in leaf_arity..ambient {
        let eq_suffix = eq_table(&alpha[i + 1..]);
        let mut norm = vec![ext2_zero(); NORM_EVALS];
        let mut eval = vec![ext2_zero(); EVAL_EVALS];
        for e in 0..n_commit {
            let mut ne = [ext2_zero(); NORM_EVALS];
            for idx in 0..hb {
                let w_lo = w_small[e][idx];
                let w_d = ext2_sub(w_small[e][hb + idx], w_lo);
                let o_lo = o_small[e][idx];
                let o_d = ext2_sub(o_small[e][hb + idx], o_lo);
                let eqs = eq_suffix[idx];
                let w0 = w_lo;
                let w1 = ext2_add(w0, w_d);
                let w2 = ext2_add(w1, w_d);
                let w3 = ext2_add(w2, w_d);
                ne[0] = ext2_add(ne[0], ext2_mul(eqs, range_poly(w0)));
                ne[1] = ext2_add(ne[1], ext2_mul(eqs, range_poly(w1)));
                ne[2] = ext2_add(ne[2], ext2_mul(eqs, range_poly(w2)));
                ne[3] = ext2_add(ne[3], ext2_mul(eqs, range_poly(w3)));
                let o0 = o_lo;
                let o1 = ext2_add(o0, o_d);
                let o2 = ext2_add(o1, o_d);
                eval[0] = ext2_add(eval[0], ext2_mul(o0, w0));
                eval[1] = ext2_add(eval[1], ext2_mul(o1, w1));
                eval[2] = ext2_add(eval[2], ext2_mul(o2, w2));
            }
            for k in 0..NORM_EVALS {
                norm[k] = ext2_add(norm[k], ext2_mul(commit_tags[e], ne[k]));
            }
        }
        let msg = RoundMsg { norm, eval };
        for v in msg.norm.iter().chain(msg.eval.iter()) {
            transcript.append_ext2(b"link-round", v);
        }
        let r = transcript.challenge_ext2(b"link-challenge");
        xi.push(r);
        for e in 0..n_commit {
            for idx in 0..hb {
                let wl = w_small[e][idx];
                w_small[e][idx] = ext2_add(wl, ext2_mul(r, ext2_sub(w_small[e][hb + idx], wl)));
                let ol = o_small[e][idx];
                o_small[e][idx] = ext2_add(ol, ext2_mul(r, ext2_sub(o_small[e][hb + idx], ol)));
            }
            w_small[e].truncate(hb.max(1));
            o_small[e].truncate(hb.max(1));
        }
        rounds.push(msg);
        hb = (hb >> 1).max(if i + 1 < ambient { 1 } else { 0 });
    }
    let terminal: Vec<Ext2> = (0..n_commit).map(|e| w_small[e][0]).collect();
    ((rounds, xi), terminal)
}

/// Time the GPU round kernel at a given witness density.
///
/// Round 0 is half the total sumcheck work and the only round at the original
/// density, so it is the honest unit to report: the whole sumcheck costs about
/// twice this at density 1, and less than that when support skipping bites in
/// the early rounds.
///
/// `spacing` controls the sparsity pattern. One-hot Shout advice puts exactly
/// one nonzero every `2^table_commit_log` slots, so the pattern matters as much
/// as the density: a warp retires early only when all 32 of its lanes are zero.
pub fn bench_gpu_round(n_commit: usize, arity: usize, spacing: usize) -> Option<f64> {
    use std::time::Instant;
    let size = 1usize << arity;

    let mut seed = 0x243F6A8885A308D3u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let witnesses: Vec<LinkWitness> = (0..n_commit)
        .map(|_| {
            let coeffs: Vec<u64> = (0..size)
                .map(|i| if spacing <= 1 || i % spacing == 0 { next() & 1 } else { 0 })
                .collect();
            let mut bits = vec![0u64; size / 64];
            for (i, v) in coeffs.iter().enumerate() {
                if *v == 1 { bits[i / 64] |= 1u64 << (i % 64); }
            }
            LinkWitness::binary(coeffs, bits)
        })
        .collect();
    // Realistic query shape: each commitment packs 2^PREFIX leaves, and every
    // leaf carries its own claim at a Boolean-prefixed point. That is what makes
    // the query-weight construction O(total witness size) instead of
    // O(queries x ambient).
    const PREFIX: usize = 4;
    let mut queries: Vec<LinkQuery> = Vec::new();
    for (e, w) in witnesses.iter().enumerate() {
        for blk in 0..(1usize << PREFIX) {
            let mut point: Vec<Ext2> = (0..PREFIX)
                .map(|k| {
                    let bit = (blk >> (PREFIX - 1 - k)) & 1;
                    Ext2::new(AlmostGoldilocksField(bit as u64), AlmostGoldilocksField(0))
                })
                .collect();
            point.extend((0..arity - PREFIX).map(|_| {
                Ext2::new(
                    AlmostGoldilocksField(next() >> 4),
                    AlmostGoldilocksField(next() >> 4),
                )
            }));
            let value = mle_eval_for_bench(&w.coeffs, &point);
            queries.push(LinkQuery { commitment: e, point, value, prefix_len: PREFIX });
        }
    }

    // warm up (allocations, context)
    let mut t0 = Transcript::new(b"warm");
    prove_link_gpu(&witnesses, &queries, arity, &mut t0)?;

    let t = Instant::now();
    let mut tr = Transcript::new(b"bench");
    let out = prove_link_gpu(&witnesses, &queries, arity, &mut tr)?;
    let secs = t.elapsed().as_secs_f64();
    std::hint::black_box(out);

    let _ = secs;
    // Round kernels only; see LAST_ROUND_NANOS.
    let round_ns = LAST_ROUND_NANOS.load(Ordering::Relaxed) as f64;
    Some(round_ns / ((n_commit * size) as f64))
}

/// Time one optimized round message against the same loop with the norm half
/// removed, on identical tables.
///
/// The projection uses the measured GPU same-point sumcheck (degree 2) as its
/// link baseline, so what it needs is the *incremental* cost of proving the norm
/// bound. Both sides here carry the same optimizations — incremental
/// interpolation, hoisted scalars, support skipping — so the ratio measures the
/// degree penalty rather than implementation quality.
///
/// Returns `(full_ns_per_coeff, eval_only_ns_per_coeff)`.
pub fn bench_round_halves(n_commit: usize, arity: usize) -> (f64, f64) {
    use std::time::Instant;
    let size = 1usize << arity;
    let half = size >> 1;

    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut rnd = || Ext2::new(
        AlmostGoldilocksField(next() >> 4),
        AlmostGoldilocksField(next() >> 4),
    );

    let tables: Vec<Tables> = (0..n_commit)
        .map(|_| Tables {
            w: (0..size).map(|_| rnd()).collect(),
            omega: (0..size).map(|_| rnd()).collect(),
        })
        .collect();
    let eq_suffix: Vec<Ext2> = (0..half).map(|_| rnd()).collect();
    let commit_tags: Vec<Ext2> = (0..n_commit).map(|_| rnd()).collect();

    let coeffs = (n_commit * size) as f64;

    let t = Instant::now();
    let full = round_message_partial(&tables, 0..n_commit, &eq_suffix, &commit_tags, half, false);
    let full_ns = t.elapsed().as_secs_f64() * 1e9 / coeffs;
    std::hint::black_box(full);

    // Evaluation half only: degree 2, three points, same optimizations.
    let t = Instant::now();
    let mut acc = [ext2_zero(); EVAL_EVALS];
    for e in 0..n_commit {
        let tb = &tables[e];
        for idx in 0..half {
            let w_lo = tb.w[idx];
            let w_hi = tb.w[half + idx];
            if is_zero(w_lo) && is_zero(w_hi) {
                continue;
            }
            let w_d = ext2_sub(w_hi, w_lo);
            let o_lo = tb.omega[idx];
            let o_d = ext2_sub(tb.omega[half + idx], o_lo);
            let w1 = ext2_add(w_lo, w_d);
            let w2 = ext2_add(w1, w_d);
            let o1 = ext2_add(o_lo, o_d);
            let o2 = ext2_add(o1, o_d);
            acc[0] = ext2_add(acc[0], ext2_mul(o_lo, w_lo));
            acc[1] = ext2_add(acc[1], ext2_mul(o1, w1));
            acc[2] = ext2_add(acc[2], ext2_mul(o2, w2));
        }
    }
    let eval_ns = t.elapsed().as_secs_f64() * 1e9 / coeffs;
    std::hint::black_box(acc);

    (full_ns, eval_ns)
}

/// Cost of one round message at a given witness density, to show how far
/// support skipping carries. `density` is the fraction of nonzero entries.
pub fn bench_round_density(n_commit: usize, arity: usize, density: f64) -> f64 {
    use std::time::Instant;
    let size = 1usize << arity;
    let half = size >> 1;
    let mut seed = 0x243F6A8885A308D3u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let thresh = (density * (u32::MAX as f64)) as u64;
    let tables: Vec<Tables> = (0..n_commit)
        .map(|_| Tables {
            w: (0..size)
                .map(|_| {
                    if (next() >> 32) < thresh {
                        Ext2::new(AlmostGoldilocksField(next() >> 4), AlmostGoldilocksField(0))
                    } else {
                        ext2_zero()
                    }
                })
                .collect(),
            omega: (0..size)
                .map(|_| Ext2::new(AlmostGoldilocksField(next() >> 4), AlmostGoldilocksField(0)))
                .collect(),
        })
        .collect();
    let eq_suffix: Vec<Ext2> = (0..half)
        .map(|_| Ext2::new(AlmostGoldilocksField(next() >> 4), AlmostGoldilocksField(0)))
        .collect();
    let commit_tags: Vec<Ext2> = (0..n_commit)
        .map(|_| Ext2::new(AlmostGoldilocksField(next() >> 4), AlmostGoldilocksField(0)))
        .collect();

    let t = Instant::now();
    let out = round_message_partial(&tables, 0..n_commit, &eq_suffix, &commit_tags, half, false);
    let ns = t.elapsed().as_secs_f64() * 1e9 / ((n_commit * size) as f64);
    std::hint::black_box(out);
    ns
}

/// MLE evaluation helper, exposed for the microbenchmark so it can build honest
/// claim values without duplicating the eq convention.
pub fn mle_eval_for_bench(coeffs: &[u64], point: &[Ext2]) -> Ext2 {
    let eqt = eq_table(point);
    let mut acc = ext2_zero();
    for (i, c) in coeffs.iter().enumerate() {
        acc = ext2_add(acc, ext2_mul(eqt[i], ext2_from_u64(*c)));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn ext2(&mut self) -> Ext2 {
            Ext2::new(
                AlmostGoldilocksField(self.next() >> 4),
                AlmostGoldilocksField(self.next() >> 4),
            )
        }
    }

    /// Direct MLE evaluation, independent of the sumcheck machinery.
    fn mle_eval(coeffs: &[u64], point: &[Ext2]) -> Ext2 {
        let eqt = eq_table(point);
        let mut acc = ext2_zero();
        for (i, c) in coeffs.iter().enumerate() {
            acc = ext2_add(acc, ext2_mul(eqt[i], ext2_from_u64(*c)));
        }
        acc
    }

    fn binary_witness(arity: usize, rng: &mut Rng) -> LinkWitness {
        LinkWitness::dense((0..(1usize << arity)).map(|_| rng.next() & 1).collect())
    }

    /// Build queries whose points carry a Boolean block prefix, matching how a
    /// packed leaf's claim is lifted to the ambient domain.
    fn packed_query(
        commitment: usize,
        w: &LinkWitness,
        block_index: usize,
        block_arity: usize,
        ambient_arity: usize,
        rng: &mut Rng,
    ) -> LinkQuery {
        let prefix_len = ambient_arity - block_arity;
        let mut point: Vec<Ext2> = (0..prefix_len)
            .map(|k| {
                let bit = (block_index >> (prefix_len - 1 - k)) & 1;
                ext2_from_u64(bit as u64)
            })
            .collect();
        point.extend((0..block_arity).map(|_| rng.ext2()));
        let value = mle_eval(&w.coeffs, &point);
        LinkQuery { commitment, point, value, prefix_len }
    }

    #[test]
    fn range_polynomial_vanishes_exactly_on_the_allowed_set() {
        for v in [0u64, 1] {
            assert!(ext2_field_eq(range_poly(ext2_from_u64(v)), ext2_zero()));
        }
        // -1 mod q
        let q_minus_1 = ((1u128 << 64) - (1u128 << 32) + 1 - 32 - 1) as u64;
        assert!(ext2_field_eq(range_poly(ext2_from_u64(q_minus_1)), ext2_zero()));
        // 2 is out of range and must not vanish
        assert!(!ext2_field_eq(range_poly(ext2_from_u64(2)), ext2_zero()));
    }

    #[test]
    fn honest_link_verifies_and_returns_same_point_claims() {
        let mut rng = Rng(0xA5A5_1234_5678_9ABC);
        let ambient = 8usize;
        let ws: Vec<LinkWitness> = (0..3).map(|_| binary_witness(ambient, &mut rng)).collect();
        let mut qs = Vec::new();
        for (e, w) in ws.iter().enumerate() {
            qs.push(packed_query(e, w, 1, ambient - 2, ambient, &mut rng));
            qs.push(packed_query(e, w, 2, ambient - 2, ambient, &mut rng));
        }

        let mut tp = Transcript::new(b"link-test");
        let (proof, xi) = prove_link(&ws, &qs, ambient, 1, &mut tp);

        let mut tv = Transcript::new(b"link-test");
        let out = verify_link(ws.len(), &qs, ambient, &proof, &mut tv);
        let (xi_v, terminal) = out.expect("honest link must verify");

        assert_eq!(xi_v, xi, "verifier must derive the same ξ");
        for (e, w) in ws.iter().enumerate() {
            assert!(
                ext2_field_eq(terminal[e], mle_eval(&w.coeffs, &xi)),
                "terminal a_e must be the witness evaluated at ξ"
            );
        }
    }

    #[test]
    fn sharding_does_not_change_the_proof() {
        let mut rng = Rng(0x0BAD_C0DE_0BAD_C0DE);
        let ambient = 7usize;
        let ws: Vec<LinkWitness> = (0..7).map(|_| binary_witness(ambient, &mut rng)).collect();
        let qs: Vec<LinkQuery> = ws
            .iter()
            .enumerate()
            .map(|(e, w)| packed_query(e, w, 0, ambient - 1, ambient, &mut rng))
            .collect();

        let mut base = None;
        for shards in [1usize, 2, 3, 7] {
            let mut t = Transcript::new(b"shard");
            let (proof, _) = prove_link(&ws, &qs, ambient, shards, &mut t);
            let flat: Vec<Ext2> = proof
                .rounds
                .iter()
                .flat_map(|m| m.norm.iter().chain(m.eval.iter()).cloned())
                .chain(proof.terminal.iter().cloned())
                .collect();
            match &base {
                None => base = Some(flat),
                Some(b) => {
                    assert_eq!(b.len(), flat.len());
                    for (x, y) in b.iter().zip(flat.iter()) {
                        assert!(ext2_field_eq(*x, *y), "shard count changed the proof");
                    }
                }
            }
        }
    }

    #[test]
    fn out_of_range_coefficient_is_rejected() {
        let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
        let ambient = 7usize;
        let mut ws: Vec<LinkWitness> = (0..2).map(|_| binary_witness(ambient, &mut rng)).collect();
        // 2 is outside {-1, 0, 1}: R_Bx does not vanish, so the norm half is
        // nonzero and the sumcheck's claimed total no longer holds.
        ws[1].coeffs[5] = 2;
        let qs: Vec<LinkQuery> = ws
            .iter()
            .enumerate()
            .map(|(e, w)| packed_query(e, w, 0, ambient, ambient, &mut rng))
            .collect();

        let mut tp = Transcript::new(b"oor");
        let (proof, _) = prove_link(&ws, &qs, ambient, 1, &mut tp);
        let mut tv = Transcript::new(b"oor");
        assert!(
            verify_link(ws.len(), &qs, ambient, &proof, &mut tv).is_none(),
            "an out-of-range coefficient must be rejected"
        );
    }

    #[test]
    fn wrong_claimed_value_is_rejected() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        let ambient = 6usize;
        let ws: Vec<LinkWitness> = (0..2).map(|_| binary_witness(ambient, &mut rng)).collect();
        let mut qs: Vec<LinkQuery> = ws
            .iter()
            .enumerate()
            .map(|(e, w)| packed_query(e, w, 0, ambient, ambient, &mut rng))
            .collect();

        let mut tp = Transcript::new(b"bad-y");
        let (proof, _) = prove_link(&ws, &qs, ambient, 1, &mut tp);
        // Prove honestly, then claim a different value: the verifier's initial
        // claim no longer matches the transcript the prover committed to.
        qs[0].value = ext2_add(qs[0].value, ext2_one());
        let mut tv = Transcript::new(b"bad-y");
        assert!(verify_link(ws.len(), &qs, ambient, &proof, &mut tv).is_none());
    }

    #[test]
    fn tampered_terminal_value_is_rejected() {
        let mut rng = Rng(0x0F0F_0F0F_0F0F_0F0F);
        let ambient = 6usize;
        let ws: Vec<LinkWitness> = (0..2).map(|_| binary_witness(ambient, &mut rng)).collect();
        let qs: Vec<LinkQuery> = ws
            .iter()
            .enumerate()
            .map(|(e, w)| packed_query(e, w, 0, ambient, ambient, &mut rng))
            .collect();

        let mut tp = Transcript::new(b"bad-a");
        let (mut proof, _) = prove_link(&ws, &qs, ambient, 1, &mut tp);
        proof.terminal[1] = ext2_add(proof.terminal[1], ext2_one());
        let mut tv = Transcript::new(b"bad-a");
        assert!(
            verify_link(ws.len(), &qs, ambient, &proof, &mut tv).is_none(),
            "a_e feeds the masked RLC; it must be bound by the link"
        );
    }

    #[test]
    fn hiding_block_of_ternary_symbols_passes_the_range_check() {
        // -1 is encoded as q-1; the range polynomial must accept it, otherwise
        // the hiding block could not live inside the committed witness.
        let mut rng = Rng(0xC0FF_EE00_C0FF_EE00);
        let ambient = 6usize;
        let q_minus_1 = ((1u128 << 64) - (1u128 << 32) + 1 - 32 - 1) as u64;
        let mut w = binary_witness(ambient, &mut rng);
        for i in 0..8 {
            w.coeffs[i] = q_minus_1;
        }
        let ws = vec![w];
        let qs = vec![packed_query(0, &ws[0], 0, ambient, ambient, &mut rng)];

        let mut tp = Transcript::new(b"ternary");
        let (proof, _) = prove_link(&ws, &qs, ambient, 1, &mut tp);
        let mut tv = Transcript::new(b"ternary");
        assert!(verify_link(1, &qs, ambient, &proof, &mut tv).is_some());
    }

    #[test]
    fn gpu_link_proof_is_identical_to_cpu() {
        // The GPU path drives the same host transcript and every kernel is a
        // pure function of device state, so any divergence is a real bug rather
        // than a scheduling artifact. Comparing whole proofs (not just the
        // verdict) is what makes that check sharp.
        let mut rng = Rng(0xC0DE_C0DE_1234_5678);
        let ambient = 12usize;
        let ws: Vec<LinkWitness> = (0..5)
            .map(|_| LinkWitness::dense((0..(1usize << ambient)).map(|_| rng.next() & 1).collect()))
            .collect();
        let mut qs = Vec::new();
        for (e, w) in ws.iter().enumerate() {
            qs.push(packed_query(e, w, 1, ambient - 2, ambient, &mut rng));
            qs.push(packed_query(e, w, 3, ambient - 2, ambient, &mut rng));
        }

        let mut tc = Transcript::new(b"gpu-vs-cpu");
        let (cpu, xi_cpu) = prove_link(&ws, &qs, ambient, 1, &mut tc);
        let mut tg = Transcript::new(b"gpu-vs-cpu");
        let (gpu, xi_gpu) = prove_link_gpu(&ws, &qs, ambient, &mut tg).expect("gpu path");

        assert_eq!(cpu.rounds.len(), gpu.rounds.len());
        for (r, (a, b)) in cpu.rounds.iter().zip(gpu.rounds.iter()).enumerate() {
            for k in 0..NORM_EVALS {
                assert!(ext2_field_eq(a.norm[k], b.norm[k]), "round {} norm[{}]", r, k);
            }
            for k in 0..EVAL_EVALS {
                assert!(ext2_field_eq(a.eval[k], b.eval[k]), "round {} eval[{}]", r, k);
            }
        }
        for (e, (a, b)) in cpu.terminal.iter().zip(gpu.terminal.iter()).enumerate() {
            assert!(ext2_field_eq(*a, *b), "terminal {}", e);
        }
        for (a, b) in xi_cpu.iter().zip(xi_gpu.iter()) {
            assert!(ext2_field_eq(*a, *b));
        }

        // and the GPU proof must verify under the ordinary verifier
        let mut tv = Transcript::new(b"gpu-vs-cpu");
        assert!(verify_link(ws.len(), &qs, ambient, &gpu, &mut tv).is_some());
    }

    #[test]
    fn multi_device_link_is_independent_of_device_count() {
        // Sharding must be a scheduling choice, not a protocol one: the round
        // message is a sum over commitments, so any partition of them has to
        // give the identical proof. Summing the partials in a fixed order is
        // what makes that true rather than approximately true.
        let mut rng = Rng(0x5151_2727_5151_2727);
        let leaf_arity = 7usize;
        let block_bits = 2usize;
        let ambient = leaf_arity + block_bits;
        let blocks = 1usize << block_bits;
        let n_commit = 4usize;
        let size = 1usize << ambient;

        let mut bits = vec![vec![0u64; size / 64]; n_commit];
        let mut claims = Vec::new();
        for e in 0..n_commit {
            for b in 0..(blocks - 1) {
                let leaf: Vec<u64> =
                    (0..(1usize << leaf_arity)).map(|_| rng.next() & 1).collect();
                for (l, v) in leaf.iter().enumerate() {
                    if *v == 1 {
                        let p = (l << block_bits) | b;
                        bits[e][p / 64] |= 1u64 << (p % 64);
                    }
                }
                let point: Vec<Ext2> = (0..leaf_arity).map(|_| rng.ext2()).collect();
                let eqt = eq_table(&point);
                let mut value = ext2_zero();
                for (l, v) in leaf.iter().enumerate() {
                    value = ext2_add(value, ext2_mul(eqt[l], ext2_from_u64(*v)));
                }
                claims.push(BlockClaim { commitment: e, block: b, point, value });
            }
        }

        let devs = almost_goldilocks_cuda::device_count().max(1);
        let mut baseline: Option<Vec<Ext2>> = None;
        for k in 1..=(devs.min(4) as usize) {
            let devices: Vec<i32> = (0..k as i32).collect();
            let mut t = Transcript::new(b"multi");
            let (proof, _) = prove_link_interleaved_multi(
                &bits, &claims, leaf_arity, block_bits, &devices, &mut t,
            )
            .expect("multi-device prove");
            let flat: Vec<Ext2> = proof
                .rounds
                .iter()
                .flat_map(|m| m.norm.iter().chain(m.eval.iter()).cloned())
                .chain(proof.terminal.iter().cloned())
                .collect();
            match &baseline {
                None => baseline = Some(flat),
                Some(b) => {
                    assert_eq!(b.len(), flat.len());
                    for (x, y) in b.iter().zip(flat.iter()) {
                        assert!(ext2_field_eq(*x, *y), "{} devices changed the proof", k);
                    }
                }
            }
        }
        // and it must agree with the single-device interleaved prover
        let mut t1 = Transcript::new(b"multi");
        let mut sc = LinkScratch::new();
        let (single, _) =
            prove_link_interleaved(&bits, &claims, leaf_arity, block_bits, &mut t1, &mut sc)
                .expect("single");
        let flat1: Vec<Ext2> = single
            .rounds
            .iter()
            .flat_map(|m| m.norm.iter().chain(m.eval.iter()).cloned())
            .chain(single.terminal.iter().cloned())
            .collect();
        for (x, y) in baseline.unwrap().iter().zip(flat1.iter()) {
            assert!(ext2_field_eq(*x, *y), "multi-device diverged from single-device");
        }
    }

    #[test]
    fn interleaved_path_matches_the_generic_cpu_reference() {
        // The interleaved prover never materializes a weight table; the generic
        // CPU path does. Comparing whole proofs on the same witness is what
        // pins that the pointwise weight evaluation is exact rather than close.
        let mut rng = Rng(0xABCD_1234_ABCD_1234);
        let leaf_arity = 8usize;
        let block_bits = 3usize;
        let ambient = leaf_arity + block_bits;
        let blocks = 1usize << block_bits;
        let n_commit = 3usize;
        let size = 1usize << ambient;

        // Interleaved witness: leaf b's coefficient l sits at (l << block_bits) | b.
        // Leave the top block empty, standing in for the hiding block.
        let leaves: Vec<Vec<u64>> = (0..n_commit * (blocks - 1))
            .map(|_| (0..(1usize << leaf_arity)).map(|_| rng.next() & 1).collect::<Vec<u64>>())
            .collect();
        let mut coeffs = vec![vec![0u64; size]; n_commit];
        let mut bits = vec![vec![0u64; size / 64]; n_commit];
        let mut claims = Vec::new();
        for (i, leaf) in leaves.iter().enumerate() {
            let e = i / (blocks - 1);
            let b = i % (blocks - 1);
            for (l, v) in leaf.iter().enumerate() {
                let p = (l << block_bits) | b;
                coeffs[e][p] = *v;
                if *v == 1 {
                    bits[e][p / 64] |= 1u64 << (p % 64);
                }
            }
            let point: Vec<Ext2> = (0..leaf_arity).map(|_| rng.ext2()).collect();
            // MLE of the leaf at `point`, MSB-first, matching eq_table.
            let eqt = eq_table(&point);
            let mut value = ext2_zero();
            for (l, v) in leaf.iter().enumerate() {
                value = ext2_add(value, ext2_mul(eqt[l], ext2_from_u64(*v)));
            }
            claims.push(BlockClaim { commitment: e, block: b, point, value });
        }

        // Generic path over the same packed witness, with the block bits as the
        // point's suffix (they are the low index bits under interleaving).
        let ws: Vec<LinkWitness> = coeffs.iter().map(|c| LinkWitness::dense(c.clone())).collect();
        let qs: Vec<LinkQuery> = claims
            .iter()
            .map(|c| {
                let mut point = c.point.clone();
                for k in 0..block_bits {
                    let bit = (c.block >> (block_bits - 1 - k)) & 1;
                    point.push(ext2_from_u64(bit as u64));
                }
                LinkQuery { commitment: c.commitment, point, value: c.value, prefix_len: 0 }
            })
            .collect();

        let mut tc = Transcript::new(b"interleaved");
        let (cpu, _) = prove_link(&ws, &qs, ambient, 1, &mut tc);
        let mut tg = Transcript::new(b"interleaved");
        let mut scratch = LinkScratch::new();
        let (gpu, _) = prove_link_interleaved(
            &bits, &claims, leaf_arity, block_bits, &mut tg, &mut scratch,
        )
        .expect("interleaved prove");

        assert_eq!(cpu.rounds.len(), gpu.rounds.len());
        for (r, (a, b)) in cpu.rounds.iter().zip(gpu.rounds.iter()).enumerate() {
            for k in 0..NORM_EVALS {
                assert!(ext2_field_eq(a.norm[k], b.norm[k]), "round {} norm[{}]", r, k);
            }
            for k in 0..EVAL_EVALS {
                assert!(ext2_field_eq(a.eval[k], b.eval[k]), "round {} eval[{}]", r, k);
            }
        }
        for (e, (a, b)) in cpu.terminal.iter().zip(gpu.terminal.iter()).enumerate() {
            assert!(ext2_field_eq(*a, *b), "terminal {}", e);
        }

        let mut tv = Transcript::new(b"interleaved");
        assert!(verify_link(n_commit, &qs, ambient, &gpu, &mut tv).is_some());
    }

    #[test]
    fn gpu_sparse_list_path_matches_cpu() {
        // One-hot shaped witness (one nonzero every 64 slots) so the support
        // lists are short enough to be taken — the previous GPU test is ~50%
        // dense and falls back to the dense kernel, so it does not cover this.
        let mut rng = Rng(0x1BADB002_1BADB002);
        let ambient = 14usize;
        let ws: Vec<LinkWitness> = (0..4)
            .map(|_| LinkWitness::dense(
                (0..(1usize << ambient))
                    .map(|i| if i % 64 == 0 { rng.next() & 1 } else { 0 })
                    .collect(),
            ))
            .collect();
        let qs: Vec<LinkQuery> = ws
            .iter()
            .enumerate()
            .map(|(e, w)| packed_query(e, w, 0, ambient - 2, ambient, &mut rng))
            .collect();

        let mut tc = Transcript::new(b"sparse-gpu");
        let (cpu, _) = prove_link(&ws, &qs, ambient, 1, &mut tc);
        let mut tg = Transcript::new(b"sparse-gpu");
        let (gpu, _) = prove_link_gpu(&ws, &qs, ambient, &mut tg).expect("gpu");

        for (r, (a, b)) in cpu.rounds.iter().zip(gpu.rounds.iter()).enumerate() {
            for k in 0..NORM_EVALS {
                assert!(ext2_field_eq(a.norm[k], b.norm[k]), "round {} norm[{}]", r, k);
            }
            for k in 0..EVAL_EVALS {
                assert!(ext2_field_eq(a.eval[k], b.eval[k]), "round {} eval[{}]", r, k);
            }
        }
        for (a, b) in cpu.terminal.iter().zip(gpu.terminal.iter()) {
            assert!(ext2_field_eq(*a, *b));
        }
        let mut tv = Transcript::new(b"sparse-gpu");
        assert!(verify_link(ws.len(), &qs, ambient, &gpu, &mut tv).is_some());
    }

    #[test]
    fn sparse_witness_verifies_so_the_support_skip_is_exact() {
        // The inner loop skips an index when a witness entry and its partner are
        // both zero, on the grounds that R(0) = 0 and ω·0 = 0 make the
        // contribution identically zero. That is the single largest optimization
        // in the link — sparse lookup advice is ~98% of committed elements — so
        // it needs to be exercised, not just argued.
        let mut rng = Rng(0x5EED_5EED_5EED_5EED);
        let ambient = 10usize;
        let ws: Vec<LinkWitness> = (0..3)
            .map(|_| LinkWitness::dense(
                (0..(1usize << ambient))
                    .map(|_| if rng.next() % 64 == 0 { 1 } else { 0 })
                    .collect(),
            ))
            .collect();
        for w in &ws {
            let nz = w.coeffs.iter().filter(|c| **c != 0).count();
            assert!(nz > 0 && nz < w.coeffs.len() / 8, "witness should be sparse, got {} nonzero", nz);
        }
        let qs: Vec<LinkQuery> = ws
            .iter()
            .enumerate()
            .map(|(e, w)| packed_query(e, w, 0, ambient, ambient, &mut rng))
            .collect();

        let mut tp = Transcript::new(b"sparse");
        let (proof, xi) = prove_link(&ws, &qs, ambient, 1, &mut tp);
        let mut tv = Transcript::new(b"sparse");
        let (_, terminal) = verify_link(ws.len(), &qs, ambient, &proof, &mut tv)
            .expect("sparse witness must verify");
        for (e, w) in ws.iter().enumerate() {
            assert!(ext2_field_eq(terminal[e], mle_eval(&w.coeffs, &xi)));
        }
    }

    #[test]
    fn multiple_queries_per_commitment_are_all_bound() {
        let mut rng = Rng(0x7777_8888_9999_AAAA);
        let ambient = 7usize;
        let ws = vec![binary_witness(ambient, &mut rng)];
        let mut qs: Vec<LinkQuery> = (0..4)
            .map(|b| packed_query(0, &ws[0], b, ambient - 2, ambient, &mut rng))
            .collect();

        let mut tp = Transcript::new(b"multi");
        let (proof, _) = prove_link(&ws, &qs, ambient, 1, &mut tp);
        let mut tv = Transcript::new(b"multi");
        assert!(verify_link(1, &qs, ambient, &proof, &mut tv).is_some());

        // breaking any one of them must be caught
        qs[2].value = ext2_add(qs[2].value, ext2_one());
        let mut tv2 = Transcript::new(b"multi");
        assert!(verify_link(1, &qs, ambient, &proof, &mut tv2).is_none());
    }
}
