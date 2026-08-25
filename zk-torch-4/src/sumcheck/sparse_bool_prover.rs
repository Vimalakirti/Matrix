//! Sparse-boolean sumcheck (Ext2 challenges).
//!
//! Proves `Σ_x eq(r, x) · Σ_j w_j · s_j(x) · (s_j(x) − 1) = 0` for a
//! collection of selection polynomials `s_j` (sparse, binary-valued by
//! construction) with per-term weights `w_j ∈ F_q^2`. Used by the §5.5
//! lookup proofs to validate that every aux entry is in `{0, 1}` even though
//! we only stored its position list.
//!
//! Differs from [`crate::sumcheck::CpuLinearSumcheckProverExt2`] in two ways:
//! 1. The polynomial table is sparse — each term is a sorted `Vec<(position,
//!    value)>` rather than a dense `Vec<Ext2>` of size `2^n`. Crucial since
//!    each lookup aux has at most `2^k` nonzero entries scattered across a
//!    `2^(k + t)` ambient cube.
//! 2. The bool check is degree-3 (`eq · aux · (aux − 1)`), so each round
//!    message has four eval points (`{0, 1, 2, 3}`).
//!
//! Port from zk-torch-2's `SparseBoolSumcheckProver`, lifted to Ext2.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use rayon::prelude::*;

use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, ext2_sub};

/// Sparse-boolean sumcheck prover (Ext2 variant).
pub struct SparseBoolSumcheckProverExt2 {
    pub num_var: usize,
    pub current_round: usize,
    pub challenges: Vec<AlmostGoldilocksExt2>,
    /// Per term: `(weight, sorted (position, current_value) entries)`. Values
    /// are 1 in the initial round (selection polynomial); later rounds carry
    /// Ext2-valued folded entries.
    terms: Vec<(AlmostGoldilocksExt2, Vec<(usize, AlmostGoldilocksExt2)>)>,
    /// Dense eq(r, ·) table of size `2^(num_var − current_round)`. Folded
    /// in-place each round.
    eq_dense: Vec<AlmostGoldilocksExt2>,
}

impl SparseBoolSumcheckProverExt2 {
    /// Transcript header matches a degree-3 sumcheck (3 polynomials: eq +
    /// aux + (aux − 1)). The verifier on the other side will call
    /// [`crate::sumcheck::SumcheckVerifier::verify`] with `num_poly = 3`.
    pub fn new(num_var: usize, transcript: &mut Transcript) -> Self {
        transcript.append_u64(b"num_var", num_var as u64);
        transcript.append_u64(b"num_poly", 3u64);
        Self {
            num_var,
            current_round: 0,
            challenges: Vec::new(),
            terms: Vec::new(),
            eq_dense: Vec::new(),
        }
    }

    /// Run the protocol. `weights[j]` is the term coefficient; `positions[j]`
    /// is the sorted-or-unsorted set-bit positions of selection polynomial
    /// `s_j`; `eq_challenge` is the `r` ∈ `F_q^2^num_var` that defines the
    /// outer eq factor.
    pub fn prove(
        &mut self,
        weights: &[AlmostGoldilocksExt2],
        positions: &[Vec<usize>],
        eq_challenge: &[AlmostGoldilocksExt2],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert_eq!(weights.len(), positions.len(), "weights/positions length mismatch");
        assert_eq!(
            eq_challenge.len(),
            self.num_var,
            "eq_challenge length {} != num_var {}",
            eq_challenge.len(),
            self.num_var,
        );

        // FACTORED-EQ FAST PATH (gated). The dense eq table below is `2^num_var`
        // Ext2 (≈4 GB at num_var=28) and is re-folded every round — the entire
        // cost of this sumcheck for high-arity range auxes. But each selection
        // poly is sparse (`nnz = 2^input_n` set bits in a `2^num_var` cube;
        // sparsity = `2^table_commit_log`). When the cube dwarfs the support we
        // carry eq factored per live entry and reconstruct missing partners in
        // O(1), so cost is O(nnz·num_var) not O(2^num_var). Byte-identical round
        // messages to the dense branch (verified by `sparse_matches_dense`), so
        // the verifier is untouched. Mirrors the same-point sparse backend
        // (`fold::same_point_sumcheck`). Gate: arity ≥ MIN and density below
        // 1/RATIO. `ZK4_SPARSE_BOOL=0` disables.
        {
            // Trigger when (a) arity is high enough that the dense `2^num_var`
            // eq table is a real allocation (≥2^20 ≈ 32 MB/group, GBs by 26 —
            // and groups run rayon-parallel, so the transient is multiplied),
            // and (b) the group is genuinely sparse (`nnz·2 < 2^num_var`). The
            // ratio gate is self-protecting: a range aux has `nnz = T·2^input_n`
            // for T terms over a `2^(input_n+table_commit_log)` cube, so it only
            // fails the gate when T approaches `2^table_commit_log` — i.e. when
            // the support really is near-dense and the dense path is genuinely
            // better. CNN range groups measure 2–8× sparse (many nodes per
            // arity); the win is mostly from avoiding the multi-GB eq alloc.
            let on = std::env::var("ZK4_SPARSE_BOOL").ok().as_deref() != Some("0");
            let min_arity = std::env::var("ZK4_SPARSE_BOOL_MIN_ARITY").ok()
                .and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
            let ratio = std::env::var("ZK4_SPARSE_BOOL_RATIO").ok()
                .and_then(|s| s.parse::<usize>().ok()).unwrap_or(2);
            let nnz: usize = positions.iter().map(|p| p.len()).sum();
            let dense_works = self.num_var < usize::BITS as usize;
            if on && dense_works && self.num_var >= min_arity
                && nnz.saturating_mul(ratio) < (1usize << self.num_var)
            {
                return self.prove_sparse_eq(weights, positions, eq_challenge, transcript);
            }
        }

        // Initialize sparse terms — each position carries value 1 (selection
        // polynomial). Sort once so the round-message loop can walk pairs in
        // index order.
        let one = AlmostGoldilocksExt2::one();
        self.terms = weights
            .iter()
            .zip(positions.iter())
            .map(|(&w, pos)| {
                let mut entries: Vec<(usize, AlmostGoldilocksExt2)> =
                    pos.iter().map(|&idx| (idx, one)).collect();
                entries.sort_unstable_by_key(|&(idx, _)| idx);
                (w, entries)
            })
            .collect();

        // DENSE path timing. Note this, not `prove_sparse_eq`, is what runs on
        // LLM shapes: the sparse gate needs `nnz·2 < 2^arity` and llama2's
        // groups have nnz ~10x the cube (45M vs 2^22), so they all land here.
        // Split into eq-build (O(2^arity), once) vs the round loop
        // (O(nnz) per round, nnz shrinking) because those need different GPU
        // work and only the larger one is worth porting.
        // Device-resident path. Same pair walk, same degree-3 message, same
        // fold; only the 4 Ext2 round messages cross PCIe, so the support data
        // (7 GB at arity 24) never leaves the GPU. Falls back to the CPU branch
        // below on any GPU error, which is why it returns Option.
        let gpu_mode = std::env::var("ZK4_GPU_BOOL").ok();
        if matches!(gpu_mode.as_deref(), Some("1") | Some("strict")) {
            if let Some(pf) = self.prove_on_gpu(weights, positions, eq_challenge, transcript) {
                return pf;
            }
        }

        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
        let t_eq = std::time::Instant::now();
        // eq_dense is 2^arity Ext2 -- 268 MB at arity 24 -- and BOOL_SPLIT
        // duplicates it per sub-group, so this was 22% of the bool phase's
        // thread-time with the allocations contending (0.23-1.82s spread at
        // arity 24). The GPU DP is bit-identical to the CPU Lagrange basis
        // (pinned by `gpu_eq_dp_matches_cpu_lagrange`), and the caller assigns
        // each sub-group a device, so these run spread across the pool.
        // ZK4_GPU_BOOL_EQ=0 disables; any GPU error falls back to CPU.
        // OPT-IN, not default. Building eq_dense on the GPU is 2.9x faster on
        // that sub-phase (5.46s -> 1.89s of thread-time) and worth exactly 0%
        // end to end, because the bool phase's wall time is the MAX over
        // concurrent sub-groups and eq_build is not on the critical one. Off by
        // default so the shipped path is the one the paper numbers were
        // measured with.
        let gpu_eq = std::env::var("ZK4_GPU_BOOL_EQ").ok().as_deref() == Some("1");
        self.eq_dense = if gpu_eq {
            match almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all(eq_challenge) {
                Ok(v) => v,
                Err(_) => evaluate_lagrange_basis_ext2(eq_challenge),
            }
        } else {
            evaluate_lagrange_basis_ext2(eq_challenge)
        };
        let eq_el = t_eq.elapsed();

        let mut t_msg = std::time::Duration::ZERO;
        let mut t_fold = std::time::Duration::ZERO;
        let mut round_messages = Vec::with_capacity(self.num_var);
        for _round in 0..self.num_var {
            let t0 = std::time::Instant::now();
            let msg = self.compute_round_message();
            t_msg += t0.elapsed();
            for m in &msg {
                transcript.append_ext2(b"round_message", m);
            }
            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(msg);
            let t1 = std::time::Instant::now();
            self.receive_challenge(challenge);
            t_fold += t1.elapsed();
        }

        if timing {
            let nnz: usize = positions.iter().map(|p| p.len()).sum();
            eprintln!("[bool_dense] arity={} terms={} nnz={} eq_build={:?} msg={:?} fold={:?}",
                      self.num_var, weights.len(), nnz, eq_el, t_msg, t_fold);
        }

        let final_eval = self.final_evaluation();
        SumcheckProof { final_eval, round_messages }
    }

    /// One round: compute `T(c)` for `c ∈ {0, 1, 2, 3}` in a single pass
    /// over each term's sparse entries. Uses the dense eq array for O(1)
    /// per-position lookups.
    fn compute_round_message(&self) -> Vec<AlmostGoldilocksExt2> {
        let zero = AlmostGoldilocksExt2::zero();
        let one = AlmostGoldilocksExt2::one();
        let two = one + one;
        let three = two + one;

        let partials: Vec<[AlmostGoldilocksExt2; 4]> = self
            .terms
            .par_iter()
            .map(|(w, entries)| {
                let mut t = [zero; 4];
                let mut i = 0;
                while i < entries.len() {
                    let pos = entries[i].0;
                    let rest = pos >> 1;
                    let bit = pos & 1;

                    let mut v0 = zero;
                    let mut v1 = zero;
                    if bit == 0 {
                        v0 = entries[i].1;
                        i += 1;
                        if i < entries.len() && entries[i].0 == 2 * rest + 1 {
                            v1 = entries[i].1;
                            i += 1;
                        }
                    } else {
                        v1 = entries[i].1;
                        i += 1;
                    }

                    let eq0 = self.eq_dense[2 * rest];
                    let eq1 = self.eq_dense[2 * rest + 1];

                    // c=0
                    t[0] = ext2_add(t[0], ext2_mul(ext2_mul(*w, ext2_mul(v0, ext2_sub(v0, one))), eq0));
                    // c=1
                    t[1] = ext2_add(t[1], ext2_mul(ext2_mul(*w, ext2_mul(v1, ext2_sub(v1, one))), eq1));
                    // c=2
                    let aux2 = ext2_sub(ext2_mul(two, v1), v0);
                    let eq2 = ext2_sub(ext2_mul(two, eq1), eq0);
                    t[2] = ext2_add(t[2], ext2_mul(ext2_mul(*w, ext2_mul(aux2, ext2_sub(aux2, one))), eq2));
                    // c=3
                    let aux3 = ext2_sub(ext2_mul(three, v1), ext2_mul(two, v0));
                    let eq3 = ext2_sub(ext2_mul(three, eq1), ext2_mul(two, eq0));
                    t[3] = ext2_add(t[3], ext2_mul(ext2_mul(*w, ext2_mul(aux3, ext2_sub(aux3, one))), eq3));
                }
                t
            })
            .collect();

        let mut result = [zero; 4];
        for t in &partials {
            for i in 0..4 {
                result[i] = ext2_add(result[i], t[i]);
            }
        }
        result.to_vec()
    }

    /// Fold against the verifier's challenge: halve `eq_dense`, fold each
    /// term's sparse entries. Zero-valued folded entries are dropped to keep
    /// each term sparse.
    fn receive_challenge(&mut self, challenge: AlmostGoldilocksExt2) {
        self.challenges.push(challenge);

        let half = self.eq_dense.len() / 2;
        let new_eq: Vec<AlmostGoldilocksExt2> = (0..half)
            .into_par_iter()
            .map(|rest| {
                let a = self.eq_dense[2 * rest];
                let b = self.eq_dense[2 * rest + 1];
                ext2_add(a, ext2_mul(challenge, ext2_sub(b, a)))
            })
            .collect();
        self.eq_dense = new_eq;

        let zero = AlmostGoldilocksExt2::zero();
        self.terms.par_iter_mut().for_each(|(_, entries)| {
            let mut new_entries = Vec::with_capacity(entries.len());
            let mut i = 0;
            while i < entries.len() {
                let pos = entries[i].0;
                let rest = pos >> 1;
                let bit = pos & 1;
                let mut v0 = zero;
                let mut v1 = zero;
                if bit == 0 {
                    v0 = entries[i].1;
                    i += 1;
                    if i < entries.len() && entries[i].0 == 2 * rest + 1 {
                        v1 = entries[i].1;
                        i += 1;
                    }
                } else {
                    v1 = entries[i].1;
                    i += 1;
                }
                let new_val = ext2_add(v0, ext2_mul(challenge, ext2_sub(v1, v0)));
                if !ext2_field_eq(new_val, zero) {
                    new_entries.push((rest, new_val));
                }
            }
            *entries = new_entries;
        });

        self.current_round += 1;
    }

    /// Device-resident dense-branch prover. Returns `None` (transcript
    /// untouched) if the GPU path cannot run, so the caller falls through to
    /// the CPU branch. Any failure AFTER the first transcript append cannot be
    /// unwound here, so the transcript is snapshotted and restored.
    fn prove_on_gpu(
        &mut self,
        weights: &[AlmostGoldilocksExt2],
        positions: &[Vec<usize>],
        eq_challenge: &[AlmostGoldilocksExt2],
        transcript: &mut Transcript,
    ) -> Option<SumcheckProof> {
        use almost_goldilocks_cuda::sumcheck_prover::BoolSumcheckGpu;
        let snapshot = transcript.clone();
        let round0 = self.current_round;
        let nch0 = self.challenges.len();
        let dbg = std::env::var("ZK4_GPU_BOOL_DBG").is_ok();
        // Any bail-out must undo the prover's own state too, not just the
        // transcript: a partially advanced current_round makes the CPU branch
        // run past num_var and trip final_evaluation's assert far from here.
        macro_rules! bail {
            ($ctx:expr, $e:expr) => {{
                if dbg { eprintln!("[gpu_bool] fallback at {}: {:?}", $ctx, $e); }
                // strict is for tests: silently falling back would make a
                // GPU-vs-CPU parity test compare CPU against CPU and pass.
                if std::env::var("ZK4_GPU_BOOL").ok().as_deref() == Some("strict") {
                    panic!("ZK4_GPU_BOOL=strict but the GPU bool path fell back at {}: {:?}", $ctx, $e);
                }
                *transcript = snapshot;
                self.current_round = round0;
                self.challenges.truncate(nch0);
                return None;
            }};
        }
        // Always the GPU DP here, regardless of ZK4_GPU_BOOL_EQ: this path is
        // about to upload eq to the device anyway, so building it on the host
        // would pay a 2^arity DP and then send the result across PCIe.
        let t_eq0 = std::time::Instant::now();
        let eq = match almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all(eq_challenge) {
            Ok(v) => v,
            Err(e) => bail!("eq_dp", e),
        };
        let eq_el = t_eq0.elapsed();
        // Positions must be sorted; the CPU branch sorts them because they come
        // from a HashMap. Do the same before handing them to the device.
        // BoolSumcheckGpu sorts each term's indices in place while building the
        // u32 array, so no pre-sorted deep copy is needed here.
        let t_eq = std::time::Instant::now();
        let mut gpu = match BoolSumcheckGpu::new(weights, positions, &eq) {
            Ok(g) => g,
            Err(e) => bail!("new", e),
        };
        let t_setup = t_eq.elapsed();
        let t_rounds0 = std::time::Instant::now();
        let mut round_messages = Vec::with_capacity(self.num_var);
        for _ in 0..self.num_var {
            let (msg, total) = match gpu.round_message() {
                Ok(x) => x,
                Err(e) => bail!("round_message", e),
            };
            for m in &msg { transcript.append_ext2(b"round_message", m); }
            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(msg.to_vec());
            if let Err(e) = gpu.fold(challenge, total) { bail!("fold", e); }
            self.challenges.push(challenge);
            self.current_round += 1;
        }
        let t_rounds = t_rounds0.elapsed();
        let t_fin0 = std::time::Instant::now();
        let (vals, eq0) = match gpu.finish() {
            Ok(x) => x,
            Err(e) => bail!("finish", e),
        };
        if std::env::var("ZK4_TIMING").ok().as_deref() == Some("1") {
            let nnz: usize = positions.iter().map(|p| p.len()).sum();
            eprintln!("[bool_gpu] arity={} terms={} nnz={} eq_build={:?} setup={:?} rounds={:?} finish={:?}",
                      self.num_var, weights.len(), nnz, eq_el, t_setup, t_rounds, t_fin0.elapsed());
        }
        let one = AlmostGoldilocksExt2::one();
        let mut inner = AlmostGoldilocksExt2::zero();
        for (w, v) in weights.iter().zip(vals.iter()) {
            inner = ext2_add(inner, ext2_mul(*w, ext2_mul(*v, ext2_sub(*v, one))));
        }
        let final_eval = ext2_mul(eq0, inner);
        Some(SumcheckProof { final_eval, round_messages })
    }

    fn final_evaluation(&self) -> AlmostGoldilocksExt2 {
        assert_eq!(self.current_round, self.num_var, "final eval before all rounds done");
        let zero = AlmostGoldilocksExt2::zero();
        let one = AlmostGoldilocksExt2::one();
        let eq_val = if self.eq_dense.is_empty() { zero } else { self.eq_dense[0] };
        let mut inner = zero;
        for (w, entries) in &self.terms {
            let v = if entries.is_empty() || entries[0].0 != 0 { zero } else { entries[0].1 };
            inner = ext2_add(inner, ext2_mul(*w, ext2_mul(v, ext2_sub(v, one))));
        }
        ext2_mul(eq_val, inner)
    }

    /// Factored-eq variant of [`prove`]. Carries `(idx, eq, f)` per live
    /// support entry of each term instead of a dense `2^num_var` eq table,
    /// reconstructing a missing pair-partner's eq in O(1) via the same
    /// `pair_eq`/`round_ratios` helpers the same-point sparse backend uses.
    /// Round messages and `final_eval` are byte-identical to [`prove`]'s dense
    /// branch, so the on-wire proof and the verifier are unchanged.
    fn prove_sparse_eq(
        &mut self,
        weights: &[AlmostGoldilocksExt2],
        positions: &[Vec<usize>],
        eq_challenge: &[AlmostGoldilocksExt2],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        use crate::fold::same_point_sumcheck::{sparse_eq_full, round_ratios};
        let one = AlmostGoldilocksExt2::one();

        // Per-term live support: (idx, eq(eq_challenge, idx), f). f starts at 1
        // (binary selection poly). Sorted ascending by idx so pairs are
        // adjacent (matches the dense branch's sorted-walk ordering).
        // Setup vs round-loop split under ZK4_TIMING. Setup is O(nnz · arity)
        // for `sparse_eq_full` plus a comparison sort per term (positions come
        // from a HashMap, so they arrive unordered). Whether the round loop or
        // the setup dominates decides what is worth moving to a GPU, so the two
        // are timed apart rather than lumped into one number.
        let t_setup = std::time::Instant::now();
        let mut supports: Vec<Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)>> = weights
            .par_iter()
            .zip(positions.par_iter())
            .map(|(_, pos)| {
                let mut s: Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)> = pos
                    .iter()
                    .map(|&idx| (idx as u64, sparse_eq_full(idx as u64, eq_challenge), one))
                    .collect();
                s.sort_unstable_by_key(|&(i, _, _)| i);
                s
            })
            .collect();
        let setup_el = t_setup.elapsed();
        let t_rounds = std::time::Instant::now();

        // `prefix = Π_{k<round} eq̃(eq_challenge[k], r_k)` — only the degenerate
        // (c∈{0,1}) partner-reconstruction fallback inside `pair_eq` reads it.
        let mut prefix = one;
        let mut round_messages = Vec::with_capacity(self.num_var);
        for round in 0..self.num_var {
            let (c, omc, re2o, ro2e) = round_ratios(eq_challenge, round);
            // Degree-3 round message: per term, per pair, accumulate the four
            // interpolation points exactly as the dense `compute_round_message`.
            let partials: Vec<[AlmostGoldilocksExt2; 4]> = supports
                .par_iter()
                .zip(weights.par_iter())
                .map(|(sup, &w)| bool_sparse_round_msg(sup, w, eq_challenge, round, prefix, c, omc, re2o, ro2e))
                .collect();
            let mut msg = [AlmostGoldilocksExt2::zero(); 4];
            for p in &partials { for i in 0..4 { msg[i] = ext2_add(msg[i], p[i]); } }
            let msg = msg.to_vec();
            for m in &msg { transcript.append_ext2(b"round_message", m); }
            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(msg);

            // Fold every term's support against `challenge` (LSB-first), drop
            // f=0 results — identical convention to `sparse_fold`.
            supports.par_iter_mut().for_each(|sup| {
                *sup = bool_sparse_fold(sup, eq_challenge, round, prefix, challenge, c, omc, re2o, ro2e);
            });
            // Advance prefix: eq̃(c, challenge) = (1−c)(1−r) + c·r.
            let eq_factor = ext2_add(
                ext2_mul(ext2_sub(one, c), ext2_sub(one, challenge)),
                ext2_mul(c, challenge));
            prefix = ext2_mul(prefix, eq_factor);
            self.challenges.push(challenge);
            self.current_round += 1;
        }

        if std::env::var("ZK4_TIMING").ok().as_deref() == Some("1") {
            let nnz: usize = positions.iter().map(|p| p.len()).sum();
            eprintln!("[bool_sparse] arity={} terms={} nnz={} setup={:?} rounds={:?}",
                      self.num_var, weights.len(), nnz, setup_el, t_rounds.elapsed());
        }

        // final_eval = eq(eq_challenge, R) · Σ_j w_j · f_j·(f_j−1). eq folded to
        // the singleton point is identical across non-empty terms; empty terms
        // have f=0 → contribute 0 (their eq is never needed).
        let zero = AlmostGoldilocksExt2::zero();
        let mut eq_val: Option<AlmostGoldilocksExt2> = None;
        let mut inner = zero;
        for (sup, &w) in supports.iter().zip(weights.iter()) {
            if let Some(&(idx, eq, f)) = sup.first() {
                debug_assert_eq!(idx, 0, "sparse bool: support not fully folded");
                eq_val = Some(eq);
                inner = ext2_add(inner, ext2_mul(w, ext2_mul(f, ext2_sub(f, one))));
            }
        }
        let final_eval = eq_val.map(|e| ext2_mul(e, inner)).unwrap_or(zero);
        SumcheckProof { final_eval, round_messages }
    }
}

/// Degree-3 sparse round message for one term: returns the four interpolation
/// points `T(0..3) = Σ_pairs w · eq(c)·f(c)·(f(c)−1)`, byte-identical to the
/// dense `compute_round_message`. eq partners reconstructed via `pair_eq`.
#[allow(clippy::too_many_arguments)]
fn bool_sparse_round_msg(
    support: &[(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)],
    w: AlmostGoldilocksExt2,
    claim_pt: &[AlmostGoldilocksExt2],
    round: usize,
    prefix: AlmostGoldilocksExt2,
    c: AlmostGoldilocksExt2,
    omc: AlmostGoldilocksExt2,
    re2o: Option<AlmostGoldilocksExt2>,
    ro2e: Option<AlmostGoldilocksExt2>,
) -> [AlmostGoldilocksExt2; 4] {
    use crate::fold::same_point_sumcheck::pair_eq;
    let zero = AlmostGoldilocksExt2::zero();
    let one = AlmostGoldilocksExt2::one();
    let two = ext2_add(one, one);
    let three = ext2_add(two, one);
    let mut t = [zero; 4];
    let mut i = 0;
    while i < support.len() {
        let y = support[i].0 >> 1;
        let (mut e0o, mut v0, mut e1o, mut v1) = (None, zero, None, zero);
        while i < support.len() && (support[i].0 >> 1) == y {
            if support[i].0 & 1 == 0 { e0o = Some(support[i].1); v0 = support[i].2; }
            else { e1o = Some(support[i].1); v1 = support[i].2; }
            i += 1;
        }
        let (eq0, eq1) = pair_eq(e0o, e1o, y, claim_pt, round, prefix, c, omc, re2o, ro2e);
        // c=0
        t[0] = ext2_add(t[0], ext2_mul(ext2_mul(w, ext2_mul(v0, ext2_sub(v0, one))), eq0));
        // c=1
        t[1] = ext2_add(t[1], ext2_mul(ext2_mul(w, ext2_mul(v1, ext2_sub(v1, one))), eq1));
        // c=2
        let aux2 = ext2_sub(ext2_mul(two, v1), v0);
        let eq2 = ext2_sub(ext2_mul(two, eq1), eq0);
        t[2] = ext2_add(t[2], ext2_mul(ext2_mul(w, ext2_mul(aux2, ext2_sub(aux2, one))), eq2));
        // c=3
        let aux3 = ext2_sub(ext2_mul(three, v1), ext2_mul(two, v0));
        let eq3 = ext2_sub(ext2_mul(three, eq1), ext2_mul(two, eq0));
        t[3] = ext2_add(t[3], ext2_mul(ext2_mul(w, ext2_mul(aux3, ext2_sub(aux3, one))), eq3));
    }
    t
}

/// Fold one term's `(idx, eq, f)` support by `r` (LSB-first), dropping f=0
/// results. Mirrors `same_point_sumcheck::sparse_fold`.
#[allow(clippy::too_many_arguments)]
/// Fold one term's support against `r`, returning a fresh compacted vector.
///
/// This allocates once per term per round, which looks wasteful: on llama2
/// 8L/seq64 the two dominant bool groups carry 147M and 132M support entries
/// over 24 rounds. An in-place variant (write cursor, then `truncate`) was tried
/// and MEASURED WORSE on both counts: bool 9.26s -> 10.50s and, more sharply,
/// fold tree 14.5s -> 30.4s and prove 36.9s -> 53.9s. `truncate` retains
/// capacity, so every term's support holds its ROUND-0 allocation for the whole
/// sumcheck (~11 GB across the two groups) instead of shrinking geometrically,
/// and the later phases pay for it. Returning a fresh smaller vector each round
/// frees the previous one immediately, which is what keeps the resident set
/// falling as the rounds progress. Do not "optimise" this without measuring the
/// fold tree as well as this phase.
fn bool_sparse_fold(
    support: &[(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)],
    claim_pt: &[AlmostGoldilocksExt2],
    round: usize,
    prefix: AlmostGoldilocksExt2,
    r: AlmostGoldilocksExt2,
    c: AlmostGoldilocksExt2,
    omc: AlmostGoldilocksExt2,
    re2o: Option<AlmostGoldilocksExt2>,
    ro2e: Option<AlmostGoldilocksExt2>,
) -> Vec<(u64, AlmostGoldilocksExt2, AlmostGoldilocksExt2)> {
    use crate::fold::same_point_sumcheck::pair_eq;
    let zero = AlmostGoldilocksExt2::zero();
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
            let (e0, e1) = pair_eq(e0o, e1o, y, claim_pt, round, prefix, c, omc, re2o, ro2e);
            let neq = ext2_add(e0, ext2_mul(r, ext2_sub(e1, e0)));
            out.push((y, neq, nf));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::SumcheckVerifier;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    /// Selection polynomials are binary by construction, so the bool sum
    /// `Σ_x eq(r, x) · Σ_j w_j · s_j(x) · (s_j(x) − 1)` is identically zero.
    /// The prover's first-round message must sum to zero (in `s(0) + s(1)`),
    /// and the verifier accepts.
    #[test]
    fn binary_selection_poly_bool_check_is_zero() {
        let n_var = 4;
        // Two terms: positions sets within a 16-position cube.
        let positions = vec![
            vec![0usize, 3, 7, 11],
            vec![1usize, 4, 5, 10, 15],
        ];
        let weights = vec![lift(3), lift(5)];
        let r: Vec<_> = (0..n_var).map(|i| lift(i as u64 * 17 + 11)).collect();

        let mut t_prove = Transcript::new(b"sbc");
        let mut prover = SparseBoolSumcheckProverExt2::new(n_var, &mut t_prove);
        let proof = prover.prove(&weights, &positions, &r, &mut t_prove);

        // First-round message must sum to 0 (the claim).
        let s0 = ext2_add(proof.round_messages[0][0], proof.round_messages[0][1]);
        assert!(
            ext2_field_eq(s0, AlmostGoldilocksExt2::zero()),
            "binary aux: first-round sum should be 0, got {:?}", s0,
        );

        // Verify roundtrip against the standard CPU sumcheck verifier.
        let mut t_verify = Transcript::new(b"sbc");
        let (ok, _) = SumcheckVerifier::verify(
            &proof,
            AlmostGoldilocksExt2::zero(),
            n_var,
            3,
            &mut t_verify,
        );
        assert!(ok, "binary aux should verify with claim = 0");
    }

    /// A non-binary entry (value = 2) breaks the bool check — verifier
    /// rejects.
    #[test]
    fn non_binary_term_breaks_verification() {
        // Build a "term" with one non-binary entry by stuffing value 2 into
        // position 0. The protocol stipulates s_j ∈ {0, 1} — violating that
        // makes the sum at first round nonzero, and verification fails when
        // the claimed sum is 0.
        let n_var = 3;
        let weights = vec![lift(1)];
        let positions = vec![vec![0usize, 1]]; // looks binary by setup

        let mut t_prove = Transcript::new(b"sbc-bad");
        let mut prover = SparseBoolSumcheckProverExt2::new(n_var, &mut t_prove);
        // Override one entry post-init: directly mutate the term list to
        // inject a non-binary value.
        prover.terms = vec![(
            lift(1),
            vec![(0usize, lift(2)), (1usize, AlmostGoldilocksExt2::one())],
        )];
        let r: Vec<_> = (0..n_var).map(|i| lift(i as u64 + 7)).collect();
        prover.eq_dense = evaluate_lagrange_basis_ext2(&r);

        // Manually run the protocol from here so we don't re-init.
        let mut round_messages = Vec::with_capacity(n_var);
        for _round in 0..n_var {
            let msg = prover.compute_round_message();
            for m in &msg {
                t_prove.append_ext2(b"round_message", m);
            }
            let challenge = t_prove.challenge_ext2(b"challenge");
            round_messages.push(msg);
            prover.receive_challenge(challenge);
        }
        let proof = SumcheckProof {
            final_eval: prover.final_evaluation(),
            round_messages,
        };

        // Compute the actual sum (= s0 + s1) — this is what the verifier
        // expects when claim = 0. With value = 2 at position 0, the
        // contribution `2 · (2 − 1) · eq[0] = 2 · eq[0] ≠ 0` violates the
        // boolean constraint, so claim = 0 verification rejects.
        let actual_first_sum = ext2_add(proof.round_messages[0][0], proof.round_messages[0][1]);
        assert!(
            !ext2_field_eq(actual_first_sum, AlmostGoldilocksExt2::zero()),
            "non-binary entry must produce nonzero sum",
        );

        // Replay against a header-matching verifier transcript.
        let mut t_verify = Transcript::new(b"sbc-bad");
        // Mirror prover.new()'s transcript header.
        t_verify.append_u64(b"num_var", n_var as u64);
        t_verify.append_u64(b"num_poly", 3u64);
        let (ok, _) = SumcheckVerifier::verify(
            &proof,
            AlmostGoldilocksExt2::zero(),
            n_var,
            3,
            &mut t_verify,
        );
        assert!(!ok, "non-binary aux must fail bool verification");
    }

    /// The factored-eq sparse path must produce BYTE-IDENTICAL round messages
    /// and final_eval to the dense path — that is what lets the verifier stay
    /// untouched. We compare the dense `prove` (n_var below the gate) against a
    /// direct `prove_sparse_eq` call on the same inputs + transcript seed.
    #[test]
    fn sparse_matches_dense() {
        for n_var in [8usize, 12, 16] {
            let cap = 1usize << n_var;
            let positions = vec![
                vec![0usize, 3, 7, 11, 60, 61, 130, cap - 1],
                vec![1usize, 4, 5, 10, 63, cap / 2, cap / 2 + 1],
                vec![2usize],
            ];
            let weights = vec![lift(3), lift(5), lift(9)];
            let r: Vec<_> = (0..n_var).map(|i| lift(i as u64 * 17 + 11)).collect();

            let mut td = Transcript::new(b"cmp");
            let mut pd = SparseBoolSumcheckProverExt2::new(n_var, &mut td);
            let proof_d = pd.prove(&weights, &positions, &r, &mut td);

            let mut ts = Transcript::new(b"cmp");
            let mut ps = SparseBoolSumcheckProverExt2::new(n_var, &mut ts);
            let proof_s = ps.prove_sparse_eq(&weights, &positions, &r, &mut ts);

            assert_eq!(proof_d.round_messages.len(), proof_s.round_messages.len());
            for (rd, rs) in proof_d.round_messages.iter().zip(proof_s.round_messages.iter()) {
                assert_eq!(rd.len(), rs.len());
                for (a, b) in rd.iter().zip(rs.iter()) {
                    assert!(ext2_field_eq(*a, *b), "round message mismatch at n_var={}", n_var);
                }
            }
            assert!(ext2_field_eq(proof_d.final_eval, proof_s.final_eval),
                "final_eval mismatch at n_var={}", n_var);
        }
    }

    /// End-to-end: with the gate ON (n_var ≥ default min 18), `prove` routes to
    /// the sparse path and the UNCHANGED verifier accepts a binary selection.
    #[test]
    fn sparse_path_verifies_through_gate() {
        let n_var = 20; // ≥ ZK4_SPARSE_BOOL_MIN_ARITY default (18) → sparse path
        let cap = 1usize << n_var;
        // Sparse binary selection: one set bit per "input row" scattered across
        // the cube (mimics a range aux: nnz ≪ 2^n_var).
        let positions = vec![
            (0..256).map(|i| (i * 1031) % cap).collect::<Vec<_>>(),
            (0..256).map(|i| (i * 2053 + 7) % cap).collect::<Vec<_>>(),
        ];
        let weights = vec![lift(3), lift(5)];
        let r: Vec<_> = (0..n_var).map(|i| lift(i as u64 * 13 + 5)).collect();

        let mut t_prove = Transcript::new(b"gate");
        let mut prover = SparseBoolSumcheckProverExt2::new(n_var, &mut t_prove);
        let proof = prover.prove(&weights, &positions, &r, &mut t_prove);

        let mut t_verify = Transcript::new(b"gate");
        let (ok, _) = SumcheckVerifier::verify(
            &proof, AlmostGoldilocksExt2::zero(), n_var, 3, &mut t_verify);
        assert!(ok, "sparse-gate path must verify for a binary selection");
    }

    /// Edge case: a term with one entry at position 0 only — odd-paired
    /// case. Verifier still accepts (single-entry bool check).
    #[test]
    fn single_entry_term_verifies() {
        let n_var = 4;
        let positions = vec![vec![0usize]];
        let weights = vec![lift(7)];
        let r: Vec<_> = (0..n_var).map(|i| lift(i as u64 * 3 + 1)).collect();
        let mut t_prove = Transcript::new(b"single");
        let mut prover = SparseBoolSumcheckProverExt2::new(n_var, &mut t_prove);
        let proof = prover.prove(&weights, &positions, &r, &mut t_prove);
        let mut t_verify = Transcript::new(b"single");
        let (ok, _) = SumcheckVerifier::verify(
            &proof,
            AlmostGoldilocksExt2::zero(),
            n_var,
            3,
            &mut t_verify,
        );
        assert!(ok);
    }
}

#[cfg(test)]
mod gpu_eq_parity {
    use crate::poly::evaluate_lagrange_basis_ext2;
    use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    /// The GPU eq DP must be bit-identical to `evaluate_lagrange_basis_ext2`,
    /// including index/bit ORDER. If it is not, every bool sumcheck that uses
    /// it produces a different round message and no proof verifies -- and the
    /// symptom would be a verification failure far from the cause. Pin it.
    #[test]
    fn gpu_eq_dp_matches_cpu_lagrange() {
        for n in 1..=10usize {
            let pt: Vec<AlmostGoldilocksExt2> = (0..n)
                .map(|i| AlmostGoldilocksExt2::new(
                    AlmostGoldilocksField(0x1234_5678_9abc_def0u64.wrapping_mul(i as u64 + 7) % 0xffff_ffff_0000_0001),
                    AlmostGoldilocksField(0x0fed_cba9_8765_4321u64.wrapping_mul(i as u64 + 3) % 0xffff_ffff_0000_0001),
                ))
                .collect();
            let cpu = evaluate_lagrange_basis_ext2(&pt);
            let gpu = almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all(&pt)
                .expect("gpu eq dp");
            assert_eq!(cpu.len(), gpu.len(), "n={}", n);
            for (k, (a, b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                assert_eq!((a.c0.0, a.c1.0), (b.c0.0, b.c1.0),
                    "n={} index={} cpu=({},{}) gpu=({},{})",
                    n, k, a.c0.0, a.c1.0, b.c0.0, b.c1.0);
            }
        }
    }
}

#[cfg(test)]
mod gpu_bool_parity {
    use super::SparseBoolSumcheckProverExt2;
    use crate::transcript::Transcript;
    use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    fn ext2(a: u64, b: u64) -> AlmostGoldilocksExt2 {
        const P: u64 = 0xffff_ffff_0000_0001;
        AlmostGoldilocksExt2::new(AlmostGoldilocksField(a % P), AlmostGoldilocksField(b % P))
    }

    /// The device path must reproduce the CPU dense branch exactly: same round
    /// messages, same final eval, same transcript state. A divergence here is a
    /// proof that does not verify, and the symptom would appear far from the
    /// cause, so this is pinned directly rather than left to an end-to-end run.
    #[test]
    fn gpu_bool_matches_cpu_dense() {
        for &(arity, n_terms, per_term) in
            &[(6usize, 3usize, 5usize), (8, 4, 17), (10, 6, 40), (12, 9, 100)]
        {
            let weights: Vec<_> = (0..n_terms)
                .map(|i| ext2(0x9e37_79b9_7f4a_7c15u64.wrapping_mul(i as u64 + 1),
                              0xbf58_476d_1ce4_e5b9u64.wrapping_mul(i as u64 + 3)))
                .collect();
            // Distinct sorted positions per term, deliberately including both
            // paired (2r, 2r+1) and lone-odd entries so every branch of the
            // pair walk is exercised.
            let positions: Vec<Vec<usize>> = (0..n_terms).map(|t| {
                let mut v: Vec<usize> = (0..per_term)
                    .map(|k| ((k * 7 + t * 3) * 5 + (k % 3)) % (1usize << arity))
                    .collect();
                v.sort_unstable(); v.dedup(); v
            }).collect();
            let eq_ch: Vec<_> = (0..arity)
                .map(|i| ext2(0x2545_f491_4f6c_dd1du64.wrapping_mul(i as u64 + 5),
                              0x1234_5678_9abc_def0u64.wrapping_mul(i as u64 + 2)))
                .collect();

            let run = |gpu: bool| {
                if gpu { std::env::set_var("ZK4_GPU_BOOL", "strict"); }
                else { std::env::remove_var("ZK4_GPU_BOOL"); }
                let mut t = Transcript::new(b"parity");
                let mut p = SparseBoolSumcheckProverExt2::new(arity, &mut t);
                let pf = p.prove(&weights, &positions, &eq_ch, &mut t);
                (pf, p.challenges.clone())
            };
            // ZK4_SPARSE_BOOL=0 keeps the CPU arm on the dense branch, which is
            // what the GPU path mirrors (the factored-eq branch is a different
            // algorithm and is not what runs on LLM shapes).
            std::env::set_var("ZK4_SPARSE_BOOL", "0");
            let (cpu, cpu_ch) = run(false);
            let (gpu, gpu_ch) = run(true);
            std::env::remove_var("ZK4_GPU_BOOL");

            assert_eq!(cpu.round_messages.len(), gpu.round_messages.len(),
                "arity={} rounds", arity);
            for (r, (a, b)) in cpu.round_messages.iter().zip(gpu.round_messages.iter()).enumerate() {
                for (k, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    assert_eq!((x.c0.0, x.c1.0), (y.c0.0, y.c1.0),
                        "arity={} round={} point={}", arity, r, k);
                }
            }
            assert_eq!((cpu.final_eval.c0.0, cpu.final_eval.c1.0),
                       (gpu.final_eval.c0.0, gpu.final_eval.c1.0), "arity={} final", arity);
            assert_eq!(cpu_ch.len(), gpu_ch.len());
            for (a, b) in cpu_ch.iter().zip(gpu_ch.iter()) {
                assert_eq!((a.c0.0, a.c1.0), (b.c0.0, b.c1.0), "arity={} challenge", arity);
            }
        }
    }
}

#[cfg(test)]
mod gpu_bool_isolate {
    use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    use almost_goldilocks_cuda::sumcheck_prover::BoolSumcheckGpu;

    fn e(a: u64, b: u64) -> AlmostGoldilocksExt2 {
        const P: u64 = 0xffff_ffff_0000_0001;
        AlmostGoldilocksExt2::new(AlmostGoldilocksField(a % P), AlmostGoldilocksField(b % P))
    }

    /// Isolated so a sticky CUDA error from an earlier case cannot be mistaken
    /// for a fault here. Grows the shape until something breaks.
    #[test]
    fn bool_gpu_shapes() {
        for &(arity, terms, per) in &[(6usize,3usize,5usize),(8,4,17),(10,6,40),(12,9,100),(12,9,120)] {
            let w: Vec<_> = (0..terms).map(|i| e(i as u64 + 1, i as u64 + 2)).collect();
            let pos: Vec<Vec<usize>> = (0..terms).map(|t| {
                let mut v: Vec<usize> = (0..per).map(|k| ((k*7 + t*3)*5 + (k%3)) % (1usize<<arity)).collect();
                v.sort_unstable(); v.dedup(); v
            }).collect();
            let n: usize = pos.iter().map(|p| p.len()).sum();
            let eq: Vec<_> = (0..(1usize<<arity)).map(|i| e(i as u64 * 7 + 1, i as u64 * 3 + 2)).collect();
            let mut g = match BoolSumcheckGpu::new(&w, &pos, &eq) {
                Ok(g) => g, Err(er) => panic!("new failed arity={} n={}: {:?}", arity, n, er)
            };
            for r in 0..arity {
                let (_m, total) = g.round_message()
                    .unwrap_or_else(|er| panic!("msg failed arity={} terms={} n={} round={}: {:?}", arity, terms, n, r, er));
                g.fold(e(r as u64 + 3, r as u64 + 5), total)
                    .unwrap_or_else(|er| panic!("fold failed arity={} round={} total={}: {:?}", arity, r, total, er));
            }
            g.finish().unwrap_or_else(|er| panic!("finish failed arity={}: {:?}", arity, er));
            eprintln!("  arity={} terms={} n={} OK", arity, terms, n);
        }
    }
}
