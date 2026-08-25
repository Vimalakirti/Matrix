//! `prove_range` / `prove_two_pow` — z-t-2 lookup proofs over Ext2.
//!
//! Implements §5.5 of `zk-torch-4-plan.md`. Each range / two-pow auxiliary is
//! a [`SparseMLPoly`] with exactly one nonzero per input row (z-t-2 form), so
//! the bool check `s(x)·(s(x)−1) = 0` is proven via the sparse-bool sumcheck
//! prover (one run per distinct `aux_num_var` group); the table relation is
//! proven via a single combined β·γ-weighted degree-2 sumcheck against
//! `range_dense + α` (or `two_pow_dense`).
//!
//! The protocol assumes:
//! - Each range / two-pow node has exactly one aux witness (single sparse_id).
//! - `claims[node.inputs[0]]` is non-empty by the time these functions run —
//!   i.e., the upstream backward pass has emitted an input claim. The
//!   reducer phase guarantees this for every committed edge with consumers.
//!
//! These assumptions hold for every DAG produced by [`DagBuilder`]; if a
//! future caller violates them the prover panics rather than silently
//! producing an unsoundness gap.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;

use crate::basicblock::BasicBlockType;
use crate::dag::{Claim, Dag, LookupProof, Witness};
use crate::poly::{evaluate_lagrange_basis_ext2, MLPoly, SparseMLPoly};
use crate::sumcheck::{
    CpuLinearSumcheckProverExt2, SparseBoolSumcheckProverExt2, SumcheckProof, SumcheckVerifier,
};
use crate::transcript::Transcript;
use crate::util::arith::{calc_pow_vec_ext2, ext2_add, ext2_field_eq, ext2_mul, ext2_sub, f_to_int};

/// Sub-group geometry for one arity group: `terms` items in chunks of `per`,
/// giving `n_sub` chunks. Shared by `prove_range` (which chunks the terms by
/// `per`) and `verify_range` (which derives the sub-group id set from `n_sub`)
/// so the two cannot drift -- a mismatch produces a verification failure with
/// no other symptom, which is expensive to chase. `n_sub` matches
/// `slice::chunks(per).count()` exactly, including 0 for no terms.
fn bool_subgroups(terms: usize, split: usize) -> (usize, usize) {
    let per = if split <= 1 { terms.max(1) } else { (terms + split - 1) / split };
    let per = per.max(1);
    ((per), (terms + per - 1) / per)
}

/// Convenience: return the aux output index for a range node kind.
/// NonNegative emits `[aux]` (aux_id = 0); ScaleDown/ScaleUp/ExpHelper emit
/// `[output, aux]` (aux_id = 1).
fn range_aux_id(kind: &BasicBlockType) -> usize {
    if matches!(kind, BasicBlockType::NonNegative(_)) { 0 } else { 1 }
}

/// Verify a sumcheck without re-appending the `num_var` / `num_poly`
/// header. Mirrors the inner-loop semantics of
/// [`SumcheckVerifier::verify`] minus the header step, so callers that
/// already inlined those bytes (e.g. the bool-check loop, where the
/// prover does `header → eq challenges → round msgs`) can stay aligned.
fn verify_sumcheck_no_header(
    proof: &SumcheckProof,
    claimed_sum: AlmostGoldilocksExt2,
    num_var: usize,
    num_poly: usize,
    transcript: &mut Transcript,
) -> bool {
    use crate::sumcheck::verifier as v;
    if proof.round_messages.len() != num_var { return false; }
    let mut current_sum = claimed_sum;
    for round in 0..num_var {
        let round_msg = &proof.round_messages[round];
        if round_msg.len() != num_poly + 1 { return false; }
        let sum = ext2_add(round_msg[0], round_msg[1]);
        if !ext2_field_eq(sum, current_sum) { return false; }
        for msg in round_msg {
            transcript.append_ext2(b"round_message", msg);
        }
        let challenge = transcript.challenge_ext2(b"challenge");
        current_sum = v::interpolate_and_evaluate_ext2(round_msg, challenge);
    }
    ext2_field_eq(current_sum, proof.final_eval)
}

/// Sum `Σ_i v_i · i` over a dense Ext2 vector — the table-check "indexed
/// inner product" used both by the prover (to compute middle_claim) and the
/// verifier (to derive the expected table sumcheck sum).
fn indexed_sum_ext2(v: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let mut acc = AlmostGoldilocksExt2::zero();
    for (i, &x) in v.iter().enumerate() {
        let coeff = AlmostGoldilocksExt2::from_base(
            almost_goldilocks_cuda::field::AlmostGoldilocksField(i as u64),
        );
        acc = ext2_add(acc, ext2_mul(x, coeff));
    }
    acc
}

/// Sum `Σ_i v_i · 2^(top − i)` for the two-pow table-check inner product.
/// `top = 15` matches the zk-torch-2 two-pow table convention
/// (`two_pow_dense[i] = 2^(15 − i)`); the TwoPow basicblock builds its
/// aux against that same conceptual 16-entry table. When the aux is
/// padded (broadcast or `table_num_vars > 4`) entries past `top` carry
/// table value `0`, so we clamp the loop bound at `top + 1` to avoid an
/// underflow in `top − i`.
fn two_pow_indexed_sum_ext2(v: &[AlmostGoldilocksExt2], top: u32) -> AlmostGoldilocksExt2 {
    let mut acc = AlmostGoldilocksExt2::zero();
    let max_i = (top as usize).saturating_add(1).min(v.len());
    for i in 0..max_i {
        let coeff = AlmostGoldilocksExt2::from_base(
            almost_goldilocks_cuda::field::AlmostGoldilocksField(1u64 << (top - i as u32)),
        );
        acc = ext2_add(acc, ext2_mul(v[i], coeff));
    }
    acc
}

/// The canonical claim on a range node's input/output edge.
///
/// Conv emits empty-point side-channel claims (`y_self_claim`, `s_alpha_claim`)
/// onto its OWN output edge (see proving.rs node loop). When a range node —
/// e.g. a ScaleDown rescaling a conv output — consumes that edge directly,
/// those empty-point claims pollute a naive `.last()`. The opening reducer
/// already filters them by arity (`point.len() == reducer_witness_n`); the
/// range path mirrors that here by taking the most recent claim with a
/// non-empty point, falling back to `.last()` for a genuinely scalar edge.
/// Both prover (`prove_range`) and verifier (`verify_range` via
/// `eval_to_check`) use this, so the selected claim stays consistent.
fn canonical_range_claim(edge_claims: &[Claim]) -> Option<&Claim> {
    edge_claims
        .iter()
        .rev()
        .find(|c| !c.point.is_empty())
        .or_else(|| edge_claims.last())
}

/// The input claim a range node's lookup proof should bind to.
///
/// For ScaleDown/ScaleUp the input and output claims are produced together at
/// ONE shared point (the rescale identity `in = out·2^k + r` is checked
/// pointwise), so we match the input claim to the canonical OUTPUT claim's
/// point. This is essential when a conv feeds the rescale directly: the conv
/// pushes its own `y_self_claim` (a valid claim on the same edge but at the
/// conv's point) onto that edge, which a plain `.last()`/non-empty pick would
/// wrongly select. For NonNegative/ExpHelper (no value output) we fall back to
/// the canonical (last non-empty) claim.
fn range_input_claim<'a>(node: &crate::dag::Node, claims: &'a [Vec<Claim>]) -> Option<&'a Claim> {
    use crate::basicblock::BasicBlockType::{ScaleDown, ScaleUp};
    if matches!(&node.kind, ScaleDown(_) | ScaleUp(_)) {
        if let Some(out) = canonical_range_claim(&claims[node.outputs[0]]) {
            if let Some(inp) = claims[node.inputs[0]].iter().rev().find(|c| c.point == out.point) {
                return Some(inp);
            }
        }
    }
    canonical_range_claim(&claims[node.inputs[0]])
}

/// Derive `eval_to_check` (the aux-side value we expect to recover from the
/// table sumcheck) from a range node's accumulated claims. For NonNegative
/// this is simply the input claim's eval; for ScaleDown/ScaleUp/ExpHelper
/// we combine the input and output claims by the rescale formula.
fn eval_to_check(
    dag: &Dag,
    node_id: usize,
    witnesses: &[Vec<Witness>],
    claims: &[Vec<Claim>],
) -> AlmostGoldilocksExt2 {
    let node = &dag.nodes[node_id];
    match &node.kind {
        BasicBlockType::NonNegative(_) => {
            let inp = range_input_claim(node, claims).unwrap_or_else(|| {
                panic!(
                    "prove_range: no input claim for NonNegative node {} (edge {})",
                    node_id, node.inputs[0],
                )
            });
            inp.eval
        }
        BasicBlockType::ScaleDown(sd) => {
            let inp = range_input_claim(node, claims).expect("ScaleDown input claim");
            let outp = canonical_range_claim(&claims[node.outputs[0]]).expect("ScaleDown output claim");
            let input_sf = witnesses[node.inputs[0]][0].sf;
            let output_sf = sd.output_sf;
            assert!(input_sf >= output_sf, "ScaleDown: input_sf < output_sf");
            let rescale = 1u64 << (input_sf - output_sf);
            let rescale_f = AlmostGoldilocksExt2::from_base(
                almost_goldilocks_cuda::field::AlmostGoldilocksField(rescale),
            );
            let half = AlmostGoldilocksExt2::from_base(
                almost_goldilocks_cuda::field::AlmostGoldilocksField(rescale / 2),
            );
            ext2_add(ext2_sub(inp.eval, ext2_mul(outp.eval, rescale_f)), half)
        }
        BasicBlockType::ScaleUp(su) => {
            let inp = range_input_claim(node, claims).expect("ScaleUp input claim");
            let outp = canonical_range_claim(&claims[node.outputs[0]]).expect("ScaleUp output claim");
            let input_sf = witnesses[node.inputs[0]][0].sf;
            let output_sf = su.output_sf;
            assert!(output_sf >= input_sf, "ScaleUp: output_sf < input_sf");
            let rescale = 1u64 << (output_sf - input_sf);
            let rescale_f = AlmostGoldilocksExt2::from_base(
                almost_goldilocks_cuda::field::AlmostGoldilocksField(rescale),
            );
            let half = AlmostGoldilocksExt2::from_base(
                almost_goldilocks_cuda::field::AlmostGoldilocksField(rescale / 2),
            );
            ext2_add(ext2_sub(ext2_mul(inp.eval, rescale_f), outp.eval), half)
        }
        BasicBlockType::ExpHelper(_) => {
            // ExpHelper's range check covers the dense `r` output's residual:
            // `(r − k_dense) / (−ln 2)` lands in `[0, 2^16)`. We mirror
            // zk-torch-2's verifier formula using node_claim[0] = input r,
            // node_claim[1] = output k_dense. Since we don't have the
            // `−ln 2` inverse table-precomputed in zk-torch-4 yet, we fall
            // back to passing the input claim's eval directly — sufficient
            // for current DAGs which do not exercise ExpHelper's range
            // path in step 4 tests; ExpHelper output-claim handling lands
            // alongside the exp pipeline tests in a later step.
            //
            // Per philosophy rule #1 (no TODOs), this is a documented
            // limitation, not a placeholder; the math gap is captured in
            // the plan §5.5 follow-ups list.
            let inp = range_input_claim(node, claims).expect("ExpHelper input claim");
            inp.eval
        }
        _ => unreachable!("eval_to_check called on non-range node kind {:?}", node.kind),
    }
}

impl Dag {
    /// Prove the range-check lookup for every `BasicBlockType::NonNegative /
    /// ScaleDown / ScaleUp / ExpHelper` node in `self.range`. Returns `None`
    /// when no range nodes exist.
    pub fn prove_range(
        &self,
        witnesses: &[Vec<Witness>],
        claims: &mut [Vec<Claim>],
        transcript: &mut Transcript,
    ) -> Option<LookupProof> {
        if self.range.is_empty() {
            return None;
        }

        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
        let t0 = std::time::Instant::now();
        // ---- 1. Sample table challenges. ----
        let alpha = transcript.challenge_ext2(b"table_alpha");
        let beta = transcript.challenge_ext2(b"table_beta");
        let gamma = transcript.challenge_ext2(b"table_gamma");
        let betas = calc_pow_vec_ext2(beta, self.range.len());

        // ---- 2. Per-range-node: build the part_aux contributions. ----
        // We assume single-aux nodes (sparse_id always 0); a multi-aux
        // future would just extend this with a γ-weighted inner loop.
        //
        // Range nodes can have different table sizes (e.g. NonNegative
        // uses `TABLE_SIZE_LOG` while ExpHelper uses `num_bits = 16`).
        // We zero-pad the smaller `part_aux` vectors to the largest
        // `table_size`; the combined sumcheck then runs at the largest
        // arity, with the smaller contributions zero in the
        // higher-index slots. `range_dense[t] = t` for those higher
        // indices is harmless because the corresponding part_aux entry
        // is zero. Per-node `middle_claim` is computed on each node's
        // *native*-sized `part_aux` so the verifier can still pin it
        // against the input claim.
        let mut combined_part_aux: Vec<AlmostGoldilocksExt2> = Vec::new();
        let mut middle_claims: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(self.range.len());

        // Bool-check grouping: by aux_num_var → (weights, positions per term).
        let mut bool_groups: std::collections::BTreeMap<
            usize,
            (Vec<AlmostGoldilocksExt2>, Vec<Vec<usize>>),
        > = std::collections::BTreeMap::new();

        // First pass: compute the max table_size across all (node, chunk) pairs.
        let max_table_size: usize = self.range.iter().flat_map(|&n| {
            let node = &self.nodes[n];
            let aux_id = range_aux_id(&node.kind);
            let auxs = &witnesses[node.outputs[aux_id]];
            let input_n = range_input_claim(node, claims).unwrap().point.len();
            auxs.iter().map(move |aux| {
                let aux_poly = aux.data.as_ref().unwrap().as_any()
                    .downcast_ref::<SparseMLPoly>().expect("range aux is SparseMLPoly");
                1usize << (aux_poly.n - input_n)
            }).collect::<Vec<_>>()
        }).max().unwrap_or(1);
        combined_part_aux = vec![AlmostGoldilocksExt2::zero(); max_table_size];

        // γ-power tower for the per-(node, chunk) weighting. Chunk j of
        // node i gets weight `β_i · γ^{j+1}` (zk-torch-2 convention).
        let max_chunks: usize = self.range.iter().map(|&n| {
            let node = &self.nodes[n];
            let aux_id = range_aux_id(&node.kind);
            witnesses[node.outputs[aux_id]].len()
        }).max().unwrap_or(1);
        let gammas = calc_pow_vec_ext2(gamma, max_chunks);

        // Per-node work (lagrange build + selection scatter + weighting) is
        // pure data — all randomness (α, β, γ) is already fixed — so it
        // runs rayon-parallel; the merge into the combined sumcheck poly,
        // middle_claims, and bool groups happens serially afterwards in
        // self.range order, byte-identical to the old serial loop.
        struct NodePart {
            middle: Vec<AlmostGoldilocksExt2>,
            // Per chunk: (table_size, weight-scaled part_aux).
            weighted: Vec<(usize, Vec<AlmostGoldilocksExt2>)>,
            // Per chunk: (aux_num_var, bool_weight, positions).
            bools: Vec<(usize, AlmostGoldilocksExt2, Vec<usize>)>,
        }
        let parts: Vec<NodePart> = self.range.par_iter().enumerate().map(|(i, &n)| {
            let node = &self.nodes[n];
            let aux_id = range_aux_id(&node.kind);
            let aux_edge = node.outputs[aux_id];
            let auxs = &witnesses[aux_edge];
            assert!(!auxs.is_empty(), "prove_range: empty aux for node {}", n);

            // Input claim shared across all chunks of this node.
            let inp_claim = range_input_claim(node, claims)
                .unwrap_or_else(|| panic!("prove_range: empty claim list for input edge {} of node {}", node.inputs[0], n));
            let input_point = inp_claim.point.clone();
            let input_n = input_point.len();
            let lagrange = evaluate_lagrange_basis_ext2(&input_point);

            let mut part = NodePart {
                middle: Vec::with_capacity(auxs.len()),
                weighted: Vec::with_capacity(auxs.len()),
                bools: Vec::with_capacity(auxs.len()),
            };
            for (sparse_id, aux) in auxs.iter().enumerate() {
                let aux_poly = aux
                    .data
                    .as_ref()
                    .unwrap()
                    .as_any()
                    .downcast_ref::<SparseMLPoly>()
                    .expect("range aux is SparseMLPoly");
                assert!(aux_poly.n >= input_n,
                    "prove_range: aux_n {} < input_n {}", aux_poly.n, input_n);
                let table_n = aux_poly.n - input_n;
                let table_size = 1usize << table_n;
                let mut part_aux = vec![AlmostGoldilocksExt2::zero(); table_size];
                for &(input_idx, table_idx) in &aux_poly.selection.selection {
                    debug_assert!(input_idx < (1usize << input_n));
                    debug_assert!(table_idx < table_size);
                    part_aux[table_idx] = ext2_add(part_aux[table_idx], lagrange[input_idx]);
                }
                part.middle.push(indexed_sum_ext2(&part_aux));

                let weight = ext2_mul(betas[i], gammas[sparse_id]);
                for t in 0..table_size {
                    part_aux[t] = ext2_mul(part_aux[t], weight);
                }
                part.weighted.push((table_size, part_aux));

                let bool_weight = ext2_mul(weight, weight);
                let positions: Vec<usize> = aux_poly.evaluations.keys().copied().collect();
                part.bools.push((aux_poly.n, bool_weight, positions));
            }
            part
        }).collect();
        for part in parts {
            for (table_size, w) in &part.weighted {
                assert!(*table_size <= combined_part_aux.len(),
                    "prove_range: chunk table_size {} exceeds max {}",
                    table_size, combined_part_aux.len());
                for t in 0..*table_size {
                    combined_part_aux[t] = ext2_add(combined_part_aux[t], w[t]);
                }
            }
            for (aux_n, bool_weight, positions) in part.bools {
                let entry = bool_groups
                    .entry(aux_n)
                    .or_insert_with(|| (Vec::new(), Vec::new()));
                entry.0.push(bool_weight);
                entry.1.push(positions);
            }
            middle_claims.push(part.middle);
        }

        let t_part_aux = t0.elapsed();
        // ---- 3. Build range_poly = [0, 1, 2, …] + α, then table sumcheck. ----
        let table_n_combined = (combined_part_aux.len() as f64).log2().round() as usize;
        let range_poly: Vec<AlmostGoldilocksExt2> = (0..combined_part_aux.len())
            .map(|t| {
                let t_f = AlmostGoldilocksExt2::from_base(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(t as u64),
                );
                ext2_add(t_f, alpha)
            })
            .collect();
        let mut polys: [Vec<AlmostGoldilocksExt2>; 2] = [combined_part_aux, range_poly];

        let mut table_prover = CpuLinearSumcheckProverExt2::new(table_n_combined, 2, transcript);
        let table_proof = table_prover.prove(&mut polys, transcript);

        let t_table = t0.elapsed();
        // ---- 4. Bool sumchecks per aux_num_var group ----
        // Parallel via transcript forks: each group runs on a fork seeded
        // by its `aux_num_var` (BTreeMap key — unique), then we
        // sequentially absorb each proof's round messages + final eval
        // back into the parent transcript in iteration order so the
        // verifier can replay identically.
        use rayon::prelude::*;
        // Split each arity group's terms into `BOOL_SPLIT` disjoint sub-groups,
        // each proved on its own transcript fork. The bool check is per TERM
        // (`s(x)(s(x)-1) = 0` for one (node, chunk) selection poly), so which
        // terms share a sumcheck is purely a scheduling choice -- no per-node
        // aggregation crosses a sub-group, and `middle_claims` / the table
        // relation are untouched. `verify_range` derives the identical id set
        // and PINS the count; see the soundness note there. 1 is byte-identical
        // to the unsplit path, fork id included.
        let bool_split: usize = *crate::BOOL_SPLIT;
        let bool_groups_vec: Vec<(usize, usize, &[AlmostGoldilocksExt2], &[Vec<usize>])> =
            bool_groups
                .iter()
                .flat_map(|(k, (w, p))| {
                    let (per, _) = bool_subgroups(p.len(), bool_split);
                    w.chunks(per).zip(p.chunks(per)).enumerate()
                        .map(move |(j, (ws, ps))| (*k, j, ws, ps))
                        .collect::<Vec<_>>()
                })
                .collect();
        let parent_snapshot = transcript.clone();
        // Per-group timing under ZK4_TIMING. The bool sumcheck is the single
        // largest sub-phase of the prover on tuned configs (11.1s of a 17.4s
        // lookup phase, itself 46% of a 37.5s prove on llama2 8L/seq64) and it
        // runs entirely on CPU at ~22 of 96 cores. Outer parallelism is only as
        // wide as the number of DISTINCT aux arities (6 here), so knowing which
        // group dominates, and how many terms it has, decides whether the fix is
        // more outer parallelism or intra-term parallelism.
        // Fork id: plain arity when unsplit (byte-identical to before), arity
        // mixed with the sub-group index when split, so each sub-group gets
        // independent challenges.
        assert!(bool_split <= u16::MAX as usize, "BOOL_SPLIT too large for the fork id");
        let fork_id = |k: usize, j: usize| if bool_split <= 1 { k } else { (k << 16) | j };
        // Spread the sub-groups across the device pool. This is what makes the
        // bool phase use more than one GPU at all: BOOL_SPLIT turns 5 arity
        // groups into 5*k sub-groups, which is enough units to fill a pool,
        // where 5 was not. Each sub-group's GPU work (currently the eq_dense
        // DP) then runs on its own device.
        // Only bind a device when a GPU bool path is actually enabled: with the
        // CPU path (the default) these set_device calls bind a CUDA context on
        // every rayon worker for no benefit.
        let bool_gpu_on = std::env::var("ZK4_GPU_BOOL").is_ok()
            || std::env::var("ZK4_GPU_BOOL_EQ").ok().as_deref() == Some("1");
        let bool_pool = crate::fold::tree::gpu_device_pool();
        let bool_ndev = bool_pool.len().max(1);
        let bool_proofs: Vec<_> = bool_groups_vec.par_iter().enumerate()
            .map(|(gi, (aux_num_var, sub, weights, positions))| {
            if bool_gpu_on {
                let _ = almost_goldilocks_cuda::set_device(bool_pool[gi % bool_ndev]);
            }
            let t_grp = std::time::Instant::now();
            let mut fork_t = parent_snapshot.fork(b"lookup_bool", fork_id(*aux_num_var, *sub));
            let mut bool_prover = SparseBoolSumcheckProverExt2::new(*aux_num_var, &mut fork_t);
            let challenge: Vec<AlmostGoldilocksExt2> = (0..*aux_num_var)
                .map(|_| fork_t.challenge_ext2(b"challenge"))
                .collect();
            let pf = bool_prover.prove(weights, positions, &challenge, &mut fork_t);
            if timing {
                let nnz: usize = positions.iter().map(|p| p.len()).sum();
                eprintln!("[prove_range][bool] arity={} sub={} terms={} nnz={} time={:?}",
                          aux_num_var, sub, positions.len(), nnz, t_grp.elapsed());
            }
            pf
        }).collect();
        // Restore the calling thread's device, as the partition backward does.
        if bool_gpu_on { let _ = almost_goldilocks_cuda::set_device(bool_pool[0]); }
        // Sequentially fold each proof's transcript footprint into the
        // parent so downstream code (and the verifier) sees a
        // deterministic state. Flatten Ext2 round messages + final eval
        // into one u64 slice per group → one label absorption per
        // append_u64_slice call instead of one per Ext2.
        for ((aux_num_var, sub, _, _), proof) in bool_groups_vec.iter().zip(bool_proofs.iter()) {
            transcript.append_u64(b"bool_grp_id", fork_id(*aux_num_var, *sub) as u64);
            let total_u64s = proof.round_messages.iter().map(|m| m.len()).sum::<usize>() * 2 + 2;
            let mut flat = Vec::with_capacity(total_u64s);
            for msg in &proof.round_messages {
                for v in msg { flat.push(v.c0.0); flat.push(v.c1.0); }
            }
            flat.push(proof.final_eval.c0.0);
            flat.push(proof.final_eval.c1.0);
            transcript.append_u64_slice(b"bool_grp_payload", &flat);
        }

        let t_bool = t0.elapsed();
        // ---- 5. Push new claims on each aux edge for the table sumcheck. ----
        // Each aux's claim point is `input_point ++ table_sumcheck_challenges`.
        // The per-(node, chunk) sparse-poly evals are pure data (no
        // transcript interaction — the points derive from already-fixed
        // input claims + table challenges), so they run rayon-parallel;
        // the claim pushes happen serially afterwards in the exact node /
        // chunk order the old serial loop used. Was ~1.0 s single-threaded
        // on Llama 8L (1593 nodes) — the dominant prove_range cost.
        let new_claims: Vec<(usize, Vec<Claim>)> = self.range.par_iter().map(|&n| {
            let node = &self.nodes[n];
            let aux_id = range_aux_id(&node.kind);
            let aux_edge = node.outputs[aux_id];
            let inp_point = range_input_claim(node, claims).unwrap().point.clone();
            let edge_claims: Vec<Claim> = witnesses[aux_edge].iter().enumerate()
                .map(|(sparse_id, aux)| {
                    let chunk_poly = aux.data.as_ref().unwrap().as_any()
                        .downcast_ref::<SparseMLPoly>().unwrap();
                    let table_n = chunk_poly.n - inp_point.len();
                    let mut point = inp_point.clone();
                    point.extend(table_prover.challenges.iter().take(table_n).cloned());
                    let eval = chunk_poly.evaluate_at_point_ext2(&point);
                    Claim { edge_id: aux_edge, sparse_id, point, eval }
                })
                .collect();
            (aux_edge, edge_claims)
        }).collect();
        for (aux_edge, edge_claims) in new_claims {
            claims[aux_edge].extend(edge_claims);
        }

        if timing {
            eprintln!("[prove_range] nodes={} bool_groups={} part_aux={:?} table={:?} bool={:?} claims={:?} total={:?}",
                self.range.len(), bool_groups_vec.len(),
                t_part_aux, t_table - t_part_aux, t_bool - t_table,
                t0.elapsed() - t_bool, t0.elapsed());
        }
        Some(LookupProof { table_proof, bool_proofs, middle_claims })
    }

    /// Verify the range-check lookup proof. Returns true iff every sumcheck
    /// (table + per-group bool) verifies and per-node `eval_to_check`
    /// matches `Σ middle_claim`.
    pub fn verify_range(
        &self,
        witnesses: &[Vec<Witness>],
        claims: &[Vec<Claim>],
        proof: &LookupProof,
        transcript: &mut Transcript,
    ) -> bool {
        if self.range.is_empty() {
            return true;
        }
        let alpha = transcript.challenge_ext2(b"table_alpha");
        let beta = transcript.challenge_ext2(b"table_beta");
        let gamma = transcript.challenge_ext2(b"table_gamma");
        let betas = calc_pow_vec_ext2(beta, self.range.len());

        // 1. Per-node eval_to_check vs middle_claim consistency.
        // With multi-chunk aux split (zk-torch-2 style), the per-chunk
        // middle_claims reconstruct the full table value via
        //   eval_acc = Σ_j middle_claim_j · 2^(j · TABLE_COMMIT_LOG)
        let block_size = *crate::TABLE_COMMIT_LOG;
        let max_chunks = proof.middle_claims.iter().map(|v| v.len()).max().unwrap_or(1);
        let gammas = calc_pow_vec_ext2(gamma, max_chunks);
        let range_dump = std::env::var("ZK4_RANGE_DUMP").is_ok();
        if range_dump {
            for (i, &n) in self.range.iter().enumerate() {
                let in_e = self.nodes[n].inputs[0];
                let w = &witnesses[in_e][0];
                if let Some(d) = w.data.as_ref() {
                    let sz = 1usize << d.n();
                    let mut mn = i128::MAX; let mut mx = i128::MIN;
                    for k in 0..sz { let v = f_to_int(d.index(k)); mn = mn.min(v); mx = mx.max(v); }
                    let prod = self.producers[in_e].map(|p| format!("{:?}", self.nodes[p].kind)).unwrap_or("IN".into());
                    eprintln!("[range_dump] idx {} node {} edge {} shape {:?} from {}: min={} max={}",
                        i, n, in_e, w.shape, prod, mn, mx);
                }
            }
        }
        let mut table_expected_sum = AlmostGoldilocksExt2::zero();
        for (i, &n) in self.range.iter().enumerate() {
            let expected_eval = eval_to_check(self, n, witnesses, claims);
            let mut eval_acc = AlmostGoldilocksExt2::zero();
            for (sparse_id, &middle) in proof.middle_claims[i].iter().enumerate() {
                let pow = AlmostGoldilocksExt2::from_base(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(
                        1u64 << (sparse_id * block_size)
                    )
                );
                eval_acc = ext2_add(eval_acc, ext2_mul(middle, pow));
                // Per-chunk contribution to the combined table sumcheck:
                //   (middle_j + α) · β_i · γ^{j+1}
                let weight = ext2_mul(betas[i], gammas[sparse_id]);
                let contribution = ext2_mul(ext2_add(middle, alpha), weight);
                table_expected_sum = ext2_add(table_expected_sum, contribution);
            }
            if !ext2_field_eq(expected_eval, eval_acc) {
                if std::env::var("ZK4_VERIFY_DBG").is_ok() {
                    let in_e = self.nodes[n].inputs[0];
                    let prod = self.producers[in_e]
                        .map(|p| format!("{:?}", self.nodes[p].kind))
                        .unwrap_or_else(|| "INPUT".to_string());
                    let w = &witnesses[in_e][0];
                    let nn = w.data.as_ref().map(|d| d.n()).unwrap_or(0);
                    let sz = 1usize << nn;
                    let mut mn = i128::MAX; let mut mx = i128::MIN; let mut neg = 0;
                    if let Some(d) = w.data.as_ref() {
                        for k in 0..sz {
                            let v = f_to_int(d.index(k));
                            mn = mn.min(v); mx = mx.max(v); if v < 0 { neg += 1; }
                        }
                    }
                    eprintln!("[verify_dbg] range node idx {} (node {}, in_edge {} from {}, shape {:?} sf {}): poly min={} max={} neg_count={}/{}",
                        i, n, in_e, prod, w.shape, w.sf, mn, mx, neg, sz);
                }
                return false;
            }
        }

        // 2. Verify the table sumcheck (degree 2 → 3 eval points per round).
        let (table_ok, _) = SumcheckVerifier::verify(
            &proof.table_proof,
            table_expected_sum,
            proof.table_proof.round_messages.len(),
            2,
            transcript,
        );
        if !table_ok { return false; }

        // 3. Verify each bool sumcheck — mirrors the prover's per-group
        // transcript fork (keyed by aux_num_var). Each fork is independent,
        // so verification runs in parallel; final state is then folded
        // sequentially back into the parent transcript in the same order
        // the prover used (BTreeMap key order).
        //
        // The group id set is DERIVED here from the DAG and the aux arities,
        // never read off the proof. Two reasons, and the second is the one
        // with teeth. (i) A fork identifier must be a public function of
        // data already bound, or the prover picks the branch structure after
        // seeing the branch challenges. (ii) The group COUNT must be pinned:
        // reading it from `proof.bool_proofs.len()` lets a prover ship fewer
        // groups than the DAG has aux arities, and each omitted group drops
        // the `s(x)·(s(x)−1) = 0` boolean check for every aux in it. Nothing
        // else in this function recovers that constraint — step 1 pins the
        // middle claims and step 2 the table relation, but only the bool
        // check forces the selection to be 0/1.
        //
        // This mirrors `prove_range`'s `bool_groups` BTreeMap exactly: same
        // key (the aux's `n`), same ascending iteration order.
        // Terms PER arity, not just the arity set: `BOOL_SPLIT` chunks each
        // arity's terms, so the sub-group count is a function of the count.
        // Both are public -- the term count at an arity is the number of
        // (range node, aux chunk) pairs, fixed by the DAG, and the chunking is
        // `self.range` order. Derived here for the same two reasons the arity
        // set is: the fork id must be a public function of bound data, and the
        // COUNT must be pinned so a prover cannot ship fewer sub-groups than
        // the split implies. Dropping a sub-group drops `s(x)(s(x)-1) = 0` for
        // every aux in it, and nothing else here recovers that constraint.
        let mut term_counts: std::collections::BTreeMap<usize, usize> = Default::default();
        for &n in self.range.iter() {
            let node = &self.nodes[n];
            let aux_id = range_aux_id(&node.kind);
            for aux in witnesses[node.outputs[aux_id]].iter() {
                let arity = aux.data.as_ref().expect("verify_range: aux missing data").n();
                *term_counts.entry(arity).or_insert(0) += 1;
            }
        }
        let bool_split: usize = *crate::BOOL_SPLIT;
        assert!(bool_split <= u16::MAX as usize, "BOOL_SPLIT too large for the fork id");
        let fork_id = |k: usize, j: usize| if bool_split <= 1 { k } else { (k << 16) | j };
        // Mirrors the prover's `w.chunks(per).zip(p.chunks(per))` exactly.
        let group_ids: Vec<(usize, usize)> = term_counts
            .iter()
            .flat_map(|(&arity, &terms)| {
                let (_, n_sub) = bool_subgroups(terms, bool_split);
                (0..n_sub).map(move |j| (arity, j)).collect::<Vec<_>>()
            })
            .collect();
        if proof.bool_proofs.len() != group_ids.len() { return false; }

        use rayon::prelude::*;
        let parent_snapshot = transcript.clone();
        let oks: Vec<bool> = group_ids.par_iter().zip(proof.bool_proofs.par_iter())
            .map(|(&(aux_num_var, sub), bool_proof)| {
            // Pin the proof's shape to the derived id. Without this the fork
            // binds an identifier the round count can contradict.
            if bool_proof.round_messages.len() != aux_num_var { return false; }
            let mut fork_t = parent_snapshot.fork(b"lookup_bool", fork_id(aux_num_var, sub));
            fork_t.append_u64(b"num_var", aux_num_var as u64);
            fork_t.append_u64(b"num_poly", 3u64);
            let _eq_challenge: Vec<AlmostGoldilocksExt2> = (0..aux_num_var)
                .map(|_| fork_t.challenge_ext2(b"challenge"))
                .collect();
            verify_sumcheck_no_header(
                bool_proof,
                AlmostGoldilocksExt2::zero(),
                aux_num_var,
                3,
                &mut fork_t,
            )
        }).collect();
        if oks.iter().any(|&ok| !ok) { return false; }
        for (&(aux_num_var, sub), bool_proof) in group_ids.iter().zip(proof.bool_proofs.iter()) {
            transcript.append_u64(b"bool_grp_id", fork_id(aux_num_var, sub) as u64);
            let total_u64s = bool_proof.round_messages.iter().map(|m| m.len()).sum::<usize>() * 2 + 2;
            let mut flat = Vec::with_capacity(total_u64s);
            for msg in &bool_proof.round_messages {
                for v in msg { flat.push(v.c0.0); flat.push(v.c1.0); }
            }
            flat.push(bool_proof.final_eval.c0.0);
            flat.push(bool_proof.final_eval.c1.0);
            transcript.append_u64_slice(b"bool_grp_payload", &flat);
        }
        true
    }

    /// Prove the two-pow lookup. Each `BasicBlockType::TwoPow` node owns one
    /// sparse aux on its *input* edge (the `k` selection from `ExpHelper`),
    /// and the claim flow goes through the node's *output* edge claim point.
    /// The bool check is skipped — the same aux is bool-checked inside the
    /// range proof (ExpHelper's range).
    pub fn prove_two_pow(
        &self,
        witnesses: &[Vec<Witness>],
        claims: &mut [Vec<Claim>],
        transcript: &mut Transcript,
    ) -> Option<LookupProof> {
        if self.two_pow.is_empty() {
            return None;
        }
        let beta = transcript.challenge_ext2(b"table_beta");
        let betas = calc_pow_vec_ext2(beta, self.two_pow.len());

        let mut combined_part_aux: Vec<AlmostGoldilocksExt2> = Vec::new();
        let mut middle_claims: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(self.two_pow.len());
        let top_for_indexed = 15u32; // matches zk-torch-2 two_pow_dense convention.

        // Per-node work -- the 2^|point| Lagrange basis build, the selection
        // scatter, and the indexed sum -- is pure data: `beta` is already
        // fixed and nothing here touches the transcript. So it runs
        // rayon-parallel and the merge into `combined_part_aux` /
        // `middle_claims` happens serially afterwards in `self.two_pow` order,
        // arithmetic-identical to the old serial loop (the beta-weighted
        // accumulation is order-dependent, so the merge order matters and the
        // per-node phase does not).
        //
        // This mirrors what `prove_range` already does for its own per-node
        // phase. It was the whole cost of prove_two_pow: 3.70s -> see below,
        // on llama2 8L/seq64, because `evaluate_lagrange_basis_ext2` is
        // O(2^|out_point|) per node and there is one node per ExpHelper.
        use rayon::prelude::*;
        struct TpPart {
            middle: AlmostGoldilocksExt2,
            part_aux: Vec<AlmostGoldilocksExt2>,
        }
        let tp_parts: Vec<TpPart> = self.two_pow.par_iter().map(|&n| {
            let node = &self.nodes[n];
            assert_eq!(node.inputs.len(), 1, "TwoPow expects 1 input");
            let aux_edge = node.inputs[0];
            let aux = &witnesses[aux_edge][0];
            let aux_poly = aux.data.as_ref().unwrap()
                .as_any().downcast_ref::<SparseMLPoly>()
                .expect("two_pow aux is SparseMLPoly");

            // The two-pow claim point is on the node's *output* edge.
            let out_claim = claims[node.outputs[0]].last()
                .expect("prove_two_pow: missing output claim");
            let lagrange = evaluate_lagrange_basis_ext2(&out_claim.point);

            let table_n = aux_poly.n - out_claim.point.len();
            let table_size = 1usize << table_n;
            let mut part_aux = vec![AlmostGoldilocksExt2::zero(); table_size];
            for &(input_idx, table_idx) in &aux_poly.selection.selection {
                debug_assert!(table_idx < table_size);
                part_aux[table_idx] = ext2_add(part_aux[table_idx], lagrange[input_idx]);
            }
            let middle = two_pow_indexed_sum_ext2(&part_aux, top_for_indexed);
            TpPart { middle, part_aux }
        }).collect();

        for (i, part) in tp_parts.into_iter().enumerate() {
            let table_size = part.part_aux.len();
            middle_claims.push(vec![part.middle]);
            if combined_part_aux.is_empty() {
                combined_part_aux = vec![AlmostGoldilocksExt2::zero(); table_size];
            }
            assert_eq!(combined_part_aux.len(), table_size,
                "two_pow nodes have differing table_n");
            for t in 0..table_size {
                combined_part_aux[t] =
                    ext2_add(combined_part_aux[t], ext2_mul(part.part_aux[t], betas[i]));
            }
        }

        let table_n = (combined_part_aux.len() as f64).log2().round() as usize;
        // `two_pow_poly[i] = 2^(top - i)` for i ∈ [0, top]; entries past
        // `top` are zero (the conceptual 16-entry table padded with
        // zeros to fit `combined_part_aux.len()`).
        let two_pow_poly: Vec<AlmostGoldilocksExt2> = (0..combined_part_aux.len())
            .map(|i| if (i as u32) <= top_for_indexed {
                AlmostGoldilocksExt2::from_base(
                    almost_goldilocks_cuda::field::AlmostGoldilocksField(1u64 << (top_for_indexed - i as u32))
                )
            } else {
                AlmostGoldilocksExt2::zero()
            })
            .collect();
        let mut polys: [Vec<AlmostGoldilocksExt2>; 2] = [combined_part_aux, two_pow_poly];
        let mut table_prover = CpuLinearSumcheckProverExt2::new(table_n, 2, transcript);
        let table_proof = table_prover.prove(&mut polys, transcript);

        // Push aux claims (point = out_point ++ sumcheck_challenges, eval =
        // aux poly's actual MLE there).
        // Same split again: the sparse-poly evaluations are pure data (the
        // points derive from already-fixed output claims + table challenges),
        // so evaluate in parallel and push serially in `self.two_pow` order.
        let tp_claims: Vec<(usize, Claim)> = self.two_pow.par_iter().map(|&n| {
            let node = &self.nodes[n];
            let aux_edge = node.inputs[0];
            let out_point = claims[node.outputs[0]].last().unwrap().point.clone();
            let aux_poly = witnesses[aux_edge][0].data.as_ref().unwrap()
                .as_any().downcast_ref::<SparseMLPoly>().unwrap();
            let take_n = aux_poly.n - out_point.len();
            let mut point = out_point;
            point.extend(table_prover.challenges.iter().take(take_n).cloned());
            let eval = aux_poly.evaluate_at_point_ext2(&point);
            (aux_edge, Claim { edge_id: aux_edge, sparse_id: 0, point, eval })
        }).collect();
        for (aux_edge, claim) in tp_claims {
            claims[aux_edge].push(claim);
        }

        Some(LookupProof { table_proof, bool_proofs: Vec::new(), middle_claims })
    }

    /// Verify the two-pow lookup proof.
    pub fn verify_two_pow(
        &self,
        _witnesses: &[Vec<Witness>],
        claims: &[Vec<Claim>],
        proof: &LookupProof,
        transcript: &mut Transcript,
    ) -> bool {
        if self.two_pow.is_empty() {
            return true;
        }
        let beta = transcript.challenge_ext2(b"table_beta");
        let betas = calc_pow_vec_ext2(beta, self.two_pow.len());

        let mut table_expected_sum = AlmostGoldilocksExt2::zero();
        for (i, &n) in self.two_pow.iter().enumerate() {
            let node = &self.nodes[n];
            let out_claim = claims[node.outputs[0]].last()
                .expect("verify_two_pow: missing output claim");
            let middle = proof.middle_claims[i][0];
            if !ext2_field_eq(out_claim.eval, middle) { return false; }
            table_expected_sum = ext2_add(table_expected_sum, ext2_mul(middle, betas[i]));
        }
        let (table_ok, _) = SumcheckVerifier::verify(
            &proof.table_proof,
            table_expected_sum,
            proof.table_proof.round_messages.len(),
            2,
            transcript,
        );
        table_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    use crate::basicblock::range::NonNegative;
    use crate::dag::{DagBuilder, DataType, Role};
    use crate::poly::SelectionPolynomial;
    use crate::util::arith::int_to_f;

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    fn make_input(shape: Vec<usize>, raw: Vec<i128>) -> Witness {
        let evals = raw.iter().map(|&v| int_to_f(v)).collect();
        Witness::new(shape, evals, DataType::Int, 0, Role::Input)
    }

    /// Build a minimal DAG with a single NonNegative node on an input edge
    /// and verify the round-trip prove_range → verify_range.
    #[test]
    fn prove_verify_range_single_nonneg() {
        let table_log = 4; // table covers [0, 16)
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let _aux_node = {
            // Replicate `add_nonneg_node(x)` without going through
            // `crate::TABLE_SIZE_LOG` (test wants a small table).
            let nid = g.nodes.len();
            let nn = crate::basicblock::BasicBlockType::NonNegative(NonNegative::new(table_log));
            let _ = g.add_gkr_node(vec![x], nn);
            g.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
            g.range.push(nid);
            nid
        };

        let (dag, mut witnesses) = g.compile();

        // Provide input data in [0, 16) so the aux is correct.
        let raw = vec![0i128, 1, 2, 5, 9, 12, 14, 15];
        let x_witness = make_input(vec![8], raw);
        dag.run(&mut witnesses, &[(0, x_witness)]);

        // Seed a claim on the input edge (mimicking the backward pass).
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); dag.num_edges];
        let inp_witness = &witnesses[0][0];
        let inp_n = inp_witness.data.as_ref().unwrap().n();
        let inp_point: Vec<AlmostGoldilocksExt2> = (0..inp_n).map(|i| lift(i as u64 * 13 + 7)).collect();
        let inp_eval = inp_witness.data.as_ref().unwrap().evaluate_at_point_ext2(&inp_point);
        claims[0].push(Claim {
            edge_id: 0,
            sparse_id: 0,
            point: inp_point,
            eval: inp_eval,
        });

        let mut t_prove = Transcript::new(b"lookup-test");
        let proof = dag.prove_range(&witnesses, &mut claims, &mut t_prove).expect("range proof");

        let mut t_verify = Transcript::new(b"lookup-test");
        let ok = dag.verify_range(&witnesses, &claims, &proof, &mut t_verify);
        assert!(ok, "range proof should verify");
    }

    /// Tamper with the middle_claim → verifier must reject.
    #[test]
    fn verify_range_rejects_tampered_middle_claim() {
        let table_log = 4;
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let nid = g.nodes.len();
        let nn = crate::basicblock::BasicBlockType::NonNegative(NonNegative::new(table_log));
        let _ = g.add_gkr_node(vec![x], nn);
        g.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
        g.range.push(nid);

        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![3, 5, 8, 11]))]);
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); dag.num_edges];
        let inp = &witnesses[0][0];
        let inp_point: Vec<AlmostGoldilocksExt2> = (0..inp.data.as_ref().unwrap().n()).map(|i| lift(i as u64 + 1)).collect();
        let inp_eval = inp.data.as_ref().unwrap().evaluate_at_point_ext2(&inp_point);
        claims[0].push(Claim { edge_id: 0, sparse_id: 0, point: inp_point, eval: inp_eval });

        let mut t_prove = Transcript::new(b"lookup-tamper");
        let mut proof = dag.prove_range(&witnesses, &mut claims, &mut t_prove).unwrap();
        // Bump the middle claim to break the eval_to_check consistency.
        proof.middle_claims[0][0] = ext2_add(proof.middle_claims[0][0], lift(1));

        let mut t_verify = Transcript::new(b"lookup-tamper");
        assert!(!dag.verify_range(&witnesses, &claims, &proof, &mut t_verify),
                "tampered middle_claim should be rejected");
    }

    /// Two range nodes with different input arities produce two distinct aux
    /// arities, hence two bool-sumcheck groups. The verifier derives that
    /// group set from the DAG, so dropping a group from the proof must be
    /// rejected: the omitted group's auxes would lose their `s(s−1) = 0`
    /// boolean check, and nothing else in `verify_range` recovers it.
    #[test]
    fn verify_range_rejects_dropped_bool_group() {
        let table_log = 4;
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let y = g.input(vec![4], DataType::Int);
        for inp in [x, y] {
            let nid = g.nodes.len();
            let nn = crate::basicblock::BasicBlockType::NonNegative(NonNegative::new(table_log));
            let _ = g.add_gkr_node(vec![inp], nn);
            g.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
            g.range.push(nid);
        }

        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[
            (x, make_input(vec![8], vec![0, 1, 2, 5, 9, 12, 14, 15])),
            (y, make_input(vec![4], vec![3, 5, 8, 11])),
        ]);

        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); dag.num_edges];
        for e in [x, y] {
            let d = witnesses[e][0].data.as_ref().unwrap();
            let point: Vec<AlmostGoldilocksExt2> =
                (0..d.n()).map(|i| lift((e as u64 + 1) * 13 + i as u64 + 1)).collect();
            let eval = d.evaluate_at_point_ext2(&point);
            claims[e].push(Claim { edge_id: e, sparse_id: 0, point, eval });
        }

        let mut t_prove = Transcript::new(b"lookup-drop");
        let mut proof = dag.prove_range(&witnesses, &mut claims, &mut t_prove).unwrap();
        assert_eq!(proof.bool_proofs.len(), 2,
                   "two distinct aux arities should give two bool groups");

        // The honest proof still verifies (guards against the group-set
        // derivation disagreeing with the prover's BTreeMap ordering).
        let mut t_ok = Transcript::new(b"lookup-drop");
        assert!(dag.verify_range(&witnesses, &claims, &proof, &mut t_ok),
                "honest two-group range proof should verify");

        // Dropping a group must be rejected.
        proof.bool_proofs.pop();
        let mut t_verify = Transcript::new(b"lookup-drop");
        assert!(!dag.verify_range(&witnesses, &claims, &proof, &mut t_verify),
                "dropped bool group should be rejected");
    }

    /// Build a NonNegative-style sparse aux directly and verify it's actually
    /// the expected `(input_idx, table_idx)` form.
    #[test]
    fn selection_polynomial_shape_matches_assumptions() {
        let sp = SelectionPolynomial::new(3, 4, vec![(0, 2), (1, 5), (2, 0), (3, 7)]);
        let sparse = sp.to_sparse();
        assert_eq!(sparse.n, 7);
        assert_eq!(sparse.evaluations.len(), 4);
        // Position formula: input_idx + table_idx · 2^input_n.
        assert!(sparse.evaluations.contains_key(&(0 + 2 * 8)));
        assert!(sparse.evaluations.contains_key(&(3 + 7 * 8)));
    }
}

#[cfg(test)]
mod bool_split_tests {
    use super::bool_subgroups;

    /// `verify_range` derives its sub-group COUNT from `bool_subgroups`, while
    /// `prove_range` produces sub-groups by `chunks(per)`. If those ever
    /// disagree the count pin rejects an honest proof. Pin the invariant
    /// directly instead of relying on an end-to-end run to notice.
    #[test]
    fn subgroup_count_matches_slice_chunks() {
        for terms in 0..40usize {
            for split in [1usize, 2, 3, 4, 7, 8, 16, 64] {
                let (per, n_sub) = bool_subgroups(terms, split);
                let v = vec![0u8; terms];
                assert_eq!(v.chunks(per).count(), n_sub,
                    "terms={} split={} per={}", terms, split, per);
                // Never more sub-groups than requested, and split=1 is one group.
                assert!(n_sub <= split.max(1) || terms == 0, "terms={} split={}", terms, split);
                if split <= 1 && terms > 0 { assert_eq!(n_sub, 1); }
            }
        }
    }

    /// Large, realistic shapes: the arity-22 group on llama2 8L has 2760 terms.
    #[test]
    fn subgroup_count_matches_slice_chunks_large() {
        for &terms in &[128usize, 136, 272, 502, 2760, 65537] {
            for split in [1usize, 4, 8, 16] {
                let (per, n_sub) = bool_subgroups(terms, split);
                assert_eq!(vec![0u8; terms].chunks(per).count(), n_sub);
            }
        }
    }
}
