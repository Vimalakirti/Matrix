//! Integration of the [`crate::fold`] tree with [`Dag::prove`] /
//! [`Dag::verify`] (plan §8.2 step 12 wiring). After the sumcheck-side
//! backward pass and the per-edge opening reducer (step 4), each
//! committed dense edge owns a single combined claim `(R, v)`. To open
//! against the per-bit-plane Ajtai commitments we:
//!
//! 1. Bit-decompose each committed witness into `b` binary planes.
//! 2. Build one `FoldInstance` per plane with the plane's commitment +
//!    the plane's MLE eval at `R`.
//! 3. Run the fold tree on the union of all leaves.
//!
//! The verifier mirrors this by checking the signed two's-complement
//! reconstruction `v == Σ_{i=0..b-2} 2^i · y_i − 2^{b-1} · y_{b-1}` per
//! edge, then replaying the fold tree against the prover-supplied
//! per-plane (commitment, claim) leaves.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use serde::{Deserialize, Serialize};

use crate::commit::{bit_decompose::decompose_and_pack, AjtaiKey, GpuAjtaiStore};
use crate::dag::{Dag, DagProof, EdgeId, PolyType, Role, Witness};

/// When `ZK4_DEFER_CONSTANTS=1`, every `Role::Constant` edge (pre-committed
/// model weights) is dropped from the fold-tree leaf set AND exposed as a
/// [`DeferredClaim`] in `FoldOpeningProof.deferred_constant_claims`. The
/// per-proof verifier accepts the proof modulo these deferred claims; a
/// downstream streaming accumulator (Phase 3) binds them to the underlying
/// PCS commitments via a reducer sumcheck per weight, deferring the actual
/// Ajtai opening until session end. Used in the multi-inference serving
/// scenario where opening cost amortizes across many proofs of the same
/// model. NEVER use without a streaming accumulator — a deferred proof on
/// its own is incomplete.
///
/// Replaces the earlier unsound `ZK4_SKIP_CONSTANT_OPENS` knob, which
/// silently dropped opens with no binding. See
/// `~/.claude/.../project_zk_soundness_audit.md` for the audit.
fn defer_constants() -> bool {
    // Read fresh on each call (called O(1) per proof) so tests can toggle
    // the mode within a single process. The env-var lookup is negligible
    // relative to a full proof, so caching this is not worth its cost in
    // testability.
    std::env::var("ZK4_DEFER_CONSTANTS").ok().as_deref() == Some("1")
}
use crate::fold::{
    prove_fold_tree, verify_fold_tree, FoldData, FoldInstance, FoldTreeProof,
};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, ext2_sub, get_n};

/// Wire-form per-edge plane reveals. The verifier re-derives the
/// committed-edge combined claim from these evals via the signed
/// two's-complement formula and uses them as the leaf claim values for
/// the fold tree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgePlaneEvals {
    pub edge_id: EdgeId,
    /// For SPARSE multi-chunk auxes (zk-torch-2 z-t-2 split), the chunk
    /// index `j ∈ [0, K)`. For dense edges, always `0`.
    pub sparse_id: usize,
    /// Edge's native commit arity (`= ec.arity`, ≥ 6). The
    /// `combined_point` lives at this arity.
    pub arity: usize,
    pub combined_point: Vec<AlmostGoldilocksExt2>,
    pub combined_eval: AlmostGoldilocksExt2,
    /// For DENSE: one eval per bit plane (length `b`).
    /// For SPARSE: a single eval (this chunk's MLE at `combined_point`).
    pub plane_evals: Vec<AlmostGoldilocksExt2>,
    pub is_sparse: bool,
}

/// A `Role::Constant` edge whose PCS opening is deferred to a downstream
/// streaming accumulator. The per-proof prover declares `(point, eval)`,
/// the per-proof verifier checks consistency with the sumcheck terminal
/// claim, and the accumulator (Phase 3) reduces a stream of `DeferredClaim`s
/// for the same `edge_id` into a single claim opened once at session end.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeferredClaim {
    pub edge_id: EdgeId,
    /// Extended (zero-padded) point at the edge's native commit arity.
    pub point: Vec<AlmostGoldilocksExt2>,
    pub eval: AlmostGoldilocksExt2,
    /// The edge's native commit arity (`= store.get(edge_id).unwrap().arity`).
    pub arity: usize,
}

/// Extension to [`DagProof`] carrying the fold-tree proof + per-edge
/// plane reveals. Held separately rather than embedded in `DagProof`
/// to keep the step-4 sumcheck-only path independently usable.
///
/// When `ZK4_DEFER_CONSTANTS=1`, `deferred_constant_claims` is populated
/// with the skipped `Role::Constant` edges; otherwise it is empty and the
/// proof is monolithic (sound on its own).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldOpeningProof {
    pub edge_plane_evals: Vec<EdgePlaneEvals>,
    pub fold_tree: FoldTreeProof,
    #[serde(default)]
    pub deferred_constant_claims: Vec<DeferredClaim>,
}

/// Per-proof verification outcome when running in deferred-constants mode.
/// `ok=true` means the sumcheck-side chain, the fold-tree for non-deferred
/// edges, AND the deferred-claim consistency checks all pass — but the
/// deferred claims are NOT yet bound to a PCS opening. A downstream
/// streaming accumulator (Phase 3) must consume `claims` to finish the
/// soundness chain. A standalone `ok=true` from this struct is necessary
/// but not sufficient for accepting the inference proof.
#[derive(Clone, Debug)]
pub struct DeferredResult {
    pub ok: bool,
    pub claims: Vec<DeferredClaim>,
}

impl Dag {
    /// Run the full prover pipeline: sumcheck-side proof (step 4) plus
    /// per-edge bit-plane decomposition + fold tree (step 12).
    ///
    /// Honors the `ZK4_DEFER_CONSTANTS=1` env var; tests should call
    /// [`Dag::prove_with_fold_tree_modes`] directly to parameterize.
    pub fn prove_with_fold_tree(
        &self,
        witnesses: &[Vec<Witness>],
        store: &GpuAjtaiStore,
        transcript: &mut Transcript,
    ) -> (DagProof, FoldOpeningProof) {
        self.prove_with_fold_tree_modes(witnesses, store, transcript, defer_constants())
    }

    /// Parameterized prover. Mirror of [`Dag::prove_with_fold_tree`] but
    /// takes `defer_const` explicitly so callers (tests, multi-input bench
    /// harness) can run defer mode without touching process env state.
    pub fn prove_with_fold_tree_modes(
        &self,
        witnesses: &[Vec<Witness>],
        store: &GpuAjtaiStore,
        transcript: &mut Transcript,
        defer_const: bool,
    ) -> (DagProof, FoldOpeningProof) {
        let trace_mem = std::env::var("ZK4_PROVE_MEM_TRACE").ok().as_deref() == Some("1");
        let mem = || almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);
        if trace_mem { eprintln!("    [prove mem] enter: {} MiB", mem()); }
        // Auto-route to partitioned prove if the DAG has boundaries set
        // (`Dag::set_partition_boundaries(N)` was called with N > 1).
        // Otherwise use the single-partition prove. Both produce the
        // same DagProof shape (boundary_claims empty in the single
        // path, populated in the partitioned path).
        let t_backward = std::time::Instant::now();
        let dag_proof = if !self.boundary_edges.is_empty() {
            let partitions = crate::dag::partition::partition_dag(self, &self.boundary_edges);
            self.prove_partitioned(witnesses, &partitions, transcript)
        } else {
            self.prove(witnesses, transcript)
        };
        let backward_elapsed = t_backward.elapsed();
        if trace_mem { eprintln!("    [prove mem] after backward: {} MiB", mem()); }
        let t_leaf_build = std::time::Instant::now();

        // Build the leaf set: one FoldInstance per (committed edge, plane).
        //
        // Per-edge work is INDEPENDENT (the body reads only `store`, the
        // claim, and the cached packed planes — no transcript consumption,
        // since `extend_claim_point_to` is pure zero-pad). The transcript
        // appends happen in a deterministic serial pass after this loop,
        // walking `edge_plane_evals` and `deferred_constant_claims` in
        // edge-id order. So we can shard the per-edge work across the GPU
        // pool with rayon + set_device — same pattern as commit_edges.
        let key = &store.key;
        let device_pool = crate::fold::tree::gpu_device_pool();
        let n_dev = device_pool.len().max(1);

        struct EdgeBatch {
            leaves: Vec<FoldInstance>,
            epe: Option<EdgePlaneEvals>,        // dense: one EPE per edge
            sparse_epes: Vec<EdgePlaneEvals>,   // sparse: one EPE per sparse_id
            deferred: Option<DeferredClaim>,
        }

        use rayon::prelude::*;
        let edge_batches: Vec<EdgeBatch> = (0..self.num_edges).into_par_iter().map(|e| {
            let mut batch = EdgeBatch {
                leaves: Vec::new(),
                epe: None,
                sparse_epes: Vec::new(),
                deferred: None,
            };
            use std::sync::atomic::Ordering;
            let device = device_pool[e % n_dev];
            let t_sd = std::time::Instant::now();
            let _ = almost_goldilocks_cuda::set_device(device);
            LEAF_BUILD_SETDEV_US.fetch_add(t_sd.elapsed().as_micros() as u64, Ordering::Relaxed);
            let ec = match store.get(e) { Some(c) => c, None => return batch };
            let ep = &dag_proof.edge_proofs[e];
            // Per `ZK4_DEFER_CONSTANTS`: route `Role::Constant` edges to
            // `deferred_constant_claims` (to be bound by the downstream
            // streaming accumulator) instead of building a fold-tree leaf.
            // The sumcheck terminal claim — `ep.claims.last()` — carries
            // the (point, eval) pair the verifier checks for consistency
            // and the accumulator will later reduce + open.
            if defer_const && witnesses[e][0].role == Role::Constant {
                let t_df = std::time::Instant::now();
                let claim = match ep.claims.last() {
                    Some(c) if !c.point.is_empty() => c,
                    _ => return batch,
                };
                // Store the un-extended sumcheck terminal point (length
                // `n`, not `arity`). The streaming reducer combines
                // length-n claims natively; finalize zero-pads to `arity`
                // when building fold-tree leaves. The `arity` field
                // records what to pad to at finalize time. Note that
                // `extend_claim_point_to` is a transcript no-op today,
                // so skipping it here doesn't shift the transcript
                // state relative to the verifier.
                batch.deferred = Some(DeferredClaim {
                    edge_id: e,
                    point: claim.point.clone(),
                    eval: claim.eval,
                    arity: ec.arity,
                });
                LEAF_BUILD_DEFER_US.fetch_add(t_df.elapsed().as_micros() as u64, Ordering::Relaxed);
                LEAF_BUILD_N_DEFER.fetch_add(1, Ordering::Relaxed);
                return batch;
            }

            if ec.is_sparse {
                // SPARSE multi-chunk path.
                let cached_planes = store.get_planes(e);
                for sparse_id in 0..ec.planes.len() {
                    let claim = match ep.claims.iter()
                        .find(|c| c.sparse_id == sparse_id && !c.point.is_empty())
                    {
                        Some(c) => c.clone(),
                        None => continue,
                    };
                    let extended_point = extend_claim_point_to(
                        &claim.point, ec.arity,
                    );
                    // Reuse the cached packed plane from commit if present;
                    // otherwise regenerate JUST the bitmask from the sparse
                    // positions (cheap — no eq eval). The cache may be
                    // absent/empty when ZK4_DROP_SPARSE_PLANE_CACHE skips
                    // building+caching the dense bitmask at commit time to
                    // halve sparse-aux host memory (cache + leaf copies).
                    let t_scl = std::time::Instant::now();
                    let packed: Vec<u64> = match cached_planes {
                        Some(planes) if !planes[sparse_id].is_empty() => planes[sparse_id].clone(),
                        _ => pack_sparse_plane(&witnesses[e][sparse_id], ec.arity),
                    };
                    LEAF_BUILD_SPARSE_CLONE_US.fetch_add(t_scl.elapsed().as_micros() as u64, Ordering::Relaxed);
                    let t_sev = std::time::Instant::now();
                    // The committed sparse chunk's MLE at `extended_point`
                    // is PROVABLY `claim.eval`: the verifier requires
                    // `plane_evals[0] == combined_eval == dag_claim.eval`
                    // (fold_integration.rs verify sparse branch), and the
                    // fold-tree opening cryptographically binds this leaf's
                    // `claim_val` to the committed plane — a wrong value
                    // fails `f*(R*) == y*`. So reuse the range-proof claim
                    // eval directly instead of rebuilding a full 2^arity eq
                    // table per chunk (was ~131 s thread-time / ~1.1 s wall
                    // on Llama 8L: 2273 single-plane evals, 1275 at arity 22).
                    // ZK4_VERIFY_SPARSE_EVAL=1 re-derives on GPU and asserts
                    // bit-equality (kept as an audit knob).
                    let plane_eval = claim.eval;
                    if std::env::var("ZK4_VERIFY_SPARSE_EVAL").ok().as_deref() == Some("1") {
                        let total = 1usize << ec.arity;
                        let gpu = if ec.arity >= 12 {
                            almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_device(
                                &extended_point, &[packed.as_slice()],
                            ).expect("eval_binary_planes_device failed")[0]
                        } else {
                            let eq = crate::poly::evaluate_lagrange_basis_ext2(&extended_point);
                            eval_binary_with_shared_eq(&packed, &eq, total)
                        };
                        assert!(ext2_field_eq(gpu, claim.eval),
                            "sparse plane eval != claim.eval (edge {} chunk {})", e, sparse_id);
                    }
                    LEAF_BUILD_SPARSE_EVAL_US.fetch_add(t_sev.elapsed().as_micros() as u64, Ordering::Relaxed);
                    LEAF_BUILD_N_SPARSE.fetch_add(1, Ordering::Relaxed);
                    let leaf = FoldInstance {
                        commitment: ec.planes[sparse_id].clone(),
                        data: FoldData::Binary(packed),
                        arity: ec.arity,
                        claim_pt: extended_point.clone(),
                        claim_val: plane_eval,
                    };
                    batch.leaves.push(leaf);
                    batch.sparse_epes.push(EdgePlaneEvals {
                        edge_id: e,
                        sparse_id,
                        arity: ec.arity,
                        combined_point: extended_point,
                        combined_eval: claim.eval,
                        plane_evals: vec![plane_eval],
                        is_sparse: true,
                    });
                }
                return batch;
            }

            // DENSE path: one combined claim → b bit planes.
            let combined_claim = match ep.claims.last() {
                Some(c) if !c.point.is_empty() => c.clone(),
                _ => return batch,
            };
            let extended_point = extend_claim_point_to(
                &combined_claim.point, ec.arity,
            );
            // Reuse cached planes from commit (bit_decompose is offline for
            // constants, online for activations — either way, already done).
            let t_cl = std::time::Instant::now();
            let planes_packed: Vec<Vec<u64>> = match store.get_planes(e) {
                Some(planes) => planes.clone(),
                None => {
                    let w = &witnesses[e][0];
                    decompose_witness_for_fold_native(
                        w, key, ec.arity, &extended_point,
                    ).0
                }
            };
            LEAF_BUILD_CLONE_US.fetch_add(
                t_cl.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed);
            // Per-plane MLE eval: build eq on device + batched selective
            // add for arities ≥ 12 (where the eq table is large enough
            // that GPU bandwidth beats CPU). Smaller arities stay on
            // CPU — the kernel-launch overhead dominates there.
            let total = 1usize << ec.arity;
            let t_ev = std::time::Instant::now();
            let plane_evals: Vec<_> = if ec.arity >= 12 {
                let plane_refs: Vec<&[u64]> = planes_packed.iter().map(|p| p.as_slice()).collect();
                almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_device(
                    &extended_point, &plane_refs,
                ).expect("eval_binary_planes_device failed")
            } else {
                let eq = crate::poly::evaluate_lagrange_basis_ext2(&extended_point);
                planes_packed.iter().map(|p| {
                    eval_binary_with_shared_eq(p, &eq, total)
                }).collect()
            };
            LEAF_BUILD_EVAL_US.fetch_add(
                t_ev.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed);
            LEAF_BUILD_N_DENSE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Branch on the EDGE's effective base (set per-witness in commit:
            // Constants get key.base, auxiliaries stay binary). Binary edges
            // emit b binary leaves; higher-base edges emit b_β Digit leaves.
            let edge_base = ec.base;
            if edge_base == 2 {
                let reconstructed = reconstruct_signed_two_complement(
                    &plane_evals, /*is_sparse=*/ false, key.b, 2,
                );
                assert!(
                    ext2_field_eq(reconstructed, combined_claim.eval),
                    "edge {}: per-plane evals don't reconstruct combined claim", e,
                );
                for (pi, packed) in planes_packed.into_iter().enumerate() {
                    let leaf = FoldInstance {
                        commitment: ec.planes[pi].clone(),
                        data: FoldData::Binary(packed),
                        arity: ec.arity,
                        claim_pt: extended_point.clone(),
                        claim_val: plane_evals[pi],
                    };
                    batch.leaves.push(leaf);
                }
                batch.epe = Some(EdgePlaneEvals {
                    edge_id: e, sparse_id: 0, arity: ec.arity,
                    combined_point: extended_point, combined_eval: combined_claim.eval,
                    plane_evals, is_sparse: false,
                });
            } else {
                // Higher radix: group the b binary bit-plane evals into b_β digit-plane
                // evals using y_{d_j} = Σ_{k<m-1} 2^k·y_bit - 2^{m-1}·y_top (for top,
                // else just Σ 2^k·y_bit). The signed-top weighting is consistent with
                // c_{d_top} = ... − 2^{m-1}·c_bit_{b-1} on the commit side, so
                // Σ β^j · y_{d_j} reconstructs the same value as the binary scheme.
                let k_log = edge_base.trailing_zeros() as usize;
                let b_beta = crate::commit::bit_decompose::digit_planes_for(key.b, edge_base);
                let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
                let mut digit_evals: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(b_beta);
                for j in 0..b_beta {
                    let lo = j * k_log;
                    let hi = ((j + 1) * k_log).min(key.b);
                    let m = hi - lo;
                    let is_top = j == b_beta - 1;
                    let mut y = AlmostGoldilocksExt2::zero();
                    let mut pow = AlmostGoldilocksExt2::one();
                    for kk in 0..m {
                        let bit_y = plane_evals[lo + kk];
                        if is_top && kk == m - 1 {
                            y = ext2_sub(y, ext2_mul(pow, bit_y));
                        } else {
                            y = ext2_add(y, ext2_mul(pow, bit_y));
                        }
                        pow = ext2_mul(pow, two);
                    }
                    digit_evals.push(y);
                }
                let reconstructed = reconstruct_signed_two_complement(
                    &digit_evals, /*is_sparse=*/ false, key.b, edge_base,
                );
                assert!(
                    ext2_field_eq(reconstructed, combined_claim.eval),
                    "edge {}: digit-plane evals don't reconstruct combined claim (base={})",
                    e, key.base,
                );
                // Move planes_packed into Option slots so each bit-plane is
                // consumed exactly once into its digit leaf.
                let mut slots: Vec<Option<Vec<u64>>> =
                    planes_packed.into_iter().map(Some).collect();
                for j in 0..b_beta {
                    let lo = j * k_log;
                    let hi = ((j + 1) * k_log).min(key.b);
                    let m = hi - lo;
                    let is_top = j == b_beta - 1;
                    let mut bit_planes: Vec<Vec<u64>> = Vec::with_capacity(m);
                    for kk in 0..m {
                        bit_planes.push(slots[lo + kk].take().expect("plane already consumed"));
                    }
                    let leaf = FoldInstance {
                        commitment: ec.planes[j].clone(),
                        data: FoldData::Digit {
                            base: key.base,
                            bit_planes,
                            negate_top_bit: is_top,
                        },
                        arity: ec.arity,
                        claim_pt: extended_point.clone(),
                        claim_val: digit_evals[j],
                    };
                    batch.leaves.push(leaf);
                }
                batch.epe = Some(EdgePlaneEvals {
                    edge_id: e, sparse_id: 0, arity: ec.arity,
                    combined_point: extended_point, combined_eval: combined_claim.eval,
                    plane_evals: digit_evals, is_sparse: false,
                });
            }
            batch
        }).collect();
        let leaf_par_elapsed = t_leaf_build.elapsed();
        // Restore primary device for any post-loop CUDA work.
        let _ = almost_goldilocks_cuda::set_device(device_pool[0]);

        // Serial flatten in edge-id order — preserves the deterministic
        // leaf ordering the fold tree binds to, and the transcript-bound
        // `edge_plane_evals` / `deferred_constant_claims` order the
        // verifier replays.
        let mut leaves: Vec<FoldInstance> = Vec::new();
        let mut edge_plane_evals: Vec<EdgePlaneEvals> = Vec::new();
        let mut deferred_constant_claims: Vec<DeferredClaim> = Vec::new();
        for mut batch in edge_batches {
            leaves.append(&mut batch.leaves);
            if let Some(epe) = batch.epe { edge_plane_evals.push(epe); }
            for epe in batch.sparse_epes { edge_plane_evals.push(epe); }
            if let Some(dc) = batch.deferred { deferred_constant_claims.push(dc); }
        }

        // Absorb deferred constant claims first (so the streaming
        // accumulator's later challenges, derived from per-proof
        // transcript state, are bound to these (point, eval) tuples).
        // Empty when not in defer mode.
        for dc in &deferred_constant_claims {
            transcript.append_u64(b"ft_deferred_edge", dc.edge_id as u64);
            transcript.append_u64(b"ft_deferred_arity", dc.arity as u64);
            for p in &dc.point { transcript.append_ext2(b"ft_deferred_point", p); }
            transcript.append_ext2(b"ft_deferred_eval", &dc.eval);
        }
        // Absorb the per-edge plane reveals into the transcript before
        // sampling fold-tree challenges, so the verifier replays the
        // same sequence.
        for epe in &edge_plane_evals {
            transcript.append_u64(b"ft_edge", epe.edge_id as u64);
            transcript.append_u64(b"ft_edge_is_sparse", epe.is_sparse as u64);
            transcript.append_ext2(b"ft_combined_eval", &epe.combined_eval);
            for e in &epe.plane_evals { transcript.append_ext2(b"ft_plane_eval", e); }
        }

        let leaf_build_elapsed = t_leaf_build.elapsed();
        let leaf_count = leaves.len();
        let arity_summary: std::collections::BTreeMap<usize, usize> = {
            let mut m = std::collections::BTreeMap::new();
            for l in &leaves { *m.entry(l.arity).or_insert(0) += 1; }
            m
        };
        if trace_mem { eprintln!("    [prove mem] after leaf build: {} MiB", mem()); }
        // Packed-PCS shadow run (ZK4_PCS=packed). Proves the SAME leaf set with
        // the packed path and self-verifies, so the two openings can be compared
        // on identical inputs in one prover run. The fold tree still produces the
        // shipped proof, so this measures without changing what is emitted.
        if std::env::var("ZK4_PCS").ok().as_deref() == Some("packed") {
            let meta: Vec<(usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)> = leaves
                .iter()
                .map(|l| (l.arity, l.claim_pt.clone(), l.claim_val))
                .collect();
            let t = std::time::Instant::now();
            let mut tp = transcript.clone();
            let interleaved = std::env::var("ZK4_PCS").ok().as_deref() == Some("packed");
            let built = if interleaved {
                crate::pcs::integration::prove_packed_interleaved(&leaves, key.seed, &mut tp)
            } else {
                crate::pcs::integration::prove_packed(&leaves, key.seed, &mut tp)
            };
            match built {
                Some(pp) => {
                    let prove_s = t.elapsed().as_secs_f64();
                    let t = std::time::Instant::now();
                    let mut tv = transcript.clone();
                    let ok = if interleaved {
                        crate::pcs::integration::verify_packed_interleaved(&meta, &pp, &mut tv)
                    } else {
                        crate::pcs::integration::verify_packed(&meta, &pp, &mut tv)
                    };
                    eprintln!(
                        "[packed_pcs] prove {:.3}s  verify {:.3}s  openings {}  verified {}",
                        prove_s, t.elapsed().as_secs_f64(), pp.openings.len(), ok
                    );
                }
                None => eprintln!(
                    "[packed_pcs] SKIPPED: leaf set contains a representation the packed \
                     path does not handle (non-binary digit or ternary leaf)"
                ),
            }
        }

        let t_fold_tree = std::time::Instant::now();
        let fold_tree = prove_fold_tree(leaves, key.seed, transcript);
        let fold_tree_elapsed = t_fold_tree.elapsed();
        if trace_mem { eprintln!("    [prove mem] after fold_tree:  {} MiB", mem()); }

        // Print timing breakdown (gated by env var so we don't spam tests).
        if std::env::var("ZK4_TIMING").ok().as_deref() == Some("1") {
            use std::sync::atomic::Ordering;
            let _ = LEAF_BUILD_BIT_DECOMPOSE_US.swap(0, Ordering::Relaxed);
            let _ = LEAF_BUILD_PLANE_EVAL_US.swap(0, Ordering::Relaxed);
            let clone_us = LEAF_BUILD_CLONE_US.swap(0, Ordering::Relaxed);
            let eval_us = LEAF_BUILD_EVAL_US.swap(0, Ordering::Relaxed);
            let setdev_us = LEAF_BUILD_SETDEV_US.swap(0, Ordering::Relaxed);
            let defer_us = LEAF_BUILD_DEFER_US.swap(0, Ordering::Relaxed);
            let sp_eval_us = LEAF_BUILD_SPARSE_EVAL_US.swap(0, Ordering::Relaxed);
            let sp_clone_us = LEAF_BUILD_SPARSE_CLONE_US.swap(0, Ordering::Relaxed);
            let n_defer = LEAF_BUILD_N_DEFER.swap(0, Ordering::Relaxed);
            let n_sparse = LEAF_BUILD_N_SPARSE.swap(0, Ordering::Relaxed);
            let n_dense = LEAF_BUILD_N_DENSE.swap(0, Ordering::Relaxed);
            let ms = |us: u64| us as f64 / 1_000.0;
            eprintln!("[prove] backward (sumcheck + reducer): {:?}",
                      backward_elapsed);
            eprintln!("[prove] leaf build total ({} leaves, {} const edges deferred): {:?}",
                      leaf_count, deferred_constant_claims.len(), leaf_build_elapsed);
            eprintln!("[prove]   par loop {:?}, flatten+absorb {:?}",
                      leaf_par_elapsed, leaf_build_elapsed - leaf_par_elapsed);
            eprintln!("[prove]   thread-time: set_device {:.0}ms ({} edges), defer {:.0}ms ({} edges), \
                       dense[clone {:.0}ms eval {:.0}ms] ({} edges), sparse[clone {:.0}ms eval {:.0}ms] ({} leaves)",
                      ms(setdev_us), self.num_edges,
                      ms(defer_us), n_defer,
                      ms(clone_us), ms(eval_us), n_dense,
                      ms(sp_clone_us), ms(sp_eval_us), n_sparse);
            eprintln!("[prove]   leaves by arity: {:?}", arity_summary);
            eprintln!("[prove] fold tree: {:?}", fold_tree_elapsed);
        }

        (dag_proof, FoldOpeningProof {
            edge_plane_evals,
            fold_tree,
            deferred_constant_claims,
        })
    }

    /// Mirror of `prove_with_fold_tree`: sumcheck-side verify + per-edge
    /// plane-reveal consistency + fold-tree verify.
    pub fn verify_with_fold_tree(
        &self,
        witnesses: &[Vec<Witness>],
        store: &GpuAjtaiStore,
        dag_proof: &DagProof,
        fold_proof: &FoldOpeningProof,
        transcript: &mut Transcript,
    ) -> bool {
        let sumcheck_ok = if !self.boundary_edges.is_empty() {
            let partitions = crate::dag::partition::partition_dag(self, &self.boundary_edges);
            self.verify_partitioned(witnesses, dag_proof, &partitions, transcript)
        } else {
            self.verify(witnesses, dag_proof, transcript)
        };
        if std::env::var("ZK4_VERIFY_DBG").is_ok() {
            eprintln!("[verify_dbg] sumcheck_ok = {}", sumcheck_ok);
        }
        if !sumcheck_ok { return false; }
        let key = &store.key;

        // Deferred-constant claims: validate each declared (point, eval)
        // matches the DAG's sumcheck terminal claim for the same edge.
        // Empty when not in defer mode. NB: `extend_claim_point_to`
        // currently doesn't consume transcript challenges, so processing
        // these before the plane-evals loop doesn't shift transcript
        // state relative to the prover (which interleaves both kinds in
        // edge_id order). If `extend_claim_point_to` is ever changed to
        // consume challenges, this loop and the plane-evals loop must be
        // merged into a single edge_id-ordered iteration to match the
        // prover.
        for dc in &fold_proof.deferred_constant_claims {
            let ec = match store.get(dc.edge_id) { Some(c) => c, None => return false };
            if dc.arity != ec.arity { return false; }
            let ep = &dag_proof.edge_proofs[dc.edge_id];
            let dag_claim = match ep.claims.last() {
                Some(c) if !c.point.is_empty() => c,
                _ => return false,
            };
            // Deferred (point, eval) must exactly match the DAG's
            // sumcheck terminal claim — un-extended. Finalize handles
            // the zero-extension to commit arity when building fold-tree
            // leaves.
            if dag_claim.eval != dc.eval { return false; }
            if dag_claim.point != dc.point { return false; }
        }

        // Per-edge / per-chunk plane reveals consistency. Iterate in the
        // same order the prover did so the per-entry `extend_claim_point_to`
        // transcript challenges align.
        for epe in &fold_proof.edge_plane_evals {
            let ec = match store.get(epe.edge_id) { Some(c) => c, None => return false };
            if epe.arity != ec.arity { return false; }
            let ep = &dag_proof.edge_proofs[epe.edge_id];
            if epe.is_sparse {
                // Sparse multi-chunk: one EdgePlaneEvals per chunk.
                // plane_evals.len() == 1; matches the chunk's MLE eval.
                if epe.plane_evals.len() != 1 { return false; }
                if epe.plane_evals[0] != epe.combined_eval { return false; }
                let dag_claim = match ep.claims.iter()
                    .find(|c| c.sparse_id == epe.sparse_id && !c.point.is_empty())
                {
                    Some(c) => c,
                    None => return false,
                };
                if dag_claim.eval != epe.combined_eval { return false; }
                if dag_claim.point.len() > epe.combined_point.len() { return false; }
                if dag_claim.point != epe.combined_point[..dag_claim.point.len()] { return false; }
                let expected_extension = extend_claim_point_to(
                    &dag_claim.point, epe.arity,
                );
                if expected_extension != epe.combined_point { return false; }
            } else {
                // Dense: reconstruction at the EDGE's effective base.
                let edge_base = store.get(epe.edge_id).map(|ec| ec.base).unwrap_or(2);
                let expected_plane_count = if edge_base == 2 {
                    key.b
                } else {
                    crate::commit::bit_decompose::digit_planes_for(key.b, edge_base)
                };
                if epe.plane_evals.len() != expected_plane_count { return false; }
                let reconstructed = reconstruct_signed_two_complement(
                    &epe.plane_evals, false, key.b, edge_base,
                );
                if !ext2_field_eq(reconstructed, epe.combined_eval) { return false; }
                let dag_claim = match ep.claims.last() {
                    Some(c) if !c.point.is_empty() => c,
                    _ => return false,
                };
                if dag_claim.eval != epe.combined_eval { return false; }
                if dag_claim.point.len() > epe.combined_point.len() { return false; }
                if dag_claim.point != epe.combined_point[..dag_claim.point.len()] { return false; }
                let expected_extension = extend_claim_point_to(
                    &dag_claim.point, epe.arity,
                );
                if expected_extension != epe.combined_point { return false; }
            }
        }

        // Absorb deferred constant claims first (matches prover order).
        for dc in &fold_proof.deferred_constant_claims {
            transcript.append_u64(b"ft_deferred_edge", dc.edge_id as u64);
            transcript.append_u64(b"ft_deferred_arity", dc.arity as u64);
            for p in &dc.point { transcript.append_ext2(b"ft_deferred_point", p); }
            transcript.append_ext2(b"ft_deferred_eval", &dc.eval);
        }
        // Absorb the same plane reveals as the prover did.
        for epe in &fold_proof.edge_plane_evals {
            transcript.append_u64(b"ft_edge", epe.edge_id as u64);
            transcript.append_u64(b"ft_edge_is_sparse", epe.is_sparse as u64);
            transcript.append_ext2(b"ft_combined_eval", &epe.combined_eval);
            for e in &epe.plane_evals { transcript.append_ext2(b"ft_plane_eval", e); }
        }

        // Rebuild leaf metadata for the (per-arity-bucketed) fold tree.
        // For dense: K=b planes per edge entry. For sparse: 1 plane per
        // edge entry (each entry IS one chunk).
        let mut leaves_meta: Vec<(_, _, _, _)> = Vec::new();
        for epe in &fold_proof.edge_plane_evals {
            let ec = store.get(epe.edge_id).unwrap();
            if epe.is_sparse {
                leaves_meta.push((
                    ec.planes[epe.sparse_id].clone(),
                    epe.arity,
                    epe.combined_point.clone(),
                    epe.plane_evals[0],
                ));
            } else {
                for (pi, y) in epe.plane_evals.iter().enumerate() {
                    leaves_meta.push((
                        ec.planes[pi].clone(),
                        epe.arity,
                        epe.combined_point.clone(),
                        *y,
                    ));
                }
            }
        }
        verify_fold_tree(&leaves_meta, &fold_proof.fold_tree, transcript).is_ok()
    }

    /// Streaming-mode counterpart to [`verify_with_fold_tree`]. Returns a
    /// [`DeferredResult`] carrying the deferred claims that the caller
    /// must feed into a streaming accumulator (Phase 3) to complete the
    /// soundness chain.
    ///
    /// `result.ok == true` means everything in the per-proof slice
    /// checks out, but DOES NOT mean the inference proof is acceptable
    /// on its own — the accumulator's `finalize()` must succeed too.
    pub fn verify_with_fold_tree_deferred(
        &self,
        witnesses: &[Vec<Witness>],
        store: &GpuAjtaiStore,
        dag_proof: &DagProof,
        fold_proof: &FoldOpeningProof,
        transcript: &mut Transcript,
    ) -> DeferredResult {
        let ok = self.verify_with_fold_tree(
            witnesses, store, dag_proof, fold_proof, transcript,
        );
        DeferredResult {
            ok,
            claims: fold_proof.deferred_constant_claims.clone(),
        }
    }
}

/// Extend a claim point to `target_arity` by appending **zeros** (not
/// random transcript challenges). The bit-decomposition planes are
/// zero-padded from the witness's native size to `target_arity ≥ 6`,
/// and the MLE of a zero-padded poly at `(native_pt, 0, …, 0)` exactly
/// equals the MLE of the native poly at `native_pt` (since
/// `eq(0…0, x_extra) = Π(1 − x_extra) = 1` when extra coords are 0).
/// Random extension would break this identity.
pub(crate) fn extend_claim_point_to(
    point: &[AlmostGoldilocksExt2],
    target_arity: usize,
) -> Vec<AlmostGoldilocksExt2> {
    assert!(point.len() <= target_arity,
        "claim point length {} > target arity {}", point.len(), target_arity);
    let mut out = point.to_vec();
    while out.len() < target_arity {
        out.push(AlmostGoldilocksExt2::zero());
    }
    out
}

/// Decompose a witness into bit-plane buffers + their MLE evals at the
/// `extended_point`, at the edge's NATIVE commit arity (no broadcast to
/// `max_num_vars`). Each plane has size `2^(arity - 6)` u64s.
/// Evaluate a binary multilinear polynomial at the point whose
/// Lagrange-basis table `eq` is provided. Pure selective add — no
/// field mul, no eq recomputation. Parallelized across `u64` words.
pub(crate) fn eval_binary_with_shared_eq(
    packed: &[u64],
    eq: &[AlmostGoldilocksExt2],
    total: usize,
) -> AlmostGoldilocksExt2 {
    use rayon::prelude::*;
    const CHUNK: usize = 4096; // u64 words per rayon task
    if packed.len() >= CHUNK * 2 {
        packed.par_chunks(CHUNK).enumerate().map(|(ci, words)| {
            let mut acc = AlmostGoldilocksExt2::zero();
            let base0 = ci * CHUNK * 64;
            for (jw, &word) in words.iter().enumerate() {
                if word == 0 { continue; }
                let base = base0 + jw * 64;
                let mut w = word;
                while w != 0 {
                    let k = w.trailing_zeros() as usize;
                    let idx = base + k;
                    if idx < total {
                        acc = ext2_add(acc, eq[idx]);
                    }
                    w &= w - 1;
                }
            }
            acc
        }).reduce(AlmostGoldilocksExt2::zero, ext2_add)
    } else {
        let mut acc = AlmostGoldilocksExt2::zero();
        for (j, &word) in packed.iter().enumerate() {
            if word == 0 { continue; }
            let base = j * 64;
            let mut w = word;
            while w != 0 {
                let k = w.trailing_zeros() as usize;
                let idx = base + k;
                if idx < total {
                    acc = ext2_add(acc, eq[idx]);
                }
                w &= w - 1;
            }
        }
        acc
    }
}

/// Atomic accumulators (microseconds) for timing the two halves of the
/// dense leaf-build: bit decomposition vs per-plane MLE evaluation.
/// Read+reset via `take_leaf_build_breakdown()`.
pub(crate) static LEAF_BUILD_CLONE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_EVAL_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
// Drill-down counters (thread-time µs / counts) to locate the leaf-build
// wall that is NOT dense gpu-eval. Reported + reset under ZK4_TIMING.
pub(crate) static LEAF_BUILD_SETDEV_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_DEFER_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_SPARSE_EVAL_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_SPARSE_CLONE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_N_DEFER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_N_SPARSE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_N_DENSE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_BIT_DECOMPOSE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LEAF_BUILD_PLANE_EVAL_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Pack a sparse witness chunk into its single binary plane (the
/// `2^(arity-6)` u64 bitmask of set positions) — and NOTHING else. This
/// is the cheap part of [`decompose_witness_for_fold_native`]'s sparse
/// branch, WITHOUT the `2^arity` eq-table build + selective-add eval
/// (which the leaf build discards, since it uses the range-proof
/// `claim.eval` directly). Used to regenerate a sparse leaf's plane on
/// demand when the dense bitmask isn't cached, at the same cost as the
/// clone it replaces (a zero-fill + bit sets, vs a memcpy).
pub(crate) fn pack_sparse_plane(witness: &Witness, arity: usize) -> Vec<u64> {
    let sparse = witness.data.as_ref().unwrap()
        .as_any().downcast_ref::<crate::poly::SparseMLPoly>()
        .expect("sparse witness must hold SparseMLPoly");
    let input_nv = sparse.selection.input_num_vars;
    let n_ring = 1usize << (arity - 6);
    let mut packed = vec![0u64; n_ring];
    for &(input_idx, table_idx) in &sparse.selection.selection {
        let p = input_idx + table_idx * (1usize << input_nv);
        packed[p / 64] |= 1u64 << (p % 64);
    }
    packed
}

pub(crate) fn decompose_witness_for_fold_native(
    witness: &Witness,
    key: &AjtaiKey,
    arity: usize,
    extended_point: &[AlmostGoldilocksExt2],
) -> (Vec<Vec<u64>>, Vec<AlmostGoldilocksExt2>) {
    use std::sync::atomic::Ordering;
    assert_eq!(extended_point.len(), arity,
               "extended_point len {} != arity {}", extended_point.len(), arity);
    match witness.poly_type {
        PolyType::Dense => {
            let evals = witness.data.as_ref().expect("dense w/o data").evaluations_ref();
            let t = std::time::Instant::now();
            let planes = crate::commit::bit_decompose::decompose_and_pack_native(evals, key.b, arity);
            LEAF_BUILD_BIT_DECOMPOSE_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            // Compute eq(R, x) ONCE for this edge; all b planes share
            // the same point. f_i(R) = Σ_{x: bit_i(x)=1} eq(R, x) — pure
            // selective add, no field mul (binary witness).
            let t = std::time::Instant::now();
            let eq = crate::poly::evaluate_lagrange_basis_ext2(extended_point);
            let total = 1usize << arity;
            let plane_evals: Vec<_> = planes.iter().map(|p| {
                eval_binary_with_shared_eq(p, &eq, total)
            }).collect();
            LEAF_BUILD_PLANE_EVAL_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
            (planes, plane_evals)
        }
        PolyType::Sparse => {
            // Sparse witness committed at native arity — build the same
            // position list (no broadcast).
            let sparse = witness.data.as_ref().unwrap()
                .as_any().downcast_ref::<crate::poly::SparseMLPoly>()
                .expect("sparse witness must hold SparseMLPoly");
            let positions: Vec<u64> = sparse.selection.selection.iter()
                .map(|&(input_idx, table_idx)| {
                    (input_idx + table_idx * (1usize << sparse.selection.input_num_vars)) as u64
                })
                .collect();
            // Pack into a single binary plane sized 2^(arity-6) u64.
            let n_ring = 1usize << (arity - 6);
            let mut packed = vec![0u64; n_ring];
            for &p in &positions {
                let j = (p / 64) as usize;
                let k = (p % 64) as usize;
                packed[j] |= 1u64 << k;
            }
            // Selective add over a single eq table — same protocol as
            // dense, just one plane per chunk.
            let eq = crate::poly::evaluate_lagrange_basis_ext2(extended_point);
            let total = 1usize << arity;
            let plane_eval = eval_binary_with_shared_eq(&packed, &eq, total);
            (vec![packed], vec![plane_eval])
        }
    }
}

/// Reconstruct the original value from per-plane evals.
/// At base=2 (binary): `v = Σ_{i<b-1} 2^i · y_i − 2^{b-1} · y_{b-1}`.
/// At base>2: `v = Σ_j β^j · y_{d_j}` (positive weights — the signed top-bit
/// weight is already absorbed into the homomorphically-derived digit-plane
/// commitment c_{d_top}, so the verifier just sums with β^j powers).
/// Sparse (single plane): `v = y_0` regardless of base.
pub(crate) fn reconstruct_signed_two_complement(
    plane_evals: &[AlmostGoldilocksExt2],
    is_sparse: bool,
    b: usize,
    base: usize,
) -> AlmostGoldilocksExt2 {
    if is_sparse {
        assert_eq!(plane_evals.len(), 1, "sparse expects 1 plane");
        return plane_evals[0];
    }
    if base == 2 {
        assert_eq!(plane_evals.len(), b, "dense plane count {} != b {}", plane_evals.len(), b);
        let mut acc = AlmostGoldilocksExt2::zero();
        for i in 0..(b - 1) {
            let pow = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1u64 << i));
            acc = ext2_add(acc, ext2_mul(pow, plane_evals[i]));
        }
        let pow_top = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1u64 << (b - 1)));
        acc = ext2_sub(acc, ext2_mul(pow_top, plane_evals[b - 1]));
        acc
    } else {
        // base-β: Σ_j β^j · y_{d_j}, where the signed-top weight is baked
        // into y_{d_top} via the derived digit-plane commitment.
        let b_beta = crate::commit::bit_decompose::digit_planes_for(b, base);
        assert_eq!(plane_evals.len(), b_beta,
            "dense plane count {} != b_β {} (b={}, base={})", plane_evals.len(), b_beta, b, base);
        let mut acc = AlmostGoldilocksExt2::zero();
        let mut beta_j = AlmostGoldilocksExt2::one();
        let beta_field = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(base as u64));
        for j in 0..b_beta {
            acc = ext2_add(acc, ext2_mul(beta_j, plane_evals[j]));
            beta_j = ext2_mul(beta_j, beta_field);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::ajtai::Seed;
    use crate::commit::AjtaiKey;
    use crate::dag::{DagBuilder, DataType};
    use crate::util::arith::int_to_f;

    fn demo_seed() -> Seed {
        Seed([
            0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
            0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
        ])
    }

    fn make_const(shape: Vec<usize>, raw: Vec<i128>) -> Witness {
        let evals = raw.iter().map(|&v| int_to_f(v)).collect();
        Witness::new(shape, evals, DataType::Int, 0, Role::Constant)
    }

    fn make_input(shape: Vec<usize>, raw: Vec<i128>) -> Witness {
        let evals = raw.iter().map(|&v| int_to_f(v)).collect();
        Witness::new(shape, evals, DataType::Int, 0, Role::Input)
    }

    /// Stress test: heterogeneous range tables in one DAG. NonNegative
    /// uses `TABLE_SIZE_LOG = 20` while ExpHelper (inside `g.exp`) uses
    /// `num_bits = 16` — so `prove_range` sees two range nodes with
    /// different `table_n`s. Validates the zero-padding code path in
    /// the combined sumcheck.
    #[test]
    fn end_to_end_heterogeneous_range_tables() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        // y = x + w so the NonNegative input has an upstream Add that
        // pushes a claim during the backward pass.
        let y = g.add(x, w)[0];
        // 1) NonNegative range check on y — table_log = TABLE_SIZE_LOG (= 20).
        g.add_nonneg_node(y);
        // 2) Add a small NonNegative with a SMALLER table to force
        // heterogeneity. Builder doesn't expose `add_nonneg_node` with
        // custom log, so we inline the same pattern with table_log = 4.
        let nid = g.nodes.len();
        let nn = crate::basicblock::BasicBlockType::NonNegative(
            crate::basicblock::range::NonNegative::new(4),
        );
        let _ = g.add_gkr_node(vec![y], nn);
        g.init_values.push(Some(crate::dag::Witness::new_wo_data(
            vec![1], crate::dag::DataType::Float, 0, crate::dag::Role::Auxiliary,
        )));
        g.range.push(nid);
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![1, 2, 3, 4]))]);

        // The largest aux is the NonNegative @ TABLE_SIZE_LOG = 20:
        // aux_n = input_n (= 2) + 20 = 22. max_num_vars must cover.
        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 22, /*b=*/ 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"het-range");
        let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);

        let mut t_v = Transcript::new(b"het-range");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(ok, "heterogeneous range table e2e should verify");
    }

    /// A batched conv proves as ONE node over a wider tensor, not as `B`
    /// replicated subgraphs. This is the end-to-end check that the batch
    /// index survives commitment, the backward pass and the fold-tree
    /// opening -- the block-level tests cover the sumcheck itself.
    #[test]
    fn end_to_end_batched_conv2d() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        for batch in [1usize, 2, 4] {
            let mut g = DagBuilder::new();
            // [B, C_in, H, W]; B == 1 keeps the 3-D shape so the unbatched
            // path is exercised by the same test.
            let x_shape = if batch > 1 {
                vec![batch, 1, 4, 4]
            } else {
                vec![1, 4, 4]
            };
            let x = g.input(x_shape.clone(), DataType::Int);
            let w = g.param(make_const(vec![1, 1, 2, 2], vec![1, 1, 1, 1]));
            let _y = g.conv2d(x, w, (2, 2))[0];
            let (dag, mut witnesses) = g.compile();

            let n = batch * 16;
            let data: Vec<i128> = (0..n as i128).map(|v| (v % 7) + 1).collect();
            dag.run(&mut witnesses, &[(0, make_input(x_shape, data))]);

            let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 20, /*b=*/ 21);
            let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
            dag.commit(&witnesses, &mut store);

            let mut t_p = Transcript::new(b"batched-conv");
            let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);
            let mut t_v = Transcript::new(b"batched-conv");
            assert!(
                dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v),
                "batch={} conv2d must verify end to end", batch
            );
        }
    }

    /// Which operators survive a leading batch dimension? Each case builds the
    /// op twice -- once on `B` separate images, once on one `[B, ...]` tensor --
    /// and requires the batched forward to equal the per-image forward,
    /// concatenated. An op that is genuinely elementwise passes for free; an op
    /// that reasons about spatial extent does not, and this says which is which
    /// instead of leaving it to inspection.
    ///
    /// This is an assertion, not a diagnostic: every operator listed here is
    /// one an end-to-end CNN needs, so a regression that silently drops the
    /// batch axis must fail the suite rather than print a note. The full table
    /// is also written to /tmp for inspection, since this binary reads argv[1]
    /// as its config path and so cannot be run with --nocapture.
    #[test]
    fn batched_ops_match_per_image_forward() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        // A batched CNN tensor is FOLDED: [b_pad*c_pad, H, W]. So the batched
        // graph is the per-image graph with more channels, and the per-image
        // comparison slices the leading axis.
        let b = 2usize;
        let c = 2usize;
        // (name, build, batch-safe?). `false` records a KNOWN gap: the entry
        // stays in the table so the gap is visible and so the day it starts
        // working the expectation fails rather than silently drifting.
        let ops: Vec<(&str, fn(&mut DagBuilder, EdgeId) -> EdgeId, bool)> = vec![
            ("relu", |g, x| g.relu(x), true),
            ("scale", |g, x| g.scale(x, 0, 0)[0], true),
            ("maxpool2d", |g, x| g.maxpool2d(x, 2, 2), true),
            ("pad", |g, x| g.pad(x, 1, 1), true),
            ("maxpool_general", |g, x| g.maxpool_general(x, 2, 2, 2, 2), true),
            ("reduce_mean", |g, x| g.reduce_mean(x, &[1, 2]), true),
            ("conv2d", |g, x| {
                let w = g.param(make_const(vec![2, 2, 2, 2], vec![1; 16]));
                g.conv2d(x, w, (2, 2))[0]
            }, true),
            // KNOWN GAP. Concat joins along the leading axis, which under
            // folding is the SAME axis the batch lives in, so it produces
            // [b_pad*cA + b_pad*cB] -- all of A's images then all of B's --
            // where the result needs the two interleaved per image. Blocks
            // YOLO, 3D-UNet and PointPillars, which all use channel concat.
            ("concat", |g, x| g.concat(x, x), false),
            ("conv2d_strided", |g, x| {
                let w = g.param(make_const(vec![2, 2, 2, 2], vec![1; 16]));
                g.conv2d_strided(x, w, (2, 2), (2, 2))[0]
            }, true),
        ];
        let mut report: Vec<String> = Vec::new();
        let mut want_ok: Vec<(&str, bool)> = Vec::new();
        for (name, build, supported) in ops {
            want_ok.push((name, supported));
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // batched
                let mut g = DagBuilder::new();
                let x = g.input(vec![b * c, 4, 4], DataType::Int);
                // Read the edge the op RETURNED. Composite ops (maxpool, relu)
                // append dominance/range aux nodes after it, so the last
                // witness is not the output.
                let out_edge = build(&mut g, x);
                let (dag, mut w) = g.compile();
                let data: Vec<i128> = (0..(b * 2 * 4 * 4) as i128)
                    .map(|v| (v % 9) + 1).collect();
                dag.run(&mut w, &[(0, make_input(vec![b * c, 4, 4], data.clone()))]);
                let out = w[out_edge][0].clone();

                // per image
                let mut per = Vec::new();
                for i in 0..b {
                    let mut g1 = DagBuilder::new();
                    let x1 = g1.input(vec![c, 4, 4], DataType::Int);
                    let e1 = build(&mut g1, x1);
                    let (d1, mut w1) = g1.compile();
                    let slice = data[i * 32..(i + 1) * 32].to_vec();
                    d1.run(&mut w1, &[(0, make_input(vec![c, 4, 4], slice))]);
                    per.push(w1[e1][0].clone());
                }
                (out, per)
            }));
            match res {
                Err(e) => {
                    let msg = e.downcast_ref::<String>().cloned()
                        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "?".into());
                    report.push(format!("{:<16} PANICKED  {}", name,
                        msg.lines().next().unwrap_or("")));
                }
                Ok((out, per)) => {
                    let got = out.data.as_ref().unwrap().evaluations();
                    let stride = got.len() / b;
                    let mut ok = true;
                    for (i, p) in per.iter().enumerate() {
                        let want = p.data.as_ref().unwrap().evaluations();
                        let n = want.len().min(stride);
                        if got[i * stride..i * stride + n] != want[..n] { ok = false; }
                    }
                    report.push(format!("{:<16} {}", name,
                        if ok { "ok" } else { "WRONG VALUES" }));
                }
            }
        }
        let _ = std::fs::write("/tmp/zk4_batch_probe.txt", report.join("\n"));
        for (line, (name, supported)) in report.iter().zip(want_ok) {
            let ok = line.trim_end().ends_with(" ok");
            assert_eq!(ok, supported,
                "{}: batch support is {}, expected {} -- {}",
                name, ok, supported, line.trim_end());
        }
    }

    /// Pins einsum's shape convention, which the FC head depends on.
    ///
    /// For einsum, the FIRST shape dimension occupies the LOW bits: in a
    /// [B, feat] tensor the images are INTERLEAVED (image b at b + i*B), not
    /// contiguous. Conv is the other way round -- [C, H, W] puts W lowest --
    /// so a batched CNN cannot simply hand its folded output to the FC head
    /// and expect the batch axis to line up. Getting this backwards silently
    /// computes a different matmul rather than failing, which is why it is
    /// nailed down here.
    #[test]
    fn einsum_batch_axis_is_the_low_bits() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let (b, feat, out) = (2usize, 4usize, 3usize);
        let wdata: Vec<i128> = (0..(feat * out) as i128).map(|v| (v % 5) + 1).collect();
        let xdata: Vec<i128> = (0..(b * feat) as i128).map(|v| (v % 7) + 1).collect();

        let mut g = DagBuilder::new();
        let x = g.input(vec![b, feat], DataType::Int);
        let w = g.param(make_const(vec![feat, out], wdata.clone()));
        let y = g.einsum("ij,jk->ik".to_string(), vec![x, w], false)[0];
        let (dag, mut ws) = g.compile();
        dag.run(&mut ws, &[(0, make_input(vec![b, feat], xdata.clone()))]);
        let got = ws[y][0].data.as_ref().unwrap().evaluations();

        for i in 0..b {
            // Image i's features are strided by b, not contiguous.
            let img: Vec<i128> = (0..feat).map(|f| xdata[f * b + i]).collect();
            let mut g1 = DagBuilder::new();
            let x1 = g1.input(vec![feat], DataType::Int);
            let w1 = g1.param(make_const(vec![feat, out], wdata.clone()));
            let y1 = g1.einsum("i,ij->j".to_string(), vec![x1, w1], false)[0];
            let (d1, mut w1s) = g1.compile();
            d1.run(&mut w1s, &[(0, make_input(vec![feat], img))]);
            let want = w1s[y1][0].data.as_ref().unwrap().evaluations();
            for j in 0..out {
                assert_eq!(got[j * b + i], want[j],
                    "image {} output {} : batched einsum disagrees with the \
                     per-image matmul under the interleaved convention", i, j);
            }
        }

        let key = AjtaiKey::new(demo_seed(), 20, 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&ws, &mut store);
        let mut tp = Transcript::new(b"batched-fc");
        let (dp, fp) = dag.prove_with_fold_tree(&ws, &store, &mut tp);
        let mut tv = Transcript::new(b"batched-fc");
        assert!(dag.verify_with_fold_tree(&ws, &store, &dp, &fp, &mut tv),
                "batched FC head must verify");
    }

    /// The conv -> FC boundary for a batched CNN.
    ///
    /// The folded conv output has b in the HIGH bits, and einsum puts the FIRST
    /// shape dimension in the LOW bits, so the FC head carries the batch as the
    /// LAST shape entry: [features, B]. The reshape is then a pure relabelling
    /// (w,h,c fold into f, b stays highest) and the matmul becomes
    /// "ib,ij->jb", which keeps the output in the same [features, B] form for
    /// the next layer. The bias is [O, 1] so it broadcasts across images --
    /// [O] would align against B instead, since broadcasting matches trailing
    /// dimensions.
    #[test]
    fn batched_conv_to_fc_head() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let (b, feat, out) = (2usize, 4usize, 3usize);
        let wdata: Vec<i128> = (0..(feat * out) as i128).map(|v| (v % 5) + 1).collect();
        let bias: Vec<i128> = (0..out as i128).map(|v| v + 1).collect();
        // [F, B]: feature index low, image index high.
        let xdata: Vec<i128> = (0..(feat * b) as i128).map(|v| (v % 7) + 1).collect();

        let mut g = DagBuilder::new();
        let x = g.input(vec![feat, b], DataType::Int);
        let w = g.param(make_const(vec![feat, out], wdata.clone()));
        let bb = g.param(make_const(vec![out, 1], bias.clone()));
        let y0 = g.einsum("ib,ij->jb".to_string(), vec![x, w], false)[0];
        let y = g.add(y0, bb)[0];
        let (dag, mut ws) = g.compile();
        dag.run(&mut ws, &[(0, make_input(vec![feat, b], xdata.clone()))]);
        let got = ws[y][0].data.as_ref().unwrap().evaluations();

        for i in 0..b {
            // [F, B] puts F in the low bits and B in the high bits, so image
            // i is CONTIGUOUS at offset i * feat_pad.
            let fp = feat.next_power_of_two();
            let img: Vec<i128> = (0..feat).map(|f| xdata[i * fp + f]).collect();
            let mut g1 = DagBuilder::new();
            let x1 = g1.input(vec![feat], DataType::Int);
            let w1 = g1.param(make_const(vec![feat, out], wdata.clone()));
            let b1 = g1.param(make_const(vec![out], bias.clone()));
            let t = g1.einsum("i,ij->j".to_string(), vec![x1, w1], false)[0];
            let y1 = g1.add(t, b1)[0];
            let (d1, mut w1s) = g1.compile();
            d1.run(&mut w1s, &[(0, make_input(vec![feat], img))]);
            let want = w1s[y1][0].data.as_ref().unwrap().evaluations();
            // Output is [O, B]: O low, B high, so image i is contiguous again.
            let op = out.next_power_of_two();
            for j in 0..out {
                assert_eq!(got[i * op + j], want[j],
                    "image {} logit {}: batched FC head disagrees with per-image",
                    i, j);
            }
        }

        let key = AjtaiKey::new(demo_seed(), 20, 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&ws, &mut store);
        let mut tp = Transcript::new(b"batched-fc-head");
        let (dp, fp) = dag.prove_with_fold_tree(&ws, &store, &mut tp);
        let mut tv = Transcript::new(b"batched-fc-head");
        assert!(dag.verify_with_fold_tree(&ws, &store, &dp, &fp, &mut tv),
                "batched conv->FC head must verify");
    }

    /// Stress test: exp pipeline (ExpHelper + TwoPow + range checks all
    /// stitched together by `g.exp`). The hot bug here would be the
    /// two_pow indexed-sum padding fix; the test fails if I got it
    /// wrong.
    #[test]
    fn end_to_end_exp_pipeline_two_pow() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        // exp needs Float-typed input. Tiny shape [4] → 2 vars + 16 (ExpHelper)
        // gives aux_n = 18, well within max_num_vars budget.
        let x = g.input(vec![4], DataType::Float);
        let _y = g.exp(x);
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![100, 200, 300, 400]))]);

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 22, /*b=*/ 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"exp-pipe");
        let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);

        let mut t_v = Transcript::new(b"exp-pipe");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(ok, "exp pipeline e2e should verify");
    }

    /// Stress test: many committed dense edges + multiple NonNegative
    /// range checks (mimics a deep chained-Add pipeline). Exercises
    /// the opening reducer + fold tree at higher leaf counts.
    #[test]
    fn end_to_end_many_committed_edges() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let mut cur = x;
        // 6 layers of (add a 1-vector, range-check intermediate). Final
        // values must stay in [0, 64) so the NonNegative table_log = 6
        // is non-trivially satisfied. Adding 6 ones to input [0..7] →
        // final = [6..13]; well within.
        for _ in 0..6 {
            let w = g.param(make_const(vec![8], vec![1; 8]));
            cur = g.add(cur, w)[0];
            let nid = g.nodes.len();
            let nn = crate::basicblock::BasicBlockType::NonNegative(
                crate::basicblock::range::NonNegative::new(6),
            );
            let _ = g.add_gkr_node(vec![cur], nn);
            g.init_values.push(Some(crate::dag::Witness::new_wo_data(
                vec![1], crate::dag::DataType::Float, 0, crate::dag::Role::Auxiliary,
            )));
            g.range.push(nid);
        }
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![8], vec![0, 1, 2, 3, 4, 5, 6, 7]))]);

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"many-edges");
        let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);
        // At b=21 and 6 layers ≈ 14 edges, we should have ~14·21 + a few
        // aux planes = ~300 leaves — plenty to exercise a non-trivial
        // fold tree.
        assert!(fp.edge_plane_evals.len() >= 6,
                "expected ≥ 6 committed edges, got {}", fp.edge_plane_evals.len());

        let mut t_v = Transcript::new(b"many-edges");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(ok, "many-edges DAG should verify");
    }

    /// Sparse-edge end-to-end: DAG with a NonNegative range check
    /// produces a sparse aux. The aux is committed via the sparse
    /// path (single plane); the fold tree integration must route it
    /// through `decompose_witness_for_fold`'s sparse branch.
    #[test]
    fn end_to_end_with_sparse_range_aux() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let w = g.param(make_const(vec![8], vec![1, 1, 1, 1, 1, 1, 1, 1]));
        let y = g.add(x, w)[0];
        // Explicit NonNegative with small table_log (= 4 → covers [0, 16)).
        let nid = g.nodes.len();
        let nn = crate::basicblock::BasicBlockType::NonNegative(
            crate::basicblock::range::NonNegative::new(4),
        );
        let _ = g.add_gkr_node(vec![y], nn);
        g.init_values.push(Some(crate::dag::Witness::new_wo_data(
            vec![1], DataType::Float, 0, crate::dag::Role::Auxiliary,
        )));
        g.range.push(nid);
        let (dag, mut witnesses) = g.compile();
        // Input values in [0, 16) so the range check + sparse aux are valid.
        dag.run(&mut witnesses, &[(0, make_input(vec![8], vec![0, 1, 2, 5, 9, 10, 12, 14]))]);

        // max_num_vars covers the aux's native arity = input_n + table_n.
        // input_n = log2(next_pow(8)) = 3; table_n = 4 → aux_n = 7.
        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 7, /*b=*/ 8);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"sparse-e2e");
        let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);
        // We expect at least one sparse leaf among the per-edge plane evals.
        let any_sparse = fold_proof.edge_plane_evals.iter().any(|e| e.is_sparse);
        assert!(any_sparse, "expected at least one sparse-edge leaf in fold proof");

        let mut t_v = Transcript::new(b"sparse-e2e");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_v);
        assert!(ok, "sparse end-to-end verify should accept");
    }

    /// CV foundation: a minimal fixed-point conv → ScaleDown → ReLU graph
    /// through the fold tree. Exercises the conv-sf-propagation + per-conv
    /// rescale + range-aux path that full CNNs need (isolated from depth).
    #[test]
    fn cnn_conv_scale_relu_fixed_point() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let sf = 4usize;
        let mksf = |shape: Vec<usize>, raw: Vec<i128>, role| {
            Witness::new(shape, raw.iter().map(|&v| int_to_f(v)).collect(), DataType::Int, sf, role)
        };
        let mut g = DagBuilder::new();
        let x = g.input(vec![4, 4, 4], DataType::Int); // [c_in=4, 4x4]
        // 1x1 conv, c_out=4, c_in=4 — non-negative small weights {0,1,2}.
        let wdata: Vec<i128> = (0..16).map(|i| (i % 3) as i128).collect();
        let w = g.param(mksf(vec![4, 4, 1, 1], wdata, Role::Constant));
        let conv = g.conv2d(x, w, (1, 1))[0]; // sf = 2*sf with sf-propagation
        let scaled = g.scale(conv, 2 * sf, sf)[0]; // ScaleDown → sf
        let _relu = g.relu(scaled);
        let (dag, mut witnesses) = g.compile();

        // input raw in [4,12) so conv outputs are clearly nonzero (avoids the
        // all-zero degenerate-claim case) and bounded.
        let xdata: Vec<i128> = (0..64).map(|i| (4 + (i % 8)) as i128).collect();
        dag.run(&mut witnesses, &[(x, mksf(vec![4, 4, 4], xdata, Role::Input))]);

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 22, /*b=*/ 12);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"cnn-e2e");
        let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);
        let mut t_v = Transcript::new(b"cnn-e2e");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(ok, "fixed-point conv→scale→relu should verify");
    }

    /// End-to-end: build a tiny `y = x + w` DAG, commit the constant
    /// `w`, run prove_with_fold_tree, verify_with_fold_tree.
    #[test]
    fn end_to_end_prove_verify_with_fold_tree() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let w = g.param(make_const(vec![8], (1..=8i128).collect()));
        let _y = g.add(x, w)[0];
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![8], vec![10, 20, 30, 40, 50, 60, 70, 80]))]);

        // Commit the constants — and also commit y as an "output" via
        // commit() so it has a per-plane commitment for the fold tree.
        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 6, /*b=*/ 8);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"e2e");
        let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_p);
        assert!(!fold_proof.edge_plane_evals.is_empty(),
                "expected at least one committed edge");

        let mut t_v = Transcript::new(b"e2e");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_v);
        assert!(ok, "end-to-end verify should accept");
    }

    /// Phase 1+2: deferred-constants mode round trip. The prover emits
    /// DeferredClaims for every `Role::Constant` edge instead of building
    /// fold-tree leaves; the verifier accepts the proof modulo the
    /// deferred claims and returns them via DeferredResult.
    #[test]
    fn end_to_end_defer_constants() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        // Two distinct weight params so we get ≥ 2 deferred claims.
        let w0 = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let w1 = g.param(make_const(vec![4], vec![2, 2, 2, 2]));
        let s0 = g.add(x, w0)[0];
        let _s1 = g.add(s0, w1)[0];
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![0, 0, 0, 0]))]);

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        // Defer-mode prove.
        let mut t_p = Transcript::new(b"defer-e2e");
        let (dp, fp) = dag.prove_with_fold_tree_modes(
            &witnesses, &store, &mut t_p, /*defer_const=*/ true,
        );
        assert!(!fp.deferred_constant_claims.is_empty(),
                "expected ≥ 1 deferred constant claim in defer mode");
        // The deferred edges must NOT appear in `edge_plane_evals`.
        for dc in &fp.deferred_constant_claims {
            assert!(
                !fp.edge_plane_evals.iter().any(|e| e.edge_id == dc.edge_id),
                "deferred edge {} should not also be in edge_plane_evals",
                dc.edge_id,
            );
        }

        // Defer-mode verify via the bool API.
        let mut t_v = Transcript::new(b"defer-e2e");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(ok, "defer-mode verify should accept (modulo deferred binds)");

        // Same proof verified through the deferred-result API.
        let mut t_v2 = Transcript::new(b"defer-e2e");
        let r = dag.verify_with_fold_tree_deferred(
            &witnesses, &store, &dp, &fp, &mut t_v2,
        );
        assert!(r.ok, "deferred-mode verify_with_fold_tree_deferred should accept");
        assert_eq!(r.claims.len(), fp.deferred_constant_claims.len());
    }

    /// Soundness check: tampering with a deferred claim's `eval` must
    /// make verify fail. This is the property the streaming accumulator
    /// depends on — the per-proof verifier MUST notice mismatches
    /// between the declared deferred (point, eval) and the DAG's
    /// sumcheck terminal claim.
    #[test]
    fn defer_mode_rejects_mutated_eval() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let _y = g.add(x, w)[0];
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![0, 0, 0, 0]))]);

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"defer-tamper");
        let (dp, mut fp) = dag.prove_with_fold_tree_modes(
            &witnesses, &store, &mut t_p, /*defer_const=*/ true,
        );
        assert!(!fp.deferred_constant_claims.is_empty());

        // Mutate the first deferred claim's eval (add 1).
        let one = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1));
        fp.deferred_constant_claims[0].eval =
            ext2_add(fp.deferred_constant_claims[0].eval, one);

        let mut t_v = Transcript::new(b"defer-tamper");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(!ok, "mutating a deferred eval must make verify fail");
    }

    /// Soundness check: tampering with a deferred claim's point must
    /// also fail.
    #[test]
    fn defer_mode_rejects_mutated_point() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let _y = g.add(x, w)[0];
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![0, 0, 0, 0]))]);

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        dag.commit(&witnesses, &mut store);

        let mut t_p = Transcript::new(b"defer-pt");
        let (dp, mut fp) = dag.prove_with_fold_tree_modes(
            &witnesses, &store, &mut t_p, /*defer_const=*/ true,
        );
        assert!(!fp.deferred_constant_claims.is_empty());

        // Mutate the first coordinate of the first deferred claim's point.
        let one = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1));
        fp.deferred_constant_claims[0].point[0] =
            ext2_add(fp.deferred_constant_claims[0].point[0], one);

        let mut t_v = Transcript::new(b"defer-pt");
        let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut t_v);
        assert!(!ok, "mutating a deferred point must make verify fail");
    }
}
