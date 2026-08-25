//! `Dag::prove` / `Dag::verify` — the sumcheck-side backward pass.
//!
//! Implements plan §8.2 step 11. The protocol structure mirrors zk-torch-3
//! but is otherwise field-clean (Ext2 throughout) and decoupled from any
//! PCS opening — that lives in the fold-tree step.
//!
//! Workflow on the prover side:
//! 1. Sample a random Ext2 point for every output port; seed `claims[e]`.
//! 2. Reverse topo walk via `BTreeSet`: pop the largest-id ready node, run
//!    its `prove` (after a reducer pass if it has multiple output claims),
//!    accumulate produced claims onto its input edges, and feed those
//!    input-edge producers back into the work set.
//! 3. Run `prove_two_pow` then `prove_range` (the order matters: range
//!    needs the input claim on the ExpHelper output edge to already exist).
//! 4. Snapshot accumulated claims into `EdgeProof::claims` — this is what
//!    the opening reducer and the fold-tree opening will consume.
//!
//! `Dag::verify` replays the same walk: re-derives the output claims from
//! the transcript, re-runs each `kind.verify`, and finally verifies the
//! lookup proofs. Opening verification is decoupled (step 6 fold-tree).

use std::collections::BTreeSet;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;

use crate::basicblock::{BasicBlock, BasicBlockType, Reducer};
use crate::dag::{Claim, Dag, DagProof, EdgeId, EdgeProof, NodeId, NodeProof, PolyType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{calc_pow_vec_ext2, ext2_add, ext2_field_eq, ext2_mul, ext2_sub};

// Partition-backward drill-down counters (thread-time µs / counts),
// summed across the parallel partitions. Reported + reset under ZK4_TIMING.
static PB_INIT_EVAL_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PB_NODE_PROVE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PB_REDUCER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PB_N_NODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PB_N_REDUCERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Dag {
    /// Run the backward pass + lookup proofs. The returned [`DagProof`] is
    /// self-contained on the sumcheck side; the fold-tree opening is
    /// produced by a separate phase (step 5).
    pub fn prove(
        &self,
        witnesses: &[Vec<Witness>],
        transcript: &mut Transcript,
    ) -> DagProof {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut node_proofs: Vec<Option<NodeProof>> = vec![None; self.nodes.len()];
        let mut edge_proofs: Vec<EdgeProof> =
            (0..self.num_edges).map(|_| EdgeProof::new()).collect();

        // ---- 1. Seed output-port claims. ----
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut output_claims: Vec<(usize, Vec<_>, _)> = Vec::new();
        let mut nodes_to_prove: BTreeSet<usize> = BTreeSet::new();
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role != Role::Output { continue; }
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"output_challenge")).collect();
            let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
            claims[e].push(Claim { edge_id: e, sparse_id: 0, point: point.clone(), eval });
            output_claims.push((e, point, eval));
            if let Some(p) = self.producers[e] {
                nodes_to_prove.insert(p);
            }
        }

        // Also handle `self_claim_edges` — output edges that need a claim
        // even though they may be intermediate (Conv outputs etc.).
        for &e in &self.self_claim_edges {
            if !self.output_ports.contains(&e) {
                let w = &witnesses[e][0];
                let n = w.data.as_ref().unwrap().n();
                let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"self_claim_challenge")).collect();
                let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
                claims[e].push(Claim { edge_id: e, sparse_id: 0, point: point.clone(), eval });
                output_claims.push((e, point, eval));
                if let Some(p) = self.producers[e] {
                    nodes_to_prove.insert(p);
                }
            }
        }

        // ---- 2. Backward pass. ----
        let bw_timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
        let t_nodes = std::time::Instant::now();
        while let Some(&node_id) = nodes_to_prove.iter().next_back() {
            nodes_to_prove.remove(&node_id);

            // Defer if any downstream consumer of this node's outputs is
            // still in the work set (we need every claim to be settled).
            let mut can_prove = true;
            for &edge in &self.nodes[node_id].outputs {
                if self.consumers[edge].iter().any(|c| nodes_to_prove.contains(c)) {
                    can_prove = false;
                    break;
                }
            }
            if !can_prove {
                nodes_to_prove.insert(node_id);
                continue;
            }

            let node = &self.nodes[node_id];
            let mut edge_ids: Vec<usize> = Vec::new();
            edge_ids.extend_from_slice(&node.inputs);
            edge_ids.extend_from_slice(&node.outputs);
            let local_witnesses: Vec<&Witness> = edge_ids.iter().map(|&e| &witnesses[e][0]).collect();
            let mut local_claims: Vec<&Claim> = Vec::new();
            for &e in &node.outputs {
                local_claims.extend(claims[e].iter());
            }
            // Reducer if multiple polynomial-evaluation claims share the
            // node.outputs[0] witness. Empty-point claims (e.g. conv's
            // `s_alpha_claim` scalar side-channel) are NOT multilinear
            // evaluation claims and must not be reduced: their eq table
            // has length 1 while the witness has 2^n entries. Filter on
            // arity match before counting for the reducer.
            let mut reducer_proofs_local: Option<Vec<SumcheckProof>> = None;
            #[allow(unused_assignments)]
            let mut reduced_storage: Vec<Claim> = Vec::new();
            #[allow(unused_assignments)]
            let mut side_storage: Vec<Claim> = Vec::new();
            let reducer_witness_n = witnesses[node.outputs[0]][0].data.as_ref().unwrap().n();
            let reducible: Vec<&Claim> = local_claims
                .iter()
                .copied()
                .filter(|c| c.point.len() == reducer_witness_n)
                .collect();
            if reducible.len() > 1 {
                // Snapshot side-channel claims by clone *before* we
                // mutably borrow `claims` to push the reduced claim.
                let side_claims: Vec<Claim> = local_claims
                    .iter()
                    .filter(|c| c.point.len() != reducer_witness_n)
                    .map(|c| (*c).clone())
                    .collect();
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                // The reducer combines multiple claims on `node.outputs[0]`
                // (where its witness lives). For multi-output nodes
                // (ScaleDown / ScaleUp / ExpHelper — output + aux),
                // `edge_ids.last()` is the AUX edge, not the main output;
                // using it here would tag the reduced claim with the wrong
                // edge_id and break the verifier's `produced_claims.last()
                // == outputs[0]` check.
                let reducer_edge_ids = vec![node.outputs[0]];
                let (proofs, rc) =
                    reducer.prove(&reducer_witness, &reducer_edge_ids, &reducible, transcript);
                if !rc.is_empty() {
                    drop(local_claims);
                    claims[node.outputs[0]].push(rc[0].clone());
                    reducer_proofs_local = Some(proofs);
                    side_storage = side_claims;
                    reduced_storage = rc;
                    // Order matters: basicblocks read `out_claims[0]` as
                    // the canonical multilinear claim. Reduced claim first,
                    // side-channel claims (conv's `s_alpha_claim` etc.) after.
                    let mut new_local: Vec<&Claim> = reduced_storage.iter().collect();
                    new_local.extend(side_storage.iter());
                    local_claims = new_local;
                }
            }

            // Prove the node.
            let (proofs, new_claims) =
                node.kind.prove(&local_witnesses, &edge_ids, &local_claims, transcript);

            // Snapshot node_claims = produced ++ consumed.
            let mut node_claims: Vec<Claim> = new_claims.clone();
            for c in &local_claims {
                node_claims.push((*c).clone());
            }

            // Push produced claims onto their input edges, and pull
            // upstream producers into the work set. Skip producers that
            // are the current node — conv's y_self_claim and s_alpha_claim
            // both land on conv's own output edge, and reinserting self
            // here loops forever (the reducer correctly skips empty-point
            // side-channel claims so it can't pop them off as before).
            for c in &new_claims {
                claims[c.edge_id].push(c.clone());
                if let Some(p) = self.producers[c.edge_id] {
                    if p != node_id {
                        nodes_to_prove.insert(p);
                    }
                }
            }

            // Stash the reducer proofs (if any) as the first sumcheck
            // proof slot — the verifier will pull it out the same way.
            // We use a tagged convention: prepend a special marker is
            // unnecessary because reducer_proofs_local carries its own
            // semantic (it acts on the *output* edge, then the node's
            // own proofs act on the reduced claim). We store both as a
            // contiguous Vec<SumcheckProof>: [reducer_proofs..., node_proofs...].
            //
            // We track the split with a separate field in `NodeProof` if
            // needed; for now the layout below mirrors zk-torch-3.
            let mut combined_proofs: Vec<SumcheckProof> = Vec::new();
            if let Some(red) = reducer_proofs_local.as_ref() {
                combined_proofs.extend_from_slice(red);
            }
            combined_proofs.extend_from_slice(&proofs);
            node_proofs[node_id] = Some(NodeProof {
                sumcheck_proofs: combined_proofs,
                produced_claims: node_claims,
            });
        }

        let dt_nodes = t_nodes.elapsed();

        // ---- 3. Lookup proofs. ----
        // Two-pow first so the range proof sees its aux claim being
        // attached (matches zk-torch-2 ordering).
        let t_two_pow = std::time::Instant::now();
        let two_pow_proof = self.prove_two_pow(witnesses, &mut claims, transcript);
        let dt_two_pow = t_two_pow.elapsed();
        let t_range = std::time::Instant::now();
        let range_proof = self.prove_range(witnesses, &mut claims, transcript);
        let dt_range = t_range.elapsed();

        // ---- 4. Snapshot edge claims (pre-reducer). ----
        for e in 0..self.num_edges {
            edge_proofs[e].claims = claims[e].clone();
        }

        // ---- 5. Opening reducer (parallel, fork-per-edge). ----
        let t_red = std::time::Instant::now();
        self.prove_opening_reducers(witnesses, &mut edge_proofs, transcript);
        let dt_red = t_red.elapsed();
        if bw_timing {
            eprintln!("[prove]   backward phases: node-sumcheck {:?}, two_pow {:?}, range {:?}, opening_reducer {:?}",
                dt_nodes, dt_two_pow, dt_range, dt_red);
        }

        DagProof {
            node_proofs,
            edge_proofs,
            range_proof,
            two_pow_proof,
            output_claims,
            boundary_claims: Vec::new(),
        }
    }

    /// Multi-GPU partitioned prove. Mirrors `prove` but:
    /// 1. Samples opening points for each boundary edge from the main
    ///    transcript (`self.boundary_edges`).
    /// 2. Routes initial claims (output_ports + self_claim_edges +
    ///    boundary edges) to the partition that PRODUCES them.
    /// 3. Runs each partition's backward pass in parallel via rayon,
    ///    each pinned to its assigned GPU via `set_device`. Each
    ///    partition uses a FORKED transcript so per-partition challenge
    ///    streams don't collide.
    /// 4. Merges per-partition claims, then continues with lookup
    ///    proofs + opening-reducer on the main transcript.
    ///
    /// `partitions` must come from `partition_dag(&dag, &dag.boundary_edges)`
    /// after `dag.set_partition_boundaries(N)` was called. With a single
    /// partition (default), this is equivalent to `prove`.
    pub fn prove_partitioned(
        &self,
        witnesses: &[Vec<Witness>],
        partitions: &[crate::dag::partition::PartitionDesc],
        transcript: &mut Transcript,
    ) -> DagProof {
        let mut edge_proofs: Vec<EdgeProof> =
            (0..self.num_edges).map(|_| EdgeProof::new()).collect();
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut initial_claims: Vec<(EdgeId, Claim)> = Vec::new();
        let mut output_claims_proof: Vec<(EdgeId, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)> =
            Vec::new();

        // ---- 1. Output-port claims. ----
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role != Role::Output { continue; }
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"output_challenge")).collect();
            let t_ie = std::time::Instant::now();
            let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
            PB_INIT_EVAL_US.fetch_add(t_ie.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            let claim = Claim { edge_id: e, sparse_id: 0, point: point.clone(), eval };
            claims[e].push(claim.clone());
            initial_claims.push((e, claim));
            output_claims_proof.push((e, point, eval));
        }
        // self_claim_edges (Conv outputs, etc.)
        for &e in &self.self_claim_edges {
            if !self.output_ports.contains(&e) {
                let w = &witnesses[e][0];
                let n = w.data.as_ref().unwrap().n();
                let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"self_claim_challenge")).collect();
                let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
                let claim = Claim { edge_id: e, sparse_id: 0, point: point.clone(), eval };
                claims[e].push(claim.clone());
                initial_claims.push((e, claim));
                output_claims_proof.push((e, point, eval));
            }
        }

        // ---- 2. Boundary-edge claims (sampled on main transcript). ----
        // Seed a claim for EVERY cross-partition edge (not just the designated
        // cut edges) so each producing node is proved by its partition — see
        // partition_dag's boundary classification.
        let xpart_edges = crate::dag::partition::cross_partition_edges(partitions);
        let mut boundary_claims_proof: Vec<(EdgeId, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)> =
            Vec::new();
        for &b in &xpart_edges {
            let w = &witnesses[b][0];
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"boundary_challenge")).collect();
            let t_ie = std::time::Instant::now();
            let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
            PB_INIT_EVAL_US.fetch_add(t_ie.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            let claim = Claim { edge_id: b, sparse_id: 0, point: point.clone(), eval };
            claims[b].push(claim.clone());
            initial_claims.push((b, claim));
            boundary_claims_proof.push((b, point, eval));
        }

        // ---- 3. Route initial claims to partitions by producer. ----
        let partition_node_sets: Vec<std::collections::HashSet<NodeId>> =
            partitions.iter().map(|p| p.node_ids.iter().copied().collect()).collect();
        let mut partition_starting_claims: Vec<Vec<(EdgeId, Claim)>> =
            vec![Vec::new(); partitions.len()];
        for (e, claim) in initial_claims {
            if let Some(producer) = self.producers[e] {
                for (k, set) in partition_node_sets.iter().enumerate() {
                    if set.contains(&producer) {
                        partition_starting_claims[k].push((e, claim.clone()));
                        break;
                    }
                }
            }
        }

        // ---- 4. Run partition backward passes in parallel. ----
        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
        let t_part = std::time::Instant::now();
        let device_pool = crate::fold::tree::gpu_device_pool();
        let n_dev = device_pool.len();
        let transcript_ref: &Transcript = &*transcript;
        use rayon::prelude::*;
        let results: Vec<(Vec<Option<NodeProof>>, Vec<Vec<Claim>>)> =
            partition_starting_claims
                .into_par_iter()
                .enumerate()
                .map(|(k, starting_claims)| {
                    let device = device_pool[k % n_dev];
                    let _ = almost_goldilocks_cuda::set_device(device);
                    let mut t_k = transcript_ref.fork(b"dag_partition", k);
                    self.prove_partition(&partitions[k], witnesses, starting_claims, &mut t_k)
                })
                .collect();
        let _ = almost_goldilocks_cuda::set_device(device_pool[0]);

        // ---- 5. Merge per-partition results. ----
        let mut node_proofs: Vec<Option<NodeProof>> = vec![None; self.nodes.len()];
        for (part_node_proofs, part_claims) in results {
            for (nid, np) in part_node_proofs.into_iter().enumerate() {
                if np.is_some() {
                    node_proofs[nid] = np;
                }
            }
            for (e, cs) in part_claims.iter().enumerate() {
                claims[e].extend(cs.iter().cloned());
            }
        }

        if timing {
            use std::sync::atomic::Ordering;
            let ms = |us: u64| us as f64 / 1_000.0;
            eprintln!("[prove] partition backward: {:?}", t_part.elapsed());
            eprintln!("[prove]   thread-time: init-eval {:.0}ms, node-prove {:.0}ms ({} nodes), reducer {:.0}ms ({} reducers)",
                ms(PB_INIT_EVAL_US.swap(0, Ordering::Relaxed)),
                ms(PB_NODE_PROVE_US.swap(0, Ordering::Relaxed)), PB_N_NODES.swap(0, Ordering::Relaxed),
                ms(PB_REDUCER_US.swap(0, Ordering::Relaxed)), PB_N_REDUCERS.swap(0, Ordering::Relaxed));
        }

        // ---- 6. Lookup proofs on main transcript. ----
        let t_lk = std::time::Instant::now();
        let two_pow_proof = self.prove_two_pow(witnesses, &mut claims, transcript);
        let t_tp = t_lk.elapsed();
        let range_proof = self.prove_range(witnesses, &mut claims, transcript);
        if timing {
            eprintln!("[prove] lookup proofs: two_pow={:?} range={:?}",
                      t_tp, t_lk.elapsed() - t_tp);
        }

        // ---- 7. Snapshot edge claims. ----
        for e in 0..self.num_edges {
            edge_proofs[e].claims = claims[e].clone();
        }

        // ---- 8. Opening reducer (parallel, fork-per-edge — same as prove). ----
        self.prove_opening_reducers(witnesses, &mut edge_proofs, transcript);

        DagProof {
            node_proofs,
            edge_proofs,
            range_proof,
            two_pow_proof,
            output_claims: output_claims_proof,
            boundary_claims: boundary_claims_proof,
        }
    }

    /// Per-partition backward pass. Mirrors `prove`'s node loop but
    /// restricted to `partition.node_ids` and stops at
    /// `partition.boundary_input_edges` (those are previous partition's
    /// outputs; the previous partition handles their producers).
    fn prove_partition(
        &self,
        partition: &crate::dag::partition::PartitionDesc,
        witnesses: &[Vec<Witness>],
        starting_claims: Vec<(EdgeId, Claim)>,
        transcript: &mut Transcript,
    ) -> (Vec<Option<NodeProof>>, Vec<Vec<Claim>>) {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut node_proofs: Vec<Option<NodeProof>> = vec![None; self.nodes.len()];
        let partition_nodes: std::collections::HashSet<NodeId> =
            partition.node_ids.iter().copied().collect();
        let boundary_inputs: std::collections::HashSet<EdgeId> =
            partition.boundary_input_edges.iter().copied().collect();
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut nodes_to_prove: BTreeSet<NodeId> = BTreeSet::new();
        for (e, claim) in starting_claims {
            claims[e].push(claim);
            if let Some(producer) = self.producers[e] {
                if partition_nodes.contains(&producer) {
                    nodes_to_prove.insert(producer);
                }
            }
        }

        while let Some(&node_id) = nodes_to_prove.iter().next_back() {
            nodes_to_prove.remove(&node_id);
            // Defer if downstream consumers WITHIN the partition are still pending.
            let mut can_prove = true;
            for &edge in &self.nodes[node_id].outputs {
                if self.consumers[edge].iter()
                    .any(|c| partition_nodes.contains(c) && nodes_to_prove.contains(c))
                {
                    can_prove = false;
                    break;
                }
            }
            if !can_prove {
                nodes_to_prove.insert(node_id);
                continue;
            }

            let node = &self.nodes[node_id];
            let mut edge_ids: Vec<usize> = Vec::new();
            edge_ids.extend_from_slice(&node.inputs);
            edge_ids.extend_from_slice(&node.outputs);
            let local_witnesses: Vec<&Witness> =
                edge_ids.iter().map(|&e| &witnesses[e][0]).collect();
            let mut local_claims: Vec<&Claim> = Vec::new();
            for &e in &node.outputs {
                local_claims.extend(claims[e].iter());
            }

            // Reducer if multiple matching-arity claims.
            let mut reducer_proofs_local: Option<Vec<SumcheckProof>> = None;
            #[allow(unused_assignments)]
            let mut reduced_storage: Vec<Claim> = Vec::new();
            #[allow(unused_assignments)]
            let mut side_storage: Vec<Claim> = Vec::new();
            let reducer_witness_n = witnesses[node.outputs[0]][0].data.as_ref().unwrap().n();
            let reducible: Vec<&Claim> = local_claims.iter().copied()
                .filter(|c| c.point.len() == reducer_witness_n).collect();
            if reducible.len() > 1 {
                let side_claims: Vec<Claim> = local_claims.iter()
                    .filter(|c| c.point.len() != reducer_witness_n)
                    .map(|c| (*c).clone()).collect();
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                let reducer_edge_ids = vec![node.outputs[0]];
                let t_rd = std::time::Instant::now();
                let (proofs, rc) = reducer.prove(
                    &reducer_witness, &reducer_edge_ids, &reducible, transcript,
                );
                PB_REDUCER_US.fetch_add(t_rd.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                PB_N_REDUCERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !rc.is_empty() {
                    drop(local_claims);
                    claims[node.outputs[0]].push(rc[0].clone());
                    reducer_proofs_local = Some(proofs);
                    side_storage = side_claims;
                    reduced_storage = rc;
                    let mut new_local: Vec<&Claim> = reduced_storage.iter().collect();
                    new_local.extend(side_storage.iter());
                    local_claims = new_local;
                }
            }

            let t_np = std::time::Instant::now();
            let (proofs, new_claims) =
                node.kind.prove(&local_witnesses, &edge_ids, &local_claims, transcript);
            PB_NODE_PROVE_US.fetch_add(t_np.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            PB_N_NODES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            let mut node_claims: Vec<Claim> = new_claims.clone();
            for c in &local_claims { node_claims.push((*c).clone()); }

            for c in &new_claims {
                claims[c.edge_id].push(c.clone());
                // Key partition rule: stop at boundary inputs; only enqueue
                // producers that live IN this partition.
                if boundary_inputs.contains(&c.edge_id) { continue; }
                if let Some(p) = self.producers[c.edge_id] {
                    if p != node_id && partition_nodes.contains(&p) {
                        nodes_to_prove.insert(p);
                    }
                }
            }

            let mut combined_proofs: Vec<SumcheckProof> = Vec::new();
            if let Some(red) = reducer_proofs_local.as_ref() {
                combined_proofs.extend_from_slice(red);
            }
            combined_proofs.extend_from_slice(&proofs);
            node_proofs[node_id] = Some(NodeProof {
                sumcheck_proofs: combined_proofs,
                produced_claims: node_claims,
            });
        }

        (node_proofs, claims)
    }

    /// Sumcheck-side replay. Walks the same backward order as `prove`,
    /// invokes each `kind.verify`, and verifies the lookup proofs. The
    /// fold-tree opening (step 5) is verified separately.
    ///
    /// `witnesses` is required only for shape / role information at the
    /// boundary; the verifier never reads polynomial data, so passing
    /// data-less `Witness::new_wo_data` placeholders works for honest
    /// verifier replays. In step 4 tests we pass the actual witnesses
    /// for simplicity.

    /// Opening reducer over every eligible edge (dense committed edge with
    /// ≥ 2 non-empty claims), run in PARALLEL with one transcript fork per
    /// edge — the fork seed binds the full post-merge transcript state
    /// (all node sumchecks + lookup proofs, hence all claims) plus the
    /// edge id. After the parallel phase, each edge's combined claim is
    /// absorbed into the main transcript in edge order so downstream
    /// (fold-tree) challenges bind to every reducer output. The verifier
    /// mirrors the same fork-per-edge + absorb structure.
    fn prove_opening_reducers(
        &self,
        witnesses: &[Vec<Witness>],
        edge_proofs: &mut [EdgeProof],
        transcript: &mut Transcript,
    ) {
        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
        let t0 = std::time::Instant::now();
        let eligible: Vec<(EdgeId, Vec<usize>)> = (0..self.num_edges)
            .filter_map(|e| {
                let w = &witnesses[e][0];
                if w.poly_type != PolyType::Dense { return None; }
                if w.data.is_none() { return None; }
                let is_self_claim = self.self_claim_edges.contains(&e);
                if w.role == Role::Output && !is_self_claim { return None; }
                let non_empty: Vec<usize> = edge_proofs[e].claims.iter().enumerate()
                    .filter(|(_, c)| !c.point.is_empty())
                    .map(|(i, _)| i)
                    .collect();
                if non_empty.len() <= 1 { return None; }
                Some((e, non_empty))
            })
            .collect();

        use rayon::prelude::*;
        let ep_ref: &[EdgeProof] = edge_proofs;
        let transcript_ref: &Transcript = &*transcript;
        let results: Vec<(EdgeId, crate::sumcheck::SumcheckProof, Claim)> = eligible
            .par_iter()
            .map(|(e, non_empty)| {
                let e = *e;
                let mut t_e = transcript_ref.fork(b"open_reducer", e);
                let w = &witnesses[e][0];
                let alpha = t_e.challenge_ext2(b"opening_reducer_alpha");
                let alphas = calc_pow_vec_ext2(alpha, non_empty.len());
                let n = w.data.as_ref().unwrap().n();
                let size = 1usize << n;
                let mut eq_combined = vec![AlmostGoldilocksExt2::zero(); size];
                for (idx, &ci) in non_empty.iter().enumerate() {
                    let eq_table =
                        evaluate_lagrange_basis_ext2(&ep_ref[e].claims[ci].point);
                    for j in 0..size {
                        eq_combined[j] =
                            ext2_add(eq_combined[j], ext2_mul(alphas[idx], eq_table[j]));
                    }
                }
                let x_evals = w.data.as_ref().unwrap().evaluations();
                let x_ext2: Vec<AlmostGoldilocksExt2> = x_evals.iter()
                    .map(|&v| AlmostGoldilocksExt2::from_base(v))
                    .collect();
                let mut prover = CpuLinearSumcheckProverExt2::new(n, 2, &mut t_e);
                let mut polys = [x_ext2, eq_combined];
                let sumcheck_proof = prover.prove(&mut polys, &mut t_e);
                let combined_claim = Claim {
                    edge_id: e,
                    sparse_id: 0,
                    point: prover.challenges.clone(),
                    eval: prover.final_evals[0],
                };
                (e, sumcheck_proof, combined_claim)
            })
            .collect();

        let n_edges = results.len();
        for (e, sumcheck_proof, combined_claim) in results {
            transcript.append_u64(b"or_edge", e as u64);
            for pc in &combined_claim.point {
                transcript.append_ext2(b"or_point", pc);
            }
            transcript.append_ext2(b"or_eval", &combined_claim.eval);
            edge_proofs[e].opening_reducer = Some(vec![sumcheck_proof]);
            edge_proofs[e].claims.push(combined_claim);
        }
        if timing {
            eprintln!("[prove] opening reducer: {} edges in {:?}", n_edges, t0.elapsed());
        }
    }

    /// Verifier mirror of [`Dag::prove_opening_reducers`]: same per-edge
    /// fork + same post-phase absorb order. Per-edge checks are unchanged
    /// from the old serial replay.
    fn verify_opening_reducers(&self, proof: &DagProof, transcript: &mut Transcript) -> bool {
        use rayon::prelude::*;
        let transcript_ref: &Transcript = &*transcript;
        let edges: Vec<EdgeId> = (0..self.num_edges)
            .filter(|&e| proof.edge_proofs[e].opening_reducer.is_some())
            .collect();
        let per_edge: Vec<Option<(EdgeId, Claim)>> = edges
            .par_iter()
            .map(|&e| {
                let ep = &proof.edge_proofs[e];
                let red = ep.opening_reducer.as_ref().unwrap();
                if red.len() != 1 { return None; }
                if ep.claims.len() < 2 { return None; }
                let combined = ep.claims.last().unwrap();
                let originals: Vec<&Claim> = ep.claims.iter()
                    .take(ep.claims.len() - 1)
                    .filter(|c| !c.point.is_empty())
                    .collect();
                if originals.is_empty() { return None; }
                let mut t_e = transcript_ref.fork(b"open_reducer", e);
                let alpha = t_e.challenge_ext2(b"opening_reducer_alpha");
                let alphas = calc_pow_vec_ext2(alpha, originals.len());
                let mut claimed_sum = AlmostGoldilocksExt2::zero();
                for (i, c) in originals.iter().enumerate() {
                    claimed_sum = ext2_add(claimed_sum, ext2_mul(alphas[i], c.eval));
                }
                let n = originals[0].point.len();
                let (ok, challenges) = SumcheckVerifier::verify(
                    &red[0], claimed_sum, n, 2, &mut t_e,
                );
                if !ok { return None; }
                let one = AlmostGoldilocksExt2::one();
                let mut eq_eval = AlmostGoldilocksExt2::zero();
                for (i, c) in originals.iter().enumerate() {
                    let mut eq = one;
                    for j in 0..challenges.len() {
                        let r_j = challenges[j];
                        let p_j = c.point[j];
                        let term = ext2_add(
                            ext2_mul(r_j, p_j),
                            ext2_mul(ext2_sub(one, r_j), ext2_sub(one, p_j)),
                        );
                        eq = ext2_mul(eq, term);
                    }
                    eq_eval = ext2_add(eq_eval, ext2_mul(alphas[i], eq));
                }
                let expected = ext2_mul(combined.eval, eq_eval);
                if !ext2_field_eq(red[0].final_eval, expected) { return None; }
                if combined.point != challenges { return None; }
                Some((e, combined.clone()))
            })
            .collect();
        let mut absorbed = Vec::with_capacity(per_edge.len());
        for r in per_edge {
            match r { Some(x) => absorbed.push(x), None => return false }
        }
        for (e, combined) in absorbed {
            transcript.append_u64(b"or_edge", e as u64);
            for pc in &combined.point {
                transcript.append_ext2(b"or_point", pc);
            }
            transcript.append_ext2(b"or_eval", &combined.eval);
        }
        true
    }

    /// SOUNDNESS: a stated `output_claims` eval must equal the output claim the
    /// producing node actually proved (its `produced_claims` entry for `e` at
    /// `point`). Without this the stated eval is UNCONSTRAINED — the node's
    /// `verify` checks its own `produced_claims` copy, not the verifier's
    /// seeded output claim, and an uncommitted output edge (e.g. an `Add`
    /// output, whose claim reduces to its inputs, so it is never committed) is
    /// not bound by the fold-tree opening either. With this check the stated
    /// output is pinned to the node's proved claim, which the node relation +
    /// fold-tree opening then bind to the committed inputs/weights.
    fn output_claim_bound(
        &self,
        proof: &DagProof,
        e: EdgeId,
        point: &[AlmostGoldilocksExt2],
        eval: &AlmostGoldilocksExt2,
    ) -> bool {
        // If `e` has downstream consumers, its output-port/self-claim seed is
        // just ONE of several claims on `e` (the consumers' backward claims
        // are the others), which get combined by the per-node reducer. The
        // reducer's sumcheck binds the reduced claim to ALL its inputs —
        // including this seed — so tampering the seed breaks the reducer check
        // and is already caught. The gap is only for TERMINAL edges (no
        // consumers ⇒ the seed is the sole claim ⇒ no reducer ⇒ the node
        // verifies its own `produced_claims` copy, leaving the seed eval
        // unconstrained). Bind those here.
        if !self.consumers[e].is_empty() {
            return true;
        }
        let p = match self.producers[e] { Some(p) => p, None => return false };
        let np = match &proof.node_proofs[p] { Some(np) => np, None => return false };
        np.produced_claims.iter().any(|c| {
            c.edge_id == e
                && c.point.as_slice() == point
                && crate::util::arith::ext2_field_eq(c.eval, *eval)
        })
    }

    pub fn verify(
        &self,
        witnesses: &[Vec<Witness>],
        proof: &DagProof,
        transcript: &mut Transcript,
    ) -> bool {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];

        // ---- 1. Re-derive output-port claims from the transcript. ----
        let mut output_idx = 0usize;
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role != Role::Output { continue; }
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"output_challenge")).collect();
            // The verifier expects the prover to supply `output_claims[i].eval` —
            // the proof's view of the output evaluation. We compare both point
            // and eval to ensure transcript consistency.
            if output_idx >= proof.output_claims.len() { return false; }
            let (recorded_e, recorded_point, recorded_eval) = &proof.output_claims[output_idx];
            if *recorded_e != e { return false; }
            if *recorded_point != point { return false; }
            if !self.output_claim_bound(proof, e, &point, recorded_eval) { return false; }
            claims[e].push(Claim { edge_id: e, sparse_id: 0, point, eval: *recorded_eval });
            output_idx += 1;
        }
        for &e in &self.self_claim_edges {
            if !self.output_ports.contains(&e) {
                let w = &witnesses[e][0];
                let n = w.data.as_ref().unwrap().n();
                let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"self_claim_challenge")).collect();
                if output_idx >= proof.output_claims.len() { return false; }
                let (recorded_e, recorded_point, recorded_eval) = &proof.output_claims[output_idx];
                if *recorded_e != e || *recorded_point != point { return false; }
                if !self.output_claim_bound(proof, e, &point, recorded_eval) { return false; }
                claims[e].push(Claim { edge_id: e, sparse_id: 0, point, eval: *recorded_eval });
                output_idx += 1;
            }
        }
        if output_idx != proof.output_claims.len() { return false; }

        // ---- 2. Replay the backward pass. ----
        // Only enqueue producers of edges that actually received a claim
        // (i.e. role == Output). Auxiliary edges in `output_ports` (like
        // NonNegative aux outputs that happen to have no consumers) are
        // not claimed here — their producer is reached via the lookup
        // proof phase instead.
        let mut nodes_to_prove: BTreeSet<usize> = BTreeSet::new();
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role != Role::Output { continue; }
            if let Some(p) = self.producers[e] { nodes_to_prove.insert(p); }
        }
        for &e in &self.self_claim_edges {
            if !self.output_ports.contains(&e) {
                if let Some(p) = self.producers[e] { nodes_to_prove.insert(p); }
            }
        }

        while let Some(&node_id) = nodes_to_prove.iter().next_back() {
            nodes_to_prove.remove(&node_id);

            let mut can_prove = true;
            for &edge in &self.nodes[node_id].outputs {
                if self.consumers[edge].iter().any(|c| nodes_to_prove.contains(c)) {
                    can_prove = false;
                    break;
                }
            }
            if !can_prove {
                nodes_to_prove.insert(node_id);
                continue;
            }

            let node = &self.nodes[node_id];
            let mut edge_ids: Vec<usize> = Vec::new();
            edge_ids.extend_from_slice(&node.inputs);
            edge_ids.extend_from_slice(&node.outputs);
            let local_witnesses: Vec<&Witness> = edge_ids.iter().map(|&e| &witnesses[e][0]).collect();
            let mut local_claims: Vec<&Claim> = Vec::new();
            for &e in &node.outputs {
                local_claims.extend(claims[e].iter());
            }
            #[allow(unused_assignments)]
            let mut local_side_storage: Vec<Claim> = Vec::new();

            let np = match &proof.node_proofs[node_id] {
                Some(np) => np,
                None => return false,
            };
            let all_proofs = &np.sumcheck_proofs;

            // Reducer replay: mirror the prover side. The prover only
            // reduces claims whose `point.len() == witness.n()` (multilinear
            // evaluation claims); empty-point side-channel claims (conv's
            // `s_alpha_claim`) are NOT reduced. Match that here so the
            // verifier expects the same number of reducer proof slots.
            let reducer_witness_n = witnesses[node.outputs[0]][0].data.as_ref().unwrap().n();
            let reducible: Vec<&Claim> = local_claims
                .iter()
                .copied()
                .filter(|c| c.point.len() == reducer_witness_n)
                .collect();
            let reducer_slots = if reducible.len() > 1 { 1 } else { 0 };
            if reducer_slots > 0 {
                if all_proofs.len() < reducer_slots { return false; }
                let reduced_claim = match np.produced_claims.last() {
                    Some(c) if c.edge_id == node.outputs[0] => c.clone(),
                    _ => return false,
                };
                // Snapshot side-channel claims by clone before mutating claims.
                let side_claims: Vec<Claim> = local_claims
                    .iter()
                    .filter(|c| c.point.len() != reducer_witness_n)
                    .map(|c| (*c).clone())
                    .collect();
                // reducer.verify expects [originals..., reduced_claim].
                let mut reducer_claims: Vec<Claim> = reducible.iter().map(|c| (*c).clone()).collect();
                reducer_claims.push(reduced_claim.clone());
                let reducer_claims_refs: Vec<&Claim> = reducer_claims.iter().collect();
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                let reducer_proof_refs: Vec<&SumcheckProof> = all_proofs[..reducer_slots].iter().collect();
                let reducer_ok = reducer.verify(
                    &reducer_witness,
                    &reducer_claims_refs,
                    &reducer_proof_refs,
                    transcript,
                );
                if !reducer_ok { return false; }
                drop(local_claims);
                claims[node.outputs[0]].push(reduced_claim);
                let last_idx = claims[node.outputs[0]].len() - 1;
                let reduced_ref = &claims[node.outputs[0]][last_idx];
                local_side_storage = side_claims;
                let mut new_local: Vec<&Claim> = vec![reduced_ref];
                new_local.extend(local_side_storage.iter());
                local_claims = new_local;
            }

            let node_proof_refs: Vec<&SumcheckProof> = all_proofs[reducer_slots..].iter().collect();

            // Pass the prover's stored `[produced..., consumed...]` snapshot
            // straight to `kind.verify` — mirrors zk-torch-3's local_claims
            // layout. Self-claim blocks (Conv, MaxPool, ReLU certificate)
            // emit extra entries (`y_self_claim`, `s_alpha_claim`) that they
            // read by index; rebuilding flat from `inputs + local_claims`
            // (3-4 entries) breaks those positional reads. `kind.prove`
            // already stamped this list at prove time, so the verifier
            // replays the exact same view.
            let flat_refs: Vec<&Claim> = np.produced_claims.iter().collect();

            let ok = node.kind.verify(&local_witnesses, &flat_refs, &node_proof_refs, transcript);
            if !ok {
                if std::env::var("ZK4_VERIFY_DBG").is_ok() {
                    eprintln!("[verify_dbg] node {} ({:?}) verify FAILED", node_id, node.kind);
                }
                return false;
            }

            // Propagate input-edge claims upstream so consumers / lookup
            // proofs see them; pick those whose edge_id ∈ node.inputs.
            let mut input_claims_pushed: Vec<&Claim> = Vec::new();
            for c in &np.produced_claims {
                if node.inputs.contains(&c.edge_id) {
                    input_claims_pushed.push(c);
                }
            }
            let input_claims_owned: Vec<Claim> =
                input_claims_pushed.iter().map(|c| (*c).clone()).collect();
            for c in input_claims_owned {
                let e = c.edge_id;
                claims[e].push(c);
                if let Some(p) = self.producers[e] {
                    nodes_to_prove.insert(p);
                }
            }
        }

        let dbg = std::env::var("ZK4_VERIFY_DBG").is_ok();
        // ---- 3. Verify lookup proofs. ----
        if let Some(tp) = &proof.two_pow_proof {
            if !self.verify_two_pow(witnesses, &claims, tp, transcript) {
                if dbg { eprintln!("[verify_dbg] two_pow FAILED"); }
                return false;
            }
        } else if !self.two_pow.is_empty() {
            if dbg { eprintln!("[verify_dbg] two_pow proof MISSING"); }
            return false;
        }
        if let Some(rp) = &proof.range_proof {
            if !self.verify_range(witnesses, &claims, rp, transcript) {
                if dbg { eprintln!("[verify_dbg] range FAILED"); }
                return false;
            }
        } else if !self.range.is_empty() {
            if dbg { eprintln!("[verify_dbg] range proof MISSING"); }
            return false;
        }

        // ---- 4. Verify opening reducers. ----
        // For every edge with an `opening_reducer`, replay the sumcheck:
        //   claimed_sum = Σ_i α^i · v_i  (over the K original claims)
        //   sumcheck on `x · eq_combined` should land on
        //   final_eval == x(R) · Σ_i α^i · eq(R, r_i)
        if !self.verify_opening_reducers(proof, transcript) {
            if dbg { eprintln!("[verify_dbg] opening_reducer FAILED"); }
            return false;
        }

        true
    }

    /// Multi-GPU partitioned verify. Mirrors `verify` but expects the
    /// proof to come from `prove_partitioned`: extra `boundary_claims`
    /// derived from the main transcript, and node sumchecks verified
    /// against per-partition forked transcripts.
    pub fn verify_partitioned(
        &self,
        witnesses: &[Vec<Witness>],
        proof: &DagProof,
        partitions: &[crate::dag::partition::PartitionDesc],
        transcript: &mut Transcript,
    ) -> bool {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];

        // ---- 1. Output-port claims. ----
        let mut output_idx = 0usize;
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role != Role::Output { continue; }
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"output_challenge")).collect();
            if output_idx >= proof.output_claims.len() { return false; }
            let (recorded_e, recorded_point, recorded_eval) = &proof.output_claims[output_idx];
            if *recorded_e != e || *recorded_point != point { return false; }
            if !self.output_claim_bound(proof, e, &point, recorded_eval) { return false; }
            claims[e].push(Claim { edge_id: e, sparse_id: 0, point, eval: *recorded_eval });
            output_idx += 1;
        }
        for &e in &self.self_claim_edges {
            if !self.output_ports.contains(&e) {
                let w = &witnesses[e][0];
                let n = w.data.as_ref().unwrap().n();
                let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"self_claim_challenge")).collect();
                if output_idx >= proof.output_claims.len() { return false; }
                let (recorded_e, recorded_point, recorded_eval) = &proof.output_claims[output_idx];
                if *recorded_e != e || *recorded_point != point { return false; }
                if !self.output_claim_bound(proof, e, &point, recorded_eval) { return false; }
                claims[e].push(Claim { edge_id: e, sparse_id: 0, point, eval: *recorded_eval });
                output_idx += 1;
            }
        }
        if output_idx != proof.output_claims.len() { return false; }

        // ---- 2. Boundary-edge claims. ---- (all cross-partition edges; must
        // match prove_partitioned's `cross_partition_edges` set exactly.)
        let xpart_edges = crate::dag::partition::cross_partition_edges(partitions);
        if proof.boundary_claims.len() != xpart_edges.len() { return false; }
        for (i, &b) in xpart_edges.iter().enumerate() {
            let w = &witnesses[b][0];
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<_> = (0..n).map(|_| transcript.challenge_ext2(b"boundary_challenge")).collect();
            let (recorded_e, recorded_point, recorded_eval) = &proof.boundary_claims[i];
            if *recorded_e != b || *recorded_point != point { return false; }
            claims[b].push(Claim { edge_id: b, sparse_id: 0, point, eval: *recorded_eval });
        }

        // ---- 3. Replay per-partition backward pass. ----
        // Each partition uses fork(b"dag_partition", k) — same as prover. Per
        // partition we walk its node_ids in the same canonical order
        // (BTreeSet reverse) and verify each node's stored proof.
        for (k, partition) in partitions.iter().enumerate() {
            let mut t_k = transcript.fork(b"dag_partition", k);
            let partition_nodes: std::collections::HashSet<NodeId> =
                partition.node_ids.iter().copied().collect();
            let boundary_inputs: std::collections::HashSet<EdgeId> =
                partition.boundary_input_edges.iter().copied().collect();
            let mut nodes_to_verify: BTreeSet<NodeId> = BTreeSet::new();
            // Seed from edges that already have a claim AND whose
            // producer is in this partition.
            for e in 0..self.num_edges {
                if claims[e].is_empty() { continue; }
                if let Some(p) = self.producers[e] {
                    if partition_nodes.contains(&p) {
                        nodes_to_verify.insert(p);
                    }
                }
            }

            while let Some(&node_id) = nodes_to_verify.iter().next_back() {
                nodes_to_verify.remove(&node_id);
                let mut can_verify = true;
                for &edge in &self.nodes[node_id].outputs {
                    if self.consumers[edge].iter()
                        .any(|c| partition_nodes.contains(c) && nodes_to_verify.contains(c))
                    {
                        can_verify = false;
                        break;
                    }
                }
                if !can_verify {
                    nodes_to_verify.insert(node_id);
                    continue;
                }

                let node = &self.nodes[node_id];
                let mut edge_ids: Vec<usize> = Vec::new();
                edge_ids.extend_from_slice(&node.inputs);
                edge_ids.extend_from_slice(&node.outputs);
                let local_witnesses: Vec<&Witness> =
                    edge_ids.iter().map(|&e| &witnesses[e][0]).collect();
                let mut local_claims: Vec<&Claim> = Vec::new();
                for &e in &node.outputs {
                    local_claims.extend(claims[e].iter());
                }
                #[allow(unused_assignments)]
                let mut local_side_storage: Vec<Claim> = Vec::new();

                let np = match &proof.node_proofs[node_id] {
                    Some(np) => np,
                    None => return false,
                };
                let all_proofs = &np.sumcheck_proofs;

                let reducer_witness_n = witnesses[node.outputs[0]][0].data.as_ref().unwrap().n();
                let reducible: Vec<&Claim> = local_claims.iter().copied()
                    .filter(|c| c.point.len() == reducer_witness_n).collect();
                let reducer_slots = if reducible.len() > 1 { 1 } else { 0 };
                if reducer_slots > 0 {
                    if all_proofs.len() < reducer_slots { return false; }
                    let reduced_claim = match np.produced_claims.last() {
                        Some(c) if c.edge_id == node.outputs[0] => c.clone(),
                        _ => return false,
                    };
                    let side_claims: Vec<Claim> = local_claims.iter()
                        .filter(|c| c.point.len() != reducer_witness_n)
                        .map(|c| (*c).clone()).collect();
                    let mut reducer_claims: Vec<Claim> =
                        reducible.iter().map(|c| (*c).clone()).collect();
                    reducer_claims.push(reduced_claim.clone());
                    let reducer_claims_refs: Vec<&Claim> = reducer_claims.iter().collect();
                    let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                    let reducer_proof_refs: Vec<&SumcheckProof> =
                        all_proofs[..reducer_slots].iter().collect();
                    let reducer_ok = reducer.verify(
                        &reducer_witness, &reducer_claims_refs,
                        &reducer_proof_refs, &mut t_k,
                    );
                    if !reducer_ok { return false; }
                    drop(local_claims);
                    claims[node.outputs[0]].push(reduced_claim);
                    let last_idx = claims[node.outputs[0]].len() - 1;
                    let reduced_ref = &claims[node.outputs[0]][last_idx];
                    local_side_storage = side_claims;
                    let mut new_local: Vec<&Claim> = vec![reduced_ref];
                    new_local.extend(local_side_storage.iter());
                    local_claims = new_local;
                }
                let node_proof_refs: Vec<&SumcheckProof> =
                    all_proofs[reducer_slots..].iter().collect();
                let flat_refs: Vec<&Claim> = np.produced_claims.iter().collect();
                let ok = node.kind.verify(
                    &local_witnesses, &flat_refs, &node_proof_refs, &mut t_k,
                );
                if !ok { return false; }

                let mut input_claims_pushed: Vec<&Claim> = Vec::new();
                for c in &np.produced_claims {
                    if node.inputs.contains(&c.edge_id) {
                        input_claims_pushed.push(c);
                    }
                }
                let input_claims_owned: Vec<Claim> =
                    input_claims_pushed.iter().map(|c| (*c).clone()).collect();
                for c in input_claims_owned {
                    let e = c.edge_id;
                    claims[e].push(c);
                    // Stop at boundary inputs; only enqueue producers in this partition.
                    if boundary_inputs.contains(&e) { continue; }
                    if let Some(p) = self.producers[e] {
                        if partition_nodes.contains(&p) {
                            nodes_to_verify.insert(p);
                        }
                    }
                }
            }
        }

        // ---- 4. Lookup proofs on main transcript. ----
        if let Some(tp) = &proof.two_pow_proof {
            if !self.verify_two_pow(witnesses, &claims, tp, transcript) { return false; }
        } else if !self.two_pow.is_empty() {
            return false;
        }
        if let Some(rp) = &proof.range_proof {
            if !self.verify_range(witnesses, &claims, rp, transcript) { return false; }
        } else if !self.range.is_empty() {
            return false;
        }

        // ---- 5. Verify opening reducers (same as verify). ----
        if !self.verify_opening_reducers(proof, transcript) { return false; }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    use crate::dag::{DagBuilder, DataType};
    use crate::util::arith::int_to_f;

    fn make_input(shape: Vec<usize>, raw: Vec<i128>) -> Witness {
        let evals = raw.iter().map(|&v| int_to_f(v)).collect();
        Witness::new(shape, evals, DataType::Int, 0, Role::Input)
    }

    fn make_const(shape: Vec<usize>, raw: Vec<i128>) -> Witness {
        let evals = raw.iter().map(|&v| int_to_f(v)).collect();
        Witness::new(shape, evals, DataType::Int, 0, Role::Constant)
    }

    /// Build `y = x + w` with `w` constant, register `y` as an output
    /// port, and round-trip prove → verify.
    #[test]
    /// Partition smoke test: build a chained-add DAG, partition it into
    /// 2 subcircuits at the midpoint, run `prove_partitioned` and
    /// `verify_partitioned` against the same proof. Verifies the
    /// partitioned path produces a proof identical-by-construction to
    /// the single-partition path (modulo boundary claims absorbed into
    /// the transcript).
    #[test]
    fn prove_verify_partitioned_chain() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        // Build a chain: x -> +w0 -> y0 -> +w1 -> y1 -> +w2 -> y2 -> +w3 -> y3
        let x = g.input(vec![8], DataType::Int);
        let w0 = g.param(make_const(vec![8], vec![1; 8]));
        let w1 = g.param(make_const(vec![8], vec![1; 8]));
        let w2 = g.param(make_const(vec![8], vec![1; 8]));
        let w3 = g.param(make_const(vec![8], vec![1; 8]));
        let y0 = g.add(x, w0)[0];
        let y1 = g.add(y0, w1)[0];
        let y2 = g.add(y1, w2)[0];
        let _y3 = g.add(y2, w3)[0];
        // Register `y1` as a layer boundary so we can split there.
        g.layer_boundaries.push(y1);
        let (mut dag, mut witnesses) = g.compile();

        let x_w = make_input(vec![8], (0..8i128).collect());
        dag.run(&mut witnesses, &[(0, x_w)]);

        // Partition into 2 — boundary at y1 (the middle of the chain).
        dag.set_partition_boundaries(2);
        assert_eq!(dag.boundary_edges.len(), 1, "2-partition needs 1 boundary");
        let partitions = crate::dag::partition::partition_dag(&dag, &dag.boundary_edges);
        assert_eq!(partitions.len(), 2);

        let mut t_prove = Transcript::new(b"part-chain");
        let proof = dag.prove_partitioned(&witnesses, &partitions, &mut t_prove);
        assert_eq!(proof.boundary_claims.len(), 1);

        let mut t_verify = Transcript::new(b"part-chain");
        let ok = dag.verify_partitioned(&witnesses, &proof, &partitions, &mut t_verify);
        assert!(ok, "verify_partitioned should accept");
    }

    /// A skip connection makes an edge cross the cut without being the
    /// designated boundary.  Both the producing and the consuming partition
    /// make a claim on such an edge, at different points, so it must be
    /// committed or nothing forces the two claims to be about the same tensor.
    /// Committing only `boundary_edges` (the designated cuts) misses it.
    #[test]
    fn skip_connection_edge_crosses_and_is_committed() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        // x -> +w0 -> y0 -> +w1 -> y1 -> +w2 -> y2 -> +y0 -> y3
        // The last add is the skip: it reads y0, produced before the cut.
        let x = g.input(vec![8], DataType::Int);
        let w0 = g.param(make_const(vec![8], vec![1; 8]));
        let w1 = g.param(make_const(vec![8], vec![1; 8]));
        let w2 = g.param(make_const(vec![8], vec![1; 8]));
        let y0 = g.add(x, w0)[0];
        let y1 = g.add(y0, w1)[0];
        let y2 = g.add(y1, w2)[0];
        let _y3 = g.add(y2, y0)[0];
        g.layer_boundaries.push(y1);
        let (mut dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![8], (0..8i128).collect()))]);

        dag.set_partition_boundaries(2);
        assert_eq!(dag.boundary_edges, vec![y1], "y1 is the designated cut");
        assert!(dag.crossing_edges.contains(&y1), "designated cut must cross");
        assert!(dag.crossing_edges.contains(&y0),
                "the skip edge crosses the cut but is not a designated boundary; \
                 crossing_edges = {:?}", dag.crossing_edges);
        assert!(dag.should_commit(&witnesses[y0][0], y0),
                "a crossing edge must be committed");

        let partitions = crate::dag::partition::partition_dag(&dag, &dag.boundary_edges);
        let mut t_prove = Transcript::new(b"part-skip");
        let proof = dag.prove_partitioned(&witnesses, &partitions, &mut t_prove);
        let mut t_verify = Transcript::new(b"part-skip");
        assert!(dag.verify_partitioned(&witnesses, &proof, &partitions, &mut t_verify),
                "verify_partitioned should accept the honest skip-connection proof");
    }

    fn prove_verify_add_smoke() {
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let w = g.param(make_const(vec![8], (0..8i128).map(|i| i + 1).collect()));
        let _y = g.add(x, w)[0];
        // `_y` is automatically registered as an output_port by `compile()`
        // since it has no downstream consumers.
        let (dag, mut witnesses) = g.compile();

        // Provide input data.
        let x_w = make_input(vec![8], vec![10, 20, 30, 40, 50, 60, 70, 80]);
        dag.run(&mut witnesses, &[(0, x_w)]);

        let mut t_prove = Transcript::new(b"dag-smoke");
        let proof = dag.prove(&witnesses, &mut t_prove);

        // We expect 1 output claim (y).
        assert_eq!(proof.output_claims.len(), 1);
        // No lookup nodes in this DAG.
        assert!(proof.range_proof.is_none());
        assert!(proof.two_pow_proof.is_none());

        // Replay verify.
        let mut t_verify = Transcript::new(b"dag-smoke");
        let ok = dag.verify(&witnesses, &proof, &mut t_verify);
        assert!(ok, "verify should accept honest proof");
    }

    /// Same DAG, but tamper with an output_claim eval — verifier must
    /// reject (the recorded eval is checked against transcript-derived
    /// point + add.verify's eval constraint).
    /// Prove a DAG that includes an internal NonNegative range check, then
    /// verify round-trip.
    #[test]
    fn prove_verify_with_range_node() {
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let w = g.param(make_const(vec![8], (1..=8i128).collect()));
        let y = g.add(x, w)[0];
        // Attach an explicit range check with table_log = 6 (covers [0, 64))
        // so `add_nonneg_node`'s default doesn't pull in TABLE_SIZE_LOG.
        let nid = g.nodes.len();
        let nn = BasicBlockType::NonNegative(crate::basicblock::range::NonNegative::new(6));
        let _ = g.add_gkr_node(vec![y], nn);
        g.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
        g.range.push(nid);

        let (dag, mut witnesses) = g.compile();
        let x_w = make_input(vec![8], vec![0, 1, 5, 10, 20, 30, 40, 55]);
        dag.run(&mut witnesses, &[(0, x_w)]);

        let mut t_prove = Transcript::new(b"range-smoke");
        let proof = dag.prove(&witnesses, &mut t_prove);
        assert!(proof.range_proof.is_some(), "should produce range proof");

        let mut t_verify = Transcript::new(b"range-smoke");
        let ok = dag.verify(&witnesses, &proof, &mut t_verify);
        assert!(ok, "honest range proof should verify");
    }

    /// An edge consumed by multiple downstream nodes accumulates multiple
    /// claims. The opening reducer should fold them into one combined
    /// claim, and verify should accept the combined proof.
    #[test]
    fn opening_reducer_combines_multi_claims() {
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 2, 3, 4]));
        // y1 = x + w
        let y1 = g.add(x, w)[0];
        // y2 = y1 + w   (two distinct consumers of w → two claims on w)
        let _y2 = g.add(y1, w)[0];
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![10, 20, 30, 40]))]);

        let mut t_prove = Transcript::new(b"reducer-smoke");
        let proof = dag.prove(&witnesses, &mut t_prove);

        // Sanity: at least one edge_proof should carry an opening_reducer.
        let any_reducer = proof.edge_proofs.iter().any(|ep| ep.opening_reducer.is_some());
        assert!(any_reducer, "expected at least one opening reducer for shared edge");

        let mut t_verify = Transcript::new(b"reducer-smoke");
        let ok = dag.verify(&witnesses, &proof, &mut t_verify);
        assert!(ok, "verify should accept reducer proof");
    }

    /// Two NonNegative checks in the same DAG → one combined table
    /// sumcheck, one bool-group sumcheck, accepted by the verifier.
    #[test]
    fn prove_verify_two_range_nodes() {
        let mut g = DagBuilder::new();
        let x = g.input(vec![8], DataType::Int);
        let w = g.param(make_const(vec![8], vec![1, 1, 1, 1, 1, 1, 1, 1]));
        let y = g.add(x, w)[0];
        // Range-check y AND a second derived edge.
        let z = g.add(y, w)[0];
        for &edge in &[y, z] {
            let nid = g.nodes.len();
            let nn = BasicBlockType::NonNegative(crate::basicblock::range::NonNegative::new(6));
            let _ = g.add_gkr_node(vec![edge], nn);
            g.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
            g.range.push(nid);
        }
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![8], vec![0, 1, 2, 3, 4, 5, 6, 7]))]);

        let mut t_prove = Transcript::new(b"two-range");
        let proof = dag.prove(&witnesses, &mut t_prove);
        assert!(proof.range_proof.is_some());
        let rp = proof.range_proof.as_ref().unwrap();
        assert_eq!(rp.middle_claims.len(), 2, "two range nodes → 2 middle_claims");

        let mut t_verify = Transcript::new(b"two-range");
        assert!(dag.verify(&witnesses, &proof, &mut t_verify));
    }

    /// Tamper with the opening reducer's final_eval → verifier rejects.
    #[test]
    fn verify_rejects_tampered_reducer() {
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 2, 3, 4]));
        let y1 = g.add(x, w)[0];
        let _y2 = g.add(y1, w)[0];
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![10, 20, 30, 40]))]);

        let mut t_prove = Transcript::new(b"red-tamper");
        let mut proof = dag.prove(&witnesses, &mut t_prove);

        // Find the reduced edge and bump its sumcheck final_eval.
        let bumped = proof.edge_proofs.iter_mut().find_map(|ep| {
            ep.opening_reducer.as_mut().map(|r| {
                r[0].final_eval = ext2_add(r[0].final_eval, AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1)));
                true
            })
        });
        assert_eq!(bumped, Some(true), "expected to find a reducer to tamper");

        let mut t_verify = Transcript::new(b"red-tamper");
        assert!(!dag.verify(&witnesses, &proof, &mut t_verify));
    }

    #[test]
    fn verify_rejects_tampered_output_claim() {
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 2, 3, 4]));
        let _y = g.add(x, w)[0];
        // `_y` is automatically registered as an output_port by `compile()`
        // since it has no downstream consumers.
        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[(0, make_input(vec![4], vec![10, 20, 30, 40]))]);

        let mut t_prove = Transcript::new(b"tamper");
        let mut proof = dag.prove(&witnesses, &mut t_prove);

        // Bump the output eval.
        let (_, _, eval) = &mut proof.output_claims[0];
        *eval = crate::util::arith::ext2_add(*eval, AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1)));

        let mut t_verify = Transcript::new(b"tamper");
        assert!(!dag.verify(&witnesses, &proof, &mut t_verify));
    }
}

