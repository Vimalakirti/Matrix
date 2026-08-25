pub mod builder;
pub mod deeplabv3plus;
pub mod dense;
pub mod llama;
pub mod gpt2;
pub mod bert;
pub mod gptj;
pub mod oneshot;
pub mod partition;
pub mod pointpillar;
pub mod resnet;
pub mod unet3d;
pub mod vgg;
pub mod verfcnn_vgg;
pub mod whisper;
pub mod yolo;

pub use builder::DagBuilder;
pub use dense::dense_add_relu;
pub use llama::*;
pub use gpt2::*;
pub use bert::*;
pub use gptj::*;
pub use partition::{PartitionDesc, partition_dag, edge_partition_map};
pub use vgg::*;

use std::collections::{BTreeSet, HashMap, HashSet};
use rayon::prelude::*;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use plonky2::{timed, util::timing::TimingTree};
use serde::{Deserialize, Serialize};

use goldilocks_cuda::basefold::{BasefoldCommitment, BasefoldTable, BasefoldVerifier};

use std::sync::Arc;
use goldilocks_cuda::DeviceBuffer;

use crate::basicblock::{BasicBlock, BasicBlockType, Reducer};
use crate::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldOpeningProof, BasefoldVerifierKey, GpuCommitmentStore};
use crate::commit::cpu_basefold::cpu_full_open_ext2;
use crate::poly::{evaluate_lagrange_basis_ext2, DenseMLPoly, DeviceDenseMLPoly, MLPoly, SparseMLPoly};
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{calc_pow_vec_ext2, ext2_add, ext2_field_eq, ext2_mul, ext2_sub, get_n};
use crate::SF_INT;

pub type NodeId = usize;
pub type EdgeId = usize;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct AliasId(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Auxiliary,
    Constant,
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Uint,
    Int,
    Bool,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyType {
    Dense,
    Sparse,
}

/// Witness: multilinear polynomial with metadata.
pub struct Witness {
    pub shape: Vec<usize>,
    pub data: Option<Box<dyn MLPoly>>,
    pub poly_type: PolyType,
    pub data_type: DataType,
    pub sf: usize,
    pub role: Role,
}

impl std::fmt::Debug for Witness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Witness")
            .field("shape", &self.shape)
            .field("poly_type", &self.poly_type)
            .field("data_type", &self.data_type)
            .field("sf", &self.sf)
            .field("role", &self.role)
            .field("has_data", &self.data.is_some())
            .finish()
    }
}

impl Witness {
    pub fn new(shape: Vec<usize>, data: Vec<GoldilocksField>, data_type: DataType, sf: usize, role: Role) -> Self {
        let n = get_n(&shape);
        let padded = if data.len() < (1 << n) {
            let mut d = data.clone();
            d.resize(1 << n, GoldilocksField(0));
            d
        } else {
            data
        };
        Self {
            shape,
            data: Some(Box::new(DenseMLPoly::new(n, padded))),
            poly_type: PolyType::Dense,
            data_type,
            sf,
            role,
        }
    }

    pub fn new_sparse(shape: Vec<usize>, data: SparseMLPoly, data_type: DataType, sf: usize, role: Role) -> Self {
        Self {
            shape,
            data: Some(Box::new(data)),
            poly_type: PolyType::Sparse,
            data_type,
            sf,
            role,
        }
    }

    /// Create a witness from a pre-built DenseMLPoly (e.g. bit decomposition auxiliary).
    pub fn new_dense_poly(shape: Vec<usize>, poly: DenseMLPoly, data_type: DataType, sf: usize, role: Role) -> Self {
        Self {
            shape,
            data: Some(Box::new(poly)),
            poly_type: PolyType::Dense,
            data_type,
            sf,
            role,
        }
    }

    /// Create a witness backed by a device-resident polynomial. Callers reading
    /// via `evaluations_ref()` etc. trigger an on-demand download.
    pub fn new_device(shape: Vec<usize>, buf: Arc<DeviceBuffer<u64>>, data_type: DataType, sf: usize, role: Role) -> Self {
        let n = get_n(&shape);
        assert_eq!(buf.len(), 1usize << n, "device buffer size {} does not match shape padding 1<<{}", buf.len(), n);
        Self {
            shape,
            data: Some(Box::new(DeviceDenseMLPoly::from_device(n, buf))),
            poly_type: PolyType::Dense,
            data_type,
            sf,
            role,
        }
    }

    /// True if `data` is a `DeviceDenseMLPoly`.
    pub fn is_device_resident(&self) -> bool {
        self.data.as_ref()
            .map(|d| d.as_any().downcast_ref::<DeviceDenseMLPoly>().is_some())
            .unwrap_or(false)
    }

    /// Borrow the device buffer if the witness is device-resident.
    pub fn device_buf(&self) -> Option<&Arc<DeviceBuffer<u64>>> {
        self.data.as_ref()
            .and_then(|d| d.as_any().downcast_ref::<DeviceDenseMLPoly>())
            .map(|d| &d.buf)
    }

    /// Get the data as an `Arc<DeviceBuffer<u64>>`, uploading from host if not
    /// already device-resident. Used by GPU `run_gpu` impls.
    pub fn as_device_buf(&self) -> Arc<DeviceBuffer<u64>> {
        if let Some(buf) = self.device_buf() {
            return Arc::clone(buf);
        }
        let evals = self.data.as_ref().expect("witness has no data").evaluations_ref();
        let raw: Vec<u64> = evals.iter().map(|f| f.0).collect();
        Arc::new(DeviceBuffer::<u64>::from_slice(&raw).expect("host->device upload failed"))
    }

    /// Free the device-resident witness buffer, downloading evaluations to
    /// host first. After eviction the witness behaves like a CPU-mode
    /// `DenseMLPoly` — same evaluation API, no GPU footprint. No-op if the
    /// witness isn't device-resident.
    ///
    /// Use after the commit phase: prove-time code paths read from host
    /// (`evaluations_ref`) and the einsum prover re-uploads from host
    /// per-task, so freeing here doesn't add prove-time cost. Saves the
    /// entire intermediate-witness GPU footprint (typically 80–100 GB for
    /// full LLaMA-2-7B) during prove and opening.
    pub fn evict_device_buffer(&mut self) {
        if !self.is_device_resident() { return; }
        let n;
        let evals;
        {
            let any = self.data.as_mut().unwrap().as_any_mut();
            let dp = any.downcast_mut::<DeviceDenseMLPoly>()
                .expect("checked is_device_resident");
            n = dp.n;
            evals = dp.take_host_evals();
        }
        self.data = Some(Box::new(DenseMLPoly::new(n, evals)));
    }

    /// Zero out MLE padding region (indices beyond actual shape bounds).
    /// Needed when non-power-of-2 shapes cause operations like Add broadcasting
    /// to leave non-zero values in the padded region.
    pub fn zero_pad_if_needed(&mut self) {
        // Only process if data exists and shape has non-power-of-2 dims
        let needs_fix = self.shape.iter().any(|&s| s != s.next_power_of_two());
        if !needs_fix { return; }
        if self.data.is_none() { return; }

        let padded: Vec<usize> = self.shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
        let total_padded: usize = padded.iter().product();
        let data = self.data.as_mut().unwrap();
        let ndims = self.shape.len();

        // Little-endian MLE layout: last shape dimension (W) has stride 1 (lowest bits),
        // first shape dimension (C) has the highest bits.
        // flat_idx = w + h*W_pad + d*W_pad*H_pad + c*W_pad*H_pad*D_pad
        // So we decompose from the LAST dimension first.
        let padded_rev: Vec<usize> = padded.iter().rev().cloned().collect();
        let shape_rev: Vec<usize> = self.shape.iter().rev().cloned().collect();

        for flat_idx in 0..total_padded {
            let mut remaining = flat_idx;
            let mut out_of_bounds = false;
            for dim in 0..ndims {
                let dim_idx = remaining % padded_rev[dim];
                remaining /= padded_rev[dim];
                if dim_idx >= shape_rev[dim] {
                    out_of_bounds = true;
                    break;
                }
            }
            if out_of_bounds {
                *data.index_mut(flat_idx) = GoldilocksField(0);
            }
        }
    }

    pub fn new_wo_data(shape: Vec<usize>, data_type: DataType, sf: usize, role: Role) -> Self {
        Self {
            shape,
            data: None,
            poly_type: PolyType::Dense,
            data_type,
            sf,
            role,
        }
    }

    pub fn clear_data(&mut self) {
        self.data = None;
    }

    pub fn get(&self, indices: &[usize]) -> GoldilocksField {
        let shape_next_pow: Vec<usize> = self.shape.iter().map(|&s| s.next_power_of_two()).collect();
        let index: usize = indices
            .iter()
            .enumerate()
            .map(|(i, &index)| index * shape_next_pow[..i].iter().product::<usize>())
            .sum();
        self.data.as_ref().unwrap().index(index)
    }
}

impl Clone for Witness {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            data: self.data.as_ref().map(|d| d.clone_box()),
            poly_type: self.poly_type,
            data_type: self.data_type,
            sf: self.sf,
            role: self.role,
        }
    }
}

/// Claim: assertion about a polynomial evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub edge_id: EdgeId,
    pub sparse_id: usize,
    pub point: Vec<GoldilocksExt2>,
    pub eval: GoldilocksExt2,
}

/// Lookup proof for range/two_pow.
#[derive(Debug, Clone)]
pub struct LookupProof {
    pub table_proofs: Vec<SumcheckProof>,
    pub middle_claims: Vec<Vec<GoldilocksExt2>>,
    pub bool_proofs: Vec<SumcheckProof>,
}

/// Edge proof: claims + opening proofs.
#[derive(Debug, Clone)]
pub struct EdgeProof {
    pub claims: Vec<Claim>,
    /// Reducer proof combining multiple claims into one for efficient opening.
    /// When present: claims[0..K-1] are originals, claims[K-1] is the combined claim.
    /// Only the combined claim gets a PCS opening proof.
    pub opening_reducer: Option<Vec<SumcheckProof>>,
    pub dense_opening_proof: Vec<BasefoldOpeningProof>,
    pub sparse_opening_proof: Vec<BasefoldOpeningProof>,
}

impl EdgeProof {
    pub fn new() -> Self {
        Self {
            claims: Vec::new(),
            opening_reducer: None,
            dense_opening_proof: Vec::new(),
            sparse_opening_proof: Vec::new(),
        }
    }
}

/// Proof for a single partition.
#[derive(Debug, Clone)]
pub struct PartitionProof {
    pub node_proofs: Vec<Option<(Vec<SumcheckProof>, Vec<Claim>)>>,
    pub reducer_proofs: Vec<Option<Vec<SumcheckProof>>>,
}

/// Proof from parallel proving across partitions.
#[derive(Debug, Clone)]
pub struct ParallelProof {
    pub boundary_evals: Vec<(EdgeId, Vec<GoldilocksExt2>, GoldilocksExt2)>,
    pub partition_proofs: Vec<PartitionProof>,
    pub edge_proofs: Vec<EdgeProof>,
    pub range_proof: LookupProof,
    pub two_pow_proof: LookupProof,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: BasicBlockType,
    pub inputs: Vec<EdgeId>,
    pub outputs: Vec<EdgeId>,
}

/// The DAG — computation graph for ZKML.
pub struct Dag {
    pub nodes: Vec<Node>,
    pub num_edges: usize,
    pub topo: Vec<NodeId>,
    pub topo_levels: Vec<Vec<NodeId>>,
    pub range: Vec<NodeId>,
    pub two_pow: Vec<NodeId>,
    pub consumers: Vec<Vec<NodeId>>,
    pub producers: Vec<Option<NodeId>>,
    pub input_ports: Vec<EdgeId>,
    pub output_ports: Vec<EdgeId>,
    /// All layer boundary edges recorded during model construction.
    /// These are candidate split points for parallel proving.
    pub layer_boundaries: Vec<EdgeId>,
    /// Active partition boundary edges (subset of layer_boundaries).
    /// These edges will be force-committed even if they are intermediate outputs.
    pub boundary_edges: Vec<EdgeId>,
    /// Output edges that need self-claims (e.g. Conv2D output edge).
    /// These edges must be committed and opened even though they are Role::Output.
    pub self_claim_edges: HashSet<EdgeId>,
    // Alias view
    pub edge_aliases: Vec<Vec<AliasId>>,
    pub alias_to_edge: Vec<EdgeId>,
    pub alias_to_consumer: Vec<NodeId>,
    pub alias_input_slot: Vec<usize>,
}

impl std::fmt::Debug for Dag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dag")
            .field("num_nodes", &self.nodes.len())
            .field("num_edges", &self.num_edges)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .field("boundary_edges", &self.boundary_edges)
            .finish()
    }
}

impl Dag {
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Select partition boundaries from layer_boundaries.
    /// `num_partitions` is the number of partitions (must be >= 1).
    /// Selects evenly-spaced boundary edges from `layer_boundaries`.
    /// With N layers, there are N-1 candidate boundaries (between layers).
    /// For `num_partitions` partitions, we pick `num_partitions - 1` boundaries.
    pub fn set_partition_boundaries(&mut self, num_partitions: usize) {
        assert!(num_partitions >= 1, "need at least 1 partition");
        if num_partitions == 1 || self.layer_boundaries.is_empty() {
            self.boundary_edges = vec![];
            return;
        }
        let num_boundaries = num_partitions - 1;
        let num_candidates = self.layer_boundaries.len();
        assert!(
            num_boundaries <= num_candidates,
            "cannot create {} partitions with only {} layer boundaries",
            num_partitions,
            num_candidates
        );
        // Pick evenly-spaced indices from 0..num_candidates
        let mut selected = Vec::with_capacity(num_boundaries);
        for i in 0..num_boundaries {
            // Spread boundaries evenly: index = (i+1) * num_candidates / num_partitions - 1
            let idx = ((i + 1) * num_candidates) / num_partitions;
            // Clamp to valid range
            let idx = idx.min(num_candidates - 1);
            selected.push(self.layer_boundaries[idx]);
        }
        self.boundary_edges = selected;
    }

    /// Drop the GPU buffer of every device-resident witness, downloading
    /// evaluations to host first. Idempotent; safe to call when
    /// ZKT_RUN_BACKEND=cpu (witnesses are already host-only).
    ///
    /// Call after `commit()` and before `prove()` when running with GPU
    /// forward passes on large models — the prove machinery reads from host
    /// and re-uploads per einsum task, so freeing here drops 80–100 GB of
    /// device residency for full LLaMA-2-7B with no prove-time cost.
    pub fn evict_device_witnesses(&self, witnesses: &mut [Vec<Witness>]) {
        let mut total_evicted: usize = 0;
        let mut total_elements: usize = 0;
        for ws in witnesses.iter_mut() {
            for w in ws.iter_mut() {
                if w.is_device_resident() {
                    let n = w.data.as_ref().unwrap().n();
                    w.evict_device_buffer();
                    total_evicted += 1;
                    total_elements += 1usize << n;
                }
            }
        }
        if total_evicted > 0 {
            println!(
                "  evict_device_witnesses: {} witnesses, {:.1} M elements freed from GPU",
                total_evicted,
                total_elements as f64 / 1_000_000.0,
            );
        }
    }

    /// Forward pass: compute all witnesses.
    /// Backend is selected by ZKT_RUN_BACKEND env var: "cpu" (default for now)
    /// or "gpu". GPU mode dispatches every node serially through `run_gpu`;
    /// each op falls back to CPU until its kernel is wired.
    pub fn run(&self, witnesses: &mut [Vec<Witness>], feed: &[(EdgeId, Witness)]) {
        assert_eq!(witnesses.len(), self.num_edges);

        for (eid, t) in feed {
            witnesses[*eid] = vec![t.clone()];
        }

        let backend = std::env::var("ZKT_RUN_BACKEND").unwrap_or_else(|_| "cpu".to_string());
        let use_gpu = backend.eq_ignore_ascii_case("gpu");

        // Per-node-type timing accumulators
        let mut type_times: HashMap<String, (std::time::Duration, usize)> = HashMap::new();
        let t_run = std::time::Instant::now();

        if use_gpu {
            // Single-GPU sequential dispatch: each op runs through run_gpu on the
            // current device. We deliberately don't parallelize across nodes within
            // a level — one device can't usefully run two big kernels at once and
            // rayon-launched concurrent kernels just contend for the same SM.
            for level in &self.topo_levels {
                for &nid in level {
                    let node = &self.nodes[nid];
                    let in_refs: Vec<&Witness> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                    log::debug!("running node {} | kind {:?} | backend gpu", nid, node.kind);
                    let t_node = std::time::Instant::now();
                    let outs = node.kind.run_gpu(&in_refs);
                    let elapsed = t_node.elapsed();
                    let type_key = format!("{:?}", node.kind).split('(').next().unwrap_or("Unknown").to_string();
                    let entry = type_times.entry(type_key).or_insert((std::time::Duration::ZERO, 0));
                    entry.0 += elapsed;
                    entry.1 += 1;
                    assert_eq!(outs.len(), node.outputs.len(), "op output arity mismatch");
                    for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                        witnesses[eid] = vec![out];
                    }
                }
            }
        } else {
            // CPU path: rayon-parallelize over independent nodes within a level.
            for level in &self.topo_levels {
                if level.len() == 1 {
                    let nid = level[0];
                    let node = &self.nodes[nid];
                    let in_refs: Vec<&Witness> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                    log::debug!("running node {} | kind {:?}", nid, node.kind);
                    let t_node = std::time::Instant::now();
                    let outs = node.kind.run(&in_refs);
                    let elapsed = t_node.elapsed();
                    let type_key = format!("{:?}", node.kind).split('(').next().unwrap_or("Unknown").to_string();
                    let entry = type_times.entry(type_key).or_insert((std::time::Duration::ZERO, 0));
                    entry.0 += elapsed;
                    entry.1 += 1;
                    assert_eq!(outs.len(), node.outputs.len(), "op output arity mismatch");
                    for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                        witnesses[eid] = vec![out];
                    }
                } else {
                    let results: Vec<(NodeId, Vec<Witness>, std::time::Duration)> = level.par_iter().map(|&nid| {
                        let node = &self.nodes[nid];
                        let in_refs: Vec<&Witness> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                        log::debug!("running node {} | kind {:?}", nid, node.kind);
                        let t_node = std::time::Instant::now();
                        let outs = node.kind.run(&in_refs);
                        let elapsed = t_node.elapsed();
                        (nid, outs, elapsed)
                    }).collect();

                    for (nid, outs, elapsed) in results {
                        let node = &self.nodes[nid];
                        let type_key = format!("{:?}", node.kind).split('(').next().unwrap_or("Unknown").to_string();
                        let entry = type_times.entry(type_key).or_insert((std::time::Duration::ZERO, 0));
                        entry.0 += elapsed;
                        entry.1 += 1;
                        assert_eq!(outs.len(), node.outputs.len(), "op output arity mismatch");
                        for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                            witnesses[eid] = vec![out];
                        }
                    }
                }
            }
        }

        let total = t_run.elapsed();
        println!("=== Run timing by node type (backend={}, total: {:.3}s) ===",
            if use_gpu { "gpu" } else { "cpu" }, total.as_secs_f64());
        let mut type_vec: Vec<_> = type_times.into_iter().collect();
        type_vec.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        for (k, (dur, count)) in &type_vec {
            println!("  {:<20} {:>8.3}s  ({} nodes)", k, dur.as_secs_f64(), count);
        }
    }

    fn should_commit(&self, witness: &Witness, edge_id: EdgeId) -> bool {
        // Force-commit boundary edges (they serve as cryptographic anchors
        // connecting independent partition sub-proofs).
        if self.boundary_edges.contains(&edge_id) {
            return true;
        }
        // Force-commit self-claim edges (Conv2D outputs need PCS openings).
        if self.self_claim_edges.contains(&edge_id) {
            return true;
        }
        match witness.role {
            Role::Constant | Role::Auxiliary | Role::Input => true,
            Role::Output => self.consumers[edge_id].is_empty(),
        }
    }

    /// Minimum num_vars for GPU basefold commit (smaller polys are skipped).
    const MIN_BASEFOLD_VARS: usize = 2;
    /// Maximum num_vars for GPU basefold commit. Set to 24 so that every
    /// auxiliary witness in GPT-2 / Llama-2 (single-layer, single-token)
    /// gets committed and opened — the prior cap of 22 silently dropped
    /// the n=24 FFN-output polys in Llama, leaving those edges unbound
    /// by the PCS. Pair with `GPU_OPEN_POOL_SIZE=1` to fit in GPU memory.
    const MAX_BASEFOLD_VARS: usize = 24;

    /// Commit to all witness polynomials using GPU basefold.
    /// Returns the non-weight commit duration (prover time portion).
    pub fn commit(
        &self,
        key: &BasefoldCommitKey,
        witnesses: &[Vec<Witness>],
        commitments: &mut [Option<BasefoldCommitmentData>],
        gpu_store: &mut GpuCommitmentStore,
        edge_partitions: Option<&[Option<usize>]>,
    ) -> std::time::Duration {
        let t_commit = std::time::Instant::now();
        let num_devices = goldilocks_cuda::device_count().unwrap_or(1).max(1) as usize;

        // Phase 1: Collect edges to commit. For device-resident witnesses we
        // skip the device→host→device round trip and use BasefoldCommitment::
        // commit_device directly.
        enum CommitSource {
            Host(Vec<GoldilocksField>),
            Device(Arc<DeviceBuffer<u64>>),
        }
        struct CommitTask {
            edge_id: usize,
            source: CommitSource,
            num_vars: usize,
            is_weight: bool,
            role: Role,
            producer_type: String,
        }
        let mut tasks: Vec<CommitTask> = Vec::new();
        for (e, witness) in witnesses.iter().enumerate() {
            if !witness.is_empty() {
                for w in witness.iter() {
                    if self.should_commit(w, e) && w.data.is_some() {
                        if commitments[e].is_none() {
                            let num_vars = w.data.as_ref().unwrap().n();
                            if num_vars >= Self::MIN_BASEFOLD_VARS && num_vars <= Self::MAX_BASEFOLD_VARS {
                                let source = if let Some(buf) = w.device_buf() {
                                    CommitSource::Device(Arc::clone(buf))
                                } else {
                                    let evals = match w.data.as_ref().unwrap().try_evaluations_ref() {
                                        Some(r) => r.to_vec(),
                                        None => w.data.as_ref().unwrap().evaluations(),
                                    };
                                    CommitSource::Host(evals)
                                };
                                let is_weight = w.role == Role::Constant;
                                let producer_type = if let Some(pid) = self.producers[e] {
                                    format!("{:?}", self.nodes[pid].kind).split('(').next().unwrap_or("?").to_string()
                                } else {
                                    "input".to_string()
                                };
                                tasks.push(CommitTask { edge_id: e, source, num_vars, is_weight, role: w.role, producer_type });
                            }
                        }
                    }
                }
            }
        }

        // Sort by num_vars descending so largest (slowest) commits start first for better load balance
        tasks.sort_by_key(|t| std::cmp::Reverse(t.num_vars));

        // Separate weight tasks from non-weight tasks for timing
        let weight_tasks: Vec<&CommitTask> = tasks.iter().filter(|t| t.is_weight).collect();
        let nonweight_tasks: Vec<&CommitTask> = tasks.iter().filter(|t| !t.is_weight).collect();
        let weight_elems: usize = weight_tasks.iter().map(|t| 1usize << t.num_vars).sum();
        let nonweight_elems: usize = nonweight_tasks.iter().map(|t| 1usize << t.num_vars).sum();

        // Phase 2: Parallel GPU commit across multiple devices
        let log_rate = key.log_rate;

        type CommitResult = Option<(usize, BasefoldCommitmentData, BasefoldCommitment, i32)>;

        // Commit weights (offline preprocess)
        let t_weight = std::time::Instant::now();
        let weight_results: Vec<CommitResult> = weight_tasks.par_iter().enumerate().map(|(idx, task)| {
            let device = match edge_partitions {
                Some(ep) => (ep[task.edge_id].unwrap_or(idx) % num_devices) as i32,
                None => (idx % num_devices) as i32,
            };
            let _ = goldilocks_cuda::set_device(device);
            let _ = goldilocks_cuda::init_device(); // Ensure Poseidon2 constants loaded on this device
            let result = match &task.source {
                CommitSource::Host(evals) => BasefoldCommitment::commit(evals, task.num_vars, log_rate),
                CommitSource::Device(buf) => {
                    // Single-GPU mode: the device buffer should already live on
                    // the current device. If commit_device fails (e.g., size
                    // mismatch), fall back to a host roundtrip.
                    BasefoldCommitment::commit_device(buf.as_ref(), task.num_vars, log_rate)
                }
            };
            match result {
                Ok(gpu_comm) => {
                    let root = gpu_comm.root;
                    let data = BasefoldCommitmentData { root, num_vars: task.num_vars };
                    Some((task.edge_id, data, gpu_comm, device))
                }
                Err(_) => None,
            }
        }).collect();
        let _ = goldilocks_cuda::set_device(0);
        let _ = goldilocks_cuda::synchronize();
        let weight_time = t_weight.elapsed();

        // Commit non-weights (part of prover time)
        let t_nonweight = std::time::Instant::now();
        let nonweight_results: Vec<CommitResult> = nonweight_tasks.par_iter().enumerate().map(|(idx, task)| {
            let device = match edge_partitions {
                Some(ep) => (ep[task.edge_id].unwrap_or(idx) % num_devices) as i32,
                None => (idx % num_devices) as i32,
            };
            let _ = goldilocks_cuda::set_device(device);
            let _ = goldilocks_cuda::init_device(); // Ensure Poseidon2 constants loaded on this device
            let result = match &task.source {
                CommitSource::Host(evals) => BasefoldCommitment::commit(evals, task.num_vars, log_rate),
                CommitSource::Device(buf) => {
                    // Single-GPU mode: the device buffer should already live on
                    // the current device. If commit_device fails (e.g., size
                    // mismatch), fall back to a host roundtrip.
                    BasefoldCommitment::commit_device(buf.as_ref(), task.num_vars, log_rate)
                }
            };
            match result {
                Ok(gpu_comm) => {
                    let root = gpu_comm.root;
                    let data = BasefoldCommitmentData { root, num_vars: task.num_vars };
                    Some((task.edge_id, data, gpu_comm, device))
                }
                Err(_) => None,
            }
        }).collect();
        let _ = goldilocks_cuda::set_device(0);
        let _ = goldilocks_cuda::synchronize();
        let nonweight_time = t_nonweight.elapsed();

        // Phase 3: Store results.
        let mut commit_count = 0usize;
        let mut total_elems = 0usize;
        let mut var_dist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for result in weight_results.into_iter().chain(nonweight_results.into_iter()) {
            if let Some((edge_id, data, gpu_comm, device)) = result {
                let nv = data.num_vars;
                commitments[edge_id] = Some(data);
                gpu_store.commitments[edge_id] = Some(gpu_comm);
                gpu_store.device_ids[edge_id] = Some(device);
                commit_count += 1;
                total_elems += 1 << nv;
                *var_dist.entry(nv).or_default() += 1;
            }
        }

        let mut sorted_vars: Vec<_> = var_dist.into_iter().collect();
        sorted_vars.sort_by_key(|&(nv, _)| std::cmp::Reverse(nv));
        println!("  Committed {} edges ({:.1}M elements) across {} GPUs in {:.3}s",
            commit_count, total_elems as f64 / 1e6, num_devices, t_commit.elapsed().as_secs_f64());
        println!("    Weight commits: {} edges ({:.1}M elements) in {:.3}s (offline)",
            weight_tasks.len(), weight_elems as f64 / 1e6, weight_time.as_secs_f64());
        println!("    Non-weight commits: {} edges ({:.1}M elements) in {:.3}s (prover time)",
            nonweight_tasks.len(), nonweight_elems as f64 / 1e6, nonweight_time.as_secs_f64());
        for (nv, count) in &sorted_vars {
            println!("    n={:>2}: {} edges ({:.1}M elements each)", nv, count, (1usize << nv) as f64 / 1e6);
        }

        // Detailed breakdown by role and producer type
        let mut role_stats: HashMap<String, (usize, usize)> = HashMap::new(); // (count, total_elements)
        let mut producer_stats: HashMap<String, (usize, usize)> = HashMap::new();
        let mut nonweight_producer_stats: HashMap<String, (usize, usize)> = HashMap::new();
        for t in &tasks {
            let role_key = format!("{:?}", t.role);
            let re = role_stats.entry(role_key).or_default();
            re.0 += 1;
            re.1 += 1usize << t.num_vars;

            let pe = producer_stats.entry(t.producer_type.clone()).or_default();
            pe.0 += 1;
            pe.1 += 1usize << t.num_vars;

            if !t.is_weight {
                let npe = nonweight_producer_stats.entry(format!("{} ({:?})", t.producer_type, t.role)).or_default();
                npe.0 += 1;
                npe.1 += 1usize << t.num_vars;
            }
        }
        println!("  === Commit breakdown by role ===");
        let mut rv: Vec<_> = role_stats.into_iter().collect();
        rv.sort_by_key(|x| std::cmp::Reverse(x.1.1));
        for (k, (cnt, elems)) in &rv {
            println!("    {:<12} {:>4} edges  {:>8.1}M elements", k, cnt, *elems as f64 / 1e6);
        }
        println!("  === Non-weight commits by producer + role ===");
        let mut npv: Vec<_> = nonweight_producer_stats.into_iter().collect();
        npv.sort_by_key(|x| std::cmp::Reverse(x.1.1));
        for (k, (cnt, elems)) in &npv {
            println!("    {:<35} {:>4} edges  {:>8.1}M elements", k, cnt, *elems as f64 / 1e6);
        }
        nonweight_time
    }

    /// Prove: backward pass producing sumcheck proofs and opening proofs.
    pub fn prove(
        &self,
        key: &BasefoldCommitKey,
        witnesses: &mut [Vec<Witness>],
        commitments: &[Option<BasefoldCommitmentData>],
        gpu_store: &GpuCommitmentStore,
        table: &BasefoldTable,
        transcript: &mut Transcript,
        timing: &mut TimingTree,
    ) -> (
        Vec<Option<(Vec<SumcheckProof>, Vec<Claim>)>>,
        Vec<EdgeProof>,
        LookupProof,
        LookupProof,
        Vec<Option<Vec<SumcheckProof>>>,
    ) {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut node_proofs: Vec<Option<(Vec<SumcheckProof>, Vec<Claim>)>> = vec![None; self.nodes.len()];
        let mut reducer_proofs: Vec<Option<Vec<SumcheckProof>>> = vec![None; self.nodes.len()];
        let mut edge_proofs: Vec<EdgeProof> = (0..self.num_edges).map(|_| EdgeProof::new()).collect();

        // 1. Open final outputs at random points
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut nodes_to_prove = BTreeSet::new();

        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role == Role::Output {
                let n = w.data.as_ref().unwrap().n();
                let point: Vec<GoldilocksExt2> = (0..n).map(|_| transcript.challenge_ext2(b"challenge")).collect();
                let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
                claims[e].push(Claim {
                    edge_id: e,
                    sparse_id: 0,
                    point,
                    eval,
                });
                if let Some(producer) = self.producers[e] {
                    nodes_to_prove.insert(producer);
                }
            }
        }

        // 2. Prove from outputs to inputs
        let t_backward = std::time::Instant::now();
        let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|c| c.iter().copied().collect()).collect();
        let mut type_times: HashMap<String, (std::time::Duration, usize)> = HashMap::new();
        let mut reducer_total = std::time::Duration::ZERO;

        while !nodes_to_prove.is_empty() {
            let node_id = *nodes_to_prove.iter().next_back().unwrap();
            nodes_to_prove.remove(&node_id);

            // Check if all consumers are done
            let mut can_prove = true;
            for &edge in &self.nodes[node_id].outputs {
                if consumer_sets[edge].iter().any(|c| nodes_to_prove.contains(c)) {
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

            // Reducer if multiple claims
            #[allow(unused_assignments)]
            let mut reduced_claims_storage: Vec<Claim> = Vec::new();
            if local_claims.len() > 1 {
                let t_red = std::time::Instant::now();
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                let reducer_edge_ids = vec![*edge_ids.last().unwrap()];
                let (proofs, rc) = reducer.prove(&reducer_witness, &reducer_edge_ids, &local_claims, transcript);
                reducer_total += t_red.elapsed();
                if !rc.is_empty() {
                    claims[node.outputs[0]].push(rc[0].clone());
                    reducer_proofs[node_id] = Some(proofs);
                    reduced_claims_storage = rc;
                    local_claims = reduced_claims_storage.iter().collect();
                }
            }

            // Prove the node
            log::debug!("proving node {} | kind {:?}", node_id, node.kind);
            let t_node = std::time::Instant::now();
            let (proofs, new_claims) = timed!(
                timing,
                format!("prove node {} | kind {:?}", node_id, node.kind).as_str(),
                {
                    node.kind.prove(&local_witnesses, &edge_ids, &local_claims, transcript)
                }
            );
            let elapsed = t_node.elapsed();
            let type_key = format!("{:?}", node.kind).split('(').next().unwrap_or("Unknown").to_string();
            let entry = type_times.entry(type_key).or_insert((std::time::Duration::ZERO, 0));
            entry.0 += elapsed;
            entry.1 += 1;

            let mut node_claims: Vec<Claim> = new_claims.clone();
            for c in &local_claims {
                node_claims.push((*c).clone());
            }
            node_proofs[node_id] = Some((proofs, node_claims));

            let mut claimed_edge_ids: Vec<usize> = Vec::new();
            for c in new_claims {
                if !claimed_edge_ids.contains(&c.edge_id) {
                    claimed_edge_ids.push(c.edge_id);
                }
                claims[c.edge_id].push(c);
            }

            for e in claimed_edge_ids {
                if let Some(producer) = self.producers[e] {
                    if producer != node_id {  // Don't re-add self for self-claims (Conv2D)
                        nodes_to_prove.insert(producer);
                    }
                }
            }
        }

        let backward_total = t_backward.elapsed();
        // Print timing summary by node type
        println!("=== Prove timing by node type (backward pass: {:.3}s) ===", backward_total.as_secs_f64());
        let mut type_vec: Vec<_> = type_times.into_iter().collect();
        type_vec.sort_by(|a, b| b.1.0.cmp(&a.1.0));
        for (k, (dur, count)) in &type_vec {
            println!("  {:<20} {:>8.3}s  ({} nodes)", k, dur.as_secs_f64(), count);
        }
        println!("  {:<20} {:>8.3}s", "Reducer (total)", reducer_total.as_secs_f64());

        // Skip freeing intermediate edges — deallocation is expensive (~13s for large models)

        // 3. Prove lookups
        let t_lookup = std::time::Instant::now();
        let two_pow_proof = self.prove_two_pow(witnesses, &mut claims, transcript);
        let range_proof = self.prove_range(witnesses, &mut claims, transcript);
        println!("  Lookup proofs: {:.3}s", t_lookup.elapsed().as_secs_f64());

        // Copy ALL claims (including lookup claims) to edge proofs
        for (e, c) in claims.iter().enumerate() {
            edge_proofs[e].claims.extend(c.iter().cloned());
        }

        // 4. Opening reducers: combine K claims per edge into 1
        let t_reduce = std::time::Instant::now();
        let mut opening_reduced_count = 0usize;
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if w.data.is_none() { continue; }
            let non_empty: Vec<usize> = edge_proofs[e].claims.iter().enumerate()
                .filter(|(_, c)| !c.point.is_empty()).map(|(i, _)| i).collect();
            if non_empty.len() <= 1 { continue; }
            // CPU-only reducer to avoid GPU OOM after commit phase
            let alpha: GoldilocksExt2 = transcript.challenge_ext2(b"reducer_alpha");
            let alphas = calc_pow_vec_ext2(alpha, non_empty.len());
            let n = w.data.as_ref().unwrap().n();
            let size = 1usize << n;
            let mut eq_combined = vec![GoldilocksExt2::zero(); size];
            for (idx, &ci) in non_empty.iter().enumerate() {
                let eq_table = evaluate_lagrange_basis_ext2(&edge_proofs[e].claims[ci].point);
                for j in 0..size {
                    eq_combined[j] = ext2_add(eq_combined[j], ext2_mul(alphas[idx], eq_table[j]));
                }
            }
            let x_evals = w.data.as_ref().unwrap().evaluations_ref();
            let x_ext2: Vec<GoldilocksExt2> = x_evals.iter()
                .map(|&v| GoldilocksExt2::from_base(v)).collect();
            let mut cpu_prover = CpuLinearSumcheckProverExt2::new(n, 2, transcript);
            let mut polys = vec![x_ext2, eq_combined];
            let sumcheck_proof = cpu_prover.prove(&mut polys, transcript);
            let combined_claim = Claim {
                edge_id: e, sparse_id: 0,
                point: cpu_prover.challenges.clone(),
                eval: cpu_prover.final_eval(0),
            };
            edge_proofs[e].opening_reducer = Some(vec![sumcheck_proof]);
            edge_proofs[e].claims.push(combined_claim);
            opening_reduced_count += non_empty.len() - 1;
        }
        if opening_reduced_count > 0 {
            println!("  Opening reducers: {:.3}s ({} proofs saved)", t_reduce.elapsed().as_secs_f64(), opening_reduced_count);
        }

        // 5. Collect opening tasks: 1 per committed edge (last non-empty claim)
        let master_seed = transcript.challenge_ext2(b"opening_seed");

        let mut tasks: Vec<(usize, usize, Vec<GoldilocksExt2>)> = Vec::new();
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if let Some((i, claim)) = edge_proofs[e].claims.iter().enumerate().rev()
                .find(|(_, c)| !c.point.is_empty()) {
                tasks.push((e, i, claim.point.clone()));
            }
        }

        println!("Opening tasks: {} (1 per edge)", tasks.len());

        // Print opening task statistics by num_vars
        let mut var_counts: HashMap<usize, usize> = HashMap::new();
        for (e, _, _) in &tasks {
            let num_vars = witnesses[*e][0].data.as_ref().unwrap().n();
            *var_counts.entry(num_vars).or_default() += 1;
        }
        let mut sorted: Vec<_> = var_counts.into_iter().collect();
        sorted.sort_by_key(|&(nv, _)| std::cmp::Reverse(nv));
        for (nv, count) in &sorted { println!("  n={:>2}: {} tasks", nv, count); }

        // Split tasks into CPU (small) and GPU (large) buckets
        let cpu_open_threshold: usize = std::env::var("CPU_OPEN_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(14);
        let num_devices = goldilocks_cuda::device_count().unwrap_or(1).max(1) as usize;
        let num_queries = key.num_queries;

        // Check if we need re-upload path (host_caches populated, GPU commitments freed)
        let use_reupload = gpu_store.host_caches.iter().any(|c| c.is_some());

        let open_start = std::time::Instant::now();
        let proof_map: HashMap<usize, BasefoldOpeningProof>;

        if use_reupload {
            // Re-upload path: group tasks by edge, re-upload each edge to GPU,
            // open its task, then drop the commitment (freeing GPU memory).
            //
            // Per-device serial scheme: one OS thread per device, opens its
            // assigned edges sequentially largest-arity-first. This avoids
            // OOM at high arities (n ≥ 24) — a single n=24 open transiently
            // needs ~4 GB on its device (1 GB commit + 1 GB table + ~2 GB
            // working). Putting two such opens on the same GPU blows the
            // budget; per-device serialization guarantees only one is in
            // flight per GPU. Across `num_devices` devices the scheme still
            // gets `num_devices`-way parallelism.
            println!("  Opening (re-upload mode, per-device serial): {} tasks, {} devices, {} queries",
                tasks.len(), num_devices, num_queries);
            let mut edge_tasks: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, (e, _, _)) in tasks.iter().enumerate() {
                edge_tasks.entry(*e).or_default().push(idx);
            }
            // Group edges by device, then sort each device's edge list by
            // descending arity so the largest opens happen first (avoids
            // fragmentation pile-up from many small opens).
            let mut per_device_edges: Vec<Vec<(usize, Vec<usize>)>> =
                (0..num_devices).map(|_| Vec::new()).collect();
            for (edge_id, task_indices) in edge_tasks.into_iter() {
                let dev = (gpu_store.device_ids[edge_id].unwrap_or(0) as usize) % num_devices;
                per_device_edges[dev].push((edge_id, task_indices));
            }
            for queue in per_device_edges.iter_mut() {
                queue.sort_by_key(|(e, _)| {
                    std::cmp::Reverse(witnesses[*e][0].data.as_ref().unwrap().n())
                });
            }

            let tasks_ref = &tasks;
            let host_caches = &gpu_store.host_caches;
            let per_device_tables = &gpu_store.per_device_tables;
            let edge_proofs_ref = &edge_proofs;

            let results: Vec<(usize, BasefoldOpeningProof)> = std::thread::scope(|s| {
                let handles: Vec<_> = per_device_edges.into_iter().enumerate().map(|(dev, edges)| {
                    s.spawn(move || {
                        let _ = goldilocks_cuda::set_device(dev as i32);
                        let dev_table = &per_device_tables[dev];
                        let mut local: Vec<(usize, BasefoldOpeningProof)> = Vec::new();
                        for (edge_id, task_indices) in edges {
                            let cache = host_caches[edge_id].as_ref()
                                .expect("host cache missing for re-upload opening");
                            let commitment = cache.to_device()
                                .expect("re-upload to_device failed");
                            for task_idx in task_indices {
                                let (_, claim_idx, point) = &tasks_ref[task_idx];
                                let mut t = Transcript::new(b"bf-open");
                                t.append_ext2(b"", &master_seed);
                                t.append_u64(b"", task_idx as u64);
                                let gpu_proof = commitment.open_ext2(point, dev_table, &mut t, num_queries)
                                    .expect("GPU open_ext2 failed (re-upload)");
                                let eval = edge_proofs_ref[edge_id].claims[*claim_idx].eval;
                                local.push((task_idx, BasefoldOpeningProof { eval, gpu_proof }));
                            }
                            // `commitment` drops here → GPU memory freed before next edge.
                        }
                        local
                    })
                }).collect();
                handles.into_iter().flat_map(|h| h.join().expect("device thread panicked")).collect()
            });

            let _ = goldilocks_cuda::set_device(0);
            println!("  Openings (re-upload, per-device serial): {:.3}s ({} tasks, {} devices)",
                open_start.elapsed().as_secs_f64(), results.len(), num_devices);

            proof_map = results.into_iter().collect();
        } else {
            // CPU+GPU dual-pool path (GPU commitments still alive)
            let mut gpu_tasks: Vec<usize> = Vec::new();
            let mut cpu_tasks: Vec<usize> = Vec::new();
            for (idx, (e, _, _)) in tasks.iter().enumerate() {
                let nv = witnesses[*e][0].data.as_ref().unwrap().n();
                if nv <= cpu_open_threshold {
                    cpu_tasks.push(idx);
                } else {
                    gpu_tasks.push(idx);
                }
            }
            println!("  Opening: {} GPU (n>{}), {} CPU (n≤{}), {} devices, {} queries",
                gpu_tasks.len(), cpu_open_threshold, cpu_tasks.len(), cpu_open_threshold, num_devices, num_queries);

            let gpu_commitments = &gpu_store.commitments;
            let device_ids = &gpu_store.device_ids;
            let per_device_tables = &gpu_store.per_device_tables;

            // Sort GPU tasks largest-first for better load balancing
            gpu_tasks.sort_by_key(|&idx| {
                let (e, _, _) = &tasks[idx];
                std::cmp::Reverse(witnesses[*e][0].data.as_ref().unwrap().n())
            });

            let tasks_ref = &tasks;
            let num_cores = num_cpus::get();
            let gpu_threads_per_device: usize = std::env::var("GPU_OPEN_THREADS_PER_DEVICE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(12);
            // `GPU_OPEN_POOL_SIZE` overrides the multiplicative pool sizing.
            // Useful when high-arity opens (n ≥ 24) need serial execution
            // to fit in GPU memory.
            let gpu_pool_size = std::env::var("GPU_OPEN_POOL_SIZE")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or((num_devices * gpu_threads_per_device).max(1));
            let cpu_pool_size = (num_cores - gpu_pool_size).max(1);
            let cpu_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(cpu_pool_size)
                .build().expect("failed to build CPU rayon pool");

            let (cpu_results, gpu_results) = std::thread::scope(|s| {
                let cpu_handle = s.spawn(|| {
                    let t0 = std::time::Instant::now();
                    let results: Vec<(usize, BasefoldOpeningProof)> = cpu_pool.install(|| {
                        cpu_tasks.par_iter().map(|&task_idx| {
                            let (e, _, point) = &tasks_ref[task_idx];
                            let commitment = gpu_commitments[*e].as_ref()
                                .expect("GPU commitment missing for CPU opening");
                            let mut t = Transcript::new(b"bf-open");
                            t.append_ext2(b"", &master_seed);
                            t.append_u64(b"", task_idx as u64);
                            let proof = cpu_full_open_ext2(commitment, point, table, &mut t, num_queries);
                            (task_idx, proof)
                        }).collect()
                    });
                    (results, t0.elapsed())
                });
                let gpu_handle = s.spawn(|| {
                    if gpu_tasks.is_empty() {
                        return (vec![], std::time::Duration::ZERO);
                    }
                    let t0 = std::time::Instant::now();
                    let gpu_pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(gpu_pool_size)
                        .build().expect("failed to build GPU rayon pool");
                    let results: Vec<(usize, BasefoldOpeningProof)> = gpu_pool.install(|| {
                        gpu_tasks.par_iter().enumerate().map(|(_gpu_idx, &task_idx)| {
                            let (e, _, point) = &tasks_ref[task_idx];
                            let dev = device_ids[*e].unwrap_or(0);
                            let _ = goldilocks_cuda::set_device(dev);
                            let dev_table = &per_device_tables[dev as usize % num_devices];
                            let commitment = gpu_commitments[*e].as_ref()
                                .expect("GPU commitment missing for GPU opening");
                            let mut t = Transcript::new(b"bf-open");
                            t.append_ext2(b"", &master_seed);
                            t.append_u64(b"", task_idx as u64);
                            let proof = commitment.open_ext2(point, dev_table, &mut t, num_queries)
                                .expect("GPU open_ext2 failed");
                            let eval = edge_proofs[*e].claims[tasks_ref[task_idx].1].eval;
                            (task_idx, BasefoldOpeningProof { eval, gpu_proof: proof })
                        }).collect()
                    });
                    (results, t0.elapsed())
                });
                let (cpu_res, cpu_time) = cpu_handle.join().unwrap();
                let (gpu_res, gpu_time) = gpu_handle.join().unwrap();
                let _ = goldilocks_cuda::set_device(0);
                println!("  Openings: CPU {:.3}s ({} tasks, {} threads), GPU {:.3}s ({} tasks, {} threads), wall {:.3}s",
                    cpu_time.as_secs_f64(), cpu_res.len(), cpu_pool_size,
                    gpu_time.as_secs_f64(), gpu_res.len(), gpu_pool_size,
                    open_start.elapsed().as_secs_f64());
                (cpu_res, gpu_res)
            });

            let mut map: HashMap<usize, BasefoldOpeningProof> = HashMap::new();
            map.extend(cpu_results);
            map.extend(gpu_results);
            proof_map = map;
        }

        // Distribute results to edge_proofs (1 proof per edge)
        for (task_idx, (e, _, _)) in tasks.iter().enumerate() {
            if let Some(proof) = proof_map.get(&task_idx) {
                edge_proofs[*e].dense_opening_proof.push(proof.clone());
            }
        }
        (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs)
    }

    /// Verify: check all proofs.
    /// Mirrors the prove backward pass exactly to keep transcripts in sync.
    pub fn verify(
        &self,
        node_proofs: &[Option<(Vec<SumcheckProof>, Vec<Claim>)>],
        edge_proofs: &[EdgeProof],
        range_proof: &LookupProof,
        two_pow_proof: &LookupProof,
        reducer_proofs: &[Option<Vec<SumcheckProof>>],
        witnesses: &[Vec<Witness>],
        _verifier_key: &BasefoldVerifierKey,
        commitments: &[Option<BasefoldCommitmentData>],
        table: &BasefoldTable,
        transcript: &mut Transcript,
    ) -> bool {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut verified = true;

        // 1. Re-derive output challenges (must match prover transcript)
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut nodes_to_verify = BTreeSet::new();

        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role == Role::Output {
                let n = edge_proofs[e].claims[0].point.len();
                let point: Vec<GoldilocksExt2> = (0..n)
                    .map(|_| transcript.challenge_ext2(b"challenge"))
                    .collect();
                verified = verified && edge_proofs[e].claims[0].point == point;
                claims[e].push(edge_proofs[e].claims[0].clone());
                if let Some(producer) = self.producers[e] {
                    nodes_to_verify.insert(producer);
                }
            }
        }

        // 2. Verify backward pass mirroring prover
        let consumer_sets: Vec<BTreeSet<NodeId>> = self
            .consumers
            .iter()
            .map(|c| c.iter().copied().collect())
            .collect();

        while !nodes_to_verify.is_empty() {
            let u = *nodes_to_verify.iter().next_back().unwrap();
            nodes_to_verify.remove(&u);

            let mut can_verify = true;
            for &edge in &self.nodes[u].outputs {
                if consumer_sets[edge].iter().any(|c| nodes_to_verify.contains(c)) {
                    can_verify = false;
                    break;
                }
            }
            if !can_verify {
                nodes_to_verify.insert(u);
                continue;
            }

            let node = &self.nodes[u];
            let (local_sumcheck_proofs, local_claims) = node_proofs[u].as_ref().unwrap();
            let local_claims_ref: Vec<&Claim> = local_claims.iter().collect();
            let local_witnesses: Vec<&Witness> = local_claims_ref
                .iter()
                .map(|c| &witnesses[c.edge_id][0])
                .collect();
            let local_sumcheck_proofs_ref: Vec<&SumcheckProof> =
                local_sumcheck_proofs.iter().collect();

            // Verify reducer if multiple claims on this output
            if claims[node.outputs[0]].len() > 1 {
                assert!(
                    reducer_proofs[u].is_some(),
                    "reducer proof is not found for node {u}"
                );
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                let mut reducer_claims = claims[node.outputs[0]].clone();
                reducer_claims.push(local_claims_ref[local_claims_ref.len() - 1].clone());
                let reducer_claims_ref: Vec<&Claim> = reducer_claims.iter().collect();
                let reducer_sc_proofs: Vec<&SumcheckProof> =
                    reducer_proofs[u].as_ref().unwrap().iter().collect();
                let reducer_verified = reducer.verify(
                    &reducer_witness,
                    &reducer_claims_ref,
                    &reducer_sc_proofs,
                    transcript,
                );
                verified = verified && reducer_verified;
                if !reducer_verified {
                    println!("verified reducer for node {u} failed");
                }
                // Push reduced claim to match prover's claims state
                claims[node.outputs[0]].push(local_claims_ref[local_claims_ref.len() - 1].clone());
            }

            // Verify the node
            let node_verified = self.nodes[u].kind.verify(
                &local_witnesses,
                &local_claims_ref,
                &local_sumcheck_proofs_ref,
                transcript,
            );
            if !node_verified {
                println!("verified node {u} failed (inputs: {:?}, outputs: {:?})",
                    node.inputs, node.outputs);
            }
            verified = verified && node_verified;

            // Propagate input claims — only add producers for edges that receive claims
            let mut claimed_edge_ids: Vec<usize> = Vec::new();
            for c in &local_claims_ref {
                if node.inputs.contains(&c.edge_id) {
                    claims[c.edge_id].push((*c).clone());
                    if !claimed_edge_ids.contains(&c.edge_id) {
                        claimed_edge_ids.push(c.edge_id);
                    }
                } else if self.self_claim_edges.contains(&c.edge_id) && node.outputs.contains(&c.edge_id) {
                    // Self-claim: Conv2D outputs a claim on its own output edge
                    claims[c.edge_id].push((*c).clone());
                }
            }

            for e in claimed_edge_ids {
                if let Some(producer) = self.producers[e] {
                    nodes_to_verify.insert(producer);
                }
            }
        }

        // 3. Verify lookup proofs (must match prover transcript ordering)
        let two_pow_verified = self.verify_two_pow(node_proofs, two_pow_proof, transcript);
        if !two_pow_verified {
            println!("two_pow verification failed");
            return false;
        }
        let range_verified = self.verify_range(node_proofs, witnesses, &claims, range_proof, transcript);
        if !range_verified {
            println!("range verification failed");
            return false;
        }

        // 4. Verify opening reducers (must match prover transcript)
        let reducer_for_opening = BasicBlockType::Reducer(Reducer {});
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if edge_proofs[e].opening_reducer.is_none() { continue; }
            // Original claims from backward pass (non-empty points only)
            let non_empty: Vec<&Claim> = edge_proofs[e].claims.iter()
                .rev().skip(1).collect::<Vec<_>>().into_iter().rev()  // all but last
                .filter(|c| !c.point.is_empty()).collect();
            // Combined claim is the last element
            let combined = edge_proofs[e].claims.last().unwrap();
            let mut all_claims: Vec<&Claim> = non_empty;
            all_claims.push(combined);
            let reducer_witness = vec![&witnesses[e][0]];
            let sc_proofs: Vec<&SumcheckProof> =
                edge_proofs[e].opening_reducer.as_ref().unwrap().iter().collect();
            let rv = reducer_for_opening.verify(&reducer_witness, &all_claims, &sc_proofs, transcript);
            if !rv {
                println!("Opening reducer verification failed at edge {}", e);
            }
            verified = verified && rv;
        }

        // 5. Opening proof verification
        let master_seed = transcript.challenge_ext2(b"opening_seed");

        // Collect tasks: 1 per committed edge (last non-empty claim)
        let mut tasks: Vec<(usize, usize)> = Vec::new();  // (edge_id, claim_idx)
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if let Some((i, _)) = edge_proofs[e].claims.iter().enumerate().rev()
                .find(|(_, c)| !c.point.is_empty()) {
                tasks.push((e, i));
            }
        }

        // Verify opening proofs (1 per edge, parallelized)
        let all_verified: Vec<bool> = tasks.par_iter().enumerate().map(|(task_idx, (e, i))| {
            let mut t = Transcript::new(b"bf-open");
            t.append_ext2(b"", &master_seed);
            t.append_u64(b"", task_idx as u64);

            let proof = &edge_proofs[*e].dense_opening_proof[0]; // 1 proof per edge
            let root = &commitments[*e].as_ref().unwrap().root;
            let point = &edge_proofs[*e].claims[*i].point;

            let ok = BasefoldVerifier::verify_ext2(root, point, &proof.gpu_proof, table, &mut t)
                .unwrap_or(false);
            if !ok {
                println!("Opening proof verification failed at edge {} claim {}", e, i);
            }
            let eval_ok = ext2_field_eq(proof.gpu_proof.eval, edge_proofs[*e].claims[*i].eval);
            if !eval_ok {
                println!("Opening eval mismatch at edge {} claim {}: proof.eval={:?} claim.eval={:?}",
                    e, i, proof.gpu_proof.eval, edge_proofs[*e].claims[*i].eval);
            }
            ok && eval_ok
        }).collect();
        if !all_verified.iter().all(|&v| v) {
            return false;
        }

        verified
    }

    /// Per-partition backward pass (private helper for prove_parallel).
    fn prove_partition(
        &self,
        partition: &PartitionDesc,
        witnesses: &[Vec<Witness>],
        starting_claims: Vec<(EdgeId, Claim)>,
        transcript: &mut Transcript,
    ) -> (PartitionProof, Vec<Vec<Claim>>) {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut node_proofs: Vec<Option<(Vec<SumcheckProof>, Vec<Claim>)>> = vec![None; self.nodes.len()];
        let mut reducer_proofs: Vec<Option<Vec<SumcheckProof>>> = vec![None; self.nodes.len()];

        let partition_nodes: HashSet<NodeId> = partition.node_ids.iter().copied().collect();
        let boundary_inputs: HashSet<EdgeId> = partition.boundary_input_edges.iter().copied().collect();

        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut nodes_to_prove = BTreeSet::new();

        for (e, claim) in starting_claims {
            claims[e].push(claim);
            if let Some(producer) = self.producers[e] {
                if partition_nodes.contains(&producer) {
                    nodes_to_prove.insert(producer);
                }
            }
        }

        // Consumer sets filtered to partition nodes
        let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|c| {
            c.iter().copied().filter(|n| partition_nodes.contains(n)).collect()
        }).collect();

        while !nodes_to_prove.is_empty() {
            let node_id = *nodes_to_prove.iter().next_back().unwrap();
            nodes_to_prove.remove(&node_id);

            // Check if all consumers within partition are done
            let mut can_prove = true;
            for &edge in &self.nodes[node_id].outputs {
                if consumer_sets[edge].iter().any(|c| nodes_to_prove.contains(c)) {
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

            // Reducer if multiple claims
            #[allow(unused_assignments)]
            let mut reduced_claims_storage: Vec<Claim> = Vec::new();
            if local_claims.len() > 1 {
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                let reducer_edge_ids = vec![*edge_ids.last().unwrap()];
                let (proofs, rc) = reducer.prove(&reducer_witness, &reducer_edge_ids, &local_claims, transcript);
                if !rc.is_empty() {
                    claims[node.outputs[0]].push(rc[0].clone());
                    reducer_proofs[node_id] = Some(proofs);
                    reduced_claims_storage = rc;
                    local_claims = reduced_claims_storage.iter().collect();
                }
            }

            // Prove the node
            let (proofs, new_claims) = node.kind.prove(&local_witnesses, &edge_ids, &local_claims, transcript);

            let mut node_claims: Vec<Claim> = new_claims.clone();
            for c in &local_claims {
                node_claims.push((*c).clone());
            }
            node_proofs[node_id] = Some((proofs, node_claims));

            let mut claimed_edge_ids: Vec<usize> = Vec::new();
            for c in new_claims {
                if !claimed_edge_ids.contains(&c.edge_id) {
                    claimed_edge_ids.push(c.edge_id);
                }
                claims[c.edge_id].push(c);
            }

            for e in claimed_edge_ids {
                if boundary_inputs.contains(&e) { continue; } // STOP at boundary
                if let Some(producer) = self.producers[e] {
                    if producer != node_id && partition_nodes.contains(&producer) {
                        nodes_to_prove.insert(producer);
                    }
                }
            }
        }

        (PartitionProof { node_proofs, reducer_proofs }, claims)
    }

    /// Parallel prove: partition the backward pass across partitions.
    pub fn prove_parallel(
        &self,
        key: &BasefoldCommitKey,
        witnesses: &mut [Vec<Witness>],
        commitments: &[Option<BasefoldCommitmentData>],
        gpu_store: &GpuCommitmentStore,
        table: &BasefoldTable,
        transcript: &mut Transcript,
        partitions: &[PartitionDesc],
        _timing: &mut TimingTree,
    ) -> ParallelProof {
        let t_total = std::time::Instant::now();
        let mut edge_proofs: Vec<EdgeProof> = (0..self.num_edges).map(|_| EdgeProof::new()).collect();

        // Step 1: Output claims (same as prove())
        let mut initial_claims: Vec<(EdgeId, Claim)> = Vec::new();
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role == Role::Output {
                let n = w.data.as_ref().unwrap().n();
                let point: Vec<GoldilocksExt2> = (0..n).map(|_| transcript.challenge_ext2(b"challenge")).collect();
                let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
                initial_claims.push((e, Claim { edge_id: e, sparse_id: 0, point, eval }));
            }
        }

        // Step 2: Boundary claims
        let mut boundary_evals: Vec<(EdgeId, Vec<GoldilocksExt2>, GoldilocksExt2)> = Vec::new();
        for &b in &self.boundary_edges {
            let w = &witnesses[b][0];
            let n = w.data.as_ref().unwrap().n();
            let point: Vec<GoldilocksExt2> = (0..n).map(|_| transcript.challenge_ext2(b"challenge")).collect();
            let eval = w.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
            boundary_evals.push((b, point.clone(), eval));
            initial_claims.push((b, Claim { edge_id: b, sparse_id: 0, point, eval }));
        }
        println!("  Step 1+2 (output+boundary claims): {:.3}s", t_total.elapsed().as_secs_f64());

        // Step 3: Route claims to partitions (by producer node's partition)
        let mut partition_starting_claims: Vec<Vec<(EdgeId, Claim)>> = vec![Vec::new(); partitions.len()];
        let partition_node_sets: Vec<HashSet<NodeId>> = partitions.iter()
            .map(|p| p.node_ids.iter().copied().collect())
            .collect();
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

        // Step 4: Fork transcript & prove partitions in parallel (multi-GPU)
        let num_devices = goldilocks_cuda::device_count().unwrap_or(1).max(1) as usize;
        println!("  Multi-GPU: {} devices available, {} partitions", num_devices, partitions.len());
        let t_partition = std::time::Instant::now();
        let transcript_ref: &Transcript = &*transcript;
        let results: Vec<(PartitionProof, Vec<Vec<Claim>>)> = partition_starting_claims
            .into_par_iter()
            .enumerate()
            .map(|(k, starting_claims)| {
                // Assign this partition to a GPU (round-robin)
                let device = (k % num_devices) as i32;
                let _ = goldilocks_cuda::set_device(device);
                let _ = goldilocks_cuda::synchronize();
                goldilocks_cuda::get_last_error(); // Clear any stale per-thread stream errors
                let mut t_k = transcript_ref.fork(k);
                self.prove_partition(&partitions[k], witnesses, starting_claims, &mut t_k)
            })
            .collect();
        // Reset to device 0 for opening proofs
        let _ = goldilocks_cuda::set_device(0);
        println!("  Partition proving: {:.3}s", t_partition.elapsed().as_secs_f64());

        // Step 5: Merge claims from all partitions
        let t_merge = std::time::Instant::now();
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut partition_proofs = Vec::new();
        for (proof, partition_claims) in results {
            for (e, c) in partition_claims.iter().enumerate() {
                claims[e].extend(c.iter().cloned());
            }
            partition_proofs.push(proof);
        }
        println!("  Step 5 (merge claims): {:.3}s", t_merge.elapsed().as_secs_f64());

        // Skip freeing intermediate edges — deallocation takes ~13s for large models
        // and contends with opening proofs for memory bandwidth. With 905GB RAM available,
        // keeping the data in memory is better than paying the deallocation cost.

        // Step 6: Lookup proofs (on main transcript)
        let t_lookup = std::time::Instant::now();
        let two_pow_proof = self.prove_two_pow(witnesses, &mut claims, transcript);
        let t_two_pow = t_lookup.elapsed();
        let range_proof = self.prove_range(witnesses, &mut claims, transcript);
        let t_range = t_lookup.elapsed() - t_two_pow;
        println!("  Step 6 (lookup proofs): {:.3}s (two_pow: {:.3}s [{} nodes], range: {:.3}s [{} nodes])",
            t_lookup.elapsed().as_secs_f64(), t_two_pow.as_secs_f64(), self.two_pow.len(),
            t_range.as_secs_f64(), self.range.len());

        // Copy ALL claims to edge proofs
        for (e, c) in claims.iter().enumerate() {
            edge_proofs[e].claims.extend(c.iter().cloned());
        }

        // Step 7: Opening reducers + opening proofs (same logic as prove())
        let t_reduce = std::time::Instant::now();
        let mut opening_reduced_count = 0usize;
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if w.data.is_none() { continue; }
            let non_empty: Vec<usize> = edge_proofs[e].claims.iter().enumerate()
                .filter(|(_, c)| !c.point.is_empty()).map(|(i, _)| i).collect();
            if non_empty.len() <= 1 { continue; }
            // CPU-only reducer to avoid GPU OOM after commit phase
            let alpha: GoldilocksExt2 = transcript.challenge_ext2(b"reducer_alpha");
            let alphas = calc_pow_vec_ext2(alpha, non_empty.len());
            let n = w.data.as_ref().unwrap().n();
            let size = 1usize << n;
            let mut eq_combined = vec![GoldilocksExt2::zero(); size];
            for (idx, &ci) in non_empty.iter().enumerate() {
                let eq_table = evaluate_lagrange_basis_ext2(&edge_proofs[e].claims[ci].point);
                for j in 0..size {
                    eq_combined[j] = ext2_add(eq_combined[j], ext2_mul(alphas[idx], eq_table[j]));
                }
            }
            let x_evals = w.data.as_ref().unwrap().evaluations_ref();
            let x_ext2: Vec<GoldilocksExt2> = x_evals.iter()
                .map(|&v| GoldilocksExt2::from_base(v)).collect();
            let mut cpu_prover = CpuLinearSumcheckProverExt2::new(n, 2, transcript);
            let mut polys = vec![x_ext2, eq_combined];
            let sumcheck_proof = cpu_prover.prove(&mut polys, transcript);
            let combined_claim = Claim {
                edge_id: e, sparse_id: 0,
                point: cpu_prover.challenges.clone(),
                eval: cpu_prover.final_eval(0),
            };
            edge_proofs[e].opening_reducer = Some(vec![sumcheck_proof]);
            edge_proofs[e].claims.push(combined_claim);
            opening_reduced_count += non_empty.len() - 1;
        }
        if opening_reduced_count > 0 {
            println!("  Opening reducers: {:.3}s ({} proofs saved)", t_reduce.elapsed().as_secs_f64(), opening_reduced_count);
        }

        let master_seed = transcript.challenge_ext2(b"opening_seed");

        let mut tasks: Vec<(usize, usize, Vec<GoldilocksExt2>)> = Vec::new();
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if let Some((i, claim)) = edge_proofs[e].claims.iter().enumerate().rev()
                .find(|(_, c)| !c.point.is_empty()) {
                tasks.push((e, i, claim.point.clone()));
            }
        }

        println!("Opening tasks: {} (1 per edge)", tasks.len());

        // Split tasks into CPU (small) and GPU (large) buckets
        let cpu_open_threshold: usize = std::env::var("CPU_OPEN_THRESHOLD")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(14);
        let num_queries = key.num_queries;

        // Check if we need re-upload path (host_caches populated, GPU commitments freed)
        let use_reupload = gpu_store.host_caches.iter().any(|c| c.is_some());

        let open_start = std::time::Instant::now();
        let proof_map: HashMap<usize, BasefoldOpeningProof>;

        if use_reupload {
            println!("  Opening (re-upload mode): {} tasks, {} devices, {} queries",
                tasks.len(), num_devices, num_queries);
            let mut edge_tasks: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, (e, _, _)) in tasks.iter().enumerate() {
                edge_tasks.entry(*e).or_default().push(idx);
            }
            let edge_list: Vec<(usize, Vec<usize>)> = edge_tasks.into_iter().collect();

            let gpu_threads_per_device: usize = std::env::var("GPU_OPEN_THREADS_PER_DEVICE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(12);
            // `GPU_OPEN_POOL_SIZE` overrides the multiplicative pool sizing.
            // Useful when high-arity opens (n ≥ 24) need serial execution
            // to fit in GPU memory.
            let gpu_pool_size = std::env::var("GPU_OPEN_POOL_SIZE")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or((num_devices * gpu_threads_per_device).max(1));
            let gpu_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(gpu_pool_size)
                .build().expect("failed to build GPU rayon pool for re-upload");

            let tasks_ref = &tasks;
            let host_caches = &gpu_store.host_caches;
            let device_ids = &gpu_store.device_ids;
            let per_device_tables = &gpu_store.per_device_tables;
            let edge_proofs_ref = &edge_proofs;

            let results: Vec<(usize, BasefoldOpeningProof)> = gpu_pool.install(|| {
                edge_list.par_iter().flat_map(|(edge_id, task_indices)| {
                    let cache = host_caches[*edge_id].as_ref()
                        .expect("host cache missing for re-upload opening");
                    let dev = device_ids[*edge_id].unwrap_or(0);
                    let _ = goldilocks_cuda::set_device(dev);
                    let dev_table = &per_device_tables[dev as usize % num_devices];
                    let commitment = cache.to_device().expect("re-upload to_device failed");
                    let proofs: Vec<(usize, BasefoldOpeningProof)> = task_indices.iter().map(|&task_idx| {
                        let (_, claim_idx, point) = &tasks_ref[task_idx];
                        let mut t = Transcript::new(b"bf-open");
                        t.append_ext2(b"", &master_seed);
                        t.append_u64(b"", task_idx as u64);
                        let gpu_proof = commitment.open_ext2(point, dev_table, &mut t, num_queries)
                            .expect("GPU open_ext2 failed (re-upload)");
                        let eval = edge_proofs_ref[*edge_id].claims[*claim_idx].eval;
                        (task_idx, BasefoldOpeningProof { eval, gpu_proof })
                    }).collect();
                    proofs
                }).collect()
            });

            let _ = goldilocks_cuda::set_device(0);
            println!("  Openings (re-upload): {:.3}s ({} edges, {} tasks, {} threads)",
                open_start.elapsed().as_secs_f64(), edge_list.len(), results.len(), gpu_pool_size);

            proof_map = results.into_iter().collect();
        } else {
            let mut gpu_tasks: Vec<usize> = Vec::new();
            let mut cpu_tasks: Vec<usize> = Vec::new();
            for (idx, (e, _, _)) in tasks.iter().enumerate() {
                let nv = witnesses[*e][0].data.as_ref().unwrap().n();
                if nv <= cpu_open_threshold {
                    cpu_tasks.push(idx);
                } else {
                    gpu_tasks.push(idx);
                }
            }
            println!("  Opening: {} GPU (n>{}), {} CPU (n≤{}), {} devices, {} queries",
                gpu_tasks.len(), cpu_open_threshold, cpu_tasks.len(), cpu_open_threshold, num_devices, num_queries);

            let gpu_commitments = &gpu_store.commitments;
            let device_ids = &gpu_store.device_ids;
            let per_device_tables = &gpu_store.per_device_tables;

            gpu_tasks.sort_by_key(|&idx| {
                let (e, _, _) = &tasks[idx];
                std::cmp::Reverse(witnesses[*e][0].data.as_ref().unwrap().n())
            });

            let tasks_ref = &tasks;
            let num_cores = num_cpus::get();
            let gpu_threads_per_device: usize = std::env::var("GPU_OPEN_THREADS_PER_DEVICE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(12);
            // `GPU_OPEN_POOL_SIZE` overrides the multiplicative pool sizing.
            // Useful when high-arity opens (n ≥ 24) need serial execution
            // to fit in GPU memory.
            let gpu_pool_size = std::env::var("GPU_OPEN_POOL_SIZE")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or((num_devices * gpu_threads_per_device).max(1));
            let cpu_pool_size = (num_cores - gpu_pool_size).max(1);
            let cpu_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(cpu_pool_size)
                .build().expect("failed to build CPU rayon pool");

            let (cpu_results, gpu_results) = std::thread::scope(|s| {
                let cpu_handle = s.spawn(|| {
                    let t0 = std::time::Instant::now();
                    let results: Vec<(usize, BasefoldOpeningProof)> = cpu_pool.install(|| {
                        cpu_tasks.par_iter().map(|&task_idx| {
                            let (e, _, point) = &tasks_ref[task_idx];
                            let commitment = gpu_commitments[*e].as_ref()
                                .expect("GPU commitment missing for CPU opening");
                            let mut t = Transcript::new(b"bf-open");
                            t.append_ext2(b"", &master_seed);
                            t.append_u64(b"", task_idx as u64);
                            let proof = cpu_full_open_ext2(commitment, point, table, &mut t, num_queries);
                            (task_idx, proof)
                        }).collect()
                    });
                    (results, t0.elapsed())
                });
                let gpu_handle = s.spawn(|| {
                    if gpu_tasks.is_empty() {
                        return (vec![], std::time::Duration::ZERO);
                    }
                    let t0 = std::time::Instant::now();
                    let gpu_pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(gpu_pool_size)
                        .build().expect("failed to build GPU rayon pool");
                    let results: Vec<(usize, BasefoldOpeningProof)> = gpu_pool.install(|| {
                        gpu_tasks.par_iter().enumerate().map(|(_gpu_idx, &task_idx)| {
                            let (e, _, point) = &tasks_ref[task_idx];
                            let dev = device_ids[*e].unwrap_or(0);
                            let _ = goldilocks_cuda::set_device(dev);
                            let dev_table = &per_device_tables[dev as usize % num_devices];
                            let commitment = gpu_commitments[*e].as_ref()
                                .expect("GPU commitment missing for GPU opening");
                            let mut t = Transcript::new(b"bf-open");
                            t.append_ext2(b"", &master_seed);
                            t.append_u64(b"", task_idx as u64);
                            let proof = commitment.open_ext2(point, dev_table, &mut t, num_queries)
                                .expect("GPU open_ext2 failed");
                            let eval = edge_proofs[*e].claims[tasks_ref[task_idx].1].eval;
                            (task_idx, BasefoldOpeningProof { eval, gpu_proof: proof })
                        }).collect()
                    });
                    (results, t0.elapsed())
                });
                let (cpu_res, cpu_time) = cpu_handle.join().unwrap();
                let (gpu_res, gpu_time) = gpu_handle.join().unwrap();
                let _ = goldilocks_cuda::set_device(0);
                println!("  Openings: CPU {:.3}s ({} tasks, {} threads), GPU {:.3}s ({} tasks, {} threads), wall {:.3}s",
                    cpu_time.as_secs_f64(), cpu_res.len(), cpu_pool_size,
                    gpu_time.as_secs_f64(), gpu_res.len(), gpu_pool_size,
                    open_start.elapsed().as_secs_f64());
                (cpu_res, gpu_res)
            });

            let mut map: HashMap<usize, BasefoldOpeningProof> = HashMap::new();
            map.extend(cpu_results);
            map.extend(gpu_results);
            proof_map = map;
        }

        // Distribute results to edge_proofs (1 proof per edge)
        for (task_idx, (e, _, _)) in tasks.iter().enumerate() {
            if let Some(proof) = proof_map.get(&task_idx) {
                edge_proofs[*e].dense_opening_proof.push(proof.clone());
            }
        }

        println!("  === Total prove_parallel: {:.3}s ===", t_total.elapsed().as_secs_f64());

        ParallelProof {
            boundary_evals,
            partition_proofs,
            edge_proofs,
            range_proof,
            two_pow_proof,
        }
    }

    /// Per-partition verification (private helper for verify_parallel).
    fn verify_partition(
        &self,
        partition: &PartitionDesc,
        partition_proof: &PartitionProof,
        starting_claims: Vec<(EdgeId, Claim)>,
        witnesses: &[Vec<Witness>],
        transcript: &mut Transcript,
    ) -> (bool, Vec<Vec<Claim>>) {
        let reducer = BasicBlockType::Reducer(Reducer {});
        let mut verified = true;

        let partition_nodes: HashSet<NodeId> = partition.node_ids.iter().copied().collect();
        let boundary_inputs: HashSet<EdgeId> = partition.boundary_input_edges.iter().copied().collect();

        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        let mut nodes_to_verify = BTreeSet::new();

        for (e, claim) in starting_claims {
            claims[e].push(claim);
            if let Some(producer) = self.producers[e] {
                if partition_nodes.contains(&producer) {
                    nodes_to_verify.insert(producer);
                }
            }
        }

        // Consumer sets filtered to partition nodes
        let consumer_sets: Vec<BTreeSet<NodeId>> = self.consumers.iter().map(|c| {
            c.iter().copied().filter(|n| partition_nodes.contains(n)).collect()
        }).collect();

        while !nodes_to_verify.is_empty() {
            let u = *nodes_to_verify.iter().next_back().unwrap();
            nodes_to_verify.remove(&u);

            let mut can_verify = true;
            for &edge in &self.nodes[u].outputs {
                if consumer_sets[edge].iter().any(|c| nodes_to_verify.contains(c)) {
                    can_verify = false;
                    break;
                }
            }
            if !can_verify {
                nodes_to_verify.insert(u);
                continue;
            }

            let node = &self.nodes[u];
            let (local_sumcheck_proofs, local_claims) = partition_proof.node_proofs[u].as_ref().unwrap();
            let local_claims_ref: Vec<&Claim> = local_claims.iter().collect();
            let local_witnesses: Vec<&Witness> = local_claims_ref
                .iter()
                .map(|c| &witnesses[c.edge_id][0])
                .collect();
            let local_sumcheck_proofs_ref: Vec<&SumcheckProof> =
                local_sumcheck_proofs.iter().collect();

            // Verify reducer if multiple claims on this output
            if claims[node.outputs[0]].len() > 1 {
                assert!(
                    partition_proof.reducer_proofs[u].is_some(),
                    "reducer proof is not found for node {u} (partition {})",
                    partition.partition_id
                );
                let reducer_witness = vec![&witnesses[node.outputs[0]][0]];
                let mut reducer_claims = claims[node.outputs[0]].clone();
                reducer_claims.push(local_claims_ref[local_claims_ref.len() - 1].clone());
                let reducer_claims_ref: Vec<&Claim> = reducer_claims.iter().collect();
                let reducer_sc_proofs: Vec<&SumcheckProof> =
                    partition_proof.reducer_proofs[u].as_ref().unwrap().iter().collect();
                let reducer_verified = reducer.verify(
                    &reducer_witness,
                    &reducer_claims_ref,
                    &reducer_sc_proofs,
                    transcript,
                );
                verified = verified && reducer_verified;
                if !reducer_verified {
                    println!("verified reducer for node {u} failed (partition {})", partition.partition_id);
                }
                // Push reduced claim to match prover's claims state
                claims[node.outputs[0]].push(local_claims_ref[local_claims_ref.len() - 1].clone());
            }

            // Verify the node
            let node_verified = self.nodes[u].kind.verify(
                &local_witnesses,
                &local_claims_ref,
                &local_sumcheck_proofs_ref,
                transcript,
            );
            if !node_verified {
                println!("verified node {u} failed (partition {})", partition.partition_id);
            }
            verified = verified && node_verified;

            // Propagate input claims
            let mut claimed_edge_ids: Vec<usize> = Vec::new();
            for c in &local_claims_ref {
                if node.inputs.contains(&c.edge_id) {
                    claims[c.edge_id].push((*c).clone());
                    if !claimed_edge_ids.contains(&c.edge_id) {
                        claimed_edge_ids.push(c.edge_id);
                    }
                } else if self.self_claim_edges.contains(&c.edge_id) && node.outputs.contains(&c.edge_id) {
                    claims[c.edge_id].push((*c).clone());
                }
            }

            for e in claimed_edge_ids {
                if boundary_inputs.contains(&e) { continue; } // STOP at boundary
                if let Some(producer) = self.producers[e] {
                    if partition_nodes.contains(&producer) {
                        nodes_to_verify.insert(producer);
                    }
                }
            }
        }

        (verified, claims)
    }

    /// Parallel verify: verify partition proofs and opening proofs.
    pub fn verify_parallel(
        &self,
        proof: &ParallelProof,
        witnesses: &[Vec<Witness>],
        _verifier_key: &BasefoldVerifierKey,
        commitments: &[Option<BasefoldCommitmentData>],
        table: &BasefoldTable,
        transcript: &mut Transcript,
        partitions: &[PartitionDesc],
    ) -> bool {
        let mut verified = true;

        // Step 1: Re-derive output challenges
        let mut initial_claims: Vec<(EdgeId, Claim)> = Vec::new();
        for &e in &self.output_ports {
            let w = &witnesses[e][0];
            if w.role == Role::Output {
                let n = proof.edge_proofs[e].claims[0].point.len();
                let point: Vec<GoldilocksExt2> = (0..n)
                    .map(|_| transcript.challenge_ext2(b"challenge"))
                    .collect();
                verified = verified && proof.edge_proofs[e].claims[0].point == point;
                initial_claims.push((e, proof.edge_proofs[e].claims[0].clone()));
            }
        }

        // Step 2: Re-derive boundary challenges and verify they match
        for (idx, &b) in self.boundary_edges.iter().enumerate() {
            let (edge_id, ref bpoint, beval) = proof.boundary_evals[idx];
            assert_eq!(edge_id, b);
            let n = bpoint.len();
            let point: Vec<GoldilocksExt2> = (0..n)
                .map(|_| transcript.challenge_ext2(b"challenge"))
                .collect();
            if *bpoint != point {
                println!("boundary edge {} challenge mismatch", b);
                return false;
            }
            initial_claims.push((b, Claim { edge_id: b, sparse_id: 0, point: bpoint.clone(), eval: beval }));
        }

        // Step 3: Route claims to partitions (by producer node's partition)
        let mut partition_starting_claims: Vec<Vec<(EdgeId, Claim)>> = vec![Vec::new(); partitions.len()];
        let partition_node_sets: Vec<HashSet<NodeId>> = partitions.iter()
            .map(|p| p.node_ids.iter().copied().collect())
            .collect();
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

        // Step 4: Fork transcript & verify partitions in parallel
        let transcript_ref: &Transcript = &*transcript;
        let results: Vec<(bool, Vec<Vec<Claim>>)> = partition_starting_claims
            .into_par_iter()
            .enumerate()
            .map(|(k, starting_claims)| {
                let mut t_k = transcript_ref.fork(k);
                self.verify_partition(
                    &partitions[k],
                    &proof.partition_proofs[k],
                    starting_claims,
                    witnesses,
                    &mut t_k,
                )
            })
            .collect();

        // Merge results
        let mut claims: Vec<Vec<Claim>> = vec![Vec::new(); self.num_edges];
        for (v, partition_claims) in &results {
            verified = verified && *v;
            for (e, c) in partition_claims.iter().enumerate() {
                claims[e].extend(c.iter().cloned());
            }
        }

        // Step 5: Verify lookup proofs
        // Build merged node_proofs for lookup verification
        let mut merged_node_proofs: Vec<Option<(Vec<SumcheckProof>, Vec<Claim>)>> = vec![None; self.nodes.len()];
        for pp in &proof.partition_proofs {
            for (i, np) in pp.node_proofs.iter().enumerate() {
                if np.is_some() {
                    merged_node_proofs[i] = np.clone();
                }
            }
        }

        let two_pow_verified = self.verify_two_pow(&merged_node_proofs, &proof.two_pow_proof, transcript);
        if !two_pow_verified {
            println!("two_pow verification failed");
            return false;
        }
        let range_verified = self.verify_range(&merged_node_proofs, witnesses, &claims, &proof.range_proof, transcript);
        if !range_verified {
            println!("range verification failed");
            return false;
        }

        // Step 6: Verify opening reducers (must match prover transcript)
        let edge_proofs = &proof.edge_proofs;
        let reducer_for_opening = BasicBlockType::Reducer(Reducer {});
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if edge_proofs[e].opening_reducer.is_none() { continue; }
            let non_empty: Vec<&Claim> = edge_proofs[e].claims.iter()
                .rev().skip(1).collect::<Vec<_>>().into_iter().rev()
                .filter(|c| !c.point.is_empty()).collect();
            let combined = edge_proofs[e].claims.last().unwrap();
            let mut all_claims: Vec<&Claim> = non_empty;
            all_claims.push(combined);
            let reducer_witness = vec![&witnesses[e][0]];
            let sc_proofs: Vec<&SumcheckProof> =
                edge_proofs[e].opening_reducer.as_ref().unwrap().iter().collect();
            let rv = reducer_for_opening.verify(&reducer_witness, &all_claims, &sc_proofs, transcript);
            if !rv {
                println!("Opening reducer verification failed at edge {}", e);
            }
            verified = verified && rv;
        }

        // Step 7: Verify opening proofs
        let master_seed = transcript.challenge_ext2(b"opening_seed");

        let mut tasks: Vec<(usize, usize)> = Vec::new();
        for e in 0..self.num_edges {
            let w = &witnesses[e][0];
            if w.role == Role::Output && !self.self_claim_edges.contains(&e) { continue; }
            if commitments[e].is_none() { continue; }
            if let Some((i, _)) = edge_proofs[e].claims.iter().enumerate().rev()
                .find(|(_, c)| !c.point.is_empty()) {
                tasks.push((e, i));
            }
        }

        let all_verified: Vec<bool> = tasks.par_iter().enumerate().map(|(task_idx, (e, i))| {
            let mut t = Transcript::new(b"bf-open");
            t.append_ext2(b"", &master_seed);
            t.append_u64(b"", task_idx as u64);

            let opening = &edge_proofs[*e].dense_opening_proof[0];
            let root = &commitments[*e].as_ref().unwrap().root;
            let point = &edge_proofs[*e].claims[*i].point;

            let ok = BasefoldVerifier::verify_ext2(root, point, &opening.gpu_proof, table, &mut t)
                .unwrap_or(false);
            if !ok {
                println!("Opening proof verification failed at edge {} claim {}", e, i);
            }
            let eval_ok = ext2_field_eq(opening.gpu_proof.eval, edge_proofs[*e].claims[*i].eval);
            if !eval_ok {
                println!("Opening eval mismatch at edge {} claim {}: proof.eval={:?} claim.eval={:?}",
                    e, i, opening.gpu_proof.eval, edge_proofs[*e].claims[*i].eval);
            }
            ok && eval_ok
        }).collect();
        if !all_verified.iter().all(|&v| v) {
            return false;
        }

        verified
    }

    /// Prove two_pow lookup.
    fn prove_two_pow(
        &self,
        witnesses: &[Vec<Witness>],
        claims: &mut Vec<Vec<Claim>>,
        transcript: &mut Transcript,
    ) -> LookupProof {
        if self.two_pow.is_empty() {
            return LookupProof { table_proofs: vec![], middle_claims: vec![], bool_proofs: vec![] };
        }

        let num_two_pow = self.two_pow.len();
        let beta = transcript.challenge_ext2(b"two_pow_beta");
        let betas = crate::util::arith::calc_pow_vec_ext2(beta, num_two_pow);

        let table_num_vars = 4usize; // two_pow table has 16 entries
        let table_size = 1usize << table_num_vars;
        let mut combined_aux = vec![GoldilocksExt2::zero(); table_size];
        let mut middle_claims_all: Vec<Vec<GoldilocksExt2>> = Vec::new();

        // Two-pow table values: [2^15, 2^14, ..., 2^0]
        let two_pow_table: Vec<GoldilocksExt2> = (0..table_size)
            .map(|j| GoldilocksExt2::from_base(GoldilocksField(1u64 << (15 - j))))
            .collect();

        for (idx, &nid) in self.two_pow.iter().enumerate() {
            let node = &self.nodes[nid];
            let inp_edge = node.inputs[0]; // SparseMLPoly from ExpHelper
            let out_edge = node.outputs[0];

            // Get the output claim point (from backward pass)
            let out_claim = claims[out_edge].last().expect("TwoPow output must have a claim");
            let claim_point = out_claim.point.clone();

            // Get the sparse polynomial (selection) from the input edge
            let sparse = witnesses[inp_edge][0]
                .data
                .as_ref()
                .unwrap()
                .as_any()
                .downcast_ref::<SparseMLPoly>()
                .expect("TwoPow input must be SparseMLPoly");

            let input_num_vars = sparse.selection.input_num_vars;
            // Compute eq table for input claim point
            let eq_ext2 = evaluate_lagrange_basis_ext2(&claim_point[..input_num_vars]);

            // Compute partial aux: fix input variables, sum over inputs mapping to each table entry
            let mut part_aux = vec![GoldilocksExt2::zero(); table_size];
            for &(input_idx, table_idx) in &sparse.selection.selection {
                if input_idx < eq_ext2.len() && table_idx < table_size {
                    part_aux[table_idx] = ext2_add(part_aux[table_idx], eq_ext2[input_idx]);
                }
            }

            // middle_claim = Σ_j part_aux[j] * two_pow_table[j]
            let mut mc = GoldilocksExt2::zero();
            for j in 0..table_size {
                mc = ext2_add(mc, ext2_mul(part_aux[j], two_pow_table[j]));
            }
            middle_claims_all.push(vec![mc]);

            // Accumulate weighted partial aux for table sumcheck
            for j in 0..table_size {
                combined_aux[j] = ext2_add(combined_aux[j], ext2_mul(betas[idx], part_aux[j]));
            }
        }

        // Table sumcheck: Σ_y combined_aux(y) * two_pow_table(y) = expected_sum
        let mut expected_sum = GoldilocksExt2::zero();
        for (idx, mc) in middle_claims_all.iter().enumerate() {
            expected_sum = ext2_add(expected_sum, ext2_mul(betas[idx], mc[0]));
        }

        // Build the two_pow_ext2 polynomial for sumcheck
        let table_poly = two_pow_table.clone();

        let mut prover = CpuLinearSumcheckProverExt2::new(table_num_vars, 2, transcript);
        let table_proof = prover.prove(&mut [combined_aux, table_poly].as_mut_slice(), transcript);
        let table_challenges = prover.challenges.clone();

        // Add claims for the auxiliary (sparse) edges at (claim_point || table_challenges)
        for (_idx, &nid) in self.two_pow.iter().enumerate() {
            let node = &self.nodes[nid];
            let inp_edge = node.inputs[0];
            let out_edge = node.outputs[0];

            let out_claim = claims[out_edge].last().unwrap();
            let input_num_vars = witnesses[inp_edge][0].data.as_ref().unwrap()
                .as_any().downcast_ref::<SparseMLPoly>().unwrap().selection.input_num_vars;

            let mut point = out_claim.point[..input_num_vars].to_vec();
            point.extend_from_slice(&table_challenges);
            let eval = witnesses[inp_edge][0].data.as_ref().unwrap().evaluate_at_point_ext2(&point);

            claims[inp_edge].push(Claim {
                edge_id: inp_edge,
                sparse_id: 0,
                point,
                eval,
            });
        }

        LookupProof {
            table_proofs: vec![table_proof],
            middle_claims: middle_claims_all,
            // TODO(soundness): two_pow selection polynomials are not proven boolean.
            // A malicious prover could set selection entries to non-{0,1} values.
            // Add boolean constraint proofs here, mirroring prove_range's approach.
            bool_proofs: vec![],
        }
    }

    /// Verify two_pow lookup.
    fn verify_two_pow(
        &self,
        node_proofs: &[Option<(Vec<SumcheckProof>, Vec<Claim>)>],
        two_pow_proof: &LookupProof,
        transcript: &mut Transcript,
    ) -> bool {
        if self.two_pow.is_empty() {
            return true;
        }

        let num_two_pow = self.two_pow.len();
        let beta = transcript.challenge_ext2(b"two_pow_beta");
        let betas = crate::util::arith::calc_pow_vec_ext2(beta, num_two_pow);

        // Check middle_claims match output evaluations
        for (idx, &nid) in self.two_pow.iter().enumerate() {
            let (_, node_claims) = node_proofs[nid].as_ref().unwrap();
            // node_claims for TwoPow: only the output claim
            let out_eval = node_claims.last().unwrap().eval;
            let mc = two_pow_proof.middle_claims[idx][0];
            if out_eval != mc {
                println!("two_pow middle_claim mismatch at node {nid}: out_eval={:?}, mc={:?}", out_eval, mc);
                return false;
            }
        }

        // Compute expected sum
        let mut expected_sum = GoldilocksExt2::zero();
        for (idx, mc) in two_pow_proof.middle_claims.iter().enumerate() {
            expected_sum = ext2_add(expected_sum, ext2_mul(betas[idx], mc[0]));
        }

        // Verify table sumcheck
        let table_num_vars = 4;
        let (ok, _challenges) = SumcheckVerifier::verify(
            &two_pow_proof.table_proofs[0],
            expected_sum,
            table_num_vars,
            2,
            transcript,
        );
        if !ok {
            println!("two_pow table sumcheck verification failed");
            return false;
        }

        // TODO(soundness): no boolean proof verification for two_pow selection polynomials.
        // Must be added once prove_two_pow generates bool_proofs (see TODO there).

        true
    }

    /// Prove range lookup using dense bit decomposition polynomials.
    ///
    /// Each range node has a dense bit polynomial B(x,y) with 5 table vars (32 bit positions).
    /// B(x,y) = bit y of value at position x. value(x) = Σ_y B(x,y) · 2^y.
    /// Table T(y) = 2^y.
    fn prove_range(
        &self,
        witnesses: &[Vec<Witness>],
        claims: &mut Vec<Vec<Claim>>,
        transcript: &mut Transcript,
    ) -> LookupProof {
        use crate::basicblock::scale::BIT_TABLE_VARS;

        if self.range.is_empty() {
            return LookupProof { table_proofs: vec![], middle_claims: vec![], bool_proofs: vec![] };
        }

        let num_range = self.range.len();
        let table_size = 1usize << BIT_TABLE_VARS; // 32

        // 1. Sample challenges
        let alpha = transcript.challenge_ext2(b"range_alpha");
        let beta = transcript.challenge_ext2(b"range_beta");
        let betas = crate::util::arith::calc_pow_vec_ext2(beta, num_range);

        // 2. Collect info per range node
        struct RangeNodeInfo {
            nid: NodeId,
            aux_edge: EdgeId,
            input_num_vars: usize,
            part_aux: Vec<GoldilocksExt2>, // 32-element partial evaluation
            middle_claim: GoldilocksExt2,  // Σ_y part_aux[y] · 2^y
            sum_aux: GoldilocksExt2,       // Σ_y part_aux[y]
        }

        // Pre-collect metadata and claim points (sequential)
        struct RangeNodePrep {
            nid: NodeId,
            aux_edge: EdgeId,
            input_num_vars: usize,
            claim_point: Vec<GoldilocksExt2>,
        }
        let mut preps: Vec<RangeNodePrep> = Vec::with_capacity(num_range);
        for &nid in self.range.iter() {
            let node = &self.nodes[nid];
            let aux_idx = match node.kind {
                BasicBlockType::NonNegative(_) => 0,
                _ => 1,
            };
            let aux_edge = node.outputs[aux_idx];
            let inp_edge = node.inputs[0];

            let bit_poly_n = witnesses[aux_edge][0].data.as_ref().unwrap().n();
            let input_num_vars = bit_poly_n - BIT_TABLE_VARS;

            let claim_point = claims[inp_edge].iter()
                .rev()
                .find(|c| !c.point.is_empty())
                .map(|c| c.point.clone())
                .unwrap_or_else(|| {
                    (0..input_num_vars)
                        .map(|_| transcript.challenge_ext2(b"range_inp_challenge"))
                        .collect()
                });

            preps.push(RangeNodePrep { nid, aux_edge, input_num_vars, claim_point });
        }

        // Parallel: compute eq tables, partial evaluations, middle claims
        let infos: Vec<RangeNodeInfo> = preps.par_iter().map(|prep| {
            let input_size = 1usize << prep.input_num_vars;

            let bit_evals = witnesses[prep.aux_edge][0]
                .data.as_ref().unwrap()
                .evaluations_ref();

            // Compute eq table for input variables at claim point
            let eq_ext2 = evaluate_lagrange_basis_ext2(&prep.claim_point[..prep.input_num_vars]);

            // Partial evaluation: for each y in 0..32, part_aux[y] = Σ_x B[x + y*input_size] * eq(r, x)
            let mut part_aux = vec![GoldilocksExt2::zero(); table_size];
            for y in 0..table_size {
                let base = y * input_size;
                let mut acc = GoldilocksExt2::zero();
                for x in 0..input_size {
                    if bit_evals[base + x].0 != 0 {
                        acc = ext2_add(acc, eq_ext2[x]);
                    }
                }
                part_aux[y] = acc;
            }

            // middle_claim = Σ_y part_aux[y] · 2^y (T(y) = 2^y)
            let mut mc = GoldilocksExt2::zero();
            let mut sa = GoldilocksExt2::zero();
            for y in 0..table_size {
                let two_pow_y = GoldilocksExt2::from_base(GoldilocksField(1u64 << y));
                mc = ext2_add(mc, ext2_mul(part_aux[y], two_pow_y));
                sa = ext2_add(sa, part_aux[y]);
            }

            RangeNodeInfo {
                nid: prep.nid,
                aux_edge: prep.aux_edge,
                input_num_vars: prep.input_num_vars,
                part_aux,
                middle_claim: mc,
                sum_aux: sa,
            }
        }).collect();

        let middle_claims_all: Vec<Vec<GoldilocksExt2>> = infos.iter()
            .map(|info| vec![info.middle_claim, info.sum_aux])
            .collect();

        // 3. Single table sumcheck over 5 variables
        // combined[y] = Σ_i β_i · part_aux_i[y]
        let mut combined_aux = vec![GoldilocksExt2::zero(); table_size];
        for (i, info) in infos.iter().enumerate() {
            for y in 0..table_size {
                combined_aux[y] = ext2_add(combined_aux[y], ext2_mul(betas[i], info.part_aux[y]));
            }
        }

        // expected_sum = Σ_i β_i · (mc_i + α · sa_i)
        let mut expected_sum = GoldilocksExt2::zero();
        for (i, info) in infos.iter().enumerate() {
            expected_sum = ext2_add(
                expected_sum,
                ext2_mul(betas[i], ext2_add(info.middle_claim, ext2_mul(alpha, info.sum_aux))),
            );
        }

        // T(y) + α = 2^y + α
        let table_alpha: Vec<GoldilocksExt2> = (0..table_size)
            .map(|y| ext2_add(GoldilocksExt2::from_base(GoldilocksField(1u64 << y)), alpha))
            .collect();

        let mut prover = CpuLinearSumcheckProverExt2::new(BIT_TABLE_VARS, 2, transcript);
        let proof = prover.prove(&mut [combined_aux, table_alpha].as_mut_slice(), transcript);
        let table_challenges = prover.challenges.clone();

        // 4. Add claims for auxiliary edges.
        // B(r_x, r_y) = Σ_y eq(r_y, y) · part_aux[y], avoiding O(2^n) evaluate_at_point_ext2.
        let eq_table_y = evaluate_lagrange_basis_ext2(&table_challenges[..BIT_TABLE_VARS]);
        let claims_ref: &Vec<Vec<Claim>> = &*claims;
        let aux_results: Vec<(EdgeId, Claim)> = infos.par_iter().map(|info| {
            let inp_edge = self.nodes[info.nid].inputs[0];

            let claim_point = claims_ref[inp_edge].iter()
                .rev()
                .find(|c| !c.point.is_empty())
                .map(|c| c.point.clone())
                .unwrap_or_else(|| vec![GoldilocksExt2::zero(); info.input_num_vars]);
            let mut point = claim_point[..info.input_num_vars].to_vec();
            point.extend_from_slice(&table_challenges[..BIT_TABLE_VARS]);

            // eval = Σ_y eq(r_y, y) · part_aux[y] (reuse already-computed partial eval)
            let mut eval = GoldilocksExt2::zero();
            for y in 0..table_size {
                eval = ext2_add(eval, ext2_mul(eq_table_y[y], info.part_aux[y]));
            }

            (info.aux_edge, Claim {
                edge_id: info.aux_edge,
                sparse_id: 0,
                point,
                eval,
            })
        }).collect();
        for (aux_edge, claim) in aux_results {
            claims[aux_edge].push(claim);
        }

        // 5. Boolean constraint proofs: prove B_j(x) ∈ {0,1} for all range nodes.
        //
        // For each group of range nodes with the same aux_num_var:
        //   C(x) = Σ_j beta_j * B_j(x) * (B_j(x) - 1)
        // If all bits are boolean, C(x) = 0 everywhere.
        // Prove: Σ_x eq(r, x) * C(x) = 0  (degree-2 sumcheck with 2 polynomials).
        use std::collections::BTreeMap;
        let mut bool_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, info) in infos.iter().enumerate() {
            let aux_num_var = info.input_num_vars + BIT_TABLE_VARS;
            bool_groups.entry(aux_num_var).or_default().push(i);
        }

        let mut bool_proofs = Vec::new();
        for (&aux_num_var, indices) in &bool_groups {
            let poly_size = 1usize << aux_num_var;

            // Sample eq challenges from transcript
            let eq_challenges: Vec<GoldilocksExt2> = (0..aux_num_var)
                .map(|_| transcript.challenge_ext2(b"bool_challenge"))
                .collect();
            let eq_evals = evaluate_lagrange_basis_ext2(&eq_challenges);

            // Build combined C(x) = Σ_j beta_j * B_j(x) * (B_j(x) - 1)
            let mut combined = vec![GoldilocksExt2::zero(); poly_size];
            for &idx in indices {
                let info = &infos[idx];
                let bit_evals = witnesses[info.aux_edge][0]
                    .data.as_ref().unwrap()
                    .evaluations_ref();
                let beta_i = betas[idx];

                for x in 0..poly_size {
                    let b = GoldilocksExt2::from_base(bit_evals[x]);
                    let b_minus_1 = ext2_sub(b, GoldilocksExt2::one());
                    let product = ext2_mul(ext2_mul(b, b_minus_1), beta_i);
                    combined[x] = ext2_add(combined[x], product);
                }
            }

            // Prove: Σ_x eq(r,x) * C(x) = 0
            let mut bool_prover = CpuLinearSumcheckProverExt2::new(aux_num_var, 2, transcript);
            let bool_proof = bool_prover.prove(&mut [eq_evals, combined], transcript);
            bool_proofs.push(bool_proof);
        }

        LookupProof {
            table_proofs: vec![proof],
            middle_claims: middle_claims_all,
            bool_proofs,
        }
    }

    /// Verify range lookup using dense bit decomposition polynomials.
    fn verify_range(
        &self,
        node_proofs: &[Option<(Vec<SumcheckProof>, Vec<Claim>)>],
        witnesses: &[Vec<Witness>],
        claims: &[Vec<Claim>],
        range_proof: &LookupProof,
        transcript: &mut Transcript,
    ) -> bool {
        use crate::basicblock::scale::BIT_TABLE_VARS;

        if self.range.is_empty() {
            return true;
        }

        let num_range = self.range.len();
        let sf_ext2 = GoldilocksExt2::from_base(GoldilocksField(*SF_INT as u64));

        // 1. Re-derive challenges (must match prover)
        let alpha = transcript.challenge_ext2(b"range_alpha");
        let beta = transcript.challenge_ext2(b"range_beta");
        let betas = crate::util::arith::calc_pow_vec_ext2(beta, num_range);

        // 2. Compute expected sum from middle claims and verify eval_to_check
        let mut expected_sum = GoldilocksExt2::zero();
        for (idx, &nid) in self.range.iter().enumerate() {
            let node = &self.nodes[nid];
            let inp_edge = node.inputs[0];
            let aux_idx = match node.kind {
                BasicBlockType::NonNegative(_) => 0,
                _ => 1,
            };
            let aux_edge = node.outputs[aux_idx];

            // Determine input_num_vars (same as prover)
            let bit_poly_n = witnesses[aux_edge][0].data.as_ref().unwrap().n();
            let input_num_vars = bit_poly_n - BIT_TABLE_VARS;

            // Reconstruct claim_point logic (must match prover for transcript sync)
            let inp_claim = claims[inp_edge].iter().rev().find(|c| !c.point.is_empty());
            if inp_claim.is_none() {
                // Generate same fallback challenges as prover to keep transcript in sync
                for _ in 0..input_num_vars {
                    let _ = transcript.challenge_ext2(b"range_inp_challenge");
                }
            }

            let mc = range_proof.middle_claims[idx][0];
            let sa = range_proof.middle_claims[idx][1];
            expected_sum = ext2_add(
                expected_sum,
                ext2_mul(betas[idx], ext2_add(mc, ext2_mul(alpha, sa))),
            );

            // Compute eval_to_check and verify mc == eval_to_check
            let eval_to_check = match &node.kind {
                BasicBlockType::ScaleDown(_) => {
                    // eval_to_check = input_eval - output_eval * SF_INT
                    // node_proofs[nid].1 = [inp_claim, out_claim, ...]
                    node_proofs[nid].as_ref().map(|nc| {
                        let inp_eval = nc.1[0].eval;
                        let out_eval = nc.1[1].eval;
                        ext2_sub(inp_eval, ext2_mul(out_eval, sf_ext2))
                    })
                }
                BasicBlockType::ScaleUp(_) => {
                    // ScaleUp: remainder is always 0
                    Some(GoldilocksExt2::zero())
                }
                BasicBlockType::NonNegative(_) => {
                    // For NonNeg, mc = Σ_y 2^y * B(r,y) should equal input(r) directly.
                    // NOTE: This check will FAIL for values >= 2^32 because the prover
                    // produces all-zero bits, giving mc=0 != input_eval. This is correct
                    // rejection behavior — the proof is unsound for out-of-range values.
                    // FIX: increase BIT_TABLE_VARS from 5 → 6 (see scale.rs).
                    node_proofs[nid].as_ref().map(|nc| nc.1[0].eval)
                }
                _ => None,
            };

            if let Some(etc) = eval_to_check {
                if mc != etc {
                    println!("range eval_to_check mismatch for node {nid} (kind: {:?})", node.kind);
                    return false;
                }
            }
        }

        // 3. Verify single table sumcheck over 5 variables
        if range_proof.table_proofs.is_empty() {
            println!("range: no table proofs");
            return false;
        }

        let (ok, _challenges) = SumcheckVerifier::verify(
            &range_proof.table_proofs[0],
            expected_sum,
            BIT_TABLE_VARS,
            2,
            transcript,
        );
        if !ok {
            println!("range table sumcheck verification failed");
            return false;
        }

        // 4. Verify boolean constraint proofs: B_j(x) ∈ {0,1}
        // Group range nodes by aux_num_var (must match prover ordering)
        use std::collections::BTreeMap;
        let mut bool_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, &nid) in self.range.iter().enumerate() {
            let node = &self.nodes[nid];
            let aux_idx = match node.kind {
                BasicBlockType::NonNegative(_) => 0,
                _ => 1,
            };
            let aux_edge = node.outputs[aux_idx];
            let bit_poly_n = witnesses[aux_edge][0].data.as_ref().unwrap().n();
            let input_num_vars = bit_poly_n - BIT_TABLE_VARS;
            let aux_num_var = input_num_vars + BIT_TABLE_VARS;
            bool_groups.entry(aux_num_var).or_default().push(i);
        }

        let mut bool_proof_idx = 0;
        for (&aux_num_var, _indices) in &bool_groups {
            if bool_proof_idx >= range_proof.bool_proofs.len() {
                println!("range: missing boolean proof for aux_num_var={}", aux_num_var);
                return false;
            }

            // Re-derive eq challenges (must match prover)
            for _ in 0..aux_num_var {
                let _ = transcript.challenge_ext2(b"bool_challenge");
            }

            // Verify with expected_sum = 0 (boolean constraint)
            let (ok, _) = SumcheckVerifier::verify(
                &range_proof.bool_proofs[bool_proof_idx],
                GoldilocksExt2::zero(),
                aux_num_var,
                2,
                transcript,
            );
            if !ok {
                println!("range boolean constraint proof verification failed (aux_num_var={})", aux_num_var);
                return false;
            }
            bool_proof_idx += 1;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goldilocks_cuda::basefold::BasefoldTable;
    use plonky2::util::timing::TimingTree;
    use rand::Rng;

    #[test]
    fn test_dag_run_simple_add() {
        // Build: a + b → output
        let mut g = DagBuilder::new();
        let a = g.input(vec![4], DataType::Uint);
        let b = g.input(vec![4], DataType::Uint);
        let _out = g.add(a, b);

        let (dag, mut witnesses) = g.compile();

        // Feed inputs
        let a_data = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b_data = Witness::new(
            vec![4],
            vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
            DataType::Uint,
            0,
            Role::Input,
        );

        dag.run(&mut witnesses, &[(0, a_data), (1, b_data)]);

        // Check output
        let out_edge = dag.output_ports.iter()
            .find(|&&e| witnesses[e][0].role == Role::Output)
            .copied()
            .unwrap_or(2);
        let out_evals = witnesses[out_edge][0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(out_evals[0], GoldilocksField(11));
        assert_eq!(out_evals[1], GoldilocksField(22));
        assert_eq!(out_evals[2], GoldilocksField(33));
        assert_eq!(out_evals[3], GoldilocksField(44));
    }

    #[test]
    fn test_dag_end_to_end() {
        goldilocks_cuda::init().expect("CUDA init failed");
        // Build: a + b → output
        let mut g = DagBuilder::new();
        let a = g.input(vec![4], DataType::Uint);
        let b = g.input(vec![4], DataType::Uint);
        let _out = g.add(a, b);

        let (dag, mut witnesses) = g.compile();

        // Feed inputs
        let a_data = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b_data = Witness::new(
            vec![4],
            vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
            DataType::Uint,
            0,
            Role::Input,
        );

        // Run forward pass
        dag.run(&mut witnesses, &[(0, a_data), (1, b_data)]);

        // Commit with GPU store
        let key = BasefoldCommitKey::default();
        let max_nv = witnesses.iter()
            .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
            .filter(|&n| n <= 22)
            .max().unwrap_or(4);
        let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
        let mut gpu_store = GpuCommitmentStore::new(max_nv, key.log_rate, key.seed, dag.num_edges());
        dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

        // Prove
        let mut transcript = Transcript::new(b"test_dag");
        let mut timing = TimingTree::new("test", log::Level::Info);
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);

        // Verify
        let vk = BasefoldVerifierKey::from(&key);
        let table = BasefoldTable::generate(max_nv, vk.log_rate, max_nv, vk.seed);
        let mut verify_transcript = Transcript::new(b"test_dag");
        let verified = dag.verify(
            &node_proofs,
            &edge_proofs,
            &range_proof,
            &two_pow_proof,
            &reducer_proofs,
            &witnesses,
            &vk,
            &commitments,
            &table,
            &mut verify_transcript,
        );
        assert!(verified, "End-to-end DAG verification should pass");
    }

    #[test]
    fn test_maxpool_general_prove_verify() {
        goldilocks_cuda::init().expect("CUDA init failed");
        // Test maxpool_general in isolation: X[1, 8, 8] → maxpool(3,3,2,2) → Y[1, 3, 3]
        let mut g = DagBuilder::new();
        let x = g.input(vec![1, 8, 8], DataType::Uint);
        let _y = g.maxpool_general(x, 3, 3, 2, 2);

        let (dag, mut witnesses) = g.compile();

        // Generate input (deterministic seed for reproducibility)
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let input_data: Vec<GoldilocksField> = (0..64)
            .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
            .collect();
        let x_data = Witness::new(vec![1, 8, 8], input_data, DataType::Uint, 0, Role::Input);

        dag.run(&mut witnesses, &[(0, x_data)]);

        // Commit
        let key = BasefoldCommitKey::default();
        let max_nv = witnesses.iter()
            .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
            .filter(|&n| n <= 22)
            .max().unwrap_or(4);
        let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
        let mut gpu_store = GpuCommitmentStore::new(max_nv, key.log_rate, key.seed, dag.num_edges());
        dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

        // Prove
        let mut transcript = Transcript::new(b"test_mp");
        let mut timing = TimingTree::new("test", log::Level::Info);
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);

        let vk = BasefoldVerifierKey::from(&key);
        let verifier_table = BasefoldTable::generate(max_nv, vk.log_rate, max_nv, vk.seed);
        let mut verify_transcript = Transcript::new(b"test_mp");
        let verified = dag.verify(
            &node_proofs,
            &edge_proofs,
            &range_proof,
            &two_pow_proof,
            &reducer_proofs,
            &witnesses,
            &vk,
            &commitments,
            &verifier_table,
            &mut verify_transcript,
        );
        assert!(verified, "maxpool_general prove/verify should pass");
    }

    #[test]
    fn test_pad_asym_maxpool_prove_verify() {
        goldilocks_cuda::init().expect("CUDA init failed");
        // Test pad_asym → maxpool_general: X[1, 8, 8] → pad_asym(0,1,0,1) → maxpool(3,3,2,2)
        let mut g = DagBuilder::new();
        let x = g.input(vec![1, 8, 8], DataType::Uint);
        let h = g.pad_asym(x, 0, 1, 0, 1);
        let _y = g.maxpool_general(h, 3, 3, 2, 2);

        let (dag, mut witnesses) = g.compile();

        let mut rng = rand::thread_rng();
        let input_data: Vec<GoldilocksField> = (0..64)
            .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
            .collect();
        let x_data = Witness::new(vec![1, 8, 8], input_data, DataType::Uint, 0, Role::Input);
        dag.run(&mut witnesses, &[(0, x_data)]);

        let key = BasefoldCommitKey::default();
        let max_nv = witnesses.iter()
            .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
            .filter(|&n| n <= 22)
            .max().unwrap_or(4);
        let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
        let mut gpu_store = GpuCommitmentStore::new(max_nv, key.log_rate, key.seed, dag.num_edges());
        dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

        let mut transcript = Transcript::new(b"test_pm");
        let mut timing = TimingTree::new("test", log::Level::Info);
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);

        let vk = BasefoldVerifierKey::from(&key);
        let table = BasefoldTable::generate(max_nv, vk.log_rate, max_nv, vk.seed);
        let mut verify_transcript = Transcript::new(b"test_pm");
        let verified = dag.verify(
            &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
            &witnesses, &vk, &commitments, &table, &mut verify_transcript,
        );
        assert!(verified, "pad_asym + maxpool_general prove/verify should pass");
    }

    #[test]
    fn test_relu_pad_maxpool_prove_verify() {
        goldilocks_cuda::init().expect("CUDA init failed");
        // Test relu → pad_asym → maxpool_general
        let mut g = DagBuilder::new();
        let x = g.input(vec![1, 8, 8], DataType::Uint);
        let h = g.relu(x);
        let h = g.pad_asym(h, 0, 1, 0, 1);
        let _y = g.maxpool_general(h, 3, 3, 2, 2);

        let (dag, mut witnesses) = g.compile();

        let mut rng = rand::thread_rng();
        let input_data: Vec<GoldilocksField> = (0..64)
            .map(|_| GoldilocksField((rng.gen::<u32>() % 100) as u64))
            .collect();
        let x_data = Witness::new(vec![1, 8, 8], input_data, DataType::Uint, 0, Role::Input);
        dag.run(&mut witnesses, &[(0, x_data)]);

        let key = BasefoldCommitKey::default();
        let max_nv = witnesses.iter()
            .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
            .filter(|&n| n <= 22)
            .max().unwrap_or(4);
        let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
        let mut gpu_store = GpuCommitmentStore::new(max_nv, key.log_rate, key.seed, dag.num_edges());
        dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

        let mut transcript = Transcript::new(b"test_rpm");
        let mut timing = TimingTree::new("test", log::Level::Info);
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);

        let vk = BasefoldVerifierKey::from(&key);
        let table = BasefoldTable::generate(max_nv, vk.log_rate, max_nv, vk.seed);
        let mut verify_transcript = Transcript::new(b"test_rpm");
        let verified = dag.verify(
            &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
            &witnesses, &vk, &commitments, &table, &mut verify_transcript,
        );
        assert!(verified, "relu + pad_asym + maxpool_general prove/verify should pass");
    }
}
