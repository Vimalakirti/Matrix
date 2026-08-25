//! DAG types and execution. The Ajtai commit phase and the fold-tree opening
//! protocol replace zk-torch-3's basefold commit/prove/verify; those are
//! built in steps 8-13 of the plan. Here we keep the structural pieces
//! (Dag, Node, the forward-pass `run`, partition utilities) that downstream
//! code needs to build and execute a model.

pub mod bert;
pub mod builder;
pub mod fold_integration;
pub mod streaming_accumulator;
pub mod gpt2;
pub mod gptj;
pub mod lenet;
pub mod llama;
pub mod lookups;
pub mod nanogpt;
pub mod oneshot;
pub mod partition;
pub mod pointpillar;
pub mod proving;
pub mod resnet;
pub mod unet3d;
pub mod verfcnn_vgg;
pub mod vgg;
pub mod whisper;
pub mod yolo;

pub use builder::DagBuilder;
pub use partition::{partition_dag, edge_partition_map, PartitionDesc};

use std::collections::HashSet;
use std::sync::Arc;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use almost_goldilocks_cuda::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};

use crate::poly::{DenseMLPoly, DeviceDenseMLPoly, MLPoly, SparseMLPoly};
use crate::util::arith::get_n;

pub type NodeId = usize;
pub type EdgeId = usize;

/// Stable identifier for a (consumer, slot) ↔ edge alias, used when a single
/// producer edge fans out to multiple consumers.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AliasId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    /// Per-input, per-proof auxiliary (range bit decomp, selection polys,
    /// activation tensors).
    Auxiliary,
    /// Input-independent: weights, biases, fixed lookup tables. Committed in
    /// the offline phase (§4.1 of the plan).
    Constant,
    /// External input to the DAG (e.g. embeddings, mel spectrogram).
    Input,
    /// Terminal output of the DAG.
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Uint,
    Int,
    Bool,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolyType {
    Dense,
    Sparse,
}

// ============================================================================
// Witness
// ============================================================================

/// A multilinear polynomial witness with the metadata the prover / verifier
/// need to interpret it.
///
/// `data` is `Option` because the witness can be cleared once it's no longer
/// needed (commit step keeps committed edges; everything else can be freed
/// during the backward pass to reclaim memory).
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
    /// Build from a flat host-side eval table. The table is padded with
    /// zeros to `2^get_n(shape)` if shorter.
    pub fn new(
        shape: Vec<usize>,
        data: Vec<AlmostGoldilocksField>,
        data_type: DataType,
        sf: usize,
        role: Role,
    ) -> Self {
        let n = get_n(&shape);
        let target = 1usize << n;
        let padded = if data.len() < target {
            let mut d = data.clone();
            d.resize(target, AlmostGoldilocksField(0));
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

    pub fn new_sparse(
        shape: Vec<usize>,
        data: SparseMLPoly,
        data_type: DataType,
        sf: usize,
        role: Role,
    ) -> Self {
        Self {
            shape,
            data: Some(Box::new(data)),
            poly_type: PolyType::Sparse,
            data_type,
            sf,
            role,
        }
    }

    pub fn new_dense_poly(
        shape: Vec<usize>,
        poly: DenseMLPoly,
        data_type: DataType,
        sf: usize,
        role: Role,
    ) -> Self {
        Self {
            shape,
            data: Some(Box::new(poly)),
            poly_type: PolyType::Dense,
            data_type,
            sf,
            role,
        }
    }

    /// Build from a device-resident `u64` buffer. Reads via `evaluations_ref`
    /// etc. lazily trigger a host download — the buffer is held by `Arc` so
    /// multi-consumer fan-out is cheap.
    pub fn new_device(
        shape: Vec<usize>,
        buf: Arc<DeviceBuffer<u64>>,
        data_type: DataType,
        sf: usize,
        role: Role,
    ) -> Self {
        let n = get_n(&shape);
        assert_eq!(
            buf.len(),
            1usize << n,
            "device buffer size {} does not match shape padding 1<<{}",
            buf.len(),
            n
        );
        Self {
            shape,
            data: Some(Box::new(DeviceDenseMLPoly::from_device(n, buf))),
            poly_type: PolyType::Dense,
            data_type,
            sf,
            role,
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

    /// True iff `data` is a [`DeviceDenseMLPoly`].
    pub fn is_device_resident(&self) -> bool {
        self.data
            .as_ref()
            .map(|d| d.as_any().downcast_ref::<DeviceDenseMLPoly>().is_some())
            .unwrap_or(false)
    }

    /// Borrow the device buffer if the witness is device-resident.
    pub fn device_buf(&self) -> Option<Arc<DeviceBuffer<u64>>> {
        self.data
            .as_ref()
            .and_then(|d| d.as_any().downcast_ref::<DeviceDenseMLPoly>())
            .map(|d| Arc::clone(&d.buf))
    }

    /// Materialize a device buffer for GPU consumers, uploading from host if
    /// the witness is not already device-resident. Cloning the returned `Arc`
    /// is cheap, so call sites can share buffers across kernel launches.
    pub fn as_device_buf(&self) -> Arc<DeviceBuffer<u64>> {
        if let Some(buf) = self.device_buf() {
            return buf;
        }
        let evals = self
            .data
            .as_ref()
            .expect("witness has no data")
            .evaluations_ref();
        let raw: Vec<u64> = evals.iter().map(|f| f.reduce().0).collect();
        Arc::new(DeviceBuffer::<u64>::from_slice(&raw).expect("host->device upload failed"))
    }

    /// Download a device-resident witness to host and drop the device buffer.
    /// No-op for host-resident witnesses. After eviction the witness behaves
    /// exactly like a `DenseMLPoly`-backed one (same evaluation API, no GPU
    /// footprint).
    pub fn evict_device_buffer(&mut self) {
        if !self.is_device_resident() {
            return;
        }
        let (n, evals) = {
            let any = self.data.as_mut().unwrap().as_any_mut();
            let dp = any
                .downcast_mut::<DeviceDenseMLPoly>()
                .expect("checked is_device_resident");
            (dp.n, dp.take_host_evals())
        };
        self.data = Some(Box::new(DenseMLPoly::new(n, evals)));
    }

    /// Zero the MLE padding region (indices ≥ shape bounds along any axis).
    /// Some broadcasts (e.g. Add on non-power-of-2 shapes) leave nonzero
    /// values past the logical end of the tensor — those must be cleared
    /// before commit, or the committed poly disagrees with the logical
    /// tensor.
    pub fn zero_pad_if_needed(&mut self) {
        let needs_fix = self.shape.iter().any(|&s| s != s.next_power_of_two());
        if !needs_fix {
            return;
        }
        let data = match self.data.as_mut() {
            Some(d) => d,
            None => return,
        };
        let ndims = self.shape.len();
        let padded: Vec<usize> = self.shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
        let total_padded: usize = padded.iter().product();
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
                *data.index_mut(flat_idx) = AlmostGoldilocksField(0);
            }
        }
    }

    /// Read a single element via multi-axis indexing (little-endian: axis 0
    /// has stride 1).
    pub fn get(&self, indices: &[usize]) -> AlmostGoldilocksField {
        let shape_next_pow: Vec<usize> =
            self.shape.iter().map(|&s| s.next_power_of_two()).collect();
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

// ============================================================================
// Claim
// ============================================================================

/// An assertion `f(point) == eval` about a committed (or in-flight) polynomial.
/// `sparse_id` indexes the sub-witness when an edge carries multiple parallel
/// `SparseMLPoly`s (e.g. lookup auxiliaries).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Claim {
    pub edge_id: EdgeId,
    pub sparse_id: usize,
    pub point: Vec<AlmostGoldilocksExt2>,
    pub eval: AlmostGoldilocksExt2,
}

// ============================================================================
// Step 4 — proof container types (backward pass + lookup + opening reducer).
// No PCS opening proofs here — those land in step 5 (fold tree).
// ============================================================================

/// Per-node proof emitted during the backward pass. `sumcheck_proofs` holds
/// the basicblock's sumcheck transcript (typically one proof; some blocks
/// emit more — e.g. the conv decomposition chain). `produced_claims` are the
/// new input-edge claims this node generated, which feed upstream consumers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeProof {
    pub sumcheck_proofs: Vec<crate::sumcheck::SumcheckProof>,
    pub produced_claims: Vec<Claim>,
}

/// Lookup proof for range / two_pow tables. We use the **z-t-2 sparse
/// selection-polynomial** form (one nonzero per row), so each "bool check"
/// is a `SparseBoolSumcheckProverExt2` run rather than a dense bit
/// decomposition.
///
/// Shape:
/// - `table_proof`: one degree-2 sumcheck binding the indexed-table value
///   to the table polynomial.
/// - `bool_proofs`: one per aux plane — verifies `s_j(x) · (s_j(x) − 1) = 0`
///   via the sparse-bool prover.
/// - `middle_claims`: prover-supplied folded values at the bool-check final
///   points; they pin the per-plane evaluation that ties back to the
///   plane's committed polynomial.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LookupProof {
    pub table_proof: crate::sumcheck::SumcheckProof,
    pub bool_proofs: Vec<crate::sumcheck::SumcheckProof>,
    pub middle_claims: Vec<Vec<AlmostGoldilocksExt2>>,
}

/// Per-edge proof: all claims accumulated for this edge plus (optionally)
/// the opening-reducer sumcheck that combines them into a single claim
/// before the fold-tree opening phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeProof {
    pub claims: Vec<Claim>,
    /// Reducer proof combining `claims.len()` claims into one. When `Some`,
    /// the last entry of `claims` is the combined claim (point + eval) and
    /// the preceding entries are the originals.
    pub opening_reducer: Option<Vec<crate::sumcheck::SumcheckProof>>,
}

impl EdgeProof {
    pub fn new() -> Self {
        Self { claims: Vec::new(), opening_reducer: None }
    }
}

impl Default for EdgeProof {
    fn default() -> Self { Self::new() }
}

/// Top-level DAG proof. Contains every artifact the verifier needs to
/// replay the sumcheck side of the protocol (step 4 boundary). The
/// fold-tree opening (step 5) is appended separately by the opening phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DagProof {
    pub node_proofs: Vec<Option<NodeProof>>,
    pub edge_proofs: Vec<EdgeProof>,
    pub range_proof: Option<LookupProof>,
    pub two_pow_proof: Option<LookupProof>,
    /// `(edge_id, point, eval)` triples that pin output-edge claims. The
    /// verifier reads these directly from the proof rather than reconstructing
    /// from witnesses; they replace the "compute output, hash, sample point"
    /// dance and let the verifier work without ever seeing the witness data.
    pub output_claims: Vec<(EdgeId, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)>,
    /// `(edge_id, point, eval)` triples for boundary edges in the
    /// multi-GPU partitioned prove path. Empty when not using
    /// `prove_partitioned`. The verifier re-derives `point` from its
    /// transcript and checks `eval` against the prover's recorded value.
    #[serde(default)]
    pub boundary_claims: Vec<(EdgeId, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)>,
}

// ============================================================================
// Node + Dag + forward pass
// ============================================================================

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: crate::basicblock::BasicBlockType,
    pub inputs: Vec<EdgeId>,
    pub outputs: Vec<EdgeId>,
}

/// The computation graph. Topology is built by [`DagBuilder::compile`]; the
/// forward pass is [`Dag::run`]. Backward-pass orchestration (prove/verify)
/// lives in steps 8–13 of the plan and ties into the Ajtai commit + fold
/// tree protocol; the structural plumbing here is shared between zk-torch-3
/// and zk-torch-4.
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
    /// All layer boundary edges recorded during model construction. Candidate
    /// split points for the partition-aware parallel proving path.
    pub layer_boundaries: Vec<EdgeId>,
    /// Active partition boundary edges (subset of `layer_boundaries`).
    pub boundary_edges: Vec<EdgeId>,
    /// EVERY edge read across a partition cut, derived from `boundary_edges`
    /// by `set_partition_boundaries`. A strict superset of `boundary_edges`
    /// on branchy graphs; equal to it on a clean stack. Each of these must be
    /// committed — see `should_commit`.
    pub crossing_edges: Vec<EdgeId>,
    /// Output edges that need self-claims (Conv2D / Conv3D outputs).
    pub self_claim_edges: HashSet<EdgeId>,
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

    /// Select `num_partitions - 1` evenly-spaced boundaries from
    /// `layer_boundaries`. Mirrors zk-torch-3.
    pub fn set_partition_boundaries(&mut self, num_partitions: usize) {
        assert!(num_partitions >= 1, "need at least 1 partition");
        if num_partitions == 1 || self.layer_boundaries.is_empty() {
            self.boundary_edges = vec![];
            self.crossing_edges = vec![];
            return;
        }
        // A graph cannot be cut more finely than it has layer boundaries, and
        // that is a property of the MODEL, not a caller error: PointPillars
        // has 6, so it tops out at 7 partitions however many GPUs are present.
        // Panicking made an 8-GPU run abort outright on such a model, which
        // turns "this model parallelizes less" into "this model does not run".
        // Clamp instead, and say so on stderr -- the caller prints the
        // effective partition count beside the boundary count, so the reduction
        // is visible in the log rather than silent.
        let num_candidates = self.layer_boundaries.len();
        let num_partitions = if num_partitions - 1 > num_candidates {
            eprintln!(
                "[partition] {} partitions requested but only {} layer \
                 boundaries exist; clamping to {}",
                num_partitions, num_candidates, num_candidates + 1
            );
            num_candidates + 1
        } else {
            num_partitions
        };
        let num_boundaries = num_partitions - 1;
        let mut selected = Vec::with_capacity(num_boundaries);
        for i in 0..num_boundaries {
            let idx = ((i + 1) * num_candidates) / num_partitions;
            let idx = idx.min(num_candidates - 1);
            selected.push(self.layer_boundaries[idx]);
        }
        self.boundary_edges = selected;
        // Derive the FULL crossing set. Commitment must cover every edge read
        // across a cut, not only the designated ones: the producing partition
        // proves the seeded boundary claim, while the consuming partition
        // emits its own claim on the same edge at a different point, and the
        // shared commitment is what forces both claims to be about the same
        // tensor. Without it a prover can use one tensor on the producing side
        // and another on the consuming side. On a clean stack the two sets
        // coincide; residual connections, rescale nodes, and encoder-decoder
        // cross-attention make the crossing set strictly larger.
        let designated = self.boundary_edges.clone();
        let parts = crate::dag::partition::partition_dag(self, &designated);
        self.crossing_edges = crate::dag::partition::cross_partition_edges(&parts);
    }

    /// Download every device-resident witness to host and drop the GPU
    /// buffers. Idempotent; safe to call after the GPU forward pass when we
    /// want to free HBM before the prove phase reads from host.
    pub fn evict_device_witnesses(&self, witnesses: &mut [Vec<Witness>]) {
        for ws in witnesses.iter_mut() {
            for w in ws.iter_mut() {
                if w.is_device_resident() {
                    w.evict_device_buffer();
                }
            }
        }
    }

    /// True iff the witness on `edge_id` is committed in zk-torch-4's
    /// two-phase commit scheme. Every edge read across a partition cut, and
    /// Conv self-claims, are always committed; `Role::Constant` is the
    /// offline phase (per-model);
    /// `Role::Auxiliary` / `Role::Input` are the online phase (per-input);
    /// `Role::Output` is committed only when there are no consumers (final
    /// claim binding).
    pub fn should_commit(&self, witness: &Witness, edge_id: EdgeId) -> bool {
        // `crossing_edges` already contains every designated cut, but check
        // both so a caller that sets `boundary_edges` directly still commits
        // its cuts.
        if self.boundary_edges.contains(&edge_id)
            || self.crossing_edges.contains(&edge_id)
        {
            return true;
        }
        if self.self_claim_edges.contains(&edge_id) {
            return true;
        }
        match witness.role {
            Role::Constant | Role::Auxiliary | Role::Input => true,
            // Terminal outputs are PUBLIC (their eval is recorded in
            // `output_claims` and absorbed into the transcript). Committing
            // them with Ajtai's signed-b-bit decomposition isn't safe when
            // post-pipeline values exceed 2^(b-1) — common for un-rescaled
            // network heads (ResNet 1L FC output ~2.7 GHz). zk-torch-3
            // committed these via basefold which has no signed-range
            // constraint; zk-torch-4's b-bit Ajtai does, so skip them.
            Role::Output => false,
        }
    }

    /// Forward pass: populate every witness in `witnesses`. CPU/GPU dispatch
    /// per node is driven by the `ZKT_RUN_BACKEND` env var (`"cpu"` default,
    /// `"gpu"` to dispatch through each block's `run_gpu` override).
    /// Print the lookup-auxiliary arity profile, then the cost of each
    /// candidate `table_commit_log`. Shapes are all this needs, so it runs in
    /// milliseconds at any model size — see `ZK4_SHAPE_REPORT` in [`Dag::run`].
    ///
    /// Each sparse aux is split into `ceil(table_size_log / table_commit_log)`
    /// chunks, each committed at arity `input_n + table_commit_log`. Two
    /// regimes follow, and they pull in OPPOSITE directions:
    ///
    ///  - `max_input_n + tcl <= 24`: the top bucket clears
    ///    `ZK4_GPU_SP_MAX_ARITY`, so same-point runs on GPU. Pick the LARGEST
    ///    tcl that still fits, to hold the chunk count down. Measured optimum
    ///    on llama2 8L/seq64 (max_input_n = 18) is exactly this value, 6.
    ///  - `max_input_n >= 24`: no tcl clears the cap, so same-point is CPU
    ///    whichever we pick, and the two costs trade off directly. Fold-tree
    ///    work goes as `chunks * 2^arity = (tsl/tcl) * 2^(max_input_n + tcl)`,
    ///    where the `2^tcl` factor beats the `1/tcl` one — so a SMALLER tcl
    ///    cuts fold-tree cost, roughly halving it per step down. The
    ///    range-lookup side moves the other way, since cost tracks the chunk
    ///    count. On llama2 8L/seq64, tcl 6 -> 4 took the fold tree 14.7s ->
    ///    12.2s while range went 13.6s -> 19.7s. No closed form covers that,
    ///    so this regime is reported as a range of candidates to measure
    ///    rather than a single suggestion.
    ///
    /// Note the degenerate end: at `tcl >= table_num_vars` the split is
    /// skipped entirely (see the `NO_SPARSE_SPLIT` block) and the aux keeps its
    /// native arity `input_n + table_num_vars`. That is one chunk at a wildly
    /// larger arity, not a cheap single chunk, so it is never the answer at
    /// full model sizes.
    ///
    /// This is why the optimum is not monotone in model size, and why a value
    /// tuned on a small configuration can be wrong at full scale.
    pub fn report_lookup_arities(&self, witnesses: &[Vec<Witness>]) {
        const GPU_SP_CAP: usize = 24;
        let arity_of = |edge: EdgeId| -> Option<usize> {
            witnesses.get(edge)?.first().map(|w| crate::util::arith::get_n(&w.shape))
        };
        let mut collect = |nodes: &[NodeId]| -> Vec<usize> {
            nodes.iter()
                .filter_map(|&n| self.nodes[n].inputs.first().copied())
                .filter_map(&arity_of)
                .collect()
        };
        let range = collect(&self.range);
        let two_pow = collect(&self.two_pow);

        let mut hist: std::collections::BTreeMap<usize, usize> = Default::default();
        for &a in range.iter().chain(two_pow.iter()) {
            *hist.entry(a).or_insert(0) += 1;
        }
        let max_input_n = hist.keys().copied().max().unwrap_or(0);
        let tsl = *crate::TABLE_SIZE_LOG;

        println!("[shape_report] range_nodes={} two_pow_nodes={} table_size_log={} \
                  table_commit_log={} (current)",
                 range.len(), two_pow.len(), tsl, *crate::TABLE_COMMIT_LOG);
        println!("[shape_report] lookup input arities (input_n -> count): {:?}", hist);
        println!("[shape_report] max_input_n={}  gpu_same_point_cap={}", max_input_n, GPU_SP_CAP);
        // "Reachable" needs a tcl >= 1, so it means max_input_n <= 23. But
        // reachable-at-1-only is a corner: it costs table_size_log chunks.
        let reachable = max_input_n < GPU_SP_CAP;
        if reachable && max_input_n + 2 > GPU_SP_CAP {
            println!("[shape_report] REGIME: marginal — only table_commit_log = 1 keeps the                       top bucket on GPU, and that costs {} chunks. Compare it against the                       cheapest above-cap rows on total prove time.", tsl);
        } else if reachable {
            println!("[shape_report] REGIME: the cap is reachable. Prefer the LARGEST \
                      table_commit_log with top_arity <= {} (fewer chunks, still on GPU).",
                     GPU_SP_CAP);
        } else {
            println!("[shape_report] REGIME: above the cap at EVERY table_commit_log, so \
                      same-point is CPU-bound whichever we pick. Fold-tree work goes as \
                      chunks * 2^top_arity, so a smaller value cuts it (the 2^tcl factor \
                      beats 1/tcl) while the range-lookup side rises with chunk count. \
                      Measure the candidates below; there is no closed form.");
        }
        // `fold_rel` is the fold-tree work proxy, chunks * 2^top_arity, normalised
        // to the cheapest row so the columns can be compared at a glance. The
        // range-lookup side tracks `chunks` and pulls the other way.
        println!("[shape_report] {:>4}  {:>9}  {:>7}  {:>9}  {:>9}",
                 "tcl", "top_arity", "chunks", "gpu_sp", "fold_rel");
        // From 1: a model with max_input_n = 23 (Llama-2-7B at seq512) clears the
        // cap ONLY at tcl = 1, and excluding it would misreport that model as
        // having no GPU-eligible setting at all. It is usually still the wrong
        // pick — 1 maximises the chunk count — but that is for the chunks and
        // fold_rel columns to show, not for the candidate list to hide.
        let cand: Vec<usize> = (1..=tsl.min(16)).collect();
        let work = |tcl: usize| -> f64 {
            let chunks = ((tsl + tcl - 1) / tcl) as f64;
            // At tcl >= table_num_vars the split is skipped and the aux keeps its
            // native arity, so model that explicitly rather than pretending the
            // single chunk is cheap.
            let top = if tcl >= tsl { max_input_n + tsl } else { max_input_n + tcl };
            chunks * (top as f64).exp2()
        };
        let min_work = cand.iter().map(|&t| work(t)).fold(f64::INFINITY, f64::min);
        for &tcl in &cand {
            let split = tcl < tsl;
            let top = if split { max_input_n + tcl } else { max_input_n + tsl };
            let chunks = (tsl + tcl - 1) / tcl;
            println!("[shape_report] {:>4}  {:>9}  {:>7}  {:>9}  {:>8.1}x{}",
                     tcl, top, chunks,
                     if top <= GPU_SP_CAP { "yes" } else { "NO (cpu)" },
                     work(tcl) / min_work,
                     if split { "" } else { "  <- no split, native arity" });
        }
        match cand.iter().copied().filter(|&t| max_input_n + t <= GPU_SP_CAP).max() {
            Some(t) => println!("[shape_report] SUGGESTED table_commit_log = {} \
                                 (largest that keeps the top bucket on GPU)", t),
            None => {
                let lo = cand.first().copied().unwrap_or(2);
                let by_work = cand.iter().copied()
                    .min_by(|&a, &b| work(a).partial_cmp(&work(b)).unwrap()).unwrap_or(lo);
                println!("[shape_report] NO SUGGESTION: cap unreachable. Fold-tree work is \
                          minimised at table_commit_log = {}, but that maximises chunks and \
                          so the range-lookup cost. Measure {}..{} and pick on total prove \
                          time.", by_work, lo, (by_work + 4).min(tsl.min(16)));
            }
        }
    }

    pub fn run(&self, witnesses: &mut [Vec<Witness>], feed: &[(EdgeId, Witness)]) {
        use crate::basicblock::BasicBlock;
        use rayon::prelude::*;

        assert_eq!(witnesses.len(), self.num_edges);
        for (eid, t) in feed {
            witnesses[*eid] = vec![t.clone()];
        }

        // `ZK4_SHAPE_REPORT=1` prints the lookup-auxiliary arity profile and
        // exits before the forward pass. Choosing `table_commit_log` needs one
        // number — the largest range/two_pow INPUT arity — because each aux
        // chunk commits at `input_n + table_commit_log` and the fold tree only
        // runs GPU same-point at arity <= 24. That number depends on shapes
        // alone, so there is no reason to pay for a forward pass and a proof to
        // learn it: at full model sizes those cost tens of minutes and hundreds
        // of GB, which makes a per-model sweep impractical.
        if std::env::var("ZK4_SHAPE_REPORT").is_ok() {
            self.report_lookup_arities(witnesses);
            std::process::exit(0);
        }

        let backend = std::env::var("ZKT_RUN_BACKEND").unwrap_or_else(|_| "cpu".to_string());
        let use_gpu = backend.eq_ignore_ascii_case("gpu");

        if use_gpu {
            // Multi-GPU forward pass. Within each topological level the nodes
            // are mutually independent, so we round-robin them across all
            // visible devices and run their GPU kernels concurrently (one
            // rayon task per node, each pinning its assigned device). `run_gpu`
            // takes host `Witness` refs and returns host `Witness`es, so data
            // flows producer→consumer through the host and a node may sit on a
            // different device than the node that produced its input — no
            // cross-device copy needed. A 1-node level (the common case in a
            // sequential CNN chain) still runs on the GPU, just on one device.
            // `run_gpu` returns DEVICE-resident witnesses (lazy host download).
            // We materialize each dense output to host on the device that
            // produced it — BEFORE another node (possibly on a different
            // device) consumes it, and before we leave that device's context.
            // This (a) makes cross-device producer→consumer correct (host is
            // the rendezvous) and (b) frees the op's device buffers right away
            // so device memory stays bounded to the working set instead of
            // accumulating every layer's output (the real OOM wall on big nets).
            fn to_host(mut w: Witness) -> Witness {
                if matches!(w.poly_type, PolyType::Dense) {
                    if let Some(d) = w.data.as_ref() {
                        let n = d.n();
                        let evals = d.evaluations(); // downloads on the current device
                        w.data = Some(Box::new(crate::poly::DenseMLPoly::new(n, evals)));
                    }
                }
                w
            }
            let devices = crate::fold::tree::gpu_device_pool();
            let ndev = devices.len().max(1);
            for level in &self.topo_levels {
                if level.len() == 1 || ndev == 1 {
                    for (i, &nid) in level.iter().enumerate() {
                        let _ = almost_goldilocks_cuda::set_device(devices[i % ndev]);
                        let node = &self.nodes[nid];
                        let in_refs: Vec<&Witness> =
                            node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                        // ZK4_FWD_TRACE=1 names every node as it runs. Every
                        // output is downloaded to host immediately below, so a
                        // CUDA fault surfaces in THAT download -- meaning the
                        // last node printed is the one that corrupted the
                        // context. Without this the panic names dense.rs and
                        // says nothing about which operator was responsible.
                        let trace = std::env::var("ZK4_FWD_TRACE").is_ok();
                        if trace {
                            eprintln!("[fwd] node {} {:?} in={:?}", nid, node.kind,
                                      in_refs.iter().map(|w| w.shape.clone())
                                             .collect::<Vec<_>>());
                        }
                        let outs = node.kind.run_gpu(&in_refs);
                        if trace {
                            eprintln!("[fwd] node {} ok out={:?}", nid,
                                      outs.iter().map(|w| w.shape.clone())
                                          .collect::<Vec<_>>());
                        }
                        assert_eq!(outs.len(), node.outputs.len(), "op arity mismatch");
                        // `ZK4_GPU_FWD_DBG=1` cross-checks every GPU op against
                        // its CPU counterpart. A `run_gpu` that returns a
                        // different SHAPE than `run` (same arity, padded
                        // extents) silently desynchronizes downstream claim
                        // construction, which is hard to trace from the
                        // eventual panic. Diagnostic only: it runs both paths.
                        if std::env::var("ZK4_GPU_FWD_DBG").is_ok() {
                            let cpu = node.kind.run(&in_refs);
                            for (o, c) in outs.iter().zip(cpu.iter()) {
                                if o.shape != c.shape {
                                    eprintln!("[gpu_fwd] SHAPE MISMATCH {:?}: gpu={:?} cpu={:?}",
                                              node.kind, o.shape, c.shape);
                                }
                            }
                        }
                        for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                            witnesses[eid] = vec![to_host(out)];
                        }
                    }
                } else {
                    let devices_ref = &devices;
                    let results: Vec<(NodeId, Vec<Witness>)> = level
                        .par_iter()
                        .enumerate()
                        .map(|(i, &nid)| {
                            let _ = almost_goldilocks_cuda::set_device(devices_ref[i % ndev]);
                            let node = &self.nodes[nid];
                            let in_refs: Vec<&Witness> =
                                node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                            // Materialize to host on THIS device before the
                            // result escapes the task (and its device context).
                            let outs: Vec<Witness> =
                                node.kind.run_gpu(&in_refs).into_iter().map(to_host).collect();
                            (nid, outs)
                        })
                        .collect();
                    for (nid, outs) in results {
                        let node = &self.nodes[nid];
                        assert_eq!(outs.len(), node.outputs.len(), "op arity mismatch");
                        for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                            witnesses[eid] = vec![out];
                        }
                    }
                }
            }
            // The caching allocator retains the forward pass's freed blocks in
            // each device's pool; release them so the prove phase (which is the
            // memory-heavy consumer) can allocate. Without this the first
            // prove OOMs right after a GPU forward on a large model.
            for &dev in &devices {
                let _ = almost_goldilocks_cuda::set_device(dev);
                let _ = almost_goldilocks_cuda::pool_trim(0);
            }
            let _ = almost_goldilocks_cuda::set_device(devices[0]);
        } else {
            let prof = std::env::var("ZKT_RUN_PROFILE").ok().as_deref() == Some("1");
            let mut prof_map: std::collections::BTreeMap<String, (std::time::Duration, usize)> =
                std::collections::BTreeMap::new();
            for level in &self.topo_levels {
                if level.len() == 1 {
                    let nid = level[0];
                    let node = &self.nodes[nid];
                    let in_refs: Vec<&Witness> =
                        node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                    let t0 = std::time::Instant::now();
                    let outs = node.kind.run(&in_refs);
                    if prof {
                        let ty = format!("{:?}", node.kind);
                        let ty = ty.split(['(', ' ', '{']).next().unwrap_or("?").to_string();
                        let e = prof_map.entry(ty).or_insert((std::time::Duration::ZERO, 0));
                        e.0 += t0.elapsed(); e.1 += 1;
                    }
                    assert_eq!(outs.len(), node.outputs.len(), "op arity mismatch");
                    for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                        witnesses[eid] = vec![out];
                    }
                } else {
                    let results: Vec<(NodeId, Vec<Witness>, String, std::time::Duration)> = level
                        .par_iter()
                        .map(|&nid| {
                            let node = &self.nodes[nid];
                            let in_refs: Vec<&Witness> =
                                node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
                            let t0 = std::time::Instant::now();
                            let outs = node.kind.run(&in_refs);
                            let ty = if prof {
                                let s = format!("{:?}", node.kind);
                                s.split(['(', ' ', '{']).next().unwrap_or("?").to_string()
                            } else { String::new() };
                            (nid, outs, ty, t0.elapsed())
                        })
                        .collect();
                    for (nid, outs, ty, d) in results {
                        if prof {
                            let e = prof_map.entry(ty).or_insert((std::time::Duration::ZERO, 0));
                            e.0 += d; e.1 += 1;
                        }
                        let node = &self.nodes[nid];
                        assert_eq!(outs.len(), node.outputs.len(), "op arity mismatch");
                        for (&eid, out) in node.outputs.iter().zip(outs.into_iter()) {
                            witnesses[eid] = vec![out];
                        }
                    }
                }
            }
            if prof {
                let mut v: Vec<_> = prof_map.into_iter().collect();
                v.sort_by_key(|(_, (d, _))| std::cmp::Reverse(*d));
                for (ty, (d, cnt)) in v {
                    eprintln!("[run_profile] {:>22} {:>8.1}ms ({} ops)", ty, d.as_secs_f64() * 1000.0, cnt);
                }
            }
        }

        // Post-process sparse witnesses: split each into K chunks of
        // `TABLE_COMMIT_LOG` table-index bits (zk-torch-2 style). Caps
        // every aux's arity at `input_n + TABLE_COMMIT_LOG`, even when
        // the underlying selection had a larger `table_num_vars`. This
        // is the architectural fix for the fold-tree's large-arity
        // buckets — see plan §5.5.
        if std::env::var("NO_SPARSE_SPLIT").is_err() {
            let block_size = *crate::TABLE_COMMIT_LOG;
            for w_vec in witnesses.iter_mut() {
                if w_vec.is_empty() { continue; }
                let w0 = &w_vec[0];
                if w0.poly_type != PolyType::Sparse { continue; }
                if w0.data.is_none() { continue; }
                let sp = match w0.data.as_ref().unwrap().as_any().downcast_ref::<SparseMLPoly>() {
                    Some(s) => s,
                    None => continue,
                };
                if sp.selection.table_num_vars <= block_size { continue; }
                let chunks = sp.split_table_index_into_blocks(block_size);
                let new_witnesses: Vec<Witness> = chunks.into_iter().map(|chunk_sp| {
                    Witness::new_sparse(w0.shape.clone(), chunk_sp,
                                        w0.data_type, w0.sf, w0.role)
                }).collect();
                *w_vec = new_witnesses;
            }
        }
    }

    /// Commit every `Role::Constant` edge into `store` — the offline phase
    /// of plan §4.1. Idempotent: edges that already have a commitment in
    /// the store are skipped. Run this once per model, before persisting
    /// the store to disk.
    pub fn commit_constants(
        &self,
        witnesses: &[Vec<Witness>],
        store: &mut crate::commit::GpuAjtaiStore,
    ) {
        self.commit_edges(witnesses, store, /*constants_only=*/ true, None);
    }

    /// Multi-GPU variant: each edge is committed on the GPU determined by
    /// `edge_partitions[edge_id]` (modulo the device pool). Pass
    /// `edge_partition_map(&dag, &partitions)` after a partition_dag call.
    pub fn commit_constants_partitioned(
        &self,
        witnesses: &[Vec<Witness>],
        store: &mut crate::commit::GpuAjtaiStore,
        edge_partitions: &[Option<usize>],
    ) {
        self.commit_edges(witnesses, store, true, Some(edge_partitions));
    }

    /// Commit every non-`Role::Constant` edge that needs a commitment per
    /// `should_commit` — the online phase of plan §4.2. Run after
    /// `dag.run()` populates the input-dependent witnesses.
    pub fn commit_remaining(
        &self,
        witnesses: &[Vec<Witness>],
        store: &mut crate::commit::GpuAjtaiStore,
    ) {
        self.commit_edges(witnesses, store, /*constants_only=*/ false, None);
    }

    /// Multi-GPU variant of `commit_remaining`.
    pub fn commit_remaining_partitioned(
        &self,
        witnesses: &[Vec<Witness>],
        store: &mut crate::commit::GpuAjtaiStore,
        edge_partitions: &[Option<usize>],
    ) {
        self.commit_edges(witnesses, store, false, Some(edge_partitions));
    }

    /// Convenience wrapper: offline pass then online pass on the same
    /// store. Use in single-shot proving (no precommitted weights file).
    pub fn commit(
        &self,
        witnesses: &[Vec<Witness>],
        store: &mut crate::commit::GpuAjtaiStore,
    ) {
        self.commit_constants(witnesses, store);
        self.commit_remaining(witnesses, store);
    }

    fn commit_edges(
        &self,
        witnesses: &[Vec<Witness>],
        store: &mut crate::commit::GpuAjtaiStore,
        constants_only: bool,
        edge_partitions: Option<&[Option<usize>]>,
    ) {
        assert_eq!(
            witnesses.len(),
            self.num_edges,
            "witness count {} != num_edges {}",
            witnesses.len(),
            self.num_edges,
        );
        assert_eq!(
            store.num_edges(),
            self.num_edges,
            "store size {} != num_edges {}",
            store.num_edges(),
            self.num_edges,
        );
        let key = store.key;
        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");

        // Multi-GPU device pool. Round-robin by commit index (when no
        // edge_partitions) or by `edge_partitions[edge_id] % n_dev`.
        // For a single-GPU host this collapses to staying on the
        // current device — identical to the legacy sequential commit.
        let device_pool = crate::fold::tree::gpu_device_pool();
        let n_dev = device_pool.len();

        // First pass: collect commit tasks. Each task records the
        // device it will run on. We dispatch in parallel via rayon.
        struct Task {
            edge_id: usize,
            device: i32,
        }
        let mut tasks: Vec<Task> = Vec::new();
        for edge_id in 0..self.num_edges {
            if store.get(edge_id).is_some() { continue; }
            let edge_ws = &witnesses[edge_id];
            if edge_ws.is_empty() { continue; }
            let w = &edge_ws[0];
            let is_constant = w.role == Role::Constant;
            if constants_only != is_constant { continue; }
            if w.data.is_none() { continue; }
            if !self.should_commit(w, edge_id) { continue; }
            let device = match edge_partitions {
                Some(eps) => {
                    let p = eps[edge_id].unwrap_or(tasks.len());
                    device_pool[p % n_dev]
                }
                None => device_pool[tasks.len() % n_dev],
            };
            tasks.push(Task { edge_id, device });
        }

        if tasks.is_empty() {
            return;
        }

        // Phase 2: parallel commit via rayon. Each worker pins itself
        // to the task's assigned device, runs commit_witness_set_with_planes,
        // returns the result. Witnesses are shared (& only — no &mut).
        use rayon::prelude::*;
        let t_total = std::time::Instant::now();
        let results: Vec<(usize, crate::commit::EdgeCommitment, Vec<Vec<u64>>, &'static str, bool, usize, std::time::Duration)> =
            tasks.par_iter().map(|task| {
                let _ = almost_goldilocks_cuda::set_device(task.device);
                let edge_ws = &witnesses[task.edge_id];
                let w = &edge_ws[0];
                let t = std::time::Instant::now();
                let (ec, packed_planes) =
                    crate::commit::commit_witness_set_with_planes(&key, edge_ws);
                let elapsed = t.elapsed();
                let role_name = match w.role {
                    Role::Constant => "Constant",
                    Role::Auxiliary => "Auxiliary",
                    Role::Input => "Input",
                    Role::Output => "Output",
                };
                let is_sparse = ec.is_sparse;
                let n_elems = w.data.as_ref().map(|d| d.len()).unwrap_or(0);
                (task.edge_id, ec, packed_planes, role_name, is_sparse, n_elems, elapsed)
            }).collect();
        let _ = almost_goldilocks_cuda::set_device(device_pool[0]);

        // Phase 3: single-threaded write-back into store + aggregate
        // timing buckets if requested.
        let mut counts: std::collections::BTreeMap<(&'static str, bool), (usize, usize, std::time::Duration)> = std::collections::BTreeMap::new();
        for (edge_id, ec, planes, role_name, is_sparse, n_elems, elapsed) in results {
            if timing {
                let e = counts.entry((role_name, is_sparse)).or_insert((0, 0, std::time::Duration::ZERO));
                e.0 += 1;
                e.1 += n_elems;
                e.2 += elapsed;
            }
            store.set(edge_id, ec);
            store.set_planes(edge_id, planes);
        }
        if timing {
            let phase = if constants_only { "offline" } else { "online" };
            eprintln!("[commit {}] {} tasks across {} GPU(s) in {:?}:",
                phase, tasks.len(), n_dev, t_total.elapsed());
            for ((role, sp), (n, elems, dur)) in &counts {
                eprintln!("  {:>10} sparse={:>5}: {:>4} edges, {:>10.2} M elems, {:>8.1?}",
                    role, sp, n, *elems as f64 / 1e6, dur);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    #[test]
    fn witness_new_pads_to_power_of_two_index_space() {
        let w = Witness::new(
            vec![3, 5],
            (0..15u64).map(agl).collect(),
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        // shape (3, 5) → padding to (4, 8) → 32-element MLE.
        let data = w.data.as_ref().unwrap();
        assert_eq!(data.n(), 5);
        assert_eq!(data.len(), 32);
        // First 15 entries match input; rest are zero.
        for i in 0..15 {
            assert_eq!(data.index(i).0, i as u64);
        }
        for i in 15..32 {
            assert_eq!(data.index(i).0, 0);
        }
    }

    #[test]
    fn witness_get_uses_little_endian_strides() {
        // Shape [3, 5] → padded [4, 8]; little-endian strides: axis 0 stride 1,
        // axis 1 stride 4. So (2, 3) maps to flat index 2 + 3*4 = 14.
        let mut data = vec![agl(0); 32];
        data[14] = agl(42);
        let w = Witness::new(
            vec![3, 5],
            data,
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        assert_eq!(w.get(&[2, 3]).0, 42);
    }

    #[test]
    fn witness_zero_pad_if_needed_clears_oob_cells() {
        // Shape (3, 3) padded to (4, 4) → 16 entries. Fill all 16 with a
        // sentinel; zero_pad_if_needed must clear positions where any axis
        // is out of the logical (3, 3) bounds.
        let mut w = Witness::new(
            vec![3, 3],
            vec![agl(7); 16],
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        w.zero_pad_if_needed();
        let data = w.data.as_ref().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let flat = i + j * 4;
                let want = if i < 3 && j < 3 { 7 } else { 0 };
                assert_eq!(
                    data.index(flat).0,
                    want,
                    "(i={}, j={}) flat={}",
                    i,
                    j,
                    flat
                );
            }
        }
    }

    #[test]
    fn witness_zero_pad_noop_when_already_pow2() {
        let mut w = Witness::new(
            vec![4, 4],
            vec![agl(7); 16],
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        w.zero_pad_if_needed();
        let data = w.data.as_ref().unwrap();
        for i in 0..16 {
            assert_eq!(data.index(i).0, 7);
        }
    }

    #[test]
    fn witness_new_sparse_round_trip() {
        let mut evals = std::collections::HashMap::new();
        evals.insert(1, agl(5));
        evals.insert(3, agl(7));
        let sp = SparseMLPoly::new(2, evals);
        let w = Witness::new_sparse(vec![4], sp, DataType::Uint, 0, Role::Auxiliary);
        assert_eq!(w.poly_type, PolyType::Sparse);
        assert_eq!(w.data.as_ref().unwrap().n(), 2);
        assert_eq!(w.data.as_ref().unwrap().index(1), agl(5));
        assert_eq!(w.data.as_ref().unwrap().index(0), agl(0));
    }

    #[test]
    fn witness_clear_data_drops_the_poly() {
        let mut w = Witness::new(
            vec![2],
            vec![agl(1), agl(2)],
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        assert!(w.data.is_some());
        w.clear_data();
        assert!(w.data.is_none());
    }

    #[test]
    fn witness_clone_preserves_data() {
        let w = Witness::new(
            vec![4],
            (0..4u64).map(agl).collect(),
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        let c = w.clone();
        for i in 0..4 {
            assert_eq!(c.data.as_ref().unwrap().index(i).0, i as u64);
        }
    }

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    #[test]
    fn witness_device_residency_transitions() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let evals: Vec<u64> = (0..8u64).map(|i| i + 1).collect();
        let buf = Arc::new(DeviceBuffer::<u64>::from_slice(&evals).expect("upload"));
        let mut w = Witness::new_device(vec![8], buf, DataType::Int, 0, Role::Auxiliary);
        assert!(w.is_device_resident());

        // Round-trip via as_device_buf is a no-op on a device-resident witness.
        let b = w.as_device_buf();
        assert_eq!(b.len(), 8);

        w.evict_device_buffer();
        assert!(!w.is_device_resident());
        for i in 0..8 {
            assert_eq!(w.data.as_ref().unwrap().index(i).0, (i + 1) as u64);
        }
    }

    #[test]
    fn witness_as_device_buf_uploads_when_host_resident() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let w = Witness::new(
            vec![4],
            (0..4u64).map(agl).collect(),
            DataType::Int,
            10,
            Role::Auxiliary,
        );
        assert!(!w.is_device_resident());
        let buf = w.as_device_buf();
        assert_eq!(buf.len(), 4);
        let downloaded = buf.to_vec().expect("download");
        assert_eq!(downloaded, vec![0u64, 1, 2, 3]);
    }
}
