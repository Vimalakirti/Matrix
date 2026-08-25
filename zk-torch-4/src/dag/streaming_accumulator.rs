//! Streaming accumulator for deferred constant openings (Phase 3 of the
//! cross-proof opening-aggregation design).
//!
//! In the multi-inference serving scenario, every proof of model `M`
//! produces a sumcheck terminal claim `(r_i, v_i)` for each pre-committed
//! weight polynomial `W`. The per-proof verifier defers those claims;
//! this module folds the stream of `(r_i, v_i)` into a single running
//! claim `(r_acc, v_acc)` per weight via the existing reducer block:
//!
//!     [eq(r_acc, x) + β · eq(r_new, x)] · W(x) = v_acc + β · v_new
//!
//! Sumcheck reduces this to a fresh random point `r_acc'` with terminal
//! claim `W(r_acc') = v_acc'`. State per weight stays constant size
//! (one [`crate::dag::Claim`]) regardless of how many proofs have
//! streamed through. No PCS opens happen here — those land in Phase 4's
//! `finalize()`.
//!
//! Soundness invariants enforced here:
//!   1. Deferred-claim absorption order is canonical (edge_id ascending
//!      within each proof — matches the per-proof prover's iteration).
//!   2. The reducer challenge `β` is sampled only AFTER both the prior
//!      accumulated claim and the incoming claim are in the transcript.
//!   3. The accumulator only combines claims for the same `edge_id` —
//!      cross-edge combination would mix different polynomials.

use std::collections::HashMap;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use serde::{Deserialize, Serialize};

use crate::basicblock::reducer::Reducer;
use crate::basicblock::BasicBlock;
use crate::commit::GpuAjtaiStore;
use crate::dag::fold_integration::{
    decompose_witness_for_fold_native, eval_binary_with_shared_eq, DeferredClaim, DeferredResult,
    EdgePlaneEvals, reconstruct_signed_two_complement,
};
use crate::dag::{Claim, EdgeId, Witness};
use crate::fold::{prove_fold_tree, verify_fold_tree, FoldData, FoldInstance, FoldTreeProof};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, ext2_sub};
use almost_goldilocks_cuda::extension::AlmostExt2Batch;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use almost_goldilocks_cuda::memory::DeviceBuffer;

/// One reducer-block update in the streaming sequence. Each step
/// consumes two claims (the prior accumulated state + the new incoming
/// deferred claim) for the same edge and produces one new accumulated
/// claim. The sumcheck proof binds the new claim to the same polynomial
/// as both inputs.
///
/// The new claim's `point` is the sumcheck transcript challenges, which
/// the verifier reconstructs during replay — so it isn't stored here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReducerStep {
    pub edge_id: EdgeId,
    pub sumcheck_proof: SumcheckProof,
    /// `W(r_acc')` — the new accumulated claim's evaluation at the new
    /// random point. Equals `prover.final_eval(0)` from the reducer's
    /// sumcheck. Stored so the verifier can construct the next-state
    /// claim without re-running the prover's final-eval computation.
    pub new_eval: AlmostGoldilocksExt2,
}

/// Streaming-mode accumulator over a sequence of deferred-constant
/// proofs for the same model. Maintains one running [`Claim`] per
/// `Role::Constant` edge; each `add_proof` call updates the running
/// claims via reducer sumchecks (no PCS opens). Owned and mutated by
/// the prover side; the verifier maintains its own mirror (Phase 4).
pub struct AccumulatorState {
    /// Current running claim per Constant edge.
    pub claims: HashMap<EdgeId, Claim>,
    /// Sequence of reducer-step proofs, ordered by (proof index, then
    /// edge_id ascending within proof). The streaming verifier replays
    /// this in the same order to match the prover's transcript.
    pub steps: Vec<ReducerStep>,
    /// Internal transcript, advanced on every `add_proof`. Seeded by
    /// the caller-supplied label so the verifier-side mirror can be
    /// initialized identically.
    transcript: Transcript,
    /// Per-edge cache of the Ext2-lifted witness on GPU. Constants
    /// don't change across the stream, so the `from_base` lift runs
    /// once per edge — saves ~55 % of each reducer-step's wall time
    /// at n=24 (vs naive re-lifting per call). Cache is opt-in: set
    /// `ZK4_STREAM_X_EXT2_CACHE=1` to enable.
    ///
    /// In multi-GPU mode, each buffer lives on the edge's assigned
    /// device (`edge_device[edge_id]`). Callers MUST `set_device`
    /// before touching a cached buffer.
    x_ext2_cache: HashMap<EdgeId, DeviceBuffer<u64>>,
    /// Per-edge cache of the reducer's d_acc buffer. Reused across
    /// stream iterations (zero-init'd via `cudaMemset` per call —
    /// fast). Skips the per-call `cudaMalloc(size*2)`, which is one
    /// of the synchronous CUDA ops responsible for sequentializing
    /// the accumulator. Cache is opt-in (same env flag as x_ext2).
    /// Sharded across GPUs by `edge_device` (see above).
    d_acc_cache: HashMap<EdgeId, DeviceBuffer<u64>>,
    /// Multi-GPU sharding: each Constant edge is pinned to a GPU at
    /// first sighting. All future cache allocations + reducer calls
    /// for that edge run on its assigned device. Lets the streaming
    /// reducer scale beyond one GPU's memory budget (full Llama-2
    /// at 32 layers needs ~160 GB of cache on a single GPU; 4-way
    /// sharded it's ~40 GB / GPU, comfortable). Also gives true
    /// per-device parallelism — separate CUDA contexts don't
    /// serialize on the driver-level sync ops that bit method #3
    /// (single-GPU rayon parallelism). Assignment is round-robin
    /// by edge_id over the configured device pool (see
    /// `gpu_device_pool()` in `fold/tree.rs` — uses
    /// `ZK4_GPU_DEVICES=0,2,3` if set, else all visible GPUs).
    edge_device: HashMap<EdgeId, i32>,
}

fn x_ext2_cache_enabled() -> bool {
    std::env::var("ZK4_STREAM_X_EXT2_CACHE").ok().as_deref() == Some("1")
}

/// Factored-eq (Gruen) reducer sumcheck. Default ON when the x_ext2 cache
/// is enabled: it never materializes the two per-claim eq tables (the old
/// path built 2×`2^n` eq tables + accumulated them per edge per proof —
/// ~0.8 s eq-build + the now-removed copy). The factored state keeps
/// precomputed suffix stages + host prefix scalars and folds only the
/// shared witness. Byte-identical proof, so the verifier is unchanged.
/// Set `ZK4_REDUCER_FACTORED=0` to fall back to the materialized path.
fn factored_reducer_enabled() -> bool {
    std::env::var("ZK4_REDUCER_FACTORED").ok().as_deref() != Some("0")
}

fn parallel_reducer_enabled() -> bool {
    // Default OFF: the parallel rayon-driven reducer is sound (per-edge
    // forked transcripts + canonical-order absorb-back into parent) but
    // it's NOT faster than the sequential version on the prover side.
    // The synchronous CUDA driver ops in each reducer call
    // (cudaMalloc + cudaMemcpy htod for the d_acc zero-init) serialize
    // across rayon worker threads — per-thread default streams only
    // parallelize kernel LAUNCHES, not driver-level sync ops. Measured
    // on Llama-2 N=5 (3-run mean): parallel 5919 ms vs sequential 5558
    // ms (parallel ~6 % SLOWER due to forked-transcript + thread-pool
    // overhead). Verifier side does benefit (~4× on acc-verify), but
    // that's a small absolute saving. Keep the parallel path opt-in for
    // experimentation. Enable with `ZK4_STREAM_PARALLEL_REDUCER=1`.
    std::env::var("ZK4_STREAM_PARALLEL_REDUCER").ok().as_deref() == Some("1")
}

/// Cap on the number of concurrent reducer sumchecks. Default 4: the
/// prover-side gain is modest (within noise) because per-call
/// synchronous CUDA ops (`cudaMalloc`, `cudaMemcpy(htod)` for d_acc
/// zero-init) serialize across threads regardless of per-thread default
/// stream. The verifier-side gain IS real (~4× on Llama acc-verify)
/// because verify is mostly CPU+small-kernel work. Override with
/// `ZK4_STREAM_PARALLELISM=N`. Use 1 to fall back to fully sequential.
fn parallel_reducer_threads() -> usize {
    std::env::var("ZK4_STREAM_PARALLELISM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .max(1)
}

/// Lazily-initialized thread pool sized by `parallel_reducer_threads()`.
/// Sized once on first use to match the env var at that moment; later
/// changes to `ZK4_STREAM_PARALLELISM` are ignored within a process.
fn reducer_thread_pool() -> &'static rayon::ThreadPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = parallel_reducer_threads();
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("zk4-reducer-{}", i))
            .build()
            .expect("failed to build reducer thread pool")
    })
}

/// Sequential reducer job. Uses BOTH the x_ext2 cache (Method #1's
/// premise — skip per-call witness lift) and the d_acc cache (Method
/// #2 — skip per-call `cudaMalloc(2·size)` for the eq-accumulator).
/// Falls back to the non-cached `Reducer::prove` if `use_cache` is off.
fn run_one_reducer_job_seq(
    edge_id: EdgeId,
    prior: Claim,
    incoming: Claim,
    transcript: &mut Transcript,
    witnesses: &[Vec<Witness>],
    x_ext2_cache: &HashMap<EdgeId, DeviceBuffer<u64>>,
    d_acc_cache: &mut HashMap<EdgeId, DeviceBuffer<u64>>,
    use_cache: bool,
) -> (EdgeId, ReducerStep, Claim) {
    let w = &witnesses[edge_id][0];
    let edge_ids = &[edge_id];
    let out_claims: &[&Claim] = &[&prior, &incoming];
    let (mut sumcheck_proofs, mut new_claims) = if use_cache {
        let cached_x = x_ext2_cache
            .get(&edge_id)
            .expect("sequential reducer: x_ext2 cache miss");
        let n = w.data.as_ref().unwrap().n();
        if factored_reducer_enabled() {
            Reducer.prove_with_cached_buffers_factored(
                cached_x, n, edge_ids, out_claims, transcript,
            )
        } else {
            let cached_d_acc = d_acc_cache
                .get_mut(&edge_id)
                .expect("sequential reducer: d_acc cache miss");
            Reducer.prove_with_cached_buffers(
                cached_x, cached_d_acc, n, edge_ids, out_claims, transcript,
            )
        }
    } else {
        Reducer.prove(&[w], edge_ids, out_claims, transcript)
    };
    let new_claim = new_claims.pop().expect("reducer returned no claim");
    let sumcheck_proof = sumcheck_proofs
        .pop()
        .expect("reducer returned no proof");
    let step = ReducerStep {
        edge_id,
        sumcheck_proof,
        new_eval: new_claim.eval,
    };
    (edge_id, step, new_claim)
}

/// Run one reducer K=2 sumcheck for a single weight `edge_id`. Returns
/// the reducer step (to append to the streaming proof) and the new
/// accumulated `Claim` (to update state). Pure function over its inputs
/// — safe to call from multiple rayon worker threads concurrently
/// (each on its own forked `transcript`, each issuing CUDA work on its
/// own per-thread default stream).
fn run_one_reducer_job(
    edge_id: EdgeId,
    prior: Claim,
    incoming: Claim,
    transcript: &mut Transcript,
    witnesses: &[Vec<Witness>],
    cache: &HashMap<EdgeId, DeviceBuffer<u64>>,
    use_cache: bool,
) -> (EdgeId, ReducerStep, Claim) {
    let w = &witnesses[edge_id][0];
    let edge_ids = &[edge_id];
    let out_claims: &[&Claim] = &[&prior, &incoming];
    let (mut sumcheck_proofs, mut new_claims) = if use_cache {
        let cached = cache
            .get(&edge_id)
            .expect("parallel reducer: cache miss for edge already in state");
        let n = w.data.as_ref().unwrap().n();
        if factored_reducer_enabled() {
            Reducer.prove_with_cached_buffers_factored(cached, n, edge_ids, out_claims, transcript)
        } else {
            Reducer.prove_with_cached_x_ext2(cached, n, edge_ids, out_claims, transcript)
        }
    } else {
        Reducer.prove(&[w], edge_ids, out_claims, transcript)
    };
    let new_claim = new_claims.pop().expect("reducer returned no claim");
    let sumcheck_proof = sumcheck_proofs
        .pop()
        .expect("reducer returned no proof");
    let step = ReducerStep {
        edge_id,
        sumcheck_proof,
        new_eval: new_claim.eval,
    };
    (edge_id, step, new_claim)
}

impl AccumulatorState {
    /// Create a fresh accumulator. The `label` seeds the internal
    /// transcript; the verifier must use the same bytes.
    pub fn new(label: &[u8]) -> Self {
        Self {
            claims: HashMap::new(),
            steps: Vec::new(),
            transcript: Transcript::new(label),
            x_ext2_cache: HashMap::new(),
            d_acc_cache: HashMap::new(),
            edge_device: HashMap::new(),
        }
    }

    /// Assign `edge_id` to a GPU on first sighting; idempotent.
    /// Round-robin by edge_id over the configured device pool.
    fn assign_device(&mut self, edge_id: EdgeId) -> i32 {
        if let Some(&d) = self.edge_device.get(&edge_id) {
            return d;
        }
        let pool = crate::fold::tree::gpu_device_pool();
        let d = pool[edge_id % pool.len()];
        self.edge_device.insert(edge_id, d);
        d
    }

    /// Consume a per-proof [`DeferredResult`] (the caller has already
    /// confirmed `ok == true` via `verify_with_fold_tree_deferred`) and
    /// update the accumulator. `witnesses` must hold the Constant edges'
    /// data — these are the model weights, shared across all proofs.
    ///
    /// After this call, `self.claims[edge_id]` is the new running claim
    /// for every edge that appeared in `deferred.claims`. Edges seen
    /// for the first time are stored verbatim (no reducer call yet,
    /// since we need ≥ 2 claims to combine).
    pub fn add_proof(
        &mut self,
        deferred: &DeferredResult,
        witnesses: &[Vec<Witness>],
    ) -> StreamingProofChunk {
        // Step 1 — absorb every incoming deferred claim into the
        // transcript before sampling any reducer challenge. The claims
        // are already in edge_id-ascending order (the per-proof prover
        // iterates 0..num_edges). The verifier must absorb in the same
        // order on its mirror.
        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");
        let t_start = std::time::Instant::now();
        for dc in &deferred.claims {
            absorb_deferred_claim(&mut self.transcript, dc);
        }

        // Step 2a — Pre-populate the x_ext2 and d_acc caches for every
        // edge that already has a prior accumulated claim. Done
        // sequentially because the caches are shared mutable state;
        // after this point they're read-only (x_ext2) or per-edge
        // sliceable (d_acc) and safe to share across rayon workers.
        //
        // Multi-GPU sharding: each edge is pinned to a GPU at first
        // sighting (`assign_device`). The cache allocs + the witness
        // lift run on that GPU (set_device first). The witness's
        // `as_device_buf()` uploads to the CURRENT device, so the
        // host→device copy lands on the right GPU automatically.
        if x_ext2_cache_enabled() {
            for dc in &deferred.claims {
                if self.claims.contains_key(&dc.edge_id) {
                    let w = &witnesses[dc.edge_id][0];
                    let n = w.data.as_ref().unwrap().n();
                    let size = 1usize << n;
                    let device = self.assign_device(dc.edge_id);
                    let _ = almost_goldilocks_cuda::set_device(device);
                    if !self.x_ext2_cache.contains_key(&dc.edge_id) {
                        let d_x_base = w.as_device_buf();
                        let mut d_x_ext2 = DeviceBuffer::<u64>::new(size * 2)
                            .expect("Reducer cache: x_ext2 alloc failed");
                        AlmostExt2Batch::from_base(&d_x_base, &mut d_x_ext2)
                            .expect("Reducer cache: base→Ext2 failed");
                        self.x_ext2_cache.insert(dc.edge_id, d_x_ext2);
                    }
                    // The factored reducer never touches d_acc — skip the
                    // per-edge alloc (saves ~K · 2^n · 16 B of device memory,
                    // ~14 GB for full Llama-2 weights, and cold-setup time).
                    if !factored_reducer_enabled() && !self.d_acc_cache.contains_key(&dc.edge_id) {
                        let d_acc = DeviceBuffer::<u64>::new(size * 2)
                            .expect("Reducer cache: d_acc alloc failed");
                        self.d_acc_cache.insert(dc.edge_id, d_acc);
                    }
                }
            }
            // Reset to device 0 so subsequent non-cached code paths see
            // a deterministic starting state.
            let _ = almost_goldilocks_cuda::set_device(0);
        }

        // Step 2b — Build job list. For each deferred claim whose edge
        // already has a prior accumulated claim, prepare an independent
        // reducer job with its own FORKED transcript. The fork is
        // deterministic (`fork(b"stream_reducer", edge_id)` clones state +
        // absorbs the partition tag), and parent transcript state is
        // unchanged — so the parent stays in sync with the verifier's
        // parent transcript.
        //
        // First-sighting claims (no prior state) are stored verbatim
        // and don't need any reducer work.
        let mut first_sightings: Vec<(EdgeId, Claim)> = Vec::new();
        let mut jobs: Vec<(EdgeId, Claim, Claim, Transcript)> = Vec::new();
        for dc in &deferred.claims {
            let edge_id = dc.edge_id;
            let incoming = Claim {
                edge_id,
                sparse_id: 0,
                point: dc.point.clone(),
                eval: dc.eval,
            };
            if let Some(prior) = self.claims.get(&edge_id) {
                let fork = self.transcript.fork(b"stream_reducer", edge_id);
                jobs.push((edge_id, prior.clone(), incoming, fork));
            } else {
                first_sightings.push((edge_id, incoming));
            }
        }

        // Step 2c — Run reducer jobs. Three dispatch modes:
        //   - Multi-GPU (default when device pool has > 1 device + cache
        //     enabled): partition jobs by their edge's assigned device,
        //     run one OS thread per device via std::thread::scope. Each
        //     thread sets its device once and runs its slice serially.
        //     Caches sharded across GPUs → memory + compute parallelism.
        //   - Single-GPU parallel (legacy Method 3, default off): same
        //     rayon-pool fork-transcript approach we proved doesn't help
        //     on one GPU. Still useful when no cache available.
        //   - Single-GPU sequential: one job after another. Fastest on
        //     a single GPU with the cache enabled.
        let t_setup = t_start.elapsed();
        let n_jobs = jobs.len();
        let use_cache = x_ext2_cache_enabled();
        let parallel = parallel_reducer_enabled();
        let device_pool = crate::fold::tree::gpu_device_pool();
        let multi_gpu = use_cache && device_pool.len() > 1;
        let raw_results: Vec<(EdgeId, ReducerStep, Claim)> = if multi_gpu {
            // ---- Multi-GPU path ----
            // Group jobs by device index in the pool.
            let dev_idx: std::collections::HashMap<i32, usize> =
                device_pool.iter().enumerate().map(|(i, &d)| (d, i)).collect();
            let n_dev = device_pool.len();
            let mut by_device: Vec<Vec<(EdgeId, Claim, Claim, Transcript, DeviceBuffer<u64>)>> =
                (0..n_dev).map(|_| Vec::new()).collect();
            for (edge_id, prior, incoming, fork) in jobs {
                let dev = *self.edge_device.get(&edge_id)
                    .expect("multi_gpu: edge_device missing for edge with prior state");
                let idx = *dev_idx.get(&dev).expect("device not in pool");
                // Take d_acc out of the cache for the duration of the
                // parallel section; we'll put it back after. In factored
                // mode d_acc isn't cached (and isn't used) — pass a 0-size
                // placeholder so the tuple shape is unchanged.
                let d_acc = self.d_acc_cache.remove(&edge_id)
                    .unwrap_or_else(|| DeviceBuffer::<u64>::new(0)
                        .expect("multi_gpu: placeholder d_acc alloc"));
                by_device[idx].push((edge_id, prior, incoming, fork, d_acc));
            }
            // x_ext2_cache stays shared (read-only across threads).
            let x_cache = &self.x_ext2_cache;
            let device_pool_ref = &device_pool;
            // Spawn one OS thread per device. Each pins itself with
            // `set_device` and runs its slice serially.
            let per_device_out: Vec<Vec<(EdgeId, ReducerStep, Claim, DeviceBuffer<u64>)>> =
                std::thread::scope(|s| {
                    let handles: Vec<_> = by_device.into_iter().enumerate()
                        .map(|(d_idx, jobs_for_dev)| {
                            let device = device_pool_ref[d_idx];
                            s.spawn(move || {
                                let _ = almost_goldilocks_cuda::set_device(device);
                                let mut local: Vec<(EdgeId, ReducerStep, Claim, DeviceBuffer<u64>)> =
                                    Vec::with_capacity(jobs_for_dev.len());
                                let factored = factored_reducer_enabled();
                                for (edge_id, prior, incoming, mut fork, mut d_acc) in jobs_for_dev {
                                    let w = &witnesses[edge_id][0];
                                    let n = w.data.as_ref().unwrap().n();
                                    let cached_x = x_cache.get(&edge_id)
                                        .expect("multi_gpu: x_ext2 cache miss");
                                    let out_claims: &[&Claim] = &[&prior, &incoming];
                                    let (mut sumcheck_proofs, mut new_claims) = if factored {
                                        // d_acc unused by the factored path (kept in the
                                        // tuple so the cache round-trip stays identical).
                                        Reducer.prove_with_cached_buffers_factored(
                                            cached_x, n, &[edge_id], out_claims, &mut fork,
                                        )
                                    } else {
                                        Reducer.prove_with_cached_buffers(
                                            cached_x, &mut d_acc, n,
                                            &[edge_id], out_claims, &mut fork,
                                        )
                                    };
                                    let new_claim = new_claims.pop().expect("no claim");
                                    let sumcheck_proof = sumcheck_proofs.pop().expect("no proof");
                                    let step = ReducerStep {
                                        edge_id,
                                        sumcheck_proof,
                                        new_eval: new_claim.eval,
                                    };
                                    local.push((edge_id, step, new_claim, d_acc));
                                }
                                local
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().expect("device thread panicked")).collect()
                });
            // Put d_acc buffers back into the cache + collect step
            // results in flat list (sorted later).
            let mut out: Vec<(EdgeId, ReducerStep, Claim)> = Vec::new();
            for batch in per_device_out {
                for (edge_id, step, claim, d_acc) in batch {
                    self.d_acc_cache.insert(edge_id, d_acc);
                    out.push((edge_id, step, claim));
                }
            }
            // Reset main thread's device for downstream code paths.
            let _ = almost_goldilocks_cuda::set_device(0);
            out
        } else if parallel {
            // ---- Single-GPU rayon-parallel path (legacy Method 3) ----
            let cache = &self.x_ext2_cache;
            use rayon::prelude::*;
            reducer_thread_pool().install(|| {
                jobs.into_par_iter()
                    .map(|(edge_id, prior, incoming, mut fork)| {
                        run_one_reducer_job(edge_id, prior, incoming, &mut fork, witnesses, cache, use_cache)
                    })
                    .collect()
            })
        } else {
            // ---- Single-GPU sequential path (default for 1 GPU) ----
            jobs.into_iter()
                .map(|(edge_id, prior, incoming, mut fork)| {
                    run_one_reducer_job_seq(
                        edge_id,
                        prior,
                        incoming,
                        &mut fork,
                        witnesses,
                        &self.x_ext2_cache,
                        &mut self.d_acc_cache,
                        use_cache,
                    )
                })
                .collect()
        };

        // Step 2d — Sort results by edge_id and absorb each sumcheck's
        // terminal `(edge_id, new_eval)` into the PARENT transcript in
        // canonical order. The verifier mirrors this exactly, so
        // subsequent parent-transcript challenges (next add_proof,
        // finalize) are bound to the full set of accumulator updates
        // regardless of the parallel execution order.
        let t_reduce = t_start.elapsed();
        let mut sorted = raw_results;
        sorted.sort_by_key(|(eid, _, _)| *eid);

        let mut chunk_steps: Vec<ReducerStep> = Vec::with_capacity(sorted.len());
        for (edge_id, step, new_claim) in sorted {
            self.transcript.append_u64(b"sa_step_edge", edge_id as u64);
            self.transcript.append_ext2(b"sa_step_new_eval", &step.new_eval);
            chunk_steps.push(step.clone());
            self.steps.push(step);
            self.claims.insert(edge_id, new_claim);
        }
        for (edge_id, incoming) in first_sightings {
            self.claims.insert(edge_id, incoming);
        }

        if timing {
            use std::sync::atomic::Ordering;
            use crate::basicblock::reducer::{
                REDUCER_EQ_BUILD_US, REDUCER_EQ_COPY_US, REDUCER_SCALE_US, REDUCER_SUMCHECK_US,
            };
            let eqb = REDUCER_EQ_BUILD_US.swap(0, Ordering::Relaxed) as f64 / 1e3;
            let eqc = REDUCER_EQ_COPY_US.swap(0, Ordering::Relaxed) as f64 / 1e3;
            let sca = REDUCER_SCALE_US.swap(0, Ordering::Relaxed) as f64 / 1e3;
            let smc = REDUCER_SUMCHECK_US.swap(0, Ordering::Relaxed) as f64 / 1e3;
            eprintln!(
                "[acc-update] {} jobs | wall: setup(cache+absorb) {:?}, reduce-dispatch {:?}, absorb {:?} | reducer thread-time: eq-build {:.0}ms eq-copy+alloc {:.0}ms scale-acc {:.0}ms sumcheck {:.0}ms",
                n_jobs, t_setup, t_reduce - t_setup, t_start.elapsed() - t_reduce,
                eqb, eqc, sca, smc,
            );
        }
        StreamingProofChunk { steps: chunk_steps }
    }

    /// Number of accumulated edges (= number of distinct Constant
    /// edges seen so far).
    pub fn num_edges(&self) -> usize {
        self.claims.len()
    }

    /// Number of reducer-step proofs emitted so far. With N proofs over
    /// a model with K shared Constant edges, this is K · (N − 1).
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Close the stream and produce a [`FinalizationProof`]: one Ajtai
    /// fold-tree opening covering every accumulated claim. After this,
    /// the accumulator is consumed — repeat add_proof calls would
    /// require starting a new accumulator (per Rule 4 of the soundness
    /// invariants).
    pub fn finalize(
        self,
        witnesses: &[Vec<Witness>],
        store: &GpuAjtaiStore,
    ) -> FinalizationProof {
        // Drop the streaming x_ext2 GPU cache up-front so the fold-tree
        // work below has the full device free. Holding ~K × n MB of
        // lifted-witness buffers across finalize would force its leaf
        // build to allocate from fragmented memory (measured: ~3×
        // slowdown on Llama 1L finalize at N=5).
        let Self {
            claims,
            steps,
            transcript,
            x_ext2_cache,
            d_acc_cache,
            edge_device: _,
        } = self;
        drop(x_ext2_cache);
        drop(d_acc_cache);
        let key = &store.key;
        let mut transcript = transcript;

        // Iterate accumulator state in canonical (edge_id ascending)
        // order so the verifier's mirror walks the same sequence and
        // builds the same fold-tree leaf list.
        let mut edges: Vec<(EdgeId, Claim)> = claims.into_iter().collect();
        edges.sort_by_key(|(eid, _)| *eid);

        let mut leaves: Vec<FoldInstance> = Vec::new();
        let mut edge_plane_evals: Vec<EdgePlaneEvals> = Vec::new();
        for (edge_id, claim) in &edges {
            let ec = store
                .get(*edge_id)
                .expect("finalize: edge not committed in store");
            // Zero-extend the accumulated point from its native length
            // (= log|W|) up to `ec.arity` for fold-tree leaf compatibility.
            // The MLE of a zero-padded poly at (native_pt, 0, ..., 0)
            // equals the MLE of the native poly at native_pt — so the
            // eval at the extended point is unchanged.
            let extended_point = {
                let mut p = claim.point.clone();
                assert!(
                    p.len() <= ec.arity,
                    "accumulated point length {} exceeds edge arity {}",
                    p.len(),
                    ec.arity,
                );
                while p.len() < ec.arity {
                    p.push(
                        almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::zero(),
                    );
                }
                p
            };

            // Bit-planes for this Constant. The cache is populated during
            // commit_constants, so the recompute fallback only fires if
            // the store was reloaded from disk after a previous session.
            let planes_packed: Vec<Vec<u64>> = match store.get_planes(*edge_id) {
                Some(planes) => planes.clone(),
                None => {
                    let w = &witnesses[*edge_id][0];
                    decompose_witness_for_fold_native(
                        w,
                        key,
                        ec.arity,
                        &extended_point,
                    )
                    .0
                }
            };
            let total = 1usize << ec.arity;
            let plane_evals: Vec<_> = if ec.arity >= 12 {
                let plane_refs: Vec<&[u64]> =
                    planes_packed.iter().map(|p| p.as_slice()).collect();
                almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_device(
                    &extended_point,
                    &plane_refs,
                )
                .expect("eval_binary_planes_device failed in finalize")
            } else {
                let eq = crate::poly::evaluate_lagrange_basis_ext2(&extended_point);
                planes_packed
                    .iter()
                    .map(|p| eval_binary_with_shared_eq(p, &eq, total))
                    .collect()
            };

            let edge_base = ec.base;
            if edge_base == 2 {
                let reconstructed = reconstruct_signed_two_complement(
                    &plane_evals, false, key.b, 2,
                );
                assert!(
                    ext2_field_eq(reconstructed, claim.eval),
                    "finalize: per-plane evals for edge {} don't reconstruct accumulated claim",
                    edge_id,
                );
                for (pi, packed) in planes_packed.into_iter().enumerate() {
                    leaves.push(FoldInstance {
                        commitment: ec.planes[pi].clone(),
                        data: FoldData::Binary(packed),
                        arity: ec.arity,
                        claim_pt: extended_point.clone(),
                        claim_val: plane_evals[pi],
                    });
                }
                edge_plane_evals.push(EdgePlaneEvals {
                    edge_id: *edge_id,
                    sparse_id: 0,
                    arity: ec.arity,
                    combined_point: extended_point,
                    combined_eval: claim.eval,
                    plane_evals,
                    is_sparse: false,
                });
            } else {
                // Higher-radix (Digit) Constants: group b binary planes into
                // b_β digit-plane evals (same scheme as fold_integration's
                // dense Digit path). Currently untested through the
                // streaming pipeline; the math mirrors the existing
                // prove_with_fold_tree branch.
                let k_log = edge_base.trailing_zeros() as usize;
                let b_beta =
                    crate::commit::bit_decompose::digit_planes_for(key.b, edge_base);
                let two = almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(
                    AlmostGoldilocksField(2),
                );
                let mut digit_evals = Vec::with_capacity(b_beta);
                for j in 0..b_beta {
                    let lo = j * k_log;
                    let hi = ((j + 1) * k_log).min(key.b);
                    let m = hi - lo;
                    let is_top = j == b_beta - 1;
                    let mut y =
                        almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::zero();
                    let mut pow =
                        almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::one();
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
                    &digit_evals,
                    false,
                    key.b,
                    edge_base,
                );
                assert!(
                    ext2_field_eq(reconstructed, claim.eval),
                    "finalize: digit-plane evals for edge {} don't reconstruct accumulated claim (base={})",
                    edge_id,
                    edge_base,
                );
                let mut slots: Vec<Option<Vec<u64>>> =
                    planes_packed.into_iter().map(Some).collect();
                for j in 0..b_beta {
                    let lo = j * k_log;
                    let hi = ((j + 1) * k_log).min(key.b);
                    let m = hi - lo;
                    let is_top = j == b_beta - 1;
                    let mut bit_planes: Vec<Vec<u64>> = Vec::with_capacity(m);
                    for kk in 0..m {
                        bit_planes.push(
                            slots[lo + kk].take().expect("plane already consumed"),
                        );
                    }
                    leaves.push(FoldInstance {
                        commitment: ec.planes[j].clone(),
                        data: FoldData::Digit {
                            base: key.base,
                            bit_planes,
                            negate_top_bit: is_top,
                        },
                        arity: ec.arity,
                        claim_pt: extended_point.clone(),
                        claim_val: digit_evals[j],
                    });
                }
                edge_plane_evals.push(EdgePlaneEvals {
                    edge_id: *edge_id,
                    sparse_id: 0,
                    arity: ec.arity,
                    combined_point: extended_point,
                    combined_eval: claim.eval,
                    plane_evals: digit_evals,
                    is_sparse: false,
                });
            }
        }

        // Absorb per-edge plane reveals (same protocol as fold_integration).
        for epe in &edge_plane_evals {
            transcript.append_u64(b"ft_edge", epe.edge_id as u64);
            transcript.append_u64(b"ft_edge_is_sparse", epe.is_sparse as u64);
            transcript.append_ext2(b"ft_combined_eval", &epe.combined_eval);
            for e in &epe.plane_evals {
                transcript.append_ext2(b"ft_plane_eval", e);
            }
        }

        let fold_tree = prove_fold_tree(leaves, key.seed, &mut transcript);

        FinalizationProof {
            edge_plane_evals,
            fold_tree,
            reducer_steps: steps,
        }
    }
}

/// Verifier counterpart of `run_one_reducer_job`. Returns
/// `Some((edge_id, new_eval, new_point))` on success, `None` if the
/// reducer-step sumcheck fails. Pure over its inputs — safe to call
/// from rayon workers.
fn verify_one_reducer_job(
    edge_id: EdgeId,
    prior: Claim,
    incoming: Claim,
    step: &ReducerStep,
    transcript: &mut Transcript,
    witnesses: &[Vec<Witness>],
) -> Option<(EdgeId, AlmostGoldilocksExt2, Vec<AlmostGoldilocksExt2>)> {
    let w = &witnesses[edge_id][0];
    let new_claim_stub = Claim {
        edge_id,
        sparse_id: 0,
        point: Vec::new(),
        eval: step.new_eval,
    };
    let claims_ref: &[&Claim] = &[&prior, &incoming, &new_claim_stub];
    let new_point = Reducer.verify_with_point(
        &[w],
        claims_ref,
        &[&step.sumcheck_proof],
        transcript,
    )?;
    Some((edge_id, step.new_eval, new_point))
}

/// Absorb a single `DeferredClaim` into a transcript. Used by both the
/// streaming prover and the streaming verifier to keep their transcripts
/// in lockstep — any change to this function MUST be mirrored on both
/// sides or the verifier will reject.
pub(crate) fn absorb_deferred_claim(transcript: &mut Transcript, dc: &DeferredClaim) {
    transcript.append_u64(b"sa_edge", dc.edge_id as u64);
    transcript.append_u64(b"sa_arity", dc.arity as u64);
    for p in &dc.point {
        transcript.append_ext2(b"sa_pt", p);
    }
    transcript.append_ext2(b"sa_eval", &dc.eval);
}

/// One per-call slice of the streaming proof. The prover's `add_proof`
/// emits one of these; the verifier's `verify_add_proof` consumes one of
/// these. `steps` has length equal to the number of edges that already
/// had state when this proof arrived (zero for the very first proof).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StreamingProofChunk {
    pub steps: Vec<ReducerStep>,
}

/// Final closing proof of a streaming session: ONE Ajtai fold-tree
/// opening covering every accumulated claim, plus the cumulative
/// reducer-step proofs (replayed by the verifier to reconstruct the
/// accumulator state if it wasn't tracking add_proof-by-add_proof —
/// kept here for self-containedness of the closing proof).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalizationProof {
    pub edge_plane_evals: Vec<EdgePlaneEvals>,
    pub fold_tree: FoldTreeProof,
    /// Full cumulative reducer-step sequence (= concatenation of every
    /// per-add_proof chunk). Allows a fresh verifier to replay the
    /// entire stream from scratch, given the original DeferredResults.
    pub reducer_steps: Vec<ReducerStep>,
}

/// Verifier-side mirror of [`AccumulatorState`]. Maintains its own
/// `claims` map and `transcript`, advanced via [`Self::verify_add_proof`]
/// to mirror the prover. Diverges from the prover iff the prover
/// cheated — in which case the next reducer-step verify or finalize
/// check will fail.
pub struct VerifierAccumulator {
    pub claims: HashMap<EdgeId, Claim>,
    transcript: Transcript,
}

impl VerifierAccumulator {
    pub fn new(label: &[u8]) -> Self {
        Self {
            claims: HashMap::new(),
            transcript: Transcript::new(label),
        }
    }

    pub fn num_edges(&self) -> usize {
        self.claims.len()
    }

    /// Consume one per-proof DeferredResult + matching prover chunk.
    /// Returns false if any reducer-step sumcheck fails, the chunk has
    /// the wrong number of steps, or a step's `edge_id` doesn't match
    /// the deferred-claims processing order. On false return, the
    /// accumulator state is best-effort; the caller should abort.
    pub fn verify_add_proof(
        &mut self,
        deferred: &DeferredResult,
        witnesses: &[Vec<Witness>],
        chunk: &StreamingProofChunk,
    ) -> bool {
        // Absorb deferred claims into transcript in the same canonical
        // order as the prover side.
        for dc in &deferred.claims {
            absorb_deferred_claim(&mut self.transcript, dc);
        }

        // Build verify jobs — one per deferred claim that has a prior
        // accumulated claim. Each job carries its OWN forked transcript,
        // matching the prover's `fork(b"stream_reducer", edge_id)` discipline.
        // First-sighting claims are stored verbatim (no reducer step
        // consumed).
        let mut first_sightings: Vec<(EdgeId, Claim)> = Vec::new();
        let mut jobs: Vec<(EdgeId, Claim, Claim, Transcript, &ReducerStep)> = Vec::new();
        let mut step_iter = chunk.steps.iter();
        for dc in &deferred.claims {
            let edge_id = dc.edge_id;
            let incoming = Claim {
                edge_id,
                sparse_id: 0,
                point: dc.point.clone(),
                eval: dc.eval,
            };
            if let Some(prior) = self.claims.get(&edge_id) {
                let step = match step_iter.next() {
                    Some(s) => s,
                    None => return false, // chunk underflow
                };
                if step.edge_id != edge_id {
                    return false; // edge_id mismatch — proof out of order
                }
                let fork = self.transcript.fork(b"stream_reducer", edge_id);
                jobs.push((edge_id, prior.clone(), incoming, fork, step));
            } else {
                first_sightings.push((edge_id, incoming));
            }
        }
        if step_iter.next().is_some() {
            return false; // chunk has more steps than deferred claims with prior state
        }

        // Run verify_with_point for each job. Sequential by default
        // (verify is cheap — single eq + check); could be parallelized
        // for parity with the prover but small wins given the size.
        let parallel = parallel_reducer_enabled();
        let raw_results: Result<Vec<(EdgeId, AlmostGoldilocksExt2, Vec<AlmostGoldilocksExt2>)>, ()> =
            if parallel {
                use rayon::prelude::*;
                reducer_thread_pool().install(|| {
                    jobs.into_par_iter()
                        .map(|(edge_id, prior, incoming, mut fork, step)| {
                            verify_one_reducer_job(
                                edge_id, prior, incoming, step, &mut fork, witnesses,
                            )
                            .ok_or(())
                        })
                        .collect()
                })
            } else {
                jobs.into_iter()
                    .map(|(edge_id, prior, incoming, mut fork, step)| {
                        verify_one_reducer_job(
                            edge_id, prior, incoming, step, &mut fork, witnesses,
                        )
                        .ok_or(())
                    })
                    .collect()
            };
        let results = match raw_results {
            Ok(r) => r,
            Err(_) => return false,
        };

        // Sort by edge_id (canonical order) and absorb terminal evals
        // into parent transcript, mirroring the prover's step 2d. After
        // this point parent is in sync with the prover's parent.
        let mut sorted = results;
        sorted.sort_by_key(|(eid, _, _)| *eid);
        for (edge_id, new_eval, new_point) in sorted {
            self.transcript.append_u64(b"sa_step_edge", edge_id as u64);
            self.transcript.append_ext2(b"sa_step_new_eval", &new_eval);
            self.claims.insert(
                edge_id,
                Claim {
                    edge_id,
                    sparse_id: 0,
                    point: new_point,
                    eval: new_eval,
                },
            );
        }
        for (edge_id, incoming) in first_sightings {
            self.claims.insert(edge_id, incoming);
        }
        true
    }

    /// Validate the closing fold-tree opening against the verifier's
    /// running accumulator state. Returns true iff (a) every per-edge
    /// plane-eval entry reconstructs to its accumulated claim's eval at
    /// the claim's point, and (b) the fold-tree proof verifies.
    pub fn verify_finalize(
        self,
        store: &GpuAjtaiStore,
        proof: &FinalizationProof,
    ) -> bool {
        let key = &store.key;
        let mut transcript = self.transcript;

        // Build the same edge_id-ascending iteration order the prover used.
        let mut edges: Vec<(EdgeId, Claim)> = self.claims.into_iter().collect();
        edges.sort_by_key(|(eid, _)| *eid);

        // The proof's edge_plane_evals must list exactly these edges
        // (length + edge_ids match, in order).
        if proof.edge_plane_evals.len() != edges.len() {
            return false;
        }

        for (epe, (edge_id, claim)) in proof.edge_plane_evals.iter().zip(edges.iter()) {
            if epe.edge_id != *edge_id {
                return false;
            }
            if epe.is_sparse {
                // Constants are never sparse; reject.
                return false;
            }
            let ec = match store.get(*edge_id) {
                Some(c) => c,
                None => return false,
            };
            if epe.arity != ec.arity {
                return false;
            }
            // Reconstruct the claim's eval from per-plane evals via the
            // edge's signed two's-complement scheme.
            let edge_base = ec.base;
            let expected_plane_count = if edge_base == 2 {
                key.b
            } else {
                crate::commit::bit_decompose::digit_planes_for(key.b, edge_base)
            };
            if epe.plane_evals.len() != expected_plane_count {
                return false;
            }
            let reconstructed = reconstruct_signed_two_complement(
                &epe.plane_evals, false, key.b, edge_base,
            );
            if !ext2_field_eq(reconstructed, epe.combined_eval) {
                return false;
            }
            // The accumulated claim's eval must equal the declared
            // combined eval; the combined_point must equal the
            // zero-extension of the accumulated point to `ec.arity`.
            if epe.combined_eval != claim.eval {
                return false;
            }
            if epe.combined_point.len() != ec.arity {
                return false;
            }
            if claim.point.len() > epe.combined_point.len() {
                return false;
            }
            if claim.point != epe.combined_point[..claim.point.len()] {
                return false;
            }
            for p in &epe.combined_point[claim.point.len()..] {
                if *p != almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::zero() {
                    return false;
                }
            }
        }

        // Absorb plane reveals (same protocol as fold_integration).
        for epe in &proof.edge_plane_evals {
            transcript.append_u64(b"ft_edge", epe.edge_id as u64);
            transcript.append_u64(b"ft_edge_is_sparse", epe.is_sparse as u64);
            transcript.append_ext2(b"ft_combined_eval", &epe.combined_eval);
            for e in &epe.plane_evals {
                transcript.append_ext2(b"ft_plane_eval", e);
            }
        }

        // Rebuild the fold-tree leaves metadata.
        let mut leaves_meta: Vec<(_, _, _, _)> = Vec::new();
        for epe in &proof.edge_plane_evals {
            let ec = store.get(epe.edge_id).unwrap();
            for (pi, y) in epe.plane_evals.iter().enumerate() {
                leaves_meta.push((
                    ec.planes[pi].clone(),
                    epe.arity,
                    epe.combined_point.clone(),
                    *y,
                ));
            }
        }
        verify_fold_tree(&leaves_meta, &proof.fold_tree, &mut transcript).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::ajtai::Seed;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    use crate::commit::{AjtaiKey, GpuAjtaiStore};
    use crate::dag::{DagBuilder, DataType, Role};
    use crate::util::arith::{ext2_field_eq, int_to_f};

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

    /// Build a tiny DAG with 2 weight params, run it N times with N
    /// different inputs (all in-range), prove each in defer mode, feed
    /// the deferred results into a streaming accumulator. Verify the
    /// final accumulated state has 1 claim per weight and the right
    /// number of reducer steps.
    #[test]
    fn streaming_n_proofs_correct_state_shape() {
        almost_goldilocks_cuda::init().expect("CUDA init");

        // Single-DAG, 2 weight Constants → expect 2 accumulated claims.
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w0 = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let w1 = g.param(make_const(vec![4], vec![2, 2, 2, 2]));
        let s0 = g.add(x, w0)[0];
        let _s1 = g.add(s0, w1)[0];
        let (dag, witnesses_template) = g.compile();

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);

        // Run N inferences with distinct inputs, accumulate deferred claims.
        let n_proofs = 3usize;
        let inputs: Vec<Vec<i128>> = (0..n_proofs as i128)
            .map(|i| vec![i, i + 1, i + 2, i + 3])
            .collect();

        let mut acc = AccumulatorState::new(b"streaming-test");
        let mut last_witnesses: Option<Vec<Vec<Witness>>> = None;

        for (proof_idx, input_vals) in inputs.iter().enumerate() {
            // Production serving would call `commit_constants` once at
            // startup and reuse the offline commits across all inferences.
            // For the test, a fresh per-inference store is functionally
            // equivalent: Ajtai commits are deterministic, so committing
            // the same Constant witness with the same key always yields
            // the same commitment. (`commit_edges` is idempotent, which
            // makes it inconvenient to re-commit non-constants in place.)
            let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
            let mut witnesses = witnesses_template.clone();
            dag.run(
                &mut witnesses,
                &[(0, make_input(vec![4], input_vals.clone()))],
            );
            dag.commit(&witnesses, &mut store);
            // Per-proof prover in defer mode.
            let mut t_p = Transcript::new(b"per-proof");
            let (dag_proof, fold_proof) = dag.prove_with_fold_tree_modes(
                &witnesses,
                &store,
                &mut t_p,
                /*defer_const=*/ true,
            );
            // Per-proof verifier returns DeferredResult.
            let mut t_v = Transcript::new(b"per-proof");
            let r = dag.verify_with_fold_tree_deferred(
                &witnesses,
                &store,
                &dag_proof,
                &fold_proof,
                &mut t_v,
            );
            assert!(r.ok, "per-proof verify must pass (proof {})", proof_idx);

            acc.add_proof(&r, &witnesses);
            last_witnesses = Some(witnesses);
        }

        let constant_edges: Vec<EdgeId> = (0..dag.num_edges())
            .filter(|&e| {
                last_witnesses
                    .as_ref()
                    .map(|w| w[e][0].role == Role::Constant && w[e][0].data.is_some())
                    .unwrap_or(false)
            })
            .collect();
        let n_consts = constant_edges.len();
        assert!(n_consts >= 1, "test DAG should have ≥ 1 Constant edge");

        assert_eq!(
            acc.num_edges(),
            n_consts,
            "accumulator should hold one claim per Constant edge",
        );
        assert_eq!(
            acc.num_steps(),
            n_consts * (n_proofs - 1),
            "expected K·(N-1) reducer steps for K={} constants × N={} proofs",
            n_consts,
            n_proofs,
        );

        // Sanity: each accumulated claim's eval really IS W(r_acc) for the
        // weight polynomial. This is the property the streaming verifier
        // will later check via the reducer-step replay; here we confirm
        // it directly to catch any off-by-one in the prover-side update.
        let witnesses = last_witnesses.unwrap();
        for (&edge_id, claim) in &acc.claims {
            let w = &witnesses[edge_id][0];
            let f = w.data.as_ref().unwrap();
            let direct = f.evaluate_at_point_ext2(&claim.point);
            assert!(
                ext2_field_eq(direct, claim.eval),
                "accumulated claim for edge {} does not match W(r_acc): direct={:?} acc={:?}",
                edge_id,
                direct,
                claim.eval,
            );
        }
    }

    /// Single-proof case: the accumulator stores incoming claims
    /// verbatim, no reducer steps emitted.
    #[test]
    fn streaming_single_proof_no_reducer_steps() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let _y = g.add(x, w)[0];
        let (dag, witnesses_template) = g.compile();

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);

        let mut acc = AccumulatorState::new(b"streaming-1");
        let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
        let mut witnesses = witnesses_template.clone();
        dag.run(
            &mut witnesses,
            &[(0, make_input(vec![4], vec![1, 2, 3, 4]))],
        );
        dag.commit(&witnesses, &mut store);
        let mut t_p = Transcript::new(b"per-proof");
        let (dp, fp) = dag.prove_with_fold_tree_modes(
            &witnesses,
            &store,
            &mut t_p,
            true,
        );
        let mut t_v = Transcript::new(b"per-proof");
        let r = dag.verify_with_fold_tree_deferred(
            &witnesses, &store, &dp, &fp, &mut t_v,
        );
        assert!(r.ok);

        acc.add_proof(&r, &witnesses);

        assert!(acc.num_edges() >= 1, "should have ≥ 1 accumulated edge");
        assert_eq!(
            acc.num_steps(),
            0,
            "single-proof case should produce zero reducer steps",
        );
    }

    /// Phase 4 end-to-end: N proofs streamed through both prover and
    /// verifier accumulators, finalize, verify_finalize. The full
    /// soundness chain must close out — `verify_finalize` returns true
    /// iff every reducer step + the closing fold-tree opening verifies.
    #[test]
    fn streaming_end_to_end_with_finalize() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w0 = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let w1 = g.param(make_const(vec![4], vec![2, 2, 2, 2]));
        let s0 = g.add(x, w0)[0];
        let _s1 = g.add(s0, w1)[0];
        let (dag, witnesses_template) = g.compile();

        let key = AjtaiKey::new(demo_seed(), /*max_num_vars=*/ 10, /*b=*/ 21);
        let n_proofs = 3usize;
        let inputs: Vec<Vec<i128>> = (0..n_proofs as i128)
            .map(|i| vec![i, i + 1, i + 2, i + 3])
            .collect();

        let label = b"streaming-e2e";
        let mut prover_acc = AccumulatorState::new(label);
        let mut verifier_acc = VerifierAccumulator::new(label);
        let mut chunks: Vec<StreamingProofChunk> = Vec::new();
        let mut deferred_results: Vec<DeferredResult> = Vec::new();
        let mut last_store: Option<GpuAjtaiStore> = None;
        let mut last_witnesses: Option<Vec<Vec<Witness>>> = None;

        for (proof_idx, input_vals) in inputs.iter().enumerate() {
            let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
            let mut witnesses = witnesses_template.clone();
            dag.run(&mut witnesses, &[(0, make_input(vec![4], input_vals.clone()))]);
            dag.commit(&witnesses, &mut store);

            let mut t_p = Transcript::new(b"per-proof");
            let (dp, fp) = dag.prove_with_fold_tree_modes(
                &witnesses, &store, &mut t_p, true,
            );
            let mut t_v = Transcript::new(b"per-proof");
            let r = dag.verify_with_fold_tree_deferred(
                &witnesses, &store, &dp, &fp, &mut t_v,
            );
            assert!(r.ok, "per-proof verify failed at proof {}", proof_idx);

            let chunk = prover_acc.add_proof(&r, &witnesses);
            let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
            assert!(ok, "streaming verifier rejected at proof {}", proof_idx);

            chunks.push(chunk);
            deferred_results.push(r);
            last_store = Some(store);
            last_witnesses = Some(witnesses);
        }

        // The prover and verifier should have converged on the same
        // accumulator state (same claims).
        assert_eq!(
            prover_acc.claims.len(),
            verifier_acc.claims.len(),
            "prover/verifier claim count diverged",
        );
        for (eid, prover_claim) in &prover_acc.claims {
            let v_claim = verifier_acc
                .claims
                .get(eid)
                .expect("verifier missing claim that prover has");
            assert_eq!(
                prover_claim.eval, v_claim.eval,
                "diverged eval for edge {}",
                eid,
            );
            assert_eq!(
                prover_claim.point, v_claim.point,
                "diverged point for edge {}",
                eid,
            );
        }

        // Finalize.
        let store = last_store.unwrap();
        let witnesses = last_witnesses.unwrap();
        let proof = prover_acc.finalize(&witnesses, &store);
        let ok = verifier_acc.verify_finalize(&store, &proof);
        assert!(ok, "verify_finalize must accept a faithful stream");
    }

    /// Soundness: mutating a `ReducerStep`'s `new_eval` in a chunk
    /// breaks the verifier — either the reducer-step sumcheck fails
    /// during `verify_add_proof`, or (if the mutation slips past) the
    /// downstream finalize fails because the accumulated state diverges.
    #[test]
    fn streaming_rejects_tampered_reducer_step() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], DataType::Int);
        let w0 = g.param(make_const(vec![4], vec![1, 1, 1, 1]));
        let w1 = g.param(make_const(vec![4], vec![2, 2, 2, 2]));
        let s0 = g.add(x, w0)[0];
        let _ = g.add(s0, w1)[0];
        let (dag, witnesses_template) = g.compile();

        let key = AjtaiKey::new(demo_seed(), 10, 21);
        let label = b"streaming-tamper";
        let mut prover_acc = AccumulatorState::new(label);
        let mut verifier_acc = VerifierAccumulator::new(label);

        // First proof (no reducer steps yet).
        let mut store0 = GpuAjtaiStore::new(dag.num_edges(), key);
        let mut w0_wit = witnesses_template.clone();
        dag.run(&mut w0_wit, &[(0, make_input(vec![4], vec![0, 0, 0, 0]))]);
        dag.commit(&w0_wit, &mut store0);
        let mut t_p = Transcript::new(b"per-proof");
        let (dp0, fp0) = dag.prove_with_fold_tree_modes(&w0_wit, &store0, &mut t_p, true);
        let mut t_v = Transcript::new(b"per-proof");
        let r0 = dag.verify_with_fold_tree_deferred(&w0_wit, &store0, &dp0, &fp0, &mut t_v);
        let chunk0 = prover_acc.add_proof(&r0, &w0_wit);
        assert!(verifier_acc.verify_add_proof(&r0, &w0_wit, &chunk0));

        // Second proof — produces reducer steps. Tamper with the first.
        let mut store1 = GpuAjtaiStore::new(dag.num_edges(), key);
        let mut w1_wit = witnesses_template.clone();
        dag.run(&mut w1_wit, &[(0, make_input(vec![4], vec![1, 2, 3, 4]))]);
        dag.commit(&w1_wit, &mut store1);
        let mut t_p = Transcript::new(b"per-proof");
        let (dp1, fp1) = dag.prove_with_fold_tree_modes(&w1_wit, &store1, &mut t_p, true);
        let mut t_v = Transcript::new(b"per-proof");
        let r1 = dag.verify_with_fold_tree_deferred(&w1_wit, &store1, &dp1, &fp1, &mut t_v);
        let mut chunk1 = prover_acc.add_proof(&r1, &w1_wit);
        assert!(!chunk1.steps.is_empty(), "expected ≥ 1 reducer step on 2nd proof");

        // Tamper: add 1 to the first step's new_eval.
        let one = almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(
            AlmostGoldilocksField(1),
        );
        chunk1.steps[0].new_eval = ext2_add(chunk1.steps[0].new_eval, one);

        let ok = verifier_acc.verify_add_proof(&r1, &w1_wit, &chunk1);
        assert!(!ok, "tampered ReducerStep.new_eval must be rejected");
    }
}
