//! Fold-tree scheduler (plan §6.4). Iteratively contracts a leaf set of
//! `FoldInstance`s via `same_point + multifold + split` per group of 63
//! until `≤ 63` instances remain, then runs one final
//! `same_point + multifold` (no split) and ships the final witness `f*`
//! verbatim to the verifier.
//!
//! Concrete numbers (per Almost-Goldilocks SuperNeo params):
//! - Per-group input cap `K + k = 50 + 13 = 63`
//! - Per-group output count after split: 13 ternary chunks
//! - Contraction ratio: `63 / 13 ≈ 4.85` per internal level
//!
//! This module is **sequential** for now — every group at a given level
//! runs in series. The parallel scheduler (per-group `Transcript::fork`
//! + multi-GPU dispatch from §6.4) is a follow-up; the soundness work
//! is what matters for step 12.

use almost_goldilocks_cuda::ajtai::{RingCommitment, Seed};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use serde::{Deserialize, Serialize};

use crate::fold::{
    FoldData, FoldInstance, MultifoldProof, SamePointProof, SplitProof,
    WireCommitment, multifold, same_point_sumcheck, split,
};
use crate::transcript::Transcript;

/// Max instances per group (matches AGL's `multifold` `M ≤ 63` SuperNeo
/// bound).
pub const MAX_PER_GROUP: usize = 63;

/// Group size for a bucket of the given arity. With the shared-eq same-point
/// backend (default on) eq is stored once per *unique* claim_pt, so a group's
/// GPU same-point state is ≈ `num_unique · 2^arity · 64 B` (num_unique ≤ 3) —
/// a few GB even at arity 24, no longer the 64 GB the old per-leaf interleaved
/// state needed. So the full `MAX_PER_GROUP` fits at every arity here, which
/// minimizes the number of groups and fold-tree levels. (Env override
/// `ZK4_GROUP_SIZE_CAP_ARITY` re-introduces a 31-cap at/above that arity for
/// the shared-eq-off path, which still uses the interleaved state.)
/// Smaller groups only RELAX the SuperNeo norm bound `(K+k)·T·(b−1) < 2^13`
/// (split still emits 13 chunks), so any cap is sound; it costs at most one
/// extra level. Prover and verifier MUST agree — both derive it from arity.
///
/// MEASURED: capping is a large LOSS, do not do it for performance. On llama2
/// 8L/seq64 at table_commit_log 12, 4xA100, fold tree with 63-leaf groups vs
/// `ZK4_GROUP_SIZE_CAP_ARITY=24` (31-leaf): 82.6s vs 152.5s (rep 1) and 81.1s
/// vs 147.8s (rep 2) — about 1.8x slower. Halving the group size doubles the
/// group count, so it doubles per-group fixed cost (transcript setup, pooled
/// GPU buffer allocation, one same-point/multifold/split invocation each) AND
/// adds a sequential level. The "minimize levels" rationale above is correct.
/// The same runs show mirror affinity is NOT the explanation: it read 67% in
/// both arms of rep 1, and swung 58% -> 35% in rep 2, i.e. it is noisy enough
/// (±9 points at a fixed setting) that it cannot be used to explain a delta
/// without repeated measurement.
pub fn group_size_for_arity(arity: usize) -> usize {
    if let Some(cap) = std::env::var("ZK4_GROUP_SIZE_CAP_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        if arity >= cap { return 31; }
    }
    MAX_PER_GROUP
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldGroupProof {
    pub same_point: SamePointProof,
    pub multifold: MultifoldProof,
    /// `None` only at the final level (single surviving group, no split).
    pub split: Option<SplitProof>,
    /// Per-chunk claim values `y_chunk_i = chunk_i(shared_r)` for the
    /// next level's same-point sumcheck. Empty at the final level (no
    /// next level to chain into). The verifier reads these straight
    /// into its next-level metadata — they're prover-supplied and
    /// trust-anchored only at the final `commit(f*) = c*` check.
    pub chunk_claim_vals: Vec<AlmostGoldilocksExt2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldLevelProof {
    pub groups: Vec<FoldGroupProof>,
}

/// Final-witness wire form: i16 wide coefficients if the final instance
/// is ternary, or packed `u64`s if it's binary (rare).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FinalWitness {
    Binary { n_ring: usize, packed: Vec<u64> },
    Ternary { n_ring: usize, pos: Vec<u64>, neg: Vec<u64> }, // 13 chunks
}

impl FinalWitness {
    pub fn from_fold_data(data: &FoldData) -> Self {
        match data {
            FoldData::Binary(v) => Self::Binary { n_ring: v.len(), packed: v.clone() },
            FoldData::Ternary(c) => Self::Ternary {
                n_ring: c.n_ring,
                pos: c.pos.clone(),
                neg: c.neg.clone(),
            },
            FoldData::Digit { .. } => {
                unreachable!("Digit not yet wired through final-witness (phase 2 WIP)")
            }
        }
    }

    pub fn to_fold_data(&self) -> FoldData {
        match self {
            Self::Binary { packed, .. } => FoldData::Binary(packed.clone()),
            Self::Ternary { n_ring, pos, neg } => FoldData::Ternary(
                almost_goldilocks_cuda::ajtai::TernaryChunks {
                    n_ring: *n_ring,
                    k_chunks: almost_goldilocks_cuda::ajtai::SPLIT_K_CHUNKS,
                    pos: pos.clone(),
                    neg: neg.clone(),
                },
            ),
        }
    }
}

/// Per-arity sub-tree proof. The fold tree buckets leaves by arity so
/// each sub-tree runs entirely under one M_k = first 2^k columns of
/// M_max (Option A — see plan §3 commentary). One tip per arity bucket.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BucketFoldProof {
    /// Native arity of this bucket (all leaves and the final witness
    /// live at this arity).
    pub arity: usize,
    pub levels: Vec<FoldLevelProof>,
    pub final_witness: FinalWitness,
    pub final_commitment: WireCommitment,
    pub final_pt: Vec<AlmostGoldilocksExt2>,
    pub final_val: AlmostGoldilocksExt2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldTreeProof {
    /// One sub-tree per distinct arity in the leaf set. Each bucket is
    /// independently verifiable: `c*_k = M_k · f*_k` ∧ `f*_k(R_k) = y*_k`.
    pub buckets: Vec<BucketFoldProof>,
    /// Ajtai seed used by every `commit_ternary` inside every bucket.
    /// The verifier needs this to re-derive chunk commitments for the
    /// split-homomorphism checks.
    pub seed: [u32; 8],
}

/// Run the fold tree end-to-end. Buckets leaves by arity and runs one
/// sub-tree per bucket under `M_arity` = first `2^arity` columns of
/// `M_max`. Each sub-tree produces one tip; the verifier checks each
/// tip independently.
pub fn prove_fold_tree(
    leaves: Vec<FoldInstance>,
    seed: Seed,
    transcript: &mut Transcript,
) -> FoldTreeProof {
    assert!(!leaves.is_empty(), "fold tree requires ≥ 1 leaf");

    // Bucket by arity, ascending. Deterministic order so the verifier
    // re-buckets identically.
    let mut by_arity: std::collections::BTreeMap<usize, Vec<FoldInstance>> =
        std::collections::BTreeMap::new();
    for leaf in leaves {
        by_arity.entry(leaf.arity).or_default().push(leaf);
    }

    transcript.append_u64(b"ft_num_buckets", by_arity.len() as u64);

    // Buckets are completely independent (each runs against its own
    // `transcript.fork(b"ft_bucket", arity)`). Bind the arities into the parent
    // transcript first so the fork seeds are determined.
    let arities_ordered: Vec<usize> = by_arity.keys().copied().collect();
    for &arity in &arities_ordered {
        transcript.append_u64(b"ft_bucket_arity", arity as u64);
    }
    let device_pool = gpu_device_pool();
    let n_devices = device_pool.len().max(1);
    let timing_on = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");

    // Unified scheduler. The fold tree's work-DAG is: buckets fully
    // independent; levels sequential within a bucket; groups within a
    // level independent (each on its own `transcript.fork(b"ft_group", gi)`). So:
    //
    //  - One PERSISTENT worker pool: `groups_per_gpu` OS threads per
    //    device, each pinned via `set_device` once and pulling group
    //    tasks off a single shared queue until it closes. Persistence
    //    keeps each worker's thread-local SP_POOL warm across levels
    //    AND buckets (measured: first group on a thread allocates
    //    ~10 GB of pooled buffers at arity 22-24; subsequent groups
    //    reuse them at +0).
    //  - One DRIVER thread per bucket: walks its levels in order,
    //    submits each level's groups to the queue, awaits results,
    //    absorbs commitments, moves to the next level.
    //
    // This replaces the old bucket-parallel / bucket-serial strategy
    // pair (and its dominance heuristic): the shared queue keeps every
    // GPU busy whether the work is spread over many balanced buckets
    // or concentrated in one dominant bucket, and there are no
    // batch-of-n_dev barriers — a finished worker immediately pulls
    // the next ready group from ANY bucket.
    //
    // `groups_per_gpu` > 1 because a group alternates CPU phases
    // (transcript, round messages) with GPU kernels — measured ~50 %
    // GPU idle with one group in flight per device. 3 concurrent
    // groups/GPU ≈ 3 × ~10 GB pooled + transients, fine on 80 GB.
    let groups_per_gpu: usize = std::env::var("ZK4_FOLD_GROUPS_PER_GPU").ok()
        .and_then(|s| s.parse::<usize>().ok()).filter(|&k| k >= 1).unwrap_or(3);
    let queues = std::sync::Arc::new(TaskQueues::new(device_pool.len()));
    let workers: Vec<std::thread::JoinHandle<()>> = device_pool.iter().enumerate()
        .flat_map(|(di, &dev)| {
            (0..groups_per_gpu).map(|_| {
                let q = queues.clone();
                std::thread::spawn(move || {
                    let _ = almost_goldilocks_cuda::set_device(dev);
                    while let Some(task) = q.pop(di) {
                        task();
                    }
                    // Free this worker's pooled GPU buffers before exit.
                    almost_goldilocks_cuda::sumcheck_prover::clear_thread_sp_pool();
                })
            }).collect::<Vec<_>>()
        }).collect();
    if timing_on {
        eprintln!("[fold_tree] scheduler: {} workers ({} per GPU x {} GPUs), {} buckets",
                  workers.len(), groups_per_gpu, n_devices, arities_ordered.len());
    }

    // Driver threads need a device too (the FINAL group of each bucket
    // runs on the driver, not through the queue — it's the level tail,
    // nothing to overlap with). Heaviest arity gets device[0], etc.
    let mut arity_with_idx: Vec<(usize, usize)> = arities_ordered.iter().copied().enumerate()
        .map(|(i, a)| (i, a)).collect();
    arity_with_idx.sort_by_key(|&(_, a)| std::cmp::Reverse(a));
    let mut device_for_bucket: Vec<i32> = vec![0; arities_ordered.len()];
    for (rank, &(orig_idx, _)) in arity_with_idx.iter().enumerate() {
        device_for_bucket[orig_idx] = device_pool[rank % n_devices];
    }

    let driver_handles: Vec<std::thread::JoinHandle<BucketFoldProof>> = arities_ordered
        .iter()
        .enumerate()
        .map(|(i, &arity)| {
            let mut bucket_t = transcript.fork(b"ft_bucket", arity);
            let leaves = by_arity.remove(&arity).unwrap_or_default();
            let device = device_for_bucket[i];
            let q = queues.clone();
            let pool = device_pool.clone();
            std::thread::spawn(move || {
                let _ = almost_goldilocks_cuda::set_device(device);
                let n_leaves = leaves.len();
                let t = std::time::Instant::now();
                let bp = prove_fold_tree_uniform(leaves, arity, seed, &mut bucket_t, &q, &pool);
                if timing_on {
                    eprintln!("[fold_tree] bucket arity={} leaves={} levels={} time={:?}",
                              arity, n_leaves, bp.levels.len(), t.elapsed());
                }
                bp
            })
        })
        .collect();
    // Join in arity-ascending order — same order the tips are absorbed.
    let buckets: Vec<BucketFoldProof> = driver_handles.into_iter()
        .map(|h| h.join().expect("fold-tree bucket driver panicked"))
        .collect();
    queues.close(); // workers drain remaining tasks (none) and exit
    for w in workers { let _ = w.join(); }
    if timing_on {
        use std::sync::atomic::Ordering;
        let hit = MIRROR_HIT.swap(0, Ordering::Relaxed);
        let xdev = MIRROR_XDEV.swap(0, Ordering::Relaxed);
        if hit + xdev > 0 {
            eprintln!("[fold_tree] mirror leaves: {} same-device (DtoD), {} cross-device (host fallback) — {:.0}% affinity",
                      hit, xdev, 100.0 * hit as f64 / (hit + xdev) as f64);
        }
    }

    // Absorb each bucket's tip into the parent transcript in arity-
    // ascending order so the verifier sees the same sequence.
    for bp in &buckets {
        transcript.append_u64_slice(b"ft_bucket_tip_comm", &bp.final_commitment.rows);
        transcript.append_ext2(b"ft_bucket_tip_val", &bp.final_val);
    }

    // Release thread-local SP_POOL buffers across every rayon worker.
    // The same-point sumcheck's pool retains DeviceBuffers keyed by
    // (device, size); across stream iterations these accumulate ~10 GB
    // / iter on Llama-2 and eventually OOM the GPU. Clearing here is
    // safe because the fold-tree has fully produced its host-side
    // output and no GPU kernels reference the pooled buffers anymore.
    // No-op cost in single-shot proving (pool is built and cleared
    // once per call). Skip via ZK4_KEEP_SP_POOL=1 if you want
    // intra-bench pool retention for some reason.
    if std::env::var("ZK4_KEEP_SP_POOL").ok().as_deref() != Some("1") {
        let _ = almost_goldilocks_cuda::synchronize();
        rayon::broadcast(|_| {
            almost_goldilocks_cuda::sumcheck_prover::clear_thread_sp_pool();
        });
        // Also clear the main thread's pool (rayon::broadcast doesn't
        // include the caller's thread).
        almost_goldilocks_cuda::sumcheck_prover::clear_thread_sp_pool();
    }

    FoldTreeProof { buckets, seed: seed.0 }
}

/// A unit of fold-tree work for the shared worker pool: one internal
/// group (same_point → multifold → split), boxed with everything it
/// needs (Arc'd level input + its transcript fork + a result sender).
/// Workers are pinned to a device before running tasks, so the task
/// body itself never calls `set_device`.
type GroupTask = Box<dyn FnOnce() + Send + 'static>;

/// Per-device task queues with work stealing. A group whose inputs mostly
/// live on device d (chunk mirrors from its parents) is pushed to queue d,
/// so the worker pinned to d runs it and the assembly is all same-device
/// DtoD. Level-0 groups (host leaves, no mirrors) get a BLOCK preference
/// (`gi · n_dev / n_groups`) so consecutive groups — whose chunks feed the
/// same next-level group — land on the same device, keeping affinity
/// through every level. Workers drain their own queue first, then the
/// no-preference queue, then steal (work conservation beats affinity at
/// level tails; a stolen group just pays the host-upload fallback).
///
/// Tasks are served FIFO by default. An arity-priority mode exists behind
/// `ZK4_FOLD_PRIORITY=1` and measured slower — see `priority_enabled`.
/// Ordering is free of soundness consequences either way: every group runs on
/// its own `fork(b"ft_group", gi)` and every bucket on `fork(b"ft_bucket", arity)`,
/// which is what already makes work stealing safe. Ties break FIFO by sequence
/// number so behaviour stays deterministic.
/// Arity-priority scheduling, OFF by default because it MEASURED SLOWER.
///
/// The idea was that the fold tree's wall clock is its slowest bucket (at
/// table_commit_log 12 on llama2 8L/seq64, buckets ran 2.4s to 81.4s against an
/// 81.7s fold tree), so serving the top-arity bucket first should shorten the
/// critical path. It does the opposite, because it fights the device affinity
/// this scheduler is built around: per-device queues exist so a group's chunk
/// mirrors stay on the device that produced them, and reordering across buckets
/// strands those mirrors into host-copy fallbacks that cost more than the
/// critical-path saving gains. Starving the small buckets also delays their own
/// sequential levels.
///
/// Measured, 4xA100, interleaved reps, fold tree (FIFO vs priority):
///   tcl 12 rep1  78.07s vs 84.03s     tcl 12 rep2  75.73s vs 80.43s
///   tcl  6 rep1  14.42s vs 15.32s
/// FIFO won every pair. Kept behind `ZK4_FOLD_PRIORITY=1` so the experiment is
/// reproducible rather than re-derived; do not enable it without re-measuring.
fn priority_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZK4_FOLD_PRIORITY").ok().as_deref() == Some("1"))
}

struct QEntry {
    prio: usize,
    seq: u64,
    task: GroupTask,
}
impl PartialEq for QEntry {
    fn eq(&self, o: &Self) -> bool { (self.prio, self.seq) == (o.prio, o.seq) }
}
impl Eq for QEntry {}
impl PartialOrd for QEntry {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for QEntry {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // BinaryHeap pops the maximum: higher prio wins, then LOWER seq wins.
        self.prio.cmp(&o.prio).then_with(|| o.seq.cmp(&self.seq))
    }
}

struct TaskQueues {
    /// `queues[d]` prefers device-pool index d; `queues[n_dev]` = no pref.
    queues: Vec<std::sync::Mutex<std::collections::BinaryHeap<QEntry>>>,
    seq: std::sync::atomic::AtomicU64,
    cv: std::sync::Condvar,
    gate: std::sync::Mutex<()>,
    open: std::sync::atomic::AtomicBool,
    pending: std::sync::atomic::AtomicUsize,
}

impl TaskQueues {
    fn new(n_dev: usize) -> Self {
        TaskQueues {
            queues: (0..=n_dev)
                .map(|_| std::sync::Mutex::new(std::collections::BinaryHeap::new()))
                .collect(),
            seq: std::sync::atomic::AtomicU64::new(0),
            cv: std::sync::Condvar::new(),
            gate: std::sync::Mutex::new(()),
            open: std::sync::atomic::AtomicBool::new(true),
            pending: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    fn push(&self, pref: Option<usize>, prio: usize, task: GroupTask) {
        use std::sync::atomic::Ordering;
        self.pending.fetch_add(1, Ordering::SeqCst);
        let n_dev = self.queues.len() - 1;
        let qi = pref.filter(|&p| p < n_dev).unwrap_or(n_dev);
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        // `ZK4_FOLD_PRIORITY=0` restores plain FIFO by flattening every priority
        // to 0, so the arity-priority change can be A/B'd without a rebuild.
        let prio = if priority_enabled() { prio } else { 0 };
        self.queues[qi].lock().unwrap().push(QEntry { prio, seq, task });
        self.cv.notify_all();
    }
    fn pop(&self, own: usize) -> Option<GroupTask> {
        use std::sync::atomic::Ordering;
        let n_q = self.queues.len();
        loop {
            // Own queue → no-pref queue → steal round-robin from others.
            let mut order: Vec<usize> = vec![own, n_q - 1];
            order.extend((0..n_q - 1).filter(|&q| q != own));
            for qi in order {
                if let Some(e) = self.queues[qi].lock().unwrap().pop() {
                    self.pending.fetch_sub(1, Ordering::SeqCst);
                    return Some(e.task);
                }
            }
            if !self.open.load(Ordering::SeqCst) && self.pending.load(Ordering::SeqCst) == 0 {
                return None;
            }
            // 1 ms poll guards against lost wakeups without busy-spinning.
            let g = self.gate.lock().unwrap();
            let _ = self.cv
                .wait_timeout(g, std::time::Duration::from_millis(1))
                .unwrap();
        }
    }
    fn close(&self) {
        self.open.store(false, std::sync::atomic::Ordering::SeqCst);
        self.cv.notify_all();
    }
}

/// Affinity effectiveness counters (per-leaf, ternary assembly only):
/// hit = same-device mirror DtoD; xdev = mirror exists on another device
/// (host-upload fallback, an affinity miss). Reported under ZK4_TIMING.
static MIRROR_HIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIRROR_XDEV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Device-side residue of one group's split: the 13 chunk plane buffers,
/// kept alive on the device that produced them so next-level groups can
/// assemble their inputs DtoD (same device) or via UVA peer copy (other
/// device) instead of re-uploading the host copy over pageable PCIe.
pub(crate) struct DevChunks {
    pub device: i32,
    pub n_ring: usize,
    pub pos: almost_goldilocks_cuda::memory::DeviceBuffer<u64>, // 13 × n_ring
    pub neg: almost_goldilocks_cuda::memory::DeviceBuffer<u64>,
}

/// Mirror entry carried alongside each `FoldInstance` in the working set:
/// `(parent buffers, chunk index within them)`. `None` for level-0 leaves
/// (host-only) and host-fallback groups' outputs.
type ChunkMirror = (std::sync::Arc<DevChunks>, usize);

/// Single-arity (uniform) fold tree. Internal — exposed only to the
/// bucket dispatcher above. All leaves must have `arity == max_num_vars`.
///
/// `task_tx` is the shared group-task queue serviced by the persistent
/// per-GPU worker pool in [`prove_fold_tree`]. Each level submits all
/// its groups and awaits their results in group order; the workers
/// interleave this bucket's groups with every other bucket's, so GPUs
/// stay busy even when one bucket dominates or a level's tail narrows.
fn prove_fold_tree_uniform(
    mut working_set: Vec<FoldInstance>,
    max_num_vars: usize,
    seed: Seed,
    transcript: &mut Transcript,
    queues: &TaskQueues,
    device_pool: &[i32],
) -> BucketFoldProof {
    assert!(!working_set.is_empty(), "uniform fold tree requires ≥ 1 leaf");
    for (i, w) in working_set.iter().enumerate() {
        assert_eq!(w.arity, max_num_vars,
                   "uniform sub-tree leaf {} arity {} != bucket arity {}",
                   i, w.arity, max_num_vars);
    }

    transcript.append_u64(b"ft_max_num_vars", max_num_vars as u64);
    transcript.append_u64(b"ft_num_leaves", working_set.len() as u64);

    let mut levels: Vec<FoldLevelProof> = Vec::new();
    let mut level_idx: u64 = 0;
    // Device mirrors ride alongside the working set (parallel vec): level-0
    // leaves are host-only (None); device-resident groups attach mirrors to
    // their output chunks so the next level assembles inputs without PCIe.
    let mut working_mirrors: Vec<Option<ChunkMirror>> = vec![None; working_set.len()];

    // Internal levels: each group → 13 ternary chunks. Each group runs
    // against a FORKED transcript (`parent.fork(b"ft_group", gi)`) so groups
    // within a level are independent — enabling future thread / GPU
    // parallelism. After a level completes we absorb every group's
    // multifold-combined commitment + chunk commitments back into the
    // parent in deterministic (group_idx ascending) order so the next
    // level's challenges Fiat-Shamir-bind to this level's outputs.
    let per_group = group_size_for_arity(max_num_vars);
    while working_set.len() > per_group {
        transcript.append_u64(b"ft_level", level_idx);

        // Per-group work is independent (each group runs on its own
        // `transcript.fork(b"ft_group", gi)`). Submit every group to the shared
        // worker queue — no batching, no barriers within the level. A
        // worker that finishes a slow group's neighbor immediately
        // pulls the next group (from this bucket or any other). The
        // level input moves into an Arc so tasks borrow nothing from
        // this stack frame; each task gets its own result channel and
        // we await them in group order to keep the absorb sequence
        // deterministic.
        let level_in = std::sync::Arc::new(std::mem::take(&mut working_set));
        let level_mirrors = std::sync::Arc::new(std::mem::take(&mut working_mirrors));
        let n = level_in.len();
        let n_groups = n.div_ceil(per_group);
        let mut result_rxs = Vec::with_capacity(n_groups);
        for gi in 0..n_groups {
            let (rtx, rrx) = std::sync::mpsc::channel::<(
                FoldGroupProof, Vec<FoldInstance>, Vec<Option<ChunkMirror>>,
            )>();
            let input = level_in.clone();
            let mirrors = level_mirrors.clone();
            let mut group_transcript = transcript.fork(b"ft_group", gi);
            let start = gi * per_group;
            let end = ((gi + 1) * per_group).min(n);
            // Affinity preference: majority device of the group's input
            // mirrors (level 1+); block assignment for mirror-less groups
            // (level 0) so consecutive groups co-locate with the next-level
            // group that consumes their chunks.
            let pref = {
                let mut counts = vec![0usize; device_pool.len()];
                let mut any = false;
                for mi in &level_mirrors[start..end] {
                    if let Some((dc, _)) = mi {
                        if let Some(pi) = device_pool.iter().position(|&d| d == dc.device) {
                            counts[pi] += 1;
                            any = true;
                        }
                    }
                }
                if any {
                    counts.iter().enumerate().max_by_key(|&(_, c)| *c).map(|(i, _)| i)
                } else {
                    Some(gi * device_pool.len().max(1) / n_groups.max(1))
                }
            };
            // Priority = bucket arity, so the critical-path bucket is served first.
            queues.push(pref, max_num_vars, Box::new(move || {
                let r = run_internal_group(
                    &input[start..end], &mirrors[start..end],
                    max_num_vars, seed, &mut group_transcript);
                let _ = rtx.send(r);
            }));
            result_rxs.push(rrx);
        }
        let mut groups_out: Vec<FoldGroupProof> = Vec::with_capacity(n_groups);
        let mut next_level: Vec<FoldInstance> = Vec::new();
        let mut next_mirrors: Vec<Option<ChunkMirror>> = Vec::new();
        for rrx in result_rxs {
            let (gp, chunks, cmirrors) = rrx.recv()
                .expect("fold-tree group worker dropped its result (panicked?)");
            groups_out.push(gp);
            next_level.extend(chunks);
            next_mirrors.extend(cmirrors);
        }
        // Absorb per-group commitments into the parent transcript so
        // downstream Fiat-Shamir challenges depend on this level's
        // output. Deterministic order = same group order both sides see.
        for (gi, gp) in groups_out.iter().enumerate() {
            absorb_group_commitments(transcript, gi, gp);
        }
        levels.push(FoldLevelProof { groups: groups_out });
        working_set = next_level;
        working_mirrors = next_mirrors;
        level_idx += 1;
    }

    // Final level: ≤ 63 instances, single same_point + multifold, no split.
    transcript.append_u64(b"ft_level", level_idx);
    let mut group_transcript = transcript.fork(b"ft_group", 0);
    let (group_proof, final_inst) = run_final_group(&working_set, max_num_vars, &mut group_transcript);
    absorb_group_commitments(transcript, 0, &group_proof);
    levels.push(FoldLevelProof { groups: vec![group_proof] });

    BucketFoldProof {
        arity: max_num_vars,
        levels,
        final_witness: FinalWitness::from_fold_data(&final_inst.data),
        final_commitment: WireCommitment::from_ring(&final_inst.commitment),
        final_pt: final_inst.claim_pt,
        final_val: final_inst.claim_val,
    }
}

/// Run one internal group: same_point → multifold → split. Output is
/// 13 `FoldInstance`s (one per ternary chunk) plus their device mirrors.
///
/// Dispatches to the DEVICE-RESIDENT path when eligible (all-binary or
/// all-single-chunk-ternary group, arity within the GPU window, not
/// disabled via `ZK4_DEVICE_RESIDENT_FOLD=0`): witness data is assembled
/// on the local GPU once and flows through same-point → multifold →
/// split-decompose → chunk-evals without any host round-trip; only the
/// chunk host copies (next-level fallback + final witness) come back.
/// Any GPU failure restores the transcript snapshot and reruns the whole
/// group on the host path — output proofs are valid either way.
fn run_internal_group(
    group: &[FoldInstance],
    mirrors: &[Option<ChunkMirror>],
    max_num_vars: usize,
    seed: Seed,
    transcript: &mut Transcript,
) -> (FoldGroupProof, Vec<FoldInstance>, Vec<Option<ChunkMirror>>) {
    let devres_on = std::env::var("ZK4_DEVICE_RESIDENT_FOLD").ok().as_deref() != Some("0");
    let devres_min = std::env::var("ZK4_DEVRES_MIN_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(18);
    let gpu_arity_cap = std::env::var("ZK4_MULTIFOLD_GPU_MAX_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(24);
    let all_binary = group.iter().all(|i| matches!(i.data, FoldData::Binary(_)));
    let all_tern1 = group.iter().all(|i| matches!(
        &i.data, FoldData::Ternary(c) if c.k_chunks == 1));
    let eligible = devres_on
        && max_num_vars >= devres_min
        && max_num_vars <= gpu_arity_cap
        && (all_binary || all_tern1);
    if eligible {
        let snapshot = transcript.clone();
        match run_internal_group_dev(group, mirrors, max_num_vars, seed, transcript) {
            Ok(r) => return r,
            Err(e) => {
                eprintln!("[fold_tree] device-resident group failed ({:?}) — host fallback", e);
                *transcript = snapshot;
            }
        }
    }
    let (gp, chunks) = run_internal_group_host(group, max_num_vars, seed, transcript);
    let n = chunks.len();
    (gp, chunks, vec![None; n])
}

/// Device-resident internal group. See [`run_internal_group`] for the
/// contract; returns `Err` (caller restores transcript + falls back) on
/// any GPU failure.
fn run_internal_group_dev(
    group: &[FoldInstance],
    mirrors: &[Option<ChunkMirror>],
    max_num_vars: usize,
    seed: Seed,
    transcript: &mut Transcript,
) -> Result<
    (FoldGroupProof, Vec<FoldInstance>, Vec<Option<ChunkMirror>>),
    almost_goldilocks_cuda::error::CudaError,
> {
    use almost_goldilocks_cuda::memory::DeviceBuffer;
    use almost_goldilocks_cuda::sumcheck_prover::{pool_take, pool_return};
    use crate::util::arith::{ext2_add, ext2_mul, ext2_sub};
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    let arity = max_num_vars;
    let n_ring = 1usize << (arity - 6); // eligibility guarantees arity ≥ 18
    let m = group.len();
    let all_binary = group.iter().all(|i| matches!(i.data, FoldData::Binary(_)));
    let timing_grp = std::env::var("ZK4_TIMING_GROUP").is_ok();
    let t0 = std::time::Instant::now();

    // ---- 1. Assemble the group's witness concat on the LOCAL device,
    // once. Sources, in preference order: parent chunk mirror on any
    // device (DtoD / UVA peer copy), else host data (pageable upload —
    // level-0 leaves and host-fallback parents).
    enum Concat {
        Bin(DeviceBuffer<u64>),
        Tern(DeviceBuffer<u64>, DeviceBuffer<u64>),
    }
    let concat = if all_binary {
        let mut d = pool_take(m * n_ring)?;
        for (i, inst) in group.iter().enumerate() {
            match &inst.data {
                FoldData::Binary(v) => d.write_slice_at(i * n_ring, v)?,
                _ => unreachable!(),
            }
        }
        Concat::Bin(d)
    } else {
        let local_dev = almost_goldilocks_cuda::current_device();
        let mut dp = pool_take(m * n_ring)?;
        let mut dn = pool_take(m * n_ring)?;
        for (i, inst) in group.iter().enumerate() {
            match &mirrors[i] {
                // Same-device mirror: zero-PCIe DtoD copy.
                Some((dc, idx)) if dc.device == local_dev => {
                    MIRROR_HIT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    dp.copy_range_from_device(i * n_ring, &dc.pos, idx * n_ring, n_ring)?;
                    dn.copy_range_from_device(i * n_ring, &dc.neg, idx * n_ring, n_ring)?;
                }
                // Cross-device mirror: fall back to the host copy. Both
                // cudaMemcpy-DtoD-across-devices and cudaMemcpyPeer
                // SEGFAULT inside libcuda on this driver (CUDA 13.1) when
                // racing the fold tree's concurrent kernel/alloc traffic —
                // canary DtoH reads of the same pointers succeed, and an
                // isolated stress test of the identical copy pattern
                // passes, so this is a driver concurrency bug, not a
                // lifetime bug. The host copy costs one pageable upload —
                // exactly what the pre-device-resident path paid.
                other => {
                    if other.is_some() {
                        MIRROR_XDEV.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    match &inst.data {
                        FoldData::Ternary(c) => {
                            dp.write_slice_at(i * n_ring, &c.pos)?;
                            dn.write_slice_at(i * n_ring, &c.neg)?;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        Concat::Tern(dp, dn)
    };

    let t_asm = t0.elapsed();
    // ---- 2. Same-point sumcheck from the device concat (falls back to
    // the CPU prover internally on GPU error — transcript stays valid).
    // `d_shared_eq` is the eq table at shared_r built for the f_evals
    // recovery — reused below for the chunk evals (same point).
    let (sp_proof, shared_r, d_shared_eq) = match &concat {
        Concat::Bin(d) => same_point_sumcheck::prove_same_point_gpu_batched_dev(
            group, same_point_sumcheck::SpDevInput::Binary(d), arity, transcript),
        Concat::Tern(dp, dn) => same_point_sumcheck::prove_same_point_gpu_batched_dev(
            group, same_point_sumcheck::SpDevInput::Ternary(dp, dn), arity, transcript),
    };

    let t_sp = t0.elapsed();
    // The multifold needs the leaves' claims promoted to shared_r — but
    // only their COMMITMENTS and the γ challenges enter the proof, so no
    // promoted FoldInstance materialization is needed on this path.

    // ---- 3. Multifold from the same concat; wide output stays on device.
    let mf_result = match &concat {
        Concat::Bin(d) =>
            multifold::prove_multifold_defer_y_dev(group, Some(d), None, transcript),
        Concat::Tern(dp, dn) =>
            multifold::prove_multifold_defer_y_dev(group, None, Some((dp, dn)), transcript),
    };
    // Concat buffers are dead after the fused kernel launch (stream-ordered).
    match concat {
        Concat::Bin(d) => pool_return(d),
        Concat::Tern(a, b) => { pool_return(a); pool_return(b); }
    }
    let (d_wide, mut mf_proof) = mf_result?;
    let t_mf = t0.elapsed();

    // ---- 4. Split decomposition on device + Ajtai chunk commits.
    let (d_cpos, d_cneg) =
        almost_goldilocks_cuda::ajtai::wide_to_ternary_device(&d_wide, n_ring)?;
    drop(d_wide);
    let dev_chunks = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring,
        k_chunks: almost_goldilocks_cuda::ajtai::SPLIT_K_CHUNKS,
        pos: d_cpos,
        neg: d_cneg,
    };
    let chunk_commits =
        almost_goldilocks_cuda::ajtai::commit_ternary(seed, &dev_chunks, None)?;
    let split_proof = SplitProof {
        chunk_commitments: chunk_commits.iter().map(WireCommitment::from_ring).collect(),
    };
    let t_commit = t0.elapsed();

    // ---- 5. Chunk evals at shared_r straight from the device planes.
    let k = almost_goldilocks_cuda::ajtai::SPLIT_K_CHUNKS;
    let evals = match &d_shared_eq {
        Some(d_eq) => almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_with_eq_dev(
            d_eq, arity, &[(&dev_chunks.pos, k), (&dev_chunks.neg, k)])?,
        None => almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_from_dev(
            &shared_r, &[(&dev_chunks.pos, k), (&dev_chunks.neg, k)])?,
    };
    let chunk_evals: Vec<AlmostGoldilocksExt2> =
        (0..k).map(|i| ext2_sub(evals[i], evals[k + i])).collect();

    // combined_y = Σ 2^i · chunk_eval[i] (split identity) — patches the
    // defer-y placeholder; verifier reads chunk_claim_vals only.
    let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
    let mut combined_y = AlmostGoldilocksExt2::zero();
    let mut pow_two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1));
    for ce in &chunk_evals {
        combined_y = ext2_add(combined_y, ext2_mul(pow_two, *ce));
        pow_two = ext2_mul(pow_two, two);
    }
    mf_proof.combined_claim = combined_y;

    let t_eval = t0.elapsed();
    // ---- 6. Host copies (next-level fallback, final witness, cross-run
    // serialization) + the device mirror for next-level assembly.
    let host_pos = dev_chunks.pos.to_vec()?;
    let host_neg = dev_chunks.neg.to_vec()?;
    let arc = std::sync::Arc::new(DevChunks {
        device: almost_goldilocks_cuda::current_device(),
        n_ring,
        pos: dev_chunks.pos,
        neg: dev_chunks.neg,
    });
    let mut chunk_instances: Vec<FoldInstance> = Vec::with_capacity(k);
    let mut chunk_mirrors: Vec<Option<ChunkMirror>> = Vec::with_capacity(k);
    for ci in 0..k {
        let single = almost_goldilocks_cuda::ajtai::TernaryChunks {
            n_ring,
            k_chunks: 1,
            pos: host_pos[ci * n_ring..(ci + 1) * n_ring].to_vec(),
            neg: host_neg[ci * n_ring..(ci + 1) * n_ring].to_vec(),
        };
        chunk_instances.push(FoldInstance {
            commitment: chunk_commits[ci].clone(),
            data: FoldData::Ternary(single),
            arity,
            claim_pt: shared_r.clone(),
            claim_val: chunk_evals[ci],
        });
        chunk_mirrors.push(Some((arc.clone(), ci)));
    }

    let proof = FoldGroupProof {
        same_point: sp_proof,
        multifold: mf_proof,
        split: Some(split_proof),
        chunk_claim_vals: chunk_evals,
    };
    if timing_grp {
        eprintln!("[group dev arity={} M={} bin={}] asm={:?} sp={:?} mf={:?} w2t+commit={:?} eval={:?} dl={:?}",
            arity, m, all_binary, t_asm, t_sp - t_asm, t_mf - t_sp,
            t_commit - t_mf, t_eval - t_commit, t0.elapsed() - t_eval);
    }
    Ok((proof, chunk_instances, chunk_mirrors))
}

/// Host-path internal group (pre-device-resident behavior, also the
/// fallback for digit groups, oversized arities, and GPU failures).
fn run_internal_group_host(
    group: &[FoldInstance],
    max_num_vars: usize,
    seed: Seed,
    transcript: &mut Transcript,
) -> (FoldGroupProof, Vec<FoldInstance>) {
    // Step 1: same-point sumcheck. Per-arity buckets are uniform-arity.
    // CPU with rayon parallelism currently outperforms the per-leaf GPU
    // dispatch (existing `GpuSumcheckStateExt2` has too much per-call
    // launch overhead — a custom sum-of-products kernel would be needed
    // to win here). See `prove_same_point_gpu` for the alternative path.
    // Per-arity-bucket invariant: every leaf has arity == max_num_vars,
    // so we can use the GPU batched path (one kernel launch per round,
    // all leaves' (eq, f) co-resident on device). Falls back to CPU for
    // small arities (< 16) where launch overhead dominates.
    let uniform_arity = group.iter().all(|i| i.arity == max_num_vars);
    // GPU batched same-point: enabled for arity ≥ 18 by default. With
    // on-device eq construction (`ext2_eq_dp_all_device`) and on-device
    // binary lift (`aext2_batched_lift_binary_kernel`), host upload is
    // just the per-leaf claim_pts (~arity Ext2) + packed bits (~n/64 u64s),
    // not the lifted Ext2 tables — net-positive vs CPU at arity ≥ 18.
    let min_arity = std::env::var("ZK4_GPU_SP_MIN_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(18);
    // Cap above which buckets fall back to the CPU same-point path. Default 24.
    //
    // The original justification was memory — the old per-leaf interleaved state
    // cost `2 · 2 · M · 2^A · 16 B` and blew an 80 GB device around arity 24.
    // Shared-eq (default on) obsoleted that: state is now
    // `num_unique · 2^arity · 64 B`, num_unique <= 3, so ~13 GB at arity 26 and
    // ~52 GB at 28. Both fit — but do NOT raise the cap on that headroom.
    //
    // 24 is still correct for a different reason: this GPU kernel is DENSE over
    // 2^arity, while the CPU path above the cap is the SPARSE one
    // (`FState::Sparse`, O(nnz)). Lookup auxiliaries are function-graph sparse,
    // so sparse-CPU beats dense-GPU once the cube is large. Measured on llama2
    // 8L/seq64, 4xA100, fold tree / total prove: at arity 26, cap 24 gives
    // 21.8s/44.0s vs cap 26 at 27.1s/49.6s; at arity 28, cap 24 gives
    // 39.3s/58.6s vs cap 28 at 54.4s/73.6s. Raising it cost 24% and 39%.
    //
    // To get the GPU onto this work, bring arity DOWN under 24 via
    // `table_commit_log` (see `Dag::report_lookup_arities`), not the cap up.
    let max_arity = std::env::var("ZK4_GPU_SP_MAX_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(24);
    let use_gpu = uniform_arity
        && max_num_vars >= min_arity
        && max_num_vars <= max_arity
        && !group.is_empty();
    let trace_mem = std::env::var("ZK4_PROVE_MEM_TRACE").ok().as_deref() == Some("1");
    let mem = || almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);
    let mem_enter = if trace_mem { mem() } else { 0 };
    let _t_sp = std::time::Instant::now();
    let (sp_proof, shared_r) = if use_gpu {
        same_point_sumcheck::prove_same_point_gpu_batched(group, max_num_vars, transcript)
    } else {
        same_point_sumcheck::prove_same_point(group, max_num_vars, transcript)
    };
    let _dt_sp = _t_sp.elapsed();
    let mem_after_sp = if trace_mem { mem() } else { 0 };
    // Lift each instance's claim to (shared_r, f_evals[i]).
    let promoted: Vec<FoldInstance> = group.iter().enumerate().map(|(i, inst)| {
        FoldInstance {
            commitment: inst.commitment.clone(),
            data: inst.data.clone(),
            arity: max_num_vars,
            claim_pt: shared_r.clone(),
            claim_val: sp_proof.f_evals[i],
        }
    }).collect();

    // Step 2: multifold. Defer `combined_y` — we recover it for free below
    // from the split-chunk evals via `combined = Σ 2^i·chunk_i`, which lets
    // us skip building/downloading the 2^arity host eq table entirely.
    let _t_mf = std::time::Instant::now();
    let (combined, mut mf_proof) = multifold::prove_multifold_defer_y(&promoted, transcript);
    let _dt_mf = _t_mf.elapsed();
    let mem_after_mf = if trace_mem { mem() } else { 0 };

    // Step 3: split into 13 ternary chunks + commit each via Ajtai.
    let _t_split = std::time::Instant::now();
    let (chunk_data, split_proof) = split::prove_split(&combined, seed);
    let _dt_split = _t_split.elapsed();
    let mem_after_split = if trace_mem { mem() } else { 0 };

    let chunks = match &chunk_data {
        FoldData::Ternary(c) => c.clone(),
        FoldData::Binary(_) => unreachable!("split always produces ternary"),
        FoldData::Digit { .. } => unreachable!("split always produces ternary"),
    };
    let chunk_commits: Vec<RingCommitment> = split_proof.chunk_commitments.iter()
        .map(|w| w.to_ring()).collect();

    // Evaluate all 13 chunks at `shared_r` on the GPU: each chunk's pos/neg
    // are packed bit-planes, so `chunk_i(R) = eval(pos_i) - eval(neg_i)`.
    // `eval_binary_planes_device` builds eq(R) once on device and returns
    // only the 26 scalar plane evals (no 64 MB eq-table D2H copy — that copy
    // was the dominant per-group cost, ~36 ms at arity 22). Then
    // `combined_y = Σ 2^i·chunk_eval[i]` reconstructs the multifold claim.
    // Falls back to a host eq build + dot product on any CUDA error.
    let _t_chunks = std::time::Instant::now();
    let chunk_evals = eval_ternary_chunks(&chunks, &shared_r, max_num_vars);
    if trace_mem {
        let mem_exit = mem();
        eprintln!(
            "      [group arity={} mem] enter {} -> sp {} (+{}) -> mf {} (+{}) -> split {} (+{}) -> exit {} (+{}) [total +{}]",
            max_num_vars, mem_enter,
            mem_after_sp, mem_after_sp as i64 - mem_enter as i64,
            mem_after_mf, mem_after_mf as i64 - mem_after_sp as i64,
            mem_after_split, mem_after_split as i64 - mem_after_mf as i64,
            mem_exit, mem_exit as i64 - mem_after_split as i64,
            mem_exit as i64 - mem_enter as i64,
        );
    }

    // combined_y = Σ_i 2^i · chunk_eval[i]  (the split decomposition
    // identity, evaluated). Verifier ignores an internal group's
    // combined_claim, but we set it correctly anyway.
    let mut combined_y = AlmostGoldilocksExt2::zero();
    let mut pow_two = AlmostGoldilocksExt2::one();
    let two = AlmostGoldilocksExt2::from_base(
        almost_goldilocks_cuda::field::AlmostGoldilocksField(2));
    for ce in &chunk_evals {
        combined_y = crate::util::arith::ext2_add(
            combined_y, crate::util::arith::ext2_mul(pow_two, *ce));
        pow_two = crate::util::arith::ext2_mul(pow_two, two);
    }
    mf_proof.combined_claim = combined_y;

    let mut chunk_instances: Vec<FoldInstance> = Vec::with_capacity(chunks.k_chunks);
    for ci in 0..chunks.k_chunks {
        let src_start = ci * chunks.n_ring;
        let src_end = src_start + chunks.n_ring;
        let single_chunk = almost_goldilocks_cuda::ajtai::TernaryChunks {
            n_ring: chunks.n_ring,
            k_chunks: 1,
            pos: chunks.pos[src_start..src_end].to_vec(),
            neg: chunks.neg[src_start..src_end].to_vec(),
        };
        chunk_instances.push(FoldInstance {
            commitment: chunk_commits[ci].clone(),
            data: FoldData::Ternary(single_chunk),
            arity: max_num_vars,
            claim_pt: shared_r.clone(),
            claim_val: chunk_evals[ci],
        });
    }

    let proof = FoldGroupProof {
        same_point: sp_proof,
        multifold: mf_proof,
        split: Some(split_proof),
        chunk_claim_vals: chunk_evals,
    };
    if std::env::var("ZK4_TIMING_GROUP").is_ok() {
        eprintln!("[group internal arity={} M={}] sp={:?} mf={:?} split={:?} chunk_evals={:?}",
            max_num_vars, group.len(), _dt_sp, _dt_mf, _dt_split, _t_chunks.elapsed());
    }
    (proof, chunk_instances)
}

/// Evaluate all `k_chunks` ternary chunks at `shared_r`. Each chunk's
/// `pos`/`neg` are packed bit-planes, so `chunk_i(R) = eval(pos_i) -
/// eval(neg_i)`. On GPU we evaluate all `2·k_chunks` planes in one
/// `eval_binary_planes_device` call (builds eq(R) on-device, returns only
/// scalars). Host fallback builds the eq table and dot-products per chunk.
fn eval_ternary_chunks(
    chunks: &almost_goldilocks_cuda::ajtai::TernaryChunks,
    shared_r: &[AlmostGoldilocksExt2],
    arity: usize,
) -> Vec<AlmostGoldilocksExt2> {
    let k = chunks.k_chunks;
    let n_ring = chunks.n_ring;
    if arity >= 12 {
        let mut plane_refs: Vec<&[u64]> = Vec::with_capacity(2 * k);
        for ci in 0..k {
            plane_refs.push(&chunks.pos[ci * n_ring..(ci + 1) * n_ring]);
            plane_refs.push(&chunks.neg[ci * n_ring..(ci + 1) * n_ring]);
        }
        if let Ok(scalars) =
            almost_goldilocks_cuda::eq_lagrange::eval_binary_planes_device(shared_r, &plane_refs)
        {
            return (0..k)
                .map(|ci| crate::util::arith::ext2_sub(scalars[2 * ci], scalars[2 * ci + 1]))
                .collect();
        }
    }
    // Host fallback: build eq(R) once, dot-product each chunk's pos/neg.
    let eq = crate::poly::evaluate_lagrange_basis_ext2(shared_r);
    (0..k)
        .map(|ci| {
            let single = almost_goldilocks_cuda::ajtai::TernaryChunks {
                n_ring,
                k_chunks: 1,
                pos: chunks.pos[ci * n_ring..(ci + 1) * n_ring].to_vec(),
                neg: chunks.neg[ci * n_ring..(ci + 1) * n_ring].to_vec(),
            };
            FoldData::Ternary(single).evaluate_with_eq(&eq)
        })
        .collect()
}

/// Run the final group: same_point → multifold → done. No split.
fn run_final_group(
    group: &[FoldInstance],
    max_num_vars: usize,
    transcript: &mut Transcript,
) -> (FoldGroupProof, FoldInstance) {
    let timing = std::env::var("ZK4_TIMING").is_ok() && max_num_vars >= 18;
    let t0 = std::time::Instant::now();
    // Per-arity-bucket invariant: every leaf has arity == max_num_vars,
    // so we can use the GPU batched path (one kernel launch per round,
    // all leaves' (eq, f) co-resident on device). Falls back to CPU for
    // small arities (< 16) where launch overhead dominates.
    let uniform_arity = group.iter().all(|i| i.arity == max_num_vars);
    // GPU batched same-point: enabled for arity ≥ 18 by default. With
    // on-device eq construction (`ext2_eq_dp_all_device`) and on-device
    // binary lift (`aext2_batched_lift_binary_kernel`), host upload is
    // just the per-leaf claim_pts (~arity Ext2) + packed bits (~n/64 u64s),
    // not the lifted Ext2 tables — net-positive vs CPU at arity ≥ 18.
    let min_arity = std::env::var("ZK4_GPU_SP_MIN_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(18);
    // Cap above which buckets fall back to the CPU same-point path. Default 24.
    //
    // The original justification was memory — the old per-leaf interleaved state
    // cost `2 · 2 · M · 2^A · 16 B` and blew an 80 GB device around arity 24.
    // Shared-eq (default on) obsoleted that: state is now
    // `num_unique · 2^arity · 64 B`, num_unique <= 3, so ~13 GB at arity 26 and
    // ~52 GB at 28. Both fit — but do NOT raise the cap on that headroom.
    //
    // 24 is still correct for a different reason: this GPU kernel is DENSE over
    // 2^arity, while the CPU path above the cap is the SPARSE one
    // (`FState::Sparse`, O(nnz)). Lookup auxiliaries are function-graph sparse,
    // so sparse-CPU beats dense-GPU once the cube is large. Measured on llama2
    // 8L/seq64, 4xA100, fold tree / total prove: at arity 26, cap 24 gives
    // 21.8s/44.0s vs cap 26 at 27.1s/49.6s; at arity 28, cap 24 gives
    // 39.3s/58.6s vs cap 28 at 54.4s/73.6s. Raising it cost 24% and 39%.
    //
    // To get the GPU onto this work, bring arity DOWN under 24 via
    // `table_commit_log` (see `Dag::report_lookup_arities`), not the cap up.
    let max_arity = std::env::var("ZK4_GPU_SP_MAX_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(24);
    let use_gpu = uniform_arity
        && max_num_vars >= min_arity
        && max_num_vars <= max_arity
        && !group.is_empty();
    let t0 = std::time::Instant::now();
    let (sp_proof, shared_r) = if use_gpu {
        same_point_sumcheck::prove_same_point_gpu_batched(group, max_num_vars, transcript)
    } else {
        same_point_sumcheck::prove_same_point(group, max_num_vars, transcript)
    };
    let t1 = std::time::Instant::now();
    let promoted: Vec<FoldInstance> = group.iter().enumerate().map(|(i, inst)| {
        FoldInstance {
            commitment: inst.commitment.clone(),
            data: inst.data.clone(),
            arity: max_num_vars,
            claim_pt: shared_r.clone(),
            claim_val: sp_proof.f_evals[i],
        }
    }).collect();
    let t2 = std::time::Instant::now();
    let (combined, mf_proof) = multifold::prove_multifold(&promoted, transcript);
    let t3 = std::time::Instant::now();
    let _ = t0;
    if timing {
        eprintln!("[final_group arity={} leaves={}] sp={:?} promote={:?} multifold={:?}",
            max_num_vars, group.len(), t1 - t0, t2 - t1, t3 - t2);
    }
    let proof = FoldGroupProof {
        same_point: sp_proof,
        multifold: mf_proof,
        split: None,
        chunk_claim_vals: Vec::new(),
    };
    (proof, combined)
}

/// Verify a fold-tree proof: bucket leaves by arity, verify each sub-tree
/// independently, mirror the prover's transcript exactly.
pub fn verify_fold_tree(
    leaves: &[(RingCommitment, usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)],
    proof: &FoldTreeProof,
    transcript: &mut Transcript,
) -> Result<(), crate::fold::FoldTreeError> {
    use crate::fold::FoldTreeError as E;
    assert!(!leaves.is_empty(), "fold-tree verifier requires ≥ 1 leaf");
    let mut by_arity: std::collections::BTreeMap<usize,
        Vec<(RingCommitment, usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)>> =
        std::collections::BTreeMap::new();
    for leaf in leaves {
        by_arity.entry(leaf.1).or_default().push(leaf.clone());
    }

    transcript.append_u64(b"ft_num_buckets", by_arity.len() as u64);
    if proof.buckets.len() != by_arity.len() {
        return Err(E::LevelMismatch {
            level: 0,
            reason: format!("bucket count: leaves {} != proof {}",
                            by_arity.len(), proof.buckets.len()),
        });
    }
    // Same fork pattern as the prover: bind arities into the parent
    // transcript first, then verify each bucket in parallel.
    let arities_ordered: Vec<usize> = by_arity.keys().copied().collect();
    for &arity in &arities_ordered {
        transcript.append_u64(b"ft_bucket_arity", arity as u64);
    }
    for (&arity, bp) in arities_ordered.iter().zip(proof.buckets.iter()) {
        if bp.arity != arity {
            return Err(E::LevelMismatch {
                level: 0,
                reason: format!("bucket arity mismatch {} vs proof {}", arity, bp.arity),
            });
        }
    }
    use rayon::prelude::*;
    // Multi-GPU verifier dispatch — same assignment scheme as the prover.
    let device_pool = gpu_device_pool();
    let n_devices = device_pool.len().max(1);
    let mut arity_with_idx: Vec<(usize, usize)> = arities_ordered.iter().copied().enumerate()
        .map(|(i, a)| (i, a)).collect();
    arity_with_idx.sort_by_key(|&(_, a)| std::cmp::Reverse(a));
    let mut device_for_bucket: Vec<i32> = vec![0; arities_ordered.len()];
    for (rank, &(orig_idx, _)) in arity_with_idx.iter().enumerate() {
        device_for_bucket[orig_idx] = device_pool[rank % n_devices];
    }
    let verify_inputs: Vec<(usize, &Vec<_>, &BucketFoldProof, Transcript, i32)> = arities_ordered
        .iter()
        .enumerate()
        .zip(proof.buckets.iter())
        .map(|((i, &arity), bp)| (arity, by_arity.get(&arity).unwrap(), bp, transcript.fork(b"ft_bucket", arity), device_for_bucket[i]))
        .collect();
    let results: Vec<Result<(), crate::fold::FoldTreeError>> = verify_inputs
        .into_par_iter()
        .map(|(arity, bucket_leaves, bp, mut bucket_t, device)| {
            let _ = almost_goldilocks_cuda::set_device(device);
            verify_fold_tree_uniform(bucket_leaves, arity, bp, &mut bucket_t)
        })
        .collect();
    for r in results { r?; }

    // Absorb tips (deterministic arity-ascending order).
    for bp in proof.buckets.iter() {
        transcript.append_u64_slice(b"ft_bucket_tip_comm", &bp.final_commitment.rows);
        transcript.append_ext2(b"ft_bucket_tip_val", &bp.final_val);
    }
    Ok(())
}

fn verify_fold_tree_uniform(
    leaves: &[(RingCommitment, usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)],
    max_num_vars: usize,
    proof: &BucketFoldProof,
    transcript: &mut Transcript,
) -> Result<(), crate::fold::FoldTreeError> {
    use crate::fold::FoldTreeError as E;
    transcript.append_u64(b"ft_max_num_vars", max_num_vars as u64);
    transcript.append_u64(b"ft_num_leaves", leaves.len() as u64);

    // Walk the tree the same way the prover did.
    let mut current: Vec<(RingCommitment, usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)> =
        leaves.to_vec();
    let mut level_idx: u64 = 0;

    let per_group = group_size_for_arity(max_num_vars);
    for (li, level) in proof.levels.iter().enumerate() {
        let is_final = li + 1 == proof.levels.len();
        if !is_final && current.len() <= per_group {
            return Err(E::LevelMismatch {
                level: li,
                reason: format!("expected ≤{} for final but got {} at non-final level",
                                per_group, current.len()),
            });
        }
        transcript.append_u64(b"ft_level", level_idx);
        level_idx += 1;

        let groups_in: Vec<&[(RingCommitment, usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)]> =
            current.chunks(per_group).collect();
        if groups_in.len() != level.groups.len() {
            return Err(E::LevelMismatch {
                level: li,
                reason: format!("expected {} groups, got {}", groups_in.len(), level.groups.len()),
            });
        }

        let mut next: Vec<(RingCommitment, usize, Vec<AlmostGoldilocksExt2>, AlmostGoldilocksExt2)> =
            Vec::new();
        for (gi, (gp, group_in)) in level.groups.iter().zip(groups_in.iter()).enumerate() {
            // Fork the verifier's transcript the same way the prover did.
            let mut group_transcript = transcript.fork(b"ft_group", gi);

            let meta: Vec<_> = group_in.iter()
                .map(|(_, a, r, y)| (*a, r.clone(), *y))
                .collect();
            let shared_r = same_point_sumcheck::verify_same_point(&meta, max_num_vars, &gp.same_point, &mut group_transcript)
                .ok_or(E::SamePointFailed { level: li, group: gi })?;

            let comm_refs: Vec<&RingCommitment> = group_in.iter().map(|(c, _, _, _)| c).collect();
            let (combined_c, combined_y) = multifold::verify_multifold(&comm_refs, &gp.multifold, &mut group_transcript)
                .ok_or(E::MultifoldFailed { level: li, group: gi })?;

            if is_final {
                // Last group's output is the final witness — record and break out.
                next.push((combined_c, max_num_vars, shared_r, combined_y));
            } else {
                // Internal level: verifier ALSO checks the split homomorphism.
                let split_proof = gp.split.as_ref().ok_or(E::SplitFailed { level: li, group: gi })?;
                let chunk_commits: Vec<RingCommitment> = split_proof.chunk_commitments.iter()
                    .map(|w| w.to_ring()).collect();
                if !split::verify_split_chunks_match(&combined_c, &chunk_commits) {
                    return Err(E::SplitFailed { level: li, group: gi });
                }
                // Each chunk becomes a next-level "leaf" with its own
                // commitment + prover-supplied claim value. The chain
                // is anchored at the final `commit(f*) = c*` check.
                if gp.chunk_claim_vals.len() != chunk_commits.len() {
                    return Err(E::SplitFailed { level: li, group: gi });
                }
                for (cc, cv) in chunk_commits.iter().zip(gp.chunk_claim_vals.iter()) {
                    next.push((cc.clone(), max_num_vars, shared_r.clone(), *cv));
                }
            }
        }
        // Mirror prover: absorb per-group commitments into the parent
        // transcript so subsequent levels' challenges depend on this
        // level's outputs.
        for (gi, gp) in level.groups.iter().enumerate() {
            absorb_group_commitments(transcript, gi, gp);
        }
        current = next;
    }

    // Final-level: there should be exactly one surviving instance.
    if current.len() != 1 {
        return Err(E::LevelMismatch {
            level: proof.levels.len(),
            reason: format!("expected 1 final instance, got {}", current.len()),
        });
    }
    let (c_final, _arity_final, r_final, y_final) = &current[0];

    // commit(f*) check: re-commit the shipped witness and verify equality.
    let received_c = proof.final_commitment.to_ring();
    if !rings_equal(c_final, &received_c) {
        return Err(E::FinalCommitmentMismatch);
    }
    let f_data = proof.final_witness.to_fold_data();
    if proof.final_pt != *r_final {
        return Err(E::FinalEvaluationMismatch);
    }
    let actual = f_data.evaluate_at_ext2(r_final);
    if !crate::util::arith::ext2_field_eq(actual, *y_final) {
        return Err(E::FinalEvaluationMismatch);
    }
    if !crate::util::arith::ext2_field_eq(actual, proof.final_val) {
        return Err(E::FinalEvaluationMismatch);
    }

    Ok(())
}

/// Absorb a group's combined-commitment + per-chunk commitments into
/// the parent transcript. This is what makes the next level's
/// Fiat-Shamir challenges depend on this level's outputs even though
/// the per-group transcript fork hid them from the parent during the
/// group's own protocol run.
/// Device ids that the fold-tree bucket dispatcher may use. Defaults
/// to all visible CUDA devices (after `CUDA_VISIBLE_DEVICES` filters).
/// Override with `ZK4_GPU_DEVICES=0,2,3` to use a specific subset
/// (e.g., to skip a device that doesn't have enough free memory for
/// the largest arity bucket).
pub fn gpu_device_pool() -> Vec<i32> {
    static POOL: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let pool: Vec<i32> = if let Ok(s) = std::env::var("ZK4_GPU_DEVICES") {
            let parsed: Vec<i32> = s.split(',')
                .filter_map(|t| t.trim().parse::<i32>().ok())
                .collect();
            if !parsed.is_empty() {
                parsed
            } else {
                let n = almost_goldilocks_cuda::device_count().max(1);
                (0..n).collect()
            }
        } else {
            let n = almost_goldilocks_cuda::device_count().max(1);
            (0..n).collect()
        };
        // Warm up each device by allocating a tiny buffer — forces CUDA
        // primary context init eagerly so the first bucket dispatch
        // doesn't pay ~800 ms of context creation on the critical path.
        // Done in parallel so device inits overlap.
        use rayon::prelude::*;
        pool.par_iter().for_each(|&dev| {
            if almost_goldilocks_cuda::set_device(dev).is_ok() {
                let _warm = almost_goldilocks_cuda::memory::DeviceBuffer::<u64>::new(1);
            }
        });
        pool
    }).clone()
}

fn absorb_group_commitments(parent: &mut Transcript, gi: usize, gp: &FoldGroupProof) {
    parent.append_u64(b"ft_absorb_group", gi as u64);
    // Combined commitment from the multifold step — 960 u64s. Batched
    // append (label once, then all values) saves ≥ 100K Monolith
    // permutations across a multi-level bucket vs the per-u64 path.
    parent.append_u64_slice(b"ft_mf_comm", &gp.multifold.combined_commitment.rows);
    if let Some(sp) = &gp.split {
        for cc in &sp.chunk_commitments {
            parent.append_u64_slice(b"ft_chunk_comm", &cc.rows);
        }
    }
    for v in &gp.chunk_claim_vals {
        parent.append_ext2(b"ft_chunk_claim", v);
    }
}

fn rings_equal(a: &RingCommitment, b: &RingCommitment) -> bool {
    use almost_goldilocks_cuda::ajtai::{KAPPA, RING_DIM};
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            if a.rows[i][k] != b.rows[i][k] { return false; }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::ajtai::{commit, ChunkSize};
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn demo_seed() -> Seed {
        Seed([
            0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
            0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
        ])
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    fn random_binary_leaf(rng: &mut StdRng, arity: usize, seed: Seed) -> FoldInstance {
        let n_ring = 1usize << (arity - 6);
        let packed: Vec<u64> = (0..n_ring).map(|_| rng.gen::<u64>()).collect();
        let commitment = commit(seed, &packed, Some(ChunkSize::C64)).expect("commit");
        let data = FoldData::Binary(packed);
        let claim_pt: Vec<_> = (0..arity).map(|i| lift(i as u64 * 7 + 3 + rng.gen::<u8>() as u64)).collect();
        let claim_val = data.evaluate_at_ext2(&claim_pt);
        FoldInstance { commitment, data, arity, claim_pt, claim_val }
    }

    /// Final-only tree (≤63 leaves): one same-point + multifold, ship
    /// the witness, verifier accepts.
    #[test]
    fn fold_tree_final_only_roundtrip() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut rng = StdRng::seed_from_u64(0xFACE);
        let arity = 6;
        let leaves: Vec<_> = (0..3)
            .map(|_| random_binary_leaf(&mut rng, arity, demo_seed()))
            .collect();
        let leaves_meta: Vec<_> = leaves.iter()
            .map(|inst| (inst.commitment.clone(), inst.arity, inst.claim_pt.clone(), inst.claim_val))
            .collect();

        let mut t_p = Transcript::new(b"ft-final");
        let proof = prove_fold_tree(leaves.clone(), demo_seed(), &mut t_p);
        assert_eq!(proof.buckets.len(), 1, "single bucket (one arity)");
        let bp = &proof.buckets[0];
        assert_eq!(bp.levels.len(), 1, "single final level");
        assert!(bp.levels[0].groups[0].split.is_none(), "no split at final");

        let mut t_v = Transcript::new(b"ft-final");
        verify_fold_tree(&leaves_meta, &proof, &mut t_v)
            .expect("honest fold tree should verify");
    }

    /// Internal-level tree (>63 leaves): triggers the split path and
    /// the chunk-as-next-level-instance translation. Confirms the chain
    /// of homomorphism checks survives one full split.
    #[test]
    fn fold_tree_with_internal_split_level() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut rng = StdRng::seed_from_u64(0xAA);
        let arity = 6; // small leaves to keep test fast
        let num_leaves = MAX_PER_GROUP + 1; // 64 → one internal level
        let leaves: Vec<_> = (0..num_leaves)
            .map(|_| random_binary_leaf(&mut rng, arity, demo_seed()))
            .collect();
        let leaves_meta: Vec<_> = leaves.iter()
            .map(|inst| (inst.commitment.clone(), inst.arity, inst.claim_pt.clone(), inst.claim_val))
            .collect();

        let mut t_p = Transcript::new(b"ft-internal");
        let proof = prove_fold_tree(leaves.clone(), demo_seed(), &mut t_p);
        assert_eq!(proof.buckets.len(), 1, "all leaves at one arity → one bucket");
        let bp = &proof.buckets[0];
        assert!(bp.levels.len() >= 2, "expected ≥ 2 levels in bucket, got {}", bp.levels.len());
        assert_eq!(bp.levels[0].groups.len(), 2, "first level: 63 + 1 → 2 groups");
        for g in &bp.levels[0].groups {
            assert!(g.split.is_some(), "internal-level groups must split");
        }
        assert!(bp.levels.last().unwrap().groups[0].split.is_none());

        let mut t_v = Transcript::new(b"ft-internal");
        verify_fold_tree(&leaves_meta, &proof, &mut t_v)
            .expect("internal-level fold tree should verify");
    }

    /// Force the device-resident group path at a small test arity (env
    /// override) across MULTIPLE levels — exercises device assembly from
    /// host leaves (level 0), chunk mirrors consumed DtoD (levels 1+),
    /// the on-device split decompose, and device chunk evals. The proof
    /// must verify against the unchanged verifier.
    #[test]
    fn fold_tree_device_resident_roundtrip() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        std::env::set_var("ZK4_DEVRES_MIN_ARITY", "1");
        let mut rng = StdRng::seed_from_u64(0xDE51DE);
        let arity = 8;
        // 64 + 200 leaves → level 0 has multiple groups, level 1 consumes
        // mirrored chunks, level 2 is final.
        let num_leaves = 264;
        let leaves: Vec<_> = (0..num_leaves)
            .map(|_| random_binary_leaf(&mut rng, arity, demo_seed()))
            .collect();
        let leaves_meta: Vec<_> = leaves.iter()
            .map(|inst| (inst.commitment.clone(), inst.arity, inst.claim_pt.clone(), inst.claim_val))
            .collect();

        let mut t_p = Transcript::new(b"ft-devres");
        let proof = prove_fold_tree(leaves.clone(), demo_seed(), &mut t_p);
        std::env::remove_var("ZK4_DEVRES_MIN_ARITY");
        assert!(proof.buckets[0].levels.len() >= 3,
            "expected ≥ 3 levels, got {}", proof.buckets[0].levels.len());

        let mut t_v = Transcript::new(b"ft-devres");
        verify_fold_tree(&leaves_meta, &proof, &mut t_v)
            .expect("device-resident fold tree must verify");
    }

    /// Tamper with the final_val → verifier rejects with FinalEvaluationMismatch.
    #[test]
    fn fold_tree_rejects_tampered_final_val() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut rng = StdRng::seed_from_u64(13);
        let arity = 6;
        let leaves: Vec<_> = (0..2).map(|_| random_binary_leaf(&mut rng, arity, demo_seed())).collect();
        let leaves_meta: Vec<_> = leaves.iter()
            .map(|inst| (inst.commitment.clone(), inst.arity, inst.claim_pt.clone(), inst.claim_val))
            .collect();
        let mut t_p = Transcript::new(b"ft-tamper");
        let mut proof = prove_fold_tree(leaves, demo_seed(), &mut t_p);
        proof.buckets[0].final_val = crate::util::arith::ext2_add(
            proof.buckets[0].final_val, AlmostGoldilocksExt2::one(),
        );
        let mut t_v = Transcript::new(b"ft-tamper");
        let res = verify_fold_tree(&leaves_meta, &proof, &mut t_v);
        assert!(res.is_err(), "tampered final_val must be rejected");
    }
}
