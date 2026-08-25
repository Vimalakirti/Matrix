//! PCS scaling bench: sparsity ablation, polynomial size, and device count.
//!
//! Isolates the polynomial commitment from any model. A model fixes its own
//! leaf set, so it cannot answer "how does the PCS scale in polynomial size?"
//! This synthesizes a leaf set of controlled arity, count and DENSITY, then
//! runs the same `prove_fold_tree` the real prover runs.
//!
//! Three axes, selected by env:
//!
//!   PCS_ARITY    log2 of the witness size per leaf          (default 22)
//!   PCS_LEAVES   number of leaves in the set                (default 64)
//!   PCS_DENS_LOG nonzeros per leaf = 2^(arity - DENS_LOG)   (default 6)
//!                0 means fully dense (~50% bits set).
//!   REPS         repetitions; the median is reported        (default 1)
//!
//! Sparsity is the axis the opening path actually keys on: with
//! ZK4_SPARSE_SP=1 a sufficiently sparse binary leaf is opened WITHOUT
//! materializing the dense 2^arity Ext2 equality table (~1 GB per leaf at
//! arity 26). ZK4_SPARSE_SP=0 forces that table. Round messages are
//! byte-identical between the two, so the proof MUST come out the same size;
//! the bench asserts that rather than assuming it, since a difference would
//! mean the two paths are not proving the same statement.
//!
//! Reports one CSV-ish line per configuration so the driver can parse it.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::{self, RingCommitment, Seed};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2 as Ext2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField as F;

use zk_torch_4::fold::{FoldData, FoldInstance, tree::prove_fold_tree};
use zk_torch_4::transcript::Transcript;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Peak GPU bytes in use across the device pool, sampled from inside the
/// process. `nvidia-smi --query-compute-apps=used_memory` reports 0 for this
/// binary on this machine while reporting real values for other processes, so
/// external sampling cannot be trusted here. cudaMemGetInfo is per-device
/// total-minus-free, so it includes anything else resident; the baseline taken
/// before allocation is subtracted to isolate this run's contribution.
fn spawn_mem_sampler(devices: Vec<i32>) -> (Arc<AtomicU64>, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
    let peak = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (p, s) = (peak.clone(), stop.clone());
    let h = std::thread::spawn(move || {
        let base: Vec<u64> = devices.iter().map(|&d| {
            let _ = almost_goldilocks_cuda::set_device(d);
            almost_goldilocks_cuda::memory::mem_get_info().map(|(f, t)| (t - f) as u64).unwrap_or(0)
        }).collect();
        while !s.load(Ordering::Relaxed) {
            let mut used = 0u64;
            for (i, &d) in devices.iter().enumerate() {
                let _ = almost_goldilocks_cuda::set_device(d);
                if let Ok((f, t)) = almost_goldilocks_cuda::memory::mem_get_info() {
                    used += ((t - f) as u64).saturating_sub(base[i]);
                }
            }
            p.fetch_max(used, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });
    (peak, stop, h)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
}

/// Packed binary witness with `nnz` bits set out of `2^arity`, or fully
/// random when `nnz == 0`. Bits are scattered by an LCG so the support is not
/// clustered into a few words, which would flatter the sparse path.
fn packed(arity: usize, nnz: usize, rng: &mut Rng) -> Vec<u64> {
    let words = 1usize << (arity - 6);
    if nnz == 0 {
        return (0..words).map(|_| rng.next()).collect();
    }
    let mut v = vec![0u64; words];
    let total = 1usize << arity;
    let mut pos = rng.next() as usize % total;
    for _ in 0..nnz {
        v[pos >> 6] |= 1u64 << (pos & 63);
        pos = (pos.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)) % total;
    }
    v
}

fn main() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let arity = env("PCS_ARITY", 22);
    let n_leaves = env("PCS_LEAVES", 64);
    let dens_log = env("PCS_DENS_LOG", 6);
    let reps = env("REPS", 1);
    let nnz = if dens_log == 0 { 0 } else { 1usize << (arity - dens_log.min(arity)) };

    let devices = zk_torch_4::fold::tree::gpu_device_pool().len();
    let sparse = std::env::var("ZK4_SPARSE_SP").ok().as_deref() != Some("0");
    println!("=== PCS scaling: arity={} leaves={} dens_log={} nnz/leaf={} sparse_sp={} commit={} gpus={} ===",
             arity, n_leaves, dens_log, if nnz == 0 { 1usize << (arity - 1) } else { nnz },
             sparse, std::env::var("PCS_COMMIT").unwrap_or_else(|_| "dense".into()), devices);

    let seed = Seed([2, 7, 1, 8, 2, 8, 1, 8]);
    let mut rng = Rng(0xF00D_BEEF_1234_5678);

    // ---- build the leaf set (not timed: it is bench setup, not the PCS) ----
    let witnesses: Vec<Vec<u64>> = (0..n_leaves).map(|_| packed(arity, nnz, &mut rng)).collect();
    let points: Vec<Vec<Ext2>> = (0..n_leaves)
        .map(|_| (0..arity).map(|_| Ext2::new(F(rng.next() >> 4), F(rng.next() >> 4))).collect())
        .collect();

    let (peak, stop, sampler) = spawn_mem_sampler(zk_torch_4::fold::tree::gpu_device_pool());

    let mut commit_ms = 0.0f64;
    let mut open_ms = 0.0f64;
    let mut proof_bytes = 0usize;
    let mut oks = true;

    for rep in 0..reps.max(1) {
        // ---- COMMIT. Two paths, and which one the real prover takes is
        // decided by the witness's PolyType (commit/mod.rs:348), not by an env
        // flag: a Sparse witness is committed from its POSITION LIST via
        // ajtai::commit_sparse, a dense one from packed bit-planes via
        // commit_batched. They are not interchangeable in cost. commit_sparse's
        // own doc says dense is ~16x cheaper at typical random densities, so
        // the sparse commit only pays when the support is genuinely tiny.
        // PCS_COMMIT=sparse|dense selects; default dense, matching the batched
        // path the fold-tree leaf builder uses for binary leaves.
        let commit_sparse = std::env::var("PCS_COMMIT").ok().as_deref() == Some("sparse");
        let t = Instant::now();
        let commitments: Vec<RingCommitment> = if commit_sparse {
            witnesses.iter().map(|w| {
                // Position list: bit index (j << 6 | l) for every set bit.
                let mut pos: Vec<u64> = Vec::new();
                for (j, &word) in w.iter().enumerate() {
                    let mut x = word;
                    while x != 0 {
                        let l = x.trailing_zeros() as u64;
                        pos.push(((j as u64) << 6) | l);
                        x &= x - 1;
                    }
                }
                ajtai::commit_sparse(seed, &pos, None).expect("commit_sparse")
            }).collect()
        } else {
            let refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
            refs.chunks(16)
                .flat_map(|c| ajtai::commit_batched(seed, c, None).expect("commit_batched"))
                .collect()
        };
        let c_ms = t.elapsed().as_secs_f64() * 1e3;

        // ---- OPEN: prove_fold_tree over the committed leaf set.
        let leaves: Vec<FoldInstance> = witnesses.iter().zip(points.iter()).zip(commitments.iter())
            .map(|((w, pt), c)| {
                let data = FoldData::Binary(w.clone());
                let val = data.evaluate_at_ext2(pt);
                FoldInstance { commitment: c.clone(), data, arity, claim_pt: pt.clone(), claim_val: val }
            })
            .collect();
        let mut tr = Transcript::new(b"pcs_scaling");
        let t = Instant::now();
        let proof = prove_fold_tree(leaves, seed, &mut tr);
        let o_ms = t.elapsed().as_secs_f64() * 1e3;
        let bytes = zk_torch_4::ser_len(&proof);

        if rep == 0 || o_ms < open_ms {
            commit_ms = c_ms; open_ms = o_ms; proof_bytes = bytes;
        }
        oks &= bytes > 0;
    }

    stop.store(true, Ordering::Relaxed);
    let _ = sampler.join();
    let peak_mib = peak.load(Ordering::Relaxed) / (1024 * 1024);
    // HOST peak, which is the resource the sparse path is actually about:
    // same_point_sumcheck calls the dense equality table "the host-memory wall
    // at high arity (~1 GB/leaf at 26, ~16 GB at 30)". GPU peak at these sizes
    // is set by the fold tree's ~10 GB pooled buffers, which both paths
    // allocate, so it shows no sparsity effect and is reported only for
    // capacity planning. VmHWM is the kernel's own high-water mark, so it needs
    // no sampling and cannot miss a transient.
    let host_peak_mib = std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(|l| l.to_string()))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
        .map(|kb| kb / 1024).unwrap_or(0);

    println!("[pcs] arity={} leaves={} dens_log={} sparse={} commit={} gpus={} commit_ms={:.1} open_ms={:.1} proof_bytes={} ok={}",
             arity, n_leaves, dens_log, if sparse { 1 } else { 0 },
             std::env::var("PCS_COMMIT").unwrap_or_else(|_| "dense".into()), devices,
             commit_ms, open_ms, proof_bytes, oks);
    println!("[pcs_mem] peak_mib={} host_peak_mib={}", peak_mib, host_peak_mib);
    println!("  commit {:.1} ms   open {:.1} ms   proof {:.2} MB   ({} leaves x 2^{})",
             commit_ms, open_ms, proof_bytes as f64 / (1 << 20) as f64, n_leaves, arity);
}
