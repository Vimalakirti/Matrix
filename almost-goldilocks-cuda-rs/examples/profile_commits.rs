//! Minimal driver for ncu profiling of commit kernels.
//!
//! Run with:
//!   cargo build --release --example profile_commits
//!   ncu --target-processes all --set full \
//!       --kernel-name regex:'commit_(dense_batched|ternary)_kernel' \
//!       --launch-skip 2 --launch-count 1 \
//!       ./target/release/examples/profile_commits
//!
//! Inputs are pre-uploaded so each launch reflects pure kernel work.
//! Binary commit uses B=16 (canonical batched path).
//! Ternary commit uses the 13-chunk SuperNeo configuration.

use almost_goldilocks_cuda::ajtai::{
    commit_batched, commit_ternary, multifold_witness, split_witness_device,
    ChunkSize, RingChallenge, Seed, SPLIT_K_CHUNKS, TernaryChunksDevice,
};
use almost_goldilocks_cuda::memory::{synchronize, DeviceBuffer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn rand_z(rng: &mut StdRng, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.gen::<u64>()).collect()
}

fn rand_r(rng: &mut StdRng) -> RingChallenge {
    let mut c = [0i8; 64];
    for v in c.iter_mut() {
        *v = match rng.gen_range(0..4) { 0 => -1, 1 => 0, 2 => 1, _ => 2 };
    }
    RingChallenge::new(c).unwrap()
}

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(0xA17_C0DE);

    // Profile at log_n = 20  (N_ring = 2^14 = 16384). Big enough for SMs
    // to saturate, small enough for ncu to capture quickly.
    let log_n: u32 = 20;
    let n_ring = 1usize << (log_n - 6);

    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);

    // ─── binary B=16 commit setup ───
    let bins: Vec<Vec<u64>> = (0..16).map(|_| rand_z(&mut rng, n_ring)).collect();
    let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

    // ─── ternary commit setup (build chunks from real multifold + split) ───
    let m = 50 + SPLIT_K_CHUNKS;
    let inner: Vec<Vec<u64>> = (0..m).map(|_| rand_z(&mut rng, n_ring)).collect();
    let inner_refs: Vec<&[u64]> = inner.iter().map(|v| v.as_slice()).collect();
    let chal: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_r(&mut rng)).collect();
    let z_wide = multifold_witness(&inner_refs, &chal).unwrap();
    let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).unwrap();
    let chunks: TernaryChunksDevice = split_witness_device(&d_z_wide).unwrap();

    // Warmups + launches. ncu --launch-skip 2 --launch-count 1 captures the
    // third launch of each kernel (steady-state).
    for _ in 0..4 {
        let _ = commit_batched(seed, &bin_refs, Some(ChunkSize::C256)).unwrap();
        synchronize().ok();
    }

    for _ in 0..4 {
        let _ = commit_ternary(seed, &chunks, None).unwrap();
        synchronize().ok();
    }

    println!("profile driver done (log_n={}, N_ring={})", log_n, n_ring);
}
