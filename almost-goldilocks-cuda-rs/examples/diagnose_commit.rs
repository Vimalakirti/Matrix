//! Diagnostic bench — isolate the cost of each phase inside commit_ternary
//! and commit_dense_batched by varying inputs:
//!
//!   * All-zero witness  → 0 set bits → bit-iteration loop runs 0 times.
//!                          Measures PRG + structural overhead only.
//!   * Random witness    → ~32 set bits per ring element → full work.
//!   * (Optional) all-ones witness  → 64 set bits/ring → 2× random.
//!
//! Difference between zero and random isolates the bit-iteration cost.
//! Combined with the premat result (PRG is not the dominant cost), this
//! pins down where the kernel time actually goes.

use almost_goldilocks_cuda::ajtai::{
    commit_batched, commit_ternary, ChunkSize, RingChallenge, Seed,
    SPLIT_K_CHUNKS, TernaryChunksDevice,
};
use almost_goldilocks_cuda::memory::{synchronize, DeviceBuffer};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn time_ms<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    f(); synchronize().ok();
    f(); synchronize().ok();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        synchronize().ok();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best { best = ms; }
    }
    best
}

fn build_chunks(n_ring: usize, density: Density, rng: &mut StdRng) -> TernaryChunksDevice {
    let mut pos = vec![0u64; SPLIT_K_CHUNKS * n_ring];
    let mut neg = vec![0u64; SPLIT_K_CHUNKS * n_ring];
    for i in 0..SPLIT_K_CHUNKS {
        for j in 0..n_ring {
            match density {
                Density::Zero => { /* leave pos=neg=0 */ }
                Density::Random => {
                    let mut p = 0u64; let mut n = 0u64;
                    for c in 0..64 {
                        match rng.gen_range(0..4) {
                            0 => p |= 1u64 << c,
                            1 => n |= 1u64 << c,
                            _ => {}
                        }
                    }
                    pos[i * n_ring + j] = p;
                    neg[i * n_ring + j] = n;
                }
                Density::AllOnesPos => { pos[i * n_ring + j] = u64::MAX; }
            }
        }
    }
    TernaryChunksDevice {
        n_ring,
        k_chunks: SPLIT_K_CHUNKS,
        pos: DeviceBuffer::from_slice(&pos).unwrap(),
        neg: DeviceBuffer::from_slice(&neg).unwrap(),
    }
}

#[derive(Copy, Clone)]
enum Density { Zero, Random, AllOnesPos }

fn main() {
    almost_goldilocks_cuda::init().unwrap();
    let mut rng = StdRng::seed_from_u64(0x_D1A6_C0DE);
    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);

    println!("# Diagnostic — isolate phase costs inside commit_ternary / commit_dense_batched");
    println!("# 'zero' = PRG + structural; 'random' = +half-density bit iter;");
    println!("# 'all-pos' = +max-density bit iter.  All times ms, min over 5 iters.");
    println!();

    // ─── commit_ternary ───
    println!("# commit_ternary (B=13 chunks)");
    println!("log_n,N_ring,zero_ms,random_ms,all_pos_ms,bit_iter_cost_ms");
    for &log_n in &[14usize, 16, 18, 20, 22] {
        let n_ring = 1usize << (log_n - 6);
        let zero  = build_chunks(n_ring, Density::Zero,       &mut rng);
        let rand  = build_chunks(n_ring, Density::Random,     &mut rng);
        let ones  = build_chunks(n_ring, Density::AllOnesPos, &mut rng);

        let z_ms = time_ms(5, || { let _ = commit_ternary(seed, &zero, None).unwrap(); });
        let r_ms = time_ms(5, || { let _ = commit_ternary(seed, &rand, None).unwrap(); });
        let o_ms = time_ms(5, || { let _ = commit_ternary(seed, &ones, None).unwrap(); });

        println!("{},{},{:.2},{:.2},{:.2},{:.2}",
                 log_n, n_ring, z_ms, r_ms, o_ms, r_ms - z_ms);
    }
    println!();

    // ─── commit_dense_batched (B=16 binary) ───
    println!("# commit_dense_batched (B=16)");
    println!("log_n,N_ring,zero_ms,random_ms,all_ones_ms,bit_iter_cost_ms");
    for &log_n in &[14usize, 16, 18, 20, 22] {
        let n_ring = 1usize << (log_n - 6);

        let zero_bins: Vec<Vec<u64>> = (0..16).map(|_| vec![0u64; n_ring]).collect();
        let zero_refs: Vec<&[u64]> = zero_bins.iter().map(|v| v.as_slice()).collect();

        let rand_bins: Vec<Vec<u64>> = (0..16)
            .map(|_| (0..n_ring).map(|_| rng.gen::<u64>()).collect()).collect();
        let rand_refs: Vec<&[u64]> = rand_bins.iter().map(|v| v.as_slice()).collect();

        let ones_bins: Vec<Vec<u64>> = (0..16).map(|_| vec![u64::MAX; n_ring]).collect();
        let ones_refs: Vec<&[u64]> = ones_bins.iter().map(|v| v.as_slice()).collect();

        let z_ms = time_ms(5, || { let _ = commit_batched(seed, &zero_refs, Some(ChunkSize::C256)).unwrap(); });
        let r_ms = time_ms(5, || { let _ = commit_batched(seed, &rand_refs, Some(ChunkSize::C256)).unwrap(); });
        let o_ms = time_ms(5, || { let _ = commit_batched(seed, &ones_refs, Some(ChunkSize::C256)).unwrap(); });

        println!("{},{},{:.2},{:.2},{:.2},{:.2}",
                 log_n, n_ring, z_ms, r_ms, o_ms, r_ms - z_ms);
    }
}
