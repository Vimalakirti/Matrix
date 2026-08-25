//! Ajtai commit-only benchmark on almost-Goldilocks.
//!
//! Sizes are aligned with the Basefold benchmark by total polynomial-
//! coefficient count: Basefold(num_vars=N) commits 2^N Goldilocks field
//! elements; Ajtai(N_ring) commits 64 · N_ring binary coefficients. We
//! pair Basefold num_vars=N with Ajtai N_ring = 2^(N - 6), so both schemes
//! commit to the same 2^N-coefficient logical polynomial (Basefold over
//! F_q, Ajtai over {0,1}).
//!
//! Reports per-commit time for B ∈ {1, 8, 16} (B>1 amortizes PRG over the
//! batch).
//!
//! Usage: cargo run --release --example bench_commit_only

use almost_goldilocks_cuda::ajtai::{commit, commit_batched, ChunkSize, Seed};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn make_witness(rng: &mut StdRng, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.gen::<u64>()).collect()
}

fn main() {
    almost_goldilocks_cuda::init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xA17A1_BEEF);
    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);

    // Same log_n sweep as the Basefold bench (number of polynomial coefficients = 2^log_n).
    let sizes: &[usize] = &[14, 16, 18, 20, 22, 24, 26];

    println!("# Ajtai commit benchmark (almost-Goldilocks, auto-selected ChunkSize)");
    println!("log_n,num_coeffs,N_ring,batch,iters,mean_ms,min_ms,per_commit_ms");

    for &log_n in sizes {
        // N_ring = 2^(log_n - 6), since each ring element packs 64 binary coefs.
        // Bail if log_n < 6 (n_ring would be < 1).
        if log_n < 6 { continue; }
        let n_ring: usize = 1usize << (log_n - 6);
        let n_coeffs: usize = 1usize << log_n;

        let z_single = make_witness(&mut rng, n_ring);

        // Pick iter count: larger sizes get fewer iterations.
        let iters: usize = if log_n <= 18 { 5 } else if log_n <= 22 { 3 } else { 2 };

        // ───────── B = 1 (single commit) ─────────
        let _ = commit(seed, &z_single, None).expect("warmup");
        let mut t_single = Vec::with_capacity(iters);
        for _ in 0..iters {
            almost_goldilocks_cuda::synchronize().ok();
            let t0 = std::time::Instant::now();
            let _c = commit(seed, &z_single, None).expect("commit");
            almost_goldilocks_cuda::synchronize().ok();
            t_single.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_s = t_single.iter().sum::<f64>() / t_single.len() as f64;
        let min_s = t_single.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        println!("{},{},{},{},{},{:.2},{:.2},{:.2}",
                 log_n, n_coeffs, n_ring, 1, iters, mean_s, min_s, mean_s);

        // ───────── B = 8 batched ─────────
        let witnesses_8: Vec<Vec<u64>> = (0..8).map(|_| make_witness(&mut rng, n_ring)).collect();
        let refs_8: Vec<&[u64]> = witnesses_8.iter().map(|w| w.as_slice()).collect();
        let _ = commit_batched(seed, &refs_8, None).expect("warmup");
        let mut t8 = Vec::with_capacity(iters);
        for _ in 0..iters {
            almost_goldilocks_cuda::synchronize().ok();
            let t0 = std::time::Instant::now();
            let _c = commit_batched(seed, &refs_8, None).expect("commit_batched");
            almost_goldilocks_cuda::synchronize().ok();
            t8.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_8 = t8.iter().sum::<f64>() / t8.len() as f64;
        let min_8 = t8.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        println!("{},{},{},{},{},{:.2},{:.2},{:.2}",
                 log_n, n_coeffs, n_ring, 8, iters, mean_8, min_8, mean_8 / 8.0);

        // ───────── B = 16 batched ─────────
        let witnesses_16: Vec<Vec<u64>> = (0..16).map(|_| make_witness(&mut rng, n_ring)).collect();
        let refs_16: Vec<&[u64]> = witnesses_16.iter().map(|w| w.as_slice()).collect();
        let _ = commit_batched(seed, &refs_16, None).expect("warmup");
        let mut t16 = Vec::with_capacity(iters);
        for _ in 0..iters {
            almost_goldilocks_cuda::synchronize().ok();
            let t0 = std::time::Instant::now();
            let _c = commit_batched(seed, &refs_16, None).expect("commit_batched");
            almost_goldilocks_cuda::synchronize().ok();
            t16.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean_16 = t16.iter().sum::<f64>() / t16.len() as f64;
        let min_16 = t16.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        println!("{},{},{},{},{},{:.2},{:.2},{:.2}",
                 log_n, n_coeffs, n_ring, 16, iters, mean_16, min_16, mean_16 / 16.0);
    }
}
