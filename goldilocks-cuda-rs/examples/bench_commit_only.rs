//! Basefold commit-only benchmark.
//!
//! Measures BasefoldCommitment::commit() time across a sweep of num_vars.
//! Prints a CSV-style table to stdout that the comparison harness collects.
//!
//! Usage: cargo run --release --example bench_commit_only

use goldilocks_cuda::basefold::BasefoldCommitment;
use goldilocks_cuda::field::{GoldilocksField, GOLDILOCKS_PRIME};

fn main() {
    goldilocks_cuda::init().expect("CUDA init");

    let log_rate = 1usize; // FRI rate 2
    let sizes: &[usize] = &[14, 16, 18, 20, 22, 24, 26];

    println!("# Basefold commit benchmark (Goldilocks, log_rate={})", log_rate);
    println!("log_n,num_coeffs,iters,mean_ms,min_ms");

    for &log_n in sizes {
        let n = 1usize << log_n;

        // Build evals from a deterministic mix so all sizes get the same RNG
        // pattern; the actual values don't affect commit time (it's compute-bound).
        let evals: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField(((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)) % GOLDILOCKS_PRIME))
            .collect();

        // Warmup once.
        let _ = BasefoldCommitment::commit(&evals, log_n, log_rate).expect("warmup");

        // Pick iter count to keep total wall time per row reasonable.
        let iters: usize = if log_n <= 18 { 5 } else if log_n <= 22 { 3 } else { 2 };

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            goldilocks_cuda::synchronize().ok();
            let t0 = std::time::Instant::now();
            let _comm = BasefoldCommitment::commit(&evals, log_n, log_rate).expect("commit");
            goldilocks_cuda::synchronize().ok();
            samples.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let min  = samples.iter().fold(f64::INFINITY, |a, b| a.min(*b));

        println!("{},{},{},{:.2},{:.2}", log_n, n, iters, mean, min);
    }
}
