//! Standalone sumcheck scaling benchmark.
//!
//! Tests both CPU and GPU sumcheck provers across a range of polynomial sizes
//! (num_vars from 8 to 24) with 2 and 3 polynomials, measuring time per run.
//!
//! Usage:
//!     cargo run --release --bin bench_sumcheck

use std::time::Instant;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use rand::Rng;

use zk_torch_3::poly::DenseMLPoly;
use zk_torch_3::sumcheck::{CpuLinearSumcheckProverExt2, GpuLinearSumcheckProver};
use zk_torch_3::transcript::Transcript;

fn random_ext2_vec(size: usize) -> Vec<GoldilocksExt2> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksExt2::new(
            GoldilocksField(rng.gen::<u64>() % (1u64 << 63)),
            GoldilocksField(rng.gen::<u64>() % (1u64 << 63)),
        ))
        .collect()
}

fn random_base_poly(num_vars: usize) -> DenseMLPoly {
    let size = 1usize << num_vars;
    let mut rng = rand::thread_rng();
    let evals: Vec<GoldilocksField> = (0..size)
        .map(|_| GoldilocksField(rng.gen::<u64>() % (1u64 << 32)))
        .collect();
    DenseMLPoly::new(num_vars, evals)
}

fn bench_cpu_ext2(num_vars: usize, num_polys: usize, trials: usize) -> f64 {
    let size = 1usize << num_vars;
    let mut times = Vec::with_capacity(trials);

    for _ in 0..trials {
        let mut polys: Vec<Vec<GoldilocksExt2>> = (0..num_polys)
            .map(|_| random_ext2_vec(size))
            .collect();

        let mut transcript = Transcript::new(b"bench_cpu");
        let mut prover = CpuLinearSumcheckProverExt2::new(num_vars, num_polys, &mut transcript);

        let t = Instant::now();
        let _proof = prover.prove(&mut polys, &mut transcript);
        times.push(t.elapsed().as_secs_f64());
    }

    // Return median
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn bench_gpu_ext2(num_vars: usize, num_polys: usize, trials: usize) -> f64 {
    let size = 1usize << num_vars;
    let mut times = Vec::with_capacity(trials);

    for _ in 0..trials {
        let polys: Vec<Vec<GoldilocksExt2>> = (0..num_polys)
            .map(|_| random_ext2_vec(size))
            .collect();

        let mut transcript = Transcript::new(b"bench_gpu");
        let mut prover = GpuLinearSumcheckProver::new(num_vars, num_polys, &mut transcript);

        let t = Instant::now();
        let _proof = prover.prove_ext2(&polys, &mut transcript);
        times.push(t.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn bench_gpu_base(num_vars: usize, num_polys: usize, trials: usize) -> f64 {
    let mut times = Vec::with_capacity(trials);

    for _ in 0..trials {
        let polys: Vec<DenseMLPoly> = (0..num_polys)
            .map(|_| random_base_poly(num_vars))
            .collect();

        let mut transcript = Transcript::new(b"bench_gpu_base");
        let mut prover = GpuLinearSumcheckProver::new(num_vars, num_polys, &mut transcript);

        let t = Instant::now();
        let _proof = prover.prove(&polys, &mut transcript);
        times.push(t.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    goldilocks_cuda::init().expect("CUDA init failed");

    let max_nvar: usize = std::env::var("MAX_NVAR").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(24);
    let min_nvar: usize = std::env::var("MIN_NVAR").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(8);
    let trials: usize = std::env::var("TRIALS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("=== Sumcheck Scaling Benchmark ===");
    println!("num_vars range: {} to {}, trials: {} (median)", min_nvar, max_nvar, trials);
    println!();

    // Warmup GPU
    {
        let polys: Vec<Vec<GoldilocksExt2>> = (0..2)
            .map(|_| random_ext2_vec(1 << 12))
            .collect();
        let mut t = Transcript::new(b"warmup");
        let mut p = GpuLinearSumcheckProver::new(12, 2, &mut t);
        let _ = p.prove_ext2(&polys, &mut t);
    }

    // ===== 2 polynomials =====
    println!("--- 2 polynomials (typical: value * eq) ---");
    println!("{:<10} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "num_vars", "CPU_ext2(ms)", "GPU_ext2(ms)", "GPU_base(ms)", "CPU/prev", "GPU/prev");

    let mut prev_cpu = 0.0f64;
    let mut prev_gpu = 0.0f64;

    for nvar in min_nvar..=max_nvar {
        let cpu_time = bench_cpu_ext2(nvar, 2, trials);
        let gpu_ext2_time = bench_gpu_ext2(nvar, 2, trials);
        let gpu_base_time = bench_gpu_base(nvar, 2, trials);

        let cpu_ratio = if prev_cpu > 0.0 { cpu_time / prev_cpu } else { 0.0 };
        let gpu_ratio = if prev_gpu > 0.0 { gpu_ext2_time / prev_gpu } else { 0.0 };

        println!("{:<10} {:>12.3} {:>12.3} {:>12.3} {:>10.2}x {:>10.2}x",
            nvar,
            cpu_time * 1000.0,
            gpu_ext2_time * 1000.0,
            gpu_base_time * 1000.0,
            cpu_ratio,
            gpu_ratio,
        );

        prev_cpu = cpu_time;
        prev_gpu = gpu_ext2_time;
    }

    // ===== 3 polynomials =====
    println!();
    println!("--- 3 polynomials (typical: input1 * input2 * eq) ---");
    println!("{:<10} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "num_vars", "CPU_ext2(ms)", "GPU_ext2(ms)", "GPU_base(ms)", "CPU/prev", "GPU/prev");

    prev_cpu = 0.0;
    prev_gpu = 0.0;

    for nvar in min_nvar..=max_nvar {
        let cpu_time = bench_cpu_ext2(nvar, 3, trials);
        let gpu_ext2_time = bench_gpu_ext2(nvar, 3, trials);
        let gpu_base_time = bench_gpu_base(nvar, 3, trials);

        let cpu_ratio = if prev_cpu > 0.0 { cpu_time / prev_cpu } else { 0.0 };
        let gpu_ratio = if prev_gpu > 0.0 { gpu_ext2_time / prev_gpu } else { 0.0 };

        println!("{:<10} {:>12.3} {:>12.3} {:>12.3} {:>10.2}x {:>10.2}x",
            nvar,
            cpu_time * 1000.0,
            gpu_ext2_time * 1000.0,
            gpu_base_time * 1000.0,
            cpu_ratio,
            gpu_ratio,
        );

        prev_cpu = cpu_time;
        prev_gpu = gpu_ext2_time;
    }

    // ===== Throughput analysis =====
    println!();
    println!("--- Throughput (elements/sec) for 2 polys ---");
    println!("{:<10} {:>15} {:>15} {:>15}",
        "num_vars", "CPU_ext2", "GPU_ext2", "GPU_base");

    for nvar in min_nvar..=max_nvar {
        let size = (1usize << nvar) as f64;
        let cpu_time = bench_cpu_ext2(nvar, 2, trials);
        let gpu_ext2_time = bench_gpu_ext2(nvar, 2, trials);
        let gpu_base_time = bench_gpu_base(nvar, 2, trials);

        println!("{:<10} {:>15.0} {:>15.0} {:>15.0}",
            nvar,
            size / cpu_time,
            size / gpu_ext2_time,
            size / gpu_base_time,
        );
    }
}
