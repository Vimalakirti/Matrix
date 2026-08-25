/// Microbenchmark binary for GPU threshold auto-tuning.
///
/// Measures CPU vs GPU crossover points for 4 threshold-controlled operations:
/// 1. Sumcheck (ZK_GPU_SUMCHECK_THRESHOLD)
/// 2. Partial Eval (ZK_GPU_PARTIAL_EVAL_THRESHOLD)
/// 3. Fused Permute + Partial Eval (ZK_GPU_FUSED_THRESHOLD)
/// 4. Opening Proofs (CPU_OPEN_THRESHOLD)
///
/// Usage:
///   CUDA_VISIBLE_DEVICES=0 cargo run --release --bin bench_thresholds
use std::process::Command;
use std::time::Instant;

use goldilocks_cuda::basefold::{BasefoldCommitment, BasefoldTable};
use goldilocks_cuda::{GoldilocksExt2, GoldilocksField};
use rand::Rng;

use zk_torch_3::basicblock::einsum::{partial_eval_ext2_cpu, permute_evals_by_ranges};
use zk_torch_3::commit::cpu_basefold::cpu_full_open_ext2;
use zk_torch_3::sumcheck::CpuLinearSumcheckProverExt2;
use zk_torch_3::sumcheck::GpuLinearSumcheckProver;
use zk_torch_3::transcript::Transcript;

const NUM_TRIALS: usize = 3;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn gpu_name() -> String {
    Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.lines().next().unwrap_or("unknown").trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn random_base_poly(n: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..1usize << n)
        .map(|_| GoldilocksField(rng.gen::<u64>() % zk_torch_3::GOLDILOCKS_PRIME))
        .collect()
}

fn random_ext2_poly(n: usize) -> Vec<GoldilocksExt2> {
    let mut rng = rand::thread_rng();
    (0..1usize << n)
        .map(|_| {
            GoldilocksExt2::new(
                GoldilocksField(rng.gen::<u64>() % zk_torch_3::GOLDILOCKS_PRIME),
                GoldilocksField(rng.gen::<u64>() % zk_torch_3::GOLDILOCKS_PRIME),
            )
        })
        .collect()
}

fn random_ext2_challenges(m: usize) -> Vec<GoldilocksExt2> {
    let mut rng = rand::thread_rng();
    (0..m)
        .map(|_| {
            GoldilocksExt2::new(
                GoldilocksField(rng.gen::<u64>() % zk_torch_3::GOLDILOCKS_PRIME),
                GoldilocksField(rng.gen::<u64>() % zk_torch_3::GOLDILOCKS_PRIME),
            )
        })
        .collect()
}

fn median(times: &mut Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn print_row(n: usize, cpu_ms: f64, gpu_ms: f64) {
    let winner = if cpu_ms <= gpu_ms { "CPU" } else { "GPU" };
    println!(
        "  n={:<3} CPU: {:>8.2}ms  GPU: {:>8.2}ms  winner: {}",
        n, cpu_ms, gpu_ms, winner
    );
}

// ── Bench 1: Sumcheck ────────────────────────────────────────────────────────

fn bench_sumcheck() -> usize {
    println!("--- Sumcheck (ZK_GPU_SUMCHECK_THRESHOLD) ---");

    let sweep = [8, 10, 12, 14, 16, 18, 20];
    let mut last_cpu_win = sweep[0];

    for &n in &sweep {
        // Generate random Ext2 polys
        let poly_template_1 = random_ext2_poly(n);
        let poly_template_2 = random_ext2_poly(n);

        // CPU trials
        let mut cpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            let mut polys = vec![poly_template_1.clone(), poly_template_2.clone()];
            let mut t = Transcript::new(b"bench_sc");
            let mut prover = CpuLinearSumcheckProverExt2::new(n, 2, &mut t);

            let start = Instant::now();
            let _proof = prover.prove(&mut polys, &mut t);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // GPU trials
        let mut gpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            let p1 = poly_template_1.clone();
            let p2 = poly_template_2.clone();
            let mut t = Transcript::new(b"bench_sc");
            let mut prover = GpuLinearSumcheckProver::new(n, 2, &mut t);

            goldilocks_cuda::synchronize().ok();
            let start = Instant::now();
            let _proof = prover.prove_ext2(&[p1, p2], &mut t);
            goldilocks_cuda::synchronize().ok();
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let cpu_ms = median(&mut cpu_times);
        let gpu_ms = median(&mut gpu_times);
        print_row(n, cpu_ms, gpu_ms);

        if cpu_ms <= gpu_ms {
            last_cpu_win = n;
        }
    }

    // Crossover = last n where CPU wins; threshold should be one step above
    let threshold = last_cpu_win + 2;
    println!("  Recommended: ZK_GPU_SUMCHECK_THRESHOLD={}", threshold);
    println!();
    threshold
}

// ── Bench 2: Partial Eval ────────────────────────────────────────────────────

fn bench_partial_eval() -> usize {
    println!("--- Partial Eval (ZK_GPU_PARTIAL_EVAL_THRESHOLD) ---");

    let sweep = [10, 12, 14, 16, 18, 20, 22];
    let mut last_cpu_win = sweep[0];

    for &n in &sweep {
        let evals = random_base_poly(n);
        let m = n / 2;
        let challenges = random_ext2_challenges(m);

        // CPU trials
        let mut cpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            let start = Instant::now();
            let _result = partial_eval_ext2_cpu(&evals, &challenges);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // GPU trials
        let mut gpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            goldilocks_cuda::synchronize().ok();
            let start = Instant::now();
            let _result = goldilocks_cuda::partial_eval::partial_eval_ext2(&evals, &challenges)
                .expect("GPU partial_eval_ext2 failed");
            goldilocks_cuda::synchronize().ok();
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let cpu_ms = median(&mut cpu_times);
        let gpu_ms = median(&mut gpu_times);
        print_row(n, cpu_ms, gpu_ms);

        if cpu_ms <= gpu_ms {
            last_cpu_win = n;
        }
    }

    let threshold = last_cpu_win + 2;
    println!(
        "  Recommended: ZK_GPU_PARTIAL_EVAL_THRESHOLD={}",
        threshold
    );
    println!();
    threshold
}

// ── Bench 3: Fused Permute + Partial Eval ────────────────────────────────────

fn bench_fused() -> usize {
    println!("--- Fused Permute+PartialEval (ZK_GPU_FUSED_THRESHOLD) ---");

    let sweep = [10, 12, 14, 16, 18, 20, 22];
    let mut last_cpu_win = sweep[0];

    for &n in &sweep {
        let evals = random_base_poly(n);
        let m = n / 2;
        let challenges = random_ext2_challenges(m);
        // Swap-halves permutation: [m..n, 0..m]
        let ranges: Vec<(usize, usize)> = vec![(m, n), (0, m)];

        // CPU fallback: permute + partial eval
        let mut cpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            let start = Instant::now();
            let permuted = permute_evals_by_ranges(&evals, n, &ranges);
            let _result = partial_eval_ext2_cpu(&permuted, &challenges);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // GPU fused
        let mut gpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            goldilocks_cuda::synchronize().ok();
            let start = Instant::now();
            let _result = goldilocks_cuda::partial_eval::fused_permute_partial_eval(
                &evals, &challenges, &ranges, n,
            )
            .expect("GPU fused_permute_partial_eval failed");
            goldilocks_cuda::synchronize().ok();
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let cpu_ms = median(&mut cpu_times);
        let gpu_ms = median(&mut gpu_times);
        print_row(n, cpu_ms, gpu_ms);

        if cpu_ms <= gpu_ms {
            last_cpu_win = n;
        }
    }

    let threshold = last_cpu_win + 2;
    println!("  Recommended: ZK_GPU_FUSED_THRESHOLD={}", threshold);
    println!();
    threshold
}

// ── Bench 4: Opening Proofs ──────────────────────────────────────────────────

fn bench_opening() -> usize {
    println!("--- Opening Proofs (CPU_OPEN_THRESHOLD) ---");

    let sweep = [8, 10, 12, 14, 16, 18, 20];
    let log_rate = 3;
    let num_queries = 34;
    let seed = 42u64;
    let max_n = *sweep.last().unwrap();

    // Generate table once at max size
    let mut table = BasefoldTable::generate(max_n, log_rate, max_n, seed);
    table.upload().expect("table upload failed");

    let mut last_cpu_win = sweep[0];

    for &n in &sweep {
        let evals = random_base_poly(n);
        let point = random_ext2_challenges(n);

        // Commit once for this n
        let commitment =
            BasefoldCommitment::commit(&evals, n, log_rate).expect("commit failed");

        // CPU trials
        let mut cpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            let mut t = Transcript::new(b"bench_open");
            let start = Instant::now();
            let _proof = cpu_full_open_ext2(&commitment, &point, &table, &mut t, num_queries);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // GPU trials
        let mut gpu_times = Vec::new();
        for _ in 0..NUM_TRIALS {
            let mut t = Transcript::new(b"bench_open");
            goldilocks_cuda::synchronize().ok();
            let start = Instant::now();
            let _proof = commitment
                .open_ext2(&point, &table, &mut t, num_queries)
                .expect("GPU open_ext2 failed");
            goldilocks_cuda::synchronize().ok();
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let cpu_ms = median(&mut cpu_times);
        let gpu_ms = median(&mut gpu_times);
        print_row(n, cpu_ms, gpu_ms);

        if cpu_ms <= gpu_ms {
            last_cpu_win = n;
        }
    }

    let threshold = last_cpu_win + 2;
    println!("  Recommended: CPU_OPEN_THRESHOLD={}", threshold);
    println!();
    threshold
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    goldilocks_cuda::init().expect("CUDA init failed");
    goldilocks_cuda::init_device().expect("CUDA init_device failed");

    println!("=== GPU Threshold Auto-Tune ===");
    println!("GPU: {}", gpu_name());
    println!();

    // Warmup: small GPU kernel to init CUDA context
    {
        let evals = random_base_poly(10);
        let chals = random_ext2_challenges(5);
        let _ = goldilocks_cuda::partial_eval::partial_eval_ext2(&evals, &chals);
        goldilocks_cuda::synchronize().ok();
    }

    let sc = bench_sumcheck();
    let pe = bench_partial_eval();
    let fu = bench_fused();
    let op = bench_opening();

    println!("=== Recommended Configuration ===");
    println!(
        "export ZK_GPU_SUMCHECK_THRESHOLD={} ZK_GPU_PARTIAL_EVAL_THRESHOLD={} ZK_GPU_FUSED_THRESHOLD={} CPU_OPEN_THRESHOLD={}",
        sc, pe, fu, op
    );
}
