//! Basefold PCS benchmark.
//!
//! Usage:
//!   cargo run --release --example bench_basefold
//!   cargo run --release --example bench_basefold -- --num-vars 20 --log-rate 1 --num-queries 32

use goldilocks_cuda::prelude::*;
use goldilocks_cuda::memory::synchronize;
use std::time::Instant;

// ── CLI argument parsing (minimal, no deps) ──

struct Args {
    num_vars_list: Vec<usize>,
    log_rate: usize,
    num_queries: usize,
    warmup: usize,
    iters: usize,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut num_vars_list: Option<Vec<usize>> = None;
    let mut log_rate = 1usize;
    let mut num_queries = 16usize;
    let mut warmup = 1usize;
    let mut iters = 3usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--num-vars" => {
                i += 1;
                // Accept comma-separated: --num-vars 10,14,18,20
                let vals: Vec<usize> = args[i]
                    .split(',')
                    .map(|s| s.trim().parse().expect("invalid num-vars"))
                    .collect();
                num_vars_list = Some(vals);
            }
            "--log-rate" => {
                i += 1;
                log_rate = args[i].parse().expect("invalid log-rate");
            }
            "--num-queries" => {
                i += 1;
                num_queries = args[i].parse().expect("invalid num-queries");
            }
            "--warmup" => {
                i += 1;
                warmup = args[i].parse().expect("invalid warmup");
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().expect("invalid iters");
            }
            "--help" | "-h" => {
                eprintln!("Usage: bench_basefold [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --num-vars N[,N,...]   Number of variables (default: 10,14,18,20)");
                eprintln!("  --log-rate R            Log2 of rate (default: 1)");
                eprintln!("  --num-queries Q         Number of FRI queries (default: 16)");
                eprintln!("  --warmup W              Warmup iterations (default: 1)");
                eprintln!("  --iters I               Benchmark iterations (default: 3)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        num_vars_list: num_vars_list.unwrap_or_else(|| vec![10, 14, 18, 20]),
        log_rate,
        num_queries,
        warmup,
        iters,
    }
}

// ── Timing helper ──

fn bench<F: FnMut()>(label: &str, warmup: usize, iters: usize, mut f: F) -> f64 {
    // warmup
    for _ in 0..warmup {
        f();
    }
    synchronize().unwrap();

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        synchronize().unwrap();
        let t0 = Instant::now();
        f();
        synchronize().unwrap();
        times.push(t0.elapsed().as_secs_f64());
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(0.0f64, f64::max);

    println!(
        "  {:<32}  avg {:>10.3} ms   min {:>10.3} ms   max {:>10.3} ms",
        label,
        avg * 1e3,
        min * 1e3,
        max * 1e3,
    );

    avg
}

// ── CPU multilinear eval (for generating correct evaluation claims) ──

const P: u64 = goldilocks_cuda::GOLDILOCKS_PRIME;

fn gl_add(a: u64, b: u64) -> u64 {
    let s = a.wrapping_add(b);
    if s < a || s >= P { s.wrapping_sub(P) } else { s }
}

fn gl_mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % P as u128) as u64
}

fn gl_sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a.wrapping_add(P).wrapping_sub(b) }
}

fn cpu_multilinear_eval(evals: &[GoldilocksField], point: &[GoldilocksField]) -> u64 {
    let n = evals.len();
    let nv = point.len();
    let mut sum = 0u64;
    for x in 0..n {
        let mut eq_val = 1u64;
        for i in 0..nv {
            let xi = ((x >> i) & 1) as u64;
            if xi == 1 {
                eq_val = gl_mul(eq_val, point[i].0);
            } else {
                eq_val = gl_mul(eq_val, gl_sub(1, point[i].0));
            }
        }
        sum = gl_add(sum, gl_mul(evals[x].0, eq_val));
    }
    sum
}

// ── Main ──

fn main() {
    let args = parse_args();

    goldilocks_cuda::init().expect("Failed to initialize CUDA");
    let gpu = goldilocks_cuda::device_name(0).unwrap_or_else(|_| "unknown".into());
    println!("GPU: {}", gpu);
    println!(
        "Config: log_rate={}, num_queries={}, warmup={}, iters={}",
        args.log_rate, args.num_queries, args.warmup, args.iters
    );
    println!("{}", "=".repeat(90));

    for &num_vars in &args.num_vars_list {
        let n = 1usize << num_vars;
        let cw_len = 1usize << (num_vars + args.log_rate);

        println!();
        println!(
            "num_vars={} (n=2^{}={}, codeword_len={})",
            num_vars, num_vars, n, cw_len
        );
        println!("{}", "-".repeat(90));

        // Generate random-ish evaluations
        let evals: Vec<GoldilocksField> = (0..n as u64)
            .map(|i| {
                let mut x = i.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0x6A09E667F3BCC908);
                x ^= x >> 30;
                x = x.wrapping_mul(0xBF58476D1CE4E5B9);
                GoldilocksField(x % P)
            })
            .collect();

        // Ext2 point for single open (realistic: verifier challenges are in ext2)
        let ext2_point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111_111) % P),
                    GoldilocksField(((i as u64 + 1) * 222_222) % P),
                )
            })
            .collect();

        // Base-field point for batch open
        let point: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 111_111) % P))
            .collect();

        let log_rate = args.log_rate;
        let num_queries = args.num_queries;

        // ── Commit ──
        let mut comm = None;
        bench("commit", args.warmup, args.iters, || {
            comm = Some(
                BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap(),
            );
        });
        let comm = comm.unwrap();

        // ── Table generation + upload ──
        let mut table = None;
        bench("table generate + upload", args.warmup, args.iters, || {
            let mut t = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
            t.upload().unwrap();
            table = Some(t);
        });
        let table = table.unwrap();

        // ── Single Open (ext2 point) ──
        let mut proof_ext2 = None;
        bench("open_ext2 (single poly)", args.warmup, args.iters, || {
            let mut transcript = TestTranscript::new(999);
            proof_ext2 = Some(
                comm.open_ext2(&ext2_point, &table, &mut transcript, num_queries)
                    .unwrap(),
            );
        });
        let proof_ext2 = proof_ext2.unwrap();

        // ── Batch Open: 2 polys, same point ──
        let evals_b: Vec<GoldilocksField> = (0..n as u64)
            .map(|i| {
                let mut x = i.wrapping_mul(0x517CC1B727220A95).wrapping_add(0x3C6EF372FE94F82B);
                x ^= x >> 30;
                x = x.wrapping_mul(0x94D049BB133111EB);
                GoldilocksField(x % P)
            })
            .collect();
        let comm_b = BasefoldCommitment::commit(&evals_b, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let eval_a = if num_vars <= 16 {
            cpu_multilinear_eval(&evals, &point)
        } else {
            // For large sizes, open poly A individually to get its base-field eval
            let mut t = TestTranscript::new(11111);
            let pa = comm.open(&point, &table, &mut t, num_queries).unwrap();
            synchronize().unwrap();
            pa.eval.0
        };
        let eval_b_val = if num_vars <= 16 {
            cpu_multilinear_eval(&evals_b, &point)
        } else {
            // Open poly B individually to get its eval
            let mut t = TestTranscript::new(12345);
            let pb = comm_b.open(&point, &table, &mut t, num_queries).unwrap();
            synchronize().unwrap();
            pb.eval.0
        };

        let claims = vec![
            Evaluation::new(0, 0, GoldilocksField(eval_a)),
            Evaluation::new(1, 0, GoldilocksField(eval_b_val)),
        ];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm, &comm_b];
        let points_refs: Vec<&[GoldilocksField]> = vec![&point];

        let mut batch_proof = None;
        bench("batch_open (2 poly, same point)", args.warmup, args.iters, || {
            let mut transcript = TestTranscript::new(888);
            batch_proof = Some(
                batch_open(
                    &comms_refs,
                    &points_refs,
                    &claims,
                    &table,
                    &mut transcript,
                    num_queries,
                )
                .unwrap(),
            );
        });
        let batch_proof = batch_proof.unwrap();

        // ── Batch Verify: 2 polys, same point ──
        bench("batch_verify (2 poly, same point)", args.warmup, args.iters, || {
            let mut transcript = TestTranscript::new(888);
            let valid = BasefoldVerifier::batch_verify(
                &[comm.root.clone(), comm_b.root.clone()],
                &points_refs,
                &claims,
                &batch_proof,
                &table,
                &mut transcript,
            )
            .unwrap();
            assert!(valid);
        });

        // ── Batch Open: 2 polys, 2 different points ──
        let point_b: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 333_333) % P))
            .collect();
        let eval_b2 = if num_vars <= 16 {
            cpu_multilinear_eval(&evals_b, &point_b)
        } else {
            let mut t = TestTranscript::new(54321);
            let pb = comm_b.open(&point_b, &table, &mut t, num_queries).unwrap();
            synchronize().unwrap();
            pb.eval.0
        };

        let claims_2pt = vec![
            Evaluation::new(0, 0, GoldilocksField(eval_a)),
            Evaluation::new(1, 1, GoldilocksField(eval_b2)),
        ];
        let points_2pt: Vec<&[GoldilocksField]> = vec![&point, &point_b];

        let mut batch_proof_2pt = None;
        bench("batch_open (2 poly, diff points)", args.warmup, args.iters, || {
            let mut transcript = TestTranscript::new(777);
            batch_proof_2pt = Some(
                batch_open(
                    &comms_refs,
                    &points_2pt,
                    &claims_2pt,
                    &table,
                    &mut transcript,
                    num_queries,
                )
                .unwrap(),
            );
        });
        let batch_proof_2pt = batch_proof_2pt.unwrap();

        bench("batch_verify (2 poly, diff points)", args.warmup, args.iters, || {
            let mut transcript = TestTranscript::new(777);
            let valid = BasefoldVerifier::batch_verify(
                &[comm.root.clone(), comm_b.root.clone()],
                &points_2pt,
                &claims_2pt,
                &batch_proof_2pt,
                &table,
                &mut transcript,
            )
            .unwrap();
            assert!(valid);
        });

        // ── Proof sizes ──
        let single_oracles = proof_ext2.sumcheck_oracles.len();
        let single_queries = proof_ext2.query_proofs.len();
        let batch_outer = batch_proof.outer_sumcheck_oracles.len();
        let batch_inner = batch_proof.inner_sumcheck_oracles.len();
        let batch_queries = batch_proof.combined_query_proofs.len();

        println!();
        println!("  Proof stats:");
        println!(
            "    single (ext2):  {} sumcheck oracles, {} folded roots, {} queries, final_cw len={}",
            single_oracles,
            proof_ext2.folded_roots.len(),
            single_queries,
            proof_ext2.final_codeword.len(),
        );
        println!(
            "    batch:   {} outer + {} inner oracles, {} folded roots, {} queries",
            batch_outer, batch_inner, batch_proof.folded_roots.len(), batch_queries,
        );
    }

    println!();
    println!("{}", "=".repeat(90));
    println!("Done.");
}
