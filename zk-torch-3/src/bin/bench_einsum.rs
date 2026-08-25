//! Einsum scaling benchmark: measures total prove time and isolates
//! non-sumcheck overhead (permute, partial_eval, eq construction, claim extraction).
//!
//! For matmul "ij,jk->ik":
//!   - Free indices i,k are fixed via partial_eval (NOT sumcheck)
//!   - Only summation index j goes through sumcheck
//!   - actual_sumcheck_rounds = j_bits (not i+j+k bits)
//!
//! Usage:
//!     cargo run --release --bin bench_einsum

use std::time::Instant;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use rand::Rng;

use zk_torch_3::basicblock::einsum::Einsum;
use zk_torch_3::basicblock::BasicBlock;
use zk_torch_3::dag::{Claim, DataType, Role, Witness};
use zk_torch_3::sumcheck::{CpuLinearSumcheckProverExt2, GpuLinearSumcheckProver};
use zk_torch_3::transcript::Transcript;

fn random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField(rng.gen::<u64>() % (1u64 << 32)))
        .collect()
}

fn random_ext2_vec(size: usize) -> Vec<GoldilocksExt2> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksExt2::new(
            GoldilocksField(rng.gen::<u64>() % (1u64 << 63)),
            GoldilocksField(rng.gen::<u64>() % (1u64 << 63)),
        ))
        .collect()
}

fn log2c(x: usize) -> usize {
    if x <= 1 { 0 } else { (x as f64).log2().ceil() as usize }
}

/// Run a pure sumcheck benchmark for the given rounds and num_polys.
fn bench_pure_sumcheck(rounds: usize, num_polys: usize) -> f64 {
    let size = 1usize << rounds;
    if rounds <= 14 {
        let mut polys: Vec<Vec<GoldilocksExt2>> = (0..num_polys)
            .map(|_| random_ext2_vec(size))
            .collect();
        let mut t = Transcript::new(b"pure_sc");
        let mut prover = CpuLinearSumcheckProverExt2::new(rounds, num_polys, &mut t);
        let start = Instant::now();
        let _ = prover.prove(&mut polys, &mut t);
        start.elapsed().as_secs_f64() * 1000.0
    } else {
        let polys: Vec<Vec<GoldilocksExt2>> = (0..num_polys)
            .map(|_| random_ext2_vec(size))
            .collect();
        let mut t = Transcript::new(b"pure_sc");
        let mut prover = GpuLinearSumcheckProver::new(rounds, num_polys, &mut t);
        let start = Instant::now();
        let _ = prover.prove_ext2(&polys, &mut t);
        start.elapsed().as_secs_f64() * 1000.0
    }
}

struct EinsumResult {
    total_ms: f64,
    pure_sc_ms: f64,
    sc_rounds: usize,
    sc_size: usize,
    num_polys: usize,
    input_a_n: usize,
    input_b_n: usize,
}

/// Benchmark matmul "ij,jk->ik".
fn bench_matmul(dim_i: usize, dim_j: usize, dim_k: usize) -> EinsumResult {
    let i_bits = log2c(dim_i);
    let j_bits = log2c(dim_j);
    let k_bits = log2c(dim_k);

    let i_pad = 1usize << i_bits;
    let j_pad = 1usize << j_bits;
    let k_pad = 1usize << k_bits;

    let einsum = Einsum::new(
        "ij,jk->ik",
        vec![vec![dim_i, dim_j], vec![dim_j, dim_k]],
        vec![dim_i, dim_k],
    );

    let w_a = Witness::new(
        vec![dim_i, dim_j],
        random_field_vec(i_pad * j_pad),
        DataType::Float, 0, Role::Auxiliary,
    );
    let w_b = Witness::new(
        vec![dim_j, dim_k],
        random_field_vec(j_pad * k_pad),
        DataType::Float, 0, Role::Auxiliary,
    );

    let outputs = einsum.run(&[&w_a, &w_b]);
    let w_out = &outputs[0];

    let out_n = w_out.data.as_ref().unwrap().n();
    let challenge_point = random_ext2_vec(out_n);
    let eval = w_out.data.as_ref().unwrap().evaluate_at_point_ext2(&challenge_point);

    let claim = Claim {
        edge_id: 2, sparse_id: 0,
        point: challenge_point, eval,
    };

    let witnesses = vec![&w_a, &w_b, w_out];
    let edge_ids = vec![0, 1, 2];

    let mut transcript = Transcript::new(b"bench_einsum");
    let t_total = Instant::now();
    let _ = einsum.prove(&witnesses, &edge_ids, &[&claim], &mut transcript);
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    // For matmul "ij,jk->ik":
    // free_once = [i, k], free_multi = [], summation = [j]
    // actual_sumcheck_rounds = j_bits (only the summation index)
    // num_polys = 3 (input_a after partial_eval, input_b after partial_eval, eq)
    // But eq is trivially all-ones when high_degree_challenge is empty,
    // so num_polys in the sumcheck = 2 inputs + 1 eq = 3
    let sc_rounds = j_bits;
    let sc_size = 1usize << sc_rounds;
    let num_polys = 3;

    let pure_sc_ms = bench_pure_sumcheck(sc_rounds, num_polys);

    EinsumResult {
        total_ms,
        pure_sc_ms,
        sc_rounds,
        sc_size,
        num_polys,
        input_a_n: i_bits + j_bits,
        input_b_n: j_bits + k_bits,
    }
}

fn main() {
    goldilocks_cuda::init().expect("CUDA init failed");

    // Warmup GPU
    {
        let polys: Vec<Vec<GoldilocksExt2>> = (0..3).map(|_| random_ext2_vec(1 << 14)).collect();
        let mut t = Transcript::new(b"warmup");
        let mut p = GpuLinearSumcheckProver::new(14, 3, &mut t);
        let _ = p.prove_ext2(&polys, &mut t);
    }

    println!("=== Einsum (matmul ij,jk->ik) Prove Breakdown ===");
    println!("Overhead = total_einsum - pure_sumcheck(j_bits, 3 polys)");
    println!("Overhead includes: permute, partial_eval, eq_construct, claim_extract");
    println!();

    // ===== Part 1: Vary j (summation dim) with i=1, k=4096 (LLaMA-like) =====
    println!("--- Part 1: Vary summation dim j (i=1, k=4096) ---");
    println!("{:<22} {:>5} {:>5} {:>6} {:>10} {:>10} {:>10} {:>7}",
        "Config", "j_bits", "sc_rnd", "sc_sz", "Total(ms)", "SC(ms)", "Ovhd(ms)", "Ovhd%");
    println!("{}", "-".repeat(85));

    for j in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let r = bench_matmul(1, j, 4096);
        let overhead_ms = r.total_ms - r.pure_sc_ms;
        let overhead_pct = overhead_ms / r.total_ms * 100.0;
        println!("{:<22} {:>5} {:>5} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>6.1}%",
            format!("1x{}x4096", j), log2c(j), r.sc_rounds, r.sc_size,
            r.total_ms, r.pure_sc_ms, overhead_ms, overhead_pct);
    }

    // ===== Part 2: Vary k (free dim) with i=1, j=4096 — tests partial_eval scaling =====
    println!();
    println!("--- Part 2: Vary free dim k (i=1, j=4096, sc_rounds fixed at 12) ---");
    println!("This isolates partial_eval cost: input B has {} vars, partial_eval fixes k_bits.",
        "j+k");
    println!("{:<22} {:>5} {:>5} {:>6} {:>10} {:>10} {:>10} {:>7}",
        "Config", "k_bits", "B_nvr", "B_sz", "Total(ms)", "SC(ms)", "Ovhd(ms)", "Ovhd%");
    println!("{}", "-".repeat(85));

    for k in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let r = bench_matmul(1, 4096, k);
        let overhead_ms = r.total_ms - r.pure_sc_ms;
        let overhead_pct = overhead_ms / r.total_ms * 100.0;
        println!("{:<22} {:>5} {:>5} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>6.1}%",
            format!("1x4096x{}", k), log2c(k), r.input_b_n, 1usize << r.input_b_n,
            r.total_ms, r.pure_sc_ms, overhead_ms, overhead_pct);
    }

    // ===== Part 3: Symmetric matmul NxNxN =====
    println!();
    println!("--- Part 3: Symmetric NxNxN matmul ---");
    println!("{:<22} {:>5} {:>5} {:>6} {:>10} {:>10} {:>10} {:>7}",
        "Config", "n_bits", "sc_rnd", "sc_sz", "Total(ms)", "SC(ms)", "Ovhd(ms)", "Ovhd%");
    println!("{}", "-".repeat(85));

    for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
        let r = bench_matmul(n, n, n);
        let overhead_ms = r.total_ms - r.pure_sc_ms;
        let overhead_pct = overhead_ms / r.total_ms * 100.0;
        println!("{:<22} {:>5} {:>5} {:>6} {:>10.3} {:>10.3} {:>10.3} {:>6.1}%",
            format!("{}x{}x{}", n, n, n), log2c(n), r.sc_rounds, r.sc_size,
            r.total_ms, r.pure_sc_ms, overhead_ms, overhead_pct);
    }

    // ===== Part 4: LLaMA-specific shapes (3 trials each) =====
    println!();
    println!("--- Part 4: LLaMA-specific shapes (3 trials) ---");
    let llama_configs: Vec<(usize, usize, usize, &str)> = vec![
        (1, 4096, 4096, "QKV proj (1x4096x4096)"),
        (1, 4096, 11008, "FFN gate/up (1x4096x11008)"),
        (1, 11008, 4096, "FFN down (1x11008x4096)"),
        (32, 1, 128, "Attn QK^T (32x1x128)"),
        (32, 128, 1, "Attn score*V (32x128x1)"),
    ];

    for (di, dj, dk, desc) in &llama_configs {
        println!("\n  {}:", desc);
        for trial in 0..3 {
            let r = bench_matmul(*di, *dj, *dk);
            let overhead_ms = r.total_ms - r.pure_sc_ms;
            println!("    trial {}: total={:>8.3}ms  sc={:>8.3}ms  overhead={:>8.3}ms ({:.1}%)  [sc_rounds={}, A_n={}, B_n={}]",
                trial, r.total_ms, r.pure_sc_ms, overhead_ms,
                overhead_ms / r.total_ms * 100.0,
                r.sc_rounds, r.input_a_n, r.input_b_n);
        }
    }
}
