//! GPT-J 6B end-to-end prover binary. Ports zk-torch-3's `bin/gptj.rs`
//! to the zk-torch-4 commit/prove API.
//!
//! Defaults: GPT-J 6B (`num_heads = 16, head_dim = 256, hidden_dim = 4096,
//! ffn_dim = 16384, vocab = 50400`, `num_layers = 1`, `seq_len = 1`).
//! Override via env vars. Full-config vocab matmul has arity ≥ 27 — needs
//! a large `MAX_NUM_VARS` or smaller `VOCAB` for fits.
//!
//! All weights are all-zero; input uses the layout-aware varied pattern
//! `(i / stride_last) % 16` so LayerNorm reciprocity rounds exactly.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::gptj::gpt_j_6b;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::SF_LOG;

fn zero_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    // Parallel/alloc_zeroed path; see zero_witness_vec.
    zk_torch_4::zero_witness_vec(size)
}

/// GPT-J subtracts the per-row mean before the RMS step, so we use the
/// exact-round 0/2048-alternating pattern along the LAST axis: after
/// mean subtraction every entry is ±1024 (= ±1.0 in float) and the
/// LN reciprocity gate `z ≈ sf` is satisfied exactly at the default
/// tolerance = 2. See the whisper bin for the same trick.
fn small_varied_input(size: usize, stride_last: usize) -> Vec<AlmostGoldilocksField> {
    (0..size)
        .map(|i| AlmostGoldilocksField(if (i / stride_last) % 2 == 0 { 0 } else { 2048 }))
        .collect()
}

#[allow(clippy::type_complexity)]
fn gen_gptj_weights(num_layers: usize, hidden_dim: usize, ffn_dim: usize, vocab: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>,
    Witness, Witness, Witness, Witness,
) {
    let mut attn_norm_w = Vec::new();
    let mut attn_q_w    = Vec::new();
    let mut attn_k_w    = Vec::new();
    let mut attn_v_w    = Vec::new();
    let mut attn_o_w    = Vec::new();
    let mut attn_norm_b = Vec::new();
    let mut proj_1_w    = Vec::new();
    let mut proj_2_w    = Vec::new();
    let mut proj_1_b    = Vec::new();
    let mut proj_2_b    = Vec::new();

    let hd_pad = hidden_dim.next_power_of_two();
    let ffn_pad = ffn_dim.next_power_of_two();
    let vocab_pad = vocab.next_power_of_two();

    for _ in 0..num_layers {
        attn_norm_w.push(Witness::new(vec![hidden_dim], zero_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_w.push(Witness::new(vec![hidden_dim, hidden_dim], zero_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_w.push(Witness::new(vec![hidden_dim, hidden_dim], zero_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_w.push(Witness::new(vec![hidden_dim, hidden_dim], zero_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_w.push(Witness::new(vec![hidden_dim, hidden_dim], zero_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_norm_b.push(Witness::new(vec![hidden_dim], zero_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_w.push(Witness::new(vec![hidden_dim, ffn_dim], zero_field_vec(hd_pad * ffn_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_w.push(Witness::new(vec![ffn_dim, hidden_dim], zero_field_vec(ffn_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_b.push(Witness::new(vec![ffn_dim], zero_field_vec(ffn_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_b.push(Witness::new(vec![hidden_dim], zero_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
    }
    let ln_w = Witness::new(vec![hidden_dim], zero_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant);
    let ln_b = Witness::new(vec![hidden_dim], zero_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant);
    let matmul_w = Witness::new(vec![hidden_dim, vocab], zero_field_vec(hd_pad * vocab_pad), DataType::Float, *SF_LOG, Role::Constant);
    let matmul_b = Witness::new(vec![vocab], zero_field_vec(vocab_pad), DataType::Float, *SF_LOG, Role::Constant);
    (
        attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
        attn_norm_b,
        proj_1_w, proj_2_w,
        proj_1_b, proj_2_b,
        ln_w, ln_b, matmul_w, matmul_b,
    )
}

fn demo_seed() -> Seed {
    Seed([
        0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
        0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
    ])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    // Full GPT-J 6B: vocab matmul 4096 × 50400 (padded 4096 × 65536) → arity 28.
    // Bump MAX_NUM_VARS or pass a smaller VOCAB for toy runs.
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_heads: usize = std::env::var("NUM_HEADS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let head_dim: usize = std::env::var("HEAD_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(256);
    let hidden_dim = num_heads * head_dim;     // 4096
    let ffn_dim: usize = std::env::var("FFN_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(hidden_dim * 4);
    let vocab: usize = std::env::var("VOCAB").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(50400);

    println!("=== GPT-J 6B on Almost-Goldilocks ===");
    println!("num_layers={} seq_len={} max_num_vars={} (threads={})",
             num_layers, seq_len, max_num_vars, rayon::current_num_threads());
    println!("hidden_dim={} ffn_dim={} num_heads={} head_dim={} vocab={}",
             hidden_dim, ffn_dim, num_heads, head_dim, vocab);

    // ---- 1. Generate weights + build DAG ----
    let t0 = Instant::now();
    let weights = gen_gptj_weights(num_layers, hidden_dim, ffn_dim, vocab);
    println!("Weight gen: {:?}", t0.elapsed());

    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq_len, hidden_dim], DataType::Float);
    let _output = g.pipe(
        &[x],
        gpt_j_6b(
            weights.0, weights.1, weights.2, weights.3, weights.4,
            weights.5,
            weights.6, weights.7,
            weights.8, weights.9,
            weights.10, weights.11, weights.12, weights.13,
            num_heads, head_dim, seq_len,
        ),
    );
    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // ---- 2. Forward pass on varied-along-hidden input ----
    let pad: usize = [1usize, seq_len, hidden_dim].iter().map(|&s| s.next_power_of_two()).product();
    let stride_last = seq_len.next_power_of_two();
    let input = Witness::new(
        vec![1, seq_len, hidden_dim],
        small_varied_input(pad, stride_last),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Forward: {:?}", t2.elapsed());

    // ---- 3. Commit ----
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses, &mut store);
    let offline_commit = t_off.elapsed();
    let t_on = Instant::now();
    dag.commit_remaining(&witnesses, &mut store);
    let online_commit = t_on.elapsed();
    println!("Commit (offline, amortized): {:?}", offline_commit);
    println!("Commit (online, prover time): {:?}", online_commit);

    // ---- 4. Prove + verify ----
    let mut t_prove = Transcript::new(b"zkml-gptj");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-gptj");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
}
