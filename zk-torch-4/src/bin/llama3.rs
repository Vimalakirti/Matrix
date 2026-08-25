//! Llama 3.1 8B end-to-end prover binary. Ports zk-torch-3's
//! `bin/llama3.rs` to the zk-torch-4 commit/prove API.
//!
//! Defaults: `HIDDEN_DIM=4096`, `NUM_HEADS=32`, `NUM_KV_HEADS=8`,
//! `HEAD_DIM=128`, `FFN_DIM=14336`, `VOCAB_SIZE=128256`. The DAG (see
//! `dag::llama::llama3_8b`) does NOT shard the logits head or the FFN
//! projections — full vocab/FFN matmuls have arities far above 22, so
//! the default `MAX_NUM_VARS` is bumped accordingly. Use smaller dims
//! (e.g. `HIDDEN_DIM=512 VOCAB_SIZE=128`) for quick local smoke runs.
//! Override `NUM_LAYERS`, `SEQ_LEN`, `MAX_NUM_VARS`, `ZK4_B`,
//! `ZK4_BASE`, `HIDDEN_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM`,
//! `FFN_DIM`, `VOCAB_SIZE` via env vars.
//!
//! Use `small_input` (≈1.0 in fixed-point) so RMSNorm's reciprocity gate
//! `r²·mean(x²) ≈ sf` holds with rounding error 0.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::llama::llama3_8b;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::SF_LOG;

fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    // Parallel/alloc_zeroed path; see zero_witness_vec.
    zk_torch_4::zero_witness_vec(size)
}

fn small_input(size: usize) -> Vec<AlmostGoldilocksField> {
    vec![AlmostGoldilocksField(1024); size]
}

fn padded_size(shape: &[usize]) -> usize {
    shape.iter().map(|&s| s.next_power_of_two()).product()
}

#[allow(clippy::type_complexity)]
fn gen_llama3_weights(
    num_layers: usize,
    hidden_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    logits_shards: usize,
    ffn_shards: usize,
) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Vec<Witness>>, Vec<Vec<Witness>>, Vec<Vec<Witness>>,
    Witness, Vec<Witness>,
) {
    // Shard the FFN along `ffn` and the lm_head along `vocab`, exactly as
    // llama_2_7b does. Unsharded, Llama-3's head is one 2^29 committed edge.
    let hd_pad = hidden_dim.next_power_of_two();
    let ffn_pad = ffn_dim.next_power_of_two();
    let vocab_pad = vocab_size.next_power_of_two();
    assert!(vocab_pad % logits_shards == 0, "vocab_pad {} not divisible by {}", vocab_pad, logits_shards);
    assert!(ffn_pad % ffn_shards == 0, "ffn_pad {} not divisible by {}", ffn_pad, ffn_shards);
    let shard_vocab = vocab_size / logits_shards;
    let shard_vocab_pad = vocab_pad / logits_shards;
    let shard_ffn = ffn_dim / ffn_shards;
    let shard_ffn_pad = ffn_pad / ffn_shards;
    let mut attn_norm_w_vec = Vec::new();
    let mut attn_q_w_vec    = Vec::new();
    let mut attn_k_w_vec    = Vec::new();
    let mut attn_v_w_vec    = Vec::new();
    let mut attn_o_w_vec    = Vec::new();
    let mut proj_norm_w_vec = Vec::new();
    let mut proj_1_w_vec    = Vec::new();
    let mut proj_2_w_vec    = Vec::new();
    let mut proj_3_w_vec    = Vec::new();

    for _ in 0..num_layers {
        attn_norm_w_vec.push(Witness::new(
            vec![hidden_dim], rand_field_vec(hidden_dim.next_power_of_two()),
            DataType::Float, *SF_LOG, Role::Constant));
        attn_q_w_vec.push(Witness::new(
            vec![hidden_dim, hidden_dim],
            rand_field_vec(padded_size(&[hidden_dim, hidden_dim])),
            DataType::Float, *SF_LOG, Role::Constant));
        attn_k_w_vec.push(Witness::new(
            vec![hidden_dim, kv_dim],
            rand_field_vec(padded_size(&[hidden_dim, kv_dim])),
            DataType::Float, *SF_LOG, Role::Constant));
        attn_v_w_vec.push(Witness::new(
            vec![hidden_dim, kv_dim],
            rand_field_vec(padded_size(&[hidden_dim, kv_dim])),
            DataType::Float, *SF_LOG, Role::Constant));
        attn_o_w_vec.push(Witness::new(
            vec![hidden_dim, hidden_dim],
            rand_field_vec(padded_size(&[hidden_dim, hidden_dim])),
            DataType::Float, *SF_LOG, Role::Constant));
        proj_norm_w_vec.push(Witness::new(
            vec![hidden_dim], rand_field_vec(hidden_dim.next_power_of_two()),
            DataType::Float, *SF_LOG, Role::Constant));
        proj_1_w_vec.push((0..ffn_shards).map(|_| Witness::new(
            vec![hidden_dim, shard_ffn],
            rand_field_vec(hd_pad * shard_ffn_pad),
            DataType::Float, *SF_LOG, Role::Constant)).collect::<Vec<_>>());
        proj_2_w_vec.push((0..ffn_shards).map(|_| Witness::new(
            vec![hidden_dim, shard_ffn],
            rand_field_vec(hd_pad * shard_ffn_pad),
            DataType::Float, *SF_LOG, Role::Constant)).collect::<Vec<_>>());
        proj_3_w_vec.push((0..ffn_shards).map(|_| Witness::new(
            vec![shard_ffn, hidden_dim],
            rand_field_vec(shard_ffn_pad * hd_pad),
            DataType::Float, *SF_LOG, Role::Constant)).collect::<Vec<_>>());
    }

    let layer_norm_w = Witness::new(
        vec![hidden_dim], rand_field_vec(hidden_dim.next_power_of_two()),
        DataType::Float, *SF_LOG, Role::Constant);
    let logits_w: Vec<Witness> = (0..logits_shards).map(|_| Witness::new(
        vec![hidden_dim, shard_vocab],
        rand_field_vec(hd_pad * shard_vocab_pad),
        DataType::Float, *SF_LOG, Role::Constant)).collect();

    (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec, proj_3_w_vec,
        layer_norm_w, logits_w,
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
    // Llama 3.1 8B's logits head (hidden × vocab = 4096 × 128256 padded
    // to 4096 × 131072 = 2^29) and FFN (hidden × ffn = 4096 × 14336 padded
    // to 4096 × 16384 = 2^26) both exceed the 22-arity budget. The DAG
    // function `llama3_8b` does not shard them, so MAX_NUM_VARS must be
    // sized to fit the largest committed edge. We default to 29 to be
    // safe at full config; users on smaller hardware should override
    // model dims (e.g. `HIDDEN_DIM=512 VOCAB_SIZE=128`) and drop this
    // accordingly.
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(29);

    let hidden_dim: usize = std::env::var("HIDDEN_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4096);
    let num_heads: usize = std::env::var("NUM_HEADS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let num_kv_heads: usize = std::env::var("NUM_KV_HEADS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(8);
    let head_dim: usize = std::env::var("HEAD_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(128);
    let ffn_dim: usize = std::env::var("FFN_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(14336);
    let vocab_size: usize = std::env::var("VOCAB_SIZE").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(128256);

    let kv_dim = num_kv_heads * head_dim;

    println!("=== Llama 3.1 8B on Almost-Goldilocks ===");
    println!("num_layers={} seq_len={} max_num_vars={} (threads={})",
             num_layers, seq_len, max_num_vars, rayon::current_num_threads());
    println!("hidden_dim={} ffn_dim={} num_heads={} num_kv_heads={} head_dim={} vocab={}",
             hidden_dim, ffn_dim, num_heads, num_kv_heads, head_dim, vocab_size);

    let t0 = Instant::now();
    let logits_shards: usize = std::env::var("LOGITS_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let ffn_shards: usize = std::env::var("FFN_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let weights = gen_llama3_weights(num_layers, hidden_dim, kv_dim, ffn_dim,
                                     vocab_size, logits_shards, ffn_shards);
    println!("Weight gen: {:?}", t0.elapsed());

    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq_len, hidden_dim], DataType::Float);
    let _ = g.pipe(
        &[x],
        llama3_8b(
            weights.0, weights.1, weights.2, weights.3, weights.4,
            weights.5, weights.6, weights.7, weights.8,
            weights.9, weights.10,
            seq_len, head_dim, num_heads, num_kv_heads, vocab_size,
        ),
    );

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    let pad: usize = padded_size(&[1, seq_len, hidden_dim]);
    let input = Witness::new(
        vec![1, seq_len, hidden_dim],
        small_input(pad),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Forward: {:?}", t2.elapsed());

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

    let mut t_prove = Transcript::new(b"zkml-llama3");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-llama3");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
    if !verified {
        eprintln!("WARN: verifier rejected — see PROVER_PERFORMANCE.md for likely causes.");
    }
}
