//! Llama-2-7B end-to-end prover binary, modelled on `bin/gpt2.rs`.
//!
//! Uses the existing [`zk_torch_4::dag::llama::llama_2_7b`] DAG which
//! includes: RoPE, llama RMSNorm (with the `RMSReciprocal::run` fix
//! from the GPT-2 work), SwiGLU MLP (3 weight matrices), grouped
//! attention-output projection, and a final logits head to vocab 32000.
//!
//! Defaults: `num_heads = 32`, `head_dim = 128` (hidden_dim = 4096),
//! `ffn_dim = 11008`, `num_layers = 1`, `seq_len = 1`, `vocab = 32000`.
//! Override via `NUM_LAYERS`, `SEQ_LEN`, and `MAX_NUM_VARS` env vars.
//!
//! Random weights (all-zero) and `small_varied_input` (~1.0 in
//! fixed-point) so RMSNorm's reciprocity gate doesn't underflow.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::llama::llama_2_7b;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::SF_LOG;

fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    // Parallel/alloc_zeroed path; see zero_witness_vec.
    zk_torch_4::zero_witness_vec(size)
}

/// Per-position constant 1.0 (= field 1024 at SF=10). Llama's RMSNorm
/// doesn't subtract the mean (unlike LayerNorm), so a constant input
/// keeps `mean(x²) = sf` exactly — the reciprocity check
/// `r²·mean(x²) ≈ sf` then holds with rounding error 0, well within
/// tolerance = 2 in the `positive_4 = tolerance - (z - sf)` gate.
fn small_input(size: usize) -> Vec<AlmostGoldilocksField> {
    vec![AlmostGoldilocksField(1024); size]
}

#[allow(clippy::type_complexity)]
fn gen_llama2_weights(
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    vocab: usize,
    logits_shards: usize,
    ffn_shards: usize,
) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Vec<Witness>>, Vec<Vec<Witness>>, Vec<Vec<Witness>>,
    Witness, Vec<Witness>,
) {
    let hd_pad = hidden_dim.next_power_of_two();
    let ffn_pad = ffn_dim.next_power_of_two();
    let vocab_pad = vocab.next_power_of_two();
    assert!(vocab_pad % logits_shards == 0,
        "vocab_pad ({}) must be divisible by logits_shards ({})", vocab_pad, logits_shards);
    assert!(ffn_pad % ffn_shards == 0,
        "ffn_pad ({}) must be divisible by ffn_shards ({})", ffn_pad, ffn_shards);
    let shard_vocab = vocab / logits_shards;
    let shard_vocab_pad = vocab_pad / logits_shards;
    let shard_ffn = ffn_dim / ffn_shards;
    let shard_ffn_pad = ffn_pad / ffn_shards;
    let mut attn_norm_w = Vec::new();
    let mut attn_q_w    = Vec::new();
    let mut attn_k_w    = Vec::new();
    let mut attn_v_w    = Vec::new();
    let mut attn_o_w    = Vec::new();
    let mut proj_norm_w = Vec::new();
    let mut proj_1_w: Vec<Vec<Witness>> = Vec::new();
    let mut proj_2_w: Vec<Vec<Witness>> = Vec::new();
    let mut proj_3_w: Vec<Vec<Witness>> = Vec::new();
    for _ in 0..num_layers {
        attn_norm_w.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_w.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_w.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_w.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_w.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_norm_w.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        // SwiGLU sharded along the ffn axis: gate/up shape [hidden, ffn/N];
        // down shape [ffn/N, hidden]. Each shard's matmul wide-poly is
        // hidden·ffn/N — at hidden=4096, ffn=16384, N=16 → arity 22.
        let p1: Vec<Witness> = (0..ffn_shards).map(|_| {
            Witness::new(vec![hidden_dim, shard_ffn],
                rand_field_vec(hd_pad * shard_ffn_pad),
                DataType::Float, *SF_LOG, Role::Constant)
        }).collect();
        let p2: Vec<Witness> = (0..ffn_shards).map(|_| {
            Witness::new(vec![hidden_dim, shard_ffn],
                rand_field_vec(hd_pad * shard_ffn_pad),
                DataType::Float, *SF_LOG, Role::Constant)
        }).collect();
        let p3: Vec<Witness> = (0..ffn_shards).map(|_| {
            Witness::new(vec![shard_ffn, hidden_dim],
                rand_field_vec(shard_ffn_pad * hd_pad),
                DataType::Float, *SF_LOG, Role::Constant)
        }).collect();
        proj_1_w.push(p1);
        proj_2_w.push(p2);
        proj_3_w.push(p3);
    }
    let layer_norm_w = Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant);
    let logits_w_shards: Vec<Witness> = (0..logits_shards).map(|_| {
        Witness::new(
            vec![hidden_dim, shard_vocab],
            rand_field_vec(hd_pad * shard_vocab_pad),
            DataType::Float, *SF_LOG, Role::Constant,
        )
    }).collect();
    (
        attn_norm_w, attn_q_w, attn_k_w, attn_v_w, attn_o_w,
        proj_norm_w, proj_1_w, proj_2_w, proj_3_w,
        layer_norm_w, logits_w_shards,
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
    // Llama-2-7B: hidden=4096 (already pow2), ffn=11008 (pads to 16384=2^14).
    // FFN sparse aux: input_n = log2(seq * ffn_pad) = log2(1*16384) = 14;
    // aux arity = 14 + TABLE_COMMIT_LOG (=8 in bench_config.yaml) = 22.
    // Logits head: hidden × vocab = 4096 × 32768 (= 2^15 padded), so dense
    // is 2^27 — needs larger max_num_vars or bigger TABLE_COMMIT_LOG.
    // For the 1-layer / seq=1 bench, max_num_vars=22 fits the body; the
    // logits matmul is the biggest single edge. Override `MAX_NUM_VARS`
    // if benchmarking deeper.
    // Default 22: matches FFN_SHARDS=16 (arity 22 for the FFN matmul,
    // 4096 × 1024 padded). Bump to 23 if you set `FFN_SHARDS=8` to match
    // its arity-23 wide-poly.
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    // Dims default to Llama-2-7B. Override via env for smaller smoke
    // tests (`NUM_HEADS=4 HEAD_DIM=64 FFN_DIM=1024 VOCAB=128`) — the
    // existing CUDA einsum kernels have known size limits for very
    // large square matmuls (e.g. the 4096×32000 logits head).
    let num_heads: usize = std::env::var("NUM_HEADS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let head_dim: usize = std::env::var("HEAD_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(128);
    let hidden_dim = num_heads * head_dim;
    let ffn_dim: usize = std::env::var("FFN_DIM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(11008);
    let vocab: usize = std::env::var("VOCAB").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32000);
    // Logits sharding: split `vocab` across N independent einsums so the
    // wide-poly per-shard stays GPU-SP-sized. For full Llama-2-7B
    // (hidden=4096, vocab=32000 → vocab_pad=32768), N=32 keeps each shard
    // at arity 21 (4096 × 1024) — well within the GPU same-point budget.
    // LOGITS_SHARDS=32 keeps the logits matmul (4096 × 32000 padded
    // → arity 27) chunked into per-shard arity-21 einsums — within the
    // GPU same-point / multifold budget.
    let logits_shards: usize = std::env::var("LOGITS_SHARDS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    // FFN sharding: split each of SwiGLU's gate/up/down projections
    // along the ffn axis into N pieces.
    //   N=16 (default): arity 22, lowest GPU memory pressure — scales
    //     to full 32L Llama-2-7B per PROVER_PERFORMANCE.md (136 s prove
    //     at 32L on uncontested A100, FFN_SHARDS=16 ZK4_BASE=2).
    //   N=8 + ZK4_BASE=16: best for 1L (7.8 s vs 12.7 s default), but
    //     OOMs at ≥ 4L on contested GPUs because base=16's larger
    //     digit-plane buffers exceed available device memory.
    // Keep N=16 default for robustness; users benchmarking 1-2L should
    // override with `FFN_SHARDS=8 ZK4_BASE=16` (see §9 of perf doc).
    let ffn_shards: usize = std::env::var("FFN_SHARDS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("=== Llama-2-7B on Almost-Goldilocks ===");
    println!("num_layers={} seq_len={} max_num_vars={} (threads={})",
             num_layers, seq_len, max_num_vars, rayon::current_num_threads());
    println!("hidden_dim={} ffn_dim={} num_heads={} head_dim={} vocab={} logits_shards={} ffn_shards={}",
             hidden_dim, ffn_dim, num_heads, head_dim, vocab, logits_shards, ffn_shards);

    // ---- 1. Generate weights + build the DAG ----
    let t0 = Instant::now();
    let weights = gen_llama2_weights(num_layers, hidden_dim, ffn_dim, vocab, logits_shards, ffn_shards);
    println!("Weight gen: {:?}", t0.elapsed());

    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq_len, hidden_dim], DataType::Float);
    let _output = g.pipe(
        &[x],
        llama_2_7b(
            weights.0, weights.1, weights.2, weights.3, weights.4,
            weights.5, weights.6, weights.7, weights.8,
            weights.9, weights.10,
            num_heads, head_dim, seq_len,
        ),
    );
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        // Report the EFFECTIVE partition count: set_partition_boundaries
        // clamps to what the graph supports, so the requested value can
        // overstate it and would land in the CSV as such.
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // ---- 2. Forward pass on small-varied input ----
    let pad: usize = [1usize, seq_len, hidden_dim].iter()
        .map(|&s| s.next_power_of_two()).product();
    let input = Witness::new(
        vec![1, seq_len, hidden_dim],
        small_input(pad),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Forward: {:?}", t2.elapsed());

    // ---- 3. Commit (offline = constants; online = activations) ----
    // b = signed two's-complement bit width of committed values (the
    // bit-plane count). Env-tunable for the b-reduction feasibility probe:
    // smaller b = fewer leaves = faster fold tree, but only SOUND if the
    // committed values actually fit in b bits (else the decomposition wraps).
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    // base=2: safest across layer counts. base=16 cuts 1L prove ~40 %
    // (PROVER_PERFORMANCE.md §9) but its larger per-leaf digit-plane
    // buffers OOM the GPU at ≥ 4L on contested or 80 GB cards.
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let edge_parts_owned: Option<Vec<Option<usize>>> = if !dag.boundary_edges.is_empty() {
        let parts = zk_torch_4::dag::partition_dag(&dag, &dag.boundary_edges);
        Some(zk_torch_4::dag::edge_partition_map(&dag, &parts))
    } else {
        None
    };
    let t_off = Instant::now();
    match &edge_parts_owned {
        Some(ep) => dag.commit_constants_partitioned(&witnesses, &mut store, ep),
        None => dag.commit_constants(&witnesses, &mut store),
    }
    let offline_commit = t_off.elapsed();
    let t_on = Instant::now();
    match &edge_parts_owned {
        Some(ep) => dag.commit_remaining_partitioned(&witnesses, &mut store, ep),
        None => dag.commit_remaining(&witnesses, &mut store),
    }
    let online_commit = t_on.elapsed();
    println!("Commit (offline, amortized): {:?}", offline_commit);
    println!("Commit (online, prover time): {:?}", online_commit);

    // ---- 4. Prove ----
    let mut t_prove = Transcript::new(b"zkml-llama2");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    // ---- 5. Verify ----
    let mut t_verify = Transcript::new(b"zkml-llama2");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    let n_deferred = fold_proof.deferred_constant_claims.len();
    if n_deferred > 0 {
        println!(
            "\nVerified (modulo {} deferred constant claims — needs streaming finalize to be sound): {}",
            n_deferred, verified,
        );
    } else {
        // Serialized proof size, reported by the evaluation harness.
        let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
            + bincode::serialize(&fold_proof).unwrap().len();
        println!("Proof size: {} bytes", proof_bytes);

        println!("\nVerified: {}", verified);
    }
    if !verified {
        eprintln!("WARN: verifier rejected — see PROVER_PERFORMANCE.md for likely causes.");
    }
}
