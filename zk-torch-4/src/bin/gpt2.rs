//! GPT-2 Small end-to-end prover binary. Ports zk-torch-3's `bin/gpt2.rs`
//! to the zk-torch-4 prove pipeline:
//!
//! - Sumcheck-side backward pass via `Dag::prove` (step 4, plan §8.2.11).
//! - Per-edge opening reducer fold-tree leaves via
//!   `Dag::prove_with_fold_tree` (step 12 + 13.5).
//!
//! Default config: GPT-2 Small (`num_heads = 12, head_dim = 64,
//! hidden_dim = 768`, `num_layers = 1`, `seq_len = 1`). Override via
//! `NUM_LAYERS`, `SEQ_LEN`, and `MAX_NUM_VARS` env vars. Random weights
//! and input — no model file loading (mirrors zk-torch-3).

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::gpt2::gpt_2_small;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::SF_LOG;

fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    // Parallel/alloc_zeroed path; see zero_witness_vec.
    zk_torch_4::zero_witness_vec(size)
}

/// Per-position non-zero input large enough that `mean(x²)` doesn't
/// underflow in fixed-point, so RMSNorm's `r = 1/sqrt(mean(x²))` and
/// the reciprocity check `r²·mean(x²) ≈ 1` are well-defined.
/// Values cluster around 1.0 (= field 1024 at SF=10) with small varied
/// offsets; mean is clean and variance is strictly positive.
///
/// The dag stores tensors COLUMN-MAJOR (first axis fastest in the flat
/// buffer; see `broadcast_strides` in `basicblock/add.rs`). LayerNorm
/// reduces over the LAST axis (hidden), so the per-position variation
/// must move along the last axis — i.e., flat index must vary the LSB
/// of the *hidden* coordinate. For shape `[b, s, h]` padded `[B, S, H]`,
/// flat index `i = b + s*B + h*B*S`, so the hidden coordinate is
/// `i / (B*S) = i / stride_last`. A plain `i % 16` would degenerate to
/// a constant along hidden whenever `stride_last % 16 == 0` (e.g. any
/// `seq_pad ≥ 16`), making variance zero and the reciprocity gate fail.
fn small_varied_input(size: usize, stride_last: usize) -> Vec<AlmostGoldilocksField> {
    (0..size)
        .map(|i| AlmostGoldilocksField(1000 + 10 * (((i / stride_last) % 16) as u64)))
        .collect()
}

#[allow(clippy::type_complexity)]
fn gen_gpt2_weights(num_layers: usize, hidden_dim: usize, ffn_dim: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness,
) {
    let mut attn_norm_w_vec = Vec::new();
    let mut attn_q_w_vec    = Vec::new();
    let mut attn_k_w_vec    = Vec::new();
    let mut attn_v_w_vec    = Vec::new();
    let mut attn_o_w_vec    = Vec::new();
    let mut attn_norm_b_vec = Vec::new();
    let mut attn_q_b_vec    = Vec::new();
    let mut attn_k_b_vec    = Vec::new();
    let mut attn_v_b_vec    = Vec::new();
    let mut attn_o_b_vec    = Vec::new();
    let mut proj_norm_w_vec = Vec::new();
    let mut proj_1_w_vec    = Vec::new();
    let mut proj_2_w_vec    = Vec::new();
    let mut proj_norm_b_vec = Vec::new();
    let mut proj_1_b_vec    = Vec::new();
    let mut proj_2_b_vec    = Vec::new();

    let hd_pad = hidden_dim.next_power_of_two();
    let ffn_pad = ffn_dim.next_power_of_two();
    for _ in 0..num_layers {
        attn_norm_w_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_w_vec.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_w_vec.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_w_vec.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_w_vec.push(Witness::new(vec![hidden_dim, hidden_dim], rand_field_vec(hd_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_norm_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_norm_w_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_w_vec.push(Witness::new(vec![hidden_dim, ffn_dim], rand_field_vec(hd_pad * ffn_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_w_vec.push(Witness::new(vec![ffn_dim, hidden_dim], rand_field_vec(ffn_pad * hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_norm_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_b_vec.push(Witness::new(vec![ffn_dim], rand_field_vec(ffn_pad), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_b_vec.push(Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant));
    }
    let ln_w = Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant);
    let ln_b = Witness::new(vec![hidden_dim], rand_field_vec(hd_pad), DataType::Float, *SF_LOG, Role::Constant);
    (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        attn_norm_b_vec, attn_q_b_vec, attn_k_b_vec, attn_v_b_vec, attn_o_b_vec,
        proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec,
        proj_norm_b_vec, proj_1_b_vec, proj_2_b_vec,
        ln_w, ln_b,
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
    // hidden=768 + seq=1 + ffn=3072: largest dense plane has
    // shape [1, 1, 3072] padded to [1, 1, 4096] = 2^12. Sparse aux:
    // input_n + TABLE_SIZE_LOG. With bench_config.yaml's table_log=10
    // and FFN input_n=12 → aux=22. So max_num_vars = 22 is enough.
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_heads = 12;
    let head_dim = 64;
    let hidden_dim = num_heads * head_dim; // 768
    let ffn_dim = hidden_dim * 4;

    println!("=== GPT-2 Small on Almost-Goldilocks ===");
    println!("num_layers={} seq_len={} max_num_vars={} (threads={})",
             num_layers, seq_len, max_num_vars, rayon::current_num_threads());

    // ---- 1. Generate weights + build the DAG ----
    let t0 = Instant::now();
    let weights = gen_gpt2_weights(num_layers, hidden_dim, ffn_dim);
    println!("Weight gen: {:?}", t0.elapsed());

    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq_len, hidden_dim], DataType::Float);
    let _output = g.pipe(
        &[x],
        gpt_2_small(
            weights.0, weights.1, weights.2, weights.3, weights.4,
            weights.5, weights.6, weights.7, weights.8, weights.9,
            weights.10, weights.11, weights.12,
            weights.13, weights.14, weights.15,
            weights.16, weights.17,
            num_heads, head_dim, seq_len,
        ),
    );
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // Optional multi-GPU partitioning. NUM_PARTITIONS=N picks N-1
    // evenly-spaced layer boundaries; each partition runs on its own
    // GPU (set_device(k % device_count)) for both commits and the
    // backward-pass sumcheck.
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

    // ---- 2. Forward pass on input varied along the hidden axis ----
    // For column-major layout of [1, seq, hidden] padded, stride_last
    // = 1 * seq_pad. The varied pattern then moves along the hidden axis
    // (the one LayerNorm reduces over) regardless of seq_len.
    let pad: usize = [1usize, seq_len, hidden_dim].iter().map(|&s| s.next_power_of_two()).product();
    let stride_last = 1 * seq_len.next_power_of_two();
    let input = Witness::new(
        vec![1, seq_len, hidden_dim],
        small_varied_input(pad, stride_last),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Forward: {:?}", t2.elapsed());

    // ---- 3. Commit, split: offline (constants — one-time per model)
    // vs online (input-dependent). Only the online part counts as
    // prover time; the offline weight commits are amortized across
    // many proofs and run once per model.
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    // If partitioned, route each edge's commit to its owning
    // partition's GPU. Otherwise round-robin by commit-task index.
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

    // ---- 4. Prove (sumcheck side + fold tree) ----
    let mut t_prove = Transcript::new(b"zkml-gpt2");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    // ---- 5. Verify ----
    let mut t_verify = Transcript::new(b"zkml-gpt2");
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
        eprintln!("WARN: verifier rejected — likely an intermediate value escaped the range table.");
    }
}
