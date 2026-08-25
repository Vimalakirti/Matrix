//! GPT-J-6B streaming-inference bench. Mirror of `bench_streaming_llama2` on
//! the `gpt_j_6b` DAG (RoPE, parallel attn+FFN, LayerNorm, un-tied lm_head), so
//! the two rows measure the SAME thing: each streamed proof is one forward pass
//! over SEQ_LEN tokens, weights are Role::Constant, deferred across the stream
//! and opened once at finalize.
//!
//! Deliberately NOT `bench_streaming_oneshot_gptj`, which already existed: that
//! streams full T-token AR GENERATIONS, a different workload that is not
//! comparable to the GPT-2 / BERT / Llama-2 rows in the same table.
//!
//! Env: NUM_LAYERS(1) SEQ_LEN(1) N_PROOFS(5) MAX_NUM_VARS(22) NUM_HEADS(16)
//! HEAD_DIM(256) FFN_DIM(4*hidden) VOCAB(50400) NUM_PARTITIONS(1) ZK4_B(21)
//! ZK4_BASE(2). Full vocab (4096 x 50400, padded to 2^28) needs MAX_NUM_VARS
//! raised or a smaller VOCAB.

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::gptj::gpt_j_6b;
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

/// Per-iteration input. GPT-J uses LayerNorm, which SUBTRACTS the mean, so a
/// constant row has zero variance and the norm is undefined -- the input must
/// vary along hidden. The 0/2048 alternation does that; `iter_offset` shifts
/// the phase so each streamed proof differs and the reducer exercises the real
/// fold rather than the degenerate identity case.
fn small_varied_input(size: usize, stride_last: usize, iter_offset: u64) -> Vec<AlmostGoldilocksField> {
    (0..size)
        .map(|i| {
            let phase = (i / stride_last.max(1)) as u64 + iter_offset;
            AlmostGoldilocksField(if phase % 2 == 0 { 0 } else { 2048 })
        })
        .collect()
}

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1000.0 }

fn zero_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    // Parallel/alloc_zeroed path; see zero_witness_vec.
    zk_torch_4::zero_witness_vec(size)
}

/// GPT-J subtracts the per-row mean before the RMS step, so we use the
/// exact-round 0/2048-alternating pattern along the LAST axis: after
/// mean subtraction every entry is ±1024 (= ±1.0 in float) and the
/// LN reciprocity gate `z ≈ sf` is satisfied exactly at the default
/// tolerance = 2. See the whisper bin for the same trick.

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

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_heads: usize = std::env::var("NUM_HEADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let head_dim: usize = std::env::var("HEAD_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let hidden_dim = num_heads * head_dim;
    let ffn_dim: usize = std::env::var("FFN_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(hidden_dim * 4);
    let vocab: usize = std::env::var("VOCAB").ok().and_then(|s| s.parse().ok()).unwrap_or(50400);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    println!("=== GPT-J-6B streaming-inference bench ===");
    println!("num_layers={} seq_len={} N_PROOFS={} max_num_vars={}", num_layers, seq_len, n_proofs, max_num_vars);
    println!("hidden_dim={} ffn_dim={} heads={} head_dim={} vocab={}",
             hidden_dim, ffn_dim, num_heads, head_dim, vocab);

    let t0 = Instant::now();
    let weights = gen_gptj_weights(num_layers, hidden_dim, ffn_dim, vocab);
    let weight_gen = t0.elapsed();

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
    let (mut dag, witnesses_template) = g.compile();
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
    let compile = t1.elapsed();

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);

    let t2 = Instant::now();
    dag.commit_constants(&witnesses_template, &mut store);
    let offline_commit = t2.elapsed();

    println!("Setup:");
    println!("  Weight gen      : {:>9.2}ms", ms(weight_gen));
    println!("  Compile         : {:>9.2}ms  ({} nodes, {} edges)",
             ms(compile), dag.nodes.len(), dag.num_edges());
    println!("  Offline commit  : {:>9.2}ms  (amortized; one-time per model)", ms(offline_commit));

    // ---- Streaming loop ----
    let label = b"zkml-gptj-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);

    let pad: usize = [1usize, seq_len, hidden_dim].iter().map(|&s| s.next_power_of_two()).product();

    let mut t_run = Duration::ZERO;
    let mut t_commit_online = Duration::ZERO;
    let mut t_prove = Duration::ZERO;
    let mut t_verify_per_proof = Duration::ZERO;
    let mut t_acc_update = Duration::ZERO;
    let mut t_acc_verify = Duration::ZERO;
    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} proofs:", n_proofs);
    for i in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let input = Witness::new(
            vec![1, seq_len, hidden_dim],
            small_varied_input(pad, hidden_dim.next_power_of_two(), i as u64),
            DataType::Float, *SF_LOG, Role::Input,
        );

        let mem_before = almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);

        let s0 = Instant::now();
        dag.run(&mut witnesses, &[(0, input)]);
        let d_run = s0.elapsed();
        t_run += d_run;
        let mem_after_run = almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);

        store.clear_non_constants(&witnesses);
        let mem_after_clear = almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);

        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed();
        t_commit_online += d_commit;
        let mem_after_commit = almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);

        let mut tp = Transcript::new(b"per-proof");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(
            &witnesses, &store, &mut tp, /*defer=*/ true,
        );
        let d_prove = s2.elapsed();
        t_prove += d_prove;
        let mem_after_prove = almost_goldilocks_cuda::mem_get_info().map(|(f, t)| (t - f) / (1024 * 1024)).unwrap_or(0);
        eprintln!(
            "  [iter {} mem-trace] start {} -> after run {} (+{}) -> after clear {} (+{}) -> after commit {} (+{}) -> after prove {} (+{})",
            i + 1, mem_before,
            mem_after_run, mem_after_run as i64 - mem_before as i64,
            mem_after_clear, mem_after_clear as i64 - mem_after_run as i64,
            mem_after_commit, mem_after_commit as i64 - mem_after_clear as i64,
            mem_after_prove, mem_after_prove as i64 - mem_after_commit as i64,
        );

        let mut tv = Transcript::new(b"per-proof");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(
            &witnesses, &store, &dp, &fp, &mut tv,
        );
        let d_verify = s3.elapsed();
        t_verify_per_proof += d_verify;
        if !r.ok {
            eprintln!("per-proof verify failed at iteration {}; aborting bench", i);
            return;
        }

        let s4 = Instant::now();
        let chunk = prover_acc.add_proof(&r, &witnesses);
        proof_bytes += ser_len(&dp) + ser_len(&fp) + ser_len(&chunk);
        // Keep one proof's component split; every iteration has the
        // same shape, so the last is representative.
        breakdown = Some(zk_torch_4::proof_size_report(
            &dp.node_proofs, &dp.edge_proofs, &dp.range_proof,
            &dp.two_pow_proof, &dp.output_claims, &dp, &fp));
        let d_acc_update = s4.elapsed();
        t_acc_update += d_acc_update;

        let s5 = Instant::now();
        let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
        let d_acc_verify = s5.elapsed();
        t_acc_verify += d_acc_verify;
        if !ok {
            eprintln!("streaming verifier rejected at iteration {}; aborting", i);
            return;
        }

        // Explicitly evict any device-resident witness buffers from
        // this iteration's witnesses before drop. Without this, the
        // Arc<DeviceBuffer> wrappers leak across iterations via
        // device-resident cached copies built up during dag.run /
        // commit_remaining / prove.
        for ws in witnesses.iter_mut() {
            for w in ws.iter_mut() {
                w.evict_device_buffer();
            }
        }
        // Trim the CUDA memory pool (best-effort — only affects
        // cudaMallocAsync-backed allocations; our code uses synchronous
        // cudaMalloc so this is mostly a no-op on the current cuda-rs).
        let _ = almost_goldilocks_cuda::pool_trim(0);
        let gpu_mem = almost_goldilocks_cuda::mem_get_info()
            .map(|(free, total)| (total - free) / (1024 * 1024))
            .unwrap_or(0);
        println!(
            "  [{:>2}/{}] run {:>8.2}ms  commit {:>8.2}ms  prove {:>8.2}ms  verify {:>7.2}ms  acc-update {:>8.2}ms  acc-verify {:>7.2}ms  gpu-mem {:>6} MiB",
            i + 1, n_proofs,
            ms(d_run), ms(d_commit),
            ms(d_prove), ms(d_verify),
            ms(d_acc_update), ms(d_acc_verify),
            gpu_mem,
        );
    }

    // ---- Finalize ----
    let n_steps_total = prover_acc.num_steps();
    let n_const_edges = prover_acc.num_edges();
    let s_fp = Instant::now();
    let final_proof = prover_acc.finalize(&witnesses_template, &store);
    let t_finalize = s_fp.elapsed();

    let s_fv = Instant::now();
    let ok = verifier_acc.verify_finalize(&store, &final_proof);
    let t_finalize_verify = s_fv.elapsed();
    if !ok {
        eprintln!("verify_finalize REJECTED — soundness chain broken");
        return;
    }

    let per_proof_total =
        t_run + t_commit_online + t_prove + t_verify_per_proof + t_acc_update + t_acc_verify;
    let n = n_proofs as f64;
    println!("\nStream summary (N = {} proofs):", n_proofs);
    println!("  Cumulative:");
    println!("    run                 : {:>9.2}ms", ms(t_run));
    println!("    commit-remaining    : {:>9.2}ms", ms(t_commit_online));
    println!("    prove (defer mode)  : {:>9.2}ms", ms(t_prove));
    println!("    verify (per-proof)  : {:>9.2}ms", ms(t_verify_per_proof));
    println!("    acc-update          : {:>9.2}ms", ms(t_acc_update));
    println!("    acc-verify          : {:>9.2}ms", ms(t_acc_verify));
    println!("    (per-proof total)   : {:>9.2}ms", ms(per_proof_total));
    println!("  Finalize (paid once at end of stream):");
    println!("    finalize prove      : {:>9.2}ms", ms(t_finalize));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("    finalize verify     : {:>9.2}ms", ms(t_finalize_verify));
    println!("  Amortized per-proof prover wall time (apples-to-apples vs monolithic prove):");
    println!("    prove (defer)               : {:>7.2}ms  per-proof avg",
             ms(t_prove) / n);
    println!("    + acc-update                : {:>7.2}ms  per-proof avg",
             ms(t_acc_update) / n);
    println!("    + finalize / N              : {:>7.2}ms",
             ms(t_finalize) / n);
    println!("    = streaming per-proof total : {:>7.2}ms",
             (ms(t_prove) + ms(t_acc_update) + ms(t_finalize)) / n);
    println!("    Compare to monolithic prove (env -ZK4_DEFER_CONSTANTS, same N)");
    println!("    to find streaming break-even.");
    println!(
        "\nVerified: {} ({} Constant edges × {} proofs = {} deferred claims → {} reducer steps → 1 fold-tree open over {} edges)",
        ok, n_const_edges, n_proofs, n_const_edges * n_proofs,
        n_steps_total, final_proof.edge_plane_evals.len(),
    );
}
