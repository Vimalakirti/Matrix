//! Whisper end-to-end prover binary. Ports zk-torch-3's `bin/whisper.rs`
//! to the zk-torch-4 commit/prove API.
//!
//! Defaults: tiny config (n_state=384, n_head=6, 4 encoder + 4 decoder
//! layers, n_mels=80, audio_ctx=1500, text_ctx=448). Override
//! `NUM_ENC_LAYERS`, `NUM_DEC_LAYERS`, `N_MELS`, `N_AUDIO_CTX`,
//! `N_TEXT_CTX`, `N_STATE`, `N_HEAD`, `MAX_NUM_VARS`, `ZK4_B`, `ZK4_BASE`
//! via env vars.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;
use rayon::prelude::*;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::whisper::{whisper_model, EncoderBlockWeights, DecoderBlockWeights};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::SF_LOG;

const PAR_THRESHOLD: usize = 1 << 16;

/// All weights/biases are zero in the smoke run so post-multiplication
/// einsum outputs (Σ over `n_state` or `mlp_dim` terms of two SF=10 floats)
/// don't overflow b=21 signed before ScaleDown brings them back. This
/// matches the gpt2/llama2 convention.
fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    // Parallel/alloc_zeroed path; see zero_witness_vec.
    zk_torch_4::zero_witness_vec(size)
}

/// Alternating 0 / 2 along the LAST tensor axis. See `make_pos_emb` for
/// the column-major rationale. For mel_data shape `[n_mels, 2·n_audio_ctx]`
/// the stride-of-last is `n_mels_pad`; for dec_data `[1, n_text_ctx, n_state]`
/// it's `n_text_ctx_pad`. The caller passes the matching `stride_last` and
/// the REAL (un-padded) extent of the last axis as `real_last`: indices at
/// or beyond it are the pow-2 padding region and MUST stay zero — the DAG's
/// LayerNorm sums (`bsi->bs` einsum) run over the padded axis but divide by
/// the real `n`, so pattern values leaked into the padding skew the mean by
/// `pad/n` (4/3 for n_state=384) and push the RMS reciprocity gate
/// `|r²·mean(x²) − sf| ≤ 2` out of tolerance.
fn small_input_vec(size: usize, stride_last: usize, real_last: usize) -> Vec<AlmostGoldilocksField> {
    (0..size)
        .map(|i| {
            let d = i / stride_last;
            AlmostGoldilocksField(if d < real_last && d % 2 == 1 { 2048 } else { 0 })
        })
        .collect()
}

fn make_weight(shape: Vec<usize>, padded_size: usize) -> Witness {
    Witness::new(shape, rand_field_vec(padded_size), DataType::Float, *SF_LOG, Role::Constant)
}

/// Like `make_weight` but with per-position variation (~1.0 in float).
/// Used for the encoder/decoder positional embedding so the transformer
/// blocks' LayerNorm input has non-zero variance even though every other
/// weight/bias is zero in the smoke run (the conv1d preamble outputs zero
/// because conv weight = 0).
fn make_pos_emb(shape: Vec<usize>, padded_size: usize) -> Witness {
    // Alternating 0 / 2 along the LAST tensor axis (= field 0 / 2048
    // at SF=10). The dag's column-major layout has the last dim with
    // the biggest stride: for shape [1, seq, dim] padded, flat index
    // `i` decomposes as `i = b + s*1 + d*stride_dim` with
    // `stride_dim = seq_pad`. Alternating on `(i / stride_dim) % 2`
    // varies along `d` (the LN-reduced axis) so x_minus_mean = ±1.0
    // in float, mean(x²) = 1 exactly, r = 1, z = sf, and the gate at
    // default `tolerance = 2` passes. For `seq_pad = 32` here, that's
    // `(i / 32) % 2`.
    //
    // The pow-2 padding tail of the last axis (d ≥ n_state, e.g.
    // 384..512) MUST stay zero: the LN mean sums the padded axis but
    // divides by the real `n`, so pattern values in the padding inflate
    // the mean by `pad/n`, make x_minus_mean asymmetric, and land
    // `r²·mean(x²)` outside the ±2 reciprocity tolerance (see
    // `small_input_vec`).
    let d_real = *shape.last().unwrap();
    let s_pad = shape.iter().rev().nth(1).copied().unwrap_or(1).next_power_of_two();
    let data: Vec<AlmostGoldilocksField> = (0..padded_size)
        .map(|i| {
            let d = i / s_pad;
            AlmostGoldilocksField(if d < d_real && d % 2 == 1 { 2048 } else { 0 })
        })
        .collect();
    Witness::new(shape, data, DataType::Float, *SF_LOG, Role::Constant)
}

fn make_bias_1d(shape: Vec<usize>, padded_size: usize) -> Witness {
    Witness::new(shape, rand_field_vec(padded_size), DataType::Float, *SF_LOG, Role::Constant)
}

fn make_uint_weight(shape: Vec<usize>, padded_size: usize) -> Witness {
    Witness::new(shape, rand_field_vec(padded_size), DataType::Uint, 0, Role::Constant)
}

/// Conv1D weight `W[C_out, C_in, K]` — all-zero in the smoke run.
fn make_conv1d_weight(c_out: usize, c_in: usize, kernel_size: usize) -> Witness {
    let c_out_pad = c_out.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let k_pad = kernel_size.next_power_of_two();
    let size = c_out_pad * c_in_pad * k_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![c_out, c_in, kernel_size], data, DataType::Uint, 0, Role::Constant)
}

fn make_encoder_block_weights(n_state: usize, mlp_dim: usize) -> EncoderBlockWeights {
    let ns_pad = n_state.next_power_of_two();
    let mlp_pad = mlp_dim.next_power_of_two();
    EncoderBlockWeights {
        attn_ln_w: make_bias_1d(vec![n_state], ns_pad),
        attn_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_q: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_k: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_v: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_o: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        b_q: make_bias_1d(vec![n_state], ns_pad),
        b_k: make_bias_1d(vec![n_state], ns_pad),
        b_v: make_bias_1d(vec![n_state], ns_pad),
        b_o: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_w: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_mlp1: make_weight(vec![n_state, mlp_dim], ns_pad * mlp_pad),
        w_mlp2: make_weight(vec![mlp_dim, n_state], mlp_pad * ns_pad),
        b_mlp1: make_bias_1d(vec![mlp_dim], mlp_pad),
        b_mlp2: make_bias_1d(vec![n_state], ns_pad),
    }
}

fn make_decoder_block_weights(n_state: usize, mlp_dim: usize) -> DecoderBlockWeights {
    let ns_pad = n_state.next_power_of_two();
    let mlp_pad = mlp_dim.next_power_of_two();
    DecoderBlockWeights {
        attn_ln_w: make_bias_1d(vec![n_state], ns_pad),
        attn_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_q: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_k: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_v: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_o: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        b_q: make_bias_1d(vec![n_state], ns_pad),
        b_k: make_bias_1d(vec![n_state], ns_pad),
        b_v: make_bias_1d(vec![n_state], ns_pad),
        b_o: make_bias_1d(vec![n_state], ns_pad),
        cross_ln_w: make_bias_1d(vec![n_state], ns_pad),
        cross_ln_b: make_bias_1d(vec![n_state], ns_pad),
        xw_q: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xw_k: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xw_v: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xw_o: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xb_q: make_bias_1d(vec![n_state], ns_pad),
        xb_k: make_bias_1d(vec![n_state], ns_pad),
        xb_v: make_bias_1d(vec![n_state], ns_pad),
        xb_o: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_w: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_mlp1: make_weight(vec![n_state, mlp_dim], ns_pad * mlp_pad),
        w_mlp2: make_weight(vec![mlp_dim, n_state], mlp_pad * ns_pad),
        b_mlp1: make_bias_1d(vec![mlp_dim], mlp_pad),
        b_mlp2: make_bias_1d(vec![n_state], ns_pad),
    }
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

    let num_enc_layers: usize = std::env::var("NUM_ENC_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let num_dec_layers: usize = std::env::var("NUM_DEC_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let n_mels: usize = std::env::var("N_MELS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(80);
    let n_audio_ctx: usize = std::env::var("N_AUDIO_CTX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1500);
    let n_text_ctx: usize = std::env::var("N_TEXT_CTX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(448);
    let n_state: usize = std::env::var("N_STATE").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(384);
    let n_head: usize = std::env::var("N_HEAD").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(6);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);

    let head_dim = n_state / n_head;
    let mlp_dim = 4 * n_state;

    let ns_pad = n_state.next_power_of_two();
    let n_mels_pad = n_mels.next_power_of_two();

    println!("=== Whisper on Almost-Goldilocks (enc={}, dec={}, n_state={}, n_head={}) ===",
        num_enc_layers, num_dec_layers, n_state, n_head);
    println!("n_mels={} n_audio_ctx={} n_text_ctx={} head_dim={} mlp_dim={}",
        n_mels, n_audio_ctx, n_text_ctx, head_dim, mlp_dim);
    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());

    let t0 = Instant::now();
    let mut g = DagBuilder::new();

    // Inputs
    let mel_input = g.input(vec![n_mels, 2 * n_audio_ctx], DataType::Float);
    let dec_input = g.input(vec![1, n_text_ctx, n_state], DataType::Float);

    // Conv weights
    let conv1_out_len = 2 * n_audio_ctx;
    let conv1_w = g.param(make_conv1d_weight(n_state, n_mels, 3));
    let conv1_bias = g.param(make_uint_weight(
        vec![n_state, conv1_out_len],
        ns_pad * conv1_out_len.next_power_of_two(),
    ));
    let conv2_w = g.param(make_conv1d_weight(n_state, n_state, 3));
    let conv2_bias = g.param(make_uint_weight(
        vec![n_state, n_audio_ctx],
        ns_pad * n_audio_ctx.next_power_of_two(),
    ));

    let enc_pos_emb = g.param(make_pos_emb(
        vec![1, n_audio_ctx, n_state],
        n_audio_ctx.next_power_of_two() * ns_pad,
    ));
    let dec_pos_emb = g.param(make_pos_emb(
        vec![1, n_text_ctx, n_state],
        n_text_ctx.next_power_of_two() * ns_pad,
    ));

    let enc_blocks: Vec<EncoderBlockWeights> = (0..num_enc_layers)
        .into_par_iter()
        .map(|_| make_encoder_block_weights(n_state, mlp_dim))
        .collect();
    let enc_final_ln_w = make_bias_1d(vec![n_state], ns_pad);
    let enc_final_ln_b = make_bias_1d(vec![n_state], ns_pad);

    let dec_blocks: Vec<DecoderBlockWeights> = (0..num_dec_layers)
        .into_par_iter()
        .map(|_| make_decoder_block_weights(n_state, mlp_dim))
        .collect();
    let dec_final_ln_w = make_bias_1d(vec![n_state], ns_pad);
    let dec_final_ln_b = make_bias_1d(vec![n_state], ns_pad);
    println!("Weight gen: {:?}", t0.elapsed());

    let _ = whisper_model(
        &mut g,
        mel_input, dec_input,
        conv1_w, conv1_bias, conv2_w, conv2_bias, enc_pos_emb,
        enc_blocks, enc_final_ln_w, enc_final_ln_b,
        dec_pos_emb, dec_blocks, dec_final_ln_w, dec_final_ln_b,
        n_head, head_dim, n_state, n_audio_ctx, n_text_ctx,
    );

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // For column-major layout, stride-of-last = product of leading
    // padded dims. mel_data [n_mels, 2*n_audio_ctx] → stride = n_mels_pad.
    // dec_data [1, n_text_ctx, n_state] → stride = 1 * n_text_ctx_pad.
    let mel_data = Witness::new(
        vec![n_mels, 2 * n_audio_ctx],
        small_input_vec(
            n_mels_pad * (2 * n_audio_ctx).next_power_of_two(),
            n_mels_pad,
            2 * n_audio_ctx,
        ),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let dec_data = Witness::new(
        vec![1, n_text_ctx, n_state],
        small_input_vec(
            n_text_ctx.next_power_of_two() * ns_pad,
            n_text_ctx.next_power_of_two(),
            n_state,
        ),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, mel_data), (1, dec_data)]);
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

    let mut t_prove = Transcript::new(b"zkml-whisper");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-whisper");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
}
