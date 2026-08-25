//! One-shot autoregressive Whisper decoder prover.
//!
//! Proves a full T-token text generation in ONE proof: audio mel → encoder
//! (cross-attention context) and dec tokens → embedding_lookup → Whisper
//! decoder (causal self-attn + cross-attn to the encoder, seq=T) →
//! lm_head(W_LM) → argmax_check. The encoder runs once on the audio; the
//! decoder's causal self-attention makes a single full-sequence forward
//! reproduce every AR step (SKIP-AR style — the public shift constraint is
//! orchestrated outside the circuit, like oneshot_gpt2).
//!
//! Decoder uses LayerNorm; `dec_pos_emb` is set to ZERO here so the
//! (high-variance 0/2048) token embedding alone feeds the first decoder
//! LayerNorm (a fixed nonzero pos_emb could cancel the embedding to a
//! constant → zero variance → range reject; see bench_streaming_whisper).
//! Two-pass prover: dummy argmax selector → read logits → set true per-row
//! argmax → rerun → commit → prove → verify.
//!
//! Run with `bench_config.yaml` as args[1]. One-shot lm_head is un-sharded →
//! keep VOCAB modest. Env: NUM_ENC_LAYERS(1) NUM_DEC_LAYERS(1) N_MELS(80)
//! N_AUDIO_CTX(32) N_TEXT_CTX(16) N_STATE(128) N_HEAD(2) VOCAB(256)
//! MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::whisper::{whisper_model, EncoderBlockWeights, DecoderBlockWeights};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::util::arith::f_to_int;
use zk_torch_4::SF_LOG;

fn zerov(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// Audio/mel input: 0/2048 along the last axis (LayerNorm-safe; conv weights
/// are 0 so the encoder output is bias+pos_emb regardless — any valid pattern).
fn input_vec(size: usize, stride_last: usize) -> Vec<AlmostGoldilocksField> {
    (0..size).map(|i| AlmostGoldilocksField(if (i / stride_last) % 2 == 0 { 0 } else { 2048 })).collect()
}

/// Decoder token embedding `(vocab, n_state)` — 0/2048 alternating along the
/// n_state axis (col-major (v,d) at v + d·vocab_pad), full padded width. The
/// decoder LayerNorm reduces over n_state → variance 1, valid.
fn embedding_matrix(vocab: usize, n_state: usize) -> Vec<AlmostGoldilocksField> {
    let vp = vocab.next_power_of_two();
    let sp = n_state.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(vp * sp);
    for d in 0..sp {
        let val = if d % 2 == 0 { AlmostGoldilocksField(0) } else { AlmostGoldilocksField(2048) };
        for v in 0..vocab { data[v + d * vp] = val; }
    }
    data
}

fn w(shape: Vec<usize>, n: usize) -> Witness { Witness::new(shape, zerov(n), DataType::Float, *SF_LOG, Role::Constant) }
fn b1(shape: Vec<usize>, n: usize) -> Witness { Witness::new(shape, zerov(n), DataType::Float, *SF_LOG, Role::Constant) }
fn uw(shape: Vec<usize>, n: usize) -> Witness { Witness::new(shape, zerov(n), DataType::Uint, 0, Role::Constant) }

fn make_pos_emb(shape: Vec<usize>, padded_size: usize) -> Witness {
    let s_pad = shape.iter().rev().nth(1).copied().unwrap_or(1).next_power_of_two();
    // The pow-2 padding tail of the last axis MUST stay zero: the LN mean
    // sums the padded axis but divides by the real extent (see whisper.rs).
    let d_real = shape.last().copied().unwrap_or(1);
    let data: Vec<AlmostGoldilocksField> = (0..padded_size)
        .map(|i| {
            let d = i / s_pad;
            AlmostGoldilocksField(if d < d_real && d % 2 == 1 { 2048 } else { 0 })
        }).collect();
    Witness::new(shape, data, DataType::Float, *SF_LOG, Role::Constant)
}

fn make_conv1d_weight(c_out: usize, c_in: usize, k: usize) -> Witness {
    let n = c_out.next_power_of_two() * c_in.next_power_of_two() * k.next_power_of_two();
    Witness::new(vec![c_out, c_in, k], zerov(n), DataType::Uint, 0, Role::Constant)
}

fn enc_block(n_state: usize, mlp_dim: usize) -> EncoderBlockWeights {
    let (ns, mp) = (n_state.next_power_of_two(), mlp_dim.next_power_of_two());
    EncoderBlockWeights {
        attn_ln_w: b1(vec![n_state], ns), attn_ln_b: b1(vec![n_state], ns),
        w_q: w(vec![n_state, n_state], ns * ns), w_k: w(vec![n_state, n_state], ns * ns),
        w_v: w(vec![n_state, n_state], ns * ns), w_o: w(vec![n_state, n_state], ns * ns),
        b_q: b1(vec![n_state], ns), b_k: b1(vec![n_state], ns), b_v: b1(vec![n_state], ns), b_o: b1(vec![n_state], ns),
        mlp_ln_w: b1(vec![n_state], ns), mlp_ln_b: b1(vec![n_state], ns),
        w_mlp1: w(vec![n_state, mlp_dim], ns * mp), w_mlp2: w(vec![mlp_dim, n_state], mp * ns),
        b_mlp1: b1(vec![mlp_dim], mp), b_mlp2: b1(vec![n_state], ns),
    }
}

fn dec_block(n_state: usize, mlp_dim: usize) -> DecoderBlockWeights {
    let (ns, mp) = (n_state.next_power_of_two(), mlp_dim.next_power_of_two());
    DecoderBlockWeights {
        attn_ln_w: b1(vec![n_state], ns), attn_ln_b: b1(vec![n_state], ns),
        w_q: w(vec![n_state, n_state], ns * ns), w_k: w(vec![n_state, n_state], ns * ns),
        w_v: w(vec![n_state, n_state], ns * ns), w_o: w(vec![n_state, n_state], ns * ns),
        b_q: b1(vec![n_state], ns), b_k: b1(vec![n_state], ns), b_v: b1(vec![n_state], ns), b_o: b1(vec![n_state], ns),
        cross_ln_w: b1(vec![n_state], ns), cross_ln_b: b1(vec![n_state], ns),
        xw_q: w(vec![n_state, n_state], ns * ns), xw_k: w(vec![n_state, n_state], ns * ns),
        xw_v: w(vec![n_state, n_state], ns * ns), xw_o: w(vec![n_state, n_state], ns * ns),
        xb_q: b1(vec![n_state], ns), xb_k: b1(vec![n_state], ns), xb_v: b1(vec![n_state], ns), xb_o: b1(vec![n_state], ns),
        mlp_ln_w: b1(vec![n_state], ns), mlp_ln_b: b1(vec![n_state], ns),
        w_mlp1: w(vec![n_state, mlp_dim], ns * mp), w_mlp2: w(vec![mlp_dim, n_state], mp * ns),
        b_mlp1: b1(vec![mlp_dim], mp), b_mlp2: b1(vec![n_state], ns),
    }
}

fn demo_seed() -> Seed {
    Seed([0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
          0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE])
}

fn argmax_row(logits: &Witness, pos: usize, vocab: usize) -> usize {
    let mut best = 0usize; let mut bv = i128::MIN;
    for v in 0..vocab {
        let val = f_to_int(logits.get(&[pos, v]));
        if val > bv { bv = val; best = v; }
    }
    best
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_enc_layers: usize = std::env::var("NUM_ENC_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let num_dec_layers: usize = std::env::var("NUM_DEC_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_mels: usize = std::env::var("N_MELS").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let n_audio_ctx: usize = std::env::var("N_AUDIO_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let n_text_ctx: usize = std::env::var("N_TEXT_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_state: usize = std::env::var("N_STATE").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let n_head: usize = std::env::var("N_HEAD").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let vocab: usize = std::env::var("VOCAB").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let bb: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let head_dim = n_state / n_head;
    let mlp_dim = 4 * n_state;
    let ns_pad = n_state.next_power_of_two();
    let n_mels_pad = n_mels.next_power_of_two();

    println!("=== One-Shot AR Whisper decoder (enc={}, dec={}, n_state={}) ===",
        num_enc_layers, num_dec_layers, n_state);
    println!("n_audio_ctx={} n_text_ctx(gen len)={} vocab={} partitions={}",
        n_audio_ctx, n_text_ctx, vocab, num_partitions);

    let mut rng = StdRng::seed_from_u64(42);
    let token_ids: Vec<usize> = (0..n_text_ctx).map(|_| rng.gen::<usize>() % vocab).collect();

    let t0 = Instant::now();
    let mut g = DagBuilder::new();
    // Audio input (encoder), Role::Input.
    let mel_input = g.input(vec![n_mels, 2 * n_audio_ctx], DataType::Float);
    // Decoder token embedding → dec_input hidden.
    let w_e = g.param(Witness::new(vec![vocab, n_state], embedding_matrix(vocab, n_state),
        DataType::Float, *SF_LOG, Role::Constant));
    let (dec_emb, _emb_sel) = g.embedding_lookup(w_e, n_text_ctx, vocab, &token_ids, Role::Constant);
    let dec_in = g.change_shape(dec_emb, vec![1, n_text_ctx, n_state]);

    let conv1_out_len = 2 * n_audio_ctx;
    let conv1_w = g.param(make_conv1d_weight(n_state, n_mels, 3));
    let conv1_bias = g.param(uw(vec![n_state, conv1_out_len], ns_pad * conv1_out_len.next_power_of_two()));
    let conv2_w = g.param(make_conv1d_weight(n_state, n_state, 3));
    let conv2_bias = g.param(uw(vec![n_state, n_audio_ctx], ns_pad * n_audio_ctx.next_power_of_two()));
    let enc_pos_emb = g.param(make_pos_emb(vec![1, n_audio_ctx, n_state], n_audio_ctx.next_power_of_two() * ns_pad));
    // dec_pos_emb = ZERO (embedding alone feeds the decoder LayerNorm).
    let dec_pos_emb = g.param(Witness::new(vec![1, n_text_ctx, n_state],
        zerov(n_text_ctx.next_power_of_two() * ns_pad), DataType::Float, *SF_LOG, Role::Constant));
    let enc_blocks: Vec<EncoderBlockWeights> = (0..num_enc_layers).into_par_iter().map(|_| enc_block(n_state, mlp_dim)).collect();
    let dec_blocks: Vec<DecoderBlockWeights> = (0..num_dec_layers).into_par_iter().map(|_| dec_block(n_state, mlp_dim)).collect();

    let dec_hidden = whisper_model(
        &mut g, mel_input, dec_in,
        conv1_w, conv1_bias, conv2_w, conv2_bias, enc_pos_emb,
        enc_blocks, b1(vec![n_state], ns_pad), b1(vec![n_state], ns_pad),
        dec_pos_emb, dec_blocks, b1(vec![n_state], ns_pad), b1(vec![n_state], ns_pad),
        n_head, head_dim, n_state, n_audio_ctx, n_text_ctx,
    );

    // lm_head + argmax_check (un-tied W_LM).
    let w_lm = g.param(Witness::new(vec![vocab, n_state], embedding_matrix(vocab, n_state),
        DataType::Float, *SF_LOG, Role::Constant));
    let logits = g.lm_head(dec_hidden, w_lm, n_text_ctx, vocab);
    let dummy = vec![0usize; n_text_ctx];
    let argmax_sel = g.argmax_check(logits, n_text_ctx, vocab, &dummy, Role::Constant);
    let (mut dag, mut witnesses) = g.compile();
    println!("Build+compile: {:?}  ({} nodes, {} edges)", t0.elapsed(), dag.nodes.len(), dag.num_edges());

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        // Report the EFFECTIVE partition count: set_partition_boundaries
        // clamps to what the graph supports, so the requested value can
        // overstate it and would land in the CSV as such.
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }

    // mel forward input (Role::Input), set via override.
    let mel_size = n_mels_pad * (2 * n_audio_ctx).next_power_of_two();
    let mel = Witness::new(vec![n_mels, 2 * n_audio_ctx], input_vec(mel_size, n_mels_pad), DataType::Float, *SF_LOG, Role::Input);
    let t1 = Instant::now();
    dag.run(&mut witnesses, &[(mel_input, mel)]);
    println!("Forward (pass 1): {:?}", t1.elapsed());

    let next_tokens: Vec<usize> = (0..n_text_ctx).map(|i| argmax_row(&witnesses[logits][0], i, vocab)).collect();
    witnesses[argmax_sel] = vec![DagBuilder::build_one_hot_selector_witness(n_text_ctx, vocab, &next_tokens, Role::Constant)];
    let mel2 = Witness::new(vec![n_mels, 2 * n_audio_ctx], input_vec(mel_size, n_mels_pad), DataType::Float, *SF_LOG, Role::Input);
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(mel_input, mel2)]);
    println!("Forward (pass 2): {:?}", t2.elapsed());

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, bb, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let edge_parts: Option<Vec<Option<usize>>> = if !dag.boundary_edges.is_empty() {
        let parts = zk_torch_4::dag::partition_dag(&dag, &dag.boundary_edges);
        Some(zk_torch_4::dag::edge_partition_map(&dag, &parts))
    } else { None };
    let t_off = Instant::now();
    match &edge_parts {
        Some(ep) => dag.commit_constants_partitioned(&witnesses, &mut store, ep),
        None => dag.commit_constants(&witnesses, &mut store),
    }
    println!("Commit (offline): {:?}", t_off.elapsed());
    let t_on = Instant::now();
    match &edge_parts {
        Some(ep) => dag.commit_remaining_partitioned(&witnesses, &mut store, ep),
        None => dag.commit_remaining(&witnesses, &mut store),
    }
    println!("Commit (online): {:?}", t_on.elapsed());

    let mut tp = Transcript::new(b"oneshot-whisper");
    let t4 = Instant::now();
    let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut tp);
    println!("Prove: {:?}", t4.elapsed());
    let mut tv = Transcript::new(b"oneshot-whisper");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut tv);
    println!("Verify: {:?}", t5.elapsed());
    println!("\nVerified: {}", verified);
    assert!(verified, "one-shot Whisper proof failed to verify");
}
