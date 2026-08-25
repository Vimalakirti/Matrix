//! Streaming Whisper (encoder + decoder with cross-attention) composed with
//! the cross-proof streaming accumulator. This is the strongest test of the
//! accumulator on a non-trivial graph: the decoder's cross-attention reuses
//! the encoder K/V, the case the old "reducer mixed-arity" note flagged.
//! Each streamed proof is one (audio, text) pair; all WEIGHTS are
//! Role::Constant (deferred → amortized into one finalize opening); the two
//! INPUTS (mel + dec) are Role::Input, committed/opened per-proof.
//!
//! Run with `bench_config.yaml` as args[1]. Env: NUM_ENC_LAYERS(1)
//! NUM_DEC_LAYERS(1) N_MELS(80) N_AUDIO_CTX(32) N_TEXT_CTX(16) N_STATE(128)
//! N_HEAD(2) N_PROOFS(3) MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rayon::prelude::*;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::whisper::{whisper_model, EncoderBlockWeights, DecoderBlockWeights};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }
fn zerov(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// Per-proof input: 0/2048 along the last axis (LayerNorm-safe), phase-shifted
/// by `it` so each proof commits a distinct (still-valid) input.
fn input_vec(size: usize, stride_last: usize, real_last: usize, it: usize) -> Vec<AlmostGoldilocksField> {
    (0..size)
        .map(|i| {
            let d = i / stride_last;
            // Padding tail of the last axis MUST stay zero (LN mean contract).
            AlmostGoldilocksField(if d < real_last && (d + it) % 2 == 1 { 2048 } else { 0 })
        })
        .collect()
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
        })
        .collect();
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
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let head_dim = n_state / n_head;
    let mlp_dim = 4 * n_state;
    let ns_pad = n_state.next_power_of_two();
    let n_mels_pad = n_mels.next_power_of_two();

    println!("=== Streaming Whisper (enc={}, dec={}, n_state={}, n_head={}) ===",
        num_enc_layers, num_dec_layers, n_state, n_head);
    println!("n_mels={} n_audio_ctx={} n_text_ctx={} N_PROOFS={} partitions={}",
        n_mels, n_audio_ctx, n_text_ctx, n_proofs, num_partitions);

    let mut g = DagBuilder::new();
    let mel_input = g.input(vec![n_mels, 2 * n_audio_ctx], DataType::Float);
    let dec_input = g.input(vec![1, n_text_ctx, n_state], DataType::Float);
    let conv1_out_len = 2 * n_audio_ctx;
    let conv1_w = g.param(make_conv1d_weight(n_state, n_mels, 3));
    let conv1_bias = g.param(uw(vec![n_state, conv1_out_len], ns_pad * conv1_out_len.next_power_of_two()));
    let conv2_w = g.param(make_conv1d_weight(n_state, n_state, 3));
    let conv2_bias = g.param(uw(vec![n_state, n_audio_ctx], ns_pad * n_audio_ctx.next_power_of_two()));
    let enc_pos_emb = g.param(make_pos_emb(vec![1, n_audio_ctx, n_state], n_audio_ctx.next_power_of_two() * ns_pad));
    let dec_pos_emb = g.param(make_pos_emb(vec![1, n_text_ctx, n_state], n_text_ctx.next_power_of_two() * ns_pad));
    let enc_blocks: Vec<EncoderBlockWeights> = (0..num_enc_layers).into_par_iter().map(|_| enc_block(n_state, mlp_dim)).collect();
    let enc_final_ln_w = b1(vec![n_state], ns_pad);
    let enc_final_ln_b = b1(vec![n_state], ns_pad);
    let dec_blocks: Vec<DecoderBlockWeights> = (0..num_dec_layers).into_par_iter().map(|_| dec_block(n_state, mlp_dim)).collect();
    let dec_final_ln_w = b1(vec![n_state], ns_pad);
    let dec_final_ln_b = b1(vec![n_state], ns_pad);

    let _ = whisper_model(
        &mut g, mel_input, dec_input,
        conv1_w, conv1_bias, conv2_w, conv2_bias, enc_pos_emb,
        enc_blocks, enc_final_ln_w, enc_final_ln_b,
        dec_pos_emb, dec_blocks, dec_final_ln_w, dec_final_ln_b,
        n_head, head_dim, n_state, n_audio_ctx, n_text_ctx,
    );
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    assert_eq!(witnesses_template[mel_input][0].role, Role::Input, "mel input must be Role::Input");
    assert_eq!(witnesses_template[dec_input][0].role, Role::Input, "dec input must be Role::Input");

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        // Report the EFFECTIVE partition count: set_partition_boundaries
        // clamps to what the graph supports, so the requested value can
        // overstate it and would land in the CSV as such.
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses_template, &mut store);
    println!("Offline commit (weights, amortized): {:.2}ms", ms(t_off.elapsed()));

    let mel_size = n_mels_pad * (2 * n_audio_ctx).next_power_of_two();
    let mel_stride = n_mels_pad;
    let dec_size = n_text_ctx.next_power_of_two() * ns_pad;
    let dec_stride = n_text_ctx.next_power_of_two();

    let label = b"zkml-whisper-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut checked_role = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} (audio,text) inferences:", n_proofs);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        // mel may vary per-proof (conv weights are 0 → encoder output is
        // bias+pos_emb regardless, so any valid mel is harmless) → distinct
        // committed inputs. dec MUST stay aligned with the fixed additive
        // dec_pos_emb: a phase-shifted (inverted 0/2048) dec cancels pos_emb
        // to a constant → zero LayerNorm variance → range check rejects.
        let mel = Witness::new(vec![n_mels, 2 * n_audio_ctx], input_vec(mel_size, mel_stride, 2 * n_audio_ctx, it), DataType::Float, *SF_LOG, Role::Input);
        let dec = Witness::new(vec![1, n_text_ctx, n_state], input_vec(dec_size, dec_stride, n_state, 0), DataType::Float, *SF_LOG, Role::Input);

        let s0 = Instant::now();
        dag.run(&mut witnesses, &[(mel_input, mel), (dec_input, dec)]);
        let d_run = s0.elapsed(); t_run += d_run;

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-inf");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-inf");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-inference verify failed at {}", it); return; }

        if !checked_role {
            for dc in &r.claims {
                assert!(dc.edge_id != mel_input && dc.edge_id != dec_input,
                    "input edge {} was deferred as a shared weight — unsound", dc.edge_id);
            }
            checked_role = true;
        }

        let s4 = Instant::now();
        let chunk = prover_acc.add_proof(&r, &witnesses);
        let d_acc = s4.elapsed(); t_acc += d_acc;
        proof_bytes += ser_len(&dp) + ser_len(&fp) + ser_len(&chunk);
        // Keep one proof's component split; every iteration has the
        // same shape, so the last is representative.
        breakdown = Some(zk_torch_4::proof_size_report(
            &dp.node_proofs, &dp.edge_proofs, &dp.range_proof,
            &dp.two_pow_proof, &dp.output_claims, &dp, &fp));
        let s5 = Instant::now();
        let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
        let d_accv = s5.elapsed(); t_accv += d_accv;
        if !ok { eprintln!("streaming verifier rejected at {}", it); return; }

        println!("  [{:>2}/{}] run {:>7.1}ms commit {:>6.1}ms prove {:>7.1}ms verify {:>6.1}ms acc {:>7.1}ms acc-v {:>6.1}ms",
            it + 1, n_proofs, ms(d_run), ms(d_commit), ms(d_prove), ms(d_verify), ms(d_acc), ms(d_accv));
    }

    let n_steps = prover_acc.num_steps();
    let n_const = prover_acc.num_edges();
    let s_fp = Instant::now();
    let final_proof = prover_acc.finalize(&witnesses_template, &store);
    let t_finalize = s_fp.elapsed();
    let s_fv = Instant::now();
    let ok = verifier_acc.verify_finalize(&store, &final_proof);
    let t_fv = s_fv.elapsed();
    if !ok { eprintln!("verify_finalize REJECTED — soundness chain broken"); return; }

    let n = n_proofs as f64;
    println!("\n=== Results ({} weight edges deferred, {} reducer steps) ===", n_const, n_steps);
    println!("  prove(defer)  per-inf : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-inf : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (Whisper enc+dec, weights amortized across {} inferences)", n_proofs);
}
