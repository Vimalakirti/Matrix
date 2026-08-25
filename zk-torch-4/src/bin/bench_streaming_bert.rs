//! Streaming BERT (encoder, seq_len>1) composed with the cross-proof
//! streaming accumulator. BERT is NOT autoregressive — it processes the full
//! sequence in one forward, so "one-shot / seq>1" is native (no embedding
//! lookup / argmax / two-pass like the decoder bins). Across the N inferences
//! the WEIGHTS (bert_large params) are Role::Constant — deferred per-proof and
//! amortized into ONE finalize fold-tree opening — while the per-inference
//! INPUT (hidden state [1,seq,hidden]) is Role::Input, committed/opened
//! per-proof and never deferred.
//!
//! This closes out the transformer set for streaming (encoder, no AR). Run
//! with `bench_config.yaml` as args[1]. Env: NUM_LAYERS(1) SEQ_LEN(8)
//! N_PROOFS(5) MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::bert::bert_large;
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }
fn zeros(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// High-variance LayerNorm-safe input (0/2048 alternating), with a per-proof
/// phase shift `it` so each inference commits a distinct input while staying
/// valid (still 0/2048 half-half → variance 1).
fn varied_input(size: usize, stride_last: usize, it: usize) -> Vec<AlmostGoldilocksField> {
    (0..size)
        .map(|i| AlmostGoldilocksField(if (i / stride_last + it) % 2 == 0 { 0 } else { 2048 }))
        .collect()
}

/// LN weight = 1.0 (field 1024 at SF=10) for the first `dim` entries (zero
/// padding) — preserves variance through the LN chain (see bert bin).
fn unit_norm_weight(dim: usize) -> Witness {
    let dim_pad = dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(dim_pad);
    for i in 0..dim { data[i] = AlmostGoldilocksField(1024); }
    Witness::new(vec![dim], data, DataType::Float, *SF_LOG, Role::Constant)
}

/// Exact copy of the bert bin's weight gen (20-tuple): per layer attn norm+qkvo
/// weights+biases, proj norm+1+2 weights+biases; then final ln_w/ln_b + the
/// classifier matmul_w[hidden,num_classes]/matmul_b. Norm weights use
/// unit_norm_weight so the LN reciprocity gate holds.
#[allow(clippy::type_complexity)]
fn gen_bert_weights(num_layers: usize, hd: usize, ffn: usize, num_classes: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness, Witness, Witness,
) {
    let c = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let hdp = hd.next_power_of_two();
    let ffnp = ffn.next_power_of_two();
    let ncp = num_classes.next_power_of_two();
    let (mut anw, mut aqw, mut akw, mut avw, mut aow) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut anb, mut aqb, mut akb, mut avb, mut aob) = (vec![], vec![], vec![], vec![], vec![]);
    let (mut pnw, mut p1w, mut p2w, mut pnb, mut p1b, mut p2b) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..num_layers {
        anw.push(unit_norm_weight(hd));
        aqw.push(c(vec![hd, hd], hdp * hdp)); akw.push(c(vec![hd, hd], hdp * hdp));
        avw.push(c(vec![hd, hd], hdp * hdp)); aow.push(c(vec![hd, hd], hdp * hdp));
        anb.push(c(vec![hd], hdp)); aqb.push(c(vec![hd], hdp)); akb.push(c(vec![hd], hdp));
        avb.push(c(vec![hd], hdp)); aob.push(c(vec![hd], hdp));
        pnw.push(unit_norm_weight(hd));
        p1w.push(c(vec![hd, ffn], hdp * ffnp)); p2w.push(c(vec![ffn, hd], ffnp * hdp));
        pnb.push(c(vec![hd], hdp)); p1b.push(c(vec![ffn], ffnp)); p2b.push(c(vec![hd], hdp));
    }
    let ln_w = unit_norm_weight(hd);
    let ln_b = c(vec![hd], hdp);
    let matmul_w = c(vec![hd, num_classes], hdp * ncp);
    let matmul_b = c(vec![num_classes], ncp);
    (anw, aqw, akw, avw, aow, anb, aqb, akb, avb, aob, pnw, p1w, p2w, pnb, p1b, p2b,
     ln_w, ln_b, matmul_w, matmul_b)
}

fn demo_seed() -> Seed {
    Seed([0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
          0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_heads = 16; let head_dim = 64;
    let hidden_dim = num_heads * head_dim; // 1024
    let ffn_dim = hidden_dim * 4;          // 4096
    let num_classes = 2usize;

    println!("=== Streaming BERT (encoder, seq_len={}) ===", seq_len);
    println!("num_layers={} seq_len={} N_PROOFS={} hidden={} max_num_vars={} partitions={}",
             num_layers, seq_len, n_proofs, hidden_dim, max_num_vars, num_partitions);

    let weights = gen_bert_weights(num_layers, hidden_dim, ffn_dim, num_classes);
    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq_len, hidden_dim], DataType::Float);
    let _output = g.pipe(&[x], bert_large(
        weights.0, weights.1, weights.2, weights.3, weights.4,
        weights.5, weights.6, weights.7, weights.8, weights.9,
        weights.10, weights.11, weights.12, weights.13, weights.14, weights.15,
        weights.16, weights.17, num_heads, head_dim, seq_len, weights.18, weights.19));
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    assert_eq!(witnesses_template[x][0].role, Role::Input,
        "BERT input edge must be Role::Input (per-proof), not deferred");

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

    let pad: usize = [1usize, seq_len, hidden_dim].iter().map(|&s| s.next_power_of_two()).product();
    let stride_last = seq_len.next_power_of_two();

    let label = b"zkml-bert-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut checked_role = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} inferences (seq_len={} each):", n_proofs, seq_len);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let input = Witness::new(vec![1, seq_len, hidden_dim],
            varied_input(pad, stride_last, it), DataType::Float, *SF_LOG, Role::Input);

        let s0 = Instant::now();
        dag.run(&mut witnesses, &[(x, input)]);
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
                assert!(dc.edge_id != x, "input edge {} was deferred as a shared weight — unsound", x);
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
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  = streaming per-inf   : {:>8.2}ms",
        (ms(t_prove) + ms(t_acc) + ms(t_finalize)) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (BERT encoder, seq_len={}, weights amortized across {} inferences)", seq_len, n_proofs);
}
