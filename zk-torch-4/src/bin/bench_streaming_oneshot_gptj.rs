//! Streaming one-shot AR GPT-J (RoPE, parallel attn+FFN, LayerNorm, un-tied
//! lm_head). Each streamed proof is a full T-token generation; WEIGHTS are
//! Role::Constant (deferred → amortized), per-gen SELECTORS are Role::Input.
//! Mirror of bench_streaming_oneshot_llama2 with the gpt_j_6b_hidden body and
//! the LayerNorm-safe 0/2048 embedding (GPT-J has no positional add — RoPE —
//! so the embedding alone feeds the first LayerNorm; low variance cancels to
//! a negative variance and trips the range check, see oneshot_gptj).
//!
//! Run with `bench_config.yaml` as args[1]. Env: NUM_LAYERS(1) SEQ_LEN(4)
//! VOCAB_SIZE(256) N_PROOFS(5) NUM_HEADS(8) HEAD_DIM(64) FFN_DIM(2048)
//! MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::gptj::gpt_j_6b_hidden;
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::util::arith::f_to_int;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }
fn zeros(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// LayerNorm-safe high-variance embedding (0/2048 alternating along hidden).
fn embedding_matrix(vocab: usize, hidden: usize) -> Vec<AlmostGoldilocksField> {
    let vp = vocab.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(vp * hp);
    for d in 0..hp {
        let val = if d % 2 == 0 { AlmostGoldilocksField(0) } else { AlmostGoldilocksField(2048) };
        for v in 0..vocab { data[v + d * vp] = val; }
    }
    data
}

#[allow(clippy::type_complexity)]
fn gen_gptj_weights(num_layers: usize, hidden: usize, ffn: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Witness, Witness,
) {
    let c = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let hp = hidden.next_power_of_two();
    let ffp = ffn.next_power_of_two();
    let (mut anw, mut aqw, mut akw, mut avw, mut aow, mut anb) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut p1w, mut p2w, mut p1b, mut p2b) = (vec![], vec![], vec![], vec![]);
    for _ in 0..num_layers {
        anw.push(c(vec![hidden], hp));
        aqw.push(c(vec![hidden, hidden], hp * hp));
        akw.push(c(vec![hidden, hidden], hp * hp));
        avw.push(c(vec![hidden, hidden], hp * hp));
        aow.push(c(vec![hidden, hidden], hp * hp));
        anb.push(c(vec![hidden], hp));
        p1w.push(c(vec![hidden, ffn], hp * ffp));
        p2w.push(c(vec![ffn, hidden], ffp * hp));
        p1b.push(c(vec![ffn], ffp));
        p2b.push(c(vec![hidden], hp));
    }
    (anw, aqw, akw, avw, aow, anb, p1w, p2w, p1b, p2b, c(vec![hidden], hp), c(vec![hidden], hp))
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

/// Global argmax across vocab shards (see oneshot_gpt2). One shard ⇒ argmax_row.
fn argmax_row_sharded(
    witnesses: &[Vec<Witness>],
    logits_edges: &[EdgeId],
    shard_ranges: &[(usize, usize)],
    pos: usize,
) -> usize {
    let mut best_tok = 0usize; let mut bv = i128::MIN;
    for (k, &(start, len)) in shard_ranges.iter().enumerate() {
        let lg = &witnesses[logits_edges[k]][0];
        for off in 0..len {
            let val = f_to_int(lg.get(&[pos, off]));
            if val > bv { bv = val; best_tok = start + off; }
        }
    }
    best_tok
}

/// A `(len, hidden)` block of the GPT-J embedding head (0/2048-alternating by
/// hidden coord, rows identical), used as a vocab-sharded un-tied head weight.
fn embedding_matrix_shard(hidden: usize, len: usize) -> Vec<AlmostGoldilocksField> {
    let lp = len.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(lp * hp);
    for d in 0..hp {
        let val = if d % 2 == 0 { AlmostGoldilocksField(0) } else { AlmostGoldilocksField(2048) };
        for off in 0..len { data[off + d * lp] = val; }
    }
    data
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let vocab_size: usize = std::env::var("VOCAB_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let num_heads: usize = std::env::var("NUM_HEADS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let head_dim: usize = std::env::var("HEAD_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let ffn_dim: usize = std::env::var("FFN_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let vocab_shards: usize = std::env::var("VOCAB_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let hidden_dim = num_heads * head_dim;

    println!("=== Streaming one-shot AR GPT-J ===");
    println!("num_layers={} seq_len={} vocab={} N_PROOFS={} hidden={} heads={} head_dim={} ffn={} partitions={}",
             num_layers, seq_len, vocab_size, n_proofs, hidden_dim, num_heads, head_dim, ffn_dim, num_partitions);

    let weights = gen_gptj_weights(num_layers, hidden_dim, ffn_dim);
    let w_e_w = Witness::new(vec![vocab_size, hidden_dim], embedding_matrix(vocab_size, hidden_dim),
        DataType::Float, *SF_LOG, Role::Constant);
    let init_tokens = vec![0usize; seq_len];
    let mut g = DagBuilder::new();
    let w_e = g.param(w_e_w);
    let (h0, emb_sel) = g.embedding_lookup(w_e, seq_len, vocab_size, &init_tokens, Role::Input);
    let h_in = g.change_shape(h0, vec![1, seq_len, hidden_dim]);
    let h_out = g.pipe(&[h_in], gpt_j_6b_hidden(
        weights.0, weights.1, weights.2, weights.3, weights.4, weights.5,
        weights.6, weights.7, weights.8, weights.9, weights.10, weights.11,
        num_heads, head_dim, seq_len))[0];
    let (logits_edges, argmax_sels, shard_ranges): (Vec<EdgeId>, Vec<EdgeId>, Vec<(usize, usize)>) =
        if vocab_shards > 1 {
            let ranges = DagBuilder::vocab_shard_ranges(vocab_size, vocab_shards);
            let shard_vocabs: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
            let w_lm_ids: Vec<EdgeId> = shard_vocabs.iter().map(|&len| {
                let data = embedding_matrix_shard(hidden_dim, len);
                g.param(Witness::new(vec![len, hidden_dim], data, DataType::Float, *SF_LOG, Role::Constant))
            }).collect();
            let logits_shards = g.lm_head_sharded(h_out, &w_lm_ids, seq_len, &shard_vocabs);
            let sels = g.argmax_check_sharded(&logits_shards, &ranges, seq_len, &init_tokens, Role::Input);
            println!("Vocab-sharded head: {} shards (sizes {:?})", ranges.len(), shard_vocabs);
            (logits_shards, sels, ranges)
        } else {
            let w_lm_w = Witness::new(vec![vocab_size, hidden_dim], embedding_matrix(vocab_size, hidden_dim),
                DataType::Float, *SF_LOG, Role::Constant);
            let w_lm = g.param(w_lm_w);
            let logits = g.lm_head(h_out, w_lm, seq_len, vocab_size);
            let sel = g.argmax_check(logits, seq_len, vocab_size, &init_tokens, Role::Input);
            (vec![logits], vec![sel], vec![(0, vocab_size)])
        };
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    assert_eq!(witnesses_template[emb_sel][0].role, Role::Input,
        "embedding selector must be Role::Input (per-proof), not deferred");
    for &sel in &argmax_sels {
        assert_eq!(witnesses_template[sel][0].role, Role::Input,
            "argmax selector must be Role::Input (per-proof), not deferred");
    }

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

    let label = b"zkml-oneshot-gptj-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut rng = StdRng::seed_from_u64(42);
    let mut checked_roles = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} generations (seq_len={} each):", n_proofs, seq_len);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let tokens: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % vocab_size).collect();
        witnesses[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(
            seq_len, vocab_size, &tokens, Role::Input)];

        let s0 = Instant::now();
        dag.run(&mut witnesses, &[]);
        let next_tokens: Vec<usize> = (0..seq_len)
            .map(|i| argmax_row_sharded(&witnesses, &logits_edges, &shard_ranges, i)).collect();
        if vocab_shards > 1 {
            for (k, &(start, len)) in shard_ranges.iter().enumerate() {
                let local: Vec<Option<usize>> = next_tokens.iter()
                    .map(|&t| if t >= start && t < start + len { Some(t - start) } else { None }).collect();
                witnesses[argmax_sels[k]] = vec![DagBuilder::build_sharded_one_hot_selector_witness(
                    seq_len, len, &local, Role::Input)];
            }
        } else {
            witnesses[argmax_sels[0]] = vec![DagBuilder::build_one_hot_selector_witness(
                seq_len, vocab_size, &next_tokens, Role::Input)];
        }
        dag.run(&mut witnesses, &[]);
        let d_run = s0.elapsed(); t_run += d_run;

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-gen");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-gen");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-gen verify failed at gen {}", it); return; }

        if !checked_roles {
            for dc in &r.claims {
                assert!(dc.edge_id != emb_sel && !argmax_sels.contains(&dc.edge_id),
                    "selector edge {} was deferred as a shared weight — unsound", dc.edge_id);
            }
            checked_roles = true;
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
        if !ok { eprintln!("streaming verifier rejected at gen {}", it); return; }

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
    println!("  prove(defer)  per-gen : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-gen : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  = streaming per-gen   : {:>8.2}ms",
        (ms(t_prove) + ms(t_acc) + ms(t_finalize)) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (full one-shot AR generations, weights amortized across {} gens)", n_proofs);
}
