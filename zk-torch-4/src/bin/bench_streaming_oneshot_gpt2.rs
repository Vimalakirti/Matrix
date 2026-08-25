//! Streaming one-shot AR GPT-2 (step 3): one-shot AR proving composed
//! with the cross-proof streaming accumulator.
//!
//! Each streamed "proof" is a FULL T-token one-shot AR generation
//! (embedding → transformer → lm_head → argmax_check), not a single
//! token. Across the N generations:
//!   - the model WEIGHTS (W_E, transformer, positional) are `Role::Constant`
//!     → deferred per-proof and amortized into ONE fold-tree opening at
//!     finalize (the streaming accumulator's job);
//!   - the per-generation one-hot SELECTORS (embedding + argmax) are
//!     `Role::Input` (via `committed_input`) → committed and opened
//!     per-proof, NEVER deferred (they change every generation).
//!
//! This is the composition the scoping called for: one-shot batches the T
//! AR steps of one generation into one proof; streaming amortizes the
//! shared-weight openings across the N generations. They stack.
//!
//! Run with `bench_config.yaml` as args[1] (sets table_size_log=10).
//! Env: NUM_LAYERS (1), SEQ_LEN (4), VOCAB_SIZE (256), N_PROOFS (5),
//! MAX_NUM_VARS (22), NUM_PARTITIONS (1).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::gpt2::gpt_2_small;
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::util::arith::f_to_int;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }
fn zeros(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// Embedding matrix `(vocab, hidden)` — every row is the RMSNorm-friendly
/// varied pattern, filled across the FULL padded hidden width (the
/// transformer LayerNorm reduces over the padded axis; see oneshot_gpt2).
fn embedding_matrix(vocab: usize, hidden: usize) -> Vec<AlmostGoldilocksField> {
    let vp = vocab.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(vp * hp);
    for d in 0..hp {
        let val = AlmostGoldilocksField(1000 + 10 * ((d % 16) as u64));
        for v in 0..vocab { data[v + d * vp] = val; }
    }
    data
}

#[allow(clippy::type_complexity)]
fn gen_gpt2_weights(num_layers: usize, hd: usize, ffn: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness,
) {
    let c = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let hdp = hd.next_power_of_two();
    let ffnp = ffn.next_power_of_two();
    let mut v: [Vec<Witness>; 16] = Default::default();
    for _ in 0..num_layers {
        v[0].push(c(vec![hd], hdp)); v[1].push(c(vec![hd, hd], hdp * hdp));
        v[2].push(c(vec![hd, hd], hdp * hdp)); v[3].push(c(vec![hd, hd], hdp * hdp));
        v[4].push(c(vec![hd, hd], hdp * hdp)); v[5].push(c(vec![hd], hdp));
        v[6].push(c(vec![hd], hdp)); v[7].push(c(vec![hd], hdp));
        v[8].push(c(vec![hd], hdp)); v[9].push(c(vec![hd], hdp));
        v[10].push(c(vec![hd], hdp)); v[11].push(c(vec![hd, ffn], hdp * ffnp));
        v[12].push(c(vec![ffn, hd], ffnp * hdp)); v[13].push(c(vec![hd], hdp));
        v[14].push(c(vec![ffn], ffnp)); v[15].push(c(vec![hd], hdp));
    }
    let ln_w = c(vec![hd], hdp); let ln_b = c(vec![hd], hdp);
    let [a, b, cc, d, e, f, g, h, i, j, k, l, m, n, o, p] = v;
    (a, b, cc, d, e, f, g, h, i, j, k, l, m, n, o, p, ln_w, ln_b)
}

fn demo_seed() -> Seed {
    Seed([0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
          0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE])
}

fn argmax_row(logits: &Witness, pos: usize, vocab: usize) -> usize {
    let mut best = 0usize; let mut best_val = i128::MIN;
    for v in 0..vocab {
        let val = f_to_int(logits.get(&[pos, v]));
        if val > best_val { best_val = val; best = v; }
    }
    best
}

/// Global argmax across vocab shards (see oneshot_gpt2). With one shard this
/// equals `argmax_row`.
fn argmax_row_sharded(
    witnesses: &[Vec<Witness>],
    logits_edges: &[EdgeId],
    shard_ranges: &[(usize, usize)],
    pos: usize,
) -> usize {
    let mut best_tok = 0usize; let mut best_val = i128::MIN;
    for (k, &(start, len)) in shard_ranges.iter().enumerate() {
        let lg = &witnesses[logits_edges[k]][0];
        for off in 0..len {
            let val = f_to_int(lg.get(&[pos, off]));
            if val > best_val { best_val = val; best_tok = start + off; }
        }
    }
    best_tok
}

/// A `(len, hidden)` row-block of the tied embedding matrix, used as an
/// un-tied LM-head weight shard (rows are value-identical, so it matches the
/// tied head exactly; see oneshot_gpt2).
fn embedding_matrix_shard(hidden: usize, len: usize) -> Vec<AlmostGoldilocksField> {
    let lp = len.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(lp * hp);
    for d in 0..hp {
        let val = AlmostGoldilocksField(1000 + 10 * ((d % 16) as u64));
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
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // VOCAB_SHARDS>1 splits the LM head + argmax range-check along the vocab
    // axis (needed for full vocab 50257/32000 — see oneshot_gpt2).
    let vocab_shards: usize = std::env::var("VOCAB_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_heads = 12; let head_dim = 64;
    let hidden_dim = num_heads * head_dim; let ffn_dim = hidden_dim * 4;

    println!("=== Streaming one-shot AR GPT-2 ===");
    println!("num_layers={} seq_len={} vocab={} N_PROOFS={} max_num_vars={} partitions={}",
             num_layers, seq_len, vocab_size, n_proofs, max_num_vars, num_partitions);

    // ---- Setup: build the one-shot circuit ONCE ----
    let weights = gen_gpt2_weights(num_layers, hidden_dim, ffn_dim);
    let w_e_w = Witness::new(vec![vocab_size, hidden_dim], embedding_matrix(vocab_size, hidden_dim),
        DataType::Float, *SF_LOG, Role::Constant);
    let pos_pad = seq_len.next_power_of_two() * hidden_dim.next_power_of_two();
    let pos_w = Witness::new(vec![seq_len, hidden_dim], zeros(pos_pad), DataType::Float, *SF_LOG, Role::Constant);

    let init_tokens = vec![0usize; seq_len];
    let mut g = DagBuilder::new();
    let w_e = g.param(w_e_w);
    // Selectors are Role::Input → committed + opened per-proof, NOT deferred.
    let (h0, emb_sel) = g.embedding_lookup(w_e, seq_len, vocab_size, &init_tokens, Role::Input);
    let h_pe = g.add_positional_encoding(h0, pos_w);
    let h_in = g.change_shape(h_pe, vec![1, seq_len, hidden_dim]);
    let h_out = g.pipe(&[h_in], gpt_2_small(
        weights.0, weights.1, weights.2, weights.3, weights.4,
        weights.5, weights.6, weights.7, weights.8, weights.9,
        weights.10, weights.11, weights.12, weights.13, weights.14, weights.15,
        weights.16, weights.17, num_heads, head_dim, seq_len))[0];
    let (logits_edges, argmax_sels, shard_ranges): (Vec<EdgeId>, Vec<EdgeId>, Vec<(usize, usize)>) =
        if vocab_shards > 1 {
            let ranges = DagBuilder::vocab_shard_ranges(vocab_size, vocab_shards);
            let shard_vocabs: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
            // Head weight shards are Role::Constant → deferred + amortized like
            // every other weight; only the selectors are per-proof.
            let w_lm_ids: Vec<EdgeId> = shard_vocabs
                .iter()
                .map(|&len| {
                    let data = embedding_matrix_shard(hidden_dim, len);
                    g.param(Witness::new(vec![len, hidden_dim], data, DataType::Float, *SF_LOG, Role::Constant))
                })
                .collect();
            let logits_shards = g.lm_head_sharded(h_out, &w_lm_ids, seq_len, &shard_vocabs);
            let sels = g.argmax_check_sharded(&logits_shards, &ranges, seq_len, &init_tokens, Role::Input);
            println!("Vocab-sharded head: {} shards (sizes {:?})", ranges.len(), shard_vocabs);
            (logits_shards, sels, ranges)
        } else {
            let logits = g.lm_head_weight_tied(h_out, w_e, seq_len, vocab_size);
            let sel = g.argmax_check(logits, seq_len, vocab_size, &init_tokens, Role::Input);
            (vec![logits], vec![sel], vec![(0, vocab_size)])
        };
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    // Safety: the per-generation selectors MUST NOT be Constant (else the
    // accumulator would defer + amortize them as one shared value).
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

    // ---- Streaming loop: each iter is one full T-token generation ----
    let label = b"zkml-oneshot-gpt2-streaming";
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
        // Fresh prompt tokens for this generation → fresh embedding selector.
        let tokens: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % vocab_size).collect();
        witnesses[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(
            seq_len, vocab_size, &tokens, Role::Input)];

        // Pass 1: forward with the prior argmax selector, read logits.
        let s0 = Instant::now();
        dag.run(&mut witnesses, &[]);
        let next_tokens: Vec<usize> = (0..seq_len)
            .map(|i| argmax_row_sharded(&witnesses, &logits_edges, &shard_ranges, i)).collect();
        // Pass 2: fix argmax selector(s) to the true argmax, rerun downstream.
        if vocab_shards > 1 {
            for (k, &(start, len)) in shard_ranges.iter().enumerate() {
                let local: Vec<Option<usize>> = next_tokens
                    .iter()
                    .map(|&t| if t >= start && t < start + len { Some(t - start) } else { None })
                    .collect();
                witnesses[argmax_sels[k]] = vec![DagBuilder::build_sharded_one_hot_selector_witness(
                    seq_len, len, &local, Role::Input)];
            }
        } else {
            witnesses[argmax_sels[0]] = vec![DagBuilder::build_one_hot_selector_witness(
                seq_len, vocab_size, &next_tokens, Role::Input)];
        }
        dag.run(&mut witnesses, &[]);
        t_run += s0.elapsed(); let d_run = s0.elapsed();

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        t_commit += s1.elapsed(); let d_commit = s1.elapsed();
        if it == 0 && std::env::var("ZK4_PLANES_DBG").is_ok() {
            let (sb, sn, db, dn) = store.planes_cache_bytes();
            eprintln!("[planes_cache] sparse: {} edges, {:.2} GB | dense: {} edges, {:.2} GB",
                sn, sb as f64 / 1e9, dn, db as f64 / 1e9);
        }

        let mut tp = Transcript::new(b"per-gen");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, /*defer=*/ true);
        t_prove += s2.elapsed(); let d_prove = s2.elapsed();

        let mut tv = Transcript::new(b"per-gen");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        t_verify += s3.elapsed(); let d_verify = s3.elapsed();
        if !r.ok { eprintln!("per-gen verify failed at gen {}", it); return; }

        // Safety (once): no deferred (= Constant, amortized) claim may be a
        // per-generation selector edge.
        if !checked_roles {
            for dc in &r.claims {
                assert!(dc.edge_id != emb_sel && !argmax_sels.contains(&dc.edge_id),
                    "selector edge {} was deferred as a shared weight — unsound", dc.edge_id);
            }
            checked_roles = true;
        }

        let s4 = Instant::now();
        let chunk = prover_acc.add_proof(&r, &witnesses);
        proof_bytes += ser_len(&dp) + ser_len(&fp) + ser_len(&chunk);
        // Keep one proof's component split; every iteration has the
        // same shape, so the last is representative.
        breakdown = Some(zk_torch_4::proof_size_report(
            &dp.node_proofs, &dp.edge_proofs, &dp.range_proof,
            &dp.two_pow_proof, &dp.output_claims, &dp, &fp));
        t_acc += s4.elapsed(); let d_acc = s4.elapsed();
        let s5 = Instant::now();
        let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
        t_accv += s5.elapsed(); let d_accv = s5.elapsed();
        if !ok { eprintln!("streaming verifier rejected at gen {}", it); return; }

        println!("  [{:>2}/{}] run {:>7.1}ms commit {:>6.1}ms prove {:>7.1}ms verify {:>6.1}ms acc {:>7.1}ms acc-v {:>6.1}ms",
            it + 1, n_proofs, ms(d_run), ms(d_commit), ms(d_prove), ms(d_verify), ms(d_acc), ms(d_accv));
    }

    // ---- Finalize: one fold-tree open over all accumulated weight claims ----
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
