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
    let num_heads = 12; let head_dim = 64;
    let hidden_dim = num_heads * head_dim; let ffn_dim = hidden_dim * 4;
    let weights = gen_gpt2_weights(num_layers, hidden_dim, ffn_dim);
    let w_e_w = Witness::new(vec![vocab_size, hidden_dim], embedding_matrix(vocab_size, hidden_dim),
        DataType::Float, *SF_LOG, Role::Constant);
    let pos_pad = seq_len.next_power_of_two() * hidden_dim.next_power_of_two();
    let pos_w = Witness::new(vec![seq_len, hidden_dim], zeros(pos_pad), DataType::Float, *SF_LOG, Role::Constant);
    let init_tokens = vec![0usize; seq_len];
    let mut g = DagBuilder::new();
    let w_e = g.param(w_e_w);
    let (h0, emb_sel) = g.embedding_lookup(w_e, seq_len, vocab_size, &init_tokens, Role::Input);
    let h_pe = g.add_positional_encoding(h0, pos_w);
    let h_in = g.change_shape(h_pe, vec![1, seq_len, hidden_dim]);
    let h_out = g.pipe(&[h_in], gpt_2_small(
        weights.0, weights.1, weights.2, weights.3, weights.4,
        weights.5, weights.6, weights.7, weights.8, weights.9,
        weights.10, weights.11, weights.12, weights.13, weights.14, weights.15,
        weights.16, weights.17, num_heads, head_dim, seq_len))[0];
    let logits = g.lm_head_weight_tied(h_out, w_e, seq_len, vocab_size);
    let sel = g.argmax_check(logits, seq_len, vocab_size, &init_tokens, Role::Input);
    let (dag, mut witnesses) = g.compile();
    let mut rng = StdRng::seed_from_u64(42);
    let tokens: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % vocab_size).collect();
    witnesses[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq_len, vocab_size, &tokens, Role::Input)];
    dag.run(&mut witnesses, &[]);
    let next: Vec<usize> = (0..seq_len).map(|i| argmax_row(&witnesses[logits][0], i, vocab_size)).collect();
    witnesses[sel] = vec![DagBuilder::build_one_hot_selector_witness(seq_len, vocab_size, &next, Role::Input)];
    dag.run(&mut witnesses, &[]);
    println!("nodes {} edges {}", dag.nodes.len(), dag.num_edges());
    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for nd in &dag.nodes { let k = format!("{:?}", nd.kind); let k = k.split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap().to_string(); *kinds.entry(k).or_default() += 1; }
    println!("node kinds: {:?}", kinds);
    println!("range nodes {} two_pow nodes {}", dag.range.len(), dag.two_pow.len());
    let mut tot: std::collections::BTreeMap<String, (usize, u128, u128, u128)> = Default::default();
    for (e, ws) in witnesses.iter().enumerate() {
        let w = &ws[0];
        let committed = dag.should_commit(w, e);
        let data = match w.data.as_ref() { Some(d) => d, None => continue };
        let len = data.len();
        let (mut maxb, mut nnz) = (0u32, 0usize);
        let evs = data.evaluations();
        for v in evs.iter() {
            let x = f_to_int(*v);
            if x != 0 { nnz += 1; }
            let b = 128 - x.unsigned_abs().leading_zeros();
            maxb = maxb.max(b);
        }
        let key = format!("{:?}/{:?}/{}", w.role, w.poly_type, if committed { "commit" } else { "-" });
        let t = tot.entry(key).or_default();
        t.0 += 1; t.1 += len as u128; t.2 += nnz as u128; t.3 += (len as u128) * (maxb.max(1) as u128);
        if std::env::var("PROBE_ALL").is_ok() && committed {
            println!("edge {e:5} {:?} {:?} shape {:?} len {} nnz {} maxbits {}", w.role, w.poly_type, w.shape, len, nnz, maxb);
        }
    }
    println!("role/type/commit: count, len, nnz, len*bits (digits)");
    for (k, v) in tot { println!("{k:40} {:6} {:>14} {:>14} {:>16}", v.0, v.1, v.2, v.3); }
}
