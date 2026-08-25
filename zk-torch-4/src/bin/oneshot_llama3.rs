//! One-shot autoregressive Llama-3.1 prover.
//!
//! tokens → embedding_lookup(W_E) → llama3 transformer (RoPE θ=500000, GQA,
//! SwiGLU, causal, seq_len=T) → lm_head(W_LM, UN-tied) → argmax_check, proved
//! in ONE proof. Same shape as `oneshot_llama` but uses `llama3_8b_hidden`
//! (GQA `num_kv_heads`). RMSNorm (no mean-subtract) → constant embedding is
//! fine. Two-pass prover (dummy argmax selector → read logits → set true
//! per-row argmax → rerun → commit → prove → verify).
//!
//! Run with `bench_config.yaml` as args[1] (table_size_log=10). One-shot
//! lm_head is UN-sharded → keep VOCAB modest. Env: NUM_LAYERS(1) SEQ_LEN(4)
//! VOCAB(256) NUM_HEADS(8) NUM_KV_HEADS(8) HEAD_DIM(64) FFN_DIM(2048)
//! MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::llama::llama3_8b_hidden;
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::util::arith::f_to_int;
use zk_torch_4::SF_LOG;

fn zeros(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// RMSNorm-friendly embedding (constant 1024 ≈ 1.0 at SF=10) across the full
/// padded buffer — Llama RMSNorm doesn't subtract the mean (see oneshot_llama).
fn embedding_matrix(vocab: usize, hidden: usize) -> Vec<AlmostGoldilocksField> {
    let vp = vocab.next_power_of_two();
    let hp = hidden.next_power_of_two();
    vec![AlmostGoldilocksField(1024); vp * hp]
}

#[allow(clippy::type_complexity)]
fn gen_llama3_weights(num_layers: usize, hidden: usize, kv_dim: usize, ffn: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Witness,
) {
    let c = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let hp = hidden.next_power_of_two();
    let kvp = kv_dim.next_power_of_two();
    let ffp = ffn.next_power_of_two();
    let (mut anw, mut aqw, mut akw, mut avw, mut aow, mut pnw) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut p1, mut p2, mut p3) = (vec![], vec![], vec![]);
    for _ in 0..num_layers {
        anw.push(c(vec![hidden], hp));
        aqw.push(c(vec![hidden, hidden], hp * hp));
        akw.push(c(vec![hidden, kv_dim], hp * kvp));
        avw.push(c(vec![hidden, kv_dim], hp * kvp));
        aow.push(c(vec![hidden, hidden], hp * hp));
        pnw.push(c(vec![hidden], hp));
        p1.push(c(vec![hidden, ffn], hp * ffp));
        p2.push(c(vec![hidden, ffn], hp * ffp));
        p3.push(c(vec![ffn, hidden], ffp * hp));
    }
    (anw, aqw, akw, avw, aow, pnw, p1, p2, p3, c(vec![hidden], hp))
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

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let vocab: usize = std::env::var("VOCAB").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let num_heads: usize = std::env::var("NUM_HEADS").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let num_kv_heads: usize = std::env::var("NUM_KV_HEADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let head_dim: usize = std::env::var("HEAD_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let ffn_dim: usize = std::env::var("FFN_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(14336);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let vocab_shards: usize = std::env::var("VOCAB_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let hidden_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    println!("=== One-Shot Llama-3.1 (full AR pipeline) ===");
    println!("num_layers={} seq_len={} vocab={} hidden={} heads={} kv_heads={} head_dim={} ffn={} max_num_vars={} partitions={}",
             num_layers, seq_len, vocab, hidden_dim, num_heads, num_kv_heads, head_dim, ffn_dim, max_num_vars, num_partitions);

    let mut rng = StdRng::seed_from_u64(42);
    let token_ids: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % vocab).collect();

    let t0 = Instant::now();
    let w = gen_llama3_weights(num_layers, hidden_dim, kv_dim, ffn_dim);
    let w_e_w = Witness::new(vec![vocab, hidden_dim], embedding_matrix(vocab, hidden_dim),
        DataType::Float, *SF_LOG, Role::Constant);
    let mut g = DagBuilder::new();
    let w_e = g.param(w_e_w);
    let (h0, _emb_sel) = g.embedding_lookup(w_e, seq_len, vocab, &token_ids, Role::Constant);
    let h_in = g.change_shape(h0, vec![1, seq_len, hidden_dim]);
    let h_out = g.pipe(&[h_in], llama3_8b_hidden(
        w.0, w.1, w.2, w.3, w.4, w.5,
        // Not FFN-sharded here; single-element inner vecs are the un-sharded
        // form llama_mlp already handled.
        w.6.into_iter().map(|x| vec![x]).collect(),
        w.7.into_iter().map(|x| vec![x]).collect(),
        w.8.into_iter().map(|x| vec![x]).collect(),
        w.9,
        seq_len, head_dim, num_heads, num_kv_heads))[0];
    let dummy = vec![0usize; seq_len];
    let (logits_edges, argmax_sels, shard_ranges): (Vec<EdgeId>, Vec<EdgeId>, Vec<(usize, usize)>) =
        if vocab_shards > 1 {
            let ranges = DagBuilder::vocab_shard_ranges(vocab, vocab_shards);
            let shard_vocabs: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
            let w_lm_ids: Vec<EdgeId> = shard_vocabs.iter().map(|&len| {
                let data = vec![AlmostGoldilocksField(1024); len.next_power_of_two() * hidden_dim.next_power_of_two()];
                g.param(Witness::new(vec![len, hidden_dim], data, DataType::Float, *SF_LOG, Role::Constant))
            }).collect();
            let logits_shards = g.lm_head_sharded(h_out, &w_lm_ids, seq_len, &shard_vocabs);
            let sels = g.argmax_check_sharded(&logits_shards, &ranges, seq_len, &dummy, Role::Constant);
            println!("Vocab-sharded head: {} shards (sizes {:?})", ranges.len(), shard_vocabs);
            (logits_shards, sels, ranges)
        } else {
            let w_lm_w = Witness::new(vec![vocab, hidden_dim],
                vec![AlmostGoldilocksField(1024); vocab.next_power_of_two() * hidden_dim.next_power_of_two()],
                DataType::Float, *SF_LOG, Role::Constant);
            let w_lm = g.param(w_lm_w);
            let logits = g.lm_head(h_out, w_lm, seq_len, vocab);
            let sel = g.argmax_check(logits, seq_len, vocab, &dummy, Role::Constant);
            (vec![logits], vec![sel], vec![(0, vocab)])
        };
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

    let t1 = Instant::now();
    dag.run(&mut witnesses, &[]);
    println!("Forward (pass 1): {:?}", t1.elapsed());

    let next_tokens: Vec<usize> = (0..seq_len)
        .map(|i| argmax_row_sharded(&witnesses, &logits_edges, &shard_ranges, i)).collect();
    if vocab_shards > 1 {
        for (k, &(start, len)) in shard_ranges.iter().enumerate() {
            let local: Vec<Option<usize>> = next_tokens.iter()
                .map(|&t| if t >= start && t < start + len { Some(t - start) } else { None }).collect();
            witnesses[argmax_sels[k]] = vec![DagBuilder::build_sharded_one_hot_selector_witness(
                seq_len, len, &local, Role::Constant)];
        }
    } else {
        witnesses[argmax_sels[0]] = vec![DagBuilder::build_one_hot_selector_witness(
            seq_len, vocab, &next_tokens, Role::Constant)];
    }
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[]);
    println!("Forward (pass 2): {:?}", t2.elapsed());

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
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

    let mut tp = Transcript::new(b"oneshot-llama3");
    let t4 = Instant::now();
    let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut tp);
    println!("Prove: {:?}", t4.elapsed());
    let mut tv = Transcript::new(b"oneshot-llama3");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut tv);
    println!("Verify: {:?}", t5.elapsed());
    println!("\nVerified: {}", verified);
    assert!(verified, "one-shot Llama-3 proof failed to verify");
}
