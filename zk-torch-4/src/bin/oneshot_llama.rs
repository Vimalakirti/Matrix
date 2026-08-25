//! One-shot autoregressive Llama-2 prover (one-shot AR for Llama).
//!
//! Full one-shot circuit:
//!   tokens → embedding_lookup(W_E) → llama transformer (RoPE, SwiGLU,
//!   causal, seq_len=T) → lm_head(W_LM, UN-tied) → argmax_check
//! proved in ONE proof. Differs from oneshot_gpt2: Llama uses RoPE (no
//! additive positional encoding) and a SEPARATE lm_head weight (not tied
//! to the embedding). Uses `llama_2_7b_hidden` (the transformer body
//! returning hidden; the bundled `llama_2_7b` head folds the seq axis
//! assuming seq_len==1).
//!
//! Two-pass prover: run with a dummy argmax selector → read logits → set
//! the selector to the true per-row argmax → rerun → commit → prove →
//! verify. SKIP-AR style (random prompt; the public shift constraint is
//! out-of-circuit and skipped here).
//!
//! Run with `bench_config.yaml` as args[1] (table_size_log=10). Defaults
//! to a SMALL model (the one-shot lm_head is un-sharded, so keep VOCAB
//! modest; full Llama vocab=32000 needs a sharded head + sharded argmax —
//! a follow-up). Env: NUM_LAYERS(1) SEQ_LEN(4) VOCAB(256) NUM_HEADS(8)
//! HEAD_DIM(64) FFN_DIM(2048) MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::llama::llama_2_7b_hidden;
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::util::arith::f_to_int;
use zk_torch_4::SF_LOG;

fn zeros(n: usize) -> Vec<AlmostGoldilocksField> { zk_torch_4::zero_witness_vec(n) }

/// Embedding matrix `(vocab, hidden)` = constant 1024 (≈1.0 at SF=10)
/// across the FULL padded buffer. Llama RMSNorm doesn't subtract the mean,
/// so a constant hidden keeps `mean(x²)` exact and the reciprocity gate
/// holds (cf. llama2 bin's `small_input`). Filling the padded width too
/// (the RMSNorm reduces over the padded axis — see oneshot_gpt2).
fn embedding_matrix(vocab: usize, hidden: usize) -> Vec<AlmostGoldilocksField> {
    let vp = vocab.next_power_of_two();
    let hp = hidden.next_power_of_two();
    vec![AlmostGoldilocksField(1024); vp * hp]
}

#[allow(clippy::type_complexity)]
fn gen_llama_weights(num_layers: usize, hd: usize, ffn: usize, ffn_shards: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Vec<Witness>>, Vec<Vec<Witness>>, Vec<Vec<Witness>>, Witness,
) {
    let c = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let hdp = hd.next_power_of_two();
    let ffnp = ffn.next_power_of_two();
    let shard_ffn = ffn / ffn_shards;
    let shard_ffn_pad = ffnp / ffn_shards;
    let (mut anw, mut aqw, mut akw, mut avw, mut aow, mut pnw) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut p1, mut p2, mut p3): (Vec<Vec<Witness>>, Vec<Vec<Witness>>, Vec<Vec<Witness>>) =
        (vec![], vec![], vec![]);
    for _ in 0..num_layers {
        anw.push(c(vec![hd], hdp));
        aqw.push(c(vec![hd, hd], hdp * hdp)); akw.push(c(vec![hd, hd], hdp * hdp));
        avw.push(c(vec![hd, hd], hdp * hdp)); aow.push(c(vec![hd, hd], hdp * hdp));
        pnw.push(c(vec![hd], hdp));
        p1.push((0..ffn_shards).map(|_| c(vec![hd, shard_ffn], hdp * shard_ffn_pad)).collect());
        p2.push((0..ffn_shards).map(|_| c(vec![hd, shard_ffn], hdp * shard_ffn_pad)).collect());
        p3.push((0..ffn_shards).map(|_| c(vec![shard_ffn, hd], shard_ffn_pad * hdp)).collect());
    }
    let ln_w = c(vec![hd], hdp);
    (anw, aqw, akw, avw, aow, pnw, p1, p2, p3, ln_w)
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
    let head_dim: usize = std::env::var("HEAD_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let ffn_dim: usize = std::env::var("FFN_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(11008);
    let ffn_shards: usize = std::env::var("FFN_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // VOCAB_SHARDS>1 splits the LM head + argmax range-check for full vocab (32000).
    let vocab_shards: usize = std::env::var("VOCAB_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let hidden_dim = num_heads * head_dim;

    println!("=== One-Shot Llama-2 (full AR pipeline) ===");
    println!("num_layers={} seq_len={} vocab={} hidden={} heads={} head_dim={} ffn={} max_num_vars={} partitions={}",
             num_layers, seq_len, vocab, hidden_dim, num_heads, head_dim, ffn_dim, max_num_vars, num_partitions);

    let mut rng = StdRng::seed_from_u64(42);
    let token_ids: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % vocab).collect();

    // ---- Build the one-shot circuit ----
    let t0 = Instant::now();
    let w = gen_llama_weights(num_layers, hidden_dim, ffn_dim, ffn_shards);
    let w_e_w = Witness::new(vec![vocab, hidden_dim], embedding_matrix(vocab, hidden_dim),
        DataType::Float, *SF_LOG, Role::Constant);
    let mut g = DagBuilder::new();
    let w_e = g.param(w_e_w);
    let (h0, _emb_sel) = g.embedding_lookup(w_e, seq_len, vocab, &token_ids, Role::Constant);
    let h_in = g.change_shape(h0, vec![1, seq_len, hidden_dim]);
    let h_out = g.pipe(&[h_in], llama_2_7b_hidden(
        w.0, w.1, w.2, w.3, w.4, w.5, w.6, w.7, w.8, w.9,
        num_heads, head_dim, seq_len))[0];
    let dummy = vec![0usize; seq_len];
    let (logits_edges, argmax_sels, shard_ranges): (Vec<EdgeId>, Vec<EdgeId>, Vec<(usize, usize)>) =
        if vocab_shards > 1 {
            let ranges = DagBuilder::vocab_shard_ranges(vocab, vocab_shards);
            let shard_vocabs: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
            let w_lm_ids: Vec<EdgeId> = shard_vocabs
                .iter()
                .map(|&len| {
                    let data = vec![AlmostGoldilocksField(1024);
                        len.next_power_of_two() * hidden_dim.next_power_of_two()];
                    g.param(Witness::new(vec![len, hidden_dim], data, DataType::Float, *SF_LOG, Role::Constant))
                })
                .collect();
            let logits_shards = g.lm_head_sharded(h_out, &w_lm_ids, seq_len, &shard_vocabs);
            let sels = g.argmax_check_sharded(&logits_shards, &ranges, seq_len, &dummy, Role::Constant);
            println!("Vocab-sharded head: {} shards (sizes {:?})", ranges.len(), shard_vocabs);
            (logits_shards, sels, ranges)
        } else {
            // Separate (un-tied) lm_head weight, shape (vocab, hidden).
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
    println!("argmax(logits): {:?}", &next_tokens);
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

    let mut tp = Transcript::new(b"oneshot-llama");
    let t4 = Instant::now();
    let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut tp);
    println!("Prove: {:?}", t4.elapsed());
    let mut tv = Transcript::new(b"oneshot-llama");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut tv);
    println!("Verify: {:?}", t5.elapsed());
    println!("\nVerified: {}", verified);
    assert!(verified, "one-shot Llama proof failed to verify");
}
