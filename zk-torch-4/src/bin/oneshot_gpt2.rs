//! One-shot autoregressive GPT-2 prover (step 2 of one-shot AR).
//!
//! Builds a single full-sequence circuit:
//!   tokens → embedding_lookup(W_E) → +positional → transformer (causal,
//!   seq_len = T) → lm_head(W_E, tied) → argmax_check
//! and proves it in ONE proof (vs T per-token proofs). Causal masking
//! makes a single full-sequence forward reproduce every AR step; the
//! argmax_check binds each position's prediction.
//!
//! Two-pass prover flow (matches zk-torch-2's oneshot bins): run forward
//! with a dummy argmax selector, read logits, set the selector to the
//! true per-row argmax, rerun downstream, then commit + prove + verify.
//! The public shift constraint (argmax(logits[i]) == token[i+1]) is an
//! AR-soundness check orchestrated outside the circuit; this standalone
//! bin uses random prompt tokens (SKIP-AR style) and only proves the
//! circuit, so it is skipped here.
//!
//! Selectors use `Role::Constant` here (single standalone proof). Under
//! the streaming accumulator (step 3) they must instead be per-proof
//! (`Role::Input`) so they are not deferred+amortized as shared weights.
//!
//! Env: NUM_LAYERS (1), SEQ_LEN (4), VOCAB_SIZE (256), MAX_NUM_VARS (22),
//! NUM_PARTITIONS (1).

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::gpt2::gpt_2_small;
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::util::arith::f_to_int;
use zk_torch_4::SF_LOG;

fn zeros(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::zero_witness_vec(size)
}

/// Embedding matrix `(vocab, hidden)` whose every row is the RMSNorm-
/// friendly varied pattern the plain gpt2 bench feeds as input: value
/// depends only on the hidden coord d (varies along the axis LayerNorm
/// reduces over), so `h0[i,:] = W_E[token[i],:]` has clean mean/variance.
/// Col-major: element (v, d) at v + d·vocab_pad.
///
/// NB: the columns are filled across the FULL padded hidden width
/// (`d ∈ 0..hidden_pad`), exactly like the plain bin's `small_varied_input`
/// fills its full padded buffer. The transformer's LayerNorm reduces over
/// the padded hidden axis, so leaving the pad columns zero (far from the
/// mean) corrupts the variance and trips its range check.
fn embedding_matrix(vocab: usize, hidden: usize) -> Vec<AlmostGoldilocksField> {
    let vp = vocab.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(vp * hp);
    for d in 0..hp {
        let val = AlmostGoldilocksField(1000 + 10 * ((d % 16) as u64));
        for v in 0..vocab {
            data[v + d * vp] = val;
        }
    }
    data
}

#[allow(clippy::type_complexity)]
fn gen_gpt2_weights(num_layers: usize, hidden_dim: usize, ffn_dim: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness,
) {
    let c = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let hd = hidden_dim; let hd_pad = hidden_dim.next_power_of_two();
    let ffn = ffn_dim; let ffn_pad = ffn_dim.next_power_of_two();
    let mut v: [Vec<Witness>; 16] = Default::default();
    for _ in 0..num_layers {
        v[0].push(c(vec![hd], hd_pad));                 // attn_norm_w
        v[1].push(c(vec![hd, hd], hd_pad * hd_pad));    // attn_q_w
        v[2].push(c(vec![hd, hd], hd_pad * hd_pad));    // attn_k_w
        v[3].push(c(vec![hd, hd], hd_pad * hd_pad));    // attn_v_w
        v[4].push(c(vec![hd, hd], hd_pad * hd_pad));    // attn_o_w
        v[5].push(c(vec![hd], hd_pad));                 // attn_norm_b
        v[6].push(c(vec![hd], hd_pad));                 // attn_q_b
        v[7].push(c(vec![hd], hd_pad));                 // attn_k_b
        v[8].push(c(vec![hd], hd_pad));                 // attn_v_b
        v[9].push(c(vec![hd], hd_pad));                 // attn_o_b
        v[10].push(c(vec![hd], hd_pad));                // proj_norm_w
        v[11].push(c(vec![hd, ffn], hd_pad * ffn_pad)); // proj_1_w
        v[12].push(c(vec![ffn, hd], ffn_pad * hd_pad)); // proj_2_w
        v[13].push(c(vec![hd], hd_pad));                // proj_norm_b
        v[14].push(c(vec![ffn], ffn_pad));             // proj_1_b
        v[15].push(c(vec![hd], hd_pad));                // proj_2_b
    }
    let ln_w = c(vec![hd], hd_pad);
    let ln_b = c(vec![hd], hd_pad);
    let [a, b, cc, d, e, f, g, h, i, j, k, l, m, n, o, p] = v;
    (a, b, cc, d, e, f, g, h, i, j, k, l, m, n, o, p, ln_w, ln_b)
}

fn demo_seed() -> Seed {
    Seed([
        0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
        0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
    ])
}

/// argmax over vocab of logits row `pos`. logits shape `(seq, vocab)`,
/// col-major: element (pos, v) at pos + v·seq_pad.
fn argmax_row(logits: &Witness, pos: usize, vocab: usize) -> usize {
    let mut best = 0usize;
    let mut best_val = i128::MIN;
    for v in 0..vocab {
        let val = f_to_int(logits.get(&[pos, v]));
        if val > best_val { best_val = val; best = v; }
    }
    best
}

/// Global argmax across vocab shards for logits row `pos`. Each
/// `logits_edges[k]` is shape `(seq, shard_ranges[k].1)`; returns the global
/// vocab id `start + local_offset`. With one shard this equals `argmax_row`.
fn argmax_row_sharded(
    witnesses: &[Vec<Witness>],
    logits_edges: &[EdgeId],
    shard_ranges: &[(usize, usize)],
    pos: usize,
) -> usize {
    let mut best_tok = 0usize;
    let mut best_val = i128::MIN;
    for (k, &(start, len)) in shard_ranges.iter().enumerate() {
        let lg = &witnesses[logits_edges[k]][0];
        for off in 0..len {
            let val = f_to_int(lg.get(&[pos, off]));
            if val > best_val { best_val = val; best_tok = start + off; }
        }
    }
    best_tok
}

/// A `(len, hidden)` row-block of the tied embedding matrix, for use as an
/// un-tied LM-head weight shard. `embedding_matrix` rows are identical (value
/// depends only on the hidden coord), so `start` only sets the global token
/// offset; the values match the tied head exactly.
fn embedding_matrix_shard(_vocab: usize, hidden: usize, _start: usize, len: usize) -> Vec<AlmostGoldilocksField> {
    let lp = len.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(lp * hp);
    for d in 0..hp {
        let val = AlmostGoldilocksField(1000 + 10 * ((d % 16) as u64));
        for off in 0..len {
            data[off + d * lp] = val;
        }
    }
    data
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let vocab_size: usize = std::env::var("VOCAB_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    // VOCAB_SHARDS>1 splits the LM head + argmax range-check along the vocab
    // axis so each fold-tree leaf fits GPU memory at full vocab (50257/32000).
    let vocab_shards: usize = std::env::var("VOCAB_SHARDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let num_heads = 12;
    let head_dim = 64;
    let hidden_dim = num_heads * head_dim; // 768
    let ffn_dim = hidden_dim * 4;

    println!("=== One-Shot GPT-2 (full AR pipeline) on Almost-Goldilocks ===");
    println!("num_layers={} seq_len={} vocab={} max_num_vars={} partitions={}",
             num_layers, seq_len, vocab_size, max_num_vars, num_partitions);

    // Prompt tokens (random; SKIP-AR — we prove the circuit, not the AR shift).
    let mut rng = StdRng::seed_from_u64(42);
    let token_ids: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % vocab_size).collect();

    // ---- 1. Build the one-shot circuit ----
    let t0 = Instant::now();
    let weights = gen_gpt2_weights(num_layers, hidden_dim, ffn_dim);
    let w_e_witness = Witness::new(
        vec![vocab_size, hidden_dim], embedding_matrix(vocab_size, hidden_dim),
        DataType::Float, *SF_LOG, Role::Constant);
    let pos_pad = seq_len.next_power_of_two() * hidden_dim.next_power_of_two();
    let pos_witness = Witness::new(
        vec![seq_len, hidden_dim], zeros(pos_pad), DataType::Float, *SF_LOG, Role::Constant);

    let mut g = DagBuilder::new();
    let w_e = g.param(w_e_witness);
    let (h0, _emb_sel) = g.embedding_lookup(w_e, seq_len, vocab_size, &token_ids, Role::Constant);
    let h_pe = g.add_positional_encoding(h0, pos_witness);
    let h_in = g.change_shape(h_pe, vec![1, seq_len, hidden_dim]);
    let h_out = g.pipe(
        &[h_in],
        gpt_2_small(
            weights.0, weights.1, weights.2, weights.3, weights.4,
            weights.5, weights.6, weights.7, weights.8, weights.9,
            weights.10, weights.11, weights.12,
            weights.13, weights.14, weights.15,
            weights.16, weights.17,
            num_heads, head_dim, seq_len,
        ),
    )[0];
    let dummy = vec![0usize; seq_len];
    let (logits_edges, argmax_sels, shard_ranges): (Vec<EdgeId>, Vec<EdgeId>, Vec<(usize, usize)>) =
        if vocab_shards > 1 {
            let ranges = DagBuilder::vocab_shard_ranges(vocab_size, vocab_shards);
            let shard_vocabs: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
            let w_lm_ids: Vec<EdgeId> = ranges
                .iter()
                .map(|&(start, len)| {
                    let data = embedding_matrix_shard(vocab_size, hidden_dim, start, len);
                    g.param(Witness::new(vec![len, hidden_dim], data, DataType::Float, *SF_LOG, Role::Constant))
                })
                .collect();
            let logits_shards = g.lm_head_sharded(h_out, &w_lm_ids, seq_len, &shard_vocabs);
            let sels = g.argmax_check_sharded(&logits_shards, &ranges, seq_len, &dummy, Role::Constant);
            println!("Vocab-sharded head: {} shards (sizes {:?})", ranges.len(), shard_vocabs);
            (logits_shards, sels, ranges)
        } else {
            let logits = g.lm_head_weight_tied(h_out, w_e, seq_len, vocab_size);
            let sel = g.argmax_check(logits, seq_len, vocab_size, &dummy, Role::Constant);
            (vec![logits], vec![sel], vec![(0, vocab_size)])
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

    // ---- 2. Pass 1: forward with dummy argmax selector ----
    let t1 = Instant::now();
    dag.run(&mut witnesses, &[]);
    println!("Forward (pass 1): {:?}", t1.elapsed());

    if std::env::var("ONESHOT_DBG").is_ok() {
        let hi = &witnesses[h_in][0];
        let vals: Vec<i128> = (0..8).map(|d| f_to_int(hi.get(&[0, 0, d]))).collect();
        let mn = (0..hidden_dim).map(|d| f_to_int(hi.get(&[0, 0, d]))).min().unwrap();
        let mx = (0..hidden_dim).map(|d| f_to_int(hi.get(&[0, 0, d]))).max().unwrap();
        println!("  h_in[0,0,0..8]={:?} min={} max={}", vals, mn, mx);
    }

    // Read logits, compute true per-row argmax (the next-token ids).
    let next_token_ids: Vec<usize> = (0..seq_len)
        .map(|i| argmax_row_sharded(&witnesses, &logits_edges, &shard_ranges, i))
        .collect();
    println!("argmax(logits): {:?}", &next_token_ids);
    if std::env::var("ONESHOT_DBG").is_ok() {
        let lg = &witnesses[logits_edges[0]][0];
        let vw = shard_ranges[0].1;
        for s in 0..seq_len {
            let row: Vec<i128> = (0..vw.min(6)).map(|v| f_to_int(lg.get(&[s, v]))).collect();
            let maxv = (0..vw).map(|v| f_to_int(lg.get(&[s, v]))).max().unwrap();
            let minv = (0..vw).map(|v| f_to_int(lg.get(&[s, v]))).min().unwrap();
            println!("  logits[{}] first6={:?} min={} max={} argmax={}", s, row, minv, maxv, next_token_ids[s]);
        }
    }

    // ---- 3. Fix the argmax selector(s) to the true argmax + rerun ----
    if vocab_shards > 1 {
        for (k, &(start, len)) in shard_ranges.iter().enumerate() {
            let local: Vec<Option<usize>> = next_token_ids
                .iter()
                .map(|&t| if t >= start && t < start + len { Some(t - start) } else { None })
                .collect();
            let s = DagBuilder::build_sharded_one_hot_selector_witness(seq_len, len, &local, Role::Constant);
            witnesses[argmax_sels[k]] = vec![s];
        }
    } else {
        let fixed_sel = DagBuilder::build_one_hot_selector_witness(
            seq_len, vocab_size, &next_token_ids, Role::Constant);
        witnesses[argmax_sels[0]] = vec![fixed_sel];
    }
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[]);
    println!("Forward (pass 2, fixed argmax): {:?}", t2.elapsed());

    // ---- 4. Commit ----
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
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

    // ---- 5. Prove + verify (single one-shot proof) ----
    let mut tp = Transcript::new(b"oneshot-gpt2");
    let t4 = Instant::now();
    let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut tp);
    println!("Prove: {:?}", t4.elapsed());

    let mut tv = Transcript::new(b"oneshot-gpt2");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut tv);
    println!("Verify: {:?}", t5.elapsed());
    let n_def = fp.deferred_constant_claims.len();
    println!("\nVerified{}: {}",
        if n_def > 0 { format!(" (modulo {} deferred constants)", n_def) } else { String::new() },
        verified);
    assert!(verified, "one-shot GPT-2 proof failed to verify");
}
