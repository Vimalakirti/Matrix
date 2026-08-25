//! Streaming nanoGPT, the exact model EZKL ships in `examples/onnx/nanoGPT`,
//! composed with the cross-proof streaming accumulator.
//!
//! block_size 64, vocab 65, n_layer 4, n_head 4, n_embd 64, bias False. Each
//! streamed proof is one forward pass over 64 token ids producing logits for
//! every position, which is what EZKL's exported graph proves. WEIGHTS are
//! Role::Constant (deferred -> amortized into one finalize opening) and the
//! per-request one-hot embedding SELECTOR is Role::Input, committed and opened
//! per proof.
//!
//! Run with `bench_config.yaml` as args[1]. Env: N_PROOFS(2) MAX_NUM_VARS(22)
//! NUM_PARTITIONS(1) SEQ_LEN(1) ZK4_B(21) ZK4_BASE(2).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::nanogpt::{nanogpt, NanoGptBlockWeights,
    NANOGPT_N_EMBD, NANOGPT_N_LAYER, NANOGPT_VOCAB};
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

/// Weights are ZERO, as every other transformer bench here does. Prover cost
/// depends on the graph's shapes, not on the values, and random weights push
/// the LayerNorm and softmax accumulators outside the range table: a first
/// version used small random values and the NonNegative check reported 64/64
/// entries out of range, then reported them NEGATIVE once the table was
/// widened, because near-zero random weights make the normalization
/// degenerate rather than merely large.
fn w(shape: Vec<usize>) -> Witness {
    let size: usize = shape.iter().map(|d| d.next_power_of_two()).product();
    Witness::new(shape, zk_torch_4::zero_witness_vec(size),
                 DataType::Float, *SF_LOG, Role::Constant)
}

/// The embedding is the one weight that must be nonzero: it is the only source
/// of signal in the graph, and an all-zero embedding leaves LayerNorm dividing
/// by a zero variance.
///
/// The spread across the hidden axis is set by `EMBD_STEP`, and it is not free
/// to choose. `llama_rms_norm` gates the reciprocal advice `r` on
///   |mean(x^2) * r^2 - 1| <= 2 / 2^SF_LOG,
/// a 0.2% band. `r` is computed in f64 from the un-rescaled input, but
/// `mean(x^2)` reaches the gate as a fixed-point value that has been rescaled
/// by 2^SF_LOG, so its quantisation error is one LSB out of `mean(x^2)`
/// itself. The gate therefore needs mean(x^2) >= 2^(SF_LOG-1) = 512 to have
/// any chance of holding. GPT-2 clears this only because hidden=768 with a
/// deviation spread of +/-433 lands mean(x^2) at ~511; the shared bench value
/// `1000 + 10*(d%16)` gives a spread of only +/-75, and at nanoGPT's hidden=64
/// that quantises mean(x^2) all the way down to the integer 2, where one LSB
/// is a 50% error and every one of the 9 LayerNorms fails the gate.
///
/// A step of 256 fixes both halves at once. Deviations become
/// 256*(d%16 - 7.5) = +/-1920, so mean(x^2) = 1360 sits well clear of 512,
/// and every deviation is a multiple of 32 so every square is an exact
/// multiple of 2^SF_LOG. The rescale is then lossless and the gate sees
/// mean(x^2)*r^2 = 1 exactly rather than merely within tolerance.
const EMBD_STEP: u64 = 256;
const EMBD_BASE: u64 = 2048;

fn embedding(vocab: usize, hidden: usize) -> Witness {
    let (vp, hp) = (vocab.next_power_of_two(), hidden.next_power_of_two());
    let mut data = zk_torch_4::zero_witness_vec(vp * hp);
    for d in 0..hp {
        let val = AlmostGoldilocksField(EMBD_BASE + EMBD_STEP * ((d % 16) as u64));
        for v in 0..vocab { data[v + d * vp] = val; }
    }
    Witness::new(vec![vocab, hidden], data, DataType::Float, *SF_LOG, Role::Constant)
}

/// bias = False in EZKL's config, so every bias is all zero. An all-zero bias
/// is arithmetically identical to having none and lets the shared gpt2_block
/// be reused unchanged.
fn zero(shape: Vec<usize>) -> Witness {
    let size: usize = shape.iter().map(|d| d.next_power_of_two()).product();
    Witness::new(shape, zk_torch_4::zero_witness_vec(size),
                 DataType::Float, *SF_LOG, Role::Constant)
}

fn demo_seed() -> Seed {
    Seed([0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
          0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let seq_len = env("SEQ_LEN", 1);
    let n_proofs = env("N_PROOFS", 2);
    let max_num_vars = env("MAX_NUM_VARS", 22);
    let num_partitions = env("NUM_PARTITIONS", 1);
    let b = env("ZK4_B", 21);
    let base = env("ZK4_BASE", 2);
    let d = NANOGPT_N_EMBD;

    println!("=== Streaming nanoGPT (EZKL examples/onnx/nanoGPT) ===");
    println!("n_layer={} n_head=4 n_embd={} vocab={} seq_len={} N_PROOFS={} partitions={}",
             NANOGPT_N_LAYER, d, NANOGPT_VOCAB, seq_len, n_proofs, num_partitions);

    let mut rng = rand::thread_rng();
    let mut tokens: Vec<usize> = (0..seq_len).map(|_| rng.gen::<usize>() % NANOGPT_VOCAB).collect();

    let blocks: Vec<NanoGptBlockWeights> = (0..NANOGPT_N_LAYER).map(|_| NanoGptBlockWeights {
        ln1_w: w(vec![d]),
        // c_attn is one fused [64, 192] in the ONNX; Split gives these three.
        q_w: w(vec![d, d]), k_w: w(vec![d, d]), v_w: w(vec![d, d]), o_w: w(vec![d, d]),
        ln1_b: zero(vec![d]),
        q_b: zero(vec![d]), k_b: zero(vec![d]), v_b: zero(vec![d]), o_b: zero(vec![d]),
        ln2_w: w(vec![d]),
        fc_w: w(vec![d, 4 * d]), proj_w: w(vec![4 * d, d]),
        ln2_b: zero(vec![d]), fc_b: zero(vec![4 * d]), proj_b: zero(vec![d]),
    }).collect();

    let mut g = DagBuilder::new();
    let _out = g.pipe(&[], nanogpt(
        embedding(NANOGPT_VOCAB, d),    // wte, also the tied lm_head
        w(vec![seq_len, d]),            // wpe
        blocks,
        w(vec![d]), zero(vec![d]),      // ln_f
        seq_len, tokens.clone(),
    ));
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    // The embedding selector is the only per-request input; it must never be
    // deferred as a shared weight.
    let sel: Vec<usize> = (0..dag.num_edges())
        .filter(|&e| witnesses_template[e].first().map(|x| x.role) == Some(Role::Input))
        .collect();
    assert_eq!(sel.len(), 1, "expected exactly one Role::Input edge (the embedding selector)");
    let emb_sel = sel[0];

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses_template, &mut store);
    println!("Offline commit (weights, amortized): {:.2}ms", ms(t_off.elapsed()));

    let label = b"zkml-nanogpt-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} forward passes (seq_len={}):", n_proofs, seq_len);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        // Fresh tokens per request -> fresh embedding selector.
        tokens = (0..seq_len).map(|_| rng.gen::<usize>() % NANOGPT_VOCAB).collect();
        witnesses[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(
            seq_len, NANOGPT_VOCAB, &tokens, Role::Input)];

        let s0 = Instant::now(); dag.run(&mut witnesses, &[]);
        let d_run = s0.elapsed(); t_run += d_run;

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now(); dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-req");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-req");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-request verify failed at {}", it); return; }
        for dc in &r.claims {
            assert!(dc.edge_id != emb_sel,
                    "embedding selector was deferred as a shared weight -- unsound");
        }

        let s4 = Instant::now();
        let chunk = prover_acc.add_proof(&r, &witnesses);
        let d_acc = s4.elapsed(); t_acc += d_acc;
        proof_bytes += ser_len(&dp) + ser_len(&fp) + ser_len(&chunk);
        breakdown = Some(zk_torch_4::proof_size_report(
            &dp.node_proofs, &dp.edge_proofs, &dp.range_proof,
            &dp.two_pow_proof, &dp.output_claims, &dp, &fp));
        let s5 = Instant::now();
        let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
        let d_accv = s5.elapsed(); t_accv += d_accv;
        if !ok { eprintln!("streaming verifier rejected at {}", it); return; }

        // Device-wide used bytes, the same probe bench_streaming_llama2 prints.
        // It is whole-device rather than per-process, so it is only meaningful
        // on a GPU this run has to itself; nvidia-smi reports 0 for these
        // binaries, so an in-process probe is the only thing that works.
        let gpu_mem = almost_goldilocks_cuda::mem_get_info()
            .map(|(free, total)| (total - free) / (1024 * 1024))
            .unwrap_or(0);
        println!("  [{:>2}/{}] run {:>7.1}ms commit {:>6.1}ms prove {:>7.1}ms verify {:>6.1}ms acc {:>7.1}ms acc-v {:>6.1}ms gpu-mem {:>6} MiB",
            it + 1, n_proofs, ms(d_run), ms(d_commit), ms(d_prove), ms(d_verify), ms(d_acc), ms(d_accv), gpu_mem);
    }

    let n_steps = prover_acc.num_steps();
    let n_const = prover_acc.num_edges();
    let s_fp = Instant::now();
    let final_proof = prover_acc.finalize(&witnesses_template, &store);
    let t_finalize = s_fp.elapsed();
    let s_fv = Instant::now();
    let ok = verifier_acc.verify_finalize(&store, &final_proof);
    let t_fv = s_fv.elapsed();
    if !ok { eprintln!("verify_finalize REJECTED -- soundness chain broken"); return; }

    let n = n_proofs as f64;
    println!("\n=== Results ({} weight edges deferred, {} reducer steps) ===", n_const, n_steps);
    println!("  prove(defer)  per-req : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-req : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  = streaming per-req   : {:>8.2}ms",
             ms(t_prove) / n + ms(t_acc) / n + ms(t_finalize) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(bk) = &breakdown { println!("{}", bk); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    let _ = (t_run, t_commit, t_verify, t_accv);
    println!("\nVerified: true (nanoGPT, weights amortized across {} forward passes)", n_proofs);
}
