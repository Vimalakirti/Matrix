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
use rayon::prelude::*;
use zk_torch_4::morpheus::{coins::{Coins, CoinProfile}, params::Profile, piop::{self, layout::Layout, Ctx, Graph, ProverData, Walk, Config}, transcript::{MTranscript, Item}};
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
    let ev = |k: &str, d: usize| -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) };
    let num_layers = ev("NUM_LAYERS", 1);
    let seq_len = ev("SEQ_LEN", 4);
    let vocab_size = ev("VOCAB_SIZE", 256);
    let num_heads = ev("NUM_HEADS", 12); let head_dim = ev("HEAD_DIM", 64);
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
    let t = Instant::now();
    dag.run(&mut witnesses, &[]);
    let next: Vec<usize> = (0..seq_len).map(|i| argmax_row(&witnesses[logits][0], i, vocab_size)).collect();
    witnesses[sel] = vec![DagBuilder::build_one_hot_selector_witness(seq_len, vocab_size, &next, Role::Input)];
    dag.run(&mut witnesses, &[]);
    println!("graph: {} nodes, {} edges; witness {:.2}s", dag.nodes.len(), dag.num_edges(), t.elapsed().as_secs_f64());

    let profile = match std::env::var("PROFILE").ok().as_deref() { Some("ternary") => Profile::ternary(), Some("mini") => Profile::mini(), _ => Profile::toy() };
    let b0 = ev("B0", 10);
    let cfg = Config { tcl: ev("TCL", 4), slots_log: ev("SLOTS_LOG", 14), ..Config::default() };
    let slots_log = cfg.slots_log;
    let mut graph = Graph::new(&dag, &witnesses, cfg);
    for &e in &[emb_sel, sel] { graph.public.insert(e, piop::witness_vals(&witnesses[e][0])); }
    let coins = Coins::new([7u8; 64], CoinProfile::Ternary);
    let t = Instant::now();
    let bank = piop::sample_bank(&coins, 0, 1usize << slots_log);
    let t_bank = t.elapsed().as_secs_f64();
    let pd = ProverData::new(&graph, &witnesses, bank);
    let t_pd = t.elapsed().as_secs_f64();
    let specs = piop::tensor_specs(&graph, &pd);
    let t_specs = t.elapsed().as_secs_f64();
    let lay = Layout::pack_min(&profile, &specs, b0);
    if std::env::var("MORPHEUS_TIMING").is_ok() { eprintln!("[prep] bank {t_bank:.2}s, prover data {:.2}s, specs {:.2}s, layout {:.2}s", t_pd - t_bank, t_specs - t_pd, t.elapsed().as_secs_f64() - t_specs); }
    {
        use zk_torch_4::morpheus::piop::layout::TKey;
        let mut by: std::collections::BTreeMap<String, (usize, u128, u128)> = Default::default();
        for &(k, n, p, bits) in &specs {
            let kind = match k { TKey::Edge(e) => format!("edge/{:?}{}", witnesses[e][0].role, if bits { "/bits" } else { "" }), TKey::Sel { node, .. } => format!("sel/{}", format!("{:?}", dag.nodes[node].kind).split(|c: char| c == '(' || c == ' ').next().unwrap()), TKey::Wg => "wg".into(), TKey::Carry => "carry".into() };
            let e = by.entry(kind).or_default();
            e.0 += 1; e.1 += 1u128 << n; e.2 += 1u128 << (n + p);
        }
        for (k, v) in by { println!("  {k:28} {:6} tensors {:>14} indices {:>14} digit slots", v.0, v.1, v.2); }
    }
    println!("layout: {} tensors -> {} sources; bank+prep {:.2}s", lay.tensors.len(), lay.n_sources, t.elapsed().as_secs_f64());
    if std::env::var("LAYOUT_ONLY").is_ok() { return; }

    use zk_torch_4::morpheus::{link, link_fast, link_gpu, pcs_gpu::PcsGpu, keys::KeySeed, auth, auth_gpu};
    let seed = KeySeed([1; 8]);
    let full = std::env::var("PCS").is_ok();
    let mut pcs = if full { let t = Instant::now(); let p = PcsGpu::new(profile.clone(), seed); println!("key caches {:.2}s", t.elapsed().as_secs_f64()); Some(p) } else { None };
    // Sources and their commitments come first: they are fixed before any graph challenge.
    // Edge values go to the device in the background while the sources are
    // filled and committed.
    let prefetch = std::env::var("MORPHEUS_NO_PREFETCH").is_err();
    let (raw, replica_jobs, c_src) = std::thread::scope(|sc| {
    if prefetch { let pdr = &pd; sc.spawn(move || pdr.prefetch()); }
    let t = Instant::now();
    let mut raw = piop::fill_sources(&graph, &pd, &lay, profile.ell());
    for (e, r) in raw.iter_mut().enumerate() { link::fill_source_suffix_gpu(&profile, &coins, e as u64, r); }
    let nz: usize = raw.par_iter().map(|r| r.iter().filter(|&&d| d != 0).count()).sum();
    println!("sources: {} x 2^{}, {} nonzero digits; fill {:.2}s", raw.len(), profile.ell(), nz, t.elapsed().as_secs_f64());
    let raw = std::sync::Arc::new(raw);
    // Authentication replicas (key caches + physical sources) on devices
    // 1..MORPHEUS_AUTH_GPUS, built in the background while the PIOP and the
    // link run.
    let auth_gpus: usize = std::env::var("MORPHEUS_AUTH_GPUS").or_else(|_| std::env::var("MORPHEUS_GPUS")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let replica_jobs: Vec<std::thread::JoinHandle<zk_torch_4::morpheus::pcs_gpu::AuthReplica>> =
        if full && std::env::var("AUTH").is_ok() && auth_gpus > 1 {
            (1..auth_gpus).map(|g| {
                let (raw2, prof2) = (raw.clone(), profile.clone());
                std::thread::spawn(move || zk_torch_4::morpheus::pcs_gpu::AuthReplica::build(g as i32, &prof2, &seed, &raw2))
            }).collect()
        } else { Vec::new() };
    let t = Instant::now();
    let c_src: Vec<Vec<Vec<u64>>> = match pcs.as_mut() {
        Some(pc) => { pc.load_sources(&raw); pc.commit_sources() }
        None => raw.iter().map(|r| auth::lin(&profile, &seed, zk_torch_4::morpheus::keys::KeyKind::Source, profile.source_payload_cols, &link::raw_to_cols(r))).collect(),
    };
    println!("source commitments {:.2}s", t.elapsed().as_secs_f64());
    (raw, replica_jobs, c_src)
    });
    let salt = coins.salt(0);
    let mut tr = MTranscript::new(profile.name, &salt);
    tr.absorb(Item::Statement, &[lay.n_sources as u64]);
    for c in &c_src { for x in c { tr.absorb(Item::SourceCommitment, x); } }

    let t = Instant::now();
    let mut ctx = Ctx::prover(tr.clone(), &pd.bank, &lay, slots_log);
    let mut walk = Walk::new(&graph, Some(&pd));
    walk.run(&mut ctx);
    let tp = t.elapsed().as_secs_f64();
    println!("PIOP prove {:.2}s: {} messages, {} rows, {} slots", tp, ctx.proof.len(), ctx.rows.len(), ctx.slots_used());
    for (k, v) in &ctx.stats { println!("  {k:12} {:6} calls {:.3}s", v.0, v.1); }
    { let u = zk_torch_4::morpheus::piop::gpu::UPLOAD.lock().unwrap(); println!("  edge uploads {} ({:.2} GB) {:.3}s", u.0, u.1 as f64 / 1e9, u.2); }
    let t = Instant::now();
    let mut vctx = Ctx::verifier(tr.clone(), ctx.proof.clone(), &lay, slots_log);
    let mut vwalk = Walk::new(&graph, None);
    vwalk.run(&mut vctx);
    println!("PIOP verify {:.2}s ok={} fail={:?} rows {}", t.elapsed().as_secs_f64(), vctx.ok, vctx.fail, vctx.rows.len());
    assert!(vctx.ok);
    assert_eq!(vctx.rows.len(), ctx.rows.len());
    if std::env::var("TAMPER").is_ok() {
        let mut bad = ctx.proof.clone();
        let k = bad.len() / 2; bad[k] = bad[k].add(&zk_torch_4::morpheus::field8::K8::ONE);
        let mut bctx = Ctx::verifier(tr.clone(), bad, &lay, slots_log);
        let mut bwalk = Walk::new(&graph, None);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bwalk.run(&mut bctx)));
        println!("tampered proof: ok={} ({:?})", res.is_ok() && bctx.ok, bctx.fail);
    }
    if std::env::var("LINK_CHECK").is_err() { return; }
    let st = link::Statement { n: lay.n_sources, bits: lay.bit_blocks(), rows: ctx.rows.clone() };
    let t = Instant::now();
    let mut t1 = ctx.tr.clone();
    let fast = match pcs.as_mut() {
        Some(pc) => link_fast::prove_with(&profile, &seed, &coins, 0, &mut t1, &st, &raw, b0, &link_gpu::gpu_backend, pc, Some(&c_src), std::env::var("MORPHEUS_LINK_V2").ok().as_deref() == Some("1")),
        None => link_fast::prove_with(&profile, &seed, &coins, 0, &mut t1, &st, &raw, b0, &link_gpu::gpu_backend, &mut link_fast::CpuServices, Some(&c_src), std::env::var("MORPHEUS_LINK_V2").ok().as_deref() == Some("1")),
    };
    println!("link prove {:.2}s", t.elapsed().as_secs_f64());
    if std::env::var("LINK_CMP").is_ok() {
        let mut t3 = ctx.tr.clone();
        let cpu = link_fast::prove(&profile, &seed, &coins, 0, &mut t3, &st, &raw, b0);
        println!("GPU path == CPU structured link: {}", cpu.proof.rounds == fast.proof.rounds && cpu.proof.a_hat == fast.proof.a_hat && cpu.proof.g_commit == fast.proof.g_commit);
    }
    let mut tv = vctx.tr.clone();
    let t = Instant::now();
    let lv = link::verify(&profile, &mut tv, &st, &c_src, &fast.proof);
    println!("link verifies: {} ({:.2}s)", lv.is_some(), t.elapsed().as_secs_f64());
    if std::env::var("AUTH").is_ok() {
        use almost_goldilocks_cuda::DeviceBuffer;
        let t = Instant::now();
        let owned;
        let r_cols: Vec<Vec<Vec<i64>>> = fast.sources.iter().map(|s| s.r.clone()).collect();
        let to_dev = |cols: &[Vec<i64>]| DeviceBuffer::from_slice(&cols.iter().flatten().map(|&v| v as i8).collect::<Vec<_>>()).unwrap();
        let dsrc: Vec<auth_gpu::DevSource> = match pcs.as_mut() {
            Some(pc) => pc.dev_sources(&r_cols),
            None => raw.iter().zip(&r_cols).map(|(rw, r)| auth_gpu::DevSource { w: to_dev(&link::raw_to_cols(rw)), r: to_dev(r) }).collect(),
        };
        let keys = match pcs.as_ref() { Some(pc) => &pc.keys, None => { owned = auth_gpu::Keys::build(&profile, &seed); &owned } };
        let mut tm = auth_gpu::AuthTiming::default();
        let tw = Instant::now();
        let mut reps: Vec<_> = replica_jobs.into_iter().map(|h| h.join().expect("replica")).collect();
        let rsrc: Vec<Vec<auth_gpu::DevSource>> = reps.iter_mut().map(|r| r.sources(&r_cols)).collect();
        let s_os: Vec<DeviceBuffer<i8>> = reps.iter().map(|r| { almost_goldilocks_cuda::set_device(r.dev).unwrap(); to_dev(&fast.s_o) }).collect();
        almost_goldilocks_cuda::set_device(0).unwrap();
        if !reps.is_empty() { println!("auth replicas ready after {:.2}s wait", tw.elapsed().as_secs_f64()); }
        let s_o0 = to_dev(&fast.s_o);
        let mut devs = vec![auth_gpu::AuthDev { dev: 0, keys, sources: &dsrc, s_o: &s_o0, runs: None }];
        for (j, r) in reps.iter().enumerate() { devs.push(auth_gpu::AuthDev { dev: r.dev, keys: &r.keys, sources: &rsrc[j], s_o: &s_os[j], runs: None }); }
        let (apf, _) = auth_gpu::prove_multi(&profile, &devs, &coins, 0, &t1, &fast.point, std::env::var("NO_AUTH_VERIFY").is_err(), &mut tm).expect("auth");
        println!("auth (GPU) {:.2}s, attempt {} {:?}", t.elapsed().as_secs_f64(), apf.attempt, tm);
        if std::env::var("NO_AUTH_VERIFY").is_err() {
            let (pt, targets, target_o) = lv.expect("link");
            let t = Instant::now();
            let ok = auth_gpu::verify(&profile, keys, &tv, &pt, &targets, &target_o, &apf);
            println!("auth verifies (GPU): {ok} ({:.2}s)", t.elapsed().as_secs_f64());
            if std::env::var("TAMPER").is_ok() {
                let mut bad = apf.clone();
                bad.z.w[0] ^= 1;
                let mut bad2 = apf.clone();
                bad2.d[1].a.0[0] = bad2.d[1].a.0[0].add(&zk_torch_4::morpheus::field8::K8::ONE);
                println!("tampered auth proofs rejected: z {}, first message {}",
                    !auth_gpu::verify(&profile, keys, &tv, &pt, &targets, &target_o, &bad),
                    !auth_gpu::verify(&profile, keys, &tv, &pt, &targets, &target_o, &bad2));
            }
            if std::env::var("AUTH_VERIFY_CPU").is_ok() {
                let t = Instant::now();
                let ok = auth::verify(&profile, &seed, &tv, &pt, &targets, &target_o, &apf);
                println!("auth verifies (CPU reference): {ok} ({:.2}s)", t.elapsed().as_secs_f64());
            }
        }
    }
}
