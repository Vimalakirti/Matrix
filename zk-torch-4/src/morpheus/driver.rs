//! Morpheus (S&P'27) driver: N requests of one model instance, monolithic or
//! with deferred fixed-weight opening, each verified; prints per-request and
//! amortized prover cost, verifier cost and proof size.
//!
//! Env: N_REQUESTS (1), DEFER (0|1), TAMPER, LAYOUT_ONLY, plus
//! request::Opts::from_env (PROFILE, B0, TCL, SLOTS_LOG, CARRY_LOG,
//! MORPHEUS_GPUS, MORPHEUS_AUTH_GPUS, NO_AUTH, NO_VERIFY, MORPHEUS_TIMING).
//! Requests reuse the instance's witness with fresh prover randomness.

use crate::dag::{Dag, Role, Witness};
use crate::morpheus::{pcs_gpu::PcsGpu, request::{self, Opts, QueueEntry, Replicas, Times}};
use std::time::Instant;

fn ev(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

fn show(label: &str, t: &Times) -> f64 {
    let tot: f64 = t.iter().filter(|(k, _)| !k.contains("extraction")).map(|(_, v)| v).sum();
    let parts: Vec<String> = t.iter().map(|(k, v)| format!("{k} {v:.2}")).collect();
    println!("{label}: {tot:.2}s  [{}]", parts.join(", "));
    tot
}

/// Entry from `Dag::run` (MORPHEUS_AT_RUN): every leaf Role::Input edge is
/// a public request input. MORPHEUS_NAME / MORPHEUS_UNITS / MORPHEUS_UNIT
/// label the report.
pub fn run_hook(dag: &Dag, wit: &mut [Vec<Witness>]) {
    let public: Vec<usize> = (0..dag.num_edges()).filter(|&e| dag.producers[e].is_none() && wit[e].first().map(|w| w.role == Role::Input && w.data.is_some()).unwrap_or(false)).collect();
    let name = std::env::var("MORPHEUS_NAME").unwrap_or_else(|_| "model".into());
    let unit: &'static str = Box::leak(std::env::var("MORPHEUS_UNIT").unwrap_or_else(|_| "request".into()).into_boxed_str());
    println!("{name}: {} nodes, {} edges (from the benchmark's forward pass); public inputs {:?}", dag.nodes.len(), dag.num_edges(), public);
    run(&name, dag, wit, &public, ev("MORPHEUS_UNITS", 1), unit);
}

pub fn run(name: &str, dag: &Dag, wit: &mut [Vec<Witness>], public: &[usize], units: usize, unit: &str) {
    run_with_ar(name, dag, wit, public, units, unit, None);
}

pub fn run_with_ar(name: &str, dag: &Dag, wit: &mut [Vec<Witness>], public: &[usize], units: usize, unit: &str, ar: Option<super::ar_statement::ArStatement>) {
    let n = ev("N_REQUESTS", 1);
    let mut o = Opts::from_env();
    let mut req_secs = Vec::new();
    let mut verify_secs = Vec::new();
    let n = if o_defer_batches().is_some() { batch_sizes(n).iter().sum() } else { n };
    println!("MORPHEUS_POLICY protocol={} profile={} auth={} verify={} requests={}", crate::morpheus::wire::PROTOCOL_VERSION, o.profile.name, o.auth, o.verify, n);
    let mut outs = Vec::new();
    let mut vqueue = Vec::new();
    let mut model_reg = None;
    let mut model_secs = 0.0;
    // One preparation per witness (per request with FRESH_INPUTS).
    let t = Instant::now();
    let mut pp = request::prepare(dag, wit, public, &mut o);
    pp.ar = ar;
    if let Some(ar) = &pp.ar { assert!(ar.validate(&pp.g), "invalid AR public statement"); }
    let prep = t.elapsed().as_secs_f64();
    println!("prep {:.2}s: {} tensors, sources model {:?} request {:?} carry {:?}", prep, pp.lay.tensors.len(), pp.model, pp.req, pp.carry);
    {
        use crate::morpheus::piop::layout::TKey;
        let mut by: std::collections::BTreeMap<String, (usize, u128, u128)> = Default::default();
        for &(k, nv, p, bits) in &pp.specs {
            let kind = match k {
                TKey::Edge(e) => format!("edge/{:?}{}", wit[e][0].role, if bits { "/bits" } else { "" }),
                TKey::Sel { node, .. } => format!("sel/{}", format!("{:?}", dag.nodes[node].kind).split(|c: char| c == '(' || c == ' ').next().unwrap()),
                TKey::Wg => "wg".into(), TKey::Carry => "carry".into(),
            };
            let e = by.entry(kind).or_default();
            e.0 += 1; e.1 += 1u128 << nv; e.2 += 1u128 << (nv + p);
        }
        for (k, v) in by { println!("  {k:28} {:6} tensors {:>14} indices {:>14} digit slots", v.0, v.1, v.2); }
    }
    // The prover data holds every value the prover needs; the forward pass's
    // own copies are released (MORPHEUS_KEEP_WITNESS=1 keeps them).
    if std::env::var_os("MORPHEUS_KEEP_WITNESS").is_none() {
        use rayon::prelude::*;
        wit.par_iter_mut().for_each(|w| for x in w.iter_mut() { x.data = None; });
        if std::env::var_os("MORPHEUS_RSS").is_some() { eprintln!("[rss] witnesses released: {:.1} GB", std::fs::read_to_string("/proc/self/statm").ok().and_then(|m| m.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok())).unwrap_or(0.0) * 4096.0 / 1e9); }
    }
    if let Some(s) = crate::morpheus::request::request_support(&pp, &o) {
        println!("public response support: {} of {} source columns", s.len(), o.profile.source_cols());
    }
    // MORPHEUS_LOOKUPS=1: every lookup node's checked size and its input's producer.
    if std::env::var_os("MORPHEUS_LOOKUPS").is_some() {
        let mut by: std::collections::BTreeMap<String, (usize, u128)> = Default::default();
        for (&node, _) in pp.g.lookups.iter() {
            let nd = &dag.nodes[node];
            let e = nd.inputs[0];
            let prod = dag.producers[e].map(|p| format!("{:?}", dag.nodes[p].kind).split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap().to_string()).unwrap_or("-".into());
            let k = format!("{} <- {} {:?}", format!("{:?}", nd.kind).split(|c: char| c == '(' || c == ' ').next().unwrap(), prod, wit[e][0].shape);
            let x = by.entry(k).or_default(); x.0 += 1; x.1 += 1u128 << pp.g.nvars[e];
        }
        let mut v: Vec<_> = by.into_iter().collect(); v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        for (k, (c, n)) in v.iter().take(25) { println!("  lookup {c:5} x {k:70} {n:>13} indices"); }
    }
    if std::env::var("LAYOUT_ONLY").is_ok() { return; }
    let t = Instant::now();
    let mut pc = PcsGpu::new(o.profile.clone(), o.seed);
    // Streamed requests use every device directly (no replicas).
    let stream = o.streams(pp.input.len() + pp.req.len() + if o.defer { 1 } else { pp.model.len() });
    let mut reps = if o.auth && o.auth_gpus > 1 && !stream { Replicas::new(&o) } else { Replicas { r: Vec::new() } };
    if stream { println!("streamed sources ({} request sources)", pp.req.len()); }
    println!("key caches {:.2}s ({} GPU(s) for authentication)", t.elapsed().as_secs_f64(), 1 + reps.r.len());
    {
        let m = request::register(&pp, &o, &mut pc, &reps);
        println!("model registration (one-time): {} sources committed in {:.2}s", m.source_count(), m.secs);
        model_secs = m.secs;
        model_reg = Some(m);
    }
    let input_reg = request::register_inputs(&pp, &o, &mut pc);
    println!("private input registration (one-time): {} sources committed in {:.2}s", input_reg.source_count(), input_reg.secs);
    // Deferred: each batch's finalization transcript absorbs its requests'
    // literal prefixes on a host thread while later requests are proven
    // (MORPHEUS_FINAL_EAGER=1 absorbs them at finalization instead).
    type PrefixMsg = (usize, u64, Vec<u8>, crate::morpheus::field8::K8, Vec<Vec<u64>>);
    let mut pre_tx: Option<std::sync::mpsc::Sender<PrefixMsg>> = None;
    let mut pre_rx: Option<std::sync::mpsc::Receiver<(usize, request::FinalPrefix, f64)>> = None;
    let mut hasher = None;
    let mut batch_of: Vec<usize> = Vec::new();
    if o.defer && std::env::var_os("MORPHEUS_FINAL_EAGER").is_none() {
        let m = model_reg.as_ref().unwrap();
        let sizes = batch_sizes(n);
        let mut pres = Vec::new();
        for (b, &jb) in sizes.iter().enumerate() { pres.push(Some(request::FinalPrefix::begin(&pp, &o, b as u64, m, jb))); batch_of.extend(std::iter::repeat(b).take(jb)); }
        let (tx, rx) = std::sync::mpsc::channel::<PrefixMsg>();
        let (btx, brx) = std::sync::mpsc::channel();
        hasher = Some(std::thread::spawn(move || {
            for (b, j, lp, dg, cc) in rx {
                let t = Instant::now();
                let p: &mut request::FinalPrefix = pres[b].as_mut().expect("batch already complete");
                p.push(j, &lp, &dg, &cc);
                drop(lp);
                let s = t.elapsed().as_secs_f64();
                if p.complete() { btx.send((b, pres[b].take().unwrap(), s)).unwrap(); }
            }
        }));
        pre_tx = Some(tx); pre_rx = Some(brx);
    }
    for j in 0..n as u64 {
        let mut out = request::prove_request(&mut pp, &o, j, model_reg.as_ref(), &input_reg, &mut pc, &mut reps);
        if let (Some(tx), Some(&b)) = (&pre_tx, batch_of.get(j as usize)) {
            tx.send((b, j, std::mem::take(&mut out.literal_prefix), out.digest, out.c_carry.clone().expect("deferred request carry"))).unwrap();
        }
        out.times.insert("prep", prep);
        let s = show(&format!("request {j} prove"), &out.times);
        if std::env::var_os("MORPHEUS_PROOF_HASH").is_some() { println!("request {j} proof hash {}", proof_hash(&out.bytes)); }
        println!("  {} PIOP rows, {} carries, {} PIOP messages, {} proof bytes", out.rows, out.carries, out.piop_len, out.bytes.len());
        req_secs.push(s);
        if o.verify {
            // The verifier sees only the serialized proof.
            let mut vt = Times::new();
            let t = Instant::now();
            let pf = crate::morpheus::wire::decode_request_compact(&o.profile, &request::request_support(&pp, &o).expect("public support"), &out.bytes).expect("proof decodes");
            println!("MORPHEUS_AUTH request={j} present={}", pf.auth.is_some());
            assert!(pf.auth.is_some(), "benchmark request lacks authentication");
            vt.insert("decode", t.elapsed().as_secs_f64());
            if j == 0 {
                let parts = crate::morpheus::wire::parts(&o.profile, &pf);
                println!("  proof parts (bytes): {}", parts.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "));
            }
            // The verifier's source key cache (setup, untimed): the prover
            // may have freed it for the link.
            pc.ensure_w();
            let r = request::verify_request(&pp, &o, j, model_reg.as_ref().map(|m| m.public()), input_reg.public(), &pf, &pc.keys, &mut vt);
            let vs = show(&format!("request {j} verify ({})", r.is_some()), &vt);
            assert!(r.is_some(), "request {j} rejected");
            verify_secs.push(vs);
            let (rows, dg, literal_prefix) = r.unwrap();
            vqueue.push((rows, dg, if o.defer { pf.c_src.last().cloned() } else { None }, literal_prefix));
        }
        outs.push(out);
    }
    let req_bytes: Vec<f64> = outs.iter().map(|out| out.bytes.len() as f64).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    // Deferred batches, finalized in order over consecutive requests
    // (DEFER_BATCHES, e.g. "2,8"; default one batch of all requests).
    let mut batches: Vec<(usize, f64, f64, f64)> = Vec::new(); // (J, prove s, verify s, bytes)
    if o.defer {
        let m = model_reg.as_mut().unwrap();
        let mut start = 0usize;
        for (b, &jb) in batch_sizes(n).iter().enumerate() {
            let range = start..start + jb;
            start += jb;
            let carries: Vec<Vec<i8>> = outs[range.clone()].iter_mut().map(|o| o.carry_raw.take().unwrap()).collect();
            let q: Vec<QueueEntry> = range.clone().map(|j| QueueEntry { j: j as u64, deferred: &outs[j].deferred, c_carry: outs[j].c_carry.as_ref().unwrap(), digest: outs[j].digest, literal_prefix: &outs[j].literal_prefix }).collect();
            // The batch's last prefix is charged to the finalization in full,
            // as if finalization started right after its last request.
            let pre = pre_rx.as_ref().map(|rx| { let (bb, p, last) = rx.recv().expect("finalization prefix"); assert_eq!(bb, b); (p, last) });
            let fo = match pre {
                Some((p, last)) => { let mut fo = request::finalize_with(&pp, &o, b as u64, m, &q, carries, &mut pc, &mut reps, Some(p)); fo.times.insert("prefix", last); fo }
                None => request::finalize(&pp, &o, b as u64, m, &q, carries, &mut pc, &mut reps),
            };
            let fs = show(&format!("finalize batch {b} (J = {jb}, {} deferred rows, {} proof bytes)", fo.rows, fo.bytes.len()), &fo.times);
            if std::env::var_os("MORPHEUS_PROOF_HASH").is_some() { println!("finalize batch {b} proof hash {}", proof_hash(&fo.bytes)); }
            let mut fv = 0.0;
            if o.verify {
                let t = Instant::now();
                let fpf = crate::morpheus::wire::decode_final_compact(&o.profile, &request::final_support(&pp, &o).expect("public support"), &fo.bytes).expect("finalization proof decodes");
                println!("MORPHEUS_AUTH batch={b} present={}", fpf.auth.is_some());
                assert!(fpf.auth.is_some(), "benchmark finalization lacks authentication");
                let dec = t.elapsed().as_secs_f64();
                let vq: Vec<QueueEntry> = range.clone().map(|j| { let (rows, dg, cc, literal_prefix) = &vqueue[j]; QueueEntry { j: j as u64, deferred: rows, c_carry: cc.as_ref().unwrap(), digest: *dg, literal_prefix } }).collect();
                let mut vt = Times::new();
                vt.insert("decode", dec);
                pc.ensure_w();
                let ok = request::verify_final(&pp, &o, b as u64, m.public(), &vq, &fpf, &pc.keys, &mut vt);
                fv = show(&format!("finalize batch {b} verify ({ok})"), &vt);
                assert!(ok, "finalization rejected");
                if std::env::var("TAMPER").is_ok() {
                    // A false weight evaluation (a deferred row with a wrong value)
                    // and a reordered queue must both be rejected.
                    let mut bad_rows = vqueue[range.start].0.clone();
                    bad_rows[0].y = bad_rows[0].y.add(&crate::morpheus::field8::K8::ONE);
                    let mut bq = vq.clone();
                    bq[0].deferred = &bad_rows;
                    let r1 = request::verify_final(&pp, &o, b as u64, m.public(), &bq, &fpf, &pc.keys, &mut Times::new());
                    let mut rq = vq.clone();
                    rq.reverse();
                    let r2 = jb < 2 || request::verify_final(&pp, &o, b as u64, m.public(), &rq, &fpf, &pc.keys, &mut Times::new());
                    println!("tampered finalization rejected: false weight claim {}, reordered queue {}", !r1, !r2);
                }
            }
            batches.push((jb, fs, fv, fo.bytes.len() as f64));
        }
    }
    drop(pre_tx);
    if let Some(h) = hasher { h.join().expect("finalization prefix thread"); }
    let u = units as f64;
    let pr = mean(&req_secs);
    println!("\n=== {} ({}; {} request(s), {}) ===", name, o.profile.name, n, if o.defer { "deferred fixed-weight opening" } else { "monolithic" });
    println!("prove per request          {:8.2} s  ({:.4} s per {})", pr, pr / u, unit);
    if o.defer {
        println!("model registration (one-time) {:8.2} s", model_secs);
        println!("request stage              {:8.2} s  ({:.4} s per {}; not accepted before finalization)", pr, pr / u, unit);
        let mut first = 0;
        for &(jb, fs, fv, fb) in &batches {
            let stage = mean(&req_secs[first..first + jb]);
            let vstage = if o.verify { mean(&verify_secs[first..first + jb]) } else { 0.0 };
            let bstage = mean(&req_bytes[first..first + jb]);
            first += jb;
            println!("batch J={jb:<3} finalize {fs:8.2} s, verify {fv:.2} s, {:.1} MiB; per accepted request {:.2} s ({:.4} s per {}), verify {:.2} s, {:.1} MiB",
                fb / (1u64 << 20) as f64, stage + fs / jb as f64, (stage + fs / jb as f64) / u, unit, vstage + fv / jb as f64, (bstage + fb / jb as f64) / (1u64 << 20) as f64);
        }
    }
    if o.verify { println!("verify per request         {:8.2} s", mean(&verify_secs)); }
    println!("proof per request          {:8.1} MiB", mean(&req_bytes) / (1u64 << 20) as f64);
}

fn o_defer_batches() -> Option<Vec<usize>> {
    std::env::var("DEFER_BATCHES").ok().map(|v| v.split(',').map(|x| x.trim().parse().expect("DEFER_BATCHES: sizes")).collect())
}
/// Batch sizes for `n` requests: DEFER_BATCHES, or one batch of all.
/// Short SHAKE256 fingerprint of serialized proof bytes (reproducibility checks).
fn proof_hash(b: &[u8]) -> String {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    let mut h = sha3::Shake256::default(); h.update(b);
    let mut o = [0u8; 16]; h.finalize_xof().read(&mut o);
    o.iter().map(|x| format!("{x:02x}")).collect()
}
fn batch_sizes(n: usize) -> Vec<usize> { o_defer_batches().unwrap_or_else(|| vec![n]) }
