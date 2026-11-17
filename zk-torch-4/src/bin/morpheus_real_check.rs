//! CPU-only check of a Morpheus transformer instance (no proving, no CUDA
//! init): builds the model exactly as morpheus_e2e does (same MODEL / shape /
//! WEIGHTS env, MORPHEUS_PROMPT_SEED picks the request), generates the
//! suffix with the fixed-point forward pass and prints
//! `morpheus::models::check_report` (tokens, range violations, request
//! tensor widths). CHECK_DUMP=<file> writes tokens + logits for
//! scripts/validate_transformer.py.

use std::time::Instant;
use zk_torch_4::morpheus::models;

fn peak_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok()))).unwrap_or(0.0) / 1e6
}

fn main() {
    env_logger::init();
    if std::env::var_os("MORPHEUS_FULL_AUX").is_none() { std::env::set_var("MORPHEUS_LEAN_AUX", "1"); }
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gpt2".into());
    let seed: u64 = std::env::var("MORPHEUS_PROMPT_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let t = Instant::now();
    let inst = models::build(&model, seed);
    println!("{}: {} nodes, {} edges; graph {:.2}s, witness {:.2}s; peak RSS {:.1} GB; scale 2^{}",
        inst.name, inst.dag.nodes.len(), inst.dag.num_edges(), t.elapsed().as_secs_f64() - inst.witness_secs, inst.witness_secs, peak_rss_gb(), *zk_torch_4::SF_LOG);
    let dump = std::env::var("CHECK_DUMP").ok().map(std::path::PathBuf::from);
    models::check_report(&inst, dump.as_deref());
    println!("[check] total {:.1}s, peak RSS {:.1} GB", t.elapsed().as_secs_f64(), peak_rss_gb());
}
