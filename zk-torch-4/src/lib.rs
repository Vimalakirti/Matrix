//! zk-torch-4: GPU-native ZKML proving over the **almost-Goldilocks** prime
//! `q = 2^64 - 2^32 - 31`, using an Ajtai SIS commitment scheme over the ring
//! `R = F_q[X] / (X^64 + 1)` and a fold-tree opening protocol (in lieu of the
//! per-edge polynomial-commitment openings of zk-torch-3).
//!
//! See `../zk-torch-4-plan.md` for the protocol design.

pub mod basicblock;
pub mod commit;
pub mod crypto;
pub mod dag;
pub mod fold;
pub mod mlperf;
pub mod pcs;
pub mod poly;
pub mod sumcheck;
pub mod transcript;
pub mod util;

use once_cell::sync::Lazy;
use std::env;
use std::fs::File;
use std::io::Read;

use crate::util::config::Config;

// ============================================================================
// Global configuration (loaded once from a YAML file at startup, or defaulted).
// Mirrors zk-torch-3 so binaries can share the same `config.yaml`.
// ============================================================================

pub static CONFIG_FILE: Lazy<String> = Lazy::new(|| {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return "config.yaml".to_string();
    }
    args[1].clone()
});

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    // A config path given explicitly on the command line MUST exist. Silently
    // falling back to `Config::default()` (table_size_log 20, table_commit_log 8)
    // when args[1] names a missing file is a measurement trap: every benchmark
    // then runs a different lookup-table geometry than the one it claims, and
    // the only symptom is a plausible-looking timing. That cost real debugging
    // time when a scratchpad of generated configs was cleaned between runs and
    // the fold tree quietly doubled, because tcl 6 -> the default 8 pushes the
    // top aux bucket from arity 24 to 26, past the GPU same-point cap. Only the
    // no-argument case is allowed to default.
    let explicit = env::args().len() >= 2;
    match File::open(&*CONFIG_FILE) {
        Ok(mut file) => {
            let mut contents = String::new();
            file.read_to_string(&mut contents).expect("Could not read config");
            serde_yaml::from_str(&contents).expect("Could not parse config")
        }
        Err(e) if explicit => panic!(
            "config file {:?} was given as args[1] but could not be opened: {e}. \
             Refusing to fall back to the built-in default (scale_factor_log {}, \
             table_size_log {}, table_commit_log {}) — that would silently change \
             the lookup-table geometry this run reports.",
            &*CONFIG_FILE,
            Config::default().sf.scale_factor_log,
            Config::default().sf.table_size_log,
            Config::default().sf.table_commit_log,
        ),
        Err(_) => Config::default(),
    }
});

/// Read `var` as an override for a config field, falling back to the YAML.
///
/// `table_commit_log` in particular has a per-MODEL optimum, not a per-family
/// one: it should sit at `24 - max_input_n` so the largest lookup auxiliary
/// lands on the fold tree's GPU same-point cap, and `max_input_n` is a property
/// of each model's shapes (`Dag::report_lookup_arities` prints it). Measured
/// across the full profile it ranges from 16 (VGG-16) to 26 (Llama-3-8B), so one
/// value per config file cannot serve every model that shares the file. These
/// overrides let the benchmark harness set the right value per row without
/// maintaining a YAML per model.
fn config_override(var: &str, from_yaml: usize) -> usize {
    match std::env::var(var).ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(v) => v,
        None => from_yaml,
    }
}

pub static SF_LOG: Lazy<usize> =
    Lazy::new(|| config_override("ZK4_SF_LOG", CONFIG.sf.scale_factor_log));
pub static TABLE_SIZE_LOG: Lazy<usize> =
    Lazy::new(|| config_override("ZK4_TABLE_SIZE_LOG", CONFIG.sf.table_size_log));
pub static TABLE_COMMIT_LOG: Lazy<usize> =
    Lazy::new(|| config_override("ZK4_TABLE_COMMIT_LOG", CONFIG.sf.table_commit_log));

/// How many disjoint sub-groups each range bool-check arity group is split
/// into. A PUBLIC parameter: prover and verifier derive the same sub-group id
/// set from it, so it must be part of the agreed parameters exactly like
/// `TABLE_COMMIT_LOG` (env override here because that is how the benches pin
/// every other geometry knob).
///
/// Why it exists: the bool sumcheck has a per-round barrier, so every round's
/// stragglers idle the pool `arity` times over. It already exposes far more
/// parallel tasks than there are cores (2760 terms vs 96 cores on llama2 8L),
/// so the win is NOT more tasks -- it is decoupling the barriers, letting one
/// sub-group's tail overlap another's body. Measured on llama2 8L/seq64:
/// bool 10.09s -> 5.02s at 4, -> 4.74s at 8 (prove_range 12.85s -> 7.78s ->
/// 7.54s). 4 captures most of it; 8 adds ~6%.
///
/// 1 is byte-identical to the unsplit protocol, including the fork id.
pub static BOOL_SPLIT: Lazy<usize> = Lazy::new(|| {
    std::env::var("ZK4_BOOL_SPLIT").ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
        .max(1)
});
pub static SF_FLOAT: Lazy<f32> = Lazy::new(|| (1 << *SF_LOG) as f32);
pub static SF_INT: Lazy<usize> = Lazy::new(|| 1 << *SF_LOG);

pub const SIGN_BIT: usize = 63;
pub const FIELD_SIZE: usize = 64;
pub const ALMOST_GOLDILOCKS_PRIME: u64 = almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME;

// Re-exports
pub use almost_goldilocks_cuda;
pub use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
pub use almost_goldilocks_cuda::field::AlmostGoldilocksField;
pub use poly::{DenseMLPoly, MLPoly};
pub use transcript::Transcript;

/// Serialized byte length of a proof component, for the evaluation harness.
///
/// The streaming bins report proof size as a sum over components (per-proof
/// DAG proof + fold proof + accumulator chunk, and the one-time finalize
/// proof) because no single object holds them all. Returns 0 if a component
/// cannot be serialized, so a reporting failure degrades the size number
/// rather than aborting a completed proof run.
pub fn ser_len<T: serde::Serialize>(x: &T) -> usize {
    bincode::serialize(x).map(|v| v.len()).unwrap_or(0)
}

/// Zero-filled witness buffer, allocated rather than written.
///
/// `vec![AlmostGoldilocksField(0); n]` does NOT hit Rust's `alloc_zeroed`
/// specialization — that covers primitives, not user newtypes — so it lowers to
/// a single-threaded store loop over the whole buffer. At benchmark sizes this
/// dominates: a 32-layer Llama-2 with sharding touches ~490 GB of host memory
/// this way, taking ~11 minutes at ~4 of 96 cores, before a single GPU kernel
/// runs. `vec![0u64; n]` DOES specialize to calloc, and the field is
/// `#[repr(transparent)]` over `u64`, so the same bytes come for free.
///
/// This is benchmark scaffolding: it fabricates synthetic weights, which a
/// deployment would load instead. It cannot affect any reported number.
pub fn zero_witness_vec(n: usize) -> Vec<AlmostGoldilocksField> {
    let mut v = vec![0u64; n];
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    std::mem::forget(v);
    // SAFETY: AlmostGoldilocksField is #[repr(transparent)] over u64, so both
    // element types have identical size and alignment and the allocation is
    // valid for the new type. AlmostGoldilocksField(0) is the all-zero bit
    // pattern, so the calloc'd contents are already the intended value.
    unsafe { Vec::from_raw_parts(ptr as *mut AlmostGoldilocksField, len, cap) }
}

/// Random witness buffer with values in `0..magnitude`, filled in parallel.
///
/// The sequential form (`(0..n).map(|_| rng.gen()).collect()`) is the other
/// half of the weight-generation cost. Chunked so each worker draws from its
/// own thread RNG; values are benchmark inputs, so the exact stream does not
/// need to be reproducible across thread counts.
pub fn rand_witness_vec(n: usize, magnitude: u32) -> Vec<AlmostGoldilocksField> {
    use rand::Rng;
    use rayon::prelude::*;
    let m = magnitude.max(1);
    let mut v = zero_witness_vec(n);
    v.par_chunks_mut(1 << 16).for_each(|chunk| {
        let mut rng = rand::thread_rng();
        for x in chunk.iter_mut() {
            *x = AlmostGoldilocksField((rng.gen::<u32>() % m) as u64);
        }
    });
    v
}

/// Byte breakdown of a proof, split the way the protocol is.
///
/// `DagProof` already separates the pieces, so no estimation is involved:
///
///   - **sumcheck, non-lookup** — `node_proofs` (per-operator sumchecks) plus
///     `edge_proofs` (the reducer sumchecks that merge repeated claims on one
///     edge into a single opening; this is the cost of deferring).
///   - **sumcheck, lookup** — `range_proof` and `two_pow_proof`.
///   - **PCS** — the fold tree, i.e. everything in `FoldTreeProof`.
///   - **output claims** — terminal evaluations carried in the clear.
///
/// The three add up to the serialized total only approximately: bincode adds
/// per-field framing, so the parts sum slightly below the whole. The residual
/// is reported rather than silently distributed.
pub fn proof_size_report<F: serde::Serialize>(
    node_proofs: &impl serde::Serialize,
    edge_proofs: &impl serde::Serialize,
    range_proof: &impl serde::Serialize,
    two_pow_proof: &impl serde::Serialize,
    output_claims: &impl serde::Serialize,
    dag_total: &impl serde::Serialize,
    fold_proof: &F,
) -> String {
    let node = ser_len(node_proofs);
    let edge = ser_len(edge_proofs);
    let range = ser_len(range_proof);
    let two_pow = ser_len(two_pow_proof);
    let outs = ser_len(output_claims);
    let dag = ser_len(dag_total);
    let pcs = ser_len(fold_proof);
    let total = dag + pcs;
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    let pct = |b: usize| if total == 0 { 0.0 } else { b as f64 * 100.0 / total as f64 };
    format!(
        "  proof breakdown (total {:.1} MB):\n\
         \x20   sumcheck non-lookup : {:>9.2} MB  {:>5.1}%   (node {:.2} + reducer/edge {:.2})\n\
         \x20   sumcheck lookup     : {:>9.2} MB  {:>5.1}%   (range {:.2} + two_pow {:.2})\n\
         \x20   PCS (fold tree)     : {:>9.2} MB  {:>5.1}%\n\
         \x20   output claims       : {:>9.2} MB  {:>5.1}%\n\
         \x20   framing residual    : {:>9.2} MB",
        mb(total),
        mb(node + edge), pct(node + edge), mb(node), mb(edge),
        mb(range + two_pow), pct(range + two_pow), mb(range), mb(two_pow),
        mb(pcs), pct(pcs),
        mb(outs), pct(outs),
        mb(dag.saturating_sub(node + edge + range + two_pow + outs)),
    )
}
