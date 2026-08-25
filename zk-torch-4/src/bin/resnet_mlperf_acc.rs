//! ResNet-50 v1.5 — MLPerf Inference v6.0 (edge) accuracy binary.
//!
//! Loads exported real weights (BatchNorm folded into conv) + preprocessed
//! ImageNet val images, runs zk-torch-4's fixed-point forward pass
//! (`dag.run()`), and reports Top-1 accuracy vs the MLPerf 99% target
//! (75.70%). Weights/inputs are produced by `scripts/export_resnet50.py`.
//! See `MLPERF_ACCURACY.md` (milestone M1).
//!
//! Usage (config yaml as args[1], per zk-4 convention — sets SF_LOG /
//! table_size_log; export with the matching --sf-log):
//!
//!     ./target/release/resnet_mlperf_acc cv_config.yaml \
//!         --weights-dir /tmp/resnet50_export --num-images 100
//!
//! Env:
//!   MAX_NUM_VARS   commit key vars (default 28 — fits 224x224, sparse range aux)
//!   ZK4_B, ZK4_BASE   Ajtai commit bit-width / base (default 21 / 2)
//!   ZK4_ACC_PROVE=k   run prove+verify on the first k images (spot check, §6)
//!
//! NOTE (M1 open item): real ImageNet input is mean-subtracted → signed. The
//! conv range path's handling of signed activations at full depth is exactly
//! what M1 validates; if Top-1 is far below target or a `[range] WARNING`
//! fires, that's the signed/range issue flagged in MLPERF_ACCURACY.md §8, not
//! the harness.

use std::path::{Path, PathBuf};
use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::resnet::resnet50_with_biases;
use zk_torch_4::dag::{DagBuilder, DataType, Role};
use zk_torch_4::mlperf::{decode_argmax, load_witness, range_health_check, read_metadata, topk};
use zk_torch_4::transcript::Transcript;

fn arg_val(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn demo_seed() -> Seed {
    Seed([
        0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
        0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
    ])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let args: Vec<String> = std::env::args().collect();
    let weights_dir = arg_val(&args, "--weights-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/resnet50_export"));
    let num_images: usize = arg_val(&args, "--num-images")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let n_prove: usize = env_usize("ZK4_ACC_PROVE", 0);
    let max_num_vars = env_usize("MAX_NUM_VARS", 28);
    let input_size = env_usize("INPUT_SIZE", 224);

    let meta = read_metadata(&weights_dir);
    let sf_log = meta.sf_log;
    let num_conv = meta.num_conv;
    let num_classes = meta.num_classes;

    println!("=== ResNet-50 v1.5 — MLPerf accuracy (zk-torch-4) ===");
    println!("  weights_dir: {}", weights_dir.display());
    println!("  num_images: {}, num_conv: {}, classes: {}", num_images, num_conv, num_classes);
    println!("  export SF_LOG: {}, runtime SF_LOG: {}", sf_log, *zk_torch_4::SF_LOG);
    if sf_log != *zk_torch_4::SF_LOG {
        eprintln!(
            "WARNING: SF_LOG mismatch (export={}, runtime config={}). Re-export with \
             --sf-log {}, or pass a config yaml whose scale_factor_log={}.",
            sf_log, *zk_torch_4::SF_LOG, *zk_torch_4::SF_LOG, sf_log
        );
    }

    // ---- Load real weights (signed fixed-point, Int, sf=sf_log) ----
    let t0 = Instant::now();
    let conv_weights = (0..num_conv)
        .map(|i| load_witness(&weights_dir.join(format!("conv_{:03}_weight.bin", i)), sf_log, DataType::Int, Role::Constant))
        .collect::<Vec<_>>();
    let conv_biases = (0..num_conv)
        .map(|i| load_witness(&weights_dir.join(format!("conv_{:03}_bias.bin", i)), sf_log, DataType::Int, Role::Constant))
        .collect::<Vec<_>>();
    let fc_weight = load_witness(&weights_dir.join("fc_weight.bin"), sf_log, DataType::Int, Role::Constant);
    let fc_bias = load_witness(&weights_dir.join("fc_bias.bin"), sf_log, DataType::Int, Role::Constant);
    println!("  loaded {} conv w+b, fc w+b in {:?}", num_conv, t0.elapsed());

    // ---- Build DAG once, reuse per image ----
    let t1 = Instant::now();
    let mut g = DagBuilder::new();
    let x = g.input(vec![3, input_size, input_size], DataType::Int);
    let out = g.pipe(
        &[x],
        resnet50_with_biases(conv_weights, conv_biases, fc_weight, Some(fc_bias), num_classes, num_conv),
    );
    let output_edge = out[0];
    let (dag, mut witnesses) = g.compile();
    println!("  DAG: {} nodes, {} edges, compiled in {:?}", dag.nodes.len(), dag.num_edges(), t1.elapsed());

    // ---- Optional commit key for spot proofs ----
    let store_key = if n_prove > 0 {
        let b = env_usize("ZK4_B", 21);
        let base = env_usize("ZK4_BASE", 2);
        Some(AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base))
    } else {
        None
    };

    // ---- Labels ----
    let labels: Vec<i32> = std::fs::read_to_string(weights_dir.join("labels.txt"))
        .expect("cannot read labels.txt")
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1).and_then(|s| s.parse().ok()))
        .collect();

    let images_dir = weights_dir.join("images");
    let n = num_images.min(labels.len());
    println!("\nRunning inference on {} images...", n);

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut proved_ok = 0usize;

    for idx in 0..n {
        let img_path = images_dir.join(format!("{:05}.bin", idx));
        if !img_path.exists() {
            continue;
        }
        let input = load_witness(&img_path, sf_log, DataType::Int, Role::Input);

        let t = Instant::now();
        dag.run(&mut witnesses, &[(0, input)]);
        let fwd = t.elapsed();

        // Range/overflow health check (see MLPERF_ACCURACY.md §1/§8).
        let rep = range_health_check(&witnesses, *zk_torch_4::TABLE_SIZE_LOG);
        if !rep.is_clean() {
            eprintln!(
                "  [{}] RANGE: {} value(s) >= 2^{} (max|.|={} at edge {}) — sample likely INVALID",
                idx, rep.over_table, rep.table_size_log, rep.max_abs, rep.max_abs_edge
            );
        }

        let (pred, _conf) = decode_argmax(&witnesses[output_edge][0], num_classes, sf_log);
        let true_label = labels[idx];
        if pred as i32 == true_label {
            correct += 1;
        }
        total += 1;

        // Spot proof for the first `n_prove` images (§6).
        if idx < n_prove {
            if let Some(key) = &store_key {
                let mut store = GpuAjtaiStore::new(dag.num_edges(), key.clone());
                dag.commit_constants(&witnesses, &mut store);
                dag.commit_remaining(&witnesses, &mut store);
                let mut tp = Transcript::new(b"zkml-resnet-acc");
                let (dp, fp) = dag.prove_with_fold_tree(&witnesses, &store, &mut tp);
                let mut tv = Transcript::new(b"zkml-resnet-acc");
                let ok = dag.verify_with_fold_tree(&witnesses, &store, &dp, &fp, &mut tv);
                if ok { proved_ok += 1; }
                println!("    [{}] spot proof: Verified={}", idx, ok);
            }
        }

        if total <= 10 || total % 10 == 0 {
            let top = topk(&witnesses[output_edge][0], num_classes, 5, sf_log);
            println!(
                "  [{}/{}] pred={} true={} {} acc={:.2}% fwd={:?} top5={:?}",
                total, n, pred, true_label,
                if pred as i32 == true_label { "OK" } else { ".." },
                100.0 * correct as f64 / total as f64, fwd,
                top.iter().map(|(c, v)| (*c, (v * 100.0).round() / 100.0)).collect::<Vec<_>>(),
            );
        }
    }

    println!("\n=== Results ===");
    println!("  images scored: {}", total);
    println!("  Top-1: {:.2}% ({}/{})", 100.0 * correct as f64 / total.max(1) as f64, correct, total);
    if n_prove > 0 {
        println!("  spot proofs verified: {}/{}", proved_ok, n_prove.min(total));
    }
    println!("  MLPerf edge target (99% of 76.46%): 75.70%");
    println!("  SF_LOG={}, TABLE_SIZE_LOG={}", sf_log, *zk_torch_4::TABLE_SIZE_LOG);
    let _ = Path::new("");
}
