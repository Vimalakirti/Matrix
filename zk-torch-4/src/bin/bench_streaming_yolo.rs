//! Streaming YOLOv11n (CV detection, batch via N inferences) composed with the
//! cross-proof streaming accumulator. Each streamed proof is one image; all
//! YOLO conv WEIGHTS (+biases) are Role::Constant (deferred → amortized into
//! one finalize opening); the per-image INPUT is Role::Input (committed/opened
//! per-proof).
//!
//! Run with `bench_config.yaml` as args[1]. Env: NUM_STAGES(2) INPUT_SIZE(64)
//! N_PROOFS(3) MAX_NUM_VARS(22) NUM_PARTITIONS(1) ZK4_B(21) ZK4_BASE(2).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;
use zk_torch_4::SF_LOG;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::yolo::{generate_yolo_weight, yolov11n};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 500)
}

/// Builds every (weight, bias) pair for YOLOv11n up to `num_stages`.
/// Matches the per-stage shape table from the yolo bin verbatim.
fn gen_all_weights(num_stages: usize, input_spatial: usize) -> Vec<(Witness, Witness)> {
    let mut weights = Vec::new();

    let s1 = input_spatial / 2;
    let s2 = input_spatial / 4;
    let s3 = input_spatial / 8;
    let s4 = input_spatial / 16;
    let s5 = input_spatial / 32;

    macro_rules! w {
        ($c_in:expr, $c_out:expr, $k:expr, $dw:expr, $h:expr, $w_dim:expr) => {
            weights.push(generate_yolo_weight($c_in, $c_out, $k, $k, $dw, $h, $w_dim));
        };
    }

    if num_stages >= 1 { w!(3, 16, 3, false, s1, s1); }

    if num_stages >= 2 { w!(16, 32, 3, false, s2, s2); }

    if num_stages >= 3 {
        w!(32, 64, 1, false, s2, s2);
        w!(32, 16, 3, false, s2, s2);
        w!(16, 32, 3, false, s2, s2);
        w!(128, 64, 1, false, s2, s2);
    }

    if num_stages >= 4 {
        w!(64, 64, 3, false, s3, s3);
        w!(64, 128, 1, false, s3, s3);
        w!(64, 32, 3, false, s3, s3);
        w!(32, 64, 3, false, s3, s3);
        w!(256, 128, 1, false, s3, s3);
    }

    if num_stages >= 5 {
        w!(128, 128, 3, false, s4, s4);
        w!(128, 128, 1, false, s4, s4);
        w!(64, 32, 1, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(64, 32, 1, false, s4, s4);
        w!(64, 64, 1, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
    }

    if num_stages >= 6 {
        w!(128, 256, 3, false, s5, s5);
        w!(256, 256, 1, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(128, 128, 1, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
        w!(256, 128, 1, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
    }

    if num_stages >= 7 {
        w!(512, 128, 1, false, s4, s4);
        w!(64, 32, 3, false, s4, s4);
        w!(32, 64, 3, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
        w!(256, 64, 1, false, s3, s3);
        w!(32, 16, 3, false, s3, s3);
        w!(16, 32, 3, false, s3, s3);
        w!(128, 64, 1, false, s3, s3);
        w!(64, 64, 3, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
        w!(64, 32, 3, false, s4, s4);
        w!(32, 64, 3, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
        w!(128, 128, 3, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(128, 128, 1, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
    }

    if num_stages >= 8 {
        // P3 head
        w!(64, 64, 3, false, s3, s3);
        w!(64, 64, 3, false, s3, s3);
        w!(64, 64, 1, false, s3, s3);
        w!(64, 64, 3, true, s3, s3);
        w!(64, 80, 1, false, s3, s3);
        w!(80, 80, 3, true, s3, s3);
        w!(80, 80, 1, false, s3, s3);
        w!(80, 80, 1, false, s3, s3);

        // P4 head
        w!(128, 64, 3, false, s4, s4);
        w!(64, 64, 3, false, s4, s4);
        w!(64, 64, 1, false, s4, s4);
        w!(128, 128, 3, true, s4, s4);
        w!(128, 80, 1, false, s4, s4);
        w!(80, 80, 3, true, s4, s4);
        w!(80, 80, 1, false, s4, s4);
        w!(80, 80, 1, false, s4, s4);

        // P5 head
        w!(256, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 1, false, s5, s5);
        w!(256, 256, 3, true, s5, s5);
        w!(256, 80, 1, false, s5, s5);
        w!(80, 80, 3, true, s5, s5);
        w!(80, 80, 1, false, s5, s5);
        w!(80, 80, 1, false, s5, s5);
    }

    weights
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

    let num_stages: usize = std::env::var("NUM_STAGES").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let input_spatial: usize = std::env::var("INPUT_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch_size: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    println!("=== Streaming YOLOv11n ({} stages, input {}x{}, batch {}) ===", num_stages, input_spatial, input_spatial, batch_size);
    println!("N_PROOFS={} max_num_vars={} partitions={}", n_proofs, max_num_vars, num_partitions);

    let t0 = Instant::now();
    let all_weights = gen_all_weights(num_stages, input_spatial);
    println!("Weight gen: {:?} ({} conv layers)", t0.elapsed(), all_weights.len());

    let mut g = DagBuilder::new();
    let xs: Vec<EdgeId> = (0..batch_size)
        .map(|_| g.input(vec![3, input_spatial, input_spatial], DataType::Uint))
        .collect();
    let _ = g.pipe(&xs, yolov11n(all_weights, num_stages));
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges (batch={})", dag.nodes.len(), dag.num_edges(), batch_size);

    for &x in &xs {
        assert_eq!(witnesses_template[x][0].role, Role::Input,
            "YOLO input edge must be Role::Input (per-proof), not deferred");
    }

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        // Report the EFFECTIVE partition count: set_partition_boundaries
        // clamps to what the graph supports, so the requested value can
        // overstate it and would land in the CSV as such.
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }

    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses_template, &mut store);
    println!("Offline commit (weights, amortized): {:.2}ms", ms(t_off.elapsed()));

    // 3 → pad to 4 channels (matches yolo bin's forward-input construction).
    let c_pad = 4;
    let input_buf_size = c_pad * input_spatial * input_spatial;

    let label = b"zkml-yolo-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut checked_role = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} images:", n_proofs);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let batch_inputs: Vec<(EdgeId, Witness)> = xs.iter().map(|&x| (x, Witness::new(
            vec![3, input_spatial, input_spatial],
            rand_field_vec(input_buf_size),
            DataType::Uint, *SF_LOG, Role::Input,
        ))).collect();

        let s0 = Instant::now();
        dag.run(&mut witnesses, &batch_inputs);
        let d_run = s0.elapsed(); t_run += d_run;

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-img");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-img");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-image verify failed at {}", it); return; }

        if !checked_role {
            for dc in &r.claims {
                assert!(!xs.contains(&dc.edge_id), "input edge {} was deferred as a shared weight — unsound", dc.edge_id);
            }
            checked_role = true;
        }

        let s4 = Instant::now();
        let chunk = prover_acc.add_proof(&r, &witnesses);
        let d_acc = s4.elapsed(); t_acc += d_acc;
        proof_bytes += ser_len(&dp) + ser_len(&fp) + ser_len(&chunk);
        // Keep one proof's component split; every iteration has the
        // same shape, so the last is representative.
        breakdown = Some(zk_torch_4::proof_size_report(
            &dp.node_proofs, &dp.edge_proofs, &dp.range_proof,
            &dp.two_pow_proof, &dp.output_claims, &dp, &fp));
        let s5 = Instant::now();
        let ok = verifier_acc.verify_add_proof(&r, &witnesses, &chunk);
        let d_accv = s5.elapsed(); t_accv += d_accv;
        if !ok { eprintln!("streaming verifier rejected at {}", it); return; }

        println!("  [{:>2}/{}] run {:>7.1}ms commit {:>6.1}ms prove {:>7.1}ms verify {:>6.1}ms acc {:>7.1}ms acc-v {:>6.1}ms",
            it + 1, n_proofs, ms(d_run), ms(d_commit), ms(d_prove), ms(d_verify), ms(d_acc), ms(d_accv));
    }

    let n_steps = prover_acc.num_steps();
    let n_const = prover_acc.num_edges();
    let s_fp = Instant::now();
    let final_proof = prover_acc.finalize(&witnesses_template, &store);
    let t_finalize = s_fp.elapsed();
    let s_fv = Instant::now();
    let ok = verifier_acc.verify_finalize(&store, &final_proof);
    let t_fv = s_fv.elapsed();
    if !ok { eprintln!("verify_finalize REJECTED — soundness chain broken"); return; }

    let n = (n_proofs * batch_size) as f64;
    println!("\n=== Results ({} weight edges deferred, {} reducer steps) ===", n_const, n_steps);
    println!("  prove(defer)  per-img : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-img : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (YOLOv11n, weights amortized across {} images = {} proofs x batch {})",
        n_proofs * batch_size, n_proofs, batch_size);
}
