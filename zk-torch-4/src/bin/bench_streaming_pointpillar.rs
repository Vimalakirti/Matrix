//! Streaming PointPillar composed with the cross-proof streaming accumulator.
//! Each streamed proof is one (pillars, coords) input pair sharing the same
//! PointPillar weights; all WEIGHTS are Role::Constant (deferred → amortized
//! into one finalize opening), the two INPUTS (pillars + coords) are
//! Role::Input, committed/opened per-proof.
//!
//! Run with `bench_config.yaml` as args[1]. Env: NY(16) NX(16) N_PILLARS(32)
//! MAX_POINTS(4) NUM_ANCHORS(2) NUM_CLASSES(5) N_PROOFS(3) MAX_NUM_VARS(22)
//! NUM_PARTITIONS(1) ZK4_B(21) ZK4_BASE(2).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::pointpillar::pointpillar;
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

fn zero_conv_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kh_pad * kw_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn zero_conv_transpose_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_in_pad * c_out_pad * kh_pad * kw_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![c_in, c_out, kh, kw], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn zero_linear_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

/// Broadcast 1D bias `[c]` → 3D `[c, h, w]`, with tiny `% 2` values so
/// the post-conv ReLU stays inside `[0, 2^TABLE_SIZE_LOG)`.
fn small_bias_3d(c_out: usize, h: usize, w: usize) -> Witness {
    let c_pad = c_out.next_power_of_two();
    let h_pad = h.next_power_of_two();
    let w_pad = w.next_power_of_two();
    let size = c_pad * h_pad * w_pad;
    let mut data = zk_torch_4::zero_witness_vec(size);
    let mut rng = rand::thread_rng();
    for c in 0..c_out {
        let v = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
        for hi in 0..h {
            for wi in 0..w {
                data[wi + hi * w_pad + c * w_pad * h_pad] = v;
            }
        }
    }
    Witness::new(vec![c_out, h, w], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn small_bias_1d(dim: usize) -> Witness {
    let dim_pad = dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(dim_pad);
    let mut rng = rand::thread_rng();
    for i in 0..dim {
        data[i] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
    }
    Witness::new(vec![dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn rand_uint_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64)).collect()
}

fn demo_seed() -> Seed {
    Seed([
        0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
        0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
    ])
}

/// Build the per-proof (pillars, coords) input witnesses with the exact
/// shapes / dtype(Uint, scale 0) / value patterns the pointpillar bin uses.
fn make_inputs(
    n_pillars: usize,
    max_points: usize,
    ny: usize,
    nx: usize,
) -> (Witness, Witness) {
    let n_pad = n_pillars.next_power_of_two();
    let mp_pad = max_points.next_power_of_two();
    let pillar_size = n_pad * mp_pad * 16; // 11 padded to 16
    let pillar_witness = Witness::new(
        vec![n_pillars, max_points, 11],
        rand_uint_vec(pillar_size),
        DataType::Uint, *SF_LOG, Role::Input,
    );

    let coord_dim_pad = 2usize.next_power_of_two();
    let coord_size = n_pad * coord_dim_pad;
    let mut coord_data = zk_torch_4::zero_witness_vec(coord_size);
    let mut rng = rand::thread_rng();
    for p in 0..n_pillars {
        // Column-major: dim 0 has stride 1, dim 1 has stride n_pad.
        coord_data[p] = AlmostGoldilocksField((rng.gen::<u32>() as usize % ny) as u64);
        coord_data[p + n_pad] = AlmostGoldilocksField((rng.gen::<u32>() as usize % nx) as u64);
    }
    let coord_witness = Witness::new(
        vec![n_pillars, 2],
        coord_data,
        DataType::Uint, 0, Role::Input,
    );
    (pillar_witness, coord_witness)
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    let ny: usize = std::env::var("NY").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let nx: usize = std::env::var("NX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_pillars: usize = std::env::var("N_PILLARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_points: usize = std::env::var("MAX_POINTS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let num_anchors: usize = std::env::var("NUM_ANCHORS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_classes: usize = std::env::var("NUM_CLASSES").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(5);
    let n_proofs: usize = std::env::var("N_PROOFS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(3);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch_size: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    println!("=== Streaming PointPillar on Almost-Goldilocks (batch {}) ===", batch_size);
    println!("BEV {}x{}, n_pillars={} max_points={} num_anchors={} num_classes={}",
             ny, nx, n_pillars, max_points, num_anchors, num_classes);
    println!("max_num_vars={} N_PROOFS={} partitions={} (threads={})",
             max_num_vars, n_proofs, num_partitions, rayon::current_num_threads());

    let t0 = Instant::now();
    let mut all_weights: Vec<(Witness, Witness)> = Vec::new();

    // VFE: Linear 11→64
    all_weights.push((zero_linear_weight(11, 64), small_bias_1d(64)));

    // Spatial tracking
    let mut sh = ny / 2;
    let mut sw = nx / 2;

    // Block 1: 6 convs (64→64, first stride 2)
    all_weights.push((zero_conv_weight(64, 64, 3, 3), small_bias_3d(64, sh, sw)));
    for _ in 0..5 {
        all_weights.push((zero_conv_weight(64, 64, 3, 3), small_bias_3d(64, sh, sw)));
    }
    let b1_h = sh;
    let b1_w = sw;

    // Block 2: 6 convs (64→128, first stride 2)
    sh /= 2; sw /= 2;
    all_weights.push((zero_conv_weight(64, 128, 3, 3), small_bias_3d(128, sh, sw)));
    for _ in 0..5 {
        all_weights.push((zero_conv_weight(128, 128, 3, 3), small_bias_3d(128, sh, sw)));
    }

    // Block 3: 6 convs (128→256, first stride 2)
    sh /= 2; sw /= 2;
    all_weights.push((zero_conv_weight(128, 256, 3, 3), small_bias_3d(256, sh, sw)));
    for _ in 0..5 {
        all_weights.push((zero_conv_weight(256, 256, 3, 3), small_bias_3d(256, sh, sw)));
    }

    // Deblock 1: ConvTranspose2d(64→128, k=1, s=1)
    all_weights.push((zero_conv_transpose_weight(64, 128, 1, 1), small_bias_3d(128, b1_h, b1_w)));
    // Deblock 2: ConvTranspose2d(128→256, k=2, s=2)
    all_weights.push((zero_conv_transpose_weight(128, 256, 2, 2), small_bias_3d(256, b1_h, b1_w)));
    // Deblock 3: two cascaded ConvTranspose2d(256→256, k=2, s=2)
    all_weights.push((zero_conv_transpose_weight(256, 256, 2, 2), small_bias_3d(256, sh * 2, sw * 2)));
    all_weights.push((zero_conv_transpose_weight(256, 256, 2, 2), small_bias_3d(256, b1_h, b1_w)));

    // Detection heads (3 × Conv2D 1×1, input = multi_concat of [128, 256, 256]
    // padded to 1024 channels per pointpillar DAG).
    let det_h = b1_h;
    let det_w = b1_w;
    let cls_out = num_anchors * num_classes;
    let box_out = num_anchors * 7;
    let dir_out = num_anchors * 2;
    let concat_channels = 1024;
    all_weights.push((zero_conv_weight(concat_channels, cls_out, 1, 1), small_bias_3d(cls_out, det_h, det_w)));
    all_weights.push((zero_conv_weight(concat_channels, box_out, 1, 1), small_bias_3d(box_out, det_h, det_w)));
    all_weights.push((zero_conv_weight(concat_channels, dir_out, 1, 1), small_bias_3d(dir_out, det_h, det_w)));

    println!("Weight gen: {:?} ({} layers)", t0.elapsed(), all_weights.len());

    // Build DAG once.
    let mut g = DagBuilder::new();
    // Batch = B (pillars, coords) input pairs, interleaved as the builder
    // consumes them (chunks of 2). Weights are shared across the batch.
    let mut xs: Vec<EdgeId> = Vec::with_capacity(batch_size * 2);
    for _ in 0..batch_size {
        xs.push(g.input(vec![n_pillars, max_points, 11], DataType::Uint));
        xs.push(g.input(vec![n_pillars, 2], DataType::Uint));
    }
    let _output = g.pipe(
        &xs,
        pointpillar(all_weights, ny, nx, n_pillars, max_points, num_anchors, num_classes),
    );

    let t1 = Instant::now();
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges, batch={})",
             t1.elapsed(), dag.nodes.len(), dag.num_edges(), batch_size);

    for &x in &xs {
        assert_eq!(witnesses_template[x][0].role, Role::Input, "PointPillar input must be Role::Input");
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

    let label = b"zkml-pointpillar-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut checked_role = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} (pillars,coords) inferences:", n_proofs);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let mut batch_inputs: Vec<(EdgeId, Witness)> = Vec::with_capacity(batch_size * 2);
        for pair in xs.chunks(2) {
            let (pw, cw) = make_inputs(n_pillars, max_points, ny, nx);
            batch_inputs.push((pair[0], pw));
            batch_inputs.push((pair[1], cw));
        }

        let s0 = Instant::now();
        dag.run(&mut witnesses, &batch_inputs);
        let d_run = s0.elapsed(); t_run += d_run;

        store.clear_non_constants(&witnesses);
        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-inf");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-inf");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-inference verify failed at {}", it); return; }

        if !checked_role {
            for dc in &r.claims {
                assert!(!xs.contains(&dc.edge_id),
                    "input edge {} was deferred as a shared weight — unsound", dc.edge_id);
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
    println!("  prove(defer)  per-inf : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-inf : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (PointPillar, weights amortized across {} inferences = {} proofs x batch {})",
        n_proofs * batch_size, n_proofs, batch_size);
}
