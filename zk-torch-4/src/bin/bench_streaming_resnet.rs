//! Streaming ResNet-50 (CV, batch via N inferences) composed with the
//! cross-proof streaming accumulator. Tests whether the accumulator/reducer
//! handles a CONV model (the Tier-2 question: conv weights reused across
//! spatial positions can produce mixed-arity claims on a shared edge). Each
//! streamed proof is one image; WEIGHTS (conv + fc) are Role::Constant
//! (deferred → amortized into one finalize opening); the per-image INPUT is
//! Role::Input (committed/opened per-proof).
//!
//! Run with `bench_config.yaml` as args[1]. Env: NUM_LAYERS(1) INPUT_SIZE(224)
//! N_PROOFS(3) MAX_NUM_VARS(22) NUM_PARTITIONS(1).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::resnet::{resnet50, resnet50_conv_configs, resnet50_output_shape, resnet50_with_biases};
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::mlperf::{load_witness, read_metadata};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, xmag())
}

// Magnitude knobs (default = original toy values, so behavior is unchanged
// unless opted in). Set `ZK4_WMAG`/`ZK4_XMAG` to drive activations up to a
// realistic fixed-point scale: a real model loaded at sf=`scale_factor_log`
// has weights/inputs whose integer reps are ~2^sf · |real value|, which makes
// the per-layer activations (hence the NonNegative range table) far larger
// than the `% 2` / `% 128` toy regime. Use this to find the true sound
// `table_size_log` (see right-size analysis), not the artificially tiny one.
fn wmag() -> u32 { std::env::var("ZK4_WMAG").ok().and_then(|s| s.parse().ok()).unwrap_or(2) }
fn xmag() -> u32 { std::env::var("ZK4_XMAG").ok().and_then(|s| s.parse().ok()).unwrap_or(128) }

fn gen_conv_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let (khp, kwp, cip, cop) = (kh.next_power_of_two(), kw.next_power_of_two(),
        c_in.next_power_of_two(), c_out.next_power_of_two());
    let mut data = zk_torch_4::zero_witness_vec(cop * cip * khp * kwp);
    let mut rng = rand::thread_rng();
    let m = wmag();
    for d in 0..c_out { for c in 0..c_in { for hi in 0..kh { for wi in 0..kw {
        let idx = wi + hi * kwp + c * kwp * khp + d * kwp * khp * cip;
        data[idx] = AlmostGoldilocksField((rng.gen::<u32>() % m) as u64);
    }}}}
    Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn gen_fc_weight(in_dim: usize, out_dim: usize) -> Witness {
    let (ip, op) = (in_dim.next_power_of_two(), out_dim.next_power_of_two());
    let mut data = zk_torch_4::zero_witness_vec(ip * op);
    let mut rng = rand::thread_rng();
    let m = wmag();
    for i in 0..in_dim { for j in 0..out_dim { data[i + j * ip] = AlmostGoldilocksField((rng.gen::<u32>() % m) as u64); } }
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

/// FC bias. Batched, the head runs in [features, B] form, so the bias is
/// declared [out_dim, 1]: broadcasting matches TRAILING dimensions and a bare
/// [out_dim] would align against the batch axis. Same data either way.
fn gen_fc_bias(out_dim: usize, batch: usize) -> Witness {
    let op = out_dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(op);
    let mut rng = rand::thread_rng();
    let m = wmag();
    for i in 0..out_dim { data[i] = AlmostGoldilocksField((rng.gen::<u32>() % m) as u64); }
    let shape = if batch > 1 { vec![out_dim, 1] } else { vec![out_dim] };
    Witness::new(shape, data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn demo_seed() -> Seed {
    Seed([0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
          0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let input_size: usize = std::env::var("INPUT_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(224);
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch_size: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_classes = 1000usize;

    println!("=== Streaming ResNet-50 ({} conv layers, input {}x{}) ===", num_layers, input_size, input_size);
    println!("N_PROOFS={} max_num_vars={} partitions={}", n_proofs, max_num_vars, num_partitions);

    let configs = resnet50_conv_configs();
    let max_conv = num_layers.min(configs.len());
    // RESNET_REAL=<dir from scripts/export_resnet50.py>: trained weights with
    // BatchNorm folded into conv biases (signed, scale 2^SF_LOG), and the
    // exported images as inputs, starting at RESNET_IMAGE (default 0).
    let real = std::env::var("RESNET_REAL").ok().map(std::path::PathBuf::from);
    let real_image: usize = std::env::var("RESNET_IMAGE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut g = DagBuilder::new();
    let mut real_x = None;
    if let Some(dir) = &real {
        assert!(batch_size == 1 && max_conv == configs.len(), "RESNET_REAL needs BATCH=1 and all 53 convolutions");
        let meta = read_metadata(dir);
        assert_eq!(meta.sf_log, *SF_LOG, "export scale must match the config's scale_factor_log");
        let ld = |name: String| load_witness(&dir.join(name), meta.sf_log, DataType::Int, Role::Constant);
        let cw: Vec<Witness> = (0..max_conv).map(|i| ld(format!("conv_{:03}_weight.bin", i))).collect();
        let cb: Vec<Witness> = (0..max_conv).map(|i| ld(format!("conv_{:03}_bias.bin", i))).collect();
        let x = g.input(vec![3, input_size, input_size], DataType::Int);
        real_x = Some(x);
        let _output = g.pipe(&[x], resnet50_with_biases(cw, cb, ld("fc_weight.bin".into()), Some(ld("fc_bias.bin".into())), num_classes, max_conv));
        println!("Real weights and images: {} (first image {})", dir.display(), real_image);
    }

    // ONE explicit [B,3,H,W] input retains logical B (including nonpowers
    // of two); its raw padded buffer is still [b_pad*4,H,W]. Rather than
    // `batch` separate inputs: the batch is bound inside conv, so the graph
    // stays the size of a single image.
    let xs: Vec<EdgeId> = if let Some(x) = real_x {
        vec![x]
    } else if batch_size > 1 {
        vec![g.input(
            vec![batch_size, 3, input_size, input_size],
            DataType::Uint,
        )]
    } else {
        vec![g.input(vec![3, input_size, input_size], DataType::Uint)]
    };
    if real.is_none() {
        let conv_weights: Vec<Witness> = configs[..max_conv].iter()
            .map(|&(c_in, c_out, kh, kw)| gen_conv_weight(c_in, c_out, kh, kw)).collect();
        let (out_c, _) = resnet50_output_shape(max_conv);
        let fc_weight = gen_fc_weight(out_c, num_classes);
        let fc_bias = gen_fc_bias(num_classes, batch_size);
        let _output = g.pipe(&xs, resnet50(conv_weights, fc_weight, Some(fc_bias), num_classes, max_conv));
    }
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges (batch={})", dag.nodes.len(), dag.num_edges(), batch_size);

    for &x in &xs {
        assert_eq!(witnesses_template[x][0].role, Role::Input,
            "ResNet input edge must be Role::Input (per-proof), not deferred");
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

    let s_pad = input_size.next_power_of_two();
    let input_buf_size = 4 * s_pad * s_pad;

    let label = b"zkml-resnet-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut checked_role = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} proofs x batch {} = {} images:", n_proofs, batch_size, n_proofs * batch_size);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let sp = input_size.next_power_of_two();
        let batch_inputs: Vec<(EdgeId, Witness)> = xs.iter().map(|&x| {
            if let Some(dir) = &real {
                let img = dir.join("images").join(format!("{:05}.bin", real_image + it));
                (x, load_witness(&img, *SF_LOG, DataType::Int, Role::Input))
            } else if batch_size > 1 {
                let lead = batch_size.next_power_of_two() * 4;
                let mut d = rand_field_vec(lead * sp * sp);
                // Keep this historical host sanitization, but correctness
                // no longer depends on it: the registered model masks the
                // explicit B/C/spatial padding before folding the tensor.
                for b in 0..batch_size.next_power_of_two() {
                    let base = (b * 4 + 3) * sp * sp;
                    for v in &mut d[base..base + sp * sp] { *v = AlmostGoldilocksField(0); }
                }
                (x, Witness::new(vec![batch_size, 3, input_size, input_size], d,
                                 DataType::Uint, *SF_LOG, Role::Input))
            } else {
                (x, Witness::new(vec![3, input_size, input_size],
                    rand_field_vec(input_buf_size), DataType::Uint, *SF_LOG, Role::Input))
            }
        }).collect();

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

    let n = n_proofs as f64;
    println!("\n=== Results ({} weight edges deferred, {} reducer steps) ===", n_const, n_steps);
    println!("  prove(defer)  per-img : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-img : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (ResNet-50, weights amortized across {} images)", n_proofs);
}
