//! Streaming 3D U-Net (CV volumetric, batch via N inferences) composed with
//! the cross-proof streaming accumulator. Each streamed proof is one volume;
//! WEIGHTS (conv3d + conv-transpose + InstanceNorm gamma/beta + output bias)
//! are Role::Constant (deferred -> amortized into one finalize opening); the
//! per-volume INPUT is Role::Input (committed/opened per-proof).
//!
//! Run with `bench_config.yaml` as args[1]. Env: NUM_LAYERS(1) maps to
//! num_levels; INPUT_D/INPUT_H/INPUT_W spatial dims; N_PROOFS(3)
//! MAX_NUM_VARS(22) NUM_PARTITIONS(1) ZK4_B(21) ZK4_BASE(2).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;
use zk_torch_4::SF_LOG;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::unet3d::{unet3d, ENCODER_LEVELS, DECODER_LEVELS};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 16)
}

/// 3D conv weight `[C_out, C_in, kD, kH, kW]` (or `[C_in, C_out, ...]` for
/// transpose). Little-endian layout `kw | kh | kd | c_in | c_out`; padded
/// entries are zero.
fn gen_conv3d_weight(c_out: usize, c_in: usize, kd: usize, kh: usize, kw: usize) -> Witness {
    let kd_pad = kd.next_power_of_two();
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kd_pad * kh_pad * kw_pad;

    let mut data = zk_torch_4::zero_witness_vec(size);
    let mut rng = rand::thread_rng();
    for co in 0..c_out {
        for ci in 0..c_in {
            for d in 0..kd {
                for h in 0..kh {
                    for w in 0..kw {
                        let idx = w + h * kw_pad + d * kw_pad * kh_pad
                            + ci * kw_pad * kh_pad * kd_pad
                            + co * kw_pad * kh_pad * kd_pad * c_in_pad;
                        data[idx] = AlmostGoldilocksField((rng.gen::<u32>() % 8) as u64);
                    }
                }
            }
        }
    }
    Witness::new(vec![c_out, c_in, kd, kh, kw], data, DataType::Uint, *SF_LOG, Role::Constant)
}

/// InstanceNorm `(gamma[C], beta[C])`. Gamma is forced nonzero so the
/// normalization gate doesn't degenerate.
fn gen_in_params(channels: usize) -> (Witness, Witness) {
    let c_pad = channels.next_power_of_two();
    let mut gamma_data = zk_torch_4::zero_witness_vec(c_pad);
    let mut beta_data = zk_torch_4::zero_witness_vec(c_pad);
    let mut rng = rand::thread_rng();
    for c in 0..channels {
        gamma_data[c] = AlmostGoldilocksField((rng.gen::<u32>() % 8 + 1) as u64);
        beta_data[c] = AlmostGoldilocksField((rng.gen::<u32>() % 8) as u64);
    }
    (
        Witness::new(vec![channels], gamma_data, DataType::Uint, *SF_LOG, Role::Constant),
        Witness::new(vec![channels], beta_data, DataType::Uint, *SF_LOG, Role::Constant),
    )
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

    let num_levels: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let input_d: usize = std::env::var("INPUT_D").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(8);
    let input_h: usize = std::env::var("INPUT_H").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let input_w: usize = std::env::var("INPUT_W").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_proofs: usize = std::env::var("N_PROOFS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch_size: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);

    let eps = 1e-5;
    let actual_levels = num_levels.min(6);

    println!("=== Streaming 3D UNet ({} encoder levels, input [{},{},{}]) ===",
             actual_levels, input_d, input_h, input_w);
    println!("N_PROOFS={} max_num_vars={} partitions={}", n_proofs, max_num_vars, num_partitions);

    // ---- 1. Generate weights (Role::Constant, amortized) ----
    let mut conv_weights = Vec::new();
    for level in 0..actual_levels {
        let (c_in, c_out, _stride) = ENCODER_LEVELS[level];
        conv_weights.push(gen_conv3d_weight(c_out, c_in, 3, 3, 3));
        conv_weights.push(gen_conv3d_weight(c_out, c_out, 3, 3, 3));
    }

    let num_decoder_levels = if actual_levels <= 1 { 0 } else { actual_levels - 1 };
    let dec_offset = 6 - actual_levels;
    for dec_level in 0..num_decoder_levels {
        let dec_config_idx = dec_offset + dec_level;
        let (_c_up_in, c_up_out, c_conv_in) = DECODER_LEVELS[dec_config_idx];
        conv_weights.push(gen_conv3d_weight(c_up_out, c_conv_in, 3, 3, 3));
        conv_weights.push(gen_conv3d_weight(c_up_out, c_up_out, 3, 3, 3));
    }

    let last_c = if num_decoder_levels > 0 {
        DECODER_LEVELS[dec_offset + num_decoder_levels - 1].1
    } else {
        ENCODER_LEVELS[actual_levels - 1].1
    };
    conv_weights.push(gen_conv3d_weight(3, last_c, 1, 1, 1));

    let mut conv_transpose_weights = Vec::new();
    for dec_level in 0..num_decoder_levels {
        let dec_config_idx = dec_offset + dec_level;
        let (c_up_in, c_up_out, _) = DECODER_LEVELS[dec_config_idx];
        conv_transpose_weights.push(gen_conv3d_weight(c_up_in, c_up_out, 2, 2, 2));
    }

    let mut in_gammas = Vec::new();
    let mut in_betas = Vec::new();
    for level in 0..actual_levels {
        let c_out = ENCODER_LEVELS[level].1;
        let (g1, b1) = gen_in_params(c_out);
        let (g2, b2) = gen_in_params(c_out);
        in_gammas.push(g1); in_gammas.push(g2);
        in_betas.push(b1); in_betas.push(b2);
    }
    for dec_level in 0..num_decoder_levels {
        let dec_config_idx = dec_offset + dec_level;
        let c_up_out = DECODER_LEVELS[dec_config_idx].1;
        let (g1, b1) = gen_in_params(c_up_out);
        let (g2, b2) = gen_in_params(c_up_out);
        in_gammas.push(g1); in_gammas.push(g2);
        in_betas.push(b1); in_betas.push(b2);
    }

    // Output bias [3, D, H, W] broadcast over spatial.
    let d_pad = input_d.next_power_of_two();
    let h_pad = input_h.next_power_of_two();
    let w_pad = input_w.next_power_of_two();
    let out_spatial = d_pad * h_pad * w_pad;
    let out_c_pad = 4; // 3.next_power_of_two()
    let bias_size = out_c_pad * out_spatial;
    let output_bias = Witness::new(
        vec![3, input_d, input_h, input_w],
        rand_field_vec(bias_size),
        DataType::Uint, *SF_LOG, Role::Constant,
    );

    // ---- 2. Build DAG once ----
    let mut g = DagBuilder::new();
    let xs: Vec<EdgeId> = (0..batch_size)
        .map(|_| g.input(vec![1, input_d, input_h, input_w], DataType::Uint))
        .collect();
    let _ = g.pipe(&xs, unet3d(
        conv_weights, conv_transpose_weights, in_gammas, in_betas,
        Some(output_bias), actual_levels, eps,
    ));

    let t1 = Instant::now();
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges, batch={})",
             t1.elapsed(), dag.nodes.len(), dag.num_edges(), batch_size);

    for &x in &xs {
        assert_eq!(witnesses_template[x][0].role, Role::Input,
            "UNet input edge must be Role::Input (per-proof), not deferred");
    }

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        // Report the EFFECTIVE partition count: set_partition_boundaries
        // clamps to what the graph supports, so the requested value can
        // overstate it and would land in the CSV as such.
        println!("Partitions: {} (boundaries: {})",
                 dag.boundary_edges.len() + 1, dag.boundary_edges.len());
    }

    // ---- 3. Commit constants once (weights, amortized) ----
    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses_template, &mut store);
    println!("Offline commit (weights, amortized): {:.2}ms", ms(t_off.elapsed()));

    let c_in_pad = 1usize.next_power_of_two();
    let input_size = c_in_pad * d_pad * h_pad * w_pad;

    let label = b"zkml-unet3d-streaming";
    let mut prover_acc = AccumulatorState::new(label);
    let mut verifier_acc = VerifierAccumulator::new(label);
    let mut checked_role = false;

    let (mut t_run, mut t_commit, mut t_prove, mut t_verify, mut t_acc, mut t_accv) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);

    let mut proof_bytes = 0usize;
    let mut breakdown: Option<String> = None;

    println!("Streaming {} volumes:", n_proofs);
    for it in 0..n_proofs {
        let mut witnesses = witnesses_template.clone();
        let batch_inputs: Vec<(EdgeId, Witness)> = xs.iter().map(|&x| (x, Witness::new(
            vec![1, input_d, input_h, input_w],
            rand_field_vec(input_size),
            DataType::Uint, *SF_LOG, Role::Input,
        ))).collect();

        let s0 = Instant::now();
        dag.run(&mut witnesses, &batch_inputs);
        let d_run = s0.elapsed(); t_run += d_run;


        store.clear_non_constants(&witnesses);
        let s1 = Instant::now();
        dag.commit_remaining(&witnesses, &mut store);
        let d_commit = s1.elapsed(); t_commit += d_commit;

        let mut tp = Transcript::new(b"per-vol");
        let s2 = Instant::now();
        let (dp, fp) = dag.prove_with_fold_tree_modes(&witnesses, &store, &mut tp, true);
        let d_prove = s2.elapsed(); t_prove += d_prove;

        let mut tv = Transcript::new(b"per-vol");
        let s3 = Instant::now();
        let r = dag.verify_with_fold_tree_deferred(&witnesses, &store, &dp, &fp, &mut tv);
        let d_verify = s3.elapsed(); t_verify += d_verify;
        if !r.ok { eprintln!("per-volume verify failed at {}", it); return; }

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
    println!("  prove(defer)  per-vol : {:>8.2}ms", ms(t_prove) / n);
    println!("  acc-update    per-vol : {:>8.2}ms", ms(t_acc) / n);
    println!("  finalize / N          : {:>8.2}ms", ms(t_finalize) / n);
    println!("  finalize (one-time)   : {:>8.2}ms  (+verify {:.2}ms)", ms(t_finalize), ms(t_fv));
    println!("  proof         per-unit: {:>8} bytes", proof_bytes / n_proofs);
    if let Some(b) = &breakdown { println!("{}", b); }
    println!("  proof     finalize    : {:>8} bytes", ser_len(&final_proof));
    println!("\nVerified: true (3D UNet, weights amortized across {} volumes = {} proofs x batch {})",
        n_proofs * batch_size, n_proofs, batch_size);
}
