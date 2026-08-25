//! 3D UNet end-to-end prover binary. Ports zk-torch-3's `bin/unet3d.rs`
//! to the zk-torch-4 commit/prove API.
//!
//! Defaults: input `[1, 16, 16, 16]` (small for quick smoke; full MLPerf
//! uses 128³) with as many encoder levels as the input geometry supports
//! (each dim must be divisible by 2^(levels-1), so 6 levels need >=32³),
//! keeping the bottleneck >= 2³ (4 levels for 16³; see the depth-default
//! comment in `main`). Override `NUM_LAYERS`, `INPUT_D`/`INPUT_H`/`INPUT_W`,
//! `MAX_NUM_VARS`, `ZK4_B`, `ZK4_BASE` via env vars.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::unet3d::{unet3d, unet3d_max_levels, ENCODER_LEVELS, DECODER_LEVELS};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;

/// Tiny magnitudes (`% 2`) so conv accumulators and InstanceNorm
/// intermediates stay inside the range lookup table AND inside the b=21
/// signed bit decomposition of the fold-tree opening (zk-torch-3's `% 500`
/// default overflows both — see resnet.rs/vgg.rs for the same convention).
fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 2)
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
                        data[idx] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
                    }
                }
            }
        }
    }
    Witness::new(vec![c_out, c_in, kd, kh, kw], data, DataType::Uint, 0, Role::Constant)
}

/// InstanceNorm `(gamma[C], beta[C])`. Gamma is forced nonzero so the
/// normalization gate doesn't degenerate.
fn gen_in_params(channels: usize) -> (Witness, Witness) {
    let c_pad = channels.next_power_of_two();
    let mut gamma_data = zk_torch_4::zero_witness_vec(c_pad);
    let mut beta_data = zk_torch_4::zero_witness_vec(c_pad);
    let mut rng = rand::thread_rng();
    for c in 0..channels {
        gamma_data[c] = AlmostGoldilocksField((rng.gen::<u32>() % 2 + 1) as u64);
        beta_data[c] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
    }
    (
        Witness::new(vec![channels], gamma_data, DataType::Uint, 0, Role::Constant),
        Witness::new(vec![channels], beta_data, DataType::Uint, 0, Role::Constant),
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

    let input_d: usize = std::env::var("INPUT_D").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let input_h: usize = std::env::var("INPUT_H").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let input_w: usize = std::env::var("INPUT_W").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    // Default depth: deepest UNet the input geometry supports (each spatial
    // dim must be divisible by 2^(levels-1)). A 1×1×1 bottleneck is fine:
    // the ConvTranspose3D degenerate-input verify bug (skipped 0-round
    // sumcheck-3 transcript replay) is fixed in conv.rs.
    let num_levels: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| unet3d_max_levels(input_d, input_h, input_w));
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);

    let eps = 1e-5;
    let actual_levels = num_levels.min(6);

    println!("=== 3D UNet on Almost-Goldilocks ({} encoder levels, input [{},{},{}]) ===",
             actual_levels, input_d, input_h, input_w);
    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());

    // ---- 1. Generate weights ----
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
        DataType::Uint, 0, Role::Constant,
    );

    // ---- 2. Build DAG ----
    let mut g = DagBuilder::new();
        // BATCH inputs per proof. The builder creates the weight edges ONCE and
    // reuses them across batch elements, so batching amortizes the weight
    // commitment instead of replicating it.
    let batch: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(batch >= 1, "BATCH must be >= 1");
    let xs: Vec<_> = (0..batch)
        .map(|_| g.input(vec![1, input_d, input_h, input_w], DataType::Uint))
        .collect();
    let x = xs[0];
    let _ = x;
    let _ = g.pipe(&xs, unet3d(
        conv_weights, conv_transpose_weights, in_gammas, in_betas,
        Some(output_bias), actual_levels, eps,
    ));

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // ---- 3. Forward pass ----
    let c_in_pad = 1usize.next_power_of_two();
    let input_size = c_in_pad * d_pad * h_pad * w_pad;
    let input = Witness::new(
        vec![1, input_d, input_h, input_w],
        rand_field_vec(input_size),
        DataType::Uint, 0, Role::Input,
    );
    let t2 = Instant::now();
    let inputs: Vec<_> = (0..batch).map(|i| (i, input.clone())).collect();
    dag.run(&mut witnesses, &inputs);
    println!("Forward: {:?}", t2.elapsed());

    // ---- 4. Commit ----
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses, &mut store);
    let offline_commit = t_off.elapsed();
    let t_on = Instant::now();
    dag.commit_remaining(&witnesses, &mut store);
    let online_commit = t_on.elapsed();
    println!("Commit (offline, amortized): {:?}", offline_commit);
    println!("Commit (online, prover time): {:?}", online_commit);

    // ---- 5. Prove + verify ----
    let mut t_prove = Transcript::new(b"zkml-unet3d");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-unet3d");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
}
