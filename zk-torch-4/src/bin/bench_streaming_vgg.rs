//! Streaming VGG-16 (CV, batch via N inferences) composed with the
//! cross-proof streaming accumulator. Hardcodes the standard `vgg16`
//! "paper" variant (Conv → bias → ReLU, three FC layers with bias) on
//! CIFAR-10 (`[3, 32, 32]`, 10 classes). Each streamed proof is one image;
//! WEIGHTS (conv + bias + fc) are Role::Constant (deferred → amortized into
//! one finalize opening); the per-image INPUT is Role::Input (committed/
//! opened per-proof).
//!
//! Run with `bench_config.yaml` as args[1]. Env: NUM_LAYERS(2) N_PROOFS(3)
//! MAX_NUM_VARS(22) NUM_PARTITIONS(1) ZK4_B(21) ZK4_BASE(2).

use std::time::{Duration, Instant};

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
use zk_torch_4::dag::vgg::{vgg16, vgg_output_shape, VGG_FC_HIDDEN};
use zk_torch_4::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::ser_len;
use zk_torch_4::SF_LOG;

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1e3 }

const VGG16_CONV_CONFIGS: [(usize, usize); 13] = [
    (3, 64), (64, 64),
    (64, 128), (128, 128),
    (128, 256), (256, 256), (256, 256),
    (256, 512), (512, 512), (512, 512),
    (512, 512), (512, 512), (512, 512),
];

const VGG16_BLOCKS: [(usize, usize); 5] = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)];

/// Magnitudes kept small (`% 2`) so the conv/ReLU accumulators stay well
/// inside the signed b=21 commit range used by the fold-tree opening.
fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 128)
}

/// Spatial dims at each conv layer for 32×32 input (same-padding 3×3 conv).
/// Halves after each *complete* block's maxpool.
fn conv_spatial_dims(blocks: &[(usize, usize)], num_layers: usize, input_size: usize) -> Vec<usize> {
    let mut spatial = input_size;
    let mut dims = Vec::new();
    let mut w_idx = 0;
    for (num_convs, _) in blocks {
        let mut block_complete = true;
        for _ in 0..*num_convs {
            if w_idx >= num_layers {
                block_complete = false;
                break;
            }
            dims.push(spatial);
            w_idx += 1;
        }
        if block_complete { spatial /= 2; }
        if w_idx >= num_layers { break; }
    }
    dims
}

/// Conv weights left ALL ZERO for the smoke test: across multiple VGG conv
/// stages the 3×3·c_in-term accumulator overflows the range table. All-zero
/// conv makes every ReLU input = the conv bias only, which is `% 2` and
/// trivially in range.
fn gen_conv_weights(conv_configs: &[(usize, usize)], num_layers: usize) -> Vec<Witness> {
    let num_convs = num_layers.min(conv_configs.len());
    let mut out = Vec::with_capacity(num_convs);
    for i in 0..num_convs {
        let (c_in, c_out) = conv_configs[i];
        let (kh, kw) = (3usize, 3usize);
        let kh_pad = kh.next_power_of_two();
        let kw_pad = kw.next_power_of_two();
        let c_in_pad = c_in.next_power_of_two();
        let c_out_pad = c_out.next_power_of_two();
        let size = c_out_pad * c_in_pad * kh_pad * kw_pad;
        let data = zk_torch_4::zero_witness_vec(size);
        out.push(Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, *SF_LOG, Role::Constant));
    }
    out
}

fn gen_conv_biases(
    conv_configs: &[(usize, usize)],
    blocks: &[(usize, usize)],
    num_layers: usize,
    input_size: usize,
    batch: usize,
) -> Vec<Witness> {
    let num_convs = num_layers.min(conv_configs.len());
    let spatial_dims = conv_spatial_dims(blocks, num_layers, input_size);
    let mut biases = Vec::with_capacity(num_convs);
    let mut rng = rand::thread_rng();
    for i in 0..num_convs {
        let (_, c_out) = conv_configs[i];
        let spatial = spatial_dims[i];
        let (h_out, w_out) = (spatial, spatial);
        let c_pad = c_out.next_power_of_two();
        let h_pad = h_out.next_power_of_two();
        let w_pad = w_out.next_power_of_two();
        // Folded batch: the bias is already a full-size tensor (the per-channel
        // value replicated across every spatial slot), so batching it means
        // tiling the same block once per image. A [C,1,1] broadcast cannot do
        // this once B and C share one axis.
        let b_pad = batch.next_power_of_two();
        let size = b_pad * c_pad * h_pad * w_pad;
        let bias_vec: Vec<AlmostGoldilocksField> = (0..c_out)
            .map(|_| AlmostGoldilocksField((rng.gen::<u32>() % 128) as u64))
            .collect();
        let mut data = zk_torch_4::zero_witness_vec(size);
        for b in 0..batch {
            for c in 0..c_out {
                for h in 0..h_out {
                    for w in 0..w_out {
                        data[b * c_pad * w_pad * h_pad
                             + w + h * w_pad + c * w_pad * h_pad] = bias_vec[c];
                    }
                }
            }
        }
        let bshape = if batch > 1 {
            vec![b_pad * c_pad, h_out, w_out]
        } else {
            vec![c_out, h_out, w_out]
        };
        biases.push(Witness::new(bshape, data, DataType::Uint, *SF_LOG, Role::Constant));
    }
    biases
}

/// FC weights left ALL ZERO so the post-FC ReLU range checks can't be
/// triggered by the natural Σ over `in_dim` terms. With all-zero weights the
/// FC output equals the bias only, which is `% 2` → trivially in range.
fn gen_fc_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

/// FC bias. When batched, the FC head runs in [features, B] form, so the bias
/// is declared [out_dim, 1]: broadcasting matches TRAILING dimensions, and a
/// bare [out_dim] would align against the batch axis instead of the features.
/// The data is unchanged -- [out_dim] and [out_dim, 1] are the same MLE.
fn gen_fc_bias(out_dim: usize, batch: usize) -> Witness {
    let out_pad = out_dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(out_pad);
    let mut rng = rand::thread_rng();
    for i in 0..out_dim {
        data[i] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
    }
    let shape = if batch > 1 { vec![out_dim, 1] } else { vec![out_dim] };
    Witness::new(shape, data, DataType::Uint, *SF_LOG, Role::Constant)
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

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let n_proofs: usize = std::env::var("N_PROOFS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok().and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let batch_size: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_classes = 10usize;
    let input_size: usize = std::env::var("INPUT_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    let max_layers = num_layers.min(VGG16_CONV_CONFIGS.len());
    println!("=== Streaming VGG-16 (paper) on CIFAR-10 ({} conv layers, 3 FC layers) ===", max_layers);
    println!("N_PROOFS={} max_num_vars={} partitions={}", n_proofs, max_num_vars, num_partitions);

    let conv_weights = gen_conv_weights(&VGG16_CONV_CONFIGS, num_layers);
    let conv_biases =
        gen_conv_biases(&VGG16_CONV_CONFIGS, &VGG16_BLOCKS, num_layers, input_size, batch_size);
    let (out_c, out_spatial) = vgg_output_shape(max_layers, input_size);
    let flat_dim = out_c * out_spatial * out_spatial;
    let fc_weights = vec![
        gen_fc_weight(flat_dim, VGG_FC_HIDDEN),
        gen_fc_weight(VGG_FC_HIDDEN, VGG_FC_HIDDEN),
        gen_fc_weight(VGG_FC_HIDDEN, num_classes),
    ];
    let fc_biases = vec![
        gen_fc_bias(VGG_FC_HIDDEN, batch_size),
        gen_fc_bias(VGG_FC_HIDDEN, batch_size),
        gen_fc_bias(num_classes, batch_size),
    ];

    let mut g = DagBuilder::new();
    // ONE folded input [b_pad*4, H, W] (RGB pads to 4 channels) rather than
    // `batch` separate inputs: the batch is bound inside conv, so the graph
    // stays the size of a single image.
    let xs: Vec<EdgeId> = if batch_size > 1 {
        vec![g.input(
            vec![batch_size.next_power_of_two() * 4, input_size, input_size],
            DataType::Uint,
        )]
    } else {
        vec![g.input(vec![3, input_size, input_size], DataType::Uint)]
    };
    let _output = g.pipe(&xs, vgg16(conv_weights, conv_biases, fc_weights, fc_biases));
    let (mut dag, witnesses_template) = g.compile();
    println!("Compile: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    for &x in &xs { assert_eq!(witnesses_template[x][0].role, Role::Input,
        "VGG input edge must be Role::Input (per-proof), not deferred"); }

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

    let label = b"zkml-vgg-streaming";
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
        // Forward pass input: 3 → pad to 4 channels.
        let sp = input_size.next_power_of_two();
        let (in_shape, in_len) = if batch_size > 1 {
            let lead = batch_size.next_power_of_two() * 4;
            (vec![lead, input_size, input_size], lead * sp * sp)
        } else {
            (vec![3, input_size, input_size], 4 * sp * sp)
        };
        let batch_inputs: Vec<(EdgeId, Witness)> = xs.iter().map(|&x| {
            let mut d = rand_field_vec(in_len);
            if batch_size > 1 {
                // RGB pads to 4 channels, so channel 3 of every image is
                // PADDING and must be zero. Unbatched, ZeroPad sanitized it
                // because it knew channels == 3; once the batch is folded into
                // the channel axis nothing downstream can tell padding from
                // data, and conv would fold over 3 channels while the opening
                // integrates over 4 -- an honest-prover verify failure.
                for b in 0..batch_size.next_power_of_two() {
                    for c in 3..4 {
                        let base = (b * 4 + c) * sp * sp;
                        for v in &mut d[base..base + sp * sp] {
                            *v = AlmostGoldilocksField(0);
                        }
                    }
                }
            }
            (x, Witness::new(in_shape.clone(), d, DataType::Uint, *SF_LOG, Role::Input))
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
    println!("\nVerified: true (VGG-16, weights amortized across {} images)", n_proofs);
}
