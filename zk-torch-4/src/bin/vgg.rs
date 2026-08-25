//! VGG-11 / VGG-16 end-to-end prover binary. Ports zk-torch-3's
//! `bin/vgg.rs`. Supports both styles via `VGG_STYLE`:
//!   * `paper` (default) — Conv → bias → ReLU, three FC layers with bias.
//!   * `verfcnn` — Conv → ReLU (no bias), single FC layer (no bias).
//!
//! Defaults: VGG-16, 2 conv layers, CIFAR-10 (`[3, 32, 32]`, 10 classes).
//! Override `VGG_STYLE` (`"paper"` | `"verfcnn"`), `VGG_VARIANT`
//! (`"11"` | `"16"`), `NUM_LAYERS`, `MAX_NUM_VARS`, `ZK4_B`, `ZK4_BASE`
//! via env vars.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::vgg::{vgg11, vgg16, vgg11_output_shape, vgg_output_shape, VGG_FC_HIDDEN};
use zk_torch_4::dag::verfcnn_vgg;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;

/// Magnitudes kept small (`% 32`) so the 27-term 3×3-conv sum (max ≈ 27·31²
/// ≈ 26k) stays well inside the signed b=21 commit range used by the
/// fold-tree opening. `% 500` (zk-torch-3 default) overflows it.
fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 2)
}

const VGG11_CONV_CONFIGS: [(usize, usize); 8] = [
    (3, 64),
    (64, 128),
    (128, 256), (256, 256),
    (256, 512), (512, 512),
    (512, 512), (512, 512),
];

const VGG16_CONV_CONFIGS: [(usize, usize); 13] = [
    (3, 64), (64, 64),
    (64, 128), (128, 128),
    (128, 256), (256, 256), (256, 256),
    (256, 512), (512, 512), (512, 512),
    (512, 512), (512, 512), (512, 512),
];

const VGG11_BLOCKS: [(usize, usize); 5] = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)];
const VGG16_BLOCKS: [(usize, usize); 5] = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)];

/// Spatial dims at each conv layer for 32×32 input (same-padding 3×3 conv).
/// Halves after each *complete* block's maxpool.
fn conv_spatial_dims(blocks: &[(usize, usize)], num_layers: usize) -> Vec<usize> {
    let mut spatial = 32usize;
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

/// Conv weights left ALL ZERO for the smoke test: across multiple
/// VGG conv stages the 3×3·c_in-term accumulator overflows the
/// `[0, 2^TABLE_SIZE_LOG)` range table (e.g. `% 2` weights · ReLU output
/// from previous stage exceeds 1024 within 2-3 stages). All-zero conv
/// makes every ReLU input = the conv bias only, which is `% 2` and
/// trivially in range. Soundness still depends on the conv proof being
/// valid for the all-zero polynomial.
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
        out.push(Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, 0, Role::Constant));
    }
    out
}

fn gen_conv_biases(
    conv_configs: &[(usize, usize)],
    blocks: &[(usize, usize)],
    num_layers: usize,
) -> Vec<Witness> {
    let num_convs = num_layers.min(conv_configs.len());
    let spatial_dims = conv_spatial_dims(blocks, num_layers);
    let mut biases = Vec::with_capacity(num_convs);
    let mut rng = rand::thread_rng();
    for i in 0..num_convs {
        let (_, c_out) = conv_configs[i];
        let spatial = spatial_dims[i];
        let (h_out, w_out) = (spatial, spatial);
        let c_pad = c_out.next_power_of_two();
        let h_pad = h_out.next_power_of_two();
        let w_pad = w_out.next_power_of_two();
        let size = c_pad * h_pad * w_pad;
        let bias_vec: Vec<AlmostGoldilocksField> = (0..c_out)
            .map(|_| AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64))
            .collect();
        let mut data = zk_torch_4::zero_witness_vec(size);
        for c in 0..c_out {
            for h in 0..h_out {
                for w in 0..w_out {
                    data[w + h * w_pad + c * w_pad * h_pad] = bias_vec[c];
                }
            }
        }
        biases.push(Witness::new(vec![c_out, h_out, w_out], data, DataType::Uint, 0, Role::Constant));
    }
    biases
}

/// FC weights left ALL ZERO so the post-FC ReLU range checks (table [0,1024))
/// can't be triggered by the natural Σ over `in_dim` terms (in_dim is up to
/// 16384 for VGG-11's first FC, which would overflow ±1024 even with binary
/// weights). With all-zero weights the FC output equals the bias only, which
/// is `% 2` → trivially in range. Smoke-test only.
fn gen_fc_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, 0, Role::Constant)
}

fn gen_fc_bias(out_dim: usize) -> Witness {
    let out_pad = out_dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(out_pad);
    let mut rng = rand::thread_rng();
    for i in 0..out_dim {
        data[i] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
    }
    Witness::new(vec![out_dim], data, DataType::Uint, 0, Role::Constant)
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

    let variant: String = std::env::var("VGG_VARIANT").unwrap_or_else(|_| "16".to_string());
    let style: String = std::env::var("VGG_STYLE").unwrap_or_else(|_| "paper".to_string());
    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(2);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    let num_classes = 10usize;

    let mut g = DagBuilder::new();
        // BATCH inputs per proof. The builder creates the weight edges ONCE and
    // reuses them across batch elements, so batching amortizes the weight
    // commitment instead of replicating it.
    let batch: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(batch >= 1, "BATCH must be >= 1");
    let xs: Vec<_> = (0..batch)
        .map(|_| g.input(vec![3, 32, 32], DataType::Uint))
        .collect();
    let x = xs[0];
    let _ = x;

    match style.as_str() {
        "verfcnn" => {
            // verfcnn-style: no conv bias, single FC layer.
            match variant.as_str() {
                "11" => {
                    let max_layers = num_layers.min(VGG11_CONV_CONFIGS.len());
                    println!("=== VGG-11 (verfcnn) on CIFAR-10 ({} conv layers) ===", max_layers);
                    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());
                    let conv_weights = gen_conv_weights(&VGG11_CONV_CONFIGS, num_layers);
                    let (out_c, out_spatial) = verfcnn_vgg::vgg11_output_shape(max_layers);
                    let fc_weight = gen_fc_weight(out_c * out_spatial * out_spatial, num_classes);
                    let _ = g.pipe(&xs, verfcnn_vgg::vgg11(conv_weights, fc_weight));
                }
                _ => {
                    let max_layers = num_layers.min(VGG16_CONV_CONFIGS.len());
                    println!("=== VGG-16 (verfcnn) on CIFAR-10 ({} conv layers) ===", max_layers);
                    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());
                    let conv_weights = gen_conv_weights(&VGG16_CONV_CONFIGS, num_layers);
                    let (out_c, out_spatial) = verfcnn_vgg::vgg_output_shape(max_layers);
                    let fc_weight = gen_fc_weight(out_c * out_spatial * out_spatial, num_classes);
                    let _ = g.pipe(&xs, verfcnn_vgg::vgg16(conv_weights, fc_weight));
                }
            }
        }
        _ => {
            // Paper style: conv bias + 3 FC layers with bias.
            match variant.as_str() {
                "11" => {
                    let max_layers = num_layers.min(VGG11_CONV_CONFIGS.len());
                    println!("=== VGG-11 (paper) on CIFAR-10 ({} conv layers, 3 FC layers) ===", max_layers);
                    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());
                    let conv_weights = gen_conv_weights(&VGG11_CONV_CONFIGS, num_layers);
                    let conv_biases = gen_conv_biases(&VGG11_CONV_CONFIGS, &VGG11_BLOCKS, num_layers);
                    let (out_c, out_spatial) = vgg11_output_shape(max_layers, 32);
                    let flat_dim = out_c * out_spatial * out_spatial;
                    let fc_weights = vec![
                        gen_fc_weight(flat_dim, VGG_FC_HIDDEN),
                        gen_fc_weight(VGG_FC_HIDDEN, VGG_FC_HIDDEN),
                        gen_fc_weight(VGG_FC_HIDDEN, num_classes),
                    ];
                    let fc_biases = vec![
                        gen_fc_bias(VGG_FC_HIDDEN),
                        gen_fc_bias(VGG_FC_HIDDEN),
                        gen_fc_bias(num_classes),
                    ];
                    let _ = g.pipe(&xs, vgg11(conv_weights, conv_biases, fc_weights, fc_biases));
                }
                _ => {
                    let max_layers = num_layers.min(VGG16_CONV_CONFIGS.len());
                    println!("=== VGG-16 (paper) on CIFAR-10 ({} conv layers, 3 FC layers) ===", max_layers);
                    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());
                    let conv_weights = gen_conv_weights(&VGG16_CONV_CONFIGS, num_layers);
                    let conv_biases = gen_conv_biases(&VGG16_CONV_CONFIGS, &VGG16_BLOCKS, num_layers);
                    let (out_c, out_spatial) = vgg_output_shape(max_layers, 32);
                    let flat_dim = out_c * out_spatial * out_spatial;
                    let fc_weights = vec![
                        gen_fc_weight(flat_dim, VGG_FC_HIDDEN),
                        gen_fc_weight(VGG_FC_HIDDEN, VGG_FC_HIDDEN),
                        gen_fc_weight(VGG_FC_HIDDEN, num_classes),
                    ];
                    let fc_biases = vec![
                        gen_fc_bias(VGG_FC_HIDDEN),
                        gen_fc_bias(VGG_FC_HIDDEN),
                        gen_fc_bias(num_classes),
                    ];
                    let _ = g.pipe(&xs, vgg16(conv_weights, conv_biases, fc_weights, fc_biases));
                }
            }
        }
    }

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // Forward pass: 3 → pad to 4 channels.
    let input = Witness::new(
        vec![3, 32, 32],
        rand_field_vec(4 * 32 * 32),
        DataType::Uint, 0, Role::Input,
    );
    let t2 = Instant::now();
    let inputs: Vec<_> = (0..batch).map(|i| (i, input.clone())).collect();
    dag.run(&mut witnesses, &inputs);
    println!("Forward: {:?}", t2.elapsed());

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

    let mut t_prove = Transcript::new(b"zkml-vgg");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-vgg");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());

    let sz = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", sz);

    println!("\nVerified: {}", verified);
}
