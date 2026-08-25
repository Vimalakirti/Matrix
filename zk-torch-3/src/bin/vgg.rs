use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::dag::vgg::{vgg11, vgg16, vgg11_output_shape, vgg_output_shape, VGG_FC_HIDDEN};
use zk_torch_3::dag::verfcnn_vgg;
use zk_torch_3::transcript::Transcript;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

/// Conv layer configs for each VGG variant: (c_in, c_out)
const VGG11_CONV_CONFIGS: [(usize, usize); 8] = [
    (3, 64),                                     // block 1
    (64, 128),                                   // block 2
    (128, 256), (256, 256),                      // block 3
    (256, 512), (512, 512),                      // block 4
    (512, 512), (512, 512),                      // block 5
];

const VGG16_CONV_CONFIGS: [(usize, usize); 13] = [
    (3, 64), (64, 64),                           // block 1
    (64, 128), (128, 128),                       // block 2
    (128, 256), (256, 256), (256, 256),          // block 3
    (256, 512), (512, 512), (512, 512),          // block 4
    (512, 512), (512, 512), (512, 512),          // block 5
];

/// Spatial dimensions at each conv layer for CIFAR-10 (32×32 input, same-padding 3×3 conv).
/// Spatial halves after each complete block's maxpool.
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
        if block_complete {
            spatial /= 2;
        }
        if w_idx >= num_layers {
            break;
        }
    }
    dims
}

fn generate_conv_weights(conv_configs: &[(usize, usize)], num_layers: usize) -> Vec<Witness> {
    let num_convs = num_layers.min(conv_configs.len());
    let mut conv_weights = Vec::with_capacity(num_convs);

    for i in 0..num_convs {
        let (c_in, c_out) = conv_configs[i];
        let kh: usize = 3;
        let kw: usize = 3;
        let kh_pad = kh.next_power_of_two(); // 4
        let kw_pad = kw.next_power_of_two(); // 4
        let c_in_pad = c_in.next_power_of_two();
        let c_out_pad = c_out.next_power_of_two();
        let size = c_out_pad * c_in_pad * kh_pad * kw_pad;
        // Must zero-pad: entries where kw >= 3 or kh >= 3 must be 0
        // Little-endian layout: kw bits (lowest) | kh bits | c_in bits | c_out bits
        let mut data = vec![GoldilocksField(0); size];
        let mut rng = rand::thread_rng();
        for d in 0..c_out {
            for c in 0..c_in {
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let idx = kw_i + kh_i * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                        data[idx] = GoldilocksField((rng.gen::<u32>() % 500) as u64);
                    }
                }
            }
        }
        conv_weights.push(Witness::new(
            vec![c_out, c_in, kh, kw],
            data,
            DataType::Uint,
            0,
            Role::Constant,
        ));
    }

    conv_weights
}

/// Generate full-size conv biases: [C_out, H_out, W_out] with bias_vec[c] broadcast over spatial dims.
fn generate_conv_biases(
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
        let spatial = spatial_dims[i]; // same-padding conv preserves spatial
        let h_out = spatial;
        let w_out = spatial;
        let c_pad = c_out.next_power_of_two();
        let h_pad = h_out.next_power_of_two();
        let w_pad = w_out.next_power_of_two();
        let size = c_pad * h_pad * w_pad;

        // Generate 1D bias vector per channel
        let bias_vec: Vec<GoldilocksField> = (0..c_out)
            .map(|_| GoldilocksField((rng.gen::<u32>() % 100) as u64))
            .collect();

        // Broadcast to full [C_out, H_out, W_out] tensor
        // Little-endian layout: w (stride 1) | h (stride W_pad) | c (stride W_pad * H_pad)
        let mut data = vec![GoldilocksField(0); size];
        for c in 0..c_out {
            for h in 0..h_out {
                for w in 0..w_out {
                    data[w + h * w_pad + c * w_pad * h_pad] = bias_vec[c];
                }
            }
        }

        biases.push(Witness::new(
            vec![c_out, h_out, w_out],
            data,
            DataType::Uint,
            0,
            Role::Constant,
        ));
    }

    biases
}

/// Generate FC weight: [in_dim, out_dim]
fn generate_fc_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for i in 0..in_dim {
        for j in 0..out_dim {
            data[i + j * in_pad] = GoldilocksField((rng.gen::<u32>() % 500) as u64);
        }
    }
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, 0, Role::Constant)
}

/// Generate FC bias: [out_dim]
fn generate_fc_bias(out_dim: usize) -> Witness {
    let out_pad = out_dim.next_power_of_two();
    let mut data = vec![GoldilocksField(0); out_pad];
    let mut rng = rand::thread_rng();
    for i in 0..out_dim {
        data[i] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
    }
    Witness::new(vec![out_dim], data, DataType::Uint, 0, Role::Constant)
}

const VGG11_BLOCKS: [(usize, usize); 5] = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)];
const VGG16_BLOCKS: [(usize, usize); 5] = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)];

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let variant: String = std::env::var("VGG_VARIANT").unwrap_or_else(|_| "16".to_string());
    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let style: String = std::env::var("VGG_STYLE").unwrap_or_else(|_| "paper".to_string());

    let thread_num = rayon::current_num_threads();
    let num_classes = 10usize;

    let mut g = DagBuilder::new();
    let x = g.input(vec![3, 32, 32], DataType::Uint);

    match style.as_str() {
        "verfcnn" => {
            // VerfCNN-style: no bias, single FC layer
            match variant.as_str() {
                "11" => {
                    let max_layers = num_layers.min(VGG11_CONV_CONFIGS.len());
                    println!("=== VGG-11 VerfCNN on CIFAR-10 ({} conv layers) ===", max_layers);
                    println!("Using {} threads", thread_num);
                    let conv_weights = generate_conv_weights(&VGG11_CONV_CONFIGS, num_layers);
                    let (out_c, out_spatial) = verfcnn_vgg::vgg11_output_shape(max_layers);
                    let fc_weight = generate_fc_weight(out_c * out_spatial * out_spatial, num_classes);
                    let _output = g.pipe(&[x], verfcnn_vgg::vgg11(conv_weights, fc_weight));
                }
                _ => {
                    let max_layers = num_layers.min(VGG16_CONV_CONFIGS.len());
                    println!("=== VGG-16 VerfCNN on CIFAR-10 ({} conv layers) ===", max_layers);
                    println!("Using {} threads", thread_num);
                    let conv_weights = generate_conv_weights(&VGG16_CONV_CONFIGS, num_layers);
                    let (out_c, out_spatial) = verfcnn_vgg::vgg_output_shape(max_layers);
                    let fc_weight = generate_fc_weight(out_c * out_spatial * out_spatial, num_classes);
                    let _output = g.pipe(&[x], verfcnn_vgg::vgg16(conv_weights, fc_weight));
                }
            }
        }
        _ => {
            // Paper-style: with bias, 3 FC layers
            match variant.as_str() {
                "11" => {
                    let max_layers = num_layers.min(VGG11_CONV_CONFIGS.len());
                    println!("=== VGG-11 (paper) on CIFAR-10 ({} conv layers, 3 FC layers) ===", max_layers);
                    println!("Using {} threads", thread_num);
                    let conv_weights = generate_conv_weights(&VGG11_CONV_CONFIGS, num_layers);
                    let conv_biases = generate_conv_biases(&VGG11_CONV_CONFIGS, &VGG11_BLOCKS, num_layers);
                    let (out_c, out_spatial) = vgg11_output_shape(max_layers);
                    let flat_dim = out_c * out_spatial * out_spatial;
                    let fc_weights = vec![
                        generate_fc_weight(flat_dim, VGG_FC_HIDDEN),
                        generate_fc_weight(VGG_FC_HIDDEN, VGG_FC_HIDDEN),
                        generate_fc_weight(VGG_FC_HIDDEN, num_classes),
                    ];
                    let fc_biases = vec![
                        generate_fc_bias(VGG_FC_HIDDEN),
                        generate_fc_bias(VGG_FC_HIDDEN),
                        generate_fc_bias(num_classes),
                    ];
                    let _output = g.pipe(&[x], vgg11(conv_weights, conv_biases, fc_weights, fc_biases));
                }
                _ => {
                    let max_layers = num_layers.min(VGG16_CONV_CONFIGS.len());
                    println!("=== VGG-16 (paper) on CIFAR-10 ({} conv layers, 3 FC layers) ===", max_layers);
                    println!("Using {} threads", thread_num);
                    let conv_weights = generate_conv_weights(&VGG16_CONV_CONFIGS, num_layers);
                    let conv_biases = generate_conv_biases(&VGG16_CONV_CONFIGS, &VGG16_BLOCKS, num_layers);
                    let (out_c, out_spatial) = vgg_output_shape(max_layers);
                    let flat_dim = out_c * out_spatial * out_spatial;
                    let fc_weights = vec![
                        generate_fc_weight(flat_dim, VGG_FC_HIDDEN),
                        generate_fc_weight(VGG_FC_HIDDEN, VGG_FC_HIDDEN),
                        generate_fc_weight(VGG_FC_HIDDEN, num_classes),
                    ];
                    let fc_biases = vec![
                        generate_fc_bias(VGG_FC_HIDDEN),
                        generate_fc_bias(VGG_FC_HIDDEN),
                        generate_fc_bias(num_classes),
                    ];
                    let _output = g.pipe(&[x], vgg16(conv_weights, conv_biases, fc_weights, fc_biases));
                }
            }
        }
    }

    println!("Compiling DAG...");
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!("DAG: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    println!("Running forward pass...");
    let input = Witness::new(
        vec![3, 32, 32],
        generate_random_field_vec(4 * 32 * 32), // 3 → pad to 4
        DataType::Uint, 0, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Run: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        // SOUND: size table to true max (unsound <=22 cap removed)
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];

    println!("Committing...");
    let t3 = Instant::now();
    let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
    println!("Commit: {:?}", t3.elapsed());
    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());
        for (i, p) in partitions.iter().enumerate() {
            println!("  Partition {}: {} nodes, {} boundary_in, {} boundary_out",
                i, p.node_ids.len(), p.boundary_input_edges.len(), p.boundary_output_edges.len());
        }

        println!("Proving (parallel, {} partitions)...", num_partitions);
        let mut transcript = Transcript::new(b"zkml-vgg");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-vgg");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-vgg");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-vgg");
        let t5 = Instant::now();
        let verified = dag.verify(
            &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
            &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    }
}
