use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::dag::unet3d::{unet3d, ENCODER_LEVELS, DECODER_LEVELS};
use zk_torch_3::transcript::Transcript;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

/// Generate 3D conv weight: [C_out, C_in, kD, kH, kW] (or [C_in, C_out, kD, kH, kW] for transpose).
/// Little-endian layout: kw | kh | kd | c_in | c_out.
/// For 3×3×3 kernels: zero-pad entries where kw >= 3, kh >= 3, or kd >= 3.
fn generate_conv3d_weight(c_out: usize, c_in: usize, kd: usize, kh: usize, kw: usize) -> Witness {
    let kd_pad = kd.next_power_of_two();
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kd_pad * kh_pad * kw_pad;

    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for co in 0..c_out {
        for ci in 0..c_in {
            for d in 0..kd {
                for h in 0..kh {
                    for w in 0..kw {
                        let idx = w + h * kw_pad + d * kw_pad * kh_pad
                            + ci * kw_pad * kh_pad * kd_pad
                            + co * kw_pad * kh_pad * kd_pad * c_in_pad;
                        data[idx] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
                    }
                }
            }
        }
    }
    Witness::new(
        vec![c_out, c_in, kd, kh, kw],
        data,
        DataType::Uint, 0, Role::Constant,
    )
}

/// Generate InstanceNorm parameters: gamma[C] and beta[C].
fn generate_instancenorm_params(channels: usize) -> (Witness, Witness) {
    let c_pad = channels.next_power_of_two();
    let mut gamma_data = vec![GoldilocksField(0); c_pad];
    let mut beta_data = vec![GoldilocksField(0); c_pad];
    let mut rng = rand::thread_rng();
    for c in 0..channels {
        gamma_data[c] = GoldilocksField((rng.gen::<u32>() % 100 + 1) as u64); // nonzero
        beta_data[c] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
    }
    (
        Witness::new(vec![channels], gamma_data, DataType::Uint, 0, Role::Constant),
        Witness::new(vec![channels], beta_data, DataType::Uint, 0, Role::Constant),
    )
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_levels: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);

    // Use smaller spatial size for testing (full model uses 128³)
    let input_d: usize = std::env::var("INPUT_D").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let input_h: usize = std::env::var("INPUT_H").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let input_w: usize = std::env::var("INPUT_W").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);

    let eps = 1e-5;
    let actual_levels = num_levels.min(6);

    let thread_num = rayon::current_num_threads();
    println!("=== 3D UNet ({} encoder levels, input [{},{},{}]) ===", actual_levels, input_d, input_h, input_w);
    println!("Using {} threads", thread_num);

    // Generate encoder conv weights (2 per level)
    let mut conv_weights = Vec::new();
    for level in 0..actual_levels {
        let (c_in, c_out, _stride) = ENCODER_LEVELS[level];
        conv_weights.push(generate_conv3d_weight(c_out, c_in, 3, 3, 3));  // first conv
        conv_weights.push(generate_conv3d_weight(c_out, c_out, 3, 3, 3)); // second conv
    }

    // Generate decoder conv weights (2 per decoder level)
    // With N encoder levels, decoder uses DECODER_LEVELS[6-N .. 4]
    let num_decoder_levels = if actual_levels <= 1 { 0 } else { actual_levels - 1 };
    let dec_offset = 6 - actual_levels; // starting index into DECODER_LEVELS
    for dec_level in 0..num_decoder_levels {
        let dec_config_idx = dec_offset + dec_level;
        let (_c_up_in, c_up_out, c_conv_in) = DECODER_LEVELS[dec_config_idx];
        conv_weights.push(generate_conv3d_weight(c_up_out, c_conv_in, 3, 3, 3));  // first conv after concat
        conv_weights.push(generate_conv3d_weight(c_up_out, c_up_out, 3, 3, 3));   // second conv
    }

    // Output 1×1×1 conv
    let last_c = if num_decoder_levels > 0 {
        DECODER_LEVELS[dec_offset + num_decoder_levels - 1].1
    } else {
        ENCODER_LEVELS[actual_levels - 1].1
    };
    conv_weights.push(generate_conv3d_weight(3, last_c, 1, 1, 1));

    // Generate ConvTranspose weights: [C_in, C_out, kD, kH, kW]
    let mut conv_transpose_weights = Vec::new();
    for dec_level in 0..num_decoder_levels {
        let dec_config_idx = dec_offset + dec_level;
        let (c_up_in, c_up_out, _) = DECODER_LEVELS[dec_config_idx];
        conv_transpose_weights.push(generate_conv3d_weight(c_up_in, c_up_out, 2, 2, 2));
    }

    // Generate InstanceNorm parameters
    // Encoder: 2 per level. Decoder: 2 per decoder level.
    let mut in_gammas = Vec::new();
    let mut in_betas = Vec::new();
    // Encoder IN params
    for level in 0..actual_levels {
        let c_out = ENCODER_LEVELS[level].1;
        let (g1, b1) = generate_instancenorm_params(c_out);
        let (g2, b2) = generate_instancenorm_params(c_out);
        in_gammas.push(g1);
        in_gammas.push(g2);
        in_betas.push(b1);
        in_betas.push(b2);
    }
    // Decoder IN params
    for dec_level in 0..num_decoder_levels {
        let dec_config_idx = dec_offset + dec_level;
        let c_up_out = DECODER_LEVELS[dec_config_idx].1;
        let (g1, b1) = generate_instancenorm_params(c_up_out);
        let (g2, b2) = generate_instancenorm_params(c_up_out);
        in_gammas.push(g1);
        in_gammas.push(g2);
        in_betas.push(b1);
        in_betas.push(b2);
    }

    // Output bias [3, D, H, W] — broadcast over spatial
    let d_pad = input_d.next_power_of_two();
    let h_pad = input_h.next_power_of_two();
    let w_pad = input_w.next_power_of_two();
    let out_spatial = d_pad * h_pad * w_pad;
    let out_c_pad = 4; // 3.next_power_of_two()
    let bias_size = out_c_pad * out_spatial;
    let output_bias = Witness::new(
        vec![3, input_d, input_h, input_w],
        generate_random_field_vec(bias_size),
        DataType::Uint, 0, Role::Constant,
    );

    // Build DAG
    let mut g = DagBuilder::new();
    let x = g.input(vec![1, input_d, input_h, input_w], DataType::Uint);
    let _output = g.pipe(&[x], unet3d(
        conv_weights, conv_transpose_weights, in_gammas, in_betas,
        Some(output_bias), actual_levels, eps,
    ));

    println!("Compiling DAG...");
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!("DAG: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    // Generate random input
    let c_in_pad = 1usize.next_power_of_two();
    let input_size = c_in_pad * d_pad * h_pad * w_pad;
    println!("Running forward pass...");
    let input = Witness::new(
        vec![1, input_d, input_h, input_w],
        generate_random_field_vec(input_size),
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
        let mut transcript = Transcript::new(b"zkml-unet3d");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-unet3d");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-unet3d");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-unet3d");
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
