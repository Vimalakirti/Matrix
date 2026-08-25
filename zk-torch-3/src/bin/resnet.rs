use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::dag::resnet::{resnet50, resnet50_conv_configs, resnet50_output_shape};
use zk_torch_3::transcript::Transcript;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

/// Generate conv weight witness for given (c_in, c_out, kH, kW).
/// Little-endian layout: kw bits | kh bits | c_in bits | c_out bits.
/// For 3×3 kernels: zero-pad entries where kw >= 3 or kh >= 3 (FlattenKernel requirement).
fn generate_conv_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kh_pad * kw_pad;

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
    Witness::new(
        vec![c_out, c_in, kh, kw],
        data,
        DataType::Uint,
        0,
        Role::Constant,
    )
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

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let num_classes = 1000usize;

    let thread_num = rayon::current_num_threads();
    let configs = resnet50_conv_configs();
    let max_conv = num_layers.min(configs.len());

    println!("=== ResNet-50 on ImageNet ({} conv layers) ===", max_conv);
    println!("Using {} threads", thread_num);

    // Generate conv weights
    let conv_weights: Vec<Witness> = configs[..max_conv].iter()
        .map(|&(c_in, c_out, kh, kw)| generate_conv_weight(c_in, c_out, kh, kw))
        .collect();

    // Generate FC weight and bias
    let (out_c, _out_spatial) = resnet50_output_shape(max_conv);
    let fc_weight = generate_fc_weight(out_c, num_classes);
    let fc_bias = generate_fc_bias(num_classes);

    // Build DAG
    let res: usize = std::env::var("RES").ok().and_then(|s| s.parse().ok()).unwrap_or(224);
    let buf_res = res.next_power_of_two();
    let mut g = DagBuilder::new();
    let x = g.input(vec![3, res, res], DataType::Uint);
    let _output = g.pipe(&[x], resnet50(conv_weights, fc_weight, Some(fc_bias), num_classes, max_conv));

    println!("Compiling DAG...");
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!("DAG: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    // Input: [3, res, res] → padded to [4, buf_res, buf_res]
    let input_size = 4 * buf_res * buf_res;
    println!("Running forward pass... (res {})", res);
    let input = Witness::new(
        vec![3, res, res],
        generate_random_field_vec(input_size),
        DataType::Uint, 0, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Run: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let true_max = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .max().unwrap_or(10);
    let capped_max = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .filter(|&n| n <= 22).max().unwrap_or(10);
    println!("max_num_vars: true={} capped<=22={}", true_max, capped_max);
    // SOUND: size the table to the true largest committed polynomial.
    let max_num_vars = true_max;
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
        let mut transcript = Transcript::new(b"zkml-resnet");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-resnet");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-resnet");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-resnet");
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
