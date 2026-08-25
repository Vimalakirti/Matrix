use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag, edge_partition_map};
use zk_torch_3::dag::yolo::{yolov11n, generate_yolo_weight};
use zk_torch_3::transcript::Transcript;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

/// Generate all weights for YOLOv11n up to a given number of stages.
/// Returns Vec<(weight, bias)> pairs.
fn generate_all_weights(num_stages: usize, input_spatial: usize) -> Vec<(Witness, Witness)> {
    let mut weights = Vec::new();

    // Spatial dims at each point (relative to input):
    let s1 = input_spatial / 2;   // After m.0
    let s2 = input_spatial / 4;   // After m.1/m.2
    let s3 = input_spatial / 8;   // After m.3/m.4
    let s4 = input_spatial / 16;  // After m.5/m.6
    let s5 = input_spatial / 32;  // After m.7/m.8/m.9

    macro_rules! w {
        ($c_in:expr, $c_out:expr, $k:expr, $dw:expr, $h:expr, $w_dim:expr) => {
            weights.push(generate_yolo_weight($c_in, $c_out, $k, $k, $dw, $h, $w_dim));
        };
    }

    // Stage 1: m.0: Conv 3->16, k3, s2
    if num_stages >= 1 {
        w!(3, 16, 3, false, s1, s1);
    }

    // Stage 2: m.1: Conv 16->32, k3, s2
    if num_stages >= 2 {
        w!(16, 32, 3, false, s2, s2);
    }

    // Stage 3: m.2: C3k 32->64
    if num_stages >= 3 {
        w!(32, 64, 1, false, s2, s2);     // cv1
        w!(32, 16, 3, false, s2, s2);     // bneck conv1
        w!(16, 32, 3, false, s2, s2);     // bneck conv2
        w!(128, 64, 1, false, s2, s2);    // cv2 (general_concat: 64+32 → [128,...])
    }

    // Stage 4: m.3 + m.4
    if num_stages >= 4 {
        w!(64, 64, 3, false, s3, s3);       // m.3 downsample

        // m.4: C3k 64->128
        w!(64, 128, 1, false, s3, s3);      // cv1
        w!(64, 32, 3, false, s3, s3);       // bneck conv1
        w!(32, 64, 3, false, s3, s3);       // bneck conv2
        w!(256, 128, 1, false, s3, s3);     // cv2 (general_concat: 128+64 → [256,...])
    }

    // Stage 5: m.5 + m.6
    if num_stages >= 5 {
        w!(128, 128, 3, false, s4, s4);     // m.5 downsample

        // m.6: C3k2 128->128
        w!(128, 128, 1, false, s4, s4);     // cv1
        w!(64, 32, 1, false, s4, s4);       // inner cv1
        w!(32, 32, 3, false, s4, s4);       // res1 conv1
        w!(32, 32, 3, false, s4, s4);       // res1 conv2
        w!(32, 32, 3, false, s4, s4);       // res2 conv1
        w!(32, 32, 3, false, s4, s4);       // res2 conv2
        w!(64, 32, 1, false, s4, s4);       // inner cv2
        w!(64, 64, 1, false, s4, s4);       // cv3
        w!(256, 128, 1, false, s4, s4);     // outer cv2 (general_concat: 128+64 → [256,...])
    }

    // Stage 6: m.7 + m.8 + m.9
    if num_stages >= 6 {
        w!(128, 256, 3, false, s5, s5);     // m.7 downsample

        // m.8: C3k2 256->256
        w!(256, 256, 1, false, s5, s5);     // cv1
        w!(128, 64, 1, false, s5, s5);      // inner cv1
        w!(64, 64, 3, false, s5, s5);       // res1 conv1
        w!(64, 64, 3, false, s5, s5);       // res1 conv2
        w!(64, 64, 3, false, s5, s5);       // res2 conv1
        w!(64, 64, 3, false, s5, s5);       // res2 conv2
        w!(128, 64, 1, false, s5, s5);      // inner cv2
        w!(128, 128, 1, false, s5, s5);     // cv3
        w!(512, 256, 1, false, s5, s5);     // outer cv2 (general_concat: 256+128 → [512,...])

        // m.9: SPPF 256->256
        w!(256, 128, 1, false, s5, s5);     // cv1
        w!(512, 256, 1, false, s5, s5);     // cv2
    }

    // Stage 7: Neck
    if num_stages >= 7 {
        // m.13: C3k 512->128 at s4 (general_concat: 256+128 → [512,...])
        w!(512, 128, 1, false, s4, s4);     // cv1
        w!(64, 32, 3, false, s4, s4);       // bneck conv1
        w!(32, 64, 3, false, s4, s4);       // bneck conv2
        w!(256, 128, 1, false, s4, s4);     // cv2 (general_concat: 128+64 → [256,...])

        // m.16: C3k 256->64 at s3
        w!(256, 64, 1, false, s3, s3);      // cv1
        w!(32, 16, 3, false, s3, s3);       // bneck conv1
        w!(16, 32, 3, false, s3, s3);       // bneck conv2
        w!(128, 64, 1, false, s3, s3);      // cv2 (general_concat: 64+32 → [128,...])

        // m.17: Conv 64->64, k3, s2 at s4
        w!(64, 64, 3, false, s4, s4);

        // m.19: C3k 256->128 at s4 (general_concat: 64+128 → [256,...])
        w!(256, 128, 1, false, s4, s4);     // cv1
        w!(64, 32, 3, false, s4, s4);       // bneck conv1
        w!(32, 64, 3, false, s4, s4);       // bneck conv2
        w!(256, 128, 1, false, s4, s4);     // cv2 (general_concat: 128+64 → [256,...])

        // m.20: Conv 128->128, k3, s2 at s5
        w!(128, 128, 3, false, s5, s5);

        // m.22: C3k2 512->256 at s5 (general_concat: 128+256 → [512,...])
        w!(512, 256, 1, false, s5, s5);     // cv1
        w!(128, 64, 1, false, s5, s5);      // inner cv1
        w!(64, 64, 3, false, s5, s5);       // res1 conv1
        w!(64, 64, 3, false, s5, s5);       // res1 conv2
        w!(64, 64, 3, false, s5, s5);       // res2 conv1
        w!(64, 64, 3, false, s5, s5);       // res2 conv2
        w!(128, 64, 1, false, s5, s5);      // inner cv2
        w!(128, 128, 1, false, s5, s5);     // cv3
        w!(512, 256, 1, false, s5, s5);     // outer cv2 (general_concat: 256+128 → [512,...])
    }

    // Stage 8: Detection heads
    if num_stages >= 8 {
        // P3 head (64ch, s3)
        w!(64, 64, 3, false, s3, s3);
        w!(64, 64, 3, false, s3, s3);
        w!(64, 64, 1, false, s3, s3);
        w!(64, 64, 3, true, s3, s3);
        w!(64, 80, 1, false, s3, s3);
        w!(80, 80, 3, true, s3, s3);
        w!(80, 80, 1, false, s3, s3);
        w!(80, 80, 1, false, s3, s3);

        // P4 head (128ch, s4)
        w!(128, 64, 3, false, s4, s4);
        w!(64, 64, 3, false, s4, s4);
        w!(64, 64, 1, false, s4, s4);
        w!(128, 128, 3, true, s4, s4);
        w!(128, 80, 1, false, s4, s4);
        w!(80, 80, 3, true, s4, s4);
        w!(80, 80, 1, false, s4, s4);
        w!(80, 80, 1, false, s4, s4);

        // P5 head (256ch, s5)
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

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_stages: usize = std::env::var("NUM_STAGES").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4); // Default: backbone only (up to m.4)
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let input_spatial: usize = std::env::var("INPUT_SIZE").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(640);

    let thread_num = rayon::current_num_threads();

    println!("=== YOLOv11n ({} stages, {}x{}) ===", num_stages, input_spatial, input_spatial);
    println!("Using {} threads", thread_num);

    // Generate weights
    println!("Generating weights...");
    let t0 = Instant::now();
    let all_weights = generate_all_weights(num_stages, input_spatial);
    println!("Weight generation: {:?} ({} conv layers)", t0.elapsed(), all_weights.len());

    // Build DAG
    let mut g = DagBuilder::new();
    let x = g.input(vec![3, input_spatial, input_spatial], DataType::Uint);
    let _outputs = g.pipe(&[x], yolov11n(all_weights, num_stages));

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
    let c_pad = 4; // 3 -> next_power_of_two = 4
    let input = Witness::new(
        vec![3, input_spatial, input_spatial],
        generate_random_field_vec(c_pad * input_spatial * input_spatial),
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

    if num_partitions > 1 {
        // Partition-aware commit: assign edges to their partition's GPU
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        let edge_partitions = edge_partition_map(&dag, &partitions);
        println!("Partitions: {}", partitions.len());
        for (i, p) in partitions.iter().enumerate() {
            println!("  Partition {}: {} nodes, {} boundary_in, {} boundary_out",
                i, p.node_ids.len(), p.boundary_input_edges.len(), p.boundary_output_edges.len());
        }

        println!("Committing (partition-aware)...");
        let t3 = Instant::now();
        let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store,
            Some(&edge_partitions));
        println!("Commit: {:?}", t3.elapsed());

        // Download commitments to CPU and free GPU for sumcheck
        gpu_store.download_and_free();

        let vk = BasefoldVerifierKey::from(&key);
        let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

        println!("Proving (parallel, {} partitions)...", num_partitions);
        let mut transcript = Transcript::new(b"zkml-yolo");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-yolo");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Committing...");
        let t3 = Instant::now();
        let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
        println!("Commit: {:?}", t3.elapsed());
        let vk = BasefoldVerifierKey::from(&key);
        let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-yolo");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-yolo");
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
