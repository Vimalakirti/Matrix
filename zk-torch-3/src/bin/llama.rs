use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::dag::llama::llama_2_7b;
use zk_torch_3::transcript::Transcript;
use zk_torch_3::SF_LOG;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

fn generate_llama_weights(num_layers: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness,
) {
    let mut attn_norm_w_vec = Vec::new();
    let mut attn_q_w_vec = Vec::new();
    let mut attn_k_w_vec = Vec::new();
    let mut attn_v_w_vec = Vec::new();
    let mut attn_o_w_vec = Vec::new();
    let mut proj_norm_w_vec = Vec::new();
    let mut proj_1_w_vec = Vec::new();
    let mut proj_2_w_vec = Vec::new();
    let mut proj_3_w_vec = Vec::new();

    for _i in 0..num_layers {
        attn_norm_w_vec.push(Witness::new(
            vec![4096],
            generate_random_field_vec(4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        attn_q_w_vec.push(Witness::new(
            vec![4096, 4096],
            generate_random_field_vec(4096 * 4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        attn_k_w_vec.push(Witness::new(
            vec![4096, 4096],
            generate_random_field_vec(4096 * 4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        attn_v_w_vec.push(Witness::new(
            vec![4096, 4096],
            generate_random_field_vec(4096 * 4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        attn_o_w_vec.push(Witness::new(
            vec![4096, 4096],
            generate_random_field_vec(4096 * 4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        proj_norm_w_vec.push(Witness::new(
            vec![4096],
            generate_random_field_vec(4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        proj_1_w_vec.push(Witness::new(
            vec![4096, 11008],
            generate_random_field_vec(4096 * 16384),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        proj_2_w_vec.push(Witness::new(
            vec![4096, 11008],
            generate_random_field_vec(4096 * 16384),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
        proj_3_w_vec.push(Witness::new(
            vec![11008, 4096],
            generate_random_field_vec(16384 * 4096),
            DataType::Float, *SF_LOG, Role::Constant,
        ));
    }

    let layer_norm_w = Witness::new(
        vec![4096],
        generate_random_field_vec(4096),
        DataType::Float, *SF_LOG, Role::Constant,
    );
    let logits_w = Witness::new(
        vec![4096, 32000],
        generate_random_field_vec(4096 * 32768),
        DataType::Float, *SF_LOG, Role::Constant,
    );

    (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec, proj_3_w_vec,
        layer_norm_w, logits_w,
    )
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);

    let num_heads = 32;
    let head_dim = 128;
    let hidden_dim = 4096;

    let thread_num = rayon::current_num_threads();
    println!("=== LLaMA-2-7B on Goldilocks ({} layers, seq_len={}) ===", num_layers, seq_len);
    println!("Using {} threads", thread_num);

    println!("Generating LLaMA-2-7B random weights...");
    let t0 = Instant::now();

    let mut g = DagBuilder::new();
    let (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec, proj_3_w_vec,
        layer_norm_w, logits_w,
    ) = generate_llama_weights(num_layers);
    println!("Weight generation: {:?}", t0.elapsed());

    let x = g.input(vec![1, seq_len, hidden_dim], DataType::Float);
    let _output = g.pipe(
        &[x],
        llama_2_7b(
            attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
            proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec, proj_3_w_vec,
            layer_norm_w, logits_w, num_heads, head_dim, seq_len,
        ),
    );

    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);

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
    let input_size = seq_len.next_power_of_two() * hidden_dim;
    let input = Witness::new(
        vec![1, seq_len, hidden_dim],
        generate_random_field_vec(input_size),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Run: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .filter(|&n| n <= 24)
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];

    println!("Committing...");
    let t3 = Instant::now();
    let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
    println!("Commit: {:?}", t3.elapsed());
    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

    // Download GPU commitments to host and free GPU memory for proving
    gpu_store.download_and_free();

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());
        for (i, p) in partitions.iter().enumerate() {
            println!("  Partition {}: {} nodes, {} boundary_in, {} boundary_out",
                i, p.node_ids.len(), p.boundary_input_edges.len(), p.boundary_output_edges.len());
        }

        println!("Proving (parallel, {} partitions)...", num_partitions);
        let mut transcript = Transcript::new(b"zkml-llama");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-llama");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-llama");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-llama");
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
