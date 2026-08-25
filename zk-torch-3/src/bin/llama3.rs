use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{
    BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore,
};
use zk_torch_3::dag::llama::llama3_8b;
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::transcript::Transcript;
use zk_torch_3::SF_LOG;

// Llama 3.1 8B config
const HIDDEN_DIM: usize = 4096;
const FFN_DIM: usize = 14336;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
const VOCAB_SIZE: usize = 128256;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

fn padded_size(shape: &[usize]) -> usize {
    shape.iter().map(|&s| s.next_power_of_two()).product()
}

fn generate_llama3_weights(
    num_layers: usize,
) -> (
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Vec<Witness>,
    Witness,
    Witness,
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
            vec![HIDDEN_DIM],
            generate_random_field_vec(HIDDEN_DIM),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // Q: [4096, 4096]
        attn_q_w_vec.push(Witness::new(
            vec![HIDDEN_DIM, HIDDEN_DIM],
            generate_random_field_vec(padded_size(&[HIDDEN_DIM, HIDDEN_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // K: [4096, 1024]
        attn_k_w_vec.push(Witness::new(
            vec![HIDDEN_DIM, KV_DIM],
            generate_random_field_vec(padded_size(&[HIDDEN_DIM, KV_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // V: [4096, 1024]
        attn_v_w_vec.push(Witness::new(
            vec![HIDDEN_DIM, KV_DIM],
            generate_random_field_vec(padded_size(&[HIDDEN_DIM, KV_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // O: [4096, 4096]
        attn_o_w_vec.push(Witness::new(
            vec![HIDDEN_DIM, HIDDEN_DIM],
            generate_random_field_vec(padded_size(&[HIDDEN_DIM, HIDDEN_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        proj_norm_w_vec.push(Witness::new(
            vec![HIDDEN_DIM],
            generate_random_field_vec(HIDDEN_DIM),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // gate_proj: [4096, 14336]
        proj_1_w_vec.push(Witness::new(
            vec![HIDDEN_DIM, FFN_DIM],
            generate_random_field_vec(padded_size(&[HIDDEN_DIM, FFN_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // up_proj: [4096, 14336]
        proj_2_w_vec.push(Witness::new(
            vec![HIDDEN_DIM, FFN_DIM],
            generate_random_field_vec(padded_size(&[HIDDEN_DIM, FFN_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        // down_proj: [14336, 4096]
        proj_3_w_vec.push(Witness::new(
            vec![FFN_DIM, HIDDEN_DIM],
            generate_random_field_vec(padded_size(&[FFN_DIM, HIDDEN_DIM])),
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
    }

    let layer_norm_w = Witness::new(
        vec![HIDDEN_DIM],
        generate_random_field_vec(HIDDEN_DIM),
        DataType::Float,
        *SF_LOG,
        Role::Constant,
    );
    // lm_head: [4096, 128256]
    let logits_w = Witness::new(
        vec![HIDDEN_DIM, VOCAB_SIZE],
        generate_random_field_vec(padded_size(&[HIDDEN_DIM, VOCAB_SIZE])),
        DataType::Float,
        *SF_LOG,
        Role::Constant,
    );

    (
        attn_norm_w_vec,
        attn_q_w_vec,
        attn_k_w_vec,
        attn_v_w_vec,
        attn_o_w_vec,
        proj_norm_w_vec,
        proj_1_w_vec,
        proj_2_w_vec,
        proj_3_w_vec,
        layer_norm_w,
        logits_w,
    )
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_layers: usize = std::env::var("NUM_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let seq_len: usize = std::env::var("SEQ_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let thread_num = rayon::current_num_threads();
    println!(
        "=== Llama 3.1 8B on Goldilocks ({} layers, seq_len={}) ===",
        num_layers, seq_len
    );
    println!("  Q heads: {}, KV heads: {}, head_dim: {}", NUM_HEADS, NUM_KV_HEADS, HEAD_DIM);
    println!("  FFN dim: {}, vocab: {}", FFN_DIM, VOCAB_SIZE);
    println!("  Using {} threads", thread_num);

    println!("Generating Llama 3.1 8B random weights...");
    let t0 = Instant::now();

    let mut g = DagBuilder::new();
    let (
        attn_norm_w_vec,
        attn_q_w_vec,
        attn_k_w_vec,
        attn_v_w_vec,
        attn_o_w_vec,
        proj_norm_w_vec,
        proj_1_w_vec,
        proj_2_w_vec,
        proj_3_w_vec,
        layer_norm_w,
        logits_w,
    ) = generate_llama3_weights(num_layers);
    println!("Weight generation: {:?}", t0.elapsed());

    let x = g.input(vec![1, seq_len, HIDDEN_DIM], DataType::Float);
    let _output = g.pipe(
        &[x],
        llama3_8b(
            attn_norm_w_vec,
            attn_q_w_vec,
            attn_k_w_vec,
            attn_v_w_vec,
            attn_o_w_vec,
            proj_norm_w_vec,
            proj_1_w_vec,
            proj_2_w_vec,
            proj_3_w_vec,
            layer_norm_w,
            logits_w,
            seq_len,
            HEAD_DIM,
            NUM_HEADS,
            NUM_KV_HEADS,
            VOCAB_SIZE,
        ),
    );

    let num_partitions: usize = std::env::var("NUM_PARTITIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    println!("Compiling DAG...");
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!(
        "DAG: {} nodes, {} edges",
        dag.nodes.len(),
        dag.num_edges()
    );

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    println!("Running forward pass...");
    let input = Witness::new(
        vec![1, seq_len, HIDDEN_DIM],
        generate_random_field_vec(padded_size(&[1, seq_len, HIDDEN_DIM])),
        DataType::Float,
        *SF_LOG,
        Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, input)]);
    println!("Run: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses
        .iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .filter(|&n| n <= 22)
        .max()
        .unwrap_or(10);
    let mut gpu_store =
        GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];

    println!("Committing...");
    let t3 = Instant::now();
    let nonweight_commit_time =
        dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
    println!("Commit: {:?}", t3.elapsed());
    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());
        for (i, p) in partitions.iter().enumerate() {
            println!(
                "  Partition {}: {} nodes, {} boundary_in, {} boundary_out",
                i,
                p.node_ids.len(),
                p.boundary_input_edges.len(),
                p.boundary_output_edges.len()
            );
        }

        println!(
            "Proving (parallel, {} partitions)...",
            num_partitions
        );
        let mut transcript = Transcript::new(b"zkml-llama3");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key,
            &mut witnesses,
            &commitments,
            &gpu_store,
            &gpu_store.table,
            &mut transcript,
            &partitions,
            &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!(
            "Prove: {:?} (+ commit {:?} = {:?})",
            prove_elapsed,
            nonweight_commit_time,
            prove_elapsed + nonweight_commit_time
        );

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-llama3");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof,
            &witnesses,
            &vk,
            &commitments,
            &vk_table,
            &mut verify_transcript,
            &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-llama3");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) = dag.prove(
            &key,
            &mut witnesses,
            &commitments,
            &gpu_store,
            &gpu_store.table,
            &mut transcript,
            &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!(
            "Prove: {:?} (+ commit {:?} = {:?})",
            prove_elapsed,
            nonweight_commit_time,
            prove_elapsed + nonweight_commit_time
        );

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-llama3");
        let t5 = Instant::now();
        let verified = dag.verify(
            &node_proofs,
            &edge_proofs,
            &range_proof,
            &two_pow_proof,
            &reducer_proofs,
            &witnesses,
            &vk,
            &commitments,
            &vk_table,
            &mut verify_transcript,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    }
}
