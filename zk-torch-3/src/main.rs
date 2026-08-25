use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, dense_add_relu};
use zk_torch_3::transcript::Transcript;

fn demo_simple_add() {
    println!("=== Demo 1: Simple Vector Addition ===");
    println!("Circuit: output = a + b  (4-element vectors)\n");

    let mut g = DagBuilder::new();
    let a = g.input(vec![4], DataType::Uint);
    let b = g.input(vec![4], DataType::Uint);
    let _out = g.add(a, b);

    let (dag, mut witnesses) = g.compile();
    println!("  DAG compiled: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());
    println!("  Inputs: {:?}, Outputs: {:?}", dag.input_ports, dag.output_ports);

    // Feed inputs
    let a_data = Witness::new(
        vec![4],
        vec![GoldilocksField(100), GoldilocksField(200), GoldilocksField(300), GoldilocksField(400)],
        DataType::Uint, 0, Role::Input,
    );
    let b_data = Witness::new(
        vec![4],
        vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
        DataType::Uint, 0, Role::Input,
    );

    // Run
    let t0 = Instant::now();
    dag.run(&mut witnesses, &[(0, a_data), (1, b_data)]);
    let run_time = t0.elapsed();

    // Print output
    let out_edge = *dag.output_ports.last().unwrap();
    let out_evals = witnesses[out_edge][0].data.as_ref().unwrap().evaluations_ref();
    println!("\n  Output: {:?}", &out_evals[..4]);

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
    dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

    // Prove
    let mut transcript = Transcript::new(b"demo_add");
    let mut timing = TimingTree::new("prove", log::Level::Info);
    let t2 = Instant::now();
    let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
        dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
    let prove_time = t2.elapsed();

    // Verify
    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);
    let mut verify_transcript = Transcript::new(b"demo_add");
    let t3 = Instant::now();
    let verified = dag.verify(
        &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
        &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript,
    );
    let verify_time = t3.elapsed();

    println!("\n  Verified: {}", verified);
    println!("  Timing:");
    println!("    Run:    {:?}", run_time);
    println!("    Prove:  {:?}", prove_time);
    println!("    Verify: {:?}", verify_time);
    println!("    Total:  {:?}", run_time + prove_time + verify_time);
    assert!(verified, "Verification failed!");
    println!("  PASSED\n");
}

fn demo_dense_layer() {
    println!("=== Demo 2: Dense Layer (matmul + bias) ===");
    println!("Circuit: output = x @ W + b  (4→4 dense layer)\n");

    let mut g = DagBuilder::new();
    let x = g.input(vec![4], DataType::Uint);

    // Weight matrix 4x4 (identity-like for easy verification)
    let w = Witness::new(
        vec![4, 4],
        vec![
            GoldilocksField(1), GoldilocksField(0), GoldilocksField(0), GoldilocksField(0),
            GoldilocksField(0), GoldilocksField(1), GoldilocksField(0), GoldilocksField(0),
            GoldilocksField(0), GoldilocksField(0), GoldilocksField(1), GoldilocksField(0),
            GoldilocksField(0), GoldilocksField(0), GoldilocksField(0), GoldilocksField(1),
        ],
        DataType::Uint, 0, Role::Constant,
    );
    // Bias vector
    let b = Witness::new(
        vec![4],
        vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
        DataType::Uint, 0, Role::Constant,
    );

    let _out = g.pipe(&[x], dense_add_relu(w, b));
    let (dag, mut witnesses) = g.compile();
    println!("  DAG compiled: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    // Feed input
    let x_data = Witness::new(
        vec![4],
        vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
        DataType::Uint, 0, Role::Input,
    );

    // Run
    let t0 = Instant::now();
    dag.run(&mut witnesses, &[(0, x_data)]);
    let run_time = t0.elapsed();

    // Print output
    let out_edge = *dag.output_ports.last().unwrap();
    let out_evals = witnesses[out_edge][0].data.as_ref().unwrap().evaluations_ref();
    println!("\n  Input:  [1, 2, 3, 4]");
    println!("  W:      identity(4x4)");
    println!("  Bias:   [10, 20, 30, 40]");
    println!("  Output: {:?}", &out_evals[..4]);

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
    dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

    // Prove
    let mut transcript = Transcript::new(b"demo_dense");
    let mut timing = TimingTree::new("prove", log::Level::Info);
    let t2 = Instant::now();
    let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
        dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
    let prove_time = t2.elapsed();

    // Verify
    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);
    let mut verify_transcript = Transcript::new(b"demo_dense");
    let t3 = Instant::now();
    let verified = dag.verify(
        &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
        &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript,
    );
    let verify_time = t3.elapsed();

    println!("\n  Verified: {}", verified);
    println!("  Timing:");
    println!("    Run:    {:?}", run_time);
    println!("    Prove:  {:?}", prove_time);
    println!("    Verify: {:?}", verify_time);
    println!("    Total:  {:?}", run_time + prove_time + verify_time);
    assert!(verified, "Verification failed!");
    println!("  PASSED\n");
}

fn main() {
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");
    println!("zk-torch-3: GPU-Native ZKML with Goldilocks Field");
    println!("Field: Goldilocks (p = 2^64 - 2^32 + 1)");
    println!("Commitment: Basefold (hash-based)");
    println!("Transcript: Poseidon2 sponge");
    println!();

    demo_simple_add();
    demo_dense_layer();

    println!("All demos passed!");
}
