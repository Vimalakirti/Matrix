//! One-shot autoregressive proving for GPT-J 6B.
//!
//! Pipeline: embedding(W_E) → gpt_j_6b body (transformer + final LN + matmul) → argmax check.
//! `gpt_j_6b` already produces `[1, seq_len, vocab_size]` logits via its internal `matmul_w`,
//! so we change_shape to `[seq_len, vocab_size]` and pass directly into argmax_check.
//!
//! GPT-J uses RoPE inside attention, not absolute positional embeddings.
//!
//! Env vars: SEQ_LEN, PROMPT_LEN, VOCAB_SIZE, NUM_LAYERS, SKIP_AR, NUM_PARTITIONS.

use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{
    BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore,
};
use zk_torch_3::dag::gptj::gpt_j_6b;
use zk_torch_3::dag::oneshot::extract_argmax_all;
use zk_torch_3::dag::{partition_dag, DagBuilder, DataType, Role, Witness};
use zk_torch_3::transcript::Transcript;
use zk_torch_3::SF_LOG;

const HIDDEN_DIM: usize = 4096;
const FFN_DIM: usize = 16384;
const NUM_HEADS: usize = 16;
const HEAD_DIM: usize = 256;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

fn padded_size(shape: &[usize]) -> usize {
    shape.iter().map(|&s| s.next_power_of_two()).product()
}

fn generate_gptj_weights(num_layers: usize, vocab_size: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness, Witness, Witness,
) {
    let mut attn_norm_w_vec = Vec::new();
    let mut attn_q_w_vec = Vec::new();
    let mut attn_k_w_vec = Vec::new();
    let mut attn_v_w_vec = Vec::new();
    let mut attn_o_w_vec = Vec::new();
    let mut attn_norm_b_vec = Vec::new();
    let mut proj_1_w_vec = Vec::new();
    let mut proj_2_w_vec = Vec::new();
    let mut proj_1_b_vec = Vec::new();
    let mut proj_2_b_vec = Vec::new();

    for _ in 0..num_layers {
        attn_norm_w_vec.push(Witness::new(vec![HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_w_vec.push(Witness::new(vec![HIDDEN_DIM, HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM * HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_w_vec.push(Witness::new(vec![HIDDEN_DIM, HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM * HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_w_vec.push(Witness::new(vec![HIDDEN_DIM, HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM * HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_w_vec.push(Witness::new(vec![HIDDEN_DIM, HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM * HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        attn_norm_b_vec.push(Witness::new(vec![HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_w_vec.push(Witness::new(vec![HIDDEN_DIM, FFN_DIM], generate_random_field_vec(HIDDEN_DIM * FFN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_w_vec.push(Witness::new(vec![FFN_DIM, HIDDEN_DIM], generate_random_field_vec(FFN_DIM * HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_b_vec.push(Witness::new(vec![FFN_DIM], generate_random_field_vec(FFN_DIM), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_b_vec.push(Witness::new(vec![HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant));
    }

    let layer_norm_w = Witness::new(vec![HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant);
    let layer_norm_b = Witness::new(vec![HIDDEN_DIM], generate_random_field_vec(HIDDEN_DIM), DataType::Float, *SF_LOG, Role::Constant);
    let matmul_w = Witness::new(
        vec![HIDDEN_DIM, vocab_size],
        generate_random_field_vec(padded_size(&[HIDDEN_DIM, vocab_size])),
        DataType::Float, *SF_LOG, Role::Constant,
    );
    let matmul_b = Witness::new(
        vec![vocab_size],
        generate_random_field_vec(vocab_size.next_power_of_two()),
        DataType::Float, *SF_LOG, Role::Constant,
    );

    (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        attn_norm_b_vec, proj_1_w_vec, proj_2_w_vec, proj_1_b_vec, proj_2_b_vec,
        layer_norm_w, layer_norm_b, matmul_w, matmul_b,
    )
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(28);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let vocab_size: usize = std::env::var("VOCAB_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let prompt_len: usize = std::env::var("PROMPT_LEN").ok().and_then(|s| s.parse().ok())
        .unwrap_or((seq_len / 2).max(1));
    assert!(prompt_len >= 1 && prompt_len <= seq_len);
    let skip_ar: bool = std::env::var("SKIP_AR").ok().as_deref() == Some("1");

    println!("=== One-shot GPT-J 6B ===");
    println!("layers={}, seq_len={}, prompt_len={}, generated={}, vocab={}, hidden={}",
        num_layers, seq_len, prompt_len, seq_len - prompt_len, vocab_size, HIDDEN_DIM);
    println!("threads={}", rayon::current_num_threads());

    let mut rng = rand::thread_rng();
    let mut token_ids: Vec<usize> = vec![0usize; seq_len];
    for i in 0..prompt_len {
        token_ids[i] = rng.gen::<usize>() % vocab_size;
    }
    if skip_ar {
        for i in prompt_len..seq_len {
            token_ids[i] = rng.gen::<usize>() % vocab_size;
        }
        println!("SKIP_AR=1: random tokens, public shift check skipped.");
    }
    println!("prompt token_ids[..{}]={:?}", prompt_len, &token_ids[..prompt_len]);

    let t0 = Instant::now();
    let (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        attn_norm_b_vec, proj_1_w_vec, proj_2_w_vec, proj_1_b_vec, proj_2_b_vec,
        layer_norm_w, layer_norm_b, matmul_w, matmul_b,
    ) = generate_gptj_weights(num_layers, vocab_size);
    let we_data = generate_random_field_vec(padded_size(&[vocab_size, HIDDEN_DIM]));
    let w_e_witness = Witness::new(
        vec![vocab_size, HIDDEN_DIM],
        we_data,
        DataType::Float, *SF_LOG, Role::Constant,
    );
    println!("Weight generation: {:?}", t0.elapsed());

    println!("Building oneshot circuit...");
    let t1 = Instant::now();
    let mut g = DagBuilder::new();

    let w_e = g.param(w_e_witness);
    let (h0, emb_selector_edge) = g.embedding_lookup(w_e, seq_len, vocab_size, &token_ids);
    let h_input = g.change_shape(h0, vec![1, seq_len, HIDDEN_DIM]);

    // gpt_j_6b returns [1, seq_len, vocab_size] logits (final LN + matmul + bias baked in).
    let logits_3d = g.pipe(
        &[h_input],
        gpt_j_6b(
            attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
            attn_norm_b_vec, proj_1_w_vec, proj_2_w_vec, proj_1_b_vec, proj_2_b_vec,
            layer_norm_w, layer_norm_b, matmul_w, matmul_b,
            NUM_HEADS, HEAD_DIM, seq_len,
        ),
    )[0];
    let logits = g.change_shape(logits_3d, vec![seq_len, vocab_size]);

    let dummy_tokens = vec![0usize; seq_len];
    let argmax_selector_edge = g.argmax_check(logits, seq_len, vocab_size, &dummy_tokens);

    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!("DAG: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    let t2 = Instant::now();
    if skip_ar {
        dag.run(&mut witnesses, &[]);
    } else {
        for pos in prompt_len..seq_len {
            dag.run(&mut witnesses, &[]);
            let next_tok = zk_torch_3::dag::oneshot::extract_argmax_at(
                &witnesses, logits, pos - 1, seq_len, vocab_size,
            );
            token_ids[pos] = next_tok;
            let new_s = DagBuilder::build_one_hot_selector(seq_len, vocab_size, &token_ids);
            witnesses[emb_selector_edge] = vec![new_s];
        }
        dag.run(&mut witnesses, &[]);
        println!("Final token_ids: {:?}", token_ids);
    }

    let next_token_ids = extract_argmax_all(&witnesses, logits, seq_len, vocab_size);
    println!("next_token_ids = {:?}", next_token_ids);

    if !skip_ar {
        for i in (prompt_len.saturating_sub(1))..seq_len.saturating_sub(1) {
            assert_eq!(
                token_ids[i + 1], next_token_ids[i],
                "shift constraint violated at i={}: token_ids[{}]={} != argmax(logits[{}])={}",
                i, i + 1, token_ids[i + 1], i, next_token_ids[i],
            );
        }
        println!("Public shift constraint check passed.");
    }

    let new_s_prime = DagBuilder::build_one_hot_selector(seq_len, vocab_size, &next_token_ids);
    witnesses[argmax_selector_edge] = vec![new_s_prime];
    dag.run(&mut witnesses, &[]);
    println!("Witness generation total: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .filter(|&n| n <= 22)
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];

    println!("Committing...");
    let t3 = Instant::now();
    let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
    println!("Commit: {:?}", t3.elapsed());

    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);
    if std::env::var("ONESHOT_DOWNLOAD_AND_FREE").ok().as_deref() == Some("1") {
        gpu_store.download_and_free();
    }
    dag.evict_device_witnesses(&mut witnesses);

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());

        println!("Proving (parallel, {} partitions)...", num_partitions);
        let mut transcript = Transcript::new(b"zkml-oneshot-gptj");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table,
            &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-oneshot-gptj");
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table,
            &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        timing.print();
        println!("\nVerified: {}", verified);
    } else {
        println!("Proving...");
        let mut transcript = Transcript::new(b"zkml-oneshot-gptj");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table,
                &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-oneshot-gptj");
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
