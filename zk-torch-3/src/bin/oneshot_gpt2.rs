//! One-shot autoregressive proving for GPT-2 Small.
//!
//! Pipeline: embedding(W_E) → positional → transformer body → LM head (weight-tied) → argmax check.
//!
//! Witness gen does K+2 forward passes (K = generated tokens) — all proof-free,
//! with the embedding selector overwritten between AR steps. Proving runs once
//! over the full seq_len-wide circuit.
//!
//! Env vars:
//!   SEQ_LEN, PROMPT_LEN, VOCAB_SIZE, NUM_LAYERS, SKIP_AR, NUM_PARTITIONS

use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{
    BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore,
};
use zk_torch_3::dag::gpt2::gpt_2_small;
use zk_torch_3::dag::oneshot::extract_argmax_all;
use zk_torch_3::dag::{partition_dag, DagBuilder, DataType, Role, Witness};
use zk_torch_3::transcript::Transcript;
use zk_torch_3::SF_LOG;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

fn padded_size(shape: &[usize]) -> usize {
    shape.iter().map(|&s| s.next_power_of_two()).product()
}

fn generate_gpt2_weights(num_layers: usize) -> (
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Vec<Witness>, Vec<Witness>, Vec<Witness>,
    Witness, Witness,
) {
    let mut attn_norm_w_vec = Vec::new();
    let mut attn_q_w_vec = Vec::new();
    let mut attn_k_w_vec = Vec::new();
    let mut attn_v_w_vec = Vec::new();
    let mut attn_o_w_vec = Vec::new();
    let mut attn_norm_b_vec = Vec::new();
    let mut attn_q_b_vec = Vec::new();
    let mut attn_k_b_vec = Vec::new();
    let mut attn_v_b_vec = Vec::new();
    let mut attn_o_b_vec = Vec::new();
    let mut proj_norm_w_vec = Vec::new();
    let mut proj_1_w_vec = Vec::new();
    let mut proj_2_w_vec = Vec::new();
    let mut proj_norm_b_vec = Vec::new();
    let mut proj_1_b_vec = Vec::new();
    let mut proj_2_b_vec = Vec::new();

    for _ in 0..num_layers {
        attn_norm_w_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_w_vec.push(Witness::new(vec![768, 768], generate_random_field_vec(1024 * 1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_w_vec.push(Witness::new(vec![768, 768], generate_random_field_vec(1024 * 1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_w_vec.push(Witness::new(vec![768, 768], generate_random_field_vec(1024 * 1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_w_vec.push(Witness::new(vec![768, 768], generate_random_field_vec(1024 * 1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_norm_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_q_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_k_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_v_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        attn_o_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        proj_norm_w_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_w_vec.push(Witness::new(vec![768, 3072], generate_random_field_vec(1024 * 4096), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_w_vec.push(Witness::new(vec![3072, 768], generate_random_field_vec(4096 * 1024), DataType::Float, *SF_LOG, Role::Constant));
        proj_norm_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
        proj_1_b_vec.push(Witness::new(vec![3072], generate_random_field_vec(4096), DataType::Float, *SF_LOG, Role::Constant));
        proj_2_b_vec.push(Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant));
    }

    let layer_norm_w = Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant);
    let layer_norm_b = Witness::new(vec![768], generate_random_field_vec(1024), DataType::Float, *SF_LOG, Role::Constant);

    (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        attn_norm_b_vec, attn_q_b_vec, attn_k_b_vec, attn_v_b_vec, attn_o_b_vec,
        proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec,
        proj_norm_b_vec, proj_1_b_vec, proj_2_b_vec,
        layer_norm_w, layer_norm_b,
    )
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let seq_len: usize = std::env::var("SEQ_LEN").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let vocab_size: usize = std::env::var("VOCAB_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let prompt_len: usize = std::env::var("PROMPT_LEN").ok().and_then(|s| s.parse().ok())
        .unwrap_or((seq_len / 2).max(1));
    assert!(prompt_len >= 1 && prompt_len <= seq_len, "PROMPT_LEN must be in [1, SEQ_LEN]");
    let skip_ar: bool = std::env::var("SKIP_AR").ok().as_deref() == Some("1");

    let num_heads = 12;
    let head_dim = 64;
    let hidden_dim = num_heads * head_dim; // 768

    println!("=== One-shot GPT-2 Small ===");
    println!("layers={}, seq_len={}, prompt_len={}, generated={}, vocab={}, hidden={}",
        num_layers, seq_len, prompt_len, seq_len - prompt_len, vocab_size, hidden_dim);
    println!("threads={}", rayon::current_num_threads());

    // Token transcript: prompt is random; generated slots start as 0 (or random under SKIP_AR).
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

    // ---- Weight gen ----
    let t0 = Instant::now();
    let (
        attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
        attn_norm_b_vec, attn_q_b_vec, attn_k_b_vec, attn_v_b_vec, attn_o_b_vec,
        proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec,
        proj_norm_b_vec, proj_1_b_vec, proj_2_b_vec,
        layer_norm_w, layer_norm_b,
    ) = generate_gpt2_weights(num_layers);
    let we_data = generate_random_field_vec(padded_size(&[vocab_size, hidden_dim]));
    let w_e_witness = Witness::new(
        vec![vocab_size, hidden_dim],
        we_data,
        DataType::Float, *SF_LOG, Role::Constant,
    );
    let pos_data = generate_random_field_vec(padded_size(&[seq_len, hidden_dim]));
    let pos_embed = Witness::new(
        vec![seq_len, hidden_dim],
        pos_data,
        DataType::Float, *SF_LOG, Role::Constant,
    );
    println!("Weight generation: {:?}", t0.elapsed());

    // ---- Build oneshot circuit ----
    println!("Building oneshot circuit...");
    let t1 = Instant::now();
    let mut g = DagBuilder::new();

    let w_e = g.param(w_e_witness);
    let (h0, emb_selector_edge) = g.embedding_lookup(w_e, seq_len, vocab_size, &token_ids);
    let h_pe = g.add_positional_encoding(h0, pos_embed);
    let h_input = g.change_shape(h_pe, vec![1, seq_len, hidden_dim]);

    let h_out = g.pipe(
        &[h_input],
        gpt_2_small(
            attn_norm_w_vec, attn_q_w_vec, attn_k_w_vec, attn_v_w_vec, attn_o_w_vec,
            attn_norm_b_vec, attn_q_b_vec, attn_k_b_vec, attn_v_b_vec, attn_o_b_vec,
            proj_norm_w_vec, proj_1_w_vec, proj_2_w_vec,
            proj_norm_b_vec, proj_1_b_vec, proj_2_b_vec,
            layer_norm_w, layer_norm_b,
            num_heads, head_dim, seq_len,
        ),
    )[0];

    let logits = g.lm_head_weight_tied(h_out, w_e, seq_len, vocab_size);

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

    // ---- Witness generation: K+2 forward passes ----
    let t2 = Instant::now();
    if skip_ar {
        // Single forward populates everything; no AR loop needed.
        dag.run(&mut witnesses, &[]);
    } else {
        // K AR iterations: fwd → read argmax at pos-1 → write next token into S.
        for pos in prompt_len..seq_len {
            dag.run(&mut witnesses, &[]);
            let next_tok = zk_torch_3::dag::oneshot::extract_argmax_at(
                &witnesses, logits, pos - 1, seq_len, vocab_size,
            );
            token_ids[pos] = next_tok;
            let new_s = DagBuilder::build_one_hot_selector(seq_len, vocab_size, &token_ids);
            witnesses[emb_selector_edge] = vec![new_s];
        }
        // Final consistency run: logits matches the fully-populated embedding selector.
        dag.run(&mut witnesses, &[]);
        println!("Final token_ids: {:?}", token_ids);
    }

    // Compute next_token_ids[i] = argmax(logits[i,:]) for every position.
    let next_token_ids = extract_argmax_all(&witnesses, logits, seq_len, vocab_size);
    println!("next_token_ids = {:?}", next_token_ids);

    // Public shift constraint (off-circuit): asserts the AR recurrence on public selectors.
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

    // Overwrite argmax selector with the now-consistent argmaxes; one more fwd populates the
    // small argmax sub-DAG (selected, diffs).
    let new_s_prime = DagBuilder::build_one_hot_selector(seq_len, vocab_size, &next_token_ids);
    witnesses[argmax_selector_edge] = vec![new_s_prime];
    dag.run(&mut witnesses, &[]);
    println!("Witness generation total: {:?}", t2.elapsed());

    // ---- Commit / Prove / Verify ----
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
    // Free witness-side device buffers before prove. Prove reads from host
    // and re-uploads per einsum task, so this saves GPU memory with zero
    // prove-time cost. No-op if witnesses are already host-only (CPU backend).
    dag.evict_device_witnesses(&mut witnesses);

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());

        println!("Proving (parallel, {} partitions)...", num_partitions);
        let mut transcript = Transcript::new(b"zkml-oneshot-gpt2");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table,
            &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-oneshot-gpt2");
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
        let mut transcript = Transcript::new(b"zkml-oneshot-gpt2");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table,
                &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-oneshot-gpt2");
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
