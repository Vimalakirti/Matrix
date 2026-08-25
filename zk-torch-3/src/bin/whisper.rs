use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;
use rayon::prelude::*;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::dag::whisper::{
    whisper_model, EncoderBlockWeights, DecoderBlockWeights,
};
use zk_torch_3::transcript::Transcript;
use zk_torch_3::SF_LOG;

/// Threshold above which we switch to parallel random generation.
const PAR_THRESHOLD: usize = 1 << 16; // 64K elements

fn rand_field_vec(size: usize) -> Vec<GoldilocksField> {
    if size >= PAR_THRESHOLD {
        (0..size).into_par_iter()
            .map_init(rand::thread_rng, |rng, _| GoldilocksField((rng.gen::<u32>() % 500) as u64))
            .collect()
    } else {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64)).collect()
    }
}

fn make_weight(shape: Vec<usize>, padded_size: usize) -> Witness {
    Witness::new(shape, rand_field_vec(padded_size), DataType::Float, *SF_LOG, Role::Constant)
}

fn make_bias_1d(shape: Vec<usize>, padded_size: usize) -> Witness {
    Witness::new(shape, rand_field_vec(padded_size), DataType::Float, *SF_LOG, Role::Constant)
}

fn make_uint_weight(shape: Vec<usize>, padded_size: usize) -> Witness {
    Witness::new(shape, rand_field_vec(padded_size), DataType::Uint, 0, Role::Constant)
}

/// Create Conv1D weight W[C_out, C_in, K] with proper zero-padding.
/// Padding positions (k >= K, c >= C_in, d >= C_out) must be zero.
fn make_conv1d_weight(c_out: usize, c_in: usize, kernel_size: usize) -> Witness {
    let c_out_pad = c_out.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let k_pad = kernel_size.next_power_of_two();
    let size = c_out_pad * c_in_pad * k_pad;
    let mut data = vec![GoldilocksField(0); size];
    // Parallelize over output channels (each channel's slice is independent)
    let chunk_size = c_in_pad * k_pad;
    data.par_chunks_mut(chunk_size)
        .enumerate()
        .filter(|&(d, _)| d < c_out)
        .for_each_init(rand::thread_rng, |rng, (_, chunk)| {
            for c in 0..c_in {
                for k in 0..kernel_size {
                    let idx = k + c * k_pad;
                    chunk[idx] = GoldilocksField((rng.gen::<u32>() % 500) as u64);
                }
            }
        });
    Witness::new(vec![c_out, c_in, kernel_size], data, DataType::Uint, 0, Role::Constant)
}

fn make_encoder_block_weights(n_state: usize, mlp_dim: usize) -> EncoderBlockWeights {
    let ns_pad = n_state.next_power_of_two();
    let mlp_pad = mlp_dim.next_power_of_two();
    EncoderBlockWeights {
        attn_ln_w: make_bias_1d(vec![n_state], ns_pad),
        attn_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_q: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_k: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_v: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_o: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        b_q: make_bias_1d(vec![n_state], ns_pad),
        b_k: make_bias_1d(vec![n_state], ns_pad),
        b_v: make_bias_1d(vec![n_state], ns_pad),
        b_o: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_w: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_mlp1: make_weight(vec![n_state, mlp_dim], ns_pad * mlp_pad),
        w_mlp2: make_weight(vec![mlp_dim, n_state], mlp_pad * ns_pad),
        b_mlp1: make_bias_1d(vec![mlp_dim], mlp_pad),
        b_mlp2: make_bias_1d(vec![n_state], ns_pad),
    }
}

fn make_decoder_block_weights(n_state: usize, mlp_dim: usize) -> DecoderBlockWeights {
    let ns_pad = n_state.next_power_of_two();
    let mlp_pad = mlp_dim.next_power_of_two();
    DecoderBlockWeights {
        attn_ln_w: make_bias_1d(vec![n_state], ns_pad),
        attn_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_q: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_k: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_v: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        w_o: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        b_q: make_bias_1d(vec![n_state], ns_pad),
        b_k: make_bias_1d(vec![n_state], ns_pad),
        b_v: make_bias_1d(vec![n_state], ns_pad),
        b_o: make_bias_1d(vec![n_state], ns_pad),
        cross_ln_w: make_bias_1d(vec![n_state], ns_pad),
        cross_ln_b: make_bias_1d(vec![n_state], ns_pad),
        xw_q: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xw_k: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xw_v: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xw_o: make_weight(vec![n_state, n_state], ns_pad * ns_pad),
        xb_q: make_bias_1d(vec![n_state], ns_pad),
        xb_k: make_bias_1d(vec![n_state], ns_pad),
        xb_v: make_bias_1d(vec![n_state], ns_pad),
        xb_o: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_w: make_bias_1d(vec![n_state], ns_pad),
        mlp_ln_b: make_bias_1d(vec![n_state], ns_pad),
        w_mlp1: make_weight(vec![n_state, mlp_dim], ns_pad * mlp_pad),
        w_mlp2: make_weight(vec![mlp_dim, n_state], mlp_pad * ns_pad),
        b_mlp1: make_bias_1d(vec![mlp_dim], mlp_pad),
        b_mlp2: make_bias_1d(vec![n_state], ns_pad),
    }
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    // Configuration via env vars
    let num_enc_layers: usize = std::env::var("NUM_ENC_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let num_dec_layers: usize = std::env::var("NUM_DEC_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let n_mels: usize = std::env::var("N_MELS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(80);
    let n_audio_ctx: usize = std::env::var("N_AUDIO_CTX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1500);
    let n_text_ctx: usize = std::env::var("N_TEXT_CTX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(448);
    let n_state: usize = std::env::var("N_STATE").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(384);
    let n_head: usize = std::env::var("N_HEAD").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(6);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);

    let head_dim = n_state / n_head;
    let mlp_dim = 4 * n_state; // Whisper uses 4x MLP expansion

    let ns_pad = n_state.next_power_of_two();
    let n_mels_pad = n_mels.next_power_of_two();
    let thread_num = rayon::current_num_threads();

    println!("=== Whisper (enc={}, dec={}, n_state={}, n_head={}) ===",
        num_enc_layers, num_dec_layers, n_state, n_head);
    println!("n_mels={}, n_audio_ctx={}, n_text_ctx={}, head_dim={}, mlp_dim={}",
        n_mels, n_audio_ctx, n_text_ctx, head_dim, mlp_dim);
    println!("Using {} threads", thread_num);

    println!("Generating random weights...");
    let t0 = Instant::now();

    let mut g = DagBuilder::new();

    // Inputs
    let mel_input = g.input(vec![n_mels, 2 * n_audio_ctx], DataType::Float);
    let dec_input = g.input(vec![1, n_text_ctx, n_state], DataType::Float);

    // Conv weights — must zero-pad k >= kernel_size positions
    let conv1_out_len = 2 * n_audio_ctx; // same length after pad(1,1)+conv(k=3,s=1)
    let conv1_w = g.param(make_conv1d_weight(n_state, n_mels, 3));
    let conv1_bias = g.param(make_uint_weight(
        vec![n_state, conv1_out_len],
        ns_pad * conv1_out_len.next_power_of_two(),
    ));
    let conv2_w = g.param(make_conv1d_weight(n_state, n_state, 3));
    let conv2_bias = g.param(make_uint_weight(
        vec![n_state, n_audio_ctx],
        ns_pad * n_audio_ctx.next_power_of_two(),
    ));

    // Sinusoidal positional embedding for encoder
    let enc_pos_emb = g.param(make_weight(
        vec![1, n_audio_ctx, n_state],
        n_audio_ctx.next_power_of_two() * ns_pad,
    ));

    // Learned positional embedding for decoder
    let dec_pos_emb = g.param(make_weight(
        vec![1, n_text_ctx, n_state],
        n_text_ctx.next_power_of_two() * ns_pad,
    ));

    // Encoder blocks (parallel)
    let enc_blocks: Vec<EncoderBlockWeights> = (0..num_enc_layers)
        .into_par_iter()
        .map(|_| make_encoder_block_weights(n_state, mlp_dim))
        .collect();
    let enc_final_ln_w = make_bias_1d(vec![n_state], ns_pad);
    let enc_final_ln_b = make_bias_1d(vec![n_state], ns_pad);

    // Decoder blocks (parallel)
    let dec_blocks: Vec<DecoderBlockWeights> = (0..num_dec_layers)
        .into_par_iter()
        .map(|_| make_decoder_block_weights(n_state, mlp_dim))
        .collect();
    let dec_final_ln_w = make_bias_1d(vec![n_state], ns_pad);
    let dec_final_ln_b = make_bias_1d(vec![n_state], ns_pad);

    println!("Weight generation: {:?}", t0.elapsed());

    let _output = whisper_model(
        &mut g,
        mel_input, dec_input,
        conv1_w, conv1_bias, conv2_w, conv2_bias, enc_pos_emb,
        enc_blocks, enc_final_ln_w, enc_final_ln_b,
        dec_pos_emb, dec_blocks, dec_final_ln_w, dec_final_ln_b,
        n_head, head_dim, n_state, n_audio_ctx, n_text_ctx,
    );

    println!("Compiling DAG...");
    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!("DAG: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    // Run forward pass
    println!("Running forward pass...");
    let mel_data = Witness::new(
        vec![n_mels, 2 * n_audio_ctx],
        rand_field_vec(n_mels_pad * (2 * n_audio_ctx).next_power_of_two()),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let dec_data = Witness::new(
        vec![1, n_text_ctx, n_state],
        rand_field_vec(n_text_ctx.next_power_of_two() * ns_pad),
        DataType::Float, *SF_LOG, Role::Input,
    );
    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, mel_data), (1, dec_data)]);
    println!("Run: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .filter(|&n| n <= 22)
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());
        for (i, p) in partitions.iter().enumerate() {
            println!("  Partition {}: {} nodes, {} boundary_in, {} boundary_out",
                i, p.node_ids.len(), p.boundary_input_edges.len(), p.boundary_output_edges.len());
        }

        println!("Committing...");
        let t3 = Instant::now();
        let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
        println!("Commit: {:?}", t3.elapsed());
        let vk = BasefoldVerifierKey::from(&key);
        let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

        println!("Proving (parallel, {} partitions)...", num_partitions);
        let mut transcript = Transcript::new(b"zkml-whisper");
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, &mut timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying (parallel)...");
        let mut verify_transcript = Transcript::new(b"zkml-whisper");
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
        let mut transcript = Transcript::new(b"zkml-whisper");
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        println!("Verifying...");
        let mut verify_transcript = Transcript::new(b"zkml-whisper");
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
