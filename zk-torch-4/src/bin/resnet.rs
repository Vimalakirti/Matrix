//! ResNet-50 end-to-end prover binary. Ports zk-torch-3's `bin/resnet.rs`
//! to the zk-torch-4 commit/prove API (Ajtai commit, fold-tree opening).
//!
//! Defaults: 1 conv layer of ImageNet ResNet-50, input `[3, 224, 224]`,
//! 1000 classes. Override `NUM_LAYERS`, `MAX_NUM_VARS`, `ZK4_B`, `ZK4_BASE`
//! via env vars. Random Uint weights and input (datatype matches the
//! integer-only conv path).

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::resnet::{resnet50, resnet50_conv_configs, resnet50_output_shape};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;
use zk_torch_4::SF_LOG;

/// Tiny magnitudes (`% 2`) so the conv-stem accumulator stays inside the
/// range lookup table (2^10 = 1024 from `bench_config.yaml::table_size_log`).
/// 7×7×3 = 147-term sum of products, each product ≤ 1·1 = 1, max ≤ 147.
fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 2)
}

/// Conv weight [C_out, C_in, kH, kW] with little-endian layout
/// `kw | kh | c_in | c_out`; padded entries (k >= K) zeroed.
fn gen_conv_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kh_pad * kw_pad;

    let mut data = zk_torch_4::zero_witness_vec(size);
    let mut rng = rand::thread_rng();
    for d in 0..c_out {
        for c in 0..c_in {
            for kh_i in 0..kh {
                for kw_i in 0..kw {
                    let idx = kw_i + kh_i * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                    data[idx] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
                }
            }
        }
    }
    // sf = *SF_LOG (not 0): resnet50's per-conv ScaleDown only activates for
    // sf_log > 0. Without it the %2 activations compound ~x(fan-in) per layer
    // and overflow the b-bit signed plane decomposition of the committed conv
    // outputs at NUM_LAYERS >= ~10 ("per-plane evals don't reconstruct").
    Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn gen_fc_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let mut data = zk_torch_4::zero_witness_vec(size);
    let mut rng = rand::thread_rng();
    for i in 0..in_dim {
        for j in 0..out_dim {
            data[i + j * in_pad] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
        }
    }
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn gen_fc_bias(out_dim: usize) -> Witness {
    let out_pad = out_dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(out_pad);
    let mut rng = rand::thread_rng();
    for i in 0..out_dim {
        data[i] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
    }
    Witness::new(vec![out_dim], data, DataType::Uint, *SF_LOG, Role::Constant)
}

fn demo_seed() -> Seed {
    Seed([
        0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
        0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
    ])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    // ImageNet uses 224. Toy smoke runs use a smaller spatial side so
    // the NonNegative aux (arity = log2(c_out·h·w_pad) + table_size_log)
    // fits in GPU memory. INPUT_SIZE=32 gives aux arity ≈ 24.
    let input_size: usize = std::env::var("INPUT_SIZE").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let num_classes = 1000usize;

    let configs = resnet50_conv_configs();
    let max_conv = num_layers.min(configs.len());

    println!("=== ResNet-50 on Almost-Goldilocks ({} conv layers, input {}x{}) ===",
             max_conv, input_size, input_size);
    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());

    // ---- 1. Generate weights + build DAG ----
    let t0 = Instant::now();
    let conv_weights: Vec<Witness> = configs[..max_conv].iter()
        .map(|&(c_in, c_out, kh, kw)| gen_conv_weight(c_in, c_out, kh, kw))
        .collect();
    let (out_c, _out_spatial) = resnet50_output_shape(max_conv);
    let fc_weight = gen_fc_weight(out_c, num_classes);
    let fc_bias = gen_fc_bias(num_classes);
    println!("Weight gen: {:?}", t0.elapsed());

    // BATCH images per proof. The builder creates the conv/FC param edges ONCE
    // and reuses them for every batch element, so batching amortizes the weight
    // commitment across the batch rather than replicating it — which is the
    // point of proving a batch instead of B separate inferences.
    let batch: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(batch >= 1, "BATCH must be >= 1");

    let mut g = DagBuilder::new();
    let xs: Vec<_> = (0..batch)
        .map(|_| g.input(vec![3, input_size, input_size], DataType::Uint))
        .collect();
    let _output = g.pipe(&xs, resnet50(conv_weights, fc_weight, Some(fc_bias), num_classes, max_conv));

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // ---- 2. Forward pass ----
    // Input [3, S, S] padded to [4, S_pad, S_pad].
    let s_pad = input_size.next_power_of_two();
    let input_buf_size = 4 * s_pad * s_pad;
    // Input sf must match the weights' sf: the builder's post-conv ScaleDown
    // rescales from 2*sf_log back to sf_log (conv-out sf = x.sf + w.sf).
    let inputs: Vec<_> = (0..batch)
        .map(|i| {
            (i, Witness::new(
                vec![3, input_size, input_size],
                rand_field_vec(input_buf_size),
                DataType::Uint, *SF_LOG, Role::Input,
            ))
        })
        .collect();
    let t2 = Instant::now();
    dag.run(&mut witnesses, &inputs);
    println!("Forward: {:?}", t2.elapsed());

    // ---- 3. Commit (offline + online) ----
    let b: usize = std::env::var("ZK4_B").ok().and_then(|s| s.parse().ok()).unwrap_or(21);
    let base: usize = std::env::var("ZK4_BASE").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let key = AjtaiKey::new_with_base(demo_seed(), max_num_vars, b, base);
    let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
    let t_off = Instant::now();
    dag.commit_constants(&witnesses, &mut store);
    let offline_commit = t_off.elapsed();
    let t_on = Instant::now();
    dag.commit_remaining(&witnesses, &mut store);
    let online_commit = t_on.elapsed();
    println!("Commit (offline, amortized): {:?}", offline_commit);
    println!("Commit (online, prover time): {:?}", online_commit);

    // ---- 4. Prove ----
    let mut t_prove = Transcript::new(b"zkml-resnet");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    // ---- 5. Verify ----
    let mut t_verify = Transcript::new(b"zkml-resnet");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
}
