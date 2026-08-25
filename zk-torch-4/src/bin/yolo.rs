//! YOLOv11n end-to-end prover binary. Ports zk-torch-3's `bin/yolo.rs`
//! to the zk-torch-4 commit/prove API.
//!
//! Defaults: 4 stages (backbone only), 640×640 input. Override
//! `NUM_STAGES`, `INPUT_SIZE`, `MAX_NUM_VARS`, `ZK4_B`, `ZK4_BASE`
//! via env vars.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::yolo::{yolov11n, generate_yolo_weight};
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;

/// Tiny magnitudes (`% 2`). zk-torch-3's `% 500` default overflows both the
/// range lookup table and the b=21 signed plane decomposition of the
/// fold-tree opening ("per-plane evals don't reconstruct combined claim") —
/// see resnet.rs / vgg.rs / unet3d.rs for the same convention.
fn rand_field_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    zk_torch_4::rand_witness_vec(size, 2)
}

/// Builds every (weight, bias) pair for YOLOv11n up to `num_stages`.
/// Matches the per-stage shape table from the zk-torch-3 bin verbatim.
fn gen_all_weights(num_stages: usize, input_spatial: usize) -> Vec<(Witness, Witness)> {
    let mut weights = Vec::new();

    let s1 = input_spatial / 2;
    let s2 = input_spatial / 4;
    let s3 = input_spatial / 8;
    let s4 = input_spatial / 16;
    let s5 = input_spatial / 32;

    macro_rules! w {
        ($c_in:expr, $c_out:expr, $k:expr, $dw:expr, $h:expr, $w_dim:expr) => {
            weights.push(generate_yolo_weight($c_in, $c_out, $k, $k, $dw, $h, $w_dim));
        };
    }

    if num_stages >= 1 { w!(3, 16, 3, false, s1, s1); }

    if num_stages >= 2 { w!(16, 32, 3, false, s2, s2); }

    if num_stages >= 3 {
        w!(32, 64, 1, false, s2, s2);
        w!(32, 16, 3, false, s2, s2);
        w!(16, 32, 3, false, s2, s2);
        w!(128, 64, 1, false, s2, s2);
    }

    if num_stages >= 4 {
        w!(64, 64, 3, false, s3, s3);
        w!(64, 128, 1, false, s3, s3);
        w!(64, 32, 3, false, s3, s3);
        w!(32, 64, 3, false, s3, s3);
        w!(256, 128, 1, false, s3, s3);
    }

    if num_stages >= 5 {
        w!(128, 128, 3, false, s4, s4);
        w!(128, 128, 1, false, s4, s4);
        w!(64, 32, 1, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(32, 32, 3, false, s4, s4);
        w!(64, 32, 1, false, s4, s4);
        w!(64, 64, 1, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
    }

    if num_stages >= 6 {
        w!(128, 256, 3, false, s5, s5);
        w!(256, 256, 1, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(128, 128, 1, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
        w!(256, 128, 1, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
    }

    if num_stages >= 7 {
        w!(512, 128, 1, false, s4, s4);
        w!(64, 32, 3, false, s4, s4);
        w!(32, 64, 3, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
        w!(256, 64, 1, false, s3, s3);
        w!(32, 16, 3, false, s3, s3);
        w!(16, 32, 3, false, s3, s3);
        w!(128, 64, 1, false, s3, s3);
        w!(64, 64, 3, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
        w!(64, 32, 3, false, s4, s4);
        w!(32, 64, 3, false, s4, s4);
        w!(256, 128, 1, false, s4, s4);
        w!(128, 128, 3, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(64, 64, 3, false, s5, s5);
        w!(128, 64, 1, false, s5, s5);
        w!(128, 128, 1, false, s5, s5);
        w!(512, 256, 1, false, s5, s5);
    }

    if num_stages >= 8 {
        // P3 head
        w!(64, 64, 3, false, s3, s3);
        w!(64, 64, 3, false, s3, s3);
        w!(64, 64, 1, false, s3, s3);
        w!(64, 64, 3, true, s3, s3);
        w!(64, 80, 1, false, s3, s3);
        w!(80, 80, 3, true, s3, s3);
        w!(80, 80, 1, false, s3, s3);
        w!(80, 80, 1, false, s3, s3);

        // P4 head
        w!(128, 64, 3, false, s4, s4);
        w!(64, 64, 3, false, s4, s4);
        w!(64, 64, 1, false, s4, s4);
        w!(128, 128, 3, true, s4, s4);
        w!(128, 80, 1, false, s4, s4);
        w!(80, 80, 3, true, s4, s4);
        w!(80, 80, 1, false, s4, s4);
        w!(80, 80, 1, false, s4, s4);

        // P5 head
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

fn demo_seed() -> Seed {
    Seed([
        0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
        0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
    ])
}

fn main() {
    env_logger::init();
    almost_goldilocks_cuda::init().expect("CUDA init");

    let num_stages: usize = std::env::var("NUM_STAGES").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let input_spatial: usize = std::env::var("INPUT_SIZE").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(640);
    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);

    println!("=== YOLOv11n on Almost-Goldilocks ({} stages, {}x{}) ===",
             num_stages, input_spatial, input_spatial);
    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());

    let t0 = Instant::now();
    let all_weights = gen_all_weights(num_stages, input_spatial);
    println!("Weight gen: {:?} ({} conv layers)", t0.elapsed(), all_weights.len());

    let mut g = DagBuilder::new();
        // BATCH inputs per proof. The builder creates the weight edges ONCE and
    // reuses them across batch elements, so batching amortizes the weight
    // commitment instead of replicating it.
    let batch: usize = std::env::var("BATCH").or_else(|_| std::env::var("BATCH_SIZE")).ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert!(batch >= 1, "BATCH must be >= 1");
    let xs: Vec<_> = (0..batch)
        .map(|_| g.input(vec![3, input_spatial, input_spatial], DataType::Uint))
        .collect();
    let x = xs[0];
    let _ = x;
    let _ = g.pipe(&xs, yolov11n(all_weights, num_stages));

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // 3 → pad to 4 channels.
    let c_pad = 4;
    let input = Witness::new(
        vec![3, input_spatial, input_spatial],
        rand_field_vec(c_pad * input_spatial * input_spatial),
        DataType::Uint, 0, Role::Input,
    );
    let t2 = Instant::now();
    let inputs: Vec<_> = (0..batch).map(|i| (i, input.clone())).collect();
    dag.run(&mut witnesses, &inputs);
    println!("Forward: {:?}", t2.elapsed());

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

    let mut t_prove = Transcript::new(b"zkml-yolo");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-yolo");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
}
