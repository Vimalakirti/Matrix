//! PointPillar end-to-end prover binary. Ports the PointPillar stage of
//! zk-torch-3's `bin/pointpainting.rs` to zk-torch-4. The DeepLabV3+ and
//! fusion stages of the original PointPainting pipeline aren't ported
//! (DeepLabV3+ DAG is not yet available in zk-torch-4); the fusion stage
//! is a small wrapper over `gather_from_grid`/`general_concat` that could
//! be added later in the same bin.
//!
//! Defaults: tiny config (`ny = 16, nx = 16, n_pillars = 32,
//! max_points = 4, num_anchors = 2, num_classes = 5`). Override via env
//! `NY/NX/N_PILLARS/MAX_POINTS/NUM_ANCHORS/NUM_CLASSES`, plus the usual
//! `MAX_NUM_VARS/ZK4_B/ZK4_BASE`.
//!
//! All conv/transpose weights are all-zero; bias witnesses use `% 2`
//! values so the ReLU output stays inside the range-table.

use std::time::Instant;

use almost_goldilocks_cuda::ajtai::Seed;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rand::Rng;

use zk_torch_4::commit::{AjtaiKey, GpuAjtaiStore};
use zk_torch_4::dag::pointpillar::pointpillar;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::transcript::Transcript;

fn zero_conv_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kh_pad * kw_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, 0, Role::Constant)
}

fn zero_conv_transpose_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_in_pad * c_out_pad * kh_pad * kw_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![c_in, c_out, kh, kw], data, DataType::Uint, 0, Role::Constant)
}

fn zero_linear_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let data = zk_torch_4::zero_witness_vec(size);
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, 0, Role::Constant)
}

/// Broadcast 1D bias `[c]` → 3D `[c, h, w]`, with tiny `% 2` values so
/// the post-conv ReLU stays inside `[0, 2^TABLE_SIZE_LOG)`.
fn small_bias_3d(c_out: usize, h: usize, w: usize) -> Witness {
    let c_pad = c_out.next_power_of_two();
    let h_pad = h.next_power_of_two();
    let w_pad = w.next_power_of_two();
    let size = c_pad * h_pad * w_pad;
    let mut data = zk_torch_4::zero_witness_vec(size);
    let mut rng = rand::thread_rng();
    for c in 0..c_out {
        let v = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
        for hi in 0..h {
            for wi in 0..w {
                data[wi + hi * w_pad + c * w_pad * h_pad] = v;
            }
        }
    }
    Witness::new(vec![c_out, h, w], data, DataType::Uint, 0, Role::Constant)
}

fn small_bias_1d(dim: usize) -> Witness {
    let dim_pad = dim.next_power_of_two();
    let mut data = zk_torch_4::zero_witness_vec(dim_pad);
    let mut rng = rand::thread_rng();
    for i in 0..dim {
        data[i] = AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64);
    }
    Witness::new(vec![dim], data, DataType::Uint, 0, Role::Constant)
}

fn rand_uint_vec(size: usize) -> Vec<AlmostGoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| AlmostGoldilocksField((rng.gen::<u32>() % 2) as u64)).collect()
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

    let max_num_vars: usize = std::env::var("MAX_NUM_VARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(22);
    let ny: usize = std::env::var("NY").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let nx: usize = std::env::var("NX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    let n_pillars: usize = std::env::var("N_PILLARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_points: usize = std::env::var("MAX_POINTS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4);
    let num_anchors: usize = std::env::var("NUM_ANCHORS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(2);
    let num_classes: usize = std::env::var("NUM_CLASSES").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("=== PointPillar on Almost-Goldilocks ===");
    println!("BEV {}x{}, n_pillars={} max_points={} num_anchors={} num_classes={}",
             ny, nx, n_pillars, max_points, num_anchors, num_classes);
    println!("max_num_vars={} (threads={})", max_num_vars, rayon::current_num_threads());

    let t0 = Instant::now();
    let mut all_weights: Vec<(Witness, Witness)> = Vec::new();

    // VFE: Linear 11→64
    all_weights.push((zero_linear_weight(11, 64), small_bias_1d(64)));

    // Spatial tracking
    let mut sh = ny / 2;
    let mut sw = nx / 2;

    // Block 1: 6 convs (64→64, first stride 2)
    all_weights.push((zero_conv_weight(64, 64, 3, 3), small_bias_3d(64, sh, sw)));
    for _ in 0..5 {
        all_weights.push((zero_conv_weight(64, 64, 3, 3), small_bias_3d(64, sh, sw)));
    }
    let b1_h = sh;
    let b1_w = sw;

    // Block 2: 6 convs (64→128, first stride 2)
    sh /= 2; sw /= 2;
    all_weights.push((zero_conv_weight(64, 128, 3, 3), small_bias_3d(128, sh, sw)));
    for _ in 0..5 {
        all_weights.push((zero_conv_weight(128, 128, 3, 3), small_bias_3d(128, sh, sw)));
    }

    // Block 3: 6 convs (128→256, first stride 2)
    sh /= 2; sw /= 2;
    all_weights.push((zero_conv_weight(128, 256, 3, 3), small_bias_3d(256, sh, sw)));
    for _ in 0..5 {
        all_weights.push((zero_conv_weight(256, 256, 3, 3), small_bias_3d(256, sh, sw)));
    }

    // Deblock 1: ConvTranspose2d(64→128, k=1, s=1)
    all_weights.push((zero_conv_transpose_weight(64, 128, 1, 1), small_bias_3d(128, b1_h, b1_w)));
    // Deblock 2: ConvTranspose2d(128→256, k=2, s=2)
    all_weights.push((zero_conv_transpose_weight(128, 256, 2, 2), small_bias_3d(256, b1_h, b1_w)));
    // Deblock 3: two cascaded ConvTranspose2d(256→256, k=2, s=2)
    all_weights.push((zero_conv_transpose_weight(256, 256, 2, 2), small_bias_3d(256, sh * 2, sw * 2)));
    all_weights.push((zero_conv_transpose_weight(256, 256, 2, 2), small_bias_3d(256, b1_h, b1_w)));

    // Detection heads (3 × Conv2D 1×1, input = multi_concat of [128, 256, 256]
    // padded to 1024 channels per pointpillar DAG).
    let det_h = b1_h;
    let det_w = b1_w;
    let cls_out = num_anchors * num_classes;
    let box_out = num_anchors * 7;
    let dir_out = num_anchors * 2;
    let concat_channels = 1024;
    all_weights.push((zero_conv_weight(concat_channels, cls_out, 1, 1), small_bias_3d(cls_out, det_h, det_w)));
    all_weights.push((zero_conv_weight(concat_channels, box_out, 1, 1), small_bias_3d(box_out, det_h, det_w)));
    all_weights.push((zero_conv_weight(concat_channels, dir_out, 1, 1), small_bias_3d(dir_out, det_h, det_w)));

    println!("Weight gen: {:?} ({} layers)", t0.elapsed(), all_weights.len());

    let mut g = DagBuilder::new();
    let pillars = g.input(vec![n_pillars, max_points, 11], DataType::Uint);
    let coords = g.input(vec![n_pillars, 2], DataType::Uint);
    let _output = g.pipe(
        &[pillars, coords],
        pointpillar(all_weights, ny, nx, n_pillars, max_points, num_anchors, num_classes),
    );

    let t1 = Instant::now();
    let (dag, mut witnesses) = g.compile();
    println!("Compile: {:?}  ({} nodes, {} edges)",
             t1.elapsed(), dag.nodes.len(), dag.num_edges());

    // Input data: pillars (random uint) + coords (random uint within ny/nx).
    let n_pad = n_pillars.next_power_of_two();
    let mp_pad = max_points.next_power_of_two();
    let pillar_size = n_pad * mp_pad * 16; // 11 padded to 16
    let pillar_witness = Witness::new(
        vec![n_pillars, max_points, 11],
        rand_uint_vec(pillar_size),
        DataType::Uint, 0, Role::Input,
    );

    let coord_dim_pad = 2usize.next_power_of_two();
    let coord_size = n_pad * coord_dim_pad;
    let mut coord_data = zk_torch_4::zero_witness_vec(coord_size);
    let mut rng = rand::thread_rng();
    for p in 0..n_pillars {
        // Column-major: dim 0 has stride 1, dim 1 has stride n_pad.
        // So index `[p, 0] → p`, `[p, 1] → p + n_pad`.
        coord_data[p] = AlmostGoldilocksField((rng.gen::<u32>() as usize % ny) as u64);
        coord_data[p + n_pad] = AlmostGoldilocksField((rng.gen::<u32>() as usize % nx) as u64);
    }
    let coord_witness = Witness::new(
        vec![n_pillars, 2],
        coord_data,
        DataType::Uint, 0, Role::Input,
    );

    let t2 = Instant::now();
    dag.run(&mut witnesses, &[(0, pillar_witness), (1, coord_witness)]);
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

    let mut t_prove = Transcript::new(b"zkml-pointpillar");
    let t4 = Instant::now();
    let (dag_proof, fold_proof) = dag.prove_with_fold_tree(&witnesses, &store, &mut t_prove);
    println!("Prove: {:?}", t4.elapsed());

    let mut t_verify = Transcript::new(b"zkml-pointpillar");
    let t5 = Instant::now();
    let verified = dag.verify_with_fold_tree(&witnesses, &store, &dag_proof, &fold_proof, &mut t_verify);
    println!("Verify: {:?}", t5.elapsed());
    // Serialized proof size, reported by the evaluation harness.
    let proof_bytes = bincode::serialize(&dag_proof).unwrap().len()
        + bincode::serialize(&fold_proof).unwrap().len();
    println!("Proof size: {} bytes", proof_bytes);

    println!("\nVerified: {}", verified);
}
