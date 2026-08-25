use std::time::Instant;

use goldilocks_cuda::GoldilocksField;
use plonky2::util::timing::TimingTree;
use rand::Rng;

use goldilocks_cuda::basefold::BasefoldTable;
use zk_torch_3::commit::basefold::{BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey, GpuCommitmentStore};
use zk_torch_3::dag::{DagBuilder, DataType, Role, Witness, partition_dag};
use zk_torch_3::dag::deeplabv3plus::{deeplabv3plus, deeplabv3plus_conv_configs};
use zk_torch_3::dag::pointpillar::pointpillar;
use zk_torch_3::transcript::Transcript;

fn generate_random_field_vec(size: usize) -> Vec<GoldilocksField> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| GoldilocksField((rng.gen::<u32>() % 500) as u64))
        .collect()
}

/// Generate conv weight: [c_out, c_in, kH, kW] with zero-padding for FlattenKernel.
fn generate_conv_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_out_pad * c_in_pad * kh_pad * kw_pad;

    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for d in 0..c_out {
        for c in 0..c_in {
            for kh_i in 0..kh {
                for kw_i in 0..kw {
                    let idx = kw_i + kh_i * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                    data[idx] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
                }
            }
        }
    }
    Witness::new(vec![c_out, c_in, kh, kw], data, DataType::Uint, 0, Role::Constant)
}

/// Generate depthwise conv weight: [C, 1, kH, kW].
fn generate_dw_conv_weight(channels: usize, kh: usize, kw: usize) -> Witness {
    let c_pad = channels.next_power_of_two();
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let size = c_pad * kh_pad * kw_pad;

    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for ch in 0..channels {
        for ki in 0..kh {
            for kj in 0..kw {
                let idx = kj + ki * kw_pad + ch * kw_pad * kh_pad;
                data[idx] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
            }
        }
    }
    Witness::new(vec![channels, 1, kh, kw], data, DataType::Uint, 0, Role::Constant)
}

/// Generate ConvTranspose2D weight: [C_in, C_out, kH, kW].
fn generate_conv_transpose_weight(c_in: usize, c_out: usize, kh: usize, kw: usize) -> Witness {
    let kh_pad = kh.next_power_of_two();
    let kw_pad = kw.next_power_of_two();
    let c_in_pad = c_in.next_power_of_two();
    let c_out_pad = c_out.next_power_of_two();
    let size = c_in_pad * c_out_pad * kh_pad * kw_pad;

    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for ci in 0..c_in {
        for co in 0..c_out {
            for ki in 0..kh {
                for kj in 0..kw {
                    let idx = kj + ki * kw_pad + co * kw_pad * kh_pad + ci * kw_pad * kh_pad * c_out_pad;
                    data[idx] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
                }
            }
        }
    }
    Witness::new(vec![c_in, c_out, kh, kw], data, DataType::Uint, 0, Role::Constant)
}

/// Generate a broadcast bias: [c_out, h, w].
fn generate_bias_3d(c_out: usize, h: usize, w: usize) -> Witness {
    let c_pad = c_out.next_power_of_two();
    let h_pad = h.next_power_of_two();
    let w_pad = w.next_power_of_two();
    let size = c_pad * h_pad * w_pad;
    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for c in 0..c_out {
        let val = GoldilocksField((rng.gen::<u32>() % 50) as u64);
        for hi in 0..h {
            for wi in 0..w {
                data[wi + hi * w_pad + c * w_pad * h_pad] = val;
            }
        }
    }
    Witness::new(vec![c_out, h, w], data, DataType::Uint, 0, Role::Constant)
}

/// Generate a 1D bias: [dim].
fn generate_bias_1d(dim: usize) -> Witness {
    let dim_pad = dim.next_power_of_two();
    let mut data = vec![GoldilocksField(0); dim_pad];
    let mut rng = rand::thread_rng();
    for i in 0..dim {
        data[i] = GoldilocksField((rng.gen::<u32>() % 50) as u64);
    }
    Witness::new(vec![dim], data, DataType::Uint, 0, Role::Constant)
}

/// Generate a weight matrix: [in_dim, out_dim].
fn generate_linear_weight(in_dim: usize, out_dim: usize) -> Witness {
    let in_pad = in_dim.next_power_of_two();
    let out_pad = out_dim.next_power_of_two();
    let size = in_pad * out_pad;
    let mut data = vec![GoldilocksField(0); size];
    let mut rng = rand::thread_rng();
    for i in 0..in_dim {
        for j in 0..out_dim {
            data[j + i * out_pad] = GoldilocksField((rng.gen::<u32>() % 100) as u64);
        }
    }
    Witness::new(vec![in_dim, out_dim], data, DataType::Uint, 0, Role::Constant)
}

fn run_stage(
    name: &str,
    g: DagBuilder,
    input_witnesses: Vec<(usize, Witness)>,
    num_partitions: usize,
    timing: &mut TimingTree,
) {
    println!("\n=== {} ===", name);

    let t1 = Instant::now();
    let (mut dag, mut witnesses) = g.compile();
    println!("Compile: {:?}", t1.elapsed());
    println!("DAG: {} nodes, {} edges", dag.nodes.len(), dag.num_edges());

    if num_partitions > 1 {
        dag.set_partition_boundaries(num_partitions);
        println!("Partition boundaries: {:?}", dag.boundary_edges);
    }

    let t2 = Instant::now();
    dag.run(&mut witnesses, &input_witnesses);
    println!("Run: {:?}", t2.elapsed());

    let key = BasefoldCommitKey::default();
    let max_num_vars = witnesses.iter()
        .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
        .filter(|&n| n <= 22)
        .max().unwrap_or(10);
    let mut gpu_store = GpuCommitmentStore::new(max_num_vars, key.log_rate, key.seed, dag.num_edges());
    let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];

    let t3 = Instant::now();
    let nonweight_commit_time = dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);
    println!("Commit: {:?}", t3.elapsed());
    let vk = BasefoldVerifierKey::from(&key);
    let vk_table = BasefoldTable::generate(max_num_vars, vk.log_rate, max_num_vars, vk.seed);

    if num_partitions > 1 {
        let partitions = partition_dag(&dag, &dag.boundary_edges);
        println!("Partitions: {}", partitions.len());

        let mut transcript = Transcript::new(name.as_bytes());
        let t4 = Instant::now();
        let parallel_proof = dag.prove_parallel(
            &key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &partitions, timing,
        );
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        let mut verify_transcript = Transcript::new(name.as_bytes());
        let t5 = Instant::now();
        let verified = dag.verify_parallel(
            &parallel_proof, &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript, &partitions,
        );
        println!("Verify: {:?}", t5.elapsed());
        println!("Verified: {}", verified);
    } else {
        let mut transcript = Transcript::new(name.as_bytes());
        let t4 = Instant::now();
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, timing);
        let prove_elapsed = t4.elapsed();
        println!("Prove: {:?} (+ commit {:?} = {:?})", prove_elapsed, nonweight_commit_time, prove_elapsed + nonweight_commit_time);

        let mut verify_transcript = Transcript::new(name.as_bytes());
        let t5 = Instant::now();
        let verified = dag.verify(
            &node_proofs, &edge_proofs, &range_proof, &two_pow_proof, &reducer_proofs,
            &witnesses, &vk, &commitments, &vk_table, &mut verify_transcript,
        );
        println!("Verify: {:?}", t5.elapsed());
        println!("Verified: {}", verified);
    }
}

fn build_deeplabv3plus(
    num_layers: usize,
    num_partitions: usize,
    input_h: usize,
    input_w: usize,
    num_classes: usize,
    timing: &mut TimingTree,
) {
    println!("\n=== Stage 1: DeepLabV3+ (ResNet-101 backbone, {} bottleneck layers) ===", num_layers);
    let t0 = Instant::now();

    let configs = deeplabv3plus_conv_configs(num_layers);
    let _total_convs = configs.len() + 1; // +1 for final classifier

    // Track spatial dimensions for bias generation
    let mut spatial_h = input_h;
    let mut spatial_w = input_w;
    let mut all_weights = Vec::new();

    // Generate stem weights (3 convs)
    // conv1: stride 2, pad 1
    {
        let (c_in, c_out, kh, kw, _) = configs[0];
        let w = generate_conv_weight(c_in, c_out, kh, kw);
        spatial_h = (spatial_h + 2 - kh) / 2 + 1; // stride 2
        spatial_w = (spatial_w + 2 - kw) / 2 + 1;
        let b = generate_bias_3d(c_out, spatial_h, spatial_w);
        all_weights.push((w, b));
    }
    // conv2, conv3: pad 1, stride 1
    for i in 1..3 {
        let (c_in, c_out, kh, kw, _) = configs[i];
        let w = generate_conv_weight(c_in, c_out, kh, kw);
        // pad 1 + conv 3×3 → same spatial
        let b = generate_bias_3d(c_out, spatial_h, spatial_w);
        all_weights.push((w, b));
    }
    // After maxpool 3×3 stride 2, pad 1
    spatial_h = (spatial_h + 2 - 3) / 2 + 1;
    spatial_w = (spatial_w + 2 - 3) / 2 + 1;

    // Generate backbone stage weights — same stopping criteria as config generator
    let mut c_in_track = 64usize;
    let mut skip_channels = 64usize;
    let mut conv_count = 3usize; // stem convs
    let stages = [(3, 64, 256, 1, 1), (4, 128, 512, 2, 1), (23, 256, 1024, 1, 2), (3, 512, 2048, 1, 4)];

    for (si, &(num_blocks, c_mid, c_out, stride_first, _dilation)) in stages.iter().enumerate() {
        for block_idx in 0..num_blocks {
            if conv_count >= num_layers { break; }
            let stride = if block_idx == 0 { stride_first } else { 1 };
            let has_projection = block_idx == 0 && (c_in_track != c_out || stride_first > 1);

            // Conv 1×1
            let sh = spatial_h;
            let sw = spatial_w;
            let w = generate_conv_weight(c_in_track, c_mid, 1, 1);
            let b = generate_bias_3d(c_mid, sh, sw);
            all_weights.push((w, b));
            conv_count += 1;

            // Conv 3×3 with stride
            let sh_after = if stride > 1 { sh / stride } else { sh };
            let sw_after = if stride > 1 { sw / stride } else { sw };
            let w = generate_conv_weight(c_mid, c_mid, 3, 3);
            let b = generate_bias_3d(c_mid, sh_after, sw_after);
            all_weights.push((w, b));
            conv_count += 1;

            // Conv 1×1
            let w = generate_conv_weight(c_mid, c_out, 1, 1);
            let b = generate_bias_3d(c_out, sh_after, sw_after);
            all_weights.push((w, b));
            conv_count += 1;

            if has_projection {
                let w = generate_conv_weight(c_in_track, c_out, 1, 1);
                let b = generate_bias_3d(c_out, sh_after, sw_after);
                all_weights.push((w, b));
                conv_count += 1;
            }

            spatial_h = sh_after;
            spatial_w = sw_after;
            c_in_track = c_out;
        }
        if si == 0 { skip_channels = c_in_track; }
        if conv_count >= num_layers { break; }
    }

    // ASPP weights (6 convs)
    let aspp_h = spatial_h;
    let aspp_w = spatial_w;

    // Branch 0: 1×1 (c_in→256), but applied to global avg pool → [256]
    // Weight is [c_in, 256] matrix for einsum, bias is [256]
    let w = generate_linear_weight(c_in_track, 256);
    let b = generate_bias_1d(256);
    all_weights.push((w, b));

    // Branch 1: 1×1 (c_in→256)
    let w = generate_conv_weight(c_in_track, 256, 1, 1);
    let b = generate_bias_3d(256, aspp_h, aspp_w);
    all_weights.push((w, b));

    // Branches 2-4: 3×3 dilated (c_in→256, dil=12/24/36) — same spatial after pad
    for _ in 0..3 {
        let w = generate_conv_weight(c_in_track, 256, 3, 3);
        let b = generate_bias_3d(256, aspp_h, aspp_w);
        all_weights.push((w, b));
    }

    // Fusion: 1×1 (2048→256) — multi_concat of 5×256 branches → 2048 channels
    let w = generate_conv_weight(2048, 256, 1, 1);
    let b = generate_bias_3d(256, aspp_h, aspp_w);
    all_weights.push((w, b));

    // Decoder weights (5 convs)
    let dec_h = aspp_h * 2; // after 2× upsample
    let dec_w = aspp_w * 2;

    // Skip projection: 1×1 (skip_channels→48)
    let skip_h = input_h / 4; // stage 1 output spatial
    let skip_w = input_w / 4;
    let w = generate_conv_weight(skip_channels, 48, 1, 1);
    let b = generate_bias_3d(48, skip_h, skip_w);
    all_weights.push((w, b));

    // DWSepConv 1: DW 3×3. general_concat(256, 48) → 512 channels (power-of-2 padding)
    let dec_concat_channels = 512;
    let w = generate_dw_conv_weight(dec_concat_channels, 3, 3);
    let b = generate_bias_3d(dec_concat_channels, dec_h, dec_w);
    all_weights.push((w, b));
    // PW 1×1 (512→256)
    let w = generate_conv_weight(dec_concat_channels, 256, 1, 1);
    let b = generate_bias_3d(256, dec_h, dec_w);
    all_weights.push((w, b));

    // DWSepConv 2: DW 3×3 (256 channels)
    let w = generate_dw_conv_weight(256, 3, 3);
    let b = generate_bias_3d(256, dec_h, dec_w);
    all_weights.push((w, b));
    // PW 1×1 (256→256)
    let w = generate_conv_weight(256, 256, 1, 1);
    let b = generate_bias_3d(256, dec_h, dec_w);
    all_weights.push((w, b));

    // Final classifier: 1×1 (256→num_classes)
    let w = generate_conv_weight(256, num_classes, 1, 1);
    let b = generate_bias_3d(num_classes, dec_h, dec_w);
    all_weights.push((w, b));

    println!("Weight generation: {:?} ({} conv layers)", t0.elapsed(), all_weights.len());

    let mut g = DagBuilder::new();
    let x = g.input(vec![3, input_h, input_w], DataType::Uint);
    let _output = g.pipe(&[x], deeplabv3plus(all_weights, num_classes, num_layers, input_h, input_w));

    let input_size = 4 * input_h.next_power_of_two() * input_w.next_power_of_two();
    let input_witness = Witness::new(
        vec![3, input_h, input_w],
        generate_random_field_vec(input_size),
        DataType::Uint, 0, Role::Input,
    );

    run_stage("DeepLabV3+", g, vec![(0, input_witness)], num_partitions, timing);
}

fn build_pointpillar(
    num_partitions: usize,
    ny: usize,
    nx: usize,
    n_pillars: usize,
    max_points: usize,
    num_anchors: usize,
    num_classes: usize,
    timing: &mut TimingTree,
) {
    println!("\n=== Stage 3: PointPillar (BEV {}×{}, {} pillars) ===", ny, nx, n_pillars);
    let t0 = Instant::now();

    let mut all_weights = Vec::new();

    // VFE: Linear 11→64
    let w = generate_linear_weight(11, 64);
    let b = generate_bias_1d(64);
    all_weights.push((w, b));

    // Spatial tracking
    let mut sh = ny / 2;
    let mut sw = nx / 2;

    // Block 1: 6 convs (64→64)
    let w = generate_conv_weight(64, 64, 3, 3);
    let b = generate_bias_3d(64, sh, sw);
    all_weights.push((w, b));
    for _ in 0..5 {
        let w = generate_conv_weight(64, 64, 3, 3);
        let b = generate_bias_3d(64, sh, sw);
        all_weights.push((w, b));
    }
    let b1_h = sh;
    let b1_w = sw;

    // Block 2: 6 convs (64→128, first stride 2)
    sh /= 2;
    sw /= 2;
    let w = generate_conv_weight(64, 128, 3, 3);
    let b = generate_bias_3d(128, sh, sw);
    all_weights.push((w, b));
    for _ in 0..5 {
        let w = generate_conv_weight(128, 128, 3, 3);
        let b = generate_bias_3d(128, sh, sw);
        all_weights.push((w, b));
    }
    let _b2_h = sh;
    let _b2_w = sw;

    // Block 3: 6 convs (128→256, first stride 2)
    sh /= 2;
    sw /= 2;
    let w = generate_conv_weight(128, 256, 3, 3);
    let b = generate_bias_3d(256, sh, sw);
    all_weights.push((w, b));
    for _ in 0..5 {
        let w = generate_conv_weight(256, 256, 3, 3);
        let b = generate_bias_3d(256, sh, sw);
        all_weights.push((w, b));
    }

    // Deblock 1: ConvTranspose2d(64→128, k=1, s=1)
    let w = generate_conv_transpose_weight(64, 128, 1, 1);
    let b = generate_bias_3d(128, b1_h, b1_w);
    all_weights.push((w, b));

    // Deblock 2: ConvTranspose2d(128→256, k=2, s=2) → [256, ny/2, nx/2]
    let w = generate_conv_transpose_weight(128, 256, 2, 2);
    let b = generate_bias_3d(256, b1_h, b1_w);
    all_weights.push((w, b));

    // Deblock 3: 2× ConvTranspose2d(256→256, k=2, s=2) cascaded
    let w = generate_conv_transpose_weight(256, 256, 2, 2);
    let b = generate_bias_3d(256, sh * 2, sw * 2);
    all_weights.push((w, b));
    let w = generate_conv_transpose_weight(256, 256, 2, 2);
    let b = generate_bias_3d(256, b1_h, b1_w);
    all_weights.push((w, b));

    // Detection heads (3 × Conv2D 1×1)
    let det_h = b1_h;
    let det_w = b1_w;
    let cls_out = num_anchors * num_classes;
    let box_out = num_anchors * 7;
    let dir_out = num_anchors * 2;

    // multi_concat of [128, 256, 256] produces 1024 channels (power-of-2 padding)
    let concat_channels = 1024;
    let w = generate_conv_weight(concat_channels, cls_out, 1, 1);
    let b = generate_bias_3d(cls_out, det_h, det_w);
    all_weights.push((w, b));
    let w = generate_conv_weight(concat_channels, box_out, 1, 1);
    let b = generate_bias_3d(box_out, det_h, det_w);
    all_weights.push((w, b));
    let w = generate_conv_weight(concat_channels, dir_out, 1, 1);
    let b = generate_bias_3d(dir_out, det_h, det_w);
    all_weights.push((w, b));

    println!("Weight generation: {:?} ({} layers)", t0.elapsed(), all_weights.len());

    let mut g = DagBuilder::new();
    let n_pad = n_pillars.next_power_of_two();
    let mp_pad = max_points.next_power_of_two();

    let pillars = g.input(vec![n_pillars, max_points, 11], DataType::Uint);
    let coords = g.input(vec![n_pillars, 2], DataType::Uint);
    let _output = g.pipe(&[pillars, coords], pointpillar(all_weights, ny, nx, n_pillars, max_points, num_anchors, num_classes));

    let pillar_size = n_pad * mp_pad * 16; // 11 padded to 16
    let pillar_witness = Witness::new(
        vec![n_pillars, max_points, 11],
        generate_random_field_vec(pillar_size),
        DataType::Uint, 0, Role::Input,
    );

    let coord_size = n_pad * 2;
    let mut coord_data = vec![GoldilocksField(0); coord_size];
    let mut rng = rand::thread_rng();
    for p in 0..n_pillars {
        coord_data[0 + p * 2] = GoldilocksField((rng.gen::<u32>() as usize % ny) as u64);
        coord_data[1 + p * 2] = GoldilocksField((rng.gen::<u32>() as usize % nx) as u64);
    }
    let coord_witness = Witness::new(
        vec![n_pillars, 2],
        coord_data,
        DataType::Uint, 0, Role::Input,
    );

    run_stage("PointPillar", g, vec![(0, pillar_witness), (1, coord_witness)], num_partitions, timing);
}

/// Stage 2: PointPainting fusion.
/// Inputs: seg_map[num_classes, seg_h, seg_w] (from DeepLabV3+),
///         pixel_coords[N_points, 2] (projected LiDAR→camera pixel coords),
///         point_features[N_points, base_features] (original LiDAR features).
/// Output: painted_features[N_points, base_features + num_classes]
///
/// For each LiDAR point:
///   1. Gather segmentation scores at its projected pixel location
///   2. Concatenate scores with original point features
fn build_fusion(
    num_partitions: usize,
    n_points: usize,
    num_classes: usize,
    base_features: usize,
    seg_h: usize,
    seg_w: usize,
    timing: &mut TimingTree,
) {
    println!("\n=== Stage 2: PointPainting Fusion ({} points, {} classes, seg {}×{}) ===",
        n_points, num_classes, seg_h, seg_w);
    let t0 = Instant::now();

    let mut g = DagBuilder::new();

    // Inputs
    let seg_map = g.input(vec![num_classes, seg_h, seg_w], DataType::Uint);
    let pixel_coords = g.input(vec![n_points, 2], DataType::Uint);
    let point_features = g.input(vec![n_points, base_features], DataType::Uint);

    // Gather: sample seg_map at each point's pixel location → [N_points, num_classes]
    let gathered = g.gather_from_grid(seg_map, pixel_coords, n_points, num_classes, seg_h, seg_w);

    // Transpose to [features, N_points] so general_concat works along feature axis (dim 0)
    let feat_t = g.einsum("pf->fp".to_string(), vec![point_features], false)[0];
    let gathered_t = g.einsum("pc->cp".to_string(), vec![gathered], false)[0];

    // Concatenate along feature axis → [base_features + num_classes, N_points]
    let concat = g.general_concat(feat_t, gathered_t);

    // Transpose back to [N_points, base_features + num_classes]
    let _output = g.einsum("fp->pf".to_string(), vec![concat], false)[0];

    println!("Weight generation: {:?}", t0.elapsed());

    // Input witnesses
    let n_pad = n_points.next_power_of_two();
    let c_pad = num_classes.next_power_of_two();
    let h_pad = seg_h.next_power_of_two();
    let w_pad = seg_w.next_power_of_two();
    let bf_pad = base_features.next_power_of_two();

    let seg_witness = Witness::new(
        vec![num_classes, seg_h, seg_w],
        generate_random_field_vec(c_pad * h_pad * w_pad),
        DataType::Uint, 0, Role::Input,
    );

    // Random pixel coords within seg_map bounds
    let coord_dim_pad = 2usize.next_power_of_two();
    let mut coord_data = vec![GoldilocksField(0); n_pad * coord_dim_pad];
    let mut rng = rand::thread_rng();
    for p in 0..n_points {
        coord_data[0 + p * coord_dim_pad] = GoldilocksField((rng.gen::<u32>() as usize % seg_h) as u64);
        coord_data[1 + p * coord_dim_pad] = GoldilocksField((rng.gen::<u32>() as usize % seg_w) as u64);
    }
    let coord_witness = Witness::new(
        vec![n_points, 2],
        coord_data,
        DataType::Uint, 0, Role::Input,
    );

    let feat_witness = Witness::new(
        vec![n_points, base_features],
        generate_random_field_vec(n_pad * bf_pad),
        DataType::Uint, 0, Role::Input,
    );

    run_stage(
        "PointPainting Fusion",
        g,
        vec![(0, seg_witness), (1, coord_witness), (2, feat_witness)],
        num_partitions,
        timing,
    );
}

fn main() {
    let mut timing = TimingTree::default();
    env_logger::init();
    goldilocks_cuda::init().expect("CUDA init failed");

    let stage: String = std::env::var("STAGE").unwrap_or("all".to_string());
    let num_layers: usize = std::env::var("NUM_LAYERS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(33);
    let num_partitions: usize = std::env::var("NUM_PARTITIONS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let input_h: usize = std::env::var("INPUT_H").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(512);
    let input_w: usize = std::env::var("INPUT_W").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(512);
    let ny: usize = std::env::var("NY").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(496);
    let nx: usize = std::env::var("NX").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(432);
    let n_pillars: usize = std::env::var("N_PILLARS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(12000);
    let max_points: usize = std::env::var("MAX_POINTS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(32);
    let num_classes: usize = std::env::var("NUM_CLASSES").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(5);
    let num_anchors = 2usize;

    // Fusion parameters: base_features is the original LiDAR point dimension (before painting)
    let base_features: usize = std::env::var("BASE_FEATURES").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(6);
    // Segmentation output spatial size (DeepLabV3+ decoder output after upsampling)
    let seg_h: usize = std::env::var("SEG_H").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(input_h / 4);
    let seg_w: usize = std::env::var("SEG_W").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(input_w / 4);
    // Number of LiDAR points for fusion (typically more than pillars since multiple points per pillar)
    let n_points: usize = std::env::var("N_POINTS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(n_pillars * max_points);

    let thread_num = rayon::current_num_threads();
    println!("=== PointPainting Pipeline ===");
    println!("Stage: {}, Threads: {}", stage, thread_num);
    println!("DeepLabV3+: {} bottleneck layers, input {}×{}, {} classes",
        num_layers, input_h, input_w, num_classes);
    println!("Fusion: {} points, {} base features, seg {}×{}",
        n_points, base_features, seg_h, seg_w);
    println!("PointPillar: BEV {}×{}, {} pillars, {} max_points",
        ny, nx, n_pillars, max_points);

    match stage.as_str() {
        "deeplabv3" => {
            build_deeplabv3plus(num_layers, num_partitions, input_h, input_w, num_classes, &mut timing);
        }
        "fusion" => {
            build_fusion(num_partitions, n_points, num_classes, base_features, seg_h, seg_w, &mut timing);
        }
        "pointpillar" => {
            build_pointpillar(num_partitions, ny, nx, n_pillars, max_points, num_anchors, num_classes, &mut timing);
        }
        "all" => {
            build_deeplabv3plus(num_layers, num_partitions, input_h, input_w, num_classes, &mut timing);
            build_fusion(num_partitions, n_points, num_classes, base_features, seg_h, seg_w, &mut timing);
            build_pointpillar(num_partitions, ny, nx, n_pillars, max_points, num_anchors, num_classes, &mut timing);
        }
        // Backwards compat
        "both" => {
            build_deeplabv3plus(num_layers, num_partitions, input_h, input_w, num_classes, &mut timing);
            build_pointpillar(num_partitions, ny, nx, n_pillars, max_points, num_anchors, num_classes, &mut timing);
        }
        _ => {
            println!("Unknown stage: {}. Use 'deeplabv3', 'fusion', 'pointpillar', or 'all'", stage);
        }
    }

    timing.print();
}
