use crate::dag::{DagBuilder, EdgeId, Witness};

// ============================================================================
// PointPillar Model Builder
// ============================================================================

/// Conv2D 3×3 + (fused BN) + ReLU with padding.
fn conv3x3_bn_relu(
    g: &mut DagBuilder,
    x: EdgeId,
    w: EdgeId,
    bias: EdgeId,
    stride: (usize, usize),
) -> EdgeId {
    let h = g.pad(x, 1, 1);
    let h = if stride.0 > 1 || stride.1 > 1 {
        g.conv2d_strided(h, w, (3, 3), stride)[0]
    } else {
        g.conv2d(h, w, (3, 3))[0]
    };
    let h = g.add(h, bias)[0];
    g.relu(h)
}

/// Conv2D 1×1 + (fused BN) + ReLU.
#[allow(dead_code)]
fn conv1x1_bn_relu(
    g: &mut DagBuilder,
    x: EdgeId,
    w: EdgeId,
    bias: EdgeId,
) -> EdgeId {
    let h = g.conv2d(x, w, (1, 1))[0];
    let h = g.add(h, bias)[0];
    g.relu(h)
}

/// ConvTranspose2D + (fused BN) + ReLU.
fn deconv_bn_relu(
    g: &mut DagBuilder,
    x: EdgeId,
    w: EdgeId,
    bias: EdgeId,
    kernel: (usize, usize),
    stride: (usize, usize),
) -> EdgeId {
    let h = g.conv_transpose2d(x, w, kernel, stride)[0];
    let h = g.add(h, bias)[0];
    g.relu(h)
}

/// Load N weight pairs from the weight list, advancing the index.
fn load_weights(g: &mut DagBuilder, all_weights: &[(Witness, Witness)], wi: &mut usize, n: usize) -> Vec<(EdgeId, EdgeId)> {
    let mut pairs = Vec::with_capacity(n);
    for _ in 0..n {
        let (w, b) = all_weights[*wi].clone();
        let we = g.param(w);
        let be = g.param(b);
        *wi += 1;
        pairs.push((we, be));
    }
    pairs
}

/// Conv layer configs for PointPillar.
/// (c_in, c_out, kH, kW, is_transpose)
pub fn pointpillar_conv_configs(_ny: usize, _nx: usize, num_anchors: usize, num_classes: usize) -> Vec<(usize, usize, usize, usize, bool)> {
    let mut configs = Vec::new();

    // PillarVFE: Linear 11→64 (implemented as einsum, not counted as conv)

    // BEV Backbone Block 1: stride-2 downsample + 5 × 3×3 conv [64→64]
    configs.push((64, 64, 3, 3, false));  // stride 2
    for _ in 0..5 {
        configs.push((64, 64, 3, 3, false));  // stride 1
    }

    // BEV Backbone Block 2: stride-2 downsample + 5 × 3×3 conv [64→128]
    configs.push((64, 128, 3, 3, false));  // stride 2
    for _ in 0..5 {
        configs.push((128, 128, 3, 3, false));  // stride 1
    }

    // BEV Backbone Block 3: stride-2 downsample + 5 × 3×3 conv [128→256]
    configs.push((128, 256, 3, 3, false));  // stride 2
    for _ in 0..5 {
        configs.push((256, 256, 3, 3, false));  // stride 1
    }

    // Deblock 1: ConvTranspose2d(64→128, k=1, s=1)
    configs.push((64, 128, 1, 1, true));

    // Deblock 2: ConvTranspose2d(128→256, k=2, s=2)
    configs.push((128, 256, 2, 2, true));

    // Deblock 3: two cascaded ConvTranspose2d for 4× upsample
    // (256→256, k=2, s=2) then (256→256, k=2, s=2)
    configs.push((256, 256, 2, 2, true));
    configs.push((256, 256, 2, 2, true));

    // Detection heads: 3 × Conv2D 1×1
    // After multi_concat of [128, 256, 256]: general_concat pads to power-of-2 channels
    // concat(128,256) → pad 128→256, concat → 512; concat(512,256) → pad 256→512, concat → 1024
    let concat_channels = 1024;
    // Classification head: concat_channels → num_anchors * num_classes
    configs.push((concat_channels, num_anchors * num_classes, 1, 1, false));
    // Box regression head: concat_channels → num_anchors * 7
    configs.push((concat_channels, num_anchors * 7, 1, 1, false));
    // Direction head: concat_channels → num_anchors * 2
    configs.push((concat_channels, num_anchors * 2, 1, 1, false));

    configs
}

/// Build a PointPillar model graph.
///
/// `all_weights`: (conv_weight, bias) pairs for each layer in order.
///   First pair is the VFE linear weight [11, 64] and bias [64].
/// `ny`, `nx`: BEV grid dimensions.
/// `n_pillars`: number of pillars.
/// `max_points`: max points per pillar.
/// `num_anchors`: number of anchors per cell.
/// `num_classes`: number of object classes.
pub fn pointpillar(
    all_weights: Vec<(Witness, Witness)>,
    ny: usize,
    nx: usize,
    n_pillars: usize,
    max_points: usize,
    num_anchors: usize,
    num_classes: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, inputs| {
        assert_eq!(inputs.len(), 2, "PointPillar expects 2 inputs: pillars + coords");
        // num_anchors/num_classes determine detection head output channels;
        // channel info is already encoded in the weight tensors passed via all_weights.
        let _ = (num_anchors, num_classes);
        let pillar_input = inputs[0];
        let coords = inputs[1];
        let mut wi = 0usize;

        // ================================================================
        // PillarVFE: Linear(11→64) + ReLU + MaxPool
        // ================================================================
        let vfe_w = load_weights(g, &all_weights, &mut wi,1);
        // pillars: [N_pillars, max_points, 11]
        // Linear: einsum "abc,cd->abd" with W[11, 64]
        let h = g.einsum("abc,cd->abd".to_string(), vec![pillar_input, vfe_w[0].0], false)[0];
        // Add bias: broadcast [64] to [N_pillars, max_points, 64]
        let h = g.add(h, vfe_w[0].1)[0];
        let h = g.relu(h);
        // MaxPool over points dimension: [N_pillars, max_points, 64] → [N_pillars, 64]
        let h = g.pillar_maxpool(h, n_pillars, max_points, 64);

        g.layer_boundaries.push(h);

        // ================================================================
        // Scatter to BEV: [N_pillars, 64] → [64, ny, nx]
        // ================================================================
        let h = g.scatter_to_bev(h, coords, n_pillars, 64, ny, nx);

        g.layer_boundaries.push(h);

        // ================================================================
        // BEV Backbone: 3 blocks + 3 deblocks + concat
        // ================================================================
        // Block 1: stride-2 downsample + 5 × conv 3×3 [64→64]
        let block1_w = load_weights(g, &all_weights, &mut wi,6);
        let mut b1 = conv3x3_bn_relu(g, h, block1_w[0].0, block1_w[0].1, (2, 2));
        for i in 1..6 {
            b1 = conv3x3_bn_relu(g, b1, block1_w[i].0, block1_w[i].1, (1, 1));
        }
        // b1: [64, ny/2, nx/2]
        g.layer_boundaries.push(b1);

        // Block 2: stride-2 downsample + 5 × conv 3×3 [64→128]
        let block2_w = load_weights(g, &all_weights, &mut wi,6);
        let mut b2 = conv3x3_bn_relu(g, b1, block2_w[0].0, block2_w[0].1, (2, 2));
        for i in 1..6 {
            b2 = conv3x3_bn_relu(g, b2, block2_w[i].0, block2_w[i].1, (1, 1));
        }
        // b2: [128, ny/4, nx/4]
        g.layer_boundaries.push(b2);

        // Block 3: stride-2 downsample + 5 × conv 3×3 [128→256]
        let block3_w = load_weights(g, &all_weights, &mut wi,6);
        let mut b3 = conv3x3_bn_relu(g, b2, block3_w[0].0, block3_w[0].1, (2, 2));
        for i in 1..6 {
            b3 = conv3x3_bn_relu(g, b3, block3_w[i].0, block3_w[i].1, (1, 1));
        }
        // b3: [256, ny/8, nx/8]
        g.layer_boundaries.push(b3);

        // Deblock 1: ConvTranspose2d(64→128, k=1, s=1) on b1
        let deblock_w = load_weights(g, &all_weights, &mut wi,4);
        let d1 = deconv_bn_relu(g, b1, deblock_w[0].0, deblock_w[0].1, (1, 1), (1, 1));
        // d1: [128, ny/2, nx/2]

        // Deblock 2: ConvTranspose2d(128→256, k=2, s=2) on b2
        let d2 = deconv_bn_relu(g, b2, deblock_w[1].0, deblock_w[1].1, (2, 2), (2, 2));
        // d2: [256, ny/2, nx/2]

        // Deblock 3: two cascaded ConvTranspose2d(256→256, k=2, s=2) on b3
        let d3 = deconv_bn_relu(g, b3, deblock_w[2].0, deblock_w[2].1, (2, 2), (2, 2));
        let d3 = deconv_bn_relu(g, d3, deblock_w[3].0, deblock_w[3].1, (2, 2), (2, 2));
        // d3: [256, ny/2, nx/2]

        // Concat all deblocks: [128 + 256 + 256 = 640, ny/2, nx/2]
        let bev_feat = g.multi_concat(vec![d1, d2, d3]);

        g.layer_boundaries.push(bev_feat);

        // ================================================================
        // Detection Heads: 3 × Conv2D 1×1
        // ================================================================
        let head_w = load_weights(g, &all_weights, &mut wi,3);

        // Classification: 640 → num_anchors * num_classes
        let cls_out = g.conv2d(bev_feat, head_w[0].0, (1, 1))[0];
        let cls_out = g.add(cls_out, head_w[0].1)[0];

        // Box regression: 640 → num_anchors * 7
        let box_out = g.conv2d(bev_feat, head_w[1].0, (1, 1))[0];
        let box_out = g.add(box_out, head_w[1].1)[0];

        // Direction: 640 → num_anchors * 2
        let dir_out = g.conv2d(bev_feat, head_w[2].0, (1, 1))[0];
        let dir_out = g.add(dir_out, head_w[2].1)[0];

        vec![cls_out, box_out, dir_out]
    }
}
