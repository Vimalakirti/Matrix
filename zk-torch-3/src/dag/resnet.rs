use crate::dag::{DagBuilder, EdgeId, Witness};

/// ResNet-50 bottleneck block.
/// Conv 1×1 (c_in → c_mid) → ReLU → Conv 3×3 stride (c_mid → c_mid, pad 1) → ReLU → Conv 1×1 (c_mid → c_out)
/// + shortcut (identity or 1×1 projection conv with stride) → Add → ReLU
///
/// `weights`: [conv1_w, conv2_w, conv3_w] and optionally [proj_w] at index 3.
/// `biases`: Optional per-conv biases, same indexing as weights.
fn bottleneck(
    g: &mut DagBuilder,
    x: EdgeId,
    stride: usize,
    has_projection: bool,
    weights: &[EdgeId],
    biases: &[Option<EdgeId>],
    sf_log: usize,
) -> EdgeId {
    // Conv 1×1 (c_in → c_mid), stride=1
    let mut h = g.conv2d(x, weights[0], (1, 1))[0];
    if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
    if let Some(b) = biases.get(0).copied().flatten() { h = g.add(h, b)[0]; }
    h = g.relu(h);

    // Conv 3×3 (c_mid → c_mid), with padding=1, stride=stride
    h = g.pad(h, 1, 1);
    h = if stride > 1 {
        g.conv2d_strided(h, weights[1], (3, 3), (stride, stride))[0]
    } else {
        g.conv2d(h, weights[1], (3, 3))[0]
    };
    if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
    if let Some(b) = biases.get(1).copied().flatten() { h = g.add(h, b)[0]; }
    h = g.relu(h);

    // Conv 1×1 (c_mid → c_out), stride=1
    h = g.conv2d(h, weights[2], (1, 1))[0];
    if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
    if let Some(b) = biases.get(2).copied().flatten() { h = g.add(h, b)[0]; }

    // Shortcut
    let shortcut = if has_projection {
        let mut s = if stride > 1 {
            g.conv2d_strided(x, weights[3], (1, 1), (stride, stride))[0]
        } else {
            g.conv2d(x, weights[3], (1, 1))[0]
        };
        if sf_log > 0 { s = g.scale(s, 2 * sf_log, sf_log)[0]; }
        if let Some(b) = biases.get(3).copied().flatten() { s = g.add(s, b)[0]; }
        s
    } else {
        x
    };

    // Add + ReLU
    h = g.add(h, shortcut)[0];
    h = g.relu(h);
    h
}

/// ResNet-50 stage configurations.
/// (num_blocks, c_mid, c_out, stride_first_block)
const RESNET50_STAGES: [(usize, usize, usize, usize); 4] = [
    (3, 64, 256, 1),    // stage1: no downsampling (stride already done by stem maxpool)
    (4, 128, 512, 2),   // stage2: first block stride 2
    (6, 256, 1024, 2),  // stage3: first block stride 2
    (3, 512, 2048, 2),  // stage4: first block stride 2
];

/// Conv layer configs for ResNet-50: (c_in, c_out, kH, kW) for each conv in order.
/// Returns list of (c_in, c_out, kernel_h, kernel_w) for all convolutions.
pub fn resnet50_conv_configs() -> Vec<(usize, usize, usize, usize)> {
    let mut configs = Vec::new();

    // Stem: conv 7×7, stride 2, c_in=3, c_out=64
    configs.push((3, 64, 7, 7));

    // Stages
    let mut c_in = 64; // after stem
    for &(num_blocks, c_mid, c_out, stride_first) in &RESNET50_STAGES {
        for block_idx in 0..num_blocks {
            let has_projection = block_idx == 0 && (c_in != c_out || stride_first > 1);
            let _stride = if block_idx == 0 { stride_first } else { 1 };

            // Conv 1×1: c_in → c_mid
            configs.push((c_in, c_mid, 1, 1));
            // Conv 3×3: c_mid → c_mid
            configs.push((c_mid, c_mid, 3, 3));
            // Conv 1×1: c_mid → c_out
            configs.push((c_mid, c_out, 1, 1));

            if has_projection {
                // Projection: c_in → c_out, 1×1
                configs.push((c_in, c_out, 1, 1));
            }

            c_in = c_out;
        }
    }

    configs
}

/// Total number of conv layers in ResNet-50.
pub fn resnet50_num_convs() -> usize {
    resnet50_conv_configs().len()
}

/// Build a ResNet-50 model graph.
///
/// `conv_weights`: weight witnesses for each conv layer in order.
/// `conv_biases`: bias witnesses for each conv layer (broadcast to output shape).
///                Empty if no biases (fused BN).
/// `fc_weight`: final FC layer weight [2048, num_classes].
/// `fc_bias`: optional FC bias [num_classes].
pub fn resnet50(
    conv_weights: Vec<Witness>,
    fc_weight: Witness,
    fc_bias: Option<Witness>,
    _num_classes: usize,
    num_layers: usize,  // max conv layers to include (for partial runs)
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    resnet50_with_biases(conv_weights, vec![], fc_weight, fc_bias, _num_classes, num_layers)
}

pub fn resnet50_with_biases(
    conv_weights: Vec<Witness>,
    conv_biases: Vec<Witness>,
    fc_weight: Witness,
    fc_bias: Option<Witness>,
    _num_classes: usize,
    num_layers: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    // When both weight and input have scale 2^sf, the conv output has scale 2^(2*sf).
    // Use ScaleDown to bring it back to 2^sf after each conv.
    let sf_log = conv_biases.first().map(|b| b.sf).unwrap_or(0);

    move |g, inputs| {
        assert_eq!(inputs.len(), 1, "ResNet expects 1 input");
        let has_biases = !conv_biases.is_empty();
        let mut h = inputs[0];
        let mut w_idx = 0;

        if w_idx >= num_layers { return vec![h]; }

        // ================================================================
        // Stem: Conv 7×7 stride 2, pad 3 → ScaleBack → Bias → ReLU → MaxPool
        // ================================================================
        h = g.pad(h, 3, 3);
        let w_e = g.param(conv_weights[w_idx].clone());
        h = g.conv2d_strided(h, w_e, (7, 7), (2, 2))[0];
        if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
        if has_biases {
            let b_e = g.param(conv_biases[w_idx].clone());
            h = g.add(h, b_e)[0];
        }
        h = g.relu(h);
        w_idx += 1;
        // After stem conv: [64, 112, 112]

        // MaxPool 3×3, stride 2, pad=[1,1,1,1] (symmetric, matching ONNX/PyTorch)
        // Input: [64, 112, 112]. After symmetric pad: [64, 114, 114]
        h = g.pad(h, 1, 1);
        h = g.maxpool_general(h, 3, 3, 2, 2);
        // After maxpool: [64, 56, 56]

        g.layer_boundaries.push(h);

        // ================================================================
        // Stages 1-4: bottleneck blocks
        // ================================================================
        let mut c_in = 64usize;

        for &(num_blocks, _c_mid, c_out, stride_first) in &RESNET50_STAGES {
            for block_idx in 0..num_blocks {
                if w_idx >= num_layers { break; }

                let stride = if block_idx == 0 { stride_first } else { 1 };
                let has_projection = block_idx == 0 && (c_in != c_out || stride_first > 1);

                // Need at least 3 conv weights for this block (+ 1 if projection)
                let num_convs_needed = if has_projection { 4 } else { 3 };
                if w_idx + num_convs_needed > conv_weights.len() { break; }

                // Collect weight and bias edges
                let mut block_weights = Vec::new();
                let mut block_biases: Vec<Option<EdgeId>> = Vec::new();
                let block_start = w_idx;
                for _ in 0..3 {
                    let w_e = g.param(conv_weights[w_idx].clone());
                    block_weights.push(w_e);
                    let b_e = if has_biases {
                        Some(g.param(conv_biases[w_idx].clone()))
                    } else {
                        None
                    };
                    block_biases.push(b_e);
                    w_idx += 1;
                }
                if has_projection {
                    let w_e = g.param(conv_weights[w_idx].clone());
                    block_weights.push(w_e);
                    let b_e = if has_biases {
                        Some(g.param(conv_biases[w_idx].clone()))
                    } else {
                        None
                    };
                    block_biases.push(b_e);
                    w_idx += 1;
                }

                h = bottleneck(g, h, stride, has_projection, &block_weights, &block_biases, sf_log);
                c_in = c_out;

                g.layer_boundaries.push(h);
            }
            if w_idx >= num_layers { break; }
        }

        // ================================================================
        // Head: Global AvgPool → FC → (no softmax/argmax in ZK)
        // ================================================================
        // Current shape: [c_in, spatial, spatial]
        // Global average pooling over axes 1, 2
        h = g.reduce_mean(h, &[1, 2]);
        // Now shape: [c_in]

        // FC: [c_in] × [c_in, num_classes] → [num_classes]
        let fc_w = g.param(fc_weight);
        h = g.einsum("i,ij->j".to_string(), vec![h, fc_w], false)[0];
        if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }

        if let Some(bias) = fc_bias {
            let fc_b = g.param(bias);
            h = g.add(h, fc_b)[0];
        }

        vec![h]
    }
}

/// Compute the output spatial size after running ResNet stem + N stages.
pub fn resnet50_output_shape(num_layers: usize) -> (usize, usize) {
    let configs = resnet50_conv_configs();
    let actual_layers = num_layers.min(configs.len());

    if actual_layers == 0 {
        return (3, 224);
    }

    // After stem: 64, 56
    let mut c_out = 64;
    let mut spatial = 56;

    if actual_layers == 1 {
        return (c_out, spatial);
    }

    let mut w_idx = 1; // skip stem
    for &(num_blocks, _c_mid, c_out_stage, stride_first) in &RESNET50_STAGES {
        for block_idx in 0..num_blocks {
            let has_projection = block_idx == 0 && (c_out != c_out_stage || stride_first > 1);
            let num_convs = if has_projection { 4 } else { 3 };
            if w_idx + num_convs > actual_layers { return (c_out, spatial); }

            if block_idx == 0 && stride_first > 1 {
                spatial /= stride_first;
            }
            c_out = c_out_stage;
            w_idx += num_convs;
        }
        if w_idx >= actual_layers { break; }
    }

    (c_out, spatial)
}
