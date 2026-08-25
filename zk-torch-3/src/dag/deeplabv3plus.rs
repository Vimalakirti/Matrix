use crate::dag::{DagBuilder, EdgeId, Witness};

// ============================================================================
// DeepLabV3+ Model Builder
// ============================================================================

/// Conv + (fused BN) + ReLU helper.
fn conv_bn_relu(
    g: &mut DagBuilder,
    x: EdgeId,
    w: EdgeId,
    bias: EdgeId,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    dilation: (usize, usize),
) -> EdgeId {
    let h = if pad.0 > 0 || pad.1 > 0 {
        g.pad(x, pad.0, pad.1)
    } else {
        x
    };
    let h = if dilation.0 > 1 || dilation.1 > 1 {
        g.conv2d_dilated(h, w, kernel, stride, dilation)[0]
    } else if stride.0 > 1 || stride.1 > 1 {
        g.conv2d_strided(h, w, kernel, stride)[0]
    } else {
        g.conv2d(h, w, kernel)[0]
    };
    let h = g.add(h, bias)[0];
    g.relu(h)
}

/// Conv + (fused BN) without activation.
fn conv_bn(
    g: &mut DagBuilder,
    x: EdgeId,
    w: EdgeId,
    bias: EdgeId,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    dilation: (usize, usize),
) -> EdgeId {
    let h = if pad.0 > 0 || pad.1 > 0 {
        g.pad(x, pad.0, pad.1)
    } else {
        x
    };
    let h = if dilation.0 > 1 || dilation.1 > 1 {
        g.conv2d_dilated(h, w, kernel, stride, dilation)[0]
    } else if stride.0 > 1 || stride.1 > 1 {
        g.conv2d_strided(h, w, kernel, stride)[0]
    } else {
        g.conv2d(h, w, kernel)[0]
    };
    g.add(h, bias)[0]
}

/// ResNet-101 bottleneck block with dilation support.
/// Conv 1×1 → ReLU → Conv 3×3(stride, dilation) → ReLU → Conv 1×1 → Add(shortcut) → ReLU
fn bottleneck_dilated(
    g: &mut DagBuilder,
    x: EdgeId,
    stride: usize,
    dilation: usize,
    has_projection: bool,
    weights: &[(EdgeId, EdgeId)], // (conv_w, bias) pairs
) -> EdgeId {
    // Conv 1×1 (c_in → c_mid), stride=1
    let mut h = conv_bn_relu(g, x, weights[0].0, weights[0].1, (1, 1), (1, 1), (0, 0), (1, 1));

    // Conv 3×3 (c_mid → c_mid) with padding=dilation, stride=stride, dilation=dilation
    let pad_size = dilation;
    h = conv_bn_relu(g, h, weights[1].0, weights[1].1, (3, 3), (stride, stride), (pad_size, pad_size), (dilation, dilation));

    // Conv 1×1 (c_mid → c_out), stride=1
    h = conv_bn(g, h, weights[2].0, weights[2].1, (1, 1), (1, 1), (0, 0), (1, 1));

    // Shortcut
    let shortcut = if has_projection {
        if stride > 1 {
            conv_bn(g, x, weights[3].0, weights[3].1, (1, 1), (stride, stride), (0, 0), (1, 1))
        } else {
            conv_bn(g, x, weights[3].0, weights[3].1, (1, 1), (1, 1), (0, 0), (1, 1))
        }
    } else {
        x
    };

    // Add + ReLU
    h = g.add(h, shortcut)[0];
    g.relu(h)
}

/// Depthwise separable conv: DWConv 3x3 + Conv 1x1, both with BN + ReLU.
fn depthwise_separable_conv(
    g: &mut DagBuilder,
    x: EdgeId,
    dw_w: EdgeId,
    dw_bias: EdgeId,
    pw_w: EdgeId,
    pw_bias: EdgeId,
) -> EdgeId {
    let h = g.pad(x, 1, 1);
    let h = g.depthwise_conv2d_strided(h, dw_w, (3, 3), (1, 1))[0];
    let h = g.add(h, dw_bias)[0];
    let h = g.relu(h);
    let h = g.conv2d(h, pw_w, (1, 1))[0];
    let h = g.add(h, pw_bias)[0];
    g.relu(h)
}

/// ResNet-101 stage configurations for output_stride=8.
/// (num_blocks, c_mid, c_out, stride, dilation)
const RESNET101_STAGES: [(usize, usize, usize, usize, usize); 4] = [
    (3, 64, 256, 1, 1),      // stage1: /4 (after stem /4)
    (4, 128, 512, 2, 1),     // stage2: /8
    (23, 256, 1024, 1, 2),   // stage3: /8 (dilation replaces stride)
    (3, 512, 2048, 1, 4),    // stage4: /8 (more dilation)
];

/// Conv layer configs for DeepLabV3+ backbone: (c_in, c_out, kH, kW, is_depthwise).
/// Returns list for weight generation.
pub fn deeplabv3plus_conv_configs(num_layers: usize) -> Vec<(usize, usize, usize, usize, bool)> {
    let mut configs = Vec::new();

    // Stem: 3 convs (deep stem V1c)
    // conv1: 3→64, 3×3, stride 2
    configs.push((3, 64, 3, 3, false));
    // conv2: 64→64, 3×3
    configs.push((64, 64, 3, 3, false));
    // conv3: 64→64, 3×3
    configs.push((64, 64, 3, 3, false));

    // Stages
    let mut c_in = 64;
    let mut skip_channels = 64; // after stage 0, becomes c_out (256)
    let mut conv_count = 3; // stem convs
    let mut _stage_done = 0;

    for (si, &(num_blocks, c_mid, c_out, stride_first, _dilation)) in RESNET101_STAGES.iter().enumerate() {
        for block_idx in 0..num_blocks {
            if conv_count >= num_layers { break; }
            let has_projection = block_idx == 0 && (c_in != c_out || stride_first > 1);

            configs.push((c_in, c_mid, 1, 1, false));    // 1×1
            configs.push((c_mid, c_mid, 3, 3, false));    // 3×3
            configs.push((c_mid, c_out, 1, 1, false));    // 1×1
            conv_count += 3;

            if has_projection {
                configs.push((c_in, c_out, 1, 1, false)); // projection
                conv_count += 1;
            }

            c_in = c_out;
        }
        if si == 0 { skip_channels = c_in; }
        _stage_done = si + 1;
        if conv_count >= num_layers { break; }
    }

    // ASPP: 5 branches + fusion conv
    configs.push((c_in, 256, 1, 1, false));  // Branch 0: global avg pool → 1×1
    configs.push((c_in, 256, 1, 1, false));  // Branch 1: 1×1
    configs.push((c_in, 256, 3, 3, false));  // Branch 2: 3×3 dilated (dil=12)
    configs.push((c_in, 256, 3, 3, false));  // Branch 3: 3×3 dilated (dil=24)
    configs.push((c_in, 256, 3, 3, false));  // Branch 4: 3×3 dilated (dil=36)
    // Fusion: 1×1 conv. multi_concat of 5×256 branches → 2048 channels (power-of-2 padding)
    let fusion_c_in = 2048; // multi_concat tree: 256*2=512, 512*2=1024, 1024+256→pad→2048
    configs.push((fusion_c_in, 256, 1, 1, false));

    // Decoder
    // Skip 1×1 (skip_channels→48)
    configs.push((skip_channels, 48, 1, 1, false));
    // DWSepConv 1: DW 3×3 + PW 1×1. general_concat(256, 48) → 512 channels (power-of-2 padding)
    let dec_concat_channels = 512;
    configs.push((dec_concat_channels, 1, 3, 3, true));  // DW 3×3
    configs.push((dec_concat_channels, 256, 1, 1, false)); // PW 1×1
    // DWSepConv 2: DW 3×3 (256 channels) + PW 1×1 (256→256)
    configs.push((256, 1, 3, 3, true));
    configs.push((256, 256, 1, 1, false));
    // Final 1×1 (256→num_classes) — added by caller

    configs
}

/// Build a DeepLabV3+ model graph.
///
/// `all_weights`: (conv_weight, bias) pairs for each conv in order.
/// `num_classes`: number of segmentation classes.
/// `num_layers`: max bottleneck conv layers (for partial runs).
///
/// Returns a closure that builds the graph.
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

pub fn deeplabv3plus(
    all_weights: Vec<(Witness, Witness)>,
    _num_classes: usize,
    num_layers: usize,
    _input_h: usize,
    _input_w: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, inputs| {
        assert_eq!(inputs.len(), 1, "DeepLabV3+ expects 1 input");
        let mut wi = 0usize;
        let mut conv_count = 0usize;

        let mut h = inputs[0];

        // ================================================================
        // Deep Stem (V1c): 3 × Conv2D 3×3 + ReLU, first with stride 2
        // ================================================================
        let stem_w = load_weights(g, &all_weights, &mut wi, 3);
        h = g.pad(h, 1, 1);
        h = g.conv2d_strided(h, stem_w[0].0, (3, 3), (2, 2))[0];
        h = g.add(h, stem_w[0].1)[0];
        h = g.relu(h);
        h = g.pad(h, 1, 1);
        h = g.conv2d(h, stem_w[1].0, (3, 3))[0];
        h = g.add(h, stem_w[1].1)[0];
        h = g.relu(h);
        h = g.pad(h, 1, 1);
        h = g.conv2d(h, stem_w[2].0, (3, 3))[0];
        h = g.add(h, stem_w[2].1)[0];
        h = g.relu(h);
        conv_count += 3;

        // MaxPool 3×3, stride 2, pad 1
        h = g.pad(h, 1, 1);
        h = g.maxpool_general(h, 3, 3, 2, 2);

        g.layer_boundaries.push(h);

        // ================================================================
        // Stages 1-4: bottleneck blocks with dilation
        // ================================================================
        let mut c_in = 64usize;
        let mut skip_output = None;

        for (stage_idx, &(num_blocks, _c_mid, c_out, stride_first, dilation)) in RESNET101_STAGES.iter().enumerate() {
            for block_idx in 0..num_blocks {
                if conv_count >= num_layers { break; }
                if wi >= all_weights.len() { break; }

                let stride = if block_idx == 0 { stride_first } else { 1 };
                let has_projection = block_idx == 0 && (c_in != c_out || stride_first > 1);
                let block_dilation = if block_idx == 0 && stride_first > 1 { 1 } else { dilation };

                let num_convs_needed = if has_projection { 4 } else { 3 };
                if wi + num_convs_needed > all_weights.len() { break; }

                let block_weights = load_weights(g, &all_weights, &mut wi, num_convs_needed);
                h = bottleneck_dilated(g, h, stride, block_dilation, has_projection, &block_weights);
                c_in = c_out;
                conv_count += num_convs_needed;

                g.layer_boundaries.push(h);
            }

            if stage_idx == 0 {
                skip_output = Some(h);
            }

            if conv_count >= num_layers { break; }
            if wi >= all_weights.len() { break; }
        }

        // If we don't have enough weights for ASPP, return early
        if wi + 6 > all_weights.len() { return vec![h]; }

        // ================================================================
        // ASPP Module: 5 parallel branches + fusion
        // ================================================================
        let aspp_input = h;
        let aspp_w = load_weights(g, &all_weights, &mut wi, 6);

        // Branch 0: Global avg pool → 1×1 → ReLU → upsample
        let branch0 = {
            let x_shape = &g.init_values[aspp_input].as_ref().unwrap().shape;
            let spatial_h = x_shape[1];
            let spatial_w = x_shape[2];
            let pooled = g.reduce_mean(aspp_input, &[1, 2]);
            let b0 = g.einsum("i,ij->j".to_string(), vec![pooled, aspp_w[0].0], false)[0];
            let b0 = g.add(b0, aspp_w[0].1)[0];
            let b0 = g.relu(b0);
            let b0 = g.change_shape(b0, vec![256, 1, 1]);
            let mut b0 = b0;
            let mut cur_h = 1usize;
            let mut cur_w = 1usize;
            while cur_h < spatial_h || cur_w < spatial_w {
                b0 = g.upsample_nearest_2x(b0);
                cur_h *= 2;
                cur_w *= 2;
            }
            if cur_h > spatial_h || cur_w > spatial_w {
                b0 = g.change_shape(b0, vec![256, spatial_h, spatial_w]);
            }
            b0
        };

        let branch1 = conv_bn_relu(g, aspp_input, aspp_w[1].0, aspp_w[1].1, (1, 1), (1, 1), (0, 0), (1, 1));
        let branch2 = conv_bn_relu(g, aspp_input, aspp_w[2].0, aspp_w[2].1, (3, 3), (1, 1), (12, 12), (12, 12));
        let branch3 = conv_bn_relu(g, aspp_input, aspp_w[3].0, aspp_w[3].1, (3, 3), (1, 1), (24, 24), (24, 24));
        let branch4 = conv_bn_relu(g, aspp_input, aspp_w[4].0, aspp_w[4].1, (3, 3), (1, 1), (36, 36), (36, 36));

        let aspp_cat = g.multi_concat(vec![branch0, branch1, branch2, branch3, branch4]);
        let aspp_out = conv_bn_relu(g, aspp_cat, aspp_w[5].0, aspp_w[5].1, (1, 1), (1, 1), (0, 0), (1, 1));

        // If we don't have decoder weights, return ASPP output
        if wi + 5 > all_weights.len() { return vec![aspp_out]; }

        // ================================================================
        // Decoder
        // ================================================================
        let decoder_w = load_weights(g, &all_weights, &mut wi, 5);

        let aspp_up = g.upsample_nearest_2x(aspp_out);

        let skip = skip_output.unwrap();
        let skip_proj = conv_bn_relu(g, skip, decoder_w[0].0, decoder_w[0].1, (1, 1), (1, 1), (0, 0), (1, 1));

        let dec_cat = g.general_concat(aspp_up, skip_proj);
        let dec = depthwise_separable_conv(g, dec_cat, decoder_w[1].0, decoder_w[1].1, decoder_w[2].0, decoder_w[2].1);
        let dec = depthwise_separable_conv(g, dec, decoder_w[3].0, decoder_w[3].1, decoder_w[4].0, decoder_w[4].1);

        // Final classifier
        if wi >= all_weights.len() { return vec![dec]; }
        let cls_w = load_weights(g, &all_weights, &mut wi, 1);
        let logits = g.conv2d(dec, cls_w[0].0, (1, 1))[0];
        let logits = g.add(logits, cls_w[0].1)[0];

        vec![logits]
    }
}

/// Total number of conv layers in the DeepLabV3+ config (backbone + ASPP + decoder + classifier).
pub fn deeplabv3plus_num_convs(num_layers: usize, _num_classes: usize) -> usize {
    deeplabv3plus_conv_configs(num_layers).len() + 1 // +1 for final classifier
}

/// Compute backbone output channel count given number of bottleneck layers processed.
pub fn deeplabv3plus_backbone_output_channels(num_layers: usize) -> usize {
    let mut c_in = 64;
    let mut conv_count = 3; // stem convs

    for &(num_blocks, _c_mid, c_out, _stride, _dilation) in &RESNET101_STAGES {
        for block_idx in 0..num_blocks {
            if conv_count >= num_layers { return c_in; }
            let has_projection = block_idx == 0 && (c_in != c_out);
            conv_count += if has_projection { 4 } else { 3 };
            c_in = c_out;
        }
    }
    c_in
}
