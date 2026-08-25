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
    let sf_log = conv_weights.first().map(|w| w.sf).unwrap_or(0)
        .max(conv_biases.first().map(|b| b.sf).unwrap_or(0));

    move |g, inputs| {
        assert!(!inputs.is_empty(), "ResNet expects >=1 input (batch)");
        let has_biases = !conv_biases.is_empty();

        // ============================================================
        // Phase 1: create the SHARED weight param edges ONCE. Each
        // batch element reuses these edges, so a batch of B images
        // commits the weights exactly once (and, under streaming, those
        // Constant edges are deferred + amortized across proofs).
        // ============================================================
        struct BlockParams { w: Vec<EdgeId>, b: Vec<Option<EdgeId>>, stride: usize, proj: bool }
        let mut stem: Option<(EdgeId, Option<EdgeId>)> = None;
        let mut blocks: Vec<BlockParams> = Vec::new();
        {
            let mut w_idx = 0;
            if w_idx < num_layers {
                let sw = g.param(conv_weights[w_idx].clone());
                let sb = if has_biases { Some(g.param(conv_biases[w_idx].clone())) } else { None };
                stem = Some((sw, sb));
                w_idx += 1;
                let mut c_in = 64usize;
                'stages: for &(num_blocks, _c_mid, c_out, stride_first) in &RESNET50_STAGES {
                    for block_idx in 0..num_blocks {
                        if w_idx >= num_layers { break 'stages; }
                        let stride = if block_idx == 0 { stride_first } else { 1 };
                        let proj = block_idx == 0 && (c_in != c_out || stride_first > 1);
                        let need = if proj { 4 } else { 3 };
                        if w_idx + need > conv_weights.len() { break 'stages; }
                        let mut wv = Vec::new();
                        let mut bv: Vec<Option<EdgeId>> = Vec::new();
                        for _ in 0..need {
                            wv.push(g.param(conv_weights[w_idx].clone()));
                            bv.push(if has_biases { Some(g.param(conv_biases[w_idx].clone())) } else { None });
                            w_idx += 1;
                        }
                        blocks.push(BlockParams { w: wv, b: bv, stride, proj });
                        c_in = c_out;
                    }
                }
            }
        }
        let fc_w = g.param(fc_weight);
        let fc_b = fc_bias.map(|b| g.param(b));

        // ============================================================
        // Phase 2: build the body once per batch element (shared edges).
        // ============================================================
        let mut outs = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            let (sw, sb) = match stem { Some(s) => s, None => { outs.push(inp); continue; } };
            // A BATCHED entry is one folded tensor [b_pad*c_pad, H, W]; it
            // still builds one chain. Conv binds the batch variables, and every
            // other op in the body is per-channel, for which a batch is simply
            // more channels. RGB pads to 4 channels.
            let lead = g.init_values[inp].as_ref().unwrap().shape[0];
            let batch = (lead / 3usize.next_power_of_two()).max(1);
            let mut h = inp;
            // Stem
            h = g.pad(h, 3, 3);
            h = g.conv2d_strided(h, sw, (7, 7), (2, 2))[0];
            if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
            if let Some(b) = sb { h = g.add(h, b)[0]; }
            h = g.relu(h);
            h = g.pad(h, 1, 1);
            h = g.maxpool_general(h, 3, 3, 2, 2);
            g.layer_boundaries.push(h);
            // Bottleneck blocks
            for bp in &blocks {
                h = bottleneck(g, h, bp.stride, bp.proj, &bp.w, &bp.b, sf_log);
                g.layer_boundaries.push(h);
            }
            // Head: GAP → FC.
            //
            // reduce_mean over [1, 2] is H and W, which is correct under the
            // folded layout: axis 0 is the combined b*c, so this is global
            // average pooling per image per channel and needs no axis shift.
            // It leaves [b_pad*c_pad], with c in the LOW bits and b in the
            // HIGH bits -- so the FC head reshapes to [c_pad, B], matching
            // einsum's convention that the first shape dimension is lowest.
            h = g.reduce_mean(h, &[1, 2]);
            if batch > 1 {
                let feat = g.init_values[h].as_ref().unwrap().shape[0] / batch;
                h = g.change_shape(h, vec![feat, batch]);
                h = g.einsum("ib,ij->jb".to_string(), vec![h, fc_w], false)[0];
                if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
                if let Some(b) = fc_b { h = g.add(h, b)[0]; }
            } else {
                h = g.einsum("i,ij->j".to_string(), vec![h, fc_w], false)[0];
                if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
                if let Some(b) = fc_b { h = g.add(h, b)[0]; }
            }
            outs.push(h);
        }
        outs
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
