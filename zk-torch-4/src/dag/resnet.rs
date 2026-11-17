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
    if let Some(b) = biases.get(0).copied().flatten() { let c = g.init_values[weights[0]].as_ref().unwrap().shape[0]; h = g.add_channel_bias_row_major(h, b, c, 2); }
    h = g.relu(h);

    // Conv 3×3 (c_mid → c_mid), with padding=1, stride=stride
    h = g.pad(h, 1, 1);
    h = if stride > 1 {
        g.conv2d_strided(h, weights[1], (3, 3), (stride, stride))[0]
    } else {
        g.conv2d(h, weights[1], (3, 3))[0]
    };
    if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
    if let Some(b) = biases.get(1).copied().flatten() { let c = g.init_values[weights[1]].as_ref().unwrap().shape[0]; h = g.add_channel_bias_row_major(h, b, c, 2); }
    h = g.relu(h);

    // Conv 1×1 (c_mid → c_out), stride=1
    h = g.conv2d(h, weights[2], (1, 1))[0];
    if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
    if let Some(b) = biases.get(2).copied().flatten() { let c = g.init_values[weights[2]].as_ref().unwrap().shape[0]; h = g.add_channel_bias_row_major(h, b, c, 2); }

    // Shortcut
    let shortcut = if has_projection {
        let mut s = if stride > 1 {
            g.conv2d_strided(x, weights[3], (1, 1), (stride, stride))[0]
        } else {
            g.conv2d(x, weights[3], (1, 1))[0]
        };
        if sf_log > 0 { s = g.scale(s, 2 * sf_log, sf_log)[0]; }
        if let Some(b) = biases.get(3).copied().flatten() { let c = g.init_values[weights[3]].as_ref().unwrap().shape[0]; s = g.add_channel_bias_row_major(s, b, c, 2); }
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
            let input_shape = g.init_values[inp].as_ref().unwrap().shape.clone();
            let batch = match input_shape.as_slice() {
                [3, _, _] => 1,
                [lead, _, _] => {
                    assert!(*lead > 0 && *lead % 4 == 0, "invalid folded RGB input");
                    *lead / 4 // Every represented folded batch is logical.
                }
                [batch, 3, _, _] => { assert!(*batch > 0); *batch }
                _ => panic!("ResNet input must be [3,H,W], [B*4,H,W], or [B,3,H,W]"),
            };
            let mut h = inp;
            if input_shape.len() == 4 {
                // Keep logical B in this closure before per-channel blocks
                // fold it into padded channels. Exclude poisoned B/C/spatial
                // padding in the registered graph, not in the input loader.
                let reversed = input_shape.iter().rev().copied().collect::<Vec<_>>();
                let view = g.change_shape(h, reversed.clone());
                let masked = g.mask(view, reversed);
                h = g.change_shape(masked, input_shape);
            }
            // Stem
            h = g.pad(h, 3, 3);
            h = g.conv2d_strided(h, sw, (7, 7), (2, 2))[0];
            if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
            if let Some(b) = sb { let c = g.init_values[sw].as_ref().unwrap().shape[0]; h = g.add_channel_bias_row_major(h, b, c, 2); }
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
            // CNN raw buffers are row-major. Reinterpret [foldedC,H,W]
            // as [W,H,foldedC] before column-major Einsum reduction.
            // Result keeps c in low bits and b in high bits for the FC view.
            h = classifier_head(g, h, fc_w, fc_b, batch, sf_log);
            outs.push(h);
        }
        outs
    }
}

/// GAP + column-major FC. `batch` is the public logical batch count retained
/// by the caller, not a count inferred from a padded intermediate. This helper
/// repairs head semantics; the Morpheus Conv2D PIOP still needs batch support.
pub(crate) fn classifier_head(
    g: &mut DagBuilder, x: EdgeId, weight: EdgeId, bias: Option<EdgeId>,
    batch: usize, sf_log: usize,
) -> EdgeId {
    assert!(batch > 0);
    let ws = g.init_values[weight].as_ref().unwrap().shape.clone();
    assert_eq!(ws.len(), 2, "ResNet FC weight must be column-major [in,out]");
    let (features, classes) = (ws[0], ws[1]);
    assert!(features > 0 && classes > 0);
    let xs = &g.init_values[x].as_ref().unwrap().shape;
    match xs.as_slice() {
        [b, c, _, _] => { assert_eq!(*b, batch); assert_eq!(*c, features); }
        [c, _, _] if batch == 1 => assert_eq!(*c, features),
        [lead, _, _] => assert!(*lead == batch * features.next_power_of_two()
            || *lead == batch.next_power_of_two() * features.next_power_of_two(),
            "folded GAP channels do not match the declared logical batch/features"),
        _ => panic!("GAP expects row-major [C,H,W] or [B,C,H,W]"),
    }
    let h = global_mean_row_major(g, x);
    let n = crate::util::arith::get_n(&g.init_values[h].as_ref().unwrap().shape);
    assert_eq!(1usize << n, features.next_power_of_two() * batch.next_power_of_two(),
        "GAP feature/batch extent does not match FC weight and logical batch");
    let shape = if batch == 1 { vec![features] } else { vec![features, batch] };
    let h = g.change_shape(h, shape.clone());
    let h = g.mask(h, shape);
    let equation = if batch == 1 { "i,ij->j" } else { "ib,ij->jb" };
    let mut out = g.einsum(equation.into(), vec![h, weight], false)[0];
    if sf_log > 0 { out = g.scale(out, 2 * sf_log, sf_log)[0]; }
    if let Some(bias) = bias {
        let bs = &g.init_values[bias].as_ref().unwrap().shape;
        assert!(bs == &vec![classes] || bs == &vec![classes, 1],
            "FC bias must contain exactly one class vector");
        let bias_shape = if batch == 1 { vec![classes] } else { vec![classes, 1] };
        let bias = g.change_shape(bias, bias_shape);
        out = g.add(out, bias)[0];
    }
    // Bias broadcasts also fill padded batches; bind those outputs to zero.
    let shape = if batch == 1 { vec![classes] } else { vec![classes, batch] };
    g.mask(out, shape)
}

/// CNN storage is row-major (W is fastest), whereas Einsum is column-major.
/// Reinterpret the SAME raw buffer as [W,H,C,(B)] before spatial reduction.
/// This is not a transpose and requires no private layout-dependent choice.
pub(crate) fn global_mean_row_major(g:&mut DagBuilder,x:EdgeId)->EdgeId {
    let shape=g.init_values[x].as_ref().unwrap().shape.clone();
    assert!(shape.len()==3 || shape.len()==4);
    let reversed=shape.iter().rev().copied().collect::<Vec<_>>();
    let view=g.change_shape(x,reversed);
    let mean=g.reduce_mean(view,&[0,1]);
    if shape.len()==4 {
        let size=shape[0].next_power_of_two()*shape[1].next_power_of_two();
        g.change_shape(mean,vec![size])
    } else {mean}
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
