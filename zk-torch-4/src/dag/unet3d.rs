use crate::dag::{DagBuilder, EdgeId, Witness};

/// 3D UNet architecture from the ONNX model (volumetric segmentation).
/// Input: [1, D, H, W] (single channel). Output: [3, D, H, W].
///
/// Encoder (6 levels):
///   Level 0: Conv(1→32, k=3, s=1, p=1) + IN + ReLU, Conv(32→32, k=3, s=1, p=1) + IN + ReLU
///   Level 1: Conv(32→64, k=3, s=2, p=1) + IN + ReLU, Conv(64→64, k=3, s=1, p=1) + IN + ReLU
///   Level 2: Conv(64→128, k=3, s=2, p=1) + IN + ReLU, Conv(128→128, k=3, s=1, p=1) + IN + ReLU
///   Level 3: Conv(128→256, k=3, s=2, p=1) + IN + ReLU, Conv(256→256, k=3, s=1, p=1) + IN + ReLU
///   Level 4: Conv(256→320, k=3, s=2, p=1) + IN + ReLU, Conv(320→320, k=3, s=1, p=1) + IN + ReLU
///   Level 5 (bottleneck): Conv(320→320, k=3, s=2, p=1) + IN + ReLU, Conv(320→320, k=3, s=1, p=1) + IN + ReLU
///
/// Decoder (5 levels):
///   Level 4: ConvT(320→320, k=2, s=2) + Concat(320+320→640), Conv(640→320, k=3, s=1, p=1) + IN + ReLU, Conv(320→320, k=3, s=1, p=1) + IN + ReLU
///   Level 3: ConvT(320→256, k=2, s=2) + Concat(256+256→512), Conv(512→256, k=3, s=1, p=1) + IN + ReLU, Conv(256→256, k=3, s=1, p=1) + IN + ReLU
///   Level 2: ConvT(256→128, k=2, s=2) + Concat(128+128→256), Conv(256→128, k=3, s=1, p=1) + IN + ReLU, Conv(128→128, k=3, s=1, p=1) + IN + ReLU
///   Level 1: ConvT(128→64, k=2, s=2) + Concat(64+64→128), Conv(128→64, k=3, s=1, p=1) + IN + ReLU, Conv(64→64, k=3, s=1, p=1) + IN + ReLU
///   Level 0: ConvT(64→32, k=2, s=2) + Concat(32+32→64), Conv(64→32, k=3, s=1, p=1) + IN + ReLU, Conv(32→32, k=3, s=1, p=1) + IN + ReLU
///
/// Output: Conv(32→3, k=1, s=1) + bias

/// Encoder level configurations: (c_in, c_out, stride_first_conv)
pub const ENCODER_LEVELS: [(usize, usize, usize); 6] = [
    (1, 32, 1),     // Level 0
    (32, 64, 2),    // Level 1
    (64, 128, 2),   // Level 2
    (128, 256, 2),  // Level 3
    (256, 256, 2),  // Level 4 (use pow2 channels for MLE padding correctness)
    (256, 256, 2),  // Level 5 (bottleneck)
];

/// Decoder level configurations: (c_upsample_in, c_upsample_out, c_conv_in (after concat))
pub const DECODER_LEVELS: [(usize, usize, usize); 5] = [
    (256, 256, 512),  // Level 4: upsample 256→256, concat with enc4 (256+256=512)
    (256, 256, 512),  // Level 3: upsample 256→256, concat with enc3 (256+256=512)
    (256, 128, 256),  // Level 2: upsample 256→128, concat with enc2 (128+128=256)
    (128, 64, 128),   // Level 1: upsample 128→64, concat with enc1 (64+64=128)
    (64, 32, 64),     // Level 0: upsample 64→32, concat with enc0 (32+32=64)
];

/// All conv layer configurations for the 3D UNet.
/// Returns (c_in, c_out, kD, kH, kW) for each conv in order.
pub fn unet3d_conv_configs() -> Vec<(usize, usize, usize, usize, usize)> {
    let mut configs = Vec::new();

    // Encoder: 6 levels × 2 convs each = 12 convs
    for &(c_in, c_out, _stride) in &ENCODER_LEVELS {
        configs.push((c_in, c_out, 3, 3, 3));   // first conv (may be strided)
        configs.push((c_out, c_out, 3, 3, 3));   // second conv
    }

    // Decoder: 5 levels × (1 convT + 2 convs) = 15 ops
    for &(c_up_in, c_up_out, c_conv_in) in &DECODER_LEVELS {
        configs.push((c_up_in, c_up_out, 2, 2, 2));   // ConvTranspose3D
        configs.push((c_conv_in, c_up_out, 3, 3, 3));  // first conv after concat
        configs.push((c_up_out, c_up_out, 3, 3, 3));   // second conv
    }

    // Output: 1×1×1 conv (32→3)
    configs.push((32, 3, 1, 1, 1));

    configs
}

/// Total number of conv/convT layers in full 3D UNet.
pub fn unet3d_num_convs() -> usize {
    unet3d_conv_configs().len()
}

/// InstanceNorm parameter configs: (channels,) for each IN layer.
/// 2 per encoder level (12 total) + 2 per decoder level (10 total) = 22.
pub fn unet3d_instancenorm_configs() -> Vec<usize> {
    let mut configs = Vec::new();

    for &(_c_in, c_out, _stride) in &ENCODER_LEVELS {
        configs.push(c_out);
        configs.push(c_out);
    }

    for &(_c_up_in, c_up_out, _c_conv_in) in &DECODER_LEVELS {
        // Note: IN after each of the 2 convs in decoder, NOT after convT
        configs.push(c_up_out);
        configs.push(c_up_out);
    }

    configs
}

/// Build a 3D UNet model graph.
///
/// `conv_weights`: weight witnesses for each conv/convT layer in order.
/// `in_gammas`, `in_betas`: InstanceNorm affine parameters.
/// `output_bias`: optional bias for the final 1×1×1 conv.
/// `num_levels`: number of encoder levels to include (1-6, default 6).
/// `eps`: epsilon for InstanceNorm.
fn rescale_conv3d(g: &mut DagBuilder, h: EdgeId, w: EdgeId) -> EdgeId {
    let sf_log = g.init_values[w].as_ref().unwrap().sf;
    if sf_log > 0 { g.scale(h, 2 * sf_log, sf_log)[0] } else { h }
}

pub fn unet3d(
    conv_weights: Vec<Witness>,
    conv_transpose_weights: Vec<Witness>,
    in_gammas: Vec<Witness>,
    in_betas: Vec<Witness>,
    output_bias: Option<Witness>,
    num_levels: usize,
    eps: f64,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, inputs| {
        assert!(!inputs.is_empty(), "UNet expects >=1 input (batch)");
        let actual_levels = num_levels.min(6);

        // Early geometry check: the encoder halves the spatial dims
        // `actual_levels - 1` times and the decoder ConvTranspose (k=2, s=2)
        // doubles them back; every halving must be exact or a decoder
        // upsample and its encoder skip connection disagree at Concat.
        for &inp in inputs {
            let shape = g.init_values[inp].as_ref().unwrap().shape.clone();
            assert_eq!(shape.len(), 4, "UNet3D input must be [C, D, H, W], got {shape:?}");
            let (d, h, w) = (shape[1], shape[2], shape[3]);
            let max_levels = unet3d_max_levels(d, h, w);
            assert!(
                actual_levels <= max_levels,
                "UNet3D: input spatial [{d},{h},{w}] does not support {actual_levels} encoder \
                 levels (each dim must be divisible by 2^(levels-1)); max depth for this \
                 input is {max_levels}. Reduce NUM_LAYERS or enlarge the input volume."
            );
        }

        // ---- Phase 1: create the SHARED param edges ONCE. Each batch volume
        // reuses these, committing weights/IN-params exactly once (and,
        // streaming, deferring + amortizing those Constant edges). ----
        let conv_w: Vec<EdgeId> = conv_weights.iter().map(|w| g.param(w.clone())).collect();
        let ct_w_edges: Vec<EdgeId> = conv_transpose_weights.iter().map(|w| g.param(w.clone())).collect();
        let gamma_e_all: Vec<EdgeId> = in_gammas.iter().map(|w| g.param(w.clone())).collect();
        let beta_e_all: Vec<EdgeId> = in_betas.iter().map(|w| g.param(w.clone())).collect();
        let out_bias_e: Option<EdgeId> = output_bias.map(|b| g.param(b));

        // ---- Phase 2: build the body once per batch volume (shared edges). ----
        let mut outs = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            let mut h = inp;
            let mut w_idx = 0;  // conv weight index (encoder convs only)
            let mut in_idx = 0; // instancenorm param index
            let mut ct_idx = 0; // conv transpose weight index

        // ================================================================
        // Encoder
        // ================================================================
        let mut skip_connections: Vec<EdgeId> = Vec::new();

        for level in 0..actual_levels {
            let stride = ENCODER_LEVELS[level].2;

            // Conv 1: possibly strided
            h = g.pad3d(h, 1, 1, 1);
            let w_e = conv_w[w_idx];
            h = if stride > 1 {
                g.conv3d_strided(h, w_e, (3, 3, 3), (stride, stride, stride))[0]
            } else {
                g.conv3d(h, w_e, (3, 3, 3))[0]
            };
            h = rescale_conv3d(g, h, w_e);
            h = g.mask_channels(h, 4); // zero-pad non-power-of-2 channels
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = gamma_e_all[in_idx];
            let beta_e = beta_e_all[in_idx];
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            // Conv 2: always stride 1
            h = g.pad3d(h, 1, 1, 1);
            let w_e = conv_w[w_idx];
            h = g.conv3d(h, w_e, (3, 3, 3))[0];
            h = rescale_conv3d(g, h, w_e);
            h = g.mask_channels(h, 4);
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = gamma_e_all[in_idx];
            let beta_e = beta_e_all[in_idx];
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            // Save skip connection (except bottleneck = last level)
            if level < actual_levels - 1 {
                skip_connections.push(h);
            }

            g.layer_boundaries.push(h);
        }

        // ================================================================
        // Decoder
        // ================================================================
        let num_decoder_levels = if actual_levels <= 1 { 0 } else { actual_levels - 1 };

        for dec_level in 0..num_decoder_levels {
            let skip_idx = skip_connections.len() - 1 - dec_level;
            let skip = skip_connections[skip_idx];

            // ConvTranspose3D (upsample)
            let ct_w = ct_w_edges[ct_idx];
            h = g.conv_transpose3d(h, ct_w, (2, 2, 2), (2, 2, 2))[0];
            h = rescale_conv3d(g, h, ct_w);
            h = g.mask_channels(h, 4);
            ct_idx += 1;

            // Concat with skip connection
            h = g.concat(h, skip);
            h = g.mask_channels(h, 4);

            // Conv 1: stride 1 with pad
            h = g.pad3d(h, 1, 1, 1);
            let w_e = conv_w[w_idx];
            h = g.conv3d(h, w_e, (3, 3, 3))[0];
            h = rescale_conv3d(g, h, w_e);
            h = g.mask_channels(h, 4);
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = gamma_e_all[in_idx];
            let beta_e = beta_e_all[in_idx];
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            // Conv 2: stride 1 with pad
            h = g.pad3d(h, 1, 1, 1);
            let w_e = conv_w[w_idx];
            h = g.conv3d(h, w_e, (3, 3, 3))[0];
            h = rescale_conv3d(g, h, w_e);
            h = g.mask_channels(h, 4);
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = gamma_e_all[in_idx];
            let beta_e = beta_e_all[in_idx];
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            g.layer_boundaries.push(h);
        }

        // ================================================================
        // Output: 1×1×1 conv (32→3) + bias
        // ================================================================
        if num_decoder_levels > 0 || actual_levels >= 1 {
            let w_e = conv_w[w_idx];
            h = g.conv3d(h, w_e, (1, 1, 1))[0];
            h = rescale_conv3d(g, h, w_e);

            if let Some(bias_e) = out_bias_e {
                h = g.add(h, bias_e)[0];
            }
        }

            outs.push(h);
        }
        outs
    }
}

/// Maximum encoder depth supported by a given input spatial size.
///
/// Each of the `levels - 1` strided encoder convs (k=3, s=2, p=1) halves the
/// spatial dims, and each decoder ConvTranspose (k=2, s=2) exactly doubles
/// them. For every decoder upsample to match its encoder skip connection,
/// every halving must be exact, i.e. each spatial dim must be divisible by
/// `2^(levels - 1)`. Capped at 6 (the architecture defines 6 levels).
pub fn unet3d_max_levels(input_d: usize, input_h: usize, input_w: usize) -> usize {
    let halvings = |n: usize| if n == 0 { 0 } else { n.trailing_zeros() as usize };
    (1 + halvings(input_d).min(halvings(input_h)).min(halvings(input_w))).min(6)
}

/// Compute the output shape [C, D, H, W] after running the 3D UNet with given number of encoder levels.
/// For num_levels encoder levels, spatial is halved (num_levels-1) times during encoding then restored.
pub fn unet3d_output_shape(num_levels: usize, input_d: usize, input_h: usize, input_w: usize) -> (usize, usize, usize, usize) {
    if num_levels <= 1 {
        // Only encoder level 0 (no stride), output channels = c_out of level 0 = 32
        // Plus 1×1×1 conv: 32→3
        return (3, input_d, input_h, input_w);
    }
    // Full encode+decode: spatial restored to original
    (3, input_d, input_h, input_w)
}
