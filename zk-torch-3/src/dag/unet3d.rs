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
        assert_eq!(inputs.len(), 1, "UNet expects 1 input");
        let mut h = inputs[0];

        let mut w_idx = 0;  // conv weight index (encoder convs only)
        let mut in_idx = 0; // instancenorm param index
        let mut ct_idx = 0; // conv transpose weight index

        let actual_levels = num_levels.min(6);

        // ================================================================
        // Encoder
        // ================================================================
        let mut skip_connections: Vec<EdgeId> = Vec::new();

        for level in 0..actual_levels {
            let stride = ENCODER_LEVELS[level].2;

            // Conv 1: possibly strided
            h = g.pad3d(h, 1, 1, 1);
            let w_e = g.param(conv_weights[w_idx].clone());
            h = if stride > 1 {
                g.conv3d_strided(h, w_e, (3, 3, 3), (stride, stride, stride))[0]
            } else {
                g.conv3d(h, w_e, (3, 3, 3))[0]
            };
            h = g.mask_channels(h, 4); // zero-pad non-power-of-2 channels
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = g.param(in_gammas[in_idx].clone());
            let beta_e = g.param(in_betas[in_idx].clone());
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            // Conv 2: always stride 1
            h = g.pad3d(h, 1, 1, 1);
            let w_e = g.param(conv_weights[w_idx].clone());
            h = g.conv3d(h, w_e, (3, 3, 3))[0];
            h = g.mask_channels(h, 4);
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = g.param(in_gammas[in_idx].clone());
            let beta_e = g.param(in_betas[in_idx].clone());
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
            let ct_w = g.param(conv_transpose_weights[ct_idx].clone());
            h = g.conv_transpose3d(h, ct_w, (2, 2, 2), (2, 2, 2))[0];
            h = g.mask_channels(h, 4);
            ct_idx += 1;

            // Concat with skip connection
            h = g.concat(h, skip);
            h = g.mask_channels(h, 4);

            // Conv 1: stride 1 with pad
            h = g.pad3d(h, 1, 1, 1);
            let w_e = g.param(conv_weights[w_idx].clone());
            h = g.conv3d(h, w_e, (3, 3, 3))[0];
            h = g.mask_channels(h, 4);
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = g.param(in_gammas[in_idx].clone());
            let beta_e = g.param(in_betas[in_idx].clone());
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            // Conv 2: stride 1 with pad
            h = g.pad3d(h, 1, 1, 1);
            let w_e = g.param(conv_weights[w_idx].clone());
            h = g.conv3d(h, w_e, (3, 3, 3))[0];
            h = g.mask_channels(h, 4);
            w_idx += 1;

            // InstanceNorm + ReLU
            let gamma_e = g.param(in_gammas[in_idx].clone());
            let beta_e = g.param(in_betas[in_idx].clone());
            h = g.instancenorm3d(h, gamma_e, beta_e, eps);
            in_idx += 1;
            h = g.relu(h);

            g.layer_boundaries.push(h);
        }

        // ================================================================
        // Output: 1×1×1 conv (32→3) + bias
        // ================================================================
        if num_decoder_levels > 0 || actual_levels >= 1 {
            let w_e = g.param(conv_weights[w_idx].clone());
            h = g.conv3d(h, w_e, (1, 1, 1))[0];

            if let Some(bias) = output_bias {
                let bias_e = g.param(bias);
                h = g.add(h, bias_e)[0];
            }
        }

        vec![h]
    }
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
