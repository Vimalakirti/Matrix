use crate::dag::{DagBuilder, EdgeId, Witness};

/// Block configs: (num_convs_per_block, out_channels)
const VGG11_BLOCKS: [(usize, usize); 5] = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)];
const VGG16_BLOCKS: [(usize, usize); 5] = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)];

/// FC hidden dimension (matches original VGG paper).
pub const VGG_FC_HIDDEN: usize = 4096;

/// Generic VGG builder matching the original VGG paper (Simonyan & Zisserman, 2014).
///
/// Architecture: conv blocks (conv + bias + relu, maxpool) → 3 FC layers with bias.
/// Input: [3, 32, 32]. Each conv uses 3×3 kernel, stride=1, pad=1 (same padding).
/// MaxPool is 2×2, stride=2.
/// FC head: FC(flat_dim → 4096) → ReLU → FC(4096 → 4096) → ReLU → FC(4096 → num_classes).
fn vgg_generic(
    blocks: &[(usize, usize)],
    conv_weights: Vec<Witness>,
    conv_biases: Vec<Witness>,
    fc_weights: Vec<Witness>,
    fc_biases: Vec<Witness>,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    let blocks = blocks.to_vec();
    move |g, x| {
        assert_eq!(x.len(), 1, "VGG expects 1 input");
        assert_eq!(conv_weights.len(), conv_biases.len(), "Each conv needs a bias");
        assert_eq!(fc_weights.len(), 3, "VGG paper has 3 FC layers");
        assert_eq!(fc_biases.len(), 3, "VGG paper has 3 FC biases");

        let mut h = x[0];
        let mut w_idx = 0;
        let num_weights = conv_weights.len();

        let mut last_out_channels = 3usize;
        let mut spatial = 32usize;

        for (num_convs, out_channels) in &blocks {
            let mut block_complete = true;
            for _ in 0..*num_convs {
                if w_idx >= num_weights {
                    block_complete = false;
                    break;
                }
                h = g.pad(h, 1, 1);
                let w_e = g.param(conv_weights[w_idx].clone());
                h = g.conv2d(h, w_e, (3, 3))[0];

                // Add conv bias (full-size [C_out, H_out, W_out] parameter)
                let bias_e = g.param(conv_biases[w_idx].clone());
                h = g.add(h, bias_e)[0];

                h = g.relu(h);
                last_out_channels = *out_channels;
                w_idx += 1;

                g.layer_boundaries.push(h);
            }
            if block_complete {
                h = g.maxpool2d(h, 2, 2);
                spatial /= 2;
            }
            if w_idx >= num_weights {
                break;
            }
        }

        let flat_dim = last_out_channels * spatial * spatial;
        h = g.change_shape(h, vec![flat_dim]);

        // FC1: flat_dim → 4096 + bias + ReLU
        let fc1_w = g.param(fc_weights[0].clone());
        h = g.einsum("i,ij->j".to_string(), vec![h, fc1_w], false)[0];
        let fc1_b = g.param(fc_biases[0].clone());
        h = g.add(h, fc1_b)[0];
        h = g.relu(h);

        // FC2: 4096 → 4096 + bias + ReLU
        let fc2_w = g.param(fc_weights[1].clone());
        h = g.einsum("i,ij->j".to_string(), vec![h, fc2_w], false)[0];
        let fc2_b = g.param(fc_biases[1].clone());
        h = g.add(h, fc2_b)[0];
        h = g.relu(h);

        // FC3: 4096 → num_classes + bias (no ReLU on output)
        let fc3_w = g.param(fc_weights[2].clone());
        h = g.einsum("i,ij->j".to_string(), vec![h, fc3_w], false)[0];
        let fc3_b = g.param(fc_biases[2].clone());
        h = g.add(h, fc3_b)[0];

        vec![h]
    }
}

/// VGG-11 (original paper) on CIFAR-10 (8 conv layers + 3 FC layers).
pub fn vgg11(
    conv_weights: Vec<Witness>,
    conv_biases: Vec<Witness>,
    fc_weights: Vec<Witness>,
    fc_biases: Vec<Witness>,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    vgg_generic(&VGG11_BLOCKS, conv_weights, conv_biases, fc_weights, fc_biases)
}

/// VGG-16 (original paper) on CIFAR-10 (13 conv layers + 3 FC layers).
pub fn vgg16(
    conv_weights: Vec<Witness>,
    conv_biases: Vec<Witness>,
    fc_weights: Vec<Witness>,
    fc_biases: Vec<Witness>,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    vgg_generic(&VGG16_BLOCKS, conv_weights, conv_biases, fc_weights, fc_biases)
}

/// Compute the output shape (channels, spatial) after running N conv layers through given blocks.
pub fn vgg_output_shape_generic(blocks: &[(usize, usize)], num_layers: usize) -> (usize, usize) {
    let mut last_c = 3usize;
    let mut spatial = 32usize;
    let mut w_idx = 0;

    for (num_convs, out_c) in blocks {
        let mut block_complete = true;
        for _ in 0..*num_convs {
            if w_idx >= num_layers {
                block_complete = false;
                break;
            }
            last_c = *out_c;
            w_idx += 1;
        }
        if block_complete {
            spatial /= 2;
        }
        if w_idx >= num_layers {
            break;
        }
    }
    (last_c, spatial)
}

/// Compute the output shape for VGG-16 blocks.
pub fn vgg_output_shape(num_layers: usize) -> (usize, usize) {
    vgg_output_shape_generic(&VGG16_BLOCKS, num_layers)
}

/// Compute the output shape for VGG-11 blocks.
pub fn vgg11_output_shape(num_layers: usize) -> (usize, usize) {
    vgg_output_shape_generic(&VGG11_BLOCKS, num_layers)
}
