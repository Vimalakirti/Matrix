use crate::dag::{DagBuilder, EdgeId, Witness};

/// Block configs: (num_convs_per_block, out_channels)
const VGG11_BLOCKS: [(usize, usize); 5] = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)];
const VGG16_BLOCKS: [(usize, usize); 5] = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)];

/// Generic VGG builder parameterized by block config.
///
/// Input: [3, 32, 32]. Each conv uses 3×3 kernel, stride=1, pad=1 (same padding).
/// MaxPool is 2×2, stride=2.
fn vgg_generic(
    blocks: &[(usize, usize)],
    conv_weights: Vec<Witness>,
    fc_weight: Witness,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    let blocks = blocks.to_vec();
    move |g, x| {
        assert_eq!(x.len(), 1, "VGG expects 1 input");
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

        let fc_e = g.param(fc_weight);
        g.einsum("i,ij->j".to_string(), vec![h, fc_e], false)
    }
}

/// VGG-11 on CIFAR-10 (8 conv layers).
///
/// Blocks: Conv64 → MaxPool → Conv128 → MaxPool → Conv256×2 → MaxPool →
///         Conv512×2 → MaxPool → Conv512×2 → MaxPool → FC(512→10)
pub fn vgg11(
    conv_weights: Vec<Witness>,
    fc_weight: Witness,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    vgg_generic(&VGG11_BLOCKS, conv_weights, fc_weight)
}

/// VGG-16 on CIFAR-10 (13 conv layers).
///
/// Blocks: Conv64×2 → MaxPool → Conv128×2 → MaxPool → Conv256×3 → MaxPool →
///         Conv512×3 → MaxPool → Conv512×3 → MaxPool → FC(512→10)
pub fn vgg16(
    conv_weights: Vec<Witness>,
    fc_weight: Witness,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    vgg_generic(&VGG16_BLOCKS, conv_weights, fc_weight)
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
