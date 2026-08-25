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
        assert!(!x.is_empty(), "VGG expects >=1 input (batch)");
        assert_eq!(conv_weights.len(), conv_biases.len(), "Each conv needs a bias");
        assert_eq!(fc_weights.len(), 3, "VGG paper has 3 FC layers");
        assert_eq!(fc_biases.len(), 3, "VGG paper has 3 FC biases");

        // Fixed-point scale: when weights carry sf>0, rescale each conv/FC
        // output back to sf so committed activations stay bounded across depth.
        let sf_log = conv_weights.first().map(|w| w.sf).unwrap_or(0);
        let num_weights = conv_weights.len();
        // Initial spatial from the input edge (square) → any resolution.
        let mut spatial = g.init_values[x[0]].as_ref().unwrap().shape[1];

        // ---- Phase 1: shared conv/FC param edges (once for the whole batch) ----
        struct ConvP { w: EdgeId, b: EdgeId, maxpool_after: bool }
        let mut convs: Vec<ConvP> = Vec::new();
        let mut last_out_channels = 3usize;
        {
            let mut w_idx = 0;
            for (num_convs, out_channels) in &blocks {
                let mut block_complete = true;
                let mut block_convs = Vec::new();
                for _ in 0..*num_convs {
                    if w_idx >= num_weights { block_complete = false; break; }
                    let w = g.param(conv_weights[w_idx].clone());
                    let bb = g.param(conv_biases[w_idx].clone());
                    block_convs.push((w, bb));
                    last_out_channels = *out_channels;
                    w_idx += 1;
                }
                let n = block_convs.len();
                for (i, (w, bb)) in block_convs.into_iter().enumerate() {
                    convs.push(ConvP { w, b: bb, maxpool_after: block_complete && i == n - 1 });
                }
                if block_complete { spatial /= 2; }
                if w_idx >= num_weights { break; }
            }
        }
        let flat_dim = last_out_channels * spatial * spatial;
        let fc1_w = g.param(fc_weights[0].clone()); let fc1_b = g.param(fc_biases[0].clone());
        let fc2_w = g.param(fc_weights[1].clone()); let fc2_b = g.param(fc_biases[1].clone());
        let fc3_w = g.param(fc_weights[2].clone()); let fc3_b = g.param(fc_biases[2].clone());

        // ---- Phase 2: build the body ----
        //
        // Each entry of `x` is an independent chain. A BATCHED entry is one
        // folded tensor [b_pad*c_pad, H, W] carrying `batch` images, and it
        // still builds ONE chain: conv binds the batch variables instead of
        // summing them, and every other op here is per-channel, for which a
        // batch is simply more channels. Building `batch` chains instead would
        // multiply the commitment leaves and cost superlinearly in the fold.
        let mut outs = Vec::with_capacity(x.len());
        for &inp in x {
            // batch = leading extent / padded input channels (3 -> 4 for RGB).
            let lead = g.init_values[inp].as_ref().unwrap().shape[0];
            let batch = (lead / 3usize.next_power_of_two()).max(1);
            let mut h = inp;
            for cp in &convs {
                h = g.pad(h, 1, 1);
                h = g.conv2d(h, cp.w, (3, 3))[0];
                if sf_log > 0 { h = g.scale(h, 2 * sf_log, sf_log)[0]; }
                h = g.add(h, cp.b)[0];
                h = g.relu(h);
                g.layer_boundaries.push(h);
                if cp.maxpool_after { h = g.maxpool2d(h, 2, 2); }
            }
            // The conv stack keeps b in the HIGH bits; einsum puts the FIRST
            // shape dimension in the LOW bits. So the FC head carries the batch
            // as the LAST entry, [features, B], which makes the reshape a pure
            // relabelling and keeps every FC output in the same form.
            if batch > 1 {
                h = g.change_shape(h, vec![flat_dim, batch]);
                h = g.einsum("ib,ij->jb".to_string(), vec![h, fc1_w], sf_log > 0)[0];
                h = g.add(h, fc1_b)[0]; h = g.relu(h);
                h = g.einsum("ib,ij->jb".to_string(), vec![h, fc2_w], sf_log > 0)[0];
                h = g.add(h, fc2_b)[0]; h = g.relu(h);
                h = g.einsum("ib,ij->jb".to_string(), vec![h, fc3_w], sf_log > 0)[0];
                h = g.add(h, fc3_b)[0];
            } else {
                h = g.change_shape(h, vec![flat_dim]);
                h = g.einsum("i,ij->j".to_string(), vec![h, fc1_w], sf_log > 0)[0];
                h = g.add(h, fc1_b)[0]; h = g.relu(h);
                h = g.einsum("i,ij->j".to_string(), vec![h, fc2_w], sf_log > 0)[0];
                h = g.add(h, fc2_b)[0]; h = g.relu(h);
                h = g.einsum("i,ij->j".to_string(), vec![h, fc3_w], sf_log > 0)[0];
                h = g.add(h, fc3_b)[0];
            }
            outs.push(h);
        }
        outs
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

/// Compute the output shape (channels, spatial) after running N conv layers
/// through given blocks, starting from `input_size` spatial resolution.
pub fn vgg_output_shape_generic(blocks: &[(usize, usize)], num_layers: usize, input_size: usize) -> (usize, usize) {
    let mut last_c = 3usize;
    let mut spatial = input_size;
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

/// Compute the output shape for VGG-16 blocks at `input_size` resolution.
pub fn vgg_output_shape(num_layers: usize, input_size: usize) -> (usize, usize) {
    vgg_output_shape_generic(&VGG16_BLOCKS, num_layers, input_size)
}

/// Compute the output shape for VGG-11 blocks at `input_size` resolution.
pub fn vgg11_output_shape(num_layers: usize, input_size: usize) -> (usize, usize) {
    vgg_output_shape_generic(&VGG11_BLOCKS, num_layers, input_size)
}
