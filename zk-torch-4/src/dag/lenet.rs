//! LeNet-5 exactly as EZKL ships it in `examples/onnx/lenet_5`.
//!
//! Read off that ONNX graph rather than from the classic paper, because the two
//! differ in the parts that matter for a prover:
//!
//!   input [1, 1, 32, 32]
//!   Conv 6x1x5x5, no pad, stride 1        -> [6, 28, 28]
//!   x^2 + x                                (Mul(y,y) then Add(.,y))
//!   AveragePool 2x2 stride 2               -> [6, 14, 14]
//!   Conv 16x6x5x5, no pad, stride 1        -> [16, 10, 10]
//!   x^2 + x
//!   AveragePool 2x2 stride 2               -> [16, 5, 5]
//!   Flatten                                -> 400
//!   Gemm 400->120, 120->84, 84->10
//!
//! Two things are NOT the textbook LeNet. The activation is the quadratic
//! `x^2 + x`, not tanh or ReLU: in the ONNX it appears as `Mul(y, y)` followed
//! by `Add(., y)`. That is a degree-2 polynomial, so it needs no range check and
//! no lookup, which is why this model is cheap for us. And pooling is AVERAGE,
//! not max; average pooling is linear and is expressed here as a depthwise
//! convolution; average pooling is linear and is built here from four strided
//! gathers plus a divide, so it costs no comparison chain and no lookup, unlike
//! the maxpool every other CNN in this table uses.

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::SF_LOG;

/// `x^2 + x`, the activation EZKL's LeNet uses after each convolution.
///
/// The einsum takes `scale_back = false` and the rescale is explicit. Automatic
/// scale-back derives the product's scale from the operands' BUILD-TIME `sf`,
/// and `g.input` declares 0 while the runtime witness carries `SF_LOG`; the
/// derived rescale is then a no-op to scale 0 and the run aborts on the first
/// ScaleUp. Every other CV builder here states the rescale explicitly for the
/// same reason.
fn quadratic_activation(g: &mut DagBuilder, x: EdgeId, eq: &str) -> EdgeId {
    let sq = g.einsum(eq.to_string(), vec![x, x], false)[0];
    let sq = if *SF_LOG > 0 { g.scale(sq, 2 * *SF_LOG, *SF_LOG)[0] } else { sq };
    g.add(sq, x)[0]
}

/// Average pooling, `2x2` stride `2`, as a strided convolution against a
/// block-diagonal `1/4` kernel: `w[co][ci][kh][kw] = 1/4` when `co == ci`.
///
/// Average pooling is linear, so it needs no comparison and no lookup, unlike
/// the maxpool the other CNNs here use. Two earlier attempts are worth
/// recording. A depthwise conv with a `[C,1,2,2]` kernel failed its
/// FlattenKernel check because the constant was written column-major, the
/// einsum operand convention, rather than the padded row-major a conv weight
/// expects. Composing `subsample2d` gathers instead failed differently:
/// `subsample2d` truncates the final window, so offset 1 on a width-28 input
/// returns 13 columns rather than 14 and the four gathers no longer broadcast
/// against each other. The layout used here was verified against a hand
/// computed average before being wired in.
fn avg_pool2x2(g: &mut DagBuilder, x: EdgeId, channels: usize) -> EdgeId {
    let cp = channels.next_power_of_two();
    let kp = 2usize.next_power_of_two();
    let mut data = crate::zero_witness_vec(cp * cp * kp * kp);
    let quarter = AlmostGoldilocksField(1u64 << (*SF_LOG - 2));
    for c in 0..channels {
        for kh in 0..2 {
            for kw in 0..2 {
                data[((c * cp + c) * kp + kh) * kp + kw] = quarter;
            }
        }
    }
    let w = Witness::new(vec![channels, channels, 2, 2], data,
                         DataType::Uint, *SF_LOG, Role::Constant);
    let w_e = g.param(w);
    let out = g.conv2d_strided(x, w_e, (2, 2), (2, 2))[0];
    // Convolution multiplies two fixed-point tensors without rescaling, so the
    // product sits at 2*sf and is brought back explicitly, as after every other
    // convolution here.
    if *SF_LOG > 0 { g.scale(out, 2 * *SF_LOG, *SF_LOG)[0] } else { out }
}

/// EZKL's LeNet-5. `conv_w` is `[conv1 (6,1,5,5), conv2 (16,6,5,5)]`, `fc_w` is
/// `[400x120, 120x84, 84x10]`, and the biases follow ONNX shapes. The graph is
/// built to match the exported ONNX operator for operator.
pub fn lenet5(
    conv_w: Vec<Witness>,
    conv_b: Vec<Witness>,
    fc_w: Vec<Witness>,
    fc_b: Vec<Witness>,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert_eq!(x.len(), 1, "LeNet-5 expects 1 input");
        assert_eq!(conv_w.len(), 2, "LeNet-5 has 2 convolutions");
        assert_eq!(fc_w.len(), 3, "LeNet-5 has 3 fully connected layers");
        let mut h = x[0];

        // conv1 -> x^2+x -> avgpool : [1,32,32] -> [6,28,28] -> [6,14,14]
        let w0 = g.param(conv_w[0].clone());
        h = g.conv2d(h, w0, (5, 5))[0];
        if *SF_LOG > 0 { h = g.scale(h, 2 * *SF_LOG, *SF_LOG)[0]; }
        let b0 = g.param(conv_b[0].clone());
        h = g.add(h, b0)[0];
        h = quadratic_activation(g, h, "chw,chw->chw");
        h = avg_pool2x2(g, h, 6);
        g.layer_boundaries.push(h);

        // conv2 -> x^2+x -> avgpool : [6,14,14] -> [16,10,10] -> [16,5,5]
        let w1 = g.param(conv_w[1].clone());
        h = g.conv2d(h, w1, (5, 5))[0];
        if *SF_LOG > 0 { h = g.scale(h, 2 * *SF_LOG, *SF_LOG)[0]; }
        let b1 = g.param(conv_b[1].clone());
        h = g.add(h, b1)[0];
        h = quadratic_activation(g, h, "chw,chw->chw");
        h = avg_pool2x2(g, h, 16);
        g.layer_boundaries.push(h);

        // Flatten to 400 and run the classifier. ONNX Gemm has transB=1, so the
        // stored weight is [out, in] and the contraction is over `out`'s second
        // axis; einsum states that directly rather than transposing first.
        h = g.change_shape(h, vec![16 * 5 * 5]);
        for i in 0..3 {
            let w_e = g.param(fc_w[i].clone());
            h = g.einsum("i,oi->o".to_string(), vec![h, w_e], false)[0];
            if *SF_LOG > 0 { h = g.scale(h, 2 * *SF_LOG, *SF_LOG)[0]; }
            let b_e = g.param(fc_b[i].clone());
            h = g.add(h, b_e)[0];
            g.layer_boundaries.push(h);
        }
        vec![h]
    }
}

