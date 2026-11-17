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
//! not max. Its finite mean uses four explicit-size strided gathers plus
//! exact integer division with authenticated range roots; its repaired cost
//! must not be read from historical no-lookup timings.

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

/// Exact 2x2 mean, using four explicit-size row-major gathers and DivConst.
/// The explicit output extent avoids the old offset-one truncation bug. The
/// authenticated row-major mask excludes spatial/channel padding BEFORE sums.
pub(crate) fn avg_pool2x2(g: &mut DagBuilder, x: EdgeId, channels: usize) -> EdgeId {
    let shape=g.init_values[x].as_ref().unwrap().shape.clone();
    let sf=g.init_values[x].as_ref().unwrap().sf;
    assert!(shape.len()==3 || shape.len()==4);
    let h=shape[shape.len()-2];let w=shape[shape.len()-1];
    assert!(h>0 && w>0 && h%2==0 && w%2==0);
    let cp=channels.next_power_of_two();
    let (batch,rows)=if shape.len()==4 {
        assert_eq!(shape[1],channels);(shape[0],shape[0].next_power_of_two()*cp)
    } else if shape[0]==channels {(1,channels)} else {
        assert_eq!(shape[0]%cp,0);(shape[0]/cp,shape[0])
    };
    let hp=h.next_power_of_two();let wp=w.next_power_of_two();
    let len=rows.next_power_of_two()*hp*wp;
    let mut mask=vec![AlmostGoldilocksField(0);len];
    for c in 0..rows {
        if c/cp>=batch || c%cp>=channels {continue;}
        for ih in 0..h {for iw in 0..w {mask[iw+wp*(ih+hp*c)]=AlmostGoldilocksField(1);}}
    }
    let mask=g.public_literal(Witness::new(vec![len],mask,DataType::Bool,0,Role::Input));
    let xf=g.change_shape(x,vec![len]);
    let masked=g.einsum("i,i->i".into(),vec![xf,mask],false)[0];
    let x=g.change_shape(masked,shape);
    let bound=(1i128<<30)-1;g.integer_interval(x,-bound,bound);
    let mut parts=Vec::new();
    for oh in 0..2 {for ow in 0..2 {
        let part=g.subsample2d_sized(x,2,2,oh,ow,h/2,w/2);
        g.init_values[part].as_mut().unwrap().sf=sf;
        parts.push(part);
    }}
    let a=g.add(parts[0],parts[1])[0];let b=g.add(parts[2],parts[3])[0];
    let sum=g.add(a,b)[0];g.div_const(sum,4)[0]
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

