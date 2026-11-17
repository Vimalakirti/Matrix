//! DeepProve's ONNX CNNs (zkml/assets/scripts: the CIFAR-10 LeNet and the
//! COVID-19 CT classifier) with their trained weights and first input,
//! exported at fixed-point scale 2^SF by scripts/bench/deepprove/export_cnn.py.
//! Conv (valid, stride 1) -> rescale -> bias -> ReLU [-> 2x2 max pool], then
//! Flatten + Gemm layers [+ Sigmoid], as in the ONNX graph. Flatten + the first
//! Gemm is one contraction over (w, h, c): a relabelling view of the conv
//! output, no data movement.
//!
//! The forward pass runs once; with MORPHEUS_AT_RUN=1 it is handed to the
//! Morpheus driver (run_models.sh). Env: CNN_FIXTURE (JSON path).

use serde::Deserialize;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::util::arith::int_to_f;

#[derive(Deserialize)]
struct Layer {
    op: String,
    #[serde(default)] co: usize,
    #[serde(default)] ci: usize,
    #[serde(default)] k: usize,
    #[serde(default, rename = "in")] fin: usize,
    #[serde(default)] out: usize,
    #[serde(default)] w: Vec<i64>,
    #[serde(default)] b: Vec<i64>,
}
#[derive(Deserialize)]
struct Fixture { sf: usize, input_shape: Vec<usize>, input: Vec<i64>, layers: Vec<Layer> }

fn wit(shape: Vec<usize>, data: Vec<i64>, sf: usize, role: Role) -> Witness {
    Witness::new(shape, data.into_iter().map(|v| int_to_f(v as i128)).collect(), DataType::Float, sf, role)
}

fn main() {
    let path = std::env::var("CNN_FIXTURE").expect("CNN_FIXTURE");
    let fx: Fixture = serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    // Every scale-dependent table follows the fixture's scale.
    std::env::set_var("ZK4_SF_LOG", fx.sf.to_string());
    almost_goldilocks_cuda::init().expect("CUDA init");
    let sf = fx.sf;
    let p2 = |n: usize| n.next_power_of_two();
    let [c0, h0, w0]: [usize; 3] = fx.input_shape.clone().try_into().unwrap();

    // Input [c, h, w], row-major with padded extents (w in the low bits).
    let mut xin = vec![0i64; p2(c0) * p2(h0) * p2(w0)];
    for c in 0..c0 { for y in 0..h0 { for x in 0..w0 {
        xin[(c * p2(h0) + y) * p2(w0) + x] = fx.input[(c * h0 + y) * w0 + x];
    } } }

    let mut g = DagBuilder::new();
    let x = g.input(vec![c0, h0, w0], DataType::Float);
    let (mut c, mut h, mut w) = (c0, h0, w0);
    let mut e = x;
    let mut flat = false;
    let mut fc_idx = 0;
    let n_layers = fx.layers.len();
    for (li, l) in fx.layers.iter().enumerate() {
        match l.op.as_str() {
            "conv" => {
                assert_eq!(l.ci, c);
                let (kp, cip) = (p2(l.k), p2(l.ci));
                let mut wd = vec![0i64; p2(l.co) * cip * kp * kp];
                for o in 0..l.co { for i in 0..l.ci { for a in 0..l.k { for bb in 0..l.k {
                    wd[((o * cip + i) * kp + a) * kp + bb] = l.w[((o * l.ci + i) * l.k + a) * l.k + bb];
                } } } }
                let wt = g.param(wit(vec![l.co, l.ci, l.k, l.k], wd, sf, Role::Constant));
                let mut bd = l.b.clone(); bd.resize(p2(l.co), 0);
                let bt = g.param(wit(vec![l.co], bd, sf, Role::Constant));
                e = g.conv2d(e, wt, (l.k, l.k))[0];
                e = g.scale(e, 2 * sf, sf)[0];
                e = g.add_channel_bias_row_major(e, bt, l.co, 2);
                c = l.co; h = h - l.k + 1; w = w - l.k + 1;
            }
            "relu" => { e = g.relu(e); }
            "maxpool" => { assert!(!flat && h % 2 == 0 && w % 2 == 0); e = g.maxpool2d(e, 2, 2); h /= 2; w /= 2; }
            "flatten" => { flat = true; }
            "fc" => {
                assert!(flat);
                let op = p2(l.out);
                let mut bd = l.b.clone(); bd.resize(op, 0);
                let bt = g.param(wit(vec![l.out], bd, sf, Role::Constant));
                if fc_idx == 0 {
                    // Flatten order c*h*w + y*w + x; contract over (x, y, c).
                    assert_eq!(l.fin, c * h * w);
                    let (wp, hp, cp) = (p2(w), p2(h), p2(c));
                    let mut wd = vec![0i64; wp * hp * cp * op];
                    for j in 0..l.out { for cc in 0..c { for y in 0..h { for xx in 0..w {
                        wd[xx + wp * (y + hp * (cc + cp * j))] = l.w[j * l.fin + (cc * h + y) * w + xx];
                    } } } }
                    let wt = g.param(wit(vec![w, h, c, l.out], wd, sf, Role::Constant));
                    let v = g.change_shape(e, vec![w, h, c]);
                    e = g.einsum("whc,whcj->j".to_string(), vec![v, wt], true)[0];
                } else {
                    let ip = p2(l.fin);
                    let mut wd = vec![0i64; ip * op];
                    for j in 0..l.out { for i in 0..l.fin { wd[i + ip * j] = l.w[j * l.fin + i]; } }
                    let wt = g.param(wit(vec![l.fin, l.out], wd, sf, Role::Constant));
                    e = g.einsum("i,ij->j".to_string(), vec![e, wt], true)[0];
                }
                e = g.add(e, bt)[0];
                fc_idx += 1;
            }
            // CNN_NO_SIGMOID=1: stop at the logit. The COVID model's logits
            // (about -1086 on every shipped input) lie outside the finite
            // exponential profile; the sigmoid there is exactly 0.
            "sigmoid" => { assert_eq!(li, n_layers - 1); if std::env::var("CNN_NO_SIGMOID").is_err() { e = g.sigmoid(e)[0]; } }
            o => panic!("unsupported layer {o}"),
        }
    }
    let (dag, template) = g.compile();
    println!("DeepProve CNN {}: {} nodes, {} edges, output edge {}", path, dag.nodes.len(), dag.num_edges(), e);
    let mut wits = template.clone();
    let inputs = vec![(x, wit(vec![c0, h0, w0], xin, sf, Role::Input))];
    dag.run(&mut wits, &inputs);
    if let Some(out) = wits[e][0].data.as_ref() {
        let v = out.evaluations();
        let shown: Vec<i128> = v.iter().take(10).map(|f| zk_torch_4::util::arith::f_to_int(*f)).collect();
        println!("output (scale 2^{sf}): {:?}", shown);
    }
}
