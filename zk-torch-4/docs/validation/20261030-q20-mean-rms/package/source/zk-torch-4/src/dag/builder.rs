use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use ndarray::ArrayD;
use std::collections::HashMap;

use crate::basicblock::*;
use crate::basicblock::BasicBlockType;
use crate::dag::{AliasId, Dag, DataType, EdgeId, Node, NodeId, Role, Witness};
use crate::util::arith::{int_to_f, next_pow};
use crate::util::shape::{broadcast_shape, pad_to_pow_of_two};
use crate::SF_LOG;

fn letters(a: usize) -> String {
    (0..a).map(|i| (b'a' + i as u8) as char).collect()
}

/// DagBuilder: DSL for constructing the computation graph.
pub struct DagBuilder {
    pub(crate) program_literals: std::collections::BTreeMap<EdgeId, Vec<u64>>,
    pub nodes: Vec<Node>,
    pub num_edges: usize,
    pub init_values: Vec<Option<Witness>>,
    pub range: Vec<NodeId>,
    pub two_pow: Vec<NodeId>,
    /// Layer boundary edges recorded during model construction.
    /// These are the hidden state edges between transformer layers.
    pub layer_boundaries: Vec<EdgeId>,
    /// Output edges that need self-claims (Conv2D output edges).
    pub self_claim_edges: Vec<EdgeId>,
}

impl DagBuilder {
    pub fn new() -> Self {
        Self {
            program_literals: Default::default(),
            nodes: Vec::new(),
            num_edges: 0,
            init_values: Vec::new(),
            range: Vec::new(),
            two_pow: Vec::new(),
            layer_boundaries: Vec::new(),
            self_claim_edges: Vec::new(),
        }
    }

    /// Create a graph input edge (no known value).
    pub fn input(&mut self, shape: Vec<usize>, data_type: DataType) -> EdgeId {
        let witness = Witness::new_wo_data(
            shape,
            data_type,
            if data_type == DataType::Float { *SF_LOG } else { 0 },
            Role::Input,
        );
        let e = self.num_edges;
        self.num_edges += 1;
        self.init_values.push(Some(witness));
        e
    }

    /// Create a parameter/constant edge with a known value.
    pub fn param(&mut self, t: Witness) -> EdgeId {
        let e = self.num_edges;
        self.num_edges += 1;
        assert_eq!(t.role, Role::Constant, "Parameters must be constants");
        self.init_values.push(Some(t));
        e
    }

    /// Immutable public program data, NOT a private model weight or request input.
    /// Its canonical values are frozen into both program and request identities.
    pub fn public_literal(&mut self, mut t: Witness) -> EdgeId {
        assert_eq!(t.poly_type, crate::dag::PolyType::Dense, "literal must be dense");
        let values = t.data.as_ref().expect("literal requires data").evaluations()
            .iter().map(|x| x.reduce().0).collect();
        t.role = Role::Input;
        let e = self.committed_input(t);
        self.program_literals.insert(e, values);
        e
    }

    /// Dense full-padded literal used for integer range constraints.
    pub(crate) fn integer_literal(&mut self, shape: Vec<usize>, value: i128, sf: usize) -> EdgeId {
        let n = shape.iter().map(|s| s.next_power_of_two()).product();
        self.public_literal(Witness::new(shape, vec![int_to_f(value); n], DataType::Float, sf, Role::Input))
    }

    /// Integer interval certificate. Both sides are lookup roots, with a
    /// common authenticated small sum; the field relation therefore lifts.
    pub(crate) fn integer_interval(&mut self, a: EdgeId, lo: i128, hi: i128) {
        assert!(lo <= hi && hi-lo < (1i128<<62), "unsupported interval budget");
        let w=self.init_values[a].as_ref().unwrap();
        let shape=w.shape.clone(); let sf=w.sf;
        let low=self.integer_literal(shape.clone(),lo,sf);
        let high=self.integer_literal(shape,hi,sf);
        let above=self.sub(a,low)[0];
        let below=self.sub(high,a)[0];
        self.init_values[above].as_mut().unwrap().role=Role::Output;
        self.init_values[below].as_mut().unwrap().role=Role::Output;
        let bits=(128-((hi-lo) as u128).leading_zeros() as usize).max(1);
        let row_bits=crate::util::arith::get_n(&self.init_values[a].as_ref().unwrap().shape);
        assert!(bits+row_bits < usize::BITS as usize ||
            (std::env::var("NO_SPARSE_SPLIT").is_err() && bits>*crate::TABLE_COMMIT_LOG),
            "wide normalization ranges require split sparse representation");
        self.add_nonneg_node_bits(above,bits);
        self.add_nonneg_node_bits(below,bits);
    }

    /// Create a leaf edge with a known value but a NON-`Constant` role
    /// (`Input`/`Auxiliary`/`Output`). Unlike [`Self::param`], the value is
    /// NOT treated as a shared model weight: it is committed and opened
    /// per-proof, and the streaming accumulator never defers it. Used for
    /// per-generation public data such as the one-shot AR one-hot
    /// selectors, whose value changes every generation. Callers overwrite
    /// the edge's witness in place before each proof and re-run the DAG.
    pub fn committed_input(&mut self, t: Witness) -> EdgeId {
        assert_ne!(t.role, Role::Constant,
            "committed_input is for per-proof values; use param() for Constants");
        let e = self.num_edges;
        self.num_edges += 1;
        self.init_values.push(Some(t));
        e
    }

    pub fn add_gkr_node(&mut self, inps: Vec<EdgeId>, basicblock: BasicBlockType) -> Vec<EdgeId> {
        let nid = self.nodes.len();
        let eid = self.num_edges;
        let outs: Vec<EdgeId> = (eid..eid + basicblock.out_arity()).collect();
        self.nodes.push(Node {
            id: nid,
            kind: basicblock,
            inputs: inps,
            outputs: outs.clone(),
        });
        self.num_edges += outs.len();
        outs
    }

    /// Wire a `DivConst` node — output's shape/sf/data_type mirror the
    /// input. Ported from zk-torch-3's `dag/llama.rs::div_const`.
    pub fn div_const(&mut self, a: EdgeId, c: usize) -> Vec<EdgeId> {
        let block = BasicBlockType::DivConst(DivConst { divisor: c as u64 });
        assert!(self.init_values[a].is_some(), "div_const: input not initialized");
        let inp = self.init_values[a].as_ref().unwrap();
        let shape = inp.shape.clone();
        let sf = inp.sf;
        let data_type = inp.data_type;
        let out = Witness::new_wo_data(shape, data_type, sf, Role::Output);
        self.init_values.push(Some(out));
        let y = self.add_gkr_node(vec![a], block)[0];
        vec![y]
    }

    fn division_guards(&mut self,a: EdgeId,y: EdgeId,c: usize) {
        let input=self.init_values[a].as_ref().unwrap();
        let output=self.init_values[y].as_ref().unwrap();
        assert_eq!(input.shape,output.shape,"division advice shape mismatch");
        assert_eq!(input.sf,output.sf,"division advice scale mismatch");
        assert!(input.shape.iter().all(|&d| d>0),"invalid division shape");
        let sf=self.init_values[a].as_ref().unwrap().sf;
        // Exact integer division on a public bounded domain. The registered
        // DAG includes these roots; advice alone is never the specification.
        assert!((1..=(1 << 20)).contains(&c), "unsupported divisor");
        const B: i128 = (1i128 << 54) - 1;
        self.integer_interval(a, -B, B);
        self.integer_interval(y, -((B / c as i128) + 1), (B / c as i128) + 1);
        let shape=self.init_values[a].as_ref().unwrap().shape.clone();
        let n=shape.iter().map(|x| x.next_power_of_two()).product();
        let af=self.change_shape(a,vec![n]);
        let yf=self.change_shape(y,vec![n]);
        let divisor=self.integer_literal(vec![n],c as i128,0);
        let cy=self.einsum("i,i->i".into(),vec![yf,divisor],false)[0];
        let rem=self.sub(af,cy)[0];
        // rho = a-c*y+floor(c/2) lies in [0,c-1]. Unlike the old
        // symmetric interval this has exactly one allowed half-tie answer.
        let half=self.integer_literal(vec![n],(c/2) as i128,sf);
        let rho=self.add(rem,half)[0];
        self.integer_interval(rho,0,c as i128-1);
    }

    pub fn add_nonneg_node(&mut self, a: EdgeId) {
        let nid = self.nodes.len();
        // Default range table covers the dynamic range of zk-torch-4's
        // signed-int representation per the global `TABLE_SIZE_LOG` config
        // (typically 20 → table covers [0, 2^20)).
        let nonneg_basicblock =
            BasicBlockType::NonNegative(NonNegative::new(*crate::TABLE_SIZE_LOG));
        let _ = self.add_gkr_node(vec![a], nonneg_basicblock);
        self.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
        self.range.push(nid);
    }

    /// Range check `a ∈ [0, 2^bits)` with an explicit width (wider than the
    /// global table; provers that chunk their lookups handle any width).
    pub fn add_nonneg_node_bits(&mut self, a: EdgeId, bits: usize) {
        let nid = self.nodes.len();
        let _ = self.add_gkr_node(vec![a], BasicBlockType::NonNegative(NonNegative::new(bits)));
        self.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
        self.range.push(nid);
    }

    pub fn change_shape(&mut self, a: EdgeId, shape: Vec<usize>) -> EdgeId {
        let change_shape_basicblock = BasicBlockType::ChangeShape(ChangeShape {
            target_shape: shape.clone(),
        });
        let outs = self.add_gkr_node(vec![a], change_shape_basicblock);
        self.init_values.push(Some(Witness::new_wo_data(
            shape,
            self.init_values[a].as_ref().unwrap().data_type,
            self.init_values[a].as_ref().unwrap().sf,
            Role::Output,
        )));
        outs[0]
    }

    pub fn reshape(&mut self, a: EdgeId, shape: Vec<usize>) -> Vec<EdgeId> {
        let witness = self.init_values[a].as_ref().unwrap().clone();
        let original_shape = witness.shape.clone();

        let out = if shape.len() > original_shape.len() {
            assert!(
                shape[shape.len() - 1] * shape[shape.len() - 2]
                    == original_shape[original_shape.len() - 1],
                "Invalid shape"
            );
            let a = self.change_shape(
                a,
                vec![
                    original_shape[0],
                    original_shape[1],
                    shape[shape.len() - 1],
                    shape[shape.len() - 2],
                ],
            );
            self.einsum("bsdh->bshd".to_string(), vec![a], false)
        } else if shape.len() < original_shape.len() {
            assert!(
                shape[shape.len() - 1]
                    == original_shape[original_shape.len() - 1]
                        * original_shape[original_shape.len() - 2],
                "Invalid shape"
            );
            let o = self.einsum("bshd->bsdh".to_string(), vec![a], false);
            let o = self.change_shape(o[0], shape);
            vec![o]
        } else {
            panic!("Not supported yet");
        };

        out
    }

    pub fn mask(&mut self, a: EdgeId, raw_mask_shape: Vec<usize>) -> EdgeId {
        let s = letters(raw_mask_shape.len());
        let val_num: usize = raw_mask_shape.iter().product();
        let vals: Vec<AlmostGoldilocksField> = (0..val_num).map(|_| AlmostGoldilocksField(1)).collect();
        let val_arr = ArrayD::from_shape_vec(raw_mask_shape.clone(), vals).unwrap();
        let pad_val_arr = pad_to_pow_of_two(&val_arr, &AlmostGoldilocksField(0));
        let col_major_output: Vec<_> = pad_val_arr.view().reversed_axes().iter().cloned().collect();
        let mask = Witness::new(raw_mask_shape, col_major_output, DataType::Float, 0, Role::Constant);
        let e = self.public_literal(mask);
        let out = self.einsum(format!("{},{}->{}", s, s, s), vec![a, e], false);
        out[0]
    }

    /// Create a causal attention mask and add it to scores.
    /// scores shape must end with [..., seq_len, seq_len].
    /// Adds a large negative value to positions where key > query (future tokens).
    /// The mask is created with the same full shape as scores so that `add` broadcasting works.
    pub fn causal_mask(&mut self, scores: EdgeId, seq_len: usize) -> EdgeId {
        let scores_shape = self.init_values[scores].as_ref().unwrap().shape.clone();
        let sf = self.init_values[scores].as_ref().unwrap().sf;
        let padded: Vec<usize> = scores_shape.iter().map(|&s| s.next_power_of_two()).collect();
        let total_padded: usize = padded.iter().product();

        // Large negative value: exp(-100) ≈ 0
        let big_neg = int_to_f(-100 * (1i128 << sf));

        // Build mask in little-endian layout matching scores shape.
        // Last two dims are [s, t] where s=query, t=key.
        // s has stride = product of all padded dims before it
        // t has stride = s_stride * s_pad
        let ndim = scores_shape.len();
        let s_pad = padded[ndim - 2];
        let t_pad = padded[ndim - 1];
        let s_stride: usize = padded[..ndim - 2].iter().product();
        let t_stride = s_stride * s_pad;

        let mut mask_data = vec![AlmostGoldilocksField(0); total_padded];
        // For each group (batch, head, ...) replicate the same causal pattern
        let num_groups: usize = padded[..ndim - 2].iter().product();
        for g in 0..num_groups {
            for t in 0..t_pad {
                for s in 0..s_pad {
                    let idx = g + s * s_stride + t * t_stride;
                    if t < seq_len && s < seq_len && t > s {
                        // Future position: large negative
                        mask_data[idx] = big_neg;
                    }
                    // else: 0 (valid position or padding)
                }
            }
        }

        let mask = Witness::new(
            scores_shape.clone(),
            mask_data,
            DataType::Float,
            sf,
            Role::Constant,
        );
        let mask_id = self.public_literal(mask);
        self.add(scores, mask_id)[0]
    }

    pub fn add(&mut self, a: EdgeId, b: EdgeId) -> Vec<EdgeId> {
        let add_basicblock = BasicBlockType::Add(Add);

        assert!(
            self.init_values[a].is_some() && self.init_values[b].is_some(),
            "Inputs must be initialized"
        );
        let out_value = if self.init_values[a].as_ref().unwrap().data.is_none()
            || self.init_values[b].as_ref().unwrap().data.is_none()
        {
            let shape = broadcast_shape(
                &self.init_values[a].as_ref().unwrap().shape,
                &self.init_values[b].as_ref().unwrap().shape,
            )
            .unwrap();
            let sf = self.init_values[a].as_ref().unwrap().sf;
            let data_type = self.init_values[a].as_ref().unwrap().data_type;
            Witness::new_wo_data(shape, data_type, sf, Role::Output)
        } else {
            let inps_values = vec![
                self.init_values[a].as_ref().unwrap(),
                self.init_values[b].as_ref().unwrap(),
            ];
            let mut out = add_basicblock.run(inps_values.as_slice()).first().unwrap().to_owned();
            // Having eager data does not make a request-dependent edge a
            // model parameter. Only model-only precomputation may be shared.
            out.role = if self.init_values[a].as_ref().unwrap().role == Role::Constant
                && self.init_values[b].as_ref().unwrap().role == Role::Constant {
                Role::Constant
            } else { Role::Output };
            out
        };
        self.init_values.push(Some(out_value));

        self.add_gkr_node(vec![a, b], add_basicblock)
    }

    pub fn sub(&mut self, a: EdgeId, b: EdgeId) -> Vec<EdgeId> {
        let sub_basicblock = BasicBlockType::Sub(Sub);

        assert!(
            self.init_values[a].is_some() && self.init_values[b].is_some(),
            "Inputs must be initialized"
        );
        let out_value = if self.init_values[a].as_ref().unwrap().data.is_none()
            || self.init_values[b].as_ref().unwrap().data.is_none()
        {
            let shape = broadcast_shape(
                &self.init_values[a].as_ref().unwrap().shape,
                &self.init_values[b].as_ref().unwrap().shape,
            )
            .unwrap();
            let sf = self.init_values[a].as_ref().unwrap().sf;
            let data_type = self.init_values[a].as_ref().unwrap().data_type;
            Witness::new_wo_data(shape, data_type, sf, Role::Output)
        } else {
            let inps_values = vec![
                self.init_values[a].as_ref().unwrap(),
                self.init_values[b].as_ref().unwrap(),
            ];
            let mut out = sub_basicblock.run(inps_values.as_slice()).first().unwrap().to_owned();
            // Having eager data does not make a request-dependent edge a
            // model parameter. Only model-only precomputation may be shared.
            out.role = if self.init_values[a].as_ref().unwrap().role == Role::Constant
                && self.init_values[b].as_ref().unwrap().role == Role::Constant {
                Role::Constant
            } else { Role::Output };
            out
        };
        self.init_values.push(Some(out_value));

        self.add_gkr_node(vec![a, b], sub_basicblock)
    }

    pub fn einsum(&mut self, equation: String, inputs: Vec<EdgeId>, scale_back: bool) -> Vec<EdgeId> {
        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|&i| self.init_values[i].as_ref().unwrap().shape.clone())
            .collect();

        // Parse equation to compute output shape
        let mut shape_map = HashMap::new();
        let input_symbols: Vec<&str> = equation
            .split("->")
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.trim())
            .collect();
        for (i, symbols) in input_symbols.iter().enumerate() {
            for (j, c) in symbols.chars().enumerate() {
                shape_map.insert(c.to_string(), input_shapes[i][j]);
            }
        }
        let output_shape: Vec<usize> = equation
            .split("->")
            .nth(1)
            .unwrap()
            .chars()
            .map(|c| *shape_map.get(&c.to_string()).unwrap())
            .collect();

        let einsum_basicblock = BasicBlockType::Einsum(
            Einsum::new(&equation, input_shapes.clone(), output_shape.clone())
        );

        let output_data_type = self.init_values[inputs[0]].as_ref().unwrap().data_type;
        let input_sf: usize = inputs
            .iter()
            .map(|&i| self.init_values[i].as_ref().unwrap().sf)
            .sum();
        let output_sf = self.init_values[inputs[0]].as_ref().unwrap().sf;

        let mut outs = self.add_gkr_node(inputs.clone(), einsum_basicblock);
        self.init_values.push(Some(Witness::new_wo_data(
            output_shape,
            output_data_type,
            input_sf,
            Role::Output,
        )));
        if scale_back {
            outs = self.scale(outs[0], input_sf, output_sf);
        }
        outs
    }

    pub fn sigmoid_const(&mut self, a: EdgeId) -> Vec<EdgeId> {
        let sigmoid_const_basicblock =
            BasicBlockType::SigmoidConst(SigmoidConst { segments: 8 });
        assert!(self.init_values[a].is_some(), "Input must be initialized");
        let inp_value = self.init_values[a].as_ref().unwrap();
        let shape = inp_value.shape.clone();
        let sf = inp_value.sf;
        let data_type = inp_value.data_type;
        let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Output);
        self.init_values.push(Some(out_value));
        self.add_gkr_node(vec![a], sigmoid_const_basicblock)
    }

    pub fn sigmoid(&mut self, a: EdgeId) -> Vec<EdgeId> {
        let sigmoid_c = self.sigmoid_const(a)[0];
        let scores = self.add(a, sigmoid_c)[0];
        let scores = self.exp(scores)[0];
        vec![scores]
    }

    pub fn scale(&mut self, a: EdgeId, input_sf: usize, output_sf: usize) -> Vec<EdgeId> {
        let nid = self.nodes.len();
        let shape = self.init_values[a].as_ref().unwrap().shape.clone();
        let data_type = self.init_values[a].as_ref().unwrap().data_type;
        let scale_basicblock = if input_sf > output_sf {
            BasicBlockType::ScaleDown(ScaleDown { output_sf })
        } else {
            BasicBlockType::ScaleUp(ScaleUp { output_sf })
        };
        self.init_values.push(Some(Witness::new_wo_data(
            shape.clone(),
            data_type,
            output_sf,
            Role::Output,
        )));
        self.init_values.push(Some(Witness::new_wo_data(
            vec![1],
            data_type,
            0,
            Role::Auxiliary,
        )));
        self.range.push(nid);
        self.add_gkr_node(vec![a], scale_basicblock)
    }

    /// Canonical unclipped exp on logical coordinates, zero on padding.
    /// The input/output masks are immutable program literals derived from the
    /// registered logical shape; this is an explicit padding-profile repair.
    /// Downstream reductions can change if they previously consumed bad pads.
    pub fn exp(&mut self, a: EdgeId) -> Vec<EdgeId> {
        let shape = self.init_values[a].as_ref().unwrap().shape.clone();
        assert!(!shape.is_empty() && shape.len()<=26 && shape.iter().all(|&s| s>0),
            "exp requires explicit nonzero supported logical extents");
        let padded_len = shape.iter().try_fold(1usize, |n,&s| {
            let d=u32::try_from(s).ok()?.checked_next_power_of_two()? as usize;
            n.checked_mul(d)
        }).expect("exp padded shape exceeds supported indexing domain");
        let padded = shape.iter().any(|s| !s.is_power_of_two());
        // Mask BEFORE helper domain checks: shifts from SoftmaxConst can put
        // positive numbers in padded cells even when all logical cells are <=0.
        let a = if padded { self.mask(a,shape.clone()) } else { a };
        let flat_shape = vec![padded_len];
        let data_type = self.init_values[a].as_ref().unwrap().data_type;

        let sf = *SF_LOG;
        assert_eq!(self.init_values[a].as_ref().unwrap().sf, sf, "exp activation scale mismatch");
        let bits = crate::basicblock::exp::exp_bits();
        let _ = crate::basicblock::exp::exp_ln2(sf);
        let exp_basicblock = BasicBlockType::ExpHelper(ExpHelper { num_bits: bits });
        self.init_values.push(Some(Witness::new_wo_data(
            shape.clone(),
            data_type,
            *SF_LOG,
            Role::Output,
        )));
        self.init_values.push(Some(Witness::new_wo_data(
            vec![1],
            data_type,
            0,
            Role::Auxiliary,
        )));
        // ExpHelper range correctness is proven by prove_two_pow, not prove_range
        let outs = self.add_gkr_node(vec![a], exp_basicblock);
        let mut r = outs[0]; // dense poly
        let k = outs[1]; // sparse poly

        r = self.scale(r, *SF_LOG, 15)[0];
        // ScaleDown alone is a field quotient relation, not integer rounding.
        // Bound every rounded quotient before using it: together with the
        // bounded numerator this excludes nonzero multiples of p in residuals.
        self.integer_interval(r, -(1<<14), 1<<14);

        // A. compute 2^(-k)
        let nid = self.nodes.len();
        self.two_pow.push(nid);
        let two_pow_basicblock = BasicBlockType::TwoPow(TwoPow);
        let mut two_pow_out = self.add_gkr_node(vec![k], two_pow_basicblock)[0];
        self.init_values.push(Some(Witness::new_wo_data(
            shape.clone(),
            data_type,
            15,
            Role::Output,
        )));
        two_pow_out = self.change_shape(two_pow_out, flat_shape.clone());

        // B. compute exp(r) by Taylor series
        let val_num: usize = shape.iter().product();

        // B1. compute 1/6
        let vals_one_sixth: Vec<AlmostGoldilocksField> =
            (0..val_num).map(|_| AlmostGoldilocksField(5461)).collect(); // 2^15 / 6
        let vals_one_sixth = ArrayD::from_shape_vec(shape.clone(), vals_one_sixth).unwrap();
        let pad_vals_one_sixth = pad_to_pow_of_two(&vals_one_sixth, &AlmostGoldilocksField(0));
        let col_major_one_sixth: Vec<_> =
            pad_vals_one_sixth.view().reversed_axes().iter().cloned().collect();
        let one_sixth = Witness::new(
            flat_shape.clone(),
            col_major_one_sixth,
            DataType::Float,
            15,
            Role::Constant,
        );
        let one_sixth = self.public_literal(one_sixth);

        // B2. compute 1/2
        let vals_half: Vec<AlmostGoldilocksField> =
            (0..val_num).map(|_| AlmostGoldilocksField(16384)).collect(); // 2^15 / 2
        let vals_half = ArrayD::from_shape_vec(shape.clone(), vals_half).unwrap();
        let pad_vals_half = pad_to_pow_of_two(&vals_half, &AlmostGoldilocksField(0));
        let col_major_half: Vec<_> =
            pad_vals_half.view().reversed_axes().iter().cloned().collect();
        let half = Witness::new(
            flat_shape.clone(),
            col_major_half,
            DataType::Float,
            15,
            Role::Constant,
        );
        let half = self.public_literal(half);

        // B3. compute fixed-point 1 at sf=15. Witness::new stores raw
        // coefficients: integer 1 here would represent 2^-15, not one.
        let vals_one: Vec<AlmostGoldilocksField> =
            (0..val_num).map(|_| AlmostGoldilocksField(1 << 15)).collect();
        let vals_one = ArrayD::from_shape_vec(shape.clone(), vals_one).unwrap();
        let pad_vals_one = pad_to_pow_of_two(&vals_one, &AlmostGoldilocksField(0));
        let col_major_one: Vec<_> = pad_vals_one.view().reversed_axes().iter().cloned().collect();
        let one = Witness::new(
            flat_shape.clone(),
            col_major_one,
            DataType::Float,
            15,
            Role::Constant,
        );
        let one = self.public_literal(one);

        // B4. compute exp(r) by Taylor series
        r = self.change_shape(r, flat_shape);
        let r_square = self.einsum("a,a->a".to_string(), vec![r, r], true);
        self.integer_interval(r_square[0], 0, 8192);
        let r_one_sixth = self.einsum("a,a->a".to_string(), vec![r, one_sixth], true);
        self.integer_interval(r_one_sixth[0], -2731, 2731);
        let r_one_sixth_plus_half = self.add(r_one_sixth[0], half)[0];
        let deg_two_plus_deg_three = self.einsum(
            "a,a->a".to_string(),
            vec![r_one_sixth_plus_half, r_square[0]],
            true,
        );
        self.integer_interval(deg_two_plus_deg_three[0], 0, 4779);
        let deg_one_plus_deg_two_plus_deg_three =
            self.add(deg_two_plus_deg_three[0], r);
        let exp_r = self.add(deg_one_plus_deg_two_plus_deg_three[0], one)[0];

        // C. compute 2^(-k) * exp(r)
        let exp_x = self.einsum("a,a->a".to_string(), vec![two_pow_out, exp_r], false);
        let exp_x = self.scale(exp_x[0], 30, *SF_LOG)[0];
        // The signed bound remains a conservative integer-lifting certificate.
        // Logical outputs are nonnegative; the explicit output mask below
        // makes every padding coordinate exactly zero.
        self.integer_interval(exp_x, -(2i128 << *SF_LOG), 2i128 << *SF_LOG);
        // All internal products are <2^31. With these quotient bounds,
        // every ScaleDown residual interval is <2^33 << p, including the
        // final (30-sf)-bit divisor and sf>15 remainder conversion.
        let exp = self.change_shape(exp_x, shape.clone());
        let exp = if padded { self.mask(exp,shape) } else { exp };
        vec![exp]
    }

    /// Zero-pad X[C, H, W] → Y[C, H+2*pad_h, W+2*pad_w].
    /// Leading-dimension size for ops that act independently per channel.
    ///
    /// `[B, C, H, W]` is bit-identical to `[b_pad*c_pad, H, W]` in this layout
    /// -- W lowest, then H, then C, then B, each padded to a power of two --
    /// so a per-channel op batches by treating the batch as extra channels.
    /// No new arithmetic is involved. The extra channels hold zero for an
    /// honest witness and stay zero through any per-channel op, which is
    /// exactly what the MLE's padding already requires.
    ///
    /// Returns (channels to configure the block with, batch, logical C).
    fn per_channel_lead(x_shape: &[usize]) -> (usize, usize, usize) {
        assert!(
            x_shape.len() == 3 || x_shape.len() == 4,
            "expected [C, H, W] or [B, C, H, W], got {:?}", x_shape
        );
        if x_shape.len() == 4 {
            let (b, c) = (x_shape[0], x_shape[1]);
            (b.next_power_of_two() * c.next_power_of_two(), b, c)
        } else {
            (x_shape[0], 1, x_shape[0])
        }
    }

    /// Output shape for a per-channel op: the FOLDED form `[b_pad*c_pad, H, W]`,
    /// which is what these blocks already emit at run time.
    ///
    /// A batched CNN tensor is folded throughout: `[B, C, H, W]` and
    /// `[b_pad*c_pad, H, W]` are the same MLE over the same variables, so the
    /// per-channel blocks need no batch awareness at all. Only ops that must
    /// separate the two axes -- convolution, which indexes weights by C --
    /// recover the split, and they do it from the weight shape rather than
    /// from a convention the caller could get wrong.
    fn per_channel_out(channels: usize, h: usize, w: usize) -> Vec<usize> {
        vec![channels, h, w]
    }

    pub fn pad(&mut self, x: EdgeId, pad_h: usize, pad_w: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let (channels, _batch, _logical_c) = Self::per_channel_lead(x_shape);
        let off = x_shape.len() - 3;
        let input_h = x_shape[off + 1];
        let input_w = x_shape[off + 2];

        let h_out = input_h + 2 * pad_h;
        let w_out = input_w + 2 * pad_w;

        let pad_block = BasicBlockType::ZeroPad(ZeroPad::new(channels, input_h, input_w, pad_h, pad_w));
        let outs = self.add_gkr_node(vec![x], pad_block);
        self.init_values.push(Some(Witness::new_wo_data(
            Self::per_channel_out(channels, h_out, w_out),
            DataType::Uint,
            0,
            Role::Output,
        )));
        outs[0]
    }

    /// 2×2 max pooling: X[C, H, W] → Y[C, H/pool_h, W/pool_w].
    /// Composes: MaxPoolHelper → Replicate2x2 → Sub → NonNeg (dominance).
    pub fn maxpool2d(&mut self, x: EdgeId, pool_h: usize, pool_w: usize) -> EdgeId {
        use crate::basicblock::maxpool::{MaxPoolHelper, Replicate2x2};

        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let (channels, _batch, _logical_c) = Self::per_channel_lead(x_shape);
        let off = x_shape.len() - 3;
        let input_h = x_shape[off + 1];
        let input_w = x_shape[off + 2];

        assert_eq!(input_h % pool_h, 0, "H must be divisible by pool_h");
        assert_eq!(input_w % pool_w, 0, "W must be divisible by pool_w");

        let h_out = input_h / pool_h;
        let w_out = input_w / pool_w;
        let out_shape = Self::per_channel_out(channels, h_out, w_out);

        // 1. MaxPoolHelper(x) → y
        let maxpool = BasicBlockType::MaxPoolHelper(MaxPoolHelper {
            channels, input_h, input_w, pool_h, pool_w,
        });
        let y_outs = self.add_gkr_node(vec![x], maxpool);
        let y = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            out_shape, DataType::Uint, 0, Role::Output,
        )));

        // 2. Replicate2x2(y) → y_rep
        let replicate = BasicBlockType::Replicate2x2(Replicate2x2 {
            channels, out_h: input_h, out_w: input_w,
        });
        let rep_outs = self.add_gkr_node(vec![y], replicate);
        let y_rep = rep_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            Self::per_channel_out(channels, input_h, input_w),
            DataType::Uint,
            0,
            Role::Output,
        )));

        // 3. Dominance: sub(y_rep, x) → diff ≥ 0 (y ≥ every input in pool window)
        let diff_outs = self.sub(y_rep, x);
        let diff = diff_outs[0];
        self.add_nonneg_node(diff);

        y
    }

    /// Asymmetric zero-pad: X[C, H, W] → Y[C, H+pt+pb, W+pl+pr].
    pub fn pad_asym(&mut self, x: EdgeId, pad_h_top: usize, pad_h_bottom: usize, pad_w_left: usize, pad_w_right: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels = x_shape[0];
        let input_h = x_shape[1];
        let input_w = x_shape[2];

        let h_out = input_h + pad_h_top + pad_h_bottom;
        let w_out = input_w + pad_w_left + pad_w_right;

        let pad_block = BasicBlockType::ZeroPadAsym(ZeroPadAsym::new(
            channels, input_h, input_w, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right,
        ));
        let outs = self.add_gkr_node(vec![x], pad_block);
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, h_out, w_out],
            DataType::Uint,
            0,
            Role::Output,
        )));
        outs[0]
    }

    /// SubSample2D: X[C, H, W] → SV[C, out_h, out_w] at strided+offset positions.
    pub fn subsample2d(&mut self, x: EdgeId, stride_h: usize, stride_w: usize, offset_h: usize, offset_w: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels = x_shape[0];
        let input_h = x_shape[1];
        let input_w = x_shape[2];

        let ss = SubSample2D::new(channels, input_h, input_w, stride_h, stride_w, offset_h, offset_w);
        let out_h = ss.out_h;
        let out_w = ss.out_w;

        let block = BasicBlockType::SubSample2D(ss);
        let outs = self.add_gkr_node(vec![x], block);
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, out_h, out_w],
            DataType::Uint,
            0,
            Role::Output,
        )));
        outs[0]
    }

    /// Conv2D with stride: FlattenKernel(W) → W_flat, Conv2D(X, W_flat) → Y.
    /// x shape: [C_in, H_in, W_in], w shape: [C_out, C_in, kH, kW].
    pub fn conv2d_strided(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize), stride: (usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        // C_in comes from the WEIGHT, which is unambiguous, and the batch is
        // whatever leading extent is left over. A folded input
        // [b_pad*c_in_pad, H, W] and an explicit [B, C_in, H, W] both work, and
        // neither depends on a convention the caller has to remember.
        let c_in = w_shape[1];
        let lead: usize = x_shape[..x_shape.len() - 2].iter().product();
        let c_in_pad = c_in.next_power_of_two();
        // An unbatched tensor carries the LOGICAL channel count (RGB is 3, not
        // 4), so `lead == c_in` is the single-image case and need not divide
        // c_in_pad. Only a folded batch uses padded channel strides.
        let batch = if lead == c_in {
            1
        } else {
            assert!(
                lead % c_in_pad == 0,
                "conv2d: leading extent {} is neither C_in {} nor a multiple of \
                 padded C_in {} (X {:?}, W {:?})",
                lead, c_in, c_in_pad, x_shape, w_shape
            );
            lead / c_in_pad
        };
        let input_h = x_shape[x_shape.len() - 2];
        let input_w = x_shape[x_shape.len() - 1];
        let c_out = w_shape[0];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kh, "W kH must match kernel_size.0");
        assert_eq!(w_shape[3], kw, "W kW must match kernel_size.1");

        let w_pad = input_w.next_power_of_two();
        let s_kernel = (kh - 1) * w_pad + kw;

        // 1. FlattenKernel: W → W_flat
        let fk = BasicBlockType::FlattenKernel(FlattenKernel {
            s_w: w_pad, kh, kw, c_out, c_in, dilation_h: 1, dilation_w: 1,
        });
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, c_in, s_kernel],
            DataType::Uint, 0, Role::Output,
        )));

        // 2. Conv2D with stride: (X, W_flat) → [Y, Y_full]
        let h_out = (input_h - kh) / sh + 1;
        let w_out = (input_w - kw) / sw + 1;
        let conv = Conv2D::new_strided(c_in, c_out, kh, kw, input_h, input_w, sh, sw)
            .with_batch(batch);
        let s_full = conv.s_in + conv.s_kernel - 1;
        let junk_free = conv.junk_free();
        let gp_mode = conv.grand_product_mode();
        let k_len = conv.gp_k_len();
        let y_outs = self.add_gkr_node(vec![x, wf_edge], BasicBlockType::Conv2D(conv));
        let y_edge = y_outs[0];
        // Folded batch layout, matching what Conv2D::run emits.
        let with_b = |v: Vec<usize>| -> Vec<usize> {
            if batch > 1 {
                let mut o = vec![batch.next_power_of_two() * v[0].next_power_of_two()];
                o.extend_from_slice(&v[1..]);
                o
            } else { v }
        };
        self.init_values.push(Some(Witness::new_wo_data(
            with_b(vec![c_out, h_out, w_out]),
            DataType::Uint, 0, Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        // Skipped for junk-free 1×1 stride-1 convs (Y bound directly).
        if !junk_free {
            self.init_values.push(Some(Witness::new_wo_data(
                with_b(vec![c_out, s_full]),
                DataType::Uint, 0, Role::Auxiliary,
            )));
            // Grand-product output binding: a third aux K (leftover advice).
            if gp_mode {
                self.init_values.push(Some(Witness::new_wo_data(
                    vec![k_len],
                    DataType::Uint, 0, Role::Auxiliary,
                )));
            }
        }

        self.self_claim_edges.push(y_edge);
        vec![y_edge]
    }

    /// Conv2D with stride and dilation: FlattenKernel(W) → W_flat, Conv2D(X, W_flat) → Y.
    /// x shape: [C_in, H_in, W_in], w shape: [C_out, C_in, kH, kW].
    pub fn conv2d_dilated(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize), stride: (usize, usize), dilation: (usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        // C_in comes from the WEIGHT, which is unambiguous, and the batch is
        // whatever leading extent is left over. A folded input
        // [b_pad*c_in_pad, H, W] and an explicit [B, C_in, H, W] both work, and
        // neither depends on a convention the caller has to remember.
        let c_in = w_shape[1];
        let lead: usize = x_shape[..x_shape.len() - 2].iter().product();
        let c_in_pad = c_in.next_power_of_two();
        // An unbatched tensor carries the LOGICAL channel count (RGB is 3, not
        // 4), so `lead == c_in` is the single-image case and need not divide
        // c_in_pad. Only a folded batch uses padded channel strides.
        let batch = if lead == c_in {
            1
        } else {
            assert!(
                lead % c_in_pad == 0,
                "conv2d: leading extent {} is neither C_in {} nor a multiple of \
                 padded C_in {} (X {:?}, W {:?})",
                lead, c_in, c_in_pad, x_shape, w_shape
            );
            lead / c_in_pad
        };
        let input_h = x_shape[x_shape.len() - 2];
        let input_w = x_shape[x_shape.len() - 1];
        let c_out = w_shape[0];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (dh, dw) = dilation;
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kh, "W kH must match kernel_size.0");
        assert_eq!(w_shape[3], kw, "W kW must match kernel_size.1");

        let w_pad = input_w.next_power_of_two();
        let s_kernel = (kh - 1) * dh * w_pad + (kw - 1) * dw + 1;

        // 1. FlattenKernel with dilation: W → W_flat
        let fk = BasicBlockType::FlattenKernel(FlattenKernel {
            s_w: w_pad, kh, kw, c_out, c_in, dilation_h: dh, dilation_w: dw,
        });
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, c_in, s_kernel],
            DataType::Uint, 0, Role::Output,
        )));

        // 2. Conv2D with stride + dilation: (X, W_flat) → [Y, Y_full]
        let conv = Conv2D::new_dilated(c_in, c_out, kh, kw, input_h, input_w, sh, sw, dh, dw)
            .with_batch(batch);
        let h_out = conv.h_out;
        let w_out = conv.w_out;
        let s_full = conv.s_in + conv.s_kernel - 1;
        let junk_free = conv.junk_free();
        let gp_mode = conv.grand_product_mode();
        let k_len = conv.gp_k_len();
        let conv_block = BasicBlockType::Conv2D(conv);
        let y_outs = self.add_gkr_node(vec![x, wf_edge], conv_block);
        let y_edge = y_outs[0];
        // Folded batch layout, matching what Conv2D::run emits.
        let with_b = |v: Vec<usize>| -> Vec<usize> {
            if batch > 1 {
                let mut o = vec![batch.next_power_of_two() * v[0].next_power_of_two()];
                o.extend_from_slice(&v[1..]);
                o
            } else { v }
        };
        self.init_values.push(Some(Witness::new_wo_data(
            with_b(vec![c_out, h_out, w_out]),
            DataType::Uint, 0, Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        // Skipped for junk-free 1×1 stride-1 convs (Y bound directly).
        if !junk_free {
            self.init_values.push(Some(Witness::new_wo_data(
                with_b(vec![c_out, s_full]),
                DataType::Uint, 0, Role::Auxiliary,
            )));
            // Grand-product output binding: a third aux K (leftover advice).
            if gp_mode {
                self.init_values.push(Some(Witness::new_wo_data(
                    vec![k_len],
                    DataType::Uint, 0, Role::Auxiliary,
                )));
            }
        }

        self.self_claim_edges.push(y_edge);
        vec![y_edge]
    }

    /// SubSample2D with explicit output size.
    pub(crate) fn subsample2d_sized(&mut self, x: EdgeId, stride_h: usize, stride_w: usize, offset_h: usize, offset_w: usize, out_h: usize, out_w: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let (channels, _batch, _logical_c) = Self::per_channel_lead(x_shape);
        let off = x_shape.len() - 3;
        let input_h = x_shape[off + 1];
        let input_w = x_shape[off + 2];

        let ss = SubSample2D::new_with_output_size(channels, input_h, input_w, stride_h, stride_w, offset_h, offset_w, out_h, out_w);

        let block = BasicBlockType::SubSample2D(ss);
        let outs = self.add_gkr_node(vec![x], block);
        self.init_values.push(Some(Witness::new_wo_data(
            Self::per_channel_out(channels, out_h, out_w),
            DataType::Uint, 0, Role::Output,
        )));
        outs[0]
    }

    /// General max pooling: X[C, H, W] → Y[C, H_out, W_out] with arbitrary kernel and stride.
    /// Uses GeneralMaxPoolHelper advice node + SubSample2D + Sub + NonNeg for dominance.
    pub fn maxpool_general(&mut self, x: EdgeId, kernel_h: usize, kernel_w: usize, stride_h: usize, stride_w: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let (channels, _batch, _logical_c) = Self::per_channel_lead(x_shape);
        let off = x_shape.len() - 3;
        let input_h = x_shape[off + 1];
        let input_w = x_shape[off + 2];

        let out_h = (input_h - kernel_h) / stride_h + 1;
        let out_w = (input_w - kernel_w) / stride_w + 1;
        let out_shape = Self::per_channel_out(channels, out_h, out_w);

        // 1. Advice node: compute Y = maxpool(X)
        let gmp = BasicBlockType::GeneralMaxPoolHelper(GeneralMaxPoolHelper {
            channels, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w,
        });
        let y_outs = self.add_gkr_node(vec![x], gmp);
        let y = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            out_shape, DataType::Uint, 0, Role::Output,
        )));

        // 2. Dominance: for each kernel position, prove Y >= SubSample(X, stride, offset)
        for kh in 0..kernel_h {
            for kw in 0..kernel_w {
                let sv = self.subsample2d_sized(x, stride_h, stride_w, kh, kw, out_h, out_w);
                let diff = self.sub(y, sv);
                self.add_nonneg_node(diff[0]);
            }
        }

        y
    }

    /// ReLU: y = max(0, x).
    /// Decomposes as: neg = max(0, -x) (advice), y = x + neg (Add), with range checks on y and neg,
    /// plus ProductZeroCheck proving neg · y = 0 (complementary slackness).
    pub fn relu(&mut self, x: EdgeId) -> EdgeId {
        let x_witness = self.init_values[x].as_ref().unwrap();
        let shape = x_witness.shape.clone();
        let sf = x_witness.sf;
        let data_type = x_witness.data_type;

        // 1. ReLUHelper(x) → neg (advice: neg = max(0, -x))
        let relu_helper = BasicBlockType::ReLUHelper(relu::ReLUHelper);
        let neg_outs = self.add_gkr_node(vec![x], relu_helper);
        let neg = neg_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            shape.clone(), data_type, sf, Role::Output,
        )));

        // 2. Add(x, neg) → y (y = x + neg = max(0, x))
        let y = self.add(x, neg)[0];

        // 3. NonNegative(y) — range check: y ≥ 0
        self.add_nonneg_node(y);

        // 4. NonNegative(neg) — range check: neg ≥ 0
        self.add_nonneg_node(neg);

        // 5. ProductZeroCheck(neg, y) → cert (proves neg · y = 0 pointwise)
        let pzc = BasicBlockType::ProductZeroCheck(relu::ProductZeroCheck);
        let cert_outs = self.add_gkr_node(vec![neg, y], pzc);
        let cert = cert_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(shape, DataType::Uint, 0, Role::Output)));
        self.self_claim_edges.push(cert);

        y
    }

    /// Logical reduction inputs have explicit signed lifts before summation.
    /// Padding is authenticated and excluded before, rather than after, a mean.
    pub(crate) fn mean_input(&mut self, x: EdgeId, count: usize) -> EdgeId {
        assert!((1..=(1<<20)).contains(&count), "unsupported mean reduction size");
        let shape=self.init_values[x].as_ref().unwrap().shape.clone();
        let x=self.mask(x,shape);
        self.integer_interval(x,-((1i128<<30)-1),(1i128<<30)-1);
        x
    }

    /// ReduceMean over one or more axes.
    /// E.g. reduce_mean(x, &[1, 2]) on shape [B, S, D] computes mean over axes 1,2 → shape [B].
    pub fn reduce_mean(&mut self, x: EdgeId, axes: &[usize]) -> EdgeId {
        let shape = self.init_values[x].as_ref().unwrap().shape.clone();
        let ndim = shape.len();
        assert!(!axes.is_empty(), "axes must be non-empty");
        for &ax in axes {
            assert!(ax < ndim, "axis {ax} out of bounds for {ndim}-dim tensor");
        }

        assert!(axes.iter().collect::<std::collections::HashSet<_>>().len()==axes.len(), "duplicate mean axis");
        let n=axes.iter().try_fold(1usize,|n,&ax|n.checked_mul(shape[ax])).expect("mean reduction size overflow");
        let x=self.mean_input(x,n);

        let in_chars: String = (0..ndim).map(|i| (b'a' + i as u8) as char).collect();
        let out_chars: String = (0..ndim)
            .filter(|i| !axes.contains(i))
            .map(|i| (b'a' + i as u8) as char)
            .collect();
        let equation = format!("{}->{}", in_chars, out_chars);

        let x_sum = self.einsum(equation, vec![x], false)[0];
        let x_mean = self.div_const(x_sum, n)[0];
        x_mean
    }

    /// Create a conv1d subgraph: Conv1D(X, W) → Y.
    /// x shape: [C_in, L_in], w shape: [C_out, C_in, K].
    /// Returns vec![y_edge].
    pub fn conv1d(&mut self, x: EdgeId, w: EdgeId, kernel_size: usize) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_len = x_shape[1];
        let c_out = w_shape[0];
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kernel_size, "W K must match kernel_size");

        let l_out = input_len - kernel_size + 1;

        // Conv1D: (X, W) → [Y, Y_full]  (no FlattenKernel needed)
        let conv = Conv1D::new(c_in, c_out, kernel_size, input_len);
        let s_full = conv.s_in + conv.s_kernel - 1;
        let y_outs = self.add_gkr_node(vec![x, w], BasicBlockType::Conv1D(conv));
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, l_out],
            DataType::Uint,
            0,
            Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, s_full],
            DataType::Uint,
            0,
            Role::Auxiliary,
        )));

        // Track Conv1D output as self-claim edge
        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// Create a strided conv1d subgraph: Conv1D(X, W) → Y.
    /// x shape: [C_in, L_in], w shape: [C_out, C_in, K].
    /// Returns vec![y_edge].
    pub fn conv1d_strided(&mut self, x: EdgeId, w: EdgeId, kernel_size: usize, stride: usize) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_len = x_shape[1];
        let c_out = w_shape[0];
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kernel_size, "W K must match kernel_size");

        let l_out = (input_len - kernel_size) / stride + 1;

        let conv = Conv1D::new_strided(c_in, c_out, kernel_size, input_len, stride);
        let s_full = conv.s_in + conv.s_kernel - 1;
        let y_outs = self.add_gkr_node(vec![x, w], BasicBlockType::Conv1D(conv));
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, l_out],
            DataType::Uint,
            0,
            Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, s_full],
            DataType::Uint,
            0,
            Role::Auxiliary,
        )));

        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// 1D zero-padding: X[C, L] → Y[C, L + pad_left + pad_right].
    /// Implemented via reshape to 3D → pad_asym → reshape back.
    pub fn pad1d(&mut self, x: EdgeId, pad_left: usize, pad_right: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        assert_eq!(x_shape.len(), 2, "pad1d expects [C, L] input");
        let channels = x_shape[0];
        let input_len = x_shape[1];

        // X[C, L] → X[C, 1, L]
        let x_3d = self.change_shape(x, vec![channels, 1, input_len]);

        // pad_asym with no height padding, only width padding
        let padded = self.pad_asym(x_3d, 0, 0, pad_left, pad_right);

        // X[C, 1, L+pl+pr] → X[C, L+pl+pr]
        let out_len = input_len + pad_left + pad_right;
        self.change_shape(padded, vec![channels, out_len])
    }

    /// Create a conv2d subgraph: FlattenKernel(W) → W_flat, Conv2D(X, W_flat) → Y.
    /// x shape: [C_in, H_in, W_in] or [B, C_in, H_in, W_in]; w shape:
    /// [C_out, C_in, kH, kW]. Returns vec![y_edge].
    ///
    /// A 4-D X means a batch of images sharing one set of weights. That is ONE
    /// conv node over a wider tensor, not `B` replicated nodes: the batch index
    /// is bound by the verifier rather than summed, so the leaf count is the
    /// same as for a single image. Replicating the subgraph per image instead
    /// multiplies the commitment leaves, which makes the fold tree fragment
    /// into low-arity buckets and costs superlinearly.
    pub fn conv2d(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        // C_in comes from the WEIGHT, which is unambiguous, and the batch is
        // whatever leading extent is left over. A folded input
        // [b_pad*c_in_pad, H, W] and an explicit [B, C_in, H, W] both work, and
        // neither depends on a convention the caller has to remember.
        let c_in = w_shape[1];
        let lead: usize = x_shape[..x_shape.len() - 2].iter().product();
        let c_in_pad = c_in.next_power_of_two();
        // An unbatched tensor carries the LOGICAL channel count (RGB is 3, not
        // 4), so `lead == c_in` is the single-image case and need not divide
        // c_in_pad. Only a folded batch uses padded channel strides.
        let batch = if lead == c_in {
            1
        } else {
            assert!(
                lead % c_in_pad == 0,
                "conv2d: leading extent {} is neither C_in {} nor a multiple of \
                 padded C_in {} (X {:?}, W {:?})",
                lead, c_in, c_in_pad, x_shape, w_shape
            );
            lead / c_in_pad
        };
        let input_h = x_shape[x_shape.len() - 2];
        let input_w = x_shape[x_shape.len() - 1];
        let c_out = w_shape[0];
        let (kh, kw) = kernel_size;
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kh, "W kH must match kernel_size.0");
        assert_eq!(w_shape[3], kw, "W kW must match kernel_size.1");

        let w_pad = input_w.next_power_of_two();
        let s_kernel = (kh - 1) * w_pad + kw;

        // 1. FlattenKernel: W → W_flat (uses w_pad stride for MLE compatibility)
        let fk = BasicBlockType::FlattenKernel(FlattenKernel {
            s_w: w_pad,
            kh,
            kw,
            c_out,
            c_in,
            dilation_h: 1,
            dilation_w: 1,
        });
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, c_in, s_kernel],
            DataType::Uint,
            0,
            Role::Output,
        )));

        // 2. Conv2D: (X, W_flat) → [Y, Y_full]
        let h_out = input_h - kh + 1;
        let w_out = input_w - kw + 1;
        let conv = Conv2D::new(c_in, c_out, kh, kw, input_h, input_w).with_batch(batch);
        let s_full = conv.s_in + conv.s_kernel - 1;
        let junk_free = conv.junk_free();
        let gp_mode = conv.grand_product_mode();
        let k_len = conv.gp_k_len();
        let y_outs = self.add_gkr_node(vec![x, wf_edge], BasicBlockType::Conv2D(conv));
        let y_edge = y_outs[0];
        // Batch stays the most significant dimension all the way through, so
        // Y and Y_full carry it exactly where X did.
        // Folded batch layout, matching what Conv2D::run emits.
        let with_b = |v: Vec<usize>| -> Vec<usize> {
            if batch > 1 {
                let mut o = vec![batch.next_power_of_two() * v[0].next_power_of_two()];
                o.extend_from_slice(&v[1..]);
                o
            } else { v }
        };
        self.init_values.push(Some(Witness::new_wo_data(
            with_b(vec![c_out, h_out, w_out]),
            DataType::Uint,
            0,
            Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        // Skipped for junk-free 1×1 stride-1 convs (Y bound directly).
        if !junk_free {
            self.init_values.push(Some(Witness::new_wo_data(
                with_b(vec![c_out, s_full]),
                DataType::Uint,
                0,
                Role::Auxiliary,
            )));
            // Grand-product output binding: a third aux K (leftover advice).
            if gp_mode {
                self.init_values.push(Some(Witness::new_wo_data(
                    vec![k_len],
                    DataType::Uint,
                    0,
                    Role::Auxiliary,
                )));
            }
        }

        // Track Conv2D output as self-claim edge
        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// Create a conv3d subgraph: FlattenKernel3D(W) → W_flat, Conv3D(X, W_flat) → Y.
    /// x shape: [C_in, D_in, H_in, W_in], w shape: [C_out, C_in, kD, kH, kW].
    /// Returns vec![y_edge].
    pub fn conv3d(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_d = x_shape[1];
        let input_h = x_shape[2];
        let input_w = x_shape[3];
        let c_out = w_shape[0];
        let (kd, kh, kw) = kernel_size;
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kd, "W kD must match kernel_size.0");
        assert_eq!(w_shape[3], kh, "W kH must match kernel_size.1");
        assert_eq!(w_shape[4], kw, "W kW must match kernel_size.2");

        let w_pad = input_w.next_power_of_two();
        let h_pad = input_h.next_power_of_two();
        let stride_w = w_pad;
        let stride_h = h_pad * w_pad;
        let s_kernel = (kd - 1) * stride_h + (kh - 1) * stride_w + kw;

        // 1. FlattenKernel3D: W → W_flat
        let fk = BasicBlockType::FlattenKernel3D(FlattenKernel3D {
            stride_h,
            stride_w,
            kd,
            kh,
            kw,
            c_out,
            c_in,
        });
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, c_in, s_kernel],
            DataType::Uint,
            0,
            Role::Output,
        )));

        // 2. Conv3D: (X, W_flat) → [Y, Y_full]
        let d_out = input_d - kd + 1;
        let h_out = input_h - kh + 1;
        let w_out = input_w - kw + 1;
        let conv = Conv3D::new(c_in, c_out, kd, kh, kw, input_d, input_h, input_w);
        let s_full = conv.s_in + conv.s_kernel - 1;
        let y_outs = self.add_gkr_node(vec![x, wf_edge], BasicBlockType::Conv3D(conv));
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, d_out, h_out, w_out],
            DataType::Uint,
            0,
            Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, s_full],
            DataType::Uint,
            0,
            Role::Auxiliary,
        )));

        // Track Conv3D output as self-claim edge
        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// Conv3D with stride: FlattenKernel3D(W) → W_flat, Conv3D(X, W_flat) → Y.
    /// x shape: [C_in, D_in, H_in, W_in], w shape: [C_out, C_in, kD, kH, kW].
    pub fn conv3d_strided(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_d = x_shape[1];
        let input_h = x_shape[2];
        let input_w = x_shape[3];
        let c_out = w_shape[0];
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        assert_eq!(w_shape[1], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kd, "W kD must match kernel_size.0");
        assert_eq!(w_shape[3], kh, "W kH must match kernel_size.1");
        assert_eq!(w_shape[4], kw, "W kW must match kernel_size.2");

        let w_pad = input_w.next_power_of_two();
        let h_pad = input_h.next_power_of_two();
        let stride_w_flat = w_pad;
        let stride_h_flat = h_pad * w_pad;
        let s_kernel = (kd - 1) * stride_h_flat + (kh - 1) * stride_w_flat + kw;

        // 1. FlattenKernel3D: W → W_flat
        let fk = BasicBlockType::FlattenKernel3D(FlattenKernel3D {
            stride_h: stride_h_flat,
            stride_w: stride_w_flat,
            kd, kh, kw, c_out, c_in,
        });
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, c_in, s_kernel],
            DataType::Uint, 0, Role::Output,
        )));

        // 2. Conv3D with stride: (X, W_flat) → [Y, Y_full]
        let d_out = (input_d - kd) / sd + 1;
        let h_out = (input_h - kh) / sh + 1;
        let w_out = (input_w - kw) / sw + 1;
        let conv = Conv3D::new_strided(c_in, c_out, kd, kh, kw, input_d, input_h, input_w, sd, sh, sw);
        let s_full = conv.s_in + conv.s_kernel - 1;
        let y_outs = self.add_gkr_node(vec![x, wf_edge], BasicBlockType::Conv3D(conv));
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, d_out, h_out, w_out],
            DataType::Uint, 0, Role::Output,
        )));
        // Aux Y_full: full 1D conv, committed advice (Role::Auxiliary).
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, s_full],
            DataType::Uint, 0, Role::Auxiliary,
        )));

        self.self_claim_edges.push(y_edge);
        vec![y_edge]
    }

    /// Create a conv_transpose1d subgraph: ConvTranspose1D(X, W) → Y.
    /// x shape: [C_in, L_in], w shape: [C_in, C_out, K].
    /// Returns vec![y_edge].
    pub fn conv_transpose1d(&mut self, x: EdgeId, w: EdgeId, kernel_size: usize, stride: usize) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_len = x_shape[1];
        let c_out = w_shape[1];
        assert_eq!(w_shape[0], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kernel_size, "W K must match kernel_size");

        let l_out = (input_len - 1) * stride + kernel_size;

        let conv = BasicBlockType::ConvTranspose1D(ConvTranspose1D::new(c_in, c_out, kernel_size, input_len, stride));
        let y_outs = self.add_gkr_node(vec![x, w], conv);
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, l_out],
            DataType::Uint,
            0,
            Role::Output,
        )));

        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// Create a conv_transpose2d subgraph: FlattenKernel(W) → W_flat, ConvTranspose2D(X, W_flat) → Y.
    /// x shape: [C_in, H_in, W_in], w shape: [C_in, C_out, kH, kW].
    /// Returns vec![y_edge].
    pub fn conv_transpose2d(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize), stride: (usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_h = x_shape[1];
        let input_w = x_shape[2];
        let c_out = w_shape[1];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        assert_eq!(w_shape[0], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kh, "W kH must match kernel_size.0");
        assert_eq!(w_shape[3], kw, "W kW must match kernel_size.1");

        let h_out = (input_h - 1) * sh + kh;
        let w_out = (input_w - 1) * sw + kw;
        let w_out_pad = w_out.next_power_of_two();

        // FlattenKernel: channels swapped (c_out=c_in, c_in=c_out), s_w=W_out_pad
        let fk = BasicBlockType::FlattenKernel(FlattenKernel {
            s_w: w_out_pad,
            kh,
            kw,
            c_out: c_in, // swapped
            c_in: c_out, // swapped
            dilation_h: 1,
            dilation_w: 1,
        });
        let s_kernel = (kh - 1) * w_out_pad + kw;
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_in, c_out, s_kernel],
            DataType::Uint,
            0,
            Role::Output,
        )));

        let conv = BasicBlockType::ConvTranspose2D(ConvTranspose2D::new(
            c_in, c_out, kh, kw, input_h, input_w, sh, sw,
        ));
        let y_outs = self.add_gkr_node(vec![x, wf_edge], conv);
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, h_out, w_out],
            DataType::Uint,
            0,
            Role::Output,
        )));

        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// Create a conv_transpose3d subgraph: FlattenKernel3D(W) → W_flat, ConvTranspose3D(X, W_flat) → Y.
    /// x shape: [C_in, D_in, H_in, W_in], w shape: [C_in, C_out, kD, kH, kW].
    /// Returns vec![y_edge].
    pub fn conv_transpose3d(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let c_in = x_shape[0];
        let input_d = x_shape[1];
        let input_h = x_shape[2];
        let input_w = x_shape[3];
        let c_out = w_shape[1];
        let (kd, kh, kw) = kernel_size;
        let (sd, sh, sw) = stride;
        assert_eq!(w_shape[0], c_in, "W c_in must match X c_in");
        assert_eq!(w_shape[2], kd, "W kD must match kernel_size.0");
        assert_eq!(w_shape[3], kh, "W kH must match kernel_size.1");
        assert_eq!(w_shape[4], kw, "W kW must match kernel_size.2");

        let d_out = (input_d - 1) * sd + kd;
        let h_out = (input_h - 1) * sh + kh;
        let w_out = (input_w - 1) * sw + kw;
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();

        // FlattenKernel3D: channels swapped
        let fk = BasicBlockType::FlattenKernel3D(FlattenKernel3D {
            stride_h: h_out_pad * w_out_pad,
            stride_w: w_out_pad,
            kd,
            kh,
            kw,
            c_out: c_in, // swapped
            c_in: c_out, // swapped
        });
        let s_kernel = (kd - 1) * h_out_pad * w_out_pad + (kh - 1) * w_out_pad + kw;
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_in, c_out, s_kernel],
            DataType::Uint,
            0,
            Role::Output,
        )));

        let conv = BasicBlockType::ConvTranspose3D(ConvTranspose3D::new(
            c_in, c_out, kd, kh, kw, input_d, input_h, input_w, sd, sh, sw,
        ));
        let y_outs = self.add_gkr_node(vec![x, wf_edge], conv);
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![c_out, d_out, h_out, w_out],
            DataType::Uint,
            0,
            Role::Output,
        )));

        self.self_claim_edges.push(y_edge);

        vec![y_edge]
    }

    /// ZeroPad3D: X[C, D, H, W] → Y[C, D+2p_d, H+2p_h, W+2p_w].
    pub fn pad3d(&mut self, x: EdgeId, pad_d: usize, pad_h: usize, pad_w: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels = x_shape[0];
        let input_d = x_shape[1];
        let input_h = x_shape[2];
        let input_w = x_shape[3];

        let d_out = input_d + 2 * pad_d;
        let h_out = input_h + 2 * pad_h;
        let w_out = input_w + 2 * pad_w;

        let pad_block = BasicBlockType::ZeroPad3D(ZeroPad3D::new(
            channels, input_d, input_h, input_w, pad_d, pad_h, pad_w,
        ));
        let outs = self.add_gkr_node(vec![x], pad_block);
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, d_out, h_out, w_out],
            DataType::Uint, 0, Role::Output,
        )));
        outs[0]
    }

    /// Concat: concatenate two tensors along the channel axis (equal-size).
    /// A[C, ...spatial...] + B[C, ...spatial...] → Y[2C, ...spatial...]
    pub fn concat(&mut self, a: EdgeId, b: EdgeId) -> EdgeId {
        let a_shape = &self.init_values[a].as_ref().unwrap().shape;
        let b_shape = &self.init_values[b].as_ref().unwrap().shape;
        assert_eq!(a_shape.len(), b_shape.len(), "Concat inputs must have same dimensionality");
        let channels_a = a_shape[0];
        assert_eq!(channels_a, b_shape[0], "Concat inputs must have equal channels");
        let spatial_dims: Vec<usize> = a_shape[1..].to_vec();
        for i in 1..a_shape.len() {
            assert_eq!(a_shape[i], b_shape[i], "Concat spatial dim {} mismatch", i);
        }

        let c_out = 2 * channels_a;
        let cat = BasicBlockType::Concat(Concat {
            channels_a, spatial_dims: spatial_dims.clone(),
        });
        let outs = self.add_gkr_node(vec![a, b], cat);
        let mut out_shape = vec![c_out];
        out_shape.extend_from_slice(&spatial_dims);
        self.init_values.push(Some(Witness::new_wo_data(
            out_shape.clone(), DataType::Uint, 0, Role::Output,
        )));
        outs[0]
    }

    /// ChannelSlice: extract channels [start, start+count) from X.
    /// X[C_in, ...spatial...] → Y[count, ...spatial...]
    pub fn channel_slice(&mut self, x: EdgeId, start: usize, count: usize) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels_in = x_shape[0];
        let spatial_dims: Vec<usize> = x_shape[1..].to_vec();
        assert!(start + count <= channels_in, "ChannelSlice out of bounds: start={start} count={count} c_in={channels_in}");
        assert!(count.is_power_of_two(), "ChannelSlice count must be power of 2, got {count}");
        assert!(channels_in.is_power_of_two(), "ChannelSlice c_in must be power of 2, got {channels_in}");
        assert!(start % count == 0, "ChannelSlice start must be aligned to count: start={start} count={count}");

        // Propagate the input's build-time fixed-point scale: ChannelSlice is a
        // pure gather (preserves scale), and downstream consumers — notably the
        // InstanceNorm einsum `scale_back`, which reads its input[0]'s build-time
        // sf to size the rescale — need the correct sf here, not 0.
        let in_sf = self.init_values[x].as_ref().unwrap().sf;
        let in_dt = self.init_values[x].as_ref().unwrap().data_type;
        let block = BasicBlockType::ChannelSlice(ChannelSlice {
            channels_in, channels_out: count, channel_start: start,
            spatial_dims: spatial_dims.clone(),
        });
        let outs = self.add_gkr_node(vec![x], block);
        let mut out_shape = vec![count];
        out_shape.extend_from_slice(&spatial_dims);
        self.init_values.push(Some(Witness::new_wo_data(
            out_shape, in_dt, in_sf, Role::Output,
        )));
        outs[0]
    }

    /// DepthwiseConv2D with stride: FlattenKernel(W, c_in=1) → W_flat, DepthwiseConv2D(X, W_flat) → Y.
    /// x shape: [C, H_in, W_in], w shape: [C, 1, kH, kW].
    pub fn depthwise_conv2d_strided(&mut self, x: EdgeId, w: EdgeId, kernel_size: (usize, usize), stride: (usize, usize)) -> Vec<EdgeId> {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let w_shape = &self.init_values[w].as_ref().unwrap().shape;

        let channels = x_shape[0];
        let input_h = x_shape[1];
        let input_w = x_shape[2];
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        assert_eq!(w_shape[0], channels, "W channels must match X channels");
        assert_eq!(w_shape[1], 1, "Depthwise conv W must have c_in=1");
        assert_eq!(w_shape[2], kh, "W kH must match kernel_size.0");
        assert_eq!(w_shape[3], kw, "W kW must match kernel_size.1");

        let w_pad = input_w.next_power_of_two();
        let s_kernel = (kh - 1) * w_pad + kw;

        // 1. FlattenKernel: W[C, 1, kH, kW] → W_flat[C, 1, S_kernel]
        //    c_out=C, c_in=1 in FlattenKernel terms
        let fk = BasicBlockType::FlattenKernel(FlattenKernel {
            s_w: w_pad, kh, kw, c_out: channels, c_in: 1, dilation_h: 1, dilation_w: 1,
        });
        let wf_outs = self.add_gkr_node(vec![w], fk);
        let wf_edge = wf_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, 1, s_kernel],
            DataType::Uint, 0, Role::Output,
        )));

        // 2. ChangeShape: W_flat[C, 1, S_kernel] → W_flat2[C, S_kernel]
        //    Remove the trivial c_in=1 dimension for DepthwiseConv2D
        let wf2 = self.change_shape(wf_edge, vec![channels, s_kernel]);

        // 3. DepthwiseConv2D: (X, W_flat2) → [Y, Y_full]
        let h_out = (input_h - kh) / sh + 1;
        let w_out = (input_w - kw) / sw + 1;
        let conv = DepthwiseConv2D::new_strided(
            channels, kh, kw, input_h, input_w, sh, sw,
        );
        let s_full = conv.s_in + conv.s_kernel - 1;
        let y_outs = self.add_gkr_node(vec![x, wf2], BasicBlockType::DepthwiseConv2D(conv));
        let y_edge = y_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, h_out, w_out],
            DataType::Uint, 0, Role::Output,
        )));
        // Aux Y_full: per-channel full 1D conv, committed advice (Role::Auxiliary).
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, s_full],
            DataType::Uint, 0, Role::Auxiliary,
        )));

        self.self_claim_edges.push(y_edge);
        vec![y_edge]
    }

    /// General concat: concatenate two tensors with potentially unequal channels.
    /// A[C_a, ...] + B[C_b, ...] → Y[C_a+C_b, ...]
    /// Uses ChangeShape + equal Concat + ChangeShape if channels differ.
    pub fn general_concat(&mut self, a: EdgeId, b: EdgeId) -> EdgeId {
        let a_shape = self.init_values[a].as_ref().unwrap().shape.clone();
        let b_shape = self.init_values[b].as_ref().unwrap().shape.clone();
        let c_a = a_shape[0];
        let c_b = b_shape[0];

        if c_a == c_b {
            return self.concat(a, b);
        }

        // Pad both to max channel count, then equal-concat.
        // Output shape is [2*c_max, spatial] where c_max = max(c_a,c_b).next_power_of_two().
        // We do NOT trim because non-power-of-2 trimming via ChangeShape leaves stale
        // data in the MLE polynomial. Extra channels are zero-padded by ChangeShape.
        let c_max = c_a.max(c_b).next_power_of_two();
        let spatial_a: Vec<usize> = a_shape[1..].to_vec();

        let a_padded = if c_a < c_max {
            let mut new_shape = vec![c_max];
            new_shape.extend_from_slice(&spatial_a);
            self.change_shape(a, new_shape)
        } else { a };

        let b_padded = if c_b < c_max {
            let spatial_b: Vec<usize> = b_shape[1..].to_vec();
            let mut new_shape = vec![c_max];
            new_shape.extend_from_slice(&spatial_b);
            self.change_shape(b, new_shape)
        } else { b };

        self.concat(a_padded, b_padded)
    }

    /// Multi-way concat: pairwise tree reduction to minimize unequal concats.
    /// For N equal-size inputs, produces exactly N channels (no padding waste).
    pub fn multi_concat(&mut self, edges: Vec<EdgeId>) -> EdgeId {
        assert!(!edges.is_empty(), "multi_concat needs at least 1 edge");
        if edges.len() == 1 {
            return edges[0];
        }
        // Tree reduction: concat pairs at each level
        let mut current = edges;
        while current.len() > 1 {
            let mut next = Vec::new();
            let mut i = 0;
            while i + 1 < current.len() {
                next.push(self.general_concat(current[i], current[i + 1]));
                i += 2;
            }
            if i < current.len() {
                next.push(current[i]); // odd one out
            }
            current = next;
        }
        current[0]
    }

    /// SiLU activation: x * sigmoid(x).
    pub fn silu(&mut self, x: EdgeId) -> EdgeId {
        let sig = self.sigmoid(x)[0];
        let x_shape = self.init_values[x].as_ref().unwrap().shape.clone();
        let letters: String = (0..x_shape.len()).map(|i| (b'a' + i as u8) as char).collect();
        let equation = format!("{},{}->{}", letters, letters, letters);
        self.einsum(equation, vec![x, sig], true)[0]
    }

    /// Nearest-neighbor 2x upsample: X[C, H, W] → Y[C, 2H, 2W].
    /// Uses Replicate2x2 block.
    pub fn upsample_nearest_2x(&mut self, x: EdgeId) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels = x_shape[0];
        let h = x_shape[1];
        let w = x_shape[2];
        let out_h = 2 * h;
        let out_w = 2 * w;

        let replicate = BasicBlockType::Replicate2x2(Replicate2x2 {
            channels, out_h, out_w,
        });
        let outs = self.add_gkr_node(vec![x], replicate);
        self.init_values.push(Some(Witness::new_wo_data(
            vec![channels, out_h, out_w],
            DataType::Uint, 0, Role::Output,
        )));
        outs[0]
    }

    /// InstanceNorm3D: X[C,D,H,W], gamma[C], beta[C] → Y[C,D,H,W].
    /// Decomposed: InstanceNormHelper (advice) → packed[2,C] (scale+offset),
    /// then Y = Einsum("c,cdhw->cdhw", scale, X) + offset (proven).
    pub fn instancenorm3d(&mut self, x: EdgeId, gamma: EdgeId, beta: EdgeId, eps: f64) -> EdgeId {
        let x_shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels = x_shape[0];
        let depth = x_shape[1];
        let height = x_shape[2];
        let width = x_shape[3];
        let sf = self.init_values[x].as_ref().unwrap().sf;
        let data_type = self.init_values[x].as_ref().unwrap().data_type;

        // 1. InstanceNormHelper(X, gamma, beta) → packed[2, C] (advice, single output)
        //    Group 0 = scale, Group 1 = offset
        let norm = BasicBlockType::InstanceNorm3D(InstanceNormHelper {
            channels, depth, height, width, eps,
        });
        let helper_outs = self.add_gkr_node(vec![x, gamma, beta], norm);
        let packed = helper_outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            vec![2, channels], data_type, sf, Role::Output,
        )));

        // 2. Unpack via ChannelSlice: packed[2, C] → scale_1c[1, C] → scale[C]
        let scale_1c = self.channel_slice(packed, 0, 1);
        let scale = self.change_shape(scale_1c, vec![channels]);

        // 3. Unpack via ChannelSlice: packed[2, C] → offset_1c[1, C] → offset[C]
        let offset_1c = self.channel_slice(packed, 1, 1);
        let offset = self.change_shape(offset_1c, vec![channels]);

        // 4. z = Einsum("a,abcd->abcd", scale, X) — proven multiply
        let z = self.einsum("a,abcd->abcd".to_string(), vec![scale, x], true)[0];

        // 5. Y = Add(z, offset_4d) — proven broadcast add
        //    offset has shape [C], but broadcast_shape right-aligns to [1,1,1,C].
        //    We need left-alignment [C,1,1,1] so channel dim broadcasts correctly.
        let offset_4d = self.change_shape(offset, vec![channels, 1, 1, 1]);
        let y = self.add(z, offset_4d)[0];

        // 6. Mask padding channels if non-power-of-2 channel count.
        //    Add broadcasting may introduce non-zero values in padding region.
        let y = self.mask_channels(y, 4);

        y
    }

    /// Mask non-power-of-2 channel dimensions by multiplying with a binary mask.
    /// mask[c] = 1 for c < actual_channels, mask[c] = 0 for c >= actual_channels.
    /// This ensures the MLE padding region is exactly zero, which is required for
    /// correct sumcheck verification in downstream operations (ConvTranspose3D, etc.).
    ///
    /// `ndims` is the total number of dimensions (e.g., 4 for [C, D, H, W]).
    /// Channel dimension is always dim 0 (highest bits in MLE layout).
    pub fn mask_channels(&mut self, x: EdgeId, ndims: usize) -> EdgeId {
        let shape = &self.init_values[x].as_ref().unwrap().shape;
        let channels = shape[0];
        let channels_pad = channels.next_power_of_two();

        // No masking needed if already power of 2
        if channels == channels_pad {
            return x;
        }

        // Create a constant binary mask: [1, 1, ..., 1, 0, 0, ..., 0]
        let _sf = self.init_values[x].as_ref().unwrap().sf;
        let data_type = self.init_values[x].as_ref().unwrap().data_type;
        let mut mask_data = vec![AlmostGoldilocksField(0); channels_pad];
        for c in 0..channels {
            mask_data[c] = AlmostGoldilocksField(1);
        }
        let mask = Witness::new(vec![channels], mask_data, data_type, 0, Role::Constant);
        let mask_edge = self.num_edges;
        self.num_edges += 1;
        self.init_values.push(Some(mask));

        // einsum "a,a<dims>->a<dims>": e.g. "a,abcd->abcd" for 4D
        let all_chars: String = (0..ndims).map(|i| (b'a' + i as u8) as char).collect();
        let eq = format!("a,{s}->{s}", s = all_chars);
        self.einsum(eq, vec![mask_edge, x], false)[0]
    }

    /// PillarMaxPool: X[N_pillars, max_points, D] → Y[N_pillars, D].
    /// With dominance (Y ≥ X_t for each t).
    pub fn pillar_maxpool(&mut self, x: EdgeId, n_pillars: usize, max_points: usize, features: usize) -> EdgeId {
        let out_shape = vec![n_pillars, features];

        // 1. PillarMaxPool(X) → Y (advice)
        let block = BasicBlockType::PillarMaxPool(PillarMaxPool {
            n_pillars, max_points, features,
        });
        let outs = self.add_gkr_node(vec![x], block);
        let y = outs[0];
        self.init_values.push(Some(Witness::new_wo_data(
            out_shape, DataType::Uint, 0, Role::Output,
        )));

        // 2. Dominance: Y ≥ X[:, t, :] for each t.
        //    X has shape [N_pillars, max_points, D] — treat as [C=N, H=T, W=D].
        //    SubSample2D(X, stride_h=1, stride_w=1, offset_h=t, offset_w=0, out_h=1, out_w=D)
        //    gives X_t[N, 1, D]. ChangeShape to [N, D], then sub(Y, X_t) ≥ 0.
        for t in 0..max_points {
            let x_t = self.subsample2d_sized(x, 1, 1, t, 0, 1, features);
            let x_t_flat = self.change_shape(x_t, vec![n_pillars, features]);
            let diff = self.sub(y, x_t_flat);
            self.add_nonneg_node(diff[0]);
        }

        y
    }

    /// ScatterToBEV: X[N_pillars, D] + coords[N_pillars, 2] → Y[D, ny, nx].
    pub fn scatter_to_bev(&mut self, x: EdgeId, coords: EdgeId, n_pillars: usize, features: usize, ny: usize, nx: usize) -> EdgeId {
        let block = BasicBlockType::ScatterToBEV(ScatterToBEV {
            n_pillars, features, ny, nx,
        });
        let outs = self.add_gkr_node(vec![x, coords], block);
        self.init_values.push(Some(Witness::new_wo_data(
            vec![features, ny, nx],
            DataType::Uint, 0, Role::Output,
        )));
        outs[0]
    }

    pub fn gather_from_grid(&mut self, grid: EdgeId, coords: EdgeId, n_points: usize, channels: usize, grid_h: usize, grid_w: usize) -> EdgeId {
        let block = BasicBlockType::GatherFromGrid(GatherFromGrid {
            n_points, channels, grid_h, grid_w,
        });
        let outs = self.add_gkr_node(vec![grid, coords], block);
        self.init_values.push(Some(Witness::new_wo_data(
            vec![n_points, channels],
            DataType::Uint, 0, Role::Output,
        )));
        outs[0]
    }

    /// Attach canonical domain gates even for standalone raw ExpHelper nodes.
    fn exp_guards(&mut self, a: EdgeId, r: EdgeId, bits: usize) -> [NodeId; 4] {
        assert!((1..=8).contains(&bits), "unsupported exp selector width");
        let sf = *SF_LOG;
        let shape = self.init_values[a].as_ref().unwrap().shape.clone();
        assert_eq!(self.init_values[a].as_ref().unwrap().sf, sf);
        let l = crate::basicblock::exp::exp_ln2(sf);
        let mut ids = [0; 4];
        // Canonical remainder and nonpositive finite-domain roots. These are
        // real NonNegative gates consumed by the PIOP, not witness assertions.
        // max(-x) = 2^bits L - floor(L/2) - 1 gives k < 2^bits.
        // All range sums and the full x = r-Lk interval are < 2^40 << p
        // for sf<=30, bits<=8, so no nonzero field multiple can counterfeit it.
        let bound = (1i128 << bits) * l - l / 2 - 1;
        let range_bits = |v: i128| -> usize { (128 - (v as u128).leading_zeros() as usize).max(1) };
        let zero = self.integer_literal(shape.clone(), 0, sf);
        let hi = self.integer_literal(shape.clone(), bound, sf);
        let neg_x = self.sub(zero, a)[0];
        self.init_values[neg_x].as_mut().unwrap().role = Role::Output;
        let slack = self.sub(hi, neg_x)[0];
        self.init_values[slack].as_mut().unwrap().role = Role::Output;
        ids[0] = self.nodes.len();
        self.add_nonneg_node_bits(neg_x, range_bits(bound));
        ids[1] = self.nodes.len();
        self.add_nonneg_node_bits(slack, range_bits(bound));
        let half = self.integer_literal(shape.clone(), l / 2, sf);
        let lm1 = self.integer_literal(shape.clone(), l - 1, sf);
        let rho = self.sub(half, r)[0];
        self.init_values[rho].as_mut().unwrap().role = Role::Output;
        let rho_slack = self.sub(lm1, rho)[0];
        self.init_values[rho_slack].as_mut().unwrap().role = Role::Output;
        ids[2] = self.nodes.len();
        self.add_nonneg_node_bits(rho, range_bits(l - 1));
        ids[3] = self.nodes.len();
        self.add_nonneg_node_bits(rho_slack, range_bits(l - 1));

        ids
    }

    /// Compile: build consumers/producers, ports, and topological order.
    pub fn compile(mut self) -> (Dag, Vec<Vec<Witness>>) {
        // Every division/RMS advice primitive receives defining constraints, including raw
        // add_gkr_node users. Legacy row-f64 reciprocal advice is unsupported.
        // The queue is intentionally live: RMS guards themselves create an
        // exact quarter division and a private bucket advice node.
        let mut cursor=0;
        while cursor<self.nodes.len() {
            let node=self.nodes[cursor].clone();cursor+=1;
            match node.kind {
                BasicBlockType::DivConst(b)=>self.division_guards(node.inputs[0],node.outputs[0],usize::try_from(b.divisor).expect("divisor too large")),
                BasicBlockType::RMSReciprocal(b)=>{
                    assert_eq!(b.dim,0,"legacy unconstrained RMS advice is unsupported; use the exact mean gate");
                    crate::dag::llama::rms_reciprocal_constraints(&mut self,node.inputs[0],node.outputs[0]);
                },
                BasicBlockType::RMSBucket(_)=>crate::dag::llama::rms_bucket_constraints(&mut self,node.inputs[0],node.outputs[0]),
                _=>{},
            }
        }
        let helpers: Vec<_> = self.nodes.iter().filter_map(|n| {
            if let BasicBlockType::ExpHelper(h) = &n.kind {
                Some((n.id, n.inputs[0], n.outputs[0], h.num_bits))
            } else { None }
        }).collect();
        let exp_guard_nodes = helpers.into_iter().map(|(n,x,r,bits)|
            (n, self.exp_guards(x,r,bits))).collect();
        let DagBuilder {
            program_literals,
            nodes,
            num_edges,
            init_values,
            range,
            two_pow,
            layer_boundaries,
            self_claim_edges,
        } = self;

        if std::env::var("ZK4_KIND_REPORT").is_ok() {
            let mut h: std::collections::BTreeMap<String, usize> = Default::default();
            for n in &nodes {
                let k = format!("{:?}", n.kind);
                let k = k.split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap().to_string();
                *h.entry(k).or_default() += 1;
            }
            let mut roles: std::collections::BTreeMap<String, (usize, u128)> = Default::default();
            for v in init_values.iter().flatten() {
                let e = roles.entry(format!("{:?}", v.role)).or_default();
                e.0 += 1; e.1 += v.shape.iter().map(|&d| d.next_power_of_two() as u128).product::<u128>();
            }
            println!("KIND_REPORT nodes={} edges={} kinds={:?} roles={:?}", nodes.len(), num_edges, h, roles);
            std::process::exit(0);
        }

        // edge -> consumers
        let mut consumers: Vec<Vec<NodeId>> = vec![Vec::new(); num_edges];
        for n in &nodes {
            for &e in &n.inputs {
                consumers[e].push(n.id);
            }
        }

        // produced edges + producers map
        let mut produced = vec![false; num_edges];
        let mut producers = vec![None; num_edges];
        for n in &nodes {
            for &e in &n.outputs {
                produced[e] = true;
                producers[e] = Some(n.id);
            }
        }

        let input_ports: Vec<EdgeId> = (0..num_edges)
            .filter(|&e| !produced[e] && init_values[e].as_ref().unwrap().role == Role::Input)
            .collect();
        let mut output_ports: Vec<EdgeId> =
            (0..num_edges).filter(|&e| consumers[e].is_empty()).collect();
        output_ports.extend(
            range
                .iter()
                .filter(|&n| matches!(nodes[*n].kind, BasicBlockType::NonNegative(_)))
                .map(|n| nodes[*n].inputs[0]),
        );

        // in-degree = #inputs that come from produced edges
        let mut indeg = vec![0usize; nodes.len()];
        for n in &nodes {
            indeg[n.id] = n.inputs.iter().filter(|&&e| produced[e]).count();
        }

        // adjacency: node -> downstream nodes via outputs' consumers
        let mut outgoing: Vec<Vec<NodeId>> = vec![Vec::new(); nodes.len()];
        for n in &nodes {
            for &e in &n.outputs {
                for &v in &consumers[e] {
                    outgoing[n.id].push(v);
                }
            }
        }

        // Kahn topo with level tracking
        let mut current_level: Vec<NodeId> = indeg
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d == 0).then_some(i))
            .collect();
        let mut topo = Vec::with_capacity(nodes.len());
        let mut topo_levels: Vec<Vec<NodeId>> = Vec::new();
        while !current_level.is_empty() {
            let mut next_level = Vec::new();
            for &u in &current_level {
                topo.push(u);
                for &v in &outgoing[u] {
                    indeg[v] -= 1;
                    if indeg[v] == 0 {
                        next_level.push(v);
                    }
                }
            }
            topo_levels.push(current_level);
            current_level = next_level;
        }
        assert_eq!(
            topo.len(),
            nodes.len(),
            "graph has a cycle or disconnected inputs"
        );

        // Build alias view
        let mut alias_to_edge: Vec<EdgeId> = Vec::new();
        let mut alias_to_consumer: Vec<NodeId> = Vec::new();
        let mut alias_input_slot: Vec<usize> = Vec::new();
        let mut edge_aliases: Vec<Vec<AliasId>> = vec![Vec::new(); num_edges];

        for (nid, node) in nodes.iter().enumerate() {
            for (slot, &e) in node.inputs.iter().enumerate() {
                let aid = AliasId(alias_to_edge.len());
                alias_to_edge.push(e);
                alias_to_consumer.push(nid);
                alias_input_slot.push(slot);
                edge_aliases[e].push(aid);
            }
        }

        let dag = Dag {
            exp_guard_nodes,
            program_literals,
            nodes,
            num_edges,
            topo,
            topo_levels,
            range,
            two_pow,
            consumers,
            producers,
            input_ports,
            output_ports,
            layer_boundaries,
            boundary_edges: Vec::new(),
            crossing_edges: Vec::new(),
            self_claim_edges: self_claim_edges.into_iter().collect(),
            edge_aliases,
            alias_to_edge,
            alias_to_consumer,
            alias_input_slot,
        };

        let init_values = init_values
            .iter()
            .map(|value| vec![value.as_ref().unwrap().clone()])
            .collect::<Vec<Vec<Witness>>>();

        (dag, init_values)
    }

    /// Compose via a recipe: f(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId>
    pub fn pipe<Fn>(&mut self, inlet: &[EdgeId], f: Fn) -> Vec<EdgeId>
    where
        Fn: FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId>,
    {
        f(self, inlet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_zero_is_one_at_output_scale_cpu_forward() {
        // Execute the actual graph's CPU operators directly, avoiding the
        // benchmark run hook and any GPU backend environment override.
        let mut g = DagBuilder::new();
        let x = g.committed_input(Witness::new(
            vec![2], vec![AlmostGoldilocksField(0); 2],
            DataType::Float, *SF_LOG, Role::Input,
        ));
        let out = g.exp(x)[0];
        let (dag, mut witnesses) = g.compile();
        for &nid in &dag.topo {
            let node = &dag.nodes[nid];
            let inputs: Vec<&Witness> = node.inputs.iter().map(|&e| &witnesses[e][0]).collect();
            let outputs = node.kind.run(&inputs);
            assert_eq!(outputs.len(), node.outputs.len());
            for (&edge, value) in node.outputs.iter().zip(outputs) {
                witnesses[edge] = vec![value];
            }
        }
        let result = &witnesses[out][0];
        assert_eq!(result.sf, *SF_LOG);
        assert!(result.data.as_ref().unwrap().evaluations_ref().iter()
            .all(|v| *v == AlmostGoldilocksField(1u64 << *SF_LOG)));
    }

    #[test]
    fn exp_logical_values_ignore_padding_and_padded_outputs_are_zero() {
        let evaluate = |shape: Vec<usize>, values: Vec<i128>| {
            let mut b=DagBuilder::new();
            let x=b.committed_input(Witness::new(shape,values.into_iter().map(int_to_f).collect(),DataType::Float,*SF_LOG,Role::Input));
            let y=b.exp(x)[0]; let (d,mut w)=b.compile();
            for &n in &d.topo {
                let nd=&d.nodes[n]; let inputs=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
                let outputs=nd.kind.run(&inputs);
                for (&e,v) in nd.outputs.iter().zip(outputs) {w[e]=vec![v];}
            }
            w[y][0].data.as_ref().unwrap().evaluations().iter().map(|v|v.reduce().0).collect::<Vec<_>>()
        };
        let logical=-(1i128<<*SF_LOG);
        let expected=evaluate(vec![1],vec![logical])[0];
        for pad in [400,-5000,10000] {
            let mut v=vec![logical;5]; v.extend([pad;3]);
            let out=evaluate(vec![5],v);
            assert_eq!(&out[..5],&[expected;5]); assert_eq!(&out[5..],&[0;3]);
        }
        // Multi-axis, column-major layout: axis0 varies fastest.
        let mut v=vec![400;16];
        for j in 0..3 {for i in 0..3 {v[i+4*j]=logical;}}
        let out=evaluate(vec![3,3],v);
        for j in 0..4 {for i in 0..4 {
            assert_eq!(out[i+4*j],if i<3 && j<3 {expected} else {0});
        }}
    }

    #[test]
    fn eager_request_arithmetic_is_not_a_model_parameter() {
        for v in [3,19] {
            let mut b=DagBuilder::new();
            let x=b.committed_input(Witness::new(vec![1],vec![int_to_f(v)],DataType::Int,0,Role::Input));
            let literal=b.public_literal(Witness::new(vec![1],vec![int_to_f(7)],DataType::Int,0,Role::Constant));
            let a=b.add(x,literal)[0]; let z=b.sub(a,literal)[0];
            assert_eq!(b.init_values[a].as_ref().unwrap().role,Role::Output);
            assert_eq!(b.init_values[z].as_ref().unwrap().role,Role::Output);
            assert_eq!(crate::util::arith::f_to_int(b.init_values[z].as_ref().unwrap().data.as_ref().unwrap().index(0)),v);
            let (d,w)=b.compile();
            let g=crate::morpheus::piop::Graph::new(&d,&w,crate::morpheus::piop::Config::default());
            assert!(!g.model[a] && !g.model[z] && !g.committed[a] && !g.committed[z]);
        }
    }

    #[test]
    fn eager_model_only_arithmetic_remains_trusted_precomputation() {
        let mut b=DagBuilder::new();
        let a=b.param(Witness::new(vec![1],vec![int_to_f(3)],DataType::Int,0,Role::Constant));
        let c=b.param(Witness::new(vec![1],vec![int_to_f(7)],DataType::Int,0,Role::Constant));
        let sum=b.add(a,c)[0]; let difference=b.sub(sum,c)[0];
        assert_eq!(b.init_values[sum].as_ref().unwrap().role,Role::Constant);
        assert_eq!(b.init_values[difference].as_ref().unwrap().role,Role::Constant);
        // Such values belong to the trusted model-registration preprocessing;
        // this is not a proof of their relationship to separate weight leaves.
    }

    #[test]
    fn test_dag_builder_simple_add() {
        // Build: input_a + input_b → output
        let mut g = DagBuilder::new();
        let a = g.input(vec![4], DataType::Uint);
        let b = g.input(vec![4], DataType::Uint);
        let out = g.add(a, b);

        assert_eq!(g.nodes.len(), 1); // One add node
        assert!(g.num_edges >= 3); // At least: a, b, output
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_dag_builder_compile() {
        let mut g = DagBuilder::new();
        let a = g.input(vec![4], DataType::Uint);
        let b = g.input(vec![4], DataType::Uint);
        let _out = g.add(a, b);

        let (dag, witnesses) = g.compile();

        assert!(!dag.topo.is_empty(), "Topological order should not be empty");
        assert_eq!(dag.input_ports.len(), 2, "Should have 2 inputs");
        assert!(!dag.output_ports.is_empty(), "Should have output ports");
        assert!(!witnesses.is_empty());
    }

    #[test]
    fn test_dag_builder_chain() {
        // input → add(input, param) → einsum
        let mut g = DagBuilder::new();
        let a = g.input(vec![4], DataType::Uint);
        let b_witness = Witness::new(
            vec![4],
            vec![AlmostGoldilocksField(1), AlmostGoldilocksField(1), AlmostGoldilocksField(1), AlmostGoldilocksField(1)],
            DataType::Uint,
            0,
            Role::Constant,
        );
        let b = g.param(b_witness);
        let sum = g.add(a, b);
        let _out = g.einsum("a->a".to_string(), vec![sum[0]], false);

        let (dag, _witnesses) = g.compile();
        assert_eq!(dag.topo.len(), 2); // add + einsum
    }

    #[test]
    fn test_dag_builder_pipe() {
        let mut g = DagBuilder::new();
        let a = g.input(vec![4], DataType::Uint);

        // Use pipe to compose a simple identity function
        let identity = |g: &mut DagBuilder, x: &[EdgeId]| -> Vec<EdgeId> {
            let e = g.einsum("a->a".to_string(), vec![x[0]], false);
            e
        };
        let _out = g.pipe(&[a], identity);

        let (dag, _witnesses) = g.compile();
        assert_eq!(dag.topo.len(), 1);
    }
}
