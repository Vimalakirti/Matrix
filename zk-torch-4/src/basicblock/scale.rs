//! [`ScaleDown`] / [`ScaleUp`] — fixed-point rescaling.
//!
//! Both produce two outputs:
//!   `output[0]`: the rescaled value (a dense polynomial).
//!   `output[1]`: a sparse selection polynomial encoding the rounding
//!                remainder, used by the DAG-level range-check protocol
//!                to prove `|remainder| ≤ rescale_factor / 2`.
//!
//! Restored to the z-t-2 shape (one nonzero per row) per §5.5 of the plan.
//! The basicblock's own `prove` is a single claim-pass-through; the lookup
//! soundness lives in `dag::prove_range`.

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, PolyType, Role, Witness};
use crate::poly::SelectionPolynomial;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f, next_pow};

/// One ScaleDown lane: round-shift `x_i` by `rescale_sf` bits and emit the
/// rounding remainder's table index. Pure + `#[inline]` so the rayon map in
/// `ScaleDown::run` is a tight per-element loop. Identical arithmetic to the
/// original serial body.
#[inline]
fn scale_down_elem(
    x_i: AlmostGoldilocksField,
    half: i128,
    rescale_sf: usize,
    rescale_factor: i128,
    rescale_factor_f: AlmostGoldilocksField,
) -> (AlmostGoldilocksField, usize) {
    let x_int = f_to_int(x_i);
    let shifted = x_int + half;
    let y_int = shifted >> rescale_sf;
    let y_f = int_to_f(y_int);
    let aux_f = x_i - y_f * rescale_factor_f;
    let aux_num = f_to_int(aux_f) + half;
    let table_index = if aux_num >= 0 && (aux_num as u128) < rescale_factor as u128 {
        aux_num as usize
    } else {
        0
    };
    (y_f, table_index)
}

// ============================================================================
// ScaleDown
// ============================================================================

#[derive(Debug, Clone)]
pub struct ScaleDown {
    /// Target output scale factor. The input's SF is read from its witness
    /// at `run` time — the DagBuilder always knows the producer's SF, so
    /// carrying it twice would be redundant.
    pub output_sf: usize,
}

impl BasicBlock for ScaleDown {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ScaleDown expects 1 input");
        let x = inputs[0];
        assert!(
            x.sf >= self.output_sf,
            "ScaleDown: input sf ({}) must be ≥ output_sf ({})",
            x.sf,
            self.output_sf
        );
        let x_shape_padded: Vec<usize> = x
            .shape
            .iter()
            .map(|&s| next_pow(s as u32) as usize)
            .collect();
        let n: usize = x_shape_padded.iter().product();
        let num_var = get_n(&x_shape_padded);
        let rescale_sf = x.sf - self.output_sf;
        let rescale_factor: i128 = 1i128 << rescale_sf;
        let half = rescale_factor / 2;
        let rescale_factor_f = int_to_f(rescale_factor);

        // Per-element, fully independent → parallelize. This rescale fires
        // after every conv and was the single largest forward-pass cost
        // (serial it dominated `dag.run` on big CV activations — e.g. ~21 s of
        // a 37 s UNet 64³ forward). Grab the host slice once so each lane is a
        // pure array read (no per-element OnceLock check / virtual dispatch).
        use rayon::prelude::*;
        let data = x.data.as_ref().unwrap();
        let computed: Vec<(AlmostGoldilocksField, usize)> = match data.try_evaluations_ref() {
            Some(slice) => (0..n)
                .into_par_iter()
                .map(|i| scale_down_elem(slice[i], half, rescale_sf, rescale_factor, rescale_factor_f))
                .collect(),
            None => (0..n)
                .into_par_iter()
                .map(|i| scale_down_elem(data.index(i), half, rescale_sf, rescale_factor, rescale_factor_f))
                .collect(),
        };
        let mut y_data = vec![AlmostGoldilocksField(0); n];
        let mut selection = Vec::with_capacity(n);
        for (i, (y_f, table_index)) in computed.into_iter().enumerate() {
            y_data[i] = y_f;
            selection.push((i, table_index));
        }

        let aux_poly = SelectionPolynomial::new(num_var, rescale_sf, selection).to_sparse_dag();
        let y = Witness::new(
            x.shape.clone(),
            y_data,
            x.data_type,
            self.output_sf,
            Role::Output,
        );
        let aux = Witness {
            shape: x.shape.clone(),
            data: Some(Box::new(aux_poly)),
            poly_type: PolyType::Sparse,
            data_type: x.data_type,
            sf: 0,
            role: Role::Auxiliary,
        };
        vec![y, aux]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Per-element scalar arithmetic with no algebraic shortcut. CPU is
        // the documented default — bandwidth-bound on either path, and the
        // signed-int decomposition needs `i128` precision that CUDA doesn't
        // natively support. See philosophy rule #7.
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Pass the y-claim through to the input as an Ext2 claim at the same
        // point: `x(r) = y(r) · rescale + (aux − rescale/2)`. The aux side is
        // proved by the DAG-level range check; here we just emit the input
        // claim and let the prove_range protocol couple it back.
        let claim = out_claims[0];
        let inp_eval = witnesses[0]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&claim.point);
        let inp = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: claim.point.clone(),
            eval: inp_eval,
        };
        (vec![], vec![inp])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        _claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        true
    }
}

// ============================================================================
// ScaleUp
// ============================================================================

#[derive(Debug, Clone)]
pub struct ScaleUp {
    /// Target output scale factor. Input SF is read from the witness.
    pub output_sf: usize,
}

impl BasicBlock for ScaleUp {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ScaleUp expects 1 input");
        let x = inputs[0];
        assert!(
            x.sf <= self.output_sf,
            "ScaleUp: input sf ({}) must be ≤ output_sf ({})",
            x.sf,
            self.output_sf
        );
        let x_shape_padded: Vec<usize> = x
            .shape
            .iter()
            .map(|&s| next_pow(s as u32) as usize)
            .collect();
        let n: usize = x_shape_padded.iter().product();
        let num_var = get_n(&x_shape_padded);
        let rescale_sf = self.output_sf - x.sf;
        let rescale_factor: i128 = 1i128 << rescale_sf;
        let half = rescale_factor / 2;
        let rescale_factor_f = int_to_f(rescale_factor);

        let mut y_data = vec![AlmostGoldilocksField(0); n];
        let mut selection = Vec::with_capacity(n);
        let data = x.data.as_ref().unwrap();
        for i in 0..n {
            let x_i = data.index(i);
            let x_int = f_to_int(x_i);
            let y_int = x_int * rescale_factor;
            y_data[i] = int_to_f(y_int);

            let aux_f = x_i * rescale_factor_f - y_data[i];
            let aux_num = f_to_int(aux_f) + half;
            let table_index = if aux_num >= 0 && (aux_num as u128) < rescale_factor as u128 {
                aux_num as usize
            } else {
                0
            };
            selection.push((i, table_index));
        }

        let aux_poly = SelectionPolynomial::new(num_var, rescale_sf, selection).to_sparse_dag();
        let y = Witness::new(
            x.shape.clone(),
            y_data,
            x.data_type,
            self.output_sf,
            Role::Output,
        );
        let aux = Witness {
            shape: x.shape.clone(),
            data: Some(Box::new(aux_poly)),
            poly_type: PolyType::Sparse,
            data_type: x.data_type,
            sf: 0,
            role: Role::Auxiliary,
        };
        vec![y, aux]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let claim = out_claims[0];
        let inp_eval = witnesses[0]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&claim.point);
        let inp = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: claim.point.clone(),
            eval: inp_eval,
        };
        (vec![], vec![inp])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        _claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;
    use crate::poly::SparseMLPoly;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn build_input(shape: Vec<usize>, vals: Vec<u64>, sf: usize) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Int,
            sf,
            Role::Input,
        )
    }

    // ---------- ScaleDown ----------

    #[test]
    fn scaledown_quotient_and_aux_partition_input() {
        // input_sf = 4, output_sf = 0 → rescale = 16, half = 8.
        let x = build_input(vec![4], vec![0, 7, 8, 15], 4);
        let sd = ScaleDown { output_sf: 0 };
        let out = sd.run(&[&x]);
        assert_eq!(out.len(), 2);
        let y_evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // y = floor((x + 8) / 16): {0→0, 7→0, 8→1, 15→1}.
        assert_eq!(y_evals, &[agl(0), agl(0), agl(1), agl(1)]);

        // aux entry at row i has table index `x_i - y_i * 16 + 8`.
        let aux = out[1]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        let sel: std::collections::HashMap<usize, usize> =
            aux.selection.selection.iter().copied().collect();
        assert_eq!(sel[&0], 8);  // x=0,  y=0 → aux = 0 + 8 = 8
        assert_eq!(sel[&1], 15); // x=7,  y=0 → aux = 7 + 8 = 15
        assert_eq!(sel[&2], 0);  // x=8,  y=1 → aux = -8 + 8 = 0
        assert_eq!(sel[&3], 7);  // x=15, y=1 → aux = -1 + 8 = 7
    }

    /// ScaleDown handles negatives by storing `q − |v|` as the field rep —
    /// `f_to_int` reads it as negative, the arithmetic shift produces the
    /// correct floor, and the aux is in `[0, rescale)`.
    #[test]
    fn scaledown_handles_negative_inputs() {
        let neg = |v: i128| -> AlmostGoldilocksField {
            crate::util::arith::int_to_f(v)
        };
        let raw = vec![neg(-7).0, neg(-1).0, agl(0).0, agl(1).0];
        let x = build_input(vec![4], raw, 4);
        let sd = ScaleDown { output_sf: 0 };
        let out = sd.run(&[&x]);
        let y_evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // y = floor((x + 8) / 16): {-7→0, -1→0, 0→0, 1→0}.
        for ev in y_evals {
            assert_eq!(ev.reduce(), agl(0));
        }
    }

    #[test]
    fn scaledown_prove_passes_claim_through() {
        let x = build_input(vec![4], vec![0, 1, 2, 3], 4);
        let sd = ScaleDown { output_sf: 0 };
        let outs = sd.run(&[&x]);
        let y = &outs[0];
        let pt = vec![
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(5)),
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(7)),
        ];
        let y_eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let y_claim = Claim { edge_id: 1, sparse_id: 0, point: pt.clone(), eval: y_eval };
        let mut t = Transcript::new(b"sd");
        let (proofs, claims) = sd.prove(&[&x], &[0], &[&y_claim], &mut t);
        assert!(proofs.is_empty());
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].point, pt);
        // The input claim's eval is f_x(pt).
        let direct = x.data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        assert_eq!(claims[0].eval, direct);
    }

    // ---------- ScaleUp ----------

    #[test]
    fn scaleup_quotient_and_aux_partition_input() {
        // input_sf = 0, output_sf = 4 → rescale = 16, half = 8.
        let x = build_input(vec![4], vec![0, 1, 2, 3], 0);
        let su = ScaleUp { output_sf: 4 };
        let out = su.run(&[&x]);
        let y_evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // y = x · 16.
        assert_eq!(y_evals, &[agl(0), agl(16), agl(32), agl(48)]);
        // aux = (x · 16 − y) + 8 = 0 + 8 = 8 for every row (no rounding error).
        let aux = out[1]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        for (_, t) in &aux.selection.selection {
            assert_eq!(*t, 8);
        }
    }

    #[test]
    fn scaleup_aux_has_one_nonzero_per_row() {
        let x = build_input(vec![4], vec![0, 1, 2, 3], 0);
        let su = ScaleUp { output_sf: 3 }; // rescale = 8
        let out = su.run(&[&x]);
        let aux = out[1]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        assert_eq!(aux.num_nonzero(), 4);
    }

    #[test]
    fn scaleup_prove_passes_claim_through() {
        let x = build_input(vec![4], vec![0, 1, 2, 3], 0);
        let su = ScaleUp { output_sf: 4 };
        let outs = su.run(&[&x]);
        let pt = vec![
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(2)),
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(11)),
        ];
        let y_eval = outs[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let y_claim = Claim { edge_id: 1, sparse_id: 0, point: pt.clone(), eval: y_eval };
        let mut t = Transcript::new(b"su");
        let (_, claims) = su.prove(&[&x], &[0], &[&y_claim], &mut t);
        let direct = x.data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        assert_eq!(claims[0].eval, direct);
    }
}
