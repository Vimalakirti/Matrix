//! [`ExpHelper`] and [`TwoPow`] — exp decomposition.
//!
//! Caller guarantees the input `x ≤ 0` (softmax subtracts row-max, sigmoid
//! emits `x + SigmoidConst ≤ 0`). `ExpHelper` decomposes `x = k · (−ln2·SF)
//! + r` with `k ∈ [0, 2^K_BITS)` and `|r| ≤ ln2/2 · SF`, emitting a sparse
//! selection polynomial for the `k` values plus a dense `r`.
//!
//! [`TwoPow`] looks up `2^(15 − k)` from the selection polynomial — the
//! result is `2^x` once multiplied by the Taylor expansion of `exp(r)` (done
//! elsewhere in the DAG).

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, PolyType, Role, Witness};
use crate::poly::{SelectionPolynomial, SparseMLPoly};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f, next_pow};
use crate::{SF_FLOAT, SF_LOG};

/// Default decomposition width: `k ∈ [0, 2^K_BITS)`. K=4 covers `exp(x)`
/// for `x ∈ [−15·ln2, 0]` at SF=15 — enough range for softmax/sigmoid
/// bounds. The DagBuilder constructs [`ExpHelper`] with a builder-chosen
/// `num_bits` (defaults to 4, can be widened by upstream callers).
pub const K_BITS: usize = 4;

// ============================================================================
// ExpHelper
// ============================================================================

#[derive(Debug, Clone)]
pub struct ExpHelper {
    pub num_bits: usize,
}

impl ExpHelper {
    pub fn new(num_bits: usize) -> Self {
        Self { num_bits }
    }
}

impl BasicBlock for ExpHelper {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ExpHelper expects 1 input");
        let x = inputs[0];
        let x_shape_padded: Vec<usize> =
            x.shape.iter().map(|&s| next_pow(s as u32) as usize).collect();
        let n: usize = x_shape_padded.iter().product();
        let num_var = get_n(&x_shape_padded);
        let ln2 = (2.0_f64.ln() * (*SF_FLOAT as f64)).round() as i128;
        let neg_ln2_f = int_to_f(-ln2);
        let table_size = 1usize << self.num_bits;

        let mut r_data = vec![AlmostGoldilocksField(0); n];
        let mut selection = Vec::with_capacity(n);
        let data = x.data.as_ref().unwrap();
        for i in 0..n {
            let x_i = data.index(i);
            let x_num = f_to_int(x_i) as f64;
            let k = ((-x_num) / (ln2 as f64)).round();
            // r = x − k · (−ln2 · SF) = x + k · ln2 · SF
            let k_f = int_to_f(k as i128);
            r_data[i] = x_i - k_f * neg_ln2_f;
            let k_idx = k as i64;
            let table_index = if k_idx >= 0 && (k_idx as usize) < table_size {
                k_idx as usize
            } else {
                0
            };
            selection.push((i, table_index));
        }

        let aux_poly = SelectionPolynomial::new(num_var, self.num_bits, selection).to_sparse();
        let r = Witness::new(x.shape.clone(), r_data, DataType::Float, *SF_LOG, Role::Output);
        let aux = Witness {
            shape: x.shape.clone(),
            data: Some(Box::new(aux_poly)),
            poly_type: PolyType::Sparse,
            data_type: x.data_type,
            sf: 0,
            role: Role::Auxiliary,
        };
        vec![r, aux]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Same reasoning as ScaleDown: bandwidth-bound per-element, with
        // a `f64::round` step that CUDA would have to mirror exactly.
        // CPU is the documented default.
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

// ============================================================================
// TwoPow
// ============================================================================

/// `TwoPow` materializes `2^(15 − k)` at each row where the
/// [`ExpHelper`] selection polynomial recorded the `k` value. Output sf is
/// fixed at 15. Soundness comes from the DAG-level two-pow lookup protocol
/// (`dag::prove_two_pow`), which checks the input selection polynomial
/// against the public table `T[k] = 2^(15 − k)`.
#[derive(Debug, Clone)]
pub struct TwoPow;

impl BasicBlock for TwoPow {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "TwoPow expects 1 input");
        let x = inputs[0];
        let sparse = x
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .expect("TwoPow input must be the SparseMLPoly from ExpHelper");
        let inp_num_vars = sparse.selection.input_num_vars;
        let mut y_data = vec![AlmostGoldilocksField(0); 1usize << inp_num_vars];
        for &(input_index, table_index) in &sparse.selection.selection {
            // table[k] = 2^(15 − k) for k ∈ [0, 16).
            assert!(table_index < 16, "TwoPow: table_index {} ≥ 16", table_index);
            y_data[input_index] = AlmostGoldilocksField(1u64 << (15 - table_index));
        }
        let y = Witness::new(x.shape.clone(), y_data, DataType::Float, 15, Role::Output);
        vec![y]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        self.run(inputs)
    }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Empty — the DAG-level two-pow protocol handles this.
        (vec![], vec![])
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

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn build_input(shape: Vec<usize>, vals: Vec<i128>) -> Witness {
        let evals: Vec<_> = vals.into_iter().map(int_to_f).collect();
        Witness::new(shape, evals, DataType::Int, *SF_LOG, Role::Input)
    }

    #[test]
    fn exphelper_emits_dense_r_and_sparse_k() {
        // x = 0 at every row → k = 0, r = 0.
        let x = build_input(vec![4], vec![0, 0, 0, 0]);
        let h = ExpHelper::new(K_BITS);
        let out = h.run(&[&x]);
        assert_eq!(out.len(), 2);
        let r_evals = out[0].data.as_ref().unwrap().evaluations_ref();
        for ev in r_evals {
            assert_eq!(ev.reduce(), agl(0));
        }
        let aux = out[1]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        assert_eq!(aux.num_nonzero(), 4);
        for (_, t) in &aux.selection.selection {
            assert_eq!(*t, 0);
        }
    }

    /// `x = -k · ln2 · SF` for `k ∈ {0, 1, 5, 15}` → selection table index = k.
    #[test]
    fn exphelper_recovers_negative_k() {
        let ln2 = (2.0_f64.ln() * (*SF_FLOAT as f64)).round() as i128;
        let x = build_input(vec![4], vec![0, -ln2, -5 * ln2, -15 * ln2]);
        let h = ExpHelper::new(K_BITS);
        let out = h.run(&[&x]);
        let aux = out[1]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        let sel: std::collections::HashMap<usize, usize> =
            aux.selection.selection.iter().copied().collect();
        assert_eq!(sel[&0], 0);
        assert_eq!(sel[&1], 1);
        assert_eq!(sel[&2], 5);
        assert_eq!(sel[&3], 15);
    }

    #[test]
    fn exphelper_prove_passes_through() {
        let x = build_input(vec![4], vec![0, 0, 0, 0]);
        let h = ExpHelper::new(K_BITS);
        let out = h.run(&[&x]);
        let pt = vec![
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(5)),
            almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::from_base(agl(7)),
        ];
        let r_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let r_claim = Claim { edge_id: 1, sparse_id: 0, point: pt.clone(), eval: r_eval };
        let mut t = Transcript::new(b"exp");
        let (_, claims) = h.prove(&[&x], &[0], &[&r_claim], &mut t);
        let direct = x.data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        assert_eq!(claims[0].eval, direct);
    }

    #[test]
    fn twopow_looks_up_15_minus_k() {
        // Build a selection polynomial with (i, k) = (0, 0), (1, 1), (2, 5), (3, 15).
        let sel = SelectionPolynomial::new(
            2,
            K_BITS,
            vec![(0, 0), (1, 1), (2, 5), (3, 15)],
        )
        .to_sparse();
        let w = Witness {
            shape: vec![4],
            data: Some(Box::new(sel)),
            poly_type: PolyType::Sparse,
            data_type: DataType::Int,
            sf: 0,
            role: Role::Auxiliary,
        };
        let out = TwoPow.run(&[&w]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0].reduce().0, 1 << 15);
        assert_eq!(evals[1].reduce().0, 1 << 14);
        assert_eq!(evals[2].reduce().0, 1 << 10);
        assert_eq!(evals[3].reduce().0, 1 << 0);
    }

    #[test]
    fn twopow_prove_returns_empty() {
        // The dag-level prove_two_pow handles this; the block itself is a no-op.
        let sel = SelectionPolynomial::new(2, K_BITS, vec![(0, 0)]).to_sparse();
        let w = Witness {
            shape: vec![4],
            data: Some(Box::new(sel)),
            poly_type: PolyType::Sparse,
            data_type: DataType::Int,
            sf: 0,
            role: Role::Auxiliary,
        };
        let mut t = Transcript::new(b"tp");
        let (proofs, claims) = TwoPow.prove(&[&w], &[0], &[], &mut t);
        assert!(proofs.is_empty() && claims.is_empty());
    }
}
