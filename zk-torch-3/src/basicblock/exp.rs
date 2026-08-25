use goldilocks_cuda::GoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::sparse::SelectionPolynomial;
use crate::poly::SparseMLPoly;

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};
use crate::SF_FLOAT;

/// ExpHelper block: decomposes input x into k and remainder r such that
/// x = k * (-ln2 * SF) + r, where k is a 4-bit integer (0..15).
#[derive(Clone, Debug)]
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
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let input_num_vars = get_n(&input.shape);
        let table_num_vars = 4; // 16 entries for k in [0, 15]
        let n = 1usize << input_num_vars;

        let ln2 = (2.0_f64.ln() * (*SF_FLOAT as f64)).round() as i128;
        let neg_ln2_f = int_to_f(-ln2);

        let mut r_data = Vec::with_capacity(n);
        let mut selection = Vec::with_capacity(n);

        for i in 0..n {
            let x_int = f_to_int(evals[i]) as f64;
            let k = ((-x_int) / (ln2 as f64)).round().max(0.0).min(15.0) as usize;
            // r = x - k * (-ln2)
            let r_val = {
                let k_field = int_to_f(k as i128);
                // r = x + k * ln2 (since neg_ln2_f = -ln2)
                let k_times_neg_ln2 = GoldilocksField(
                    ((k_field.0 as u128 * neg_ln2_f.0 as u128) % crate::GOLDILOCKS_PRIME as u128) as u64,
                );
                // r = x - k * neg_ln2_f (field subtraction)
                let x_val = evals[i];
                if x_val.0 >= k_times_neg_ln2.0 {
                    GoldilocksField(x_val.0 - k_times_neg_ln2.0)
                } else {
                    GoldilocksField(crate::GOLDILOCKS_PRIME - (k_times_neg_ln2.0 - x_val.0))
                }
            };
            r_data.push(r_val);
            selection.push((i, k));
        }

        let sel_poly = SelectionPolynomial::new(input_num_vars, table_num_vars, selection);
        let sparse = sel_poly.to_sparse();

        vec![
            Witness::new(input.shape.clone(), r_data, input.data_type, input.sf, Role::Output),
            Witness::new_sparse(input.shape.clone(), sparse, DataType::Uint, 0, Role::Auxiliary),
        ]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let inp_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: out_claims[0].point.clone(),
            eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&out_claims[0].point),
        };
        (vec![], vec![inp_claim])
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

/// TwoPow block: table lookup for 2^(15-k).
/// Input is a SparseMLPoly from ExpHelper containing the selection polynomial.
#[derive(Clone, Debug)]
pub struct TwoPow;

impl BasicBlock for TwoPow {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let sparse = input
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .expect("TwoPow input must be SparseMLPoly");
        let inp_num_vars = sparse.selection.input_num_vars;
        let n = 1usize << inp_num_vars;
        let mut y_data = vec![GoldilocksField(0); n];
        for &(input_index, table_index) in &sparse.selection.selection {
            if input_index < n && table_index < 16 {
                y_data[input_index] = GoldilocksField(1u64 << (15 - table_index));
            }
        }

        vec![Witness::new(
            input.shape.clone(),
            y_data,
            DataType::Float,
            15,
            Role::Output,
        )]
    }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
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
