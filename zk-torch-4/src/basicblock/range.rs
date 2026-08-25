//! `NonNegative` — emits a [`SparseMLPoly`] selection polynomial encoding the
//! input value at each row (z-t-2 form, per §5.5 of the plan).
//!
//! The basicblock's own `prove` returns empty — the real soundness comes from
//! the DAG-level `prove_range` lookup protocol that batches all range checks
//! into one combined sumcheck (table sumcheck + sparse-bool sumcheck). See
//! the plan §5.5 and the future `dag::prove_range` for the verifier-side
//! checks.

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, PolyType, Role, Witness};
use crate::poly::SelectionPolynomial;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, next_pow};

#[derive(Debug, Clone)]
pub struct NonNegative {
    /// `t = table_size_log`. The lookup table covers values `[0, 2^t)`.
    pub table_size_log: usize,
}

impl NonNegative {
    pub fn new(table_size_log: usize) -> Self {
        Self { table_size_log }
    }
}

impl BasicBlock for NonNegative {
    /// Produce a single sparse witness: for each input row `i`, set the
    /// `(i, x_i)` cell to 1. Out-of-range values (incl. negatives, since
    /// signed-int reps in `F_q` are `> q/2`) clamp to table index 0 — the
    /// resulting selection is wrong for that row, which the verifier catches
    /// via the dag-level table sumcheck.
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "NonNegative expects 1 input");
        let x = inputs[0];
        let x_shape_padded: Vec<usize> = x
            .shape
            .iter()
            .map(|&s| next_pow(s as u32) as usize)
            .collect();
        let n: usize = x_shape_padded.iter().product();
        let num_var = get_n(&x_shape_padded);
        let table_size = 1usize << self.table_size_log;

        let mut selection = Vec::with_capacity(n);
        let data = x.data.as_ref().expect("NonNegative: input has no data");
        // Out-of-range tracking: a value that is negative or ≥ table_size gets
        // clamped to table index 0, which is the WRONG selection — the proof
        // will then fail to verify with no other signal. This is the classic
        // "silent Verified: false" from too-small `table_size_log` (the range
        // table covers only [0, 2^table_size_log)). Detect it here, during
        // witness gen (well before the expensive prove), and warn loudly with
        // the exact value and the table_size_log needed.
        let mut oor_count: usize = 0;
        let mut max_val: i128 = 0;
        let mut neg_count: usize = 0;
        for i in 0..n {
            let v = f_to_int(data.index(i));
            let in_range = v >= 0 && (v as u128) < table_size as u128;
            if !in_range {
                oor_count += 1;
                if v < 0 { neg_count += 1; } else if v > max_val { max_val = v; }
            }
            let table_index = if in_range { v as usize } else { 0 };
            selection.push((i, table_index));
        }
        if oor_count > 0 {
            let needed = if max_val > 0 {
                (128 - (max_val as u128).leading_zeros()) as usize
            } else { self.table_size_log + 1 };
            eprintln!(
                "[range] WARNING: NonNegative range check — {}/{} value(s) fall outside \
                 the table [0, 2^{}) ({} negative, max overflow value = {}). These clamp to \
                 index 0, so the proof WILL fail to verify. Raise table_size_log to >= {} \
                 (use a config with a larger table_size_log, e.g. llama2_config.yaml for \
                 hidden>=2048).",
                oor_count, n, self.table_size_log, neg_count, max_val, needed.max(self.table_size_log + 1)
            );
        }
        let aux_poly = SelectionPolynomial::new(num_var, self.table_size_log, selection).to_sparse_dag();
        let aux = Witness {
            shape: x.shape.clone(),
            data: Some(Box::new(aux_poly)),
            poly_type: PolyType::Sparse,
            data_type: x.data_type,
            sf: 0,
            role: Role::Auxiliary,
        };
        vec![aux]
    }

    /// The aux production is deterministic memory-traffic-bound work; on a
    /// signed-int-classification check there's no algebraic shortcut to
    /// gain from a GPU kernel beyond saturating memory bandwidth. CPU is
    /// the documented default (philosophy rule #7).
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        self.run(inputs)
    }

    /// Empty — the DAG's `prove_range` handles the lookup proof.
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

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    use crate::dag::DataType;
    use crate::poly::SparseMLPoly;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn build_input(shape: Vec<usize>, vals: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Int,
            0,
            Role::Input,
        )
    }

    #[test]
    fn nonneg_run_produces_one_nonzero_per_row() {
        // shape [4], table covers [0, 8).
        let x = build_input(vec![4], vec![0, 1, 5, 7]);
        let nn = NonNegative::new(3);
        let out = nn.run(&[&x]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].poly_type, PolyType::Sparse);
        let sp = out[0]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .expect("aux is SparseMLPoly");
        // n = 2 (input rows), t = 3 → ambient n + t = 5.
        assert_eq!(sp.n, 5);
        assert_eq!(sp.num_nonzero(), 4);
        // Selection encodes (i, x_i).
        let sel = &sp.selection.selection;
        let mut by_input: std::collections::HashMap<usize, usize> =
            sel.iter().copied().collect();
        assert_eq!(by_input.remove(&0), Some(0));
        assert_eq!(by_input.remove(&1), Some(1));
        assert_eq!(by_input.remove(&2), Some(5));
        assert_eq!(by_input.remove(&3), Some(7));
    }

    #[test]
    fn nonneg_run_pads_to_pow_of_two_rows() {
        let x = build_input(vec![3], vec![0, 1, 2]); // padded to length-4 MLE
        let nn = NonNegative::new(3);
        let out = nn.run(&[&x]);
        let sp = out[0]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        // 4 rows total — the padding row maps the zero value to table index 0.
        assert_eq!(sp.num_nonzero(), 4);
        // Row 3 (padding) has value 0 → table_index 0 → entry at idx 3 + 0·4 = 3.
        assert!(sp.evaluations.contains_key(&3));
    }

    /// Negative values (stored as `q - |v|`) and values ≥ 2^t both clamp to
    /// table index 0. The selection is "wrong" in the sense that it doesn't
    /// recover the original input; the lookup verifier catches it.
    #[test]
    fn nonneg_run_clamps_out_of_range_to_zero() {
        let neg_one = (AlmostGoldilocksField(0) - AlmostGoldilocksField(1)).reduce().0;
        let x = build_input(vec![4], vec![0, neg_one, 100, 1]); // t=4 → table covers [0, 16)
        let nn = NonNegative::new(4);
        let out = nn.run(&[&x]);
        let sp = out[0]
            .data
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<SparseMLPoly>()
            .unwrap();
        let sel: std::collections::HashMap<usize, usize> =
            sp.selection.selection.iter().copied().collect();
        assert_eq!(sel[&0], 0);
        assert_eq!(sel[&1], 0, "negative clamps to 0");
        assert_eq!(sel[&2], 0, "out-of-range high clamps to 0");
        assert_eq!(sel[&3], 1);
    }

    #[test]
    fn nonneg_prove_is_empty() {
        let x = build_input(vec![4], vec![0, 1, 2, 3]);
        let nn = NonNegative::new(3);
        let mut t = Transcript::new(b"nn");
        let (proofs, claims) = nn.prove(&[&x], &[0], &[], &mut t);
        assert!(proofs.is_empty());
        assert!(claims.is_empty());
    }
}
