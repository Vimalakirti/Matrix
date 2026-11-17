//! [`ExpHelper`] and [`TwoPow`] — exp decomposition.
//!
//! `ExpHelper` decomposes `x = r - Ls*k` using exact public `Ls`. The
//! non-saturating profile requires `x<=0`, `k<2^num_bits`, and the canonical
//! remainder `0<=Ls/2-r<Ls`. DagBuilder::compile attaches real range roots
//! even to standalone raw helpers; the Morpheus verifier validates that wiring.
//! Normalization advice and causal independence are separate obligations.
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
use crate::SF_LOG;

/// Compact, non-saturating exp profile: eight bits also cover the existing
/// -100 causal-mask sentinel. Inputs outside the declared table FAIL CLOSED.
/// This does not repair the causal denominator or normalization advice.
pub const K_BITS: usize = 8;

pub fn exp_bits() -> usize {
    let bits = match std::env::var("ZK4_EXP_BITS") {
        Ok(s) => s.parse::<usize>().expect("ZK4_EXP_BITS must be an integer"),
        Err(std::env::VarError::NotPresent) => K_BITS,
        Err(std::env::VarError::NotUnicode(_)) => panic!("ZK4_EXP_BITS must be valid Unicode"),
    };
    assert!((1..=8).contains(&bits), "supported exp selector width is 1..=8");
    bits
}

/// Exact public ln(2) quantization, indexed by activation scale (not Qe).
/// These integers, and the operation order, define the numerical program.
pub const EXP_LS: [i128; 31] = [1,1,3,6,11,22,44,89,177,355,710,1420,2839,5678,11357,22713,45426,90852,181704,363409,726817,1453635,2907270,5814540,11629080,23258160,46516320,93032640,186065279,372130559,744261118];
pub fn exp_ln2(sf: usize) -> i128 {
    *EXP_LS.get(sf).expect("exp activation scale must be 0..=30")
}

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
        assert_eq!(x.sf, *SF_LOG, "exp input must use the registered activation scale");
        assert!((1..=8).contains(&self.num_bits), "unsupported exp selector width");
        let ln2 = exp_ln2(x.sf);
        let neg_ln2_f = int_to_f(-ln2);
        let table_size = 1usize << self.num_bits;

        let mut r_data = vec![AlmostGoldilocksField(0); n];
        let mut selection = Vec::with_capacity(n);
        let data = x.data.as_ref().unwrap();
        for i in 0..n {
            let x_i = data.index(i);
            let x_num = f_to_int(x_i);
            assert!(x_num <= 0, "exp profile requires nonpositive input");
            let k = (-x_num + ln2 / 2) / ln2;
            assert!(k < table_size as i128, "exp input outside registered selector domain");
            // Unique integer decomposition, no floating rounding or index clamp.
            let k_f = int_to_f(k);
            r_data[i] = x_i - k_f * neg_ln2_f;
            let table_index = k as usize;
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
            channel_shift: None,
        };
        vec![r, aux]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Exact integer witness generation; CPU is the documented default.
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
            // table[k] = 2^(15 − k) for k ∈ [0, 16), 0 beyond (wider ExpHelper tables).
            y_data[input_index] = AlmostGoldilocksField(if table_index < 16 { 1u64 << (15 - table_index) } else { 0 });
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
        let ln2 = exp_ln2(*SF_LOG);
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
    fn exphelper_rejects_positive_and_out_of_domain_without_clamping() {
        let l=exp_ln2(*SF_LOG); let h=ExpHelper::new(8);
        let bound=256*l-l/2-1;
        for v in [1,-bound-1] {
            let x=build_input(vec![1],vec![v]);
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.run(&[&x]))).is_err());
        }
        for v in [0,-bound,-l/2,-(l-l/2)] {
            let x=build_input(vec![1],vec![v]); let out=h.run(&[&x]);
            let r=crate::util::arith::f_to_int(out[0].data.as_ref().unwrap().index(0));
            assert!((0..l).contains(&(l/2-r)));
        }
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
