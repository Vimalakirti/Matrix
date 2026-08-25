//! [`ReLUHelper`] — advice op emitting `neg = max(0, −x)`.
//! [`ProductZeroCheck`] — degree-3 sumcheck proving `A(x) · B(x) = 0` for all
//! Boolean `x` (the certificate output is the all-zero polynomial).

use std::sync::Arc;

use almost_goldilocks_cuda::conv::{relu_helper as gpu_relu_helper, zero_buffer as gpu_zero_buffer};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::{AlmostGoldilocksField, ALMOST_GOLDILOCKS_PRIME};
use almost_goldilocks_cuda::memory::DeviceBuffer;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, ext2_sub, get_n};

// ============================================================================
// ReLUHelper
// ============================================================================

#[derive(Clone, Debug)]
pub struct ReLUHelper;

impl BasicBlock for ReLUHelper {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ReLUHelper expects 1 input");
        let x = inputs[0];
        let evals = x.data.as_ref().unwrap().evaluations_ref();
        let n = get_n(&x.shape);
        let size = 1usize << n;
        let half_q = ALMOST_GOLDILOCKS_PRIME / 2;

        let mut neg_data = vec![AlmostGoldilocksField(0); size];
        for i in 0..size {
            let v = evals[i].reduce().0;
            if v > half_q {
                neg_data[i] = AlmostGoldilocksField(ALMOST_GOLDILOCKS_PRIME - v);
            }
        }
        vec![Witness::new(x.shape.clone(), neg_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    /// GPU path — uses `agl_relu_helper` (1 fused-element kernel).
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ReLUHelper expects 1 input");
        let x = inputs[0];
        let n = get_n(&x.shape);
        let size = 1usize << n;
        let d_x = x.as_device_buf();
        let mut d_neg = DeviceBuffer::<u64>::new(size).expect("ReLUHelper: alloc failed");
        gpu_relu_helper(&d_x, &mut d_neg, size).expect("ReLUHelper: GPU kernel failed");
        vec![Witness::new_device(
            x.shape.clone(),
            Arc::new(d_neg),
            DataType::Uint,
            0,
            Role::Output,
        )]
    }

    /// Advice op: empty prove. Soundness comes from the downstream
    /// composition (NonNegative on `neg`, NonNegative on `y = x + neg`, and
    /// ProductZeroCheck on `(neg, y)`) — all wired by the DAG builder.
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

// ============================================================================
// ProductZeroCheck
// ============================================================================

#[derive(Clone, Debug)]
pub struct ProductZeroCheck;

impl BasicBlock for ProductZeroCheck {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "ProductZeroCheck expects 2 inputs");
        let a = inputs[0];
        let n = get_n(&a.shape);
        let size = 1usize << n;
        let cert = vec![AlmostGoldilocksField(0); size];
        vec![Witness::new(a.shape.clone(), cert, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "ProductZeroCheck expects 2 inputs");
        let a = inputs[0];
        let n = get_n(&a.shape);
        let size = 1usize << n;
        let mut d_zero = DeviceBuffer::<u64>::new(size).expect("PZC: alloc");
        gpu_zero_buffer(&mut d_zero, size).expect("PZC: zero_buffer failed");
        vec![Witness::new_device(
            a.shape.clone(),
            Arc::new(d_zero),
            DataType::Uint,
            0,
            Role::Output,
        )]
    }

    /// Degree-3 sumcheck `Σ_x eq(r, x) · A(x) · B(x) = 0`.
    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // witnesses: [A, B, cert], edges: [a, b, cert], claims: [cert_claim].
        let cert_claim = out_claims[0];
        let r = &cert_claim.point;
        let n = r.len();
        let size = 1usize << n;

        let a_data = witnesses[0].data.as_ref().unwrap();
        let b_data = witnesses[1].data.as_ref().unwrap();
        let eq_table = evaluate_lagrange_basis_ext2(r);
        let a_evals = a_data.evaluations_ref();
        let b_evals = b_data.evaluations_ref();
        let a_ext2: Vec<_> = (0..size)
            .map(|i| AlmostGoldilocksExt2::from_base(a_evals[i]))
            .collect();
        let b_ext2: Vec<_> = (0..size)
            .map(|i| AlmostGoldilocksExt2::from_base(b_evals[i]))
            .collect();

        let mut prover = CpuLinearSumcheckProverExt2::new(n, 3, transcript);
        let proof = prover.prove(&mut [eq_table, a_ext2, b_ext2], transcript);
        let u = prover.challenges.clone();

        let a_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: u.clone(),
            eval: a_data.evaluate_at_point_ext2(&u),
        };
        let b_claim = Claim {
            edge_id: edge_ids[1],
            sparse_id: 0,
            point: u,
            eval: b_data.evaluate_at_point_ext2(&prover.challenges),
        };
        (vec![proof], vec![a_claim, b_claim])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        let cert_claim = claims.last().unwrap();
        let a_claim = &claims[0];
        let b_claim = &claims[1];
        let r = &cert_claim.point;
        let n = r.len();
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            cert_claim.eval,
            n,
            3,
            transcript,
        );
        if !ok { return false; }
        let one = AlmostGoldilocksExt2::one();
        let mut eq_eval = one;
        for i in 0..n {
            let r_i = r[i];
            let u_i = challenges[i];
            let term = ext2_add(
                ext2_mul(r_i, u_i),
                ext2_mul(ext2_sub(one, r_i), ext2_sub(one, u_i)),
            );
            eq_eval = ext2_mul(eq_eval, term);
        }
        let expected = ext2_mul(ext2_mul(eq_eval, a_claim.eval), b_claim.eval);
        ext2_field_eq(expected, sumcheck_proofs[0].final_eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(agl(v))
    }

    fn make_witness(shape: Vec<usize>, vals: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Uint,
            0,
            Role::Input,
        )
    }

    #[test]
    fn relu_helper_emits_signed_magnitude_for_negatives() {
        let p = ALMOST_GOLDILOCKS_PRIME;
        let x = make_witness(vec![4], vec![5, 0, p - 3, p - 10]);
        let out = ReLUHelper.run(&[&x]);
        let neg = out[0].data.as_ref().unwrap();
        assert_eq!(neg.index(0), agl(0));
        assert_eq!(neg.index(1), agl(0));
        assert_eq!(neg.index(2), agl(3));
        assert_eq!(neg.index(3), agl(10));
    }

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    #[test]
    fn relu_helper_run_gpu_matches_cpu() {
        if !cuda_ready() { eprintln!("skipping GPU test: CUDA not available"); return; }
        let p = ALMOST_GOLDILOCKS_PRIME;
        let x = make_witness(vec![8], vec![1, 2, 3, p - 1, p - 2, p - 3, 0, p / 2]);
        let cpu = ReLUHelper.run(&[&x]);
        let gpu = ReLUHelper.run_gpu(&[&x]);
        for i in 0..8 {
            let want = cpu[0].data.as_ref().unwrap().index(i);
            let got = gpu[0].data.as_ref().unwrap().index(i);
            assert_eq!(want.reduce(), got.reduce(), "i = {}", i);
        }
    }

    /// Honest prover with `A · B = 0` (one factor is zero everywhere)
    /// produces a sumcheck that verifies. Then check tampering rejects.
    #[test]
    fn product_zero_check_roundtrip() {
        // A = [1, 2, 3, 4], B = [0, 0, 0, 0]: product is zero everywhere.
        let a = make_witness(vec![4], vec![1, 2, 3, 4]);
        let b = make_witness(vec![4], vec![0, 0, 0, 0]);
        let cert_outs = ProductZeroCheck.run(&[&a, &b]);
        let cert = &cert_outs[0];

        let pt = vec![lift(3), lift(5)];
        let cert_eval = cert.data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        assert_eq!(cert_eval, AlmostGoldilocksExt2::zero());
        let cert_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point: pt,
            eval: cert_eval,
        };

        let mut t_prove = Transcript::new(b"pzc");
        let (proofs, claims) = ProductZeroCheck.prove(
            &[&a, &b, cert],
            &[0, 1, 2],
            &[&cert_claim],
            &mut t_prove,
        );
        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 2);

        let mut t_verify = Transcript::new(b"pzc");
        let all = [&claims[0], &claims[1], &cert_claim];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(ProductZeroCheck.verify(&[&a, &b, cert], &all, &proof_refs, &mut t_verify));
    }

    #[test]
    fn product_zero_check_rejects_tampered_eval() {
        let a = make_witness(vec![4], vec![1, 2, 3, 4]);
        let b = make_witness(vec![4], vec![0, 0, 0, 0]);
        let cert_outs = ProductZeroCheck.run(&[&a, &b]);
        let cert = &cert_outs[0];
        let pt = vec![lift(3), lift(5)];
        let cert_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point: pt,
            eval: AlmostGoldilocksExt2::zero(),
        };
        let mut t_prove = Transcript::new(b"pzc-tamper");
        let (mut proofs, claims) =
            ProductZeroCheck.prove(&[&a, &b, cert], &[0, 1, 2], &[&cert_claim], &mut t_prove);
        proofs[0].final_eval = proofs[0].final_eval + lift(1);
        let mut t_verify = Transcript::new(b"pzc-tamper");
        let all = [&claims[0], &claims[1], &cert_claim];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(!ProductZeroCheck.verify(&[&a, &b, cert], &all, &proof_refs, &mut t_verify));
    }
}
