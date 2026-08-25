use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksField, GoldilocksExt2, GOLDILOCKS_PRIME};
use goldilocks_cuda::conv::{relu_helper as gpu_relu_helper, zero_buffer as gpu_zero_buffer};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::cpu_ext2_prover::CpuLinearSumcheckProverExt2;
use crate::sumcheck::{SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_mul, ext2_sub, get_n};

/// ReLUHelper: advice op that computes neg = max(0, -x).
/// Input: x (1 input) → Output: neg (1 output)
/// Combined with Add(x, neg) → y = max(0, x).
#[derive(Clone, Debug)]
pub struct ReLUHelper;

impl BasicBlock for ReLUHelper {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];
        let evals = x.data.as_ref().unwrap().evaluations_ref();
        let n = get_n(&x.shape);
        let size = 1usize << n;
        let p = GOLDILOCKS_PRIME;
        let half_p = p / 2;

        let mut neg_data = vec![GoldilocksField(0); size];
        for i in 0..size {
            let v = evals[i].0;
            if v > half_p {
                // x_i is negative (signed), neg_i = -x_i = p - x_i
                neg_data[i] = GoldilocksField(p - v);
            }
            // else x_i is non-negative, neg_i = 0
        }

        vec![Witness::new(x.shape.clone(), neg_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];
        let n = get_n(&x.shape);
        let size = 1usize << n;
        let d_x = x.as_device_buf();
        let mut d_neg = DeviceBuffer::<u64>::new(size).expect("ReLUHelper: alloc");
        gpu_relu_helper(&d_x, &mut d_neg, size).expect("ReLUHelper: gpu kernel failed");
        vec![Witness::new_device(x.shape.clone(), Arc::new(d_neg), DataType::Uint, 0, Role::Output)]
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

/// ProductZeroCheck: proves that A(x) * B(x) = 0 for all boolean x.
///
/// Inputs: A (edge 0), B (edge 1).
/// Output: certificate polynomial (all zeros).
///
/// Proof: degree-3 sumcheck over Σ_x eq(r, x) * A(x) * B(x) = 0,
/// where r is the evaluation point from the claim on the certificate output.
#[derive(Clone, Debug)]
pub struct ProductZeroCheck;

impl BasicBlock for ProductZeroCheck {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let n = get_n(&a.shape);
        let size = 1usize << n;

        // Certificate polynomial: all zeros
        let cert_data = vec![GoldilocksField(0); size];
        vec![Witness::new(a.shape.clone(), cert_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let n = get_n(&a.shape);
        let size = 1usize << n;
        let mut d_zero = DeviceBuffer::<u64>::new(size).expect("ProductZeroCheck: alloc");
        gpu_zero_buffer(&mut d_zero, size).expect("ProductZeroCheck: zero failed");
        vec![Witness::new_device(a.shape.clone(), Arc::new(d_zero), DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // witnesses layout: [A, B, cert] (inputs then outputs)
        // edge_ids layout: [a_edge, b_edge, cert_edge]
        // out_claims: [cert_claim] at point r with eval = 0
        let cert_claim = out_claims[0];
        let r = &cert_claim.point;
        let n = r.len();

        let a_edge = edge_ids[0];
        let b_edge = edge_ids[1];
        let a_data = witnesses[0].data.as_ref().unwrap();
        let b_data = witnesses[1].data.as_ref().unwrap();
        let size = 1usize << n;

        // Build eq(r, ·) table
        let eq_table = evaluate_lagrange_basis_ext2(r);

        // Build A and B as Ext2 vectors
        let a_evals = a_data.evaluations_ref();
        let b_evals = b_data.evaluations_ref();
        let a_ext2: Vec<GoldilocksExt2> = (0..size)
            .map(|i| GoldilocksExt2::from_base(a_evals[i]))
            .collect();
        let b_ext2: Vec<GoldilocksExt2> = (0..size)
            .map(|i| GoldilocksExt2::from_base(b_evals[i]))
            .collect();

        // Run degree-3 sumcheck: Σ_x eq(r,x) * A(x) * B(x) = 0
        let mut prover = CpuLinearSumcheckProverExt2::new(n, 3, transcript);
        let proof = prover.prove(
            &mut [eq_table, a_ext2, b_ext2],
            transcript,
        );

        // Create claims on A and B at sumcheck challenge point
        let u = &prover.challenges;
        let a_eval = a_data.evaluate_at_point_ext2(u);
        let b_eval = b_data.evaluate_at_point_ext2(u);

        let a_claim = Claim {
            edge_id: a_edge,
            sparse_id: 0,
            point: u.clone(),
            eval: a_eval,
        };
        let b_claim = Claim {
            edge_id: b_edge,
            sparse_id: 0,
            point: u.clone(),
            eval: b_eval,
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
        // claims: [a_claim, b_claim, cert_claim]
        let cert_claim = claims.last().unwrap();
        let a_claim = claims[0];
        let b_claim = claims[1];
        let r = &cert_claim.point;
        let n = r.len();

        // Verify sumcheck with expected_sum = cert_claim.eval (= 0 for honest prover)
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            cert_claim.eval,
            n,
            3, // 3 polynomials: eq, A, B
            transcript,
        );
        if !ok {
            println!("ProductZeroCheck: sumcheck verification failed");
            return false;
        }

        // Compute eq(r, u) where u = challenges
        let one = GoldilocksExt2::one();
        let mut eq_eval = one;
        for i in 0..n {
            let ri = r[i];
            let ui = challenges[i];
            // eq(r_i, u_i) = r_i * u_i + (1 - r_i) * (1 - u_i)
            let term1 = ext2_mul(ri, ui);
            let term2 = ext2_mul(ext2_sub(one, ri), ext2_sub(one, ui));
            eq_eval = ext2_mul(eq_eval, ext2_add(term1, term2));
        }

        // Check: eq(r, u) * a_eval * b_eval == final_eval
        let expected_final = ext2_mul(ext2_mul(eq_eval, a_claim.eval), b_claim.eval);
        if expected_final != sumcheck_proofs[0].final_eval {
            println!("ProductZeroCheck: final eval check failed");
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DataType, Role, Witness};
    use goldilocks_cuda::{GoldilocksField, GOLDILOCKS_PRIME};

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_relu_helper_run() {
        let helper = ReLUHelper;
        let p = GOLDILOCKS_PRIME;

        // x = [5, 0, p-3, p-10] (signed: [5, 0, -3, -10])
        let x = make_witness(vec![4], vec![5, 0, p - 3, p - 10]);
        let result = helper.run(&[&x]);
        let neg = &result[0];

        assert_eq!(neg.shape, vec![4]);
        // neg = max(0, -x): [0, 0, 3, 10]
        assert_eq!(neg.data.as_ref().unwrap().index(0), GoldilocksField(0));
        assert_eq!(neg.data.as_ref().unwrap().index(1), GoldilocksField(0));
        assert_eq!(neg.data.as_ref().unwrap().index(2), GoldilocksField(3));
        assert_eq!(neg.data.as_ref().unwrap().index(3), GoldilocksField(10));
    }

    #[test]
    fn test_relu_prove_verify() {
        use crate::commit::basefold::{GpuCommitmentStore, BasefoldCommitKey, BasefoldCommitmentData, BasefoldVerifierKey};
        use crate::dag::{builder::DagBuilder, DataType};
        use crate::transcript::Transcript;
        use goldilocks_cuda::basefold::BasefoldTable;
        use plonky2::util::timing::TimingTree;

        goldilocks_cuda::init().expect("CUDA init failed");
        let p = GOLDILOCKS_PRIME;

        let mut g = DagBuilder::new();
        let x_edge = g.input(vec![4], DataType::Uint);
        let y_edge = g.relu(x_edge);
        assert_ne!(x_edge, y_edge, "relu should produce a new edge");

        let (dag, mut witnesses) = g.compile();

        // Set x data: [5, 0, p-3, p-10] (signed: [5, 0, -3, -10])
        let x_data = Witness::new(
            vec![4],
            vec![
                GoldilocksField(5),
                GoldilocksField(0),
                GoldilocksField(p - 3),
                GoldilocksField(p - 10),
            ],
            DataType::Uint,
            0,
            Role::Input,
        );
        dag.run(&mut witnesses, &[(x_edge, x_data)]);

        // Check y = max(0, x): [5, 0, 0, 0]
        let y_data = witnesses[y_edge][0].data.as_ref().unwrap();
        assert_eq!(y_data.index(0), GoldilocksField(5));
        assert_eq!(y_data.index(1), GoldilocksField(0));
        assert_eq!(y_data.index(2), GoldilocksField(0));
        assert_eq!(y_data.index(3), GoldilocksField(0));

        // Commit
        let key = BasefoldCommitKey::default();
        let max_nv = witnesses.iter()
            .filter_map(|ws| ws.first().and_then(|w| w.data.as_ref().map(|d| d.n())))
            .filter(|&n| n <= 22)
            .max().unwrap_or(4);
        let mut commitments: Vec<Option<BasefoldCommitmentData>> = vec![None; dag.num_edges()];
        let mut gpu_store = GpuCommitmentStore::new(max_nv, key.log_rate, key.seed, dag.num_edges());
        dag.commit(&key, &witnesses, &mut commitments, &mut gpu_store, None);

        // Prove
        let mut transcript = Transcript::new(b"test_relu");
        let mut timing = TimingTree::new("test", log::Level::Info);
        let (node_proofs, edge_proofs, range_proof, two_pow_proof, reducer_proofs) =
            dag.prove(&key, &mut witnesses, &commitments, &gpu_store, &gpu_store.table, &mut transcript, &mut timing);

        // Verify
        let vk = BasefoldVerifierKey::from(&key);
        let table = BasefoldTable::generate(max_nv, vk.log_rate, max_nv, vk.seed);
        let mut verify_transcript = Transcript::new(b"test_relu");
        let verified = dag.verify(
            &node_proofs,
            &edge_proofs,
            &range_proof,
            &two_pow_proof,
            &reducer_proofs,
            &witnesses,
            &vk,
            &commitments,
            &table,
            &mut verify_transcript,
        );
        assert!(verified, "ReLU prove/verify should pass");
    }
}
