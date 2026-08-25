use goldilocks_cuda::{GoldilocksExt2, DeviceBuffer, Ext2Batch};
use goldilocks_cuda::eq_lagrange::ext2_eq_dp_all_device;
use goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;

use std::sync::OnceLock;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, GpuLinearSumcheckProver, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{calc_pow_vec_ext2, get_n, ext2_add, ext2_mul, ext2_sub};

/// Threshold: use GPU sumcheck only when n > this value.
/// Override with ZK_GPU_SUMCHECK_THRESHOLD env var.
fn gpu_sumcheck_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_SUMCHECK_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(14)
    })
}

/// Reducer block: combines multiple claims on the same polynomial into one
/// via random linear combination sumcheck.
#[derive(Clone, Debug)]
pub struct Reducer;

impl BasicBlock for Reducer {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert!(inputs.len() == 1, "Reducer expects 1 input");
        vec![inputs[0].clone()]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert!(witnesses.len() == 1, "Reducer expects 1 input");
        let alpha: GoldilocksExt2 = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, out_claims.len());

        let x = witnesses[0];
        let n = x.data.as_ref().unwrap().n();
        let size = 1usize << n;

        if n <= gpu_sumcheck_threshold() {
            // === CPU path ===
            // Build eq_combined on CPU: Σ alpha^i * eq(claim_i.point, x)
            let mut eq_combined = vec![GoldilocksExt2::zero(); size];
            for (idx, claim) in out_claims.iter().enumerate() {
                let eq_table = evaluate_lagrange_basis_ext2(&claim.point);
                for j in 0..size {
                    eq_combined[j] = ext2_add(eq_combined[j], ext2_mul(alphas[idx], eq_table[j]));
                }
            }

            // Convert base-field witness to Ext2
            let x_evals = x.data.as_ref().unwrap().evaluations_ref();
            let x_ext2: Vec<GoldilocksExt2> = x_evals.iter()
                .map(|&v| GoldilocksExt2::from_base(v))
                .collect();

            let mut cpu_prover = CpuLinearSumcheckProverExt2::new(n, 2, transcript);
            let mut polys = vec![x_ext2, eq_combined];
            let sumcheck_proof = cpu_prover.prove(&mut polys, transcript);
            let challenges = cpu_prover.challenges.clone();

            let claim_x = Claim {
                edge_id: edge_ids[0],
                sparse_id: 0,
                point: challenges,
                eval: cpu_prover.final_eval(0),
            };

            return (vec![sumcheck_proof], vec![claim_x]);
        }

        // === GPU path ===
        // Build eq_combined on GPU: Σ alpha^i * eq(claim_i.point, x)
        let mut d_acc = DeviceBuffer::<u64>::new(size * 2).expect("alloc failed");
        // Zero-initialize accumulator
        let zeros = vec![0u64; size * 2];
        d_acc.copy_from_slice(&zeros).expect("zero init failed");

        for (idx, claim) in out_claims.iter().enumerate() {
            // Compute eq table on GPU
            let d_r = DeviceBuffer::<GoldilocksExt2>::from_slice(&claim.point)
                .expect("GPU upload failed");
            let log_n = claim.point.len();

            let (d_buf_a, d_buf_b, result_in_a) = ext2_eq_dp_all_device(&d_r, log_n)
                .expect("ext2_eq_dp_all_device failed");

            let d_eq_result = if result_in_a { &d_buf_a } else { &d_buf_b };

            // Copy eq result to a u64 buffer view
            let mut d_eq_u64 = DeviceBuffer::<u64>::new(size * 2).expect("alloc failed");
            unsafe {
                goldilocks_cuda::memcpy_dtod(
                    d_eq_u64.as_mut_ptr() as *mut std::os::raw::c_void,
                    d_eq_result.as_ptr() as *const std::os::raw::c_void,
                    size * 2 * std::mem::size_of::<u64>(),
                ).expect("D2D copy failed");
            }

            // acc += alpha^idx * eq
            Ext2Batch::scale_accumulate(alphas[idx], &d_eq_u64, &mut d_acc)
                .expect("scale_accumulate failed");
        }

        // Convert base-field witness to Ext2 on GPU
        let x_evals = x.data.as_ref().unwrap().evaluations_ref();
        let x_u64: Vec<u64> = x_evals.iter().map(|v| v.0).collect();
        let d_x_base = DeviceBuffer::<u64>::from_slice(&x_u64).expect("GPU upload failed");
        let mut d_x_ext2 = DeviceBuffer::<u64>::new(size * 2).expect("alloc failed");
        Ext2Batch::from_base(&d_x_base, &mut d_x_ext2).expect("base→Ext2 failed");

        // Build GPU sumcheck state from device buffers
        let buf_refs: Vec<&DeviceBuffer<u64>> = vec![&d_x_ext2, &d_acc];
        let gpu_state = GpuSumcheckStateExt2::from_device_buffers(&buf_refs, size)
            .expect("from_device_buffers failed");

        let mut gpu_prover = GpuLinearSumcheckProver::new(n, 2, transcript);
        let sumcheck_proof = gpu_prover.prove_gpu_resident(gpu_state, transcript);
        let challenges = gpu_prover.challenges.clone();

        let claim_x = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges.clone(),
            eval: gpu_prover.final_eval(0),
        };

        (vec![sumcheck_proof], vec![claim_x])
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        assert!(witnesses.len() == 1, "Reducer expects 1 input");
        let alpha: GoldilocksExt2 = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, claims.len() - 1);
        let x = witnesses[0];

        // Compute expected sum = Σ alpha^i * claim_i.eval
        let mut eval = GoldilocksExt2::zero();
        for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
            eval = ext2_add(eval, ext2_mul(claim.eval, alphas[i]));
        }

        let n_verify = get_n(&x.shape);
        let (verified, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            eval,
            n_verify,
            2,
            transcript,
        );
        if !verified {
            println!("verified reducer failed 1");
            return false;
        }

        // Compute eq_eval = Σ_i alpha_i * eq(challenges, claim_i.point)
        let one = GoldilocksExt2::one();
        let mut eq_eval = GoldilocksExt2::zero();
        for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
            let mut eq = GoldilocksExt2::one();
            for j in 0..challenges.len() {
                let r_j = challenges[j];
                let p_j = claim.point[j];
                let term = ext2_add(
                    ext2_mul(r_j, p_j),
                    ext2_mul(ext2_sub(one, r_j), ext2_sub(one, p_j)),
                );
                eq = ext2_mul(eq, term);
            }
            eq_eval = ext2_add(eq_eval, ext2_mul(alphas[i], eq));
        }

        // Final eval check: proof.final_eval == x_eval * eq_eval
        let x_eval = claims[claims.len() - 1].eval;
        let expected = ext2_mul(x_eval, eq_eval);
        if sumcheck_proofs[0].final_eval != expected {
            println!("verified reducer failed: final_eval check mismatch");
            return false;
        }
        verified
    }
}
