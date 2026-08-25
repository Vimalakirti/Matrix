//! GPU-accelerated linear sumcheck prover over Ext2.
//!
//! Wraps [`almost_goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2`].
//! Drop-in replacement for [`crate::sumcheck::LinearSumcheckProver`] over
//! polynomials large enough that GPU launch overhead is amortized
//! (`n > ZK_GPU_SUMCHECK_THRESHOLD`, default 14).

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use almost_goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;

use crate::poly::DenseMLPoly;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::ext2_mul;

pub struct GpuLinearSumcheckProver {
    gpu_state: Option<GpuSumcheckStateExt2>,
    pub num_var: usize,
    pub num_poly: usize,
    pub challenges: Vec<AlmostGoldilocksExt2>,
    final_evals: Vec<AlmostGoldilocksExt2>,
}

impl GpuLinearSumcheckProver {
    pub fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript) -> Self {
        transcript.append_u64(b"num_var", num_var as u64);
        transcript.append_u64(b"num_poly", num_polys as u64);
        Self {
            gpu_state: None,
            num_var,
            num_poly: num_polys,
            challenges: Vec::new(),
            final_evals: Vec::new(),
        }
    }

    /// Prove using base-field polynomials. The wrapper uploads them to the
    /// GPU as Ext2 (c1 = 0).
    pub fn prove(
        &mut self,
        instances: &[DenseMLPoly],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert!(!instances.is_empty());
        assert_eq!(instances[0].len(), 1 << self.num_var);
        assert_eq!(instances.len(), self.num_poly);

        let eval_refs: Vec<&[AlmostGoldilocksField]> =
            instances.iter().map(|p| p.evaluations.as_slice()).collect();
        let gpu_state = GpuSumcheckStateExt2::new_from_base(&eval_refs)
            .expect("GpuSumcheckStateExt2::new_from_base failed");
        self.run_loop(gpu_state, transcript)
    }

    /// Prove using polynomials already in Ext2 (e.g., eq tables).
    pub fn prove_ext2(
        &mut self,
        instances: &[Vec<AlmostGoldilocksExt2>],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert!(!instances.is_empty());
        assert_eq!(instances[0].len(), 1 << self.num_var);
        assert_eq!(instances.len(), self.num_poly);

        let eval_refs: Vec<&[AlmostGoldilocksExt2]> =
            instances.iter().map(|p| p.as_slice()).collect();
        let gpu_state = GpuSumcheckStateExt2::new(&eval_refs)
            .expect("GpuSumcheckStateExt2::new failed");
        self.run_loop(gpu_state, transcript)
    }

    /// Prove against an already-resident GPU state (caller-managed device
    /// buffers, e.g. inside a fused commit→prove pipeline).
    pub fn prove_gpu_resident(
        &mut self,
        gpu_state: GpuSumcheckStateExt2,
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        self.run_loop(gpu_state, transcript)
    }

    fn run_loop(
        &mut self,
        mut gpu_state: GpuSumcheckStateExt2,
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        let mut round_messages = Vec::with_capacity(self.num_var);
        for _round in 0..self.num_var {
            let round_message = gpu_state
                .compute_round_message()
                .expect("GPU compute_round_message failed");
            for msg in &round_message {
                transcript.append_ext2(b"round_message", msg);
            }
            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(round_message);
            gpu_state.fold(challenge).expect("GPU fold failed");
            self.challenges.push(challenge);
        }

        self.final_evals = gpu_state
            .final_evaluations()
            .expect("GPU final_evaluations failed");
        let mut final_eval = AlmostGoldilocksExt2::one();
        for &e in &self.final_evals {
            final_eval = ext2_mul(final_eval, e);
        }

        self.gpu_state = Some(gpu_state);
        SumcheckProof { final_eval, round_messages }
    }

    pub fn final_eval(&self, poly_idx: usize) -> AlmostGoldilocksExt2 {
        self.final_evals[poly_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::SumcheckVerifier;
    use crate::util::arith::ext2_field_eq;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    /// CUDA-dependent — skipped when no device is visible.
    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    /// End-to-end GPU sumcheck on a small problem, verified by the CPU
    /// [`SumcheckVerifier`]. Validates the wire format and challenge schedule
    /// — same transcript labels as CPU, so this exercises that the two are
    /// truly interoperable.
    #[test]
    fn gpu_prove_verifies_via_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let n_var = 3;
        let p1 = DenseMLPoly::new(
            n_var,
            (0..(1 << n_var) as u64).map(|i| agl(i * 7 + 3)).collect(),
        );
        let p2 = DenseMLPoly::new(
            n_var,
            (0..(1 << n_var) as u64).map(|i| agl(i * 5 + 11)).collect(),
        );
        let mut claim = AlmostGoldilocksExt2::zero();
        for i in 0..(1 << n_var) {
            claim = claim
                + AlmostGoldilocksExt2::from_base(p1.evaluations[i])
                    * AlmostGoldilocksExt2::from_base(p2.evaluations[i]);
        }

        let mut t_prove = Transcript::new(b"gpu-roundtrip");
        let mut prover = GpuLinearSumcheckProver::new(n_var, 2, &mut t_prove);
        let proof = prover.prove(&[p1, p2], &mut t_prove);

        let mut t_verify = Transcript::new(b"gpu-roundtrip");
        let (ok, _) = SumcheckVerifier::verify(&proof, claim, n_var, 2, &mut t_verify);
        assert!(ok, "GPU sumcheck verification failed");
    }

    /// GPU vs CPU-Ext2 cross-check: same polynomials and same transcript
    /// label produce bit-identical round messages and final eval.
    #[test]
    fn gpu_matches_cpu_ext2() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let n_var = 3;
        let size = 1 << n_var;
        let p1: Vec<_> = (0..size as u64)
            .map(|i| AlmostGoldilocksExt2::from_base(agl(i * 7 + 3)))
            .collect();
        let p2: Vec<_> = (0..size as u64)
            .map(|i| AlmostGoldilocksExt2::from_base(agl(i * 5 + 11)))
            .collect();

        let mut t_gpu = Transcript::new(b"x");
        let mut gpu = GpuLinearSumcheckProver::new(n_var, 2, &mut t_gpu);
        let proof_gpu = gpu.prove_ext2(&[p1.clone(), p2.clone()], &mut t_gpu);

        let mut t_cpu = Transcript::new(b"x");
        let mut cpu = crate::sumcheck::CpuLinearSumcheckProverExt2::new(n_var, 2, &mut t_cpu);
        let proof_cpu = cpu.prove(&mut [p1, p2], &mut t_cpu);

        assert_eq!(proof_gpu.round_messages.len(), proof_cpu.round_messages.len());
        for (rg, rc) in proof_gpu.round_messages.iter().zip(proof_cpu.round_messages.iter()) {
            assert_eq!(rg.len(), rc.len());
            for (a, b) in rg.iter().zip(rc.iter()) {
                assert!(ext2_field_eq(*a, *b), "round msg mismatch");
            }
        }
        assert!(ext2_field_eq(proof_gpu.final_eval, proof_cpu.final_eval));
    }
}
