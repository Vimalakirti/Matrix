use goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;
use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

use crate::poly::DenseMLPoly;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::ext2_mul;

/// GPU-accelerated linear sumcheck prover over Ext2.
///
/// Drop-in replacement for LinearSumcheckProver that runs round message
/// computation and polynomial folding on GPU using Ext2 arithmetic.
pub struct GpuLinearSumcheckProver {
    gpu_state: Option<GpuSumcheckStateExt2>,
    pub num_var: usize,
    pub num_poly: usize,
    pub challenges: Vec<GoldilocksExt2>,
    final_evals: Vec<GoldilocksExt2>,
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

    /// Prove using base-field polynomials (converted to Ext2 on GPU upload).
    pub fn prove(
        &mut self,
        instances: &[DenseMLPoly],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert!(!instances.is_empty());
        let n_size = instances[0].len();
        assert_eq!(n_size, 1 << self.num_var);
        assert_eq!(instances.len(), self.num_poly);

        // Pack polynomial evaluations for GPU upload (base field → Ext2)
        let eval_refs: Vec<&[GoldilocksField]> =
            instances.iter().map(|p| p.evaluations.as_slice()).collect();

        let mut gpu_state =
            GpuSumcheckStateExt2::new_from_base(&eval_refs).expect("Failed to create GPU sumcheck state");

        let mut round_messages = Vec::new();

        for _round in 0..self.num_var {
            let round_message = gpu_state
                .compute_round_message()
                .expect("GPU round message failed");

            for msg in &round_message {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(round_message);

            gpu_state
                .fold(challenge)
                .expect("GPU fold failed");
            self.challenges.push(challenge);
        }

        // Get final evaluations from GPU
        self.final_evals = gpu_state
            .final_evaluations()
            .expect("GPU final eval failed");

        let mut final_eval = GoldilocksExt2::one();
        for &e in &self.final_evals {
            final_eval = ext2_mul(final_eval, e);
        }

        self.gpu_state = Some(gpu_state);

        SumcheckProof {
            final_eval,
            round_messages,
        }
    }

    /// Prove using Ext2 polynomials directly (e.g., eq polynomial already in Ext2).
    pub fn prove_ext2(
        &mut self,
        instances: &[Vec<GoldilocksExt2>],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert!(!instances.is_empty());
        let n_size = instances[0].len();
        assert_eq!(n_size, 1 << self.num_var);
        assert_eq!(instances.len(), self.num_poly);

        let eval_refs: Vec<&[GoldilocksExt2]> =
            instances.iter().map(|p| p.as_slice()).collect();

        let mut gpu_state =
            GpuSumcheckStateExt2::new(&eval_refs).expect("Failed to create GPU sumcheck state");

        let mut round_messages = Vec::new();

        for _round in 0..self.num_var {
            let round_message = gpu_state
                .compute_round_message()
                .expect("GPU round message failed");

            for msg in &round_message {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(round_message);

            gpu_state
                .fold(challenge)
                .expect("GPU fold failed");
            self.challenges.push(challenge);
        }

        self.final_evals = gpu_state
            .final_evaluations()
            .expect("GPU final eval failed");

        let mut final_eval = GoldilocksExt2::one();
        for &e in &self.final_evals {
            final_eval = ext2_mul(final_eval, e);
        }

        self.gpu_state = Some(gpu_state);

        SumcheckProof {
            final_eval,
            round_messages,
        }
    }

    /// Prove using a pre-built GPU sumcheck state (data already on GPU).
    pub fn prove_gpu_resident(
        &mut self,
        mut gpu_state: GpuSumcheckStateExt2,
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        let mut round_messages = Vec::new();

        for _round in 0..self.num_var {
            let round_message = gpu_state
                .compute_round_message()
                .expect("GPU round message failed");

            for msg in &round_message {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(round_message);

            gpu_state
                .fold(challenge)
                .expect("GPU fold failed");
            self.challenges.push(challenge);
        }

        self.final_evals = gpu_state
            .final_evaluations()
            .expect("GPU final eval failed");

        let mut final_eval = GoldilocksExt2::one();
        for &e in &self.final_evals {
            final_eval = ext2_mul(final_eval, e);
        }

        self.gpu_state = Some(gpu_state);

        SumcheckProof {
            final_eval,
            round_messages,
        }
    }

    /// Get the final evaluation for the polynomial at the given index.
    pub fn final_eval(&self, poly_idx: usize) -> GoldilocksExt2 {
        self.final_evals[poly_idx]
    }
}
