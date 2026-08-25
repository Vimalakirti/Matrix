pub mod linear_prover;
pub mod general_prover;
pub mod gpu_prover;
pub mod cpu_ext2_prover;
pub mod verifier;

pub use linear_prover::LinearSumcheckProver;
pub use general_prover::GeneralLinearSumcheckProver;
pub use gpu_prover::GpuLinearSumcheckProver;
pub use cpu_ext2_prover::CpuLinearSumcheckProverExt2;
pub use verifier::SumcheckVerifier;

use goldilocks_cuda::GoldilocksExt2;
use serde::{Deserialize, Serialize};

/// Proof produced by the sumcheck protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof {
    /// Final evaluation of the polynomial product at the challenge point.
    pub final_eval: GoldilocksExt2,
    /// Round messages: for each round, the evaluations of the round polynomial.
    pub round_messages: Vec<Vec<GoldilocksExt2>>,
}

/// Trait for sumcheck provers.
pub trait SumcheckProver {
    type Instance;

    fn new(num_var: usize, num_polys: usize, transcript: &mut crate::transcript::Transcript) -> Self;
    fn prove(
        &mut self,
        instances: &Self::Instance,
        transcript: &mut crate::transcript::Transcript,
    ) -> SumcheckProof;
}
