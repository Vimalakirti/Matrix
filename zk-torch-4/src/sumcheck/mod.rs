//! Sumcheck protocol provers and verifier (CPU + GPU paths).
//!
//! - [`linear_prover::LinearSumcheckProver`]: CPU prover over the base field.
//! - [`cpu_ext2_prover::CpuLinearSumcheckProverExt2`]: CPU prover over Ext2.
//! - [`general_prover::GeneralLinearSumcheckProver`]: degree-`d+1` CPU prover
//!   with an explicit `eq(r, x)` factor.
//! - [`gpu_prover::GpuLinearSumcheckProver`]: GPU-backed Ext2 prover wrapping
//!   `almost_goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2`.
//! - [`verifier::SumcheckVerifier`]: round-by-round checker + final evaluation.

pub mod linear_prover;
pub mod general_prover;
pub mod gpu_prover;
pub mod cpu_ext2_prover;
pub mod sparse_bool_prover;
pub mod grand_product;
pub mod verifier;

pub use cpu_ext2_prover::CpuLinearSumcheckProverExt2;
pub use general_prover::GeneralLinearSumcheckProver;
pub use gpu_prover::GpuLinearSumcheckProver;
pub use linear_prover::LinearSumcheckProver;
pub use sparse_bool_prover::SparseBoolSumcheckProverExt2;
pub use verifier::SumcheckVerifier;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use serde::{Deserialize, Serialize};

/// Sumcheck proof transmitted from prover to verifier. Round `i`'s message is
/// the prover's claimed evaluations of the round polynomial at
/// `{0, 1, …, num_poly}` — enough to interpolate the degree-`num_poly`
/// univariate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SumcheckProof {
    pub final_eval: AlmostGoldilocksExt2,
    pub round_messages: Vec<Vec<AlmostGoldilocksExt2>>,
}

/// Generic prover trait. Each prover supplies an instance type and runs the
/// full sumcheck against the given transcript.
pub trait SumcheckProver {
    type Instance;

    fn new(num_var: usize, num_polys: usize, transcript: &mut crate::transcript::Transcript) -> Self;
    fn prove(
        &mut self,
        instances: &Self::Instance,
        transcript: &mut crate::transcript::Transcript,
    ) -> SumcheckProof;
}
