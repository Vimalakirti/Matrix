pub mod basefold;
pub mod cpu_basefold;
pub mod gpu_basefold;
pub mod sparse_basefold;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use crate::poly::DenseMLPoly;

/// Commitment trait — enables commitment arithmetic.
pub trait Commitment:
    Clone + Sized + std::fmt::Debug + Default
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
{
    fn zero() -> Self {
        Self::default()
    }

    fn scale(&self, scalar: GoldilocksField) -> Self;
}

/// Multilinear polynomial commitment scheme trait.
pub trait MLPolyCommit {
    type CommitmentKey;
    type VerifierKey;
    type Commitment: Commitment;
    type Proof: Clone + std::fmt::Debug;
    type BatchProof: Clone + std::fmt::Debug;

    fn setup(n: usize, sf_log: usize, offset: i128, size: usize) -> Self::CommitmentKey;
    fn commit(poly: &DenseMLPoly, key: &Self::CommitmentKey) -> Self::Commitment;
    fn open(
        commitment: &Self::Commitment,
        poly: &DenseMLPoly,
        key: &Self::CommitmentKey,
        point: &[GoldilocksExt2],
    ) -> Self::Proof;
    fn verify(
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        key: &Self::VerifierKey,
        point: &[GoldilocksExt2],
        eval: GoldilocksExt2,
    ) -> bool;
    fn batch_open(
        commitments: &[Self::Commitment],
        polys: &[&DenseMLPoly],
        keys: &Self::CommitmentKey,
        point: &[GoldilocksExt2],
    ) -> Self::BatchProof;
    fn batch_verify(
        commitments: &[Self::Commitment],
        proof: &Self::BatchProof,
        key: &Self::VerifierKey,
        point: &[GoldilocksExt2],
        evals: &[GoldilocksExt2],
    ) -> bool;
}
