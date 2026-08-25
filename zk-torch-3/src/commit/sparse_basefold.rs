use goldilocks_cuda::GoldilocksExt2;

use crate::commit::basefold::{BasefoldCommit, BasefoldCommitKey, BasefoldCommitmentData, BasefoldOpeningProof, BasefoldVerifierKey};
use crate::commit::MLPolyCommit;
use crate::poly::SparseMLPoly;

/// Sparse Basefold commitment — densifies the sparse polynomial before committing.
/// Legacy stub: `open_sparse` delegates to `BasefoldCommit::open` which returns
/// an incomplete proof. All real opening proofs go through `gpu_open_ext2()` or
/// `cpu_full_open_ext2()`.
#[allow(dead_code)]
pub struct SparseBasefoldCommit;

#[allow(dead_code)]
impl SparseBasefoldCommit {
    pub fn commit_sparse(poly: &SparseMLPoly, key: &BasefoldCommitKey) -> BasefoldCommitmentData {
        let dense = poly.to_dense();
        BasefoldCommit::commit(&dense, key)
    }

    pub fn open_sparse(
        commitment: &BasefoldCommitmentData,
        poly: &SparseMLPoly,
        key: &BasefoldCommitKey,
        point: &[GoldilocksExt2],
    ) -> BasefoldOpeningProof {
        let dense = poly.to_dense();
        BasefoldCommit::open(commitment, &dense, key, point)
    }

    pub fn verify_sparse(
        commitment: &BasefoldCommitmentData,
        proof: &BasefoldOpeningProof,
        key: &BasefoldVerifierKey,
        point: &[GoldilocksExt2],
        eval: GoldilocksExt2,
    ) -> bool {
        BasefoldCommit::verify(commitment, proof, key, point, eval)
    }
}
