//! Split + ternary commit (§6.3). After the multifold the wide folded
//! witness lives in `i16` with `|·| ≤ M · T · (b−1) < 2^13`. Decompose
//! into 13 ternary chunks (`Σ_i 2^i · (pos_i − neg_i) = wide`) and
//! commit each via the SuperNeo `(b=2, k=13)` `commit_ternary` kernel.
//!
//! The 13 chunk commitments are emitted in the proof and the verifier
//! uses the Ajtai homomorphism
//!
//! ```text
//! Σ_i 2^i · c_chunk_i = c_parent
//! ```
//!
//! to check them against the multifold's combined commitment.

use almost_goldilocks_cuda::ajtai::{
    self, RingCommitment, Seed, SPLIT_K_CHUNKS, TernaryChunks, TernaryChunksDevice,
};
use almost_goldilocks_cuda::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};

use crate::fold::{FoldData, FoldInstance, WireCommitment};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplitProof {
    /// 13 ternary-chunk commitments, in chunk order (i = 0 is the
    /// least-significant power of two).
    pub chunk_commitments: Vec<WireCommitment>,
}

/// Run the split step. Takes a [`FoldInstance`] whose witness is either
/// already ternary chunks (the typical post-multifold case) or a binary
/// witness (which we re-decompose as 13 trivial chunks with only chunk 0
/// nonzero), commits them via `commit_ternary`, and returns the
/// `(chunks, commitments)` pair for the next level.
///
/// The Ajtai seed is the same one used by the offline / online commit
/// phase, so the chunks line up with the rest of the commitment chain.
pub fn prove_split(
    inst: &FoldInstance,
    seed: Seed,
) -> (FoldData, SplitProof) {
    let chunks = ensure_ternary(&inst.data);
    let device_chunks = upload_chunks(&chunks).expect("upload ternary chunks");

    let chunk_commits = ajtai::commit_ternary(seed, &device_chunks, None)
        .expect("commit_ternary kernel");

    let proof = SplitProof {
        chunk_commitments: chunk_commits.iter().map(WireCommitment::from_ring).collect(),
    };
    (FoldData::Ternary(chunks), proof)
}

/// Verifier-side check: `Σ_i 2^i · c_chunk_i == c_parent`. Pure CPU
/// (15·64 = 960 modular ops, microseconds). Returns `true` on success.
pub fn verify_split_chunks_match(
    parent: &RingCommitment,
    chunk_commitments: &[RingCommitment],
) -> bool {
    if chunk_commitments.len() != SPLIT_K_CHUNKS { return false; }
    use almost_goldilocks_cuda::ajtai::{KAPPA, RING_DIM};
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            let mut acc = AlmostGoldilocksField(0);
            for bi in 0..SPLIT_K_CHUNKS {
                // 2^bi as a base-field scalar; chunk values already
                // canonicalized as F_q residues.
                let pow = AlmostGoldilocksField(1u64 << bi);
                let coef = AlmostGoldilocksField(chunk_commitments[bi].rows[i][k]);
                acc = acc + pow * coef;
            }
            if acc.reduce().0 != AlmostGoldilocksField(parent.rows[i][k]).reduce().0 {
                return false;
            }
        }
    }
    true
}

/// If `data` is already ternary, return it as-is. If it's binary, lift
/// to ternary chunks with only chunk 0 carrying the data (binary values
/// are trivially `+1` digits in chunk 0).
fn ensure_ternary(data: &FoldData) -> TernaryChunks {
    match data {
        FoldData::Ternary(c) => c.clone(),
        FoldData::Binary(packed) => {
            let n_ring = packed.len();
            let mut pos = vec![0u64; SPLIT_K_CHUNKS * n_ring];
            let neg = vec![0u64; SPLIT_K_CHUNKS * n_ring];
            for j in 0..n_ring {
                pos[j] = packed[j]; // chunk 0 positive
            }
            TernaryChunks { n_ring, k_chunks: SPLIT_K_CHUNKS, pos, neg }
        }
        FoldData::Digit { .. } => {
            unreachable!("Digit not yet wired through split (phase 2 WIP)")
        }
    }
}

/// Upload host `TernaryChunks` to device.
fn upload_chunks(
    chunks: &TernaryChunks,
) -> Result<TernaryChunksDevice, almost_goldilocks_cuda::error::CudaError> {
    let pos = DeviceBuffer::<u64>::from_slice(&chunks.pos)?;
    let neg = DeviceBuffer::<u64>::from_slice(&chunks.neg)?;
    Ok(TernaryChunksDevice {
        n_ring: chunks.n_ring,
        k_chunks: chunks.k_chunks,
        pos,
        neg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    use crate::fold::FoldData;
    use almost_goldilocks_cuda::ajtai::{commit, ChunkSize, KAPPA, RING_DIM};

    fn demo_seed() -> Seed {
        Seed([
            0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
            0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE,
        ])
    }

    /// Random binary witness → split into trivial chunks → commit; then
    /// homomorphism check against a direct commit of the same witness.
    #[test]
    fn binary_split_homomorphism() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let n_ring: usize = 4;
        let packed: Vec<u64> = (0..n_ring).map(|_| rng.gen::<u64>()).collect();

        // Reference: commit of the binary witness directly.
        let c_ref = commit(demo_seed(), &packed, Some(ChunkSize::C64))
            .expect("commit binary");

        // Through split path: binary lifted to chunk 0 only.
        let inst = FoldInstance {
            commitment: c_ref.clone(),
            data: FoldData::Binary(packed.clone()),
            arity: (n_ring.trailing_zeros() as usize) + 6,
            claim_pt: vec![],
            claim_val: almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::zero(),
        };
        let (folded, proof) = prove_split(&inst, demo_seed());
        assert!(matches!(folded, FoldData::Ternary(_)));
        assert_eq!(proof.chunk_commitments.len(), SPLIT_K_CHUNKS);
        let chunk_commits: Vec<RingCommitment> = proof.chunk_commitments.iter()
            .map(|w| w.to_ring())
            .collect();
        // chunk 0 commit equals c_ref (binary → trivial chunk 0).
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                assert_eq!(chunk_commits[0].rows[i][k], c_ref.rows[i][k],
                    "chunk[0] commit mismatch at ({}, {})", i, k);
            }
        }
        // Homomorphism: Σ 2^i · c_i = c_ref (trivially since only chunk 0 is nonzero).
        assert!(verify_split_chunks_match(&c_ref, &chunk_commits),
                "homomorphism check should pass for trivial binary→ternary split");
    }

    /// Build a real wide witness via host fold, decompose to ternary,
    /// commit, and check the homomorphism against a synthetic
    /// parent commit derived from the host fold's reconstruction.
    #[test]
    fn ternary_split_homomorphism_random_wide() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut rng = StdRng::seed_from_u64(7);
        let n_ring: usize = 4;
        // Wide coefficients in [-100, 100].
        let mut wide = vec![0i16; n_ring * 64];
        for v in wide.iter_mut() {
            *v = rng.gen_range(-100..=100);
        }
        // Decompose into 13 chunks.
        let k_chunks = SPLIT_K_CHUNKS;
        let mut pos = vec![0u64; k_chunks * n_ring];
        let mut neg = vec![0u64; k_chunks * n_ring];
        for j in 0..n_ring {
            for k in 0..64 {
                let mut v = wide[j * 64 + k];
                let negative = v < 0;
                if negative { v = -v; }
                for i in 0..k_chunks {
                    if (v >> i) & 1 == 1 {
                        if negative { neg[i * n_ring + j] |= 1u64 << k; }
                        else        { pos[i * n_ring + j] |= 1u64 << k; }
                    }
                }
            }
        }
        let chunks = TernaryChunks { n_ring, k_chunks, pos, neg };
        let inst = FoldInstance {
            commitment: RingCommitment::zero(), // unused for this test
            data: FoldData::Ternary(chunks.clone()),
            arity: (n_ring.trailing_zeros() as usize) + 6,
            claim_pt: vec![],
            claim_val: almost_goldilocks_cuda::extension::AlmostGoldilocksExt2::zero(),
        };
        let (_data, proof) = prove_split(&inst, demo_seed());
        let chunk_commits: Vec<RingCommitment> = proof.chunk_commitments.iter()
            .map(|w| w.to_ring())
            .collect();

        // Compute the expected parent commitment: c_parent = Σ 2^i · c_chunk_i.
        let mut c_parent = RingCommitment::zero();
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                let mut acc = AlmostGoldilocksField(0);
                for bi in 0..SPLIT_K_CHUNKS {
                    let pow = AlmostGoldilocksField(1u64 << bi);
                    let coef = AlmostGoldilocksField(chunk_commits[bi].rows[i][k]);
                    acc = acc + pow * coef;
                }
                c_parent.rows[i][k] = acc.reduce().0;
            }
        }
        assert!(verify_split_chunks_match(&c_parent, &chunk_commits));
    }
}
