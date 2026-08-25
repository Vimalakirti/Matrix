use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use goldilocks_cuda::poseidon2::Poseidon2Hash;
use goldilocks_cuda::basefold::{
    BasefoldCommitment, BasefoldProofExt2, BasefoldTable,
    BatchBasefoldProofExt2, EvaluationExt2, HostCommitmentCache,
};

use crate::commit::{Commitment, MLPolyCommit};
use crate::poly::DenseMLPoly;

/// Basefold commitment — a Merkle root of encoded evaluations.
/// This is the lightweight verifier-side data stored in `commitments[]`.
#[derive(Clone, Debug)]
pub struct BasefoldCommitmentData {
    pub root: Poseidon2Hash,
    pub num_vars: usize,
}

impl Default for BasefoldCommitmentData {
    fn default() -> Self {
        Self {
            root: Poseidon2Hash::from_raw([0; 4]),
            num_vars: 0,
        }
    }
}

impl std::ops::Add for BasefoldCommitmentData {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        self
    }
}

impl std::ops::Sub for BasefoldCommitmentData {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        self
    }
}

impl Commitment for BasefoldCommitmentData {
    fn scale(&self, _scalar: GoldilocksField) -> Self {
        self.clone()
    }
}

/// Basefold opening proof — wraps the real GPU proof.
#[derive(Clone, Debug)]
pub struct BasefoldOpeningProof {
    pub eval: GoldilocksExt2,
    pub gpu_proof: BasefoldProofExt2,
}

/// A group of openings batched together by (device_id, num_vars).
/// Contains a single `BatchBasefoldProofExt2` proving all evaluations in the group.
#[derive(Clone, Debug)]
pub struct BatchOpeningGroup {
    pub num_vars: usize,
    pub log_rate: usize,
    pub batch_proof: BatchBasefoldProofExt2,
    /// Mapping: eval index in batch → (edge_id, claim_idx) in the DAG
    pub eval_to_task: Vec<(usize, usize)>,
    /// Commitment roots (ordered as passed to batch_open_ext2)
    pub roots: Vec<Poseidon2Hash>,
    /// Evaluation points (ordered as passed to batch_open_ext2)
    pub points: Vec<Vec<GoldilocksExt2>>,
    /// Evaluation claims (ordered as passed to batch_open_ext2)
    pub evals: Vec<EvaluationExt2>,
}

/// Batch proof for multiple polynomials.
#[derive(Clone, Debug)]
pub struct BasefoldBatchProof {
    pub individual_proofs: Vec<BasefoldOpeningProof>,
}

/// Basefold commitment key (public parameters).
#[derive(Clone, Debug)]
pub struct BasefoldCommitKey {
    pub log_rate: usize,
    pub num_queries: usize,
    pub seed: u64,
}

impl Default for BasefoldCommitKey {
    fn default() -> Self {
        Self {
            log_rate: 3,
            num_queries: 34,  // ~102 bits security (conjectured: 34 × log_rate=3)
            seed: 42,
        }
    }
}

/// Basefold verifier key.
#[derive(Clone, Debug)]
pub struct BasefoldVerifierKey {
    pub log_rate: usize,
    pub num_queries: usize,
    pub seed: u64,
}

impl From<&BasefoldCommitKey> for BasefoldVerifierKey {
    fn from(ck: &BasefoldCommitKey) -> Self {
        Self {
            log_rate: ck.log_rate,
            num_queries: ck.num_queries,
            seed: ck.seed,
        }
    }
}

/// GPU-side commitment store: holds GPU-resident `BasefoldCommitment` objects
/// alongside the shared `BasefoldTable`. Commitments stay alive on their
/// original GPU device so opening proofs can call `open_ext2()` directly
/// without rebuilding.
pub struct GpuCommitmentStore {
    pub table: BasefoldTable,
    pub commitments: Vec<Option<BasefoldCommitment>>,
    /// Which GPU device each commitment lives on (for correct device affinity during open).
    pub device_ids: Vec<Option<i32>>,
    /// Pre-cloned per-device tables for multi-GPU opening proofs.
    pub per_device_tables: Vec<BasefoldTable>,
    /// Host-cached commitment data for re-upload during opening proofs.
    /// Populated by `download_and_free()`.
    pub host_caches: Vec<Option<HostCommitmentCache>>,
}

impl GpuCommitmentStore {
    pub fn new(max_num_vars: usize, log_rate: usize, seed: u64, num_edges: usize) -> Self {
        let num_rounds = max_num_vars;
        let mut table = BasefoldTable::generate(max_num_vars, log_rate, num_rounds, seed);
        table.upload().expect("Failed to upload BasefoldTable to GPU");

        // Pre-clone tables to all GPU devices for multi-GPU opening proofs
        let num_devices = goldilocks_cuda::device_count().unwrap_or(1).max(1) as usize;
        let per_device_tables: Vec<BasefoldTable> = (0..num_devices).map(|d| {
            let _ = goldilocks_cuda::set_device(d as i32);
            let _ = goldilocks_cuda::synchronize();
            goldilocks_cuda::get_last_error();
            goldilocks_cuda::init_device().expect("init_device failed");
            table.clone_to_current_device().expect("table clone failed")
        }).collect();
        let _ = goldilocks_cuda::set_device(0);

        Self {
            table,
            commitments: (0..num_edges).map(|_| None).collect(),
            device_ids: (0..num_edges).map(|_| None).collect(),
            per_device_tables,
            host_caches: (0..num_edges).map(|_| None).collect(),
        }
    }

    /// Free GPU commitments and device tracking.
    pub fn free_commitments(&mut self) {
        for c in self.commitments.iter_mut() {
            *c = None;
        }
        for d in self.device_ids.iter_mut() {
            *d = None;
        }
    }

    /// Free the table's GPU memory. Call after prove when table is no longer needed.
    pub fn free_table(&mut self) {
        self.table.free_gpu();
    }

    /// Free all GPU memory (commitments + table).
    pub fn free_gpu(&mut self) {
        self.free_commitments();
        self.free_table();
    }

    /// Download all GPU commitments to host caches, then free GPU memory.
    /// After this call, `commitments` are all `None` (GPU freed),
    /// and `host_caches` hold the data needed for re-upload during opening proofs.
    pub fn download_and_free(&mut self) {
        let edges: Vec<usize> = (0..self.commitments.len())
            .filter(|&e| self.commitments[e].is_some())
            .collect();
        if edges.is_empty() { return; }

        let t0 = std::time::Instant::now();
        let num_devices = goldilocks_cuda::device_count().unwrap_or(1).max(1) as usize;

        // Download per device sequentially (avoids cross-device contention),
        // but edges within a device are downloaded in parallel.
        for d in 0..num_devices {
            let dev_edges: Vec<usize> = edges.iter()
                .filter(|&&e| self.device_ids[e].unwrap_or(0) == d as i32)
                .copied().collect();
            if dev_edges.is_empty() { continue; }

            let _ = goldilocks_cuda::set_device(d as i32);
            // Download + free one by one to avoid holding too much GPU memory
            for &e in &dev_edges {
                let cache = self.commitments[e].as_ref().unwrap().to_host_cache()
                    .expect("to_host_cache failed");
                self.host_caches[e] = Some(cache);
                self.commitments[e] = None; // Free GPU commitment immediately
            }
        }

        // Clear stale CUDA errors on all devices
        for d in 0..num_devices {
            let _ = goldilocks_cuda::set_device(d as i32);
            let _ = goldilocks_cuda::synchronize();
            loop {
                let err = goldilocks_cuda::get_last_error();
                if err == 0 { break; }
            }
            let _ = goldilocks_cuda::init_device();
        }
        let _ = goldilocks_cuda::set_device(0);

        let total_bytes: usize = self.host_caches.iter().filter_map(|c| c.as_ref()).map(|c| {
            (c.codeword.len() + c.bh_evals.len()) * 8
        }).sum();
        let _ = goldilocks_cuda::set_device(0);
        println!("  download_and_free: {} edges, {:.1} GB, {:.3}s",
            edges.len(), total_bytes as f64 / 1e9, t0.elapsed().as_secs_f64());
    }
}

/// Basefold polynomial commitment scheme.
pub struct BasefoldCommit;

impl MLPolyCommit for BasefoldCommit {
    type CommitmentKey = BasefoldCommitKey;
    type VerifierKey = BasefoldVerifierKey;
    type Commitment = BasefoldCommitmentData;
    type Proof = BasefoldOpeningProof;
    type BatchProof = BasefoldBatchProof;

    fn setup(_n: usize, _sf_log: usize, _offset: i128, _size: usize) -> Self::CommitmentKey {
        BasefoldCommitKey::default()
    }

    fn commit(poly: &DenseMLPoly, key: &Self::CommitmentKey) -> Self::Commitment {
        let gpu_comm = BasefoldCommitment::commit(&poly.evaluations, poly.n, key.log_rate)
            .expect("GPU BasefoldCommitment::commit failed");
        BasefoldCommitmentData {
            root: gpu_comm.root,
            num_vars: poly.n,
        }
    }

    /// Legacy stub: returns an incomplete proof (correct eval, empty Merkle/query data).
    /// This static method path cannot call `open_ext2` without a live GPU commitment.
    /// All real opening proofs go through `gpu_open_ext2()` or `cpu_full_open_ext2()`.
    fn open(
        _commitment: &Self::Commitment,
        poly: &DenseMLPoly,
        _key: &Self::CommitmentKey,
        point: &[GoldilocksExt2],
    ) -> Self::Proof {
        let eval = poly.evaluate_ext2_gpu(point);
        BasefoldOpeningProof {
            eval,
            gpu_proof: BasefoldProofExt2 {
                eval,
                sumcheck_oracles: vec![],
                folded_roots: vec![],
                final_codeword: vec![],
                query_proofs: vec![],
            },
        }
    }

    fn verify(
        _commitment: &Self::Commitment,
        proof: &Self::Proof,
        _key: &Self::VerifierKey,
        _point: &[GoldilocksExt2],
        eval: GoldilocksExt2,
    ) -> bool {
        proof.eval == eval
    }

    fn batch_open(
        commitments: &[Self::Commitment],
        polys: &[&DenseMLPoly],
        key: &Self::CommitmentKey,
        point: &[GoldilocksExt2],
    ) -> Self::BatchProof {
        let individual_proofs: Vec<BasefoldOpeningProof> = commitments
            .iter()
            .zip(polys.iter())
            .map(|(c, p)| Self::open(c, p, key, point))
            .collect();
        BasefoldBatchProof { individual_proofs }
    }

    fn batch_verify(
        _commitments: &[Self::Commitment],
        proof: &Self::BatchProof,
        _key: &Self::VerifierKey,
        _point: &[GoldilocksExt2],
        evals: &[GoldilocksExt2],
    ) -> bool {
        proof
            .individual_proofs
            .iter()
            .zip(evals.iter())
            .all(|(p, &e)| p.eval == e)
    }
}
