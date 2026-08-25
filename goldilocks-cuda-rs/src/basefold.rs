//! Basefold Polynomial Commitment Scheme - GPU-accelerated.
//!
//! Provides both low-level kernel wrappers (`BasefoldBatch`) and a high-level
//! commit→open→verify API (`BasefoldCommitment`, `BasefoldVerifier`).
//!
//! **Design principle**: All intermediate data stays on GPU. Host↔device transfers
//! happen only at input/output boundaries and Fiat-Shamir sync points.

use std::os::raw::c_int;

#[cfg(feature = "monolith")]
use crate::cpu_monolith::{hash_gl_leaf, hash_ext2_leaf, verify_auth_path};
#[cfg(not(feature = "monolith"))]
use crate::cpu_poseidon2::{hash_gl_leaf, hash_ext2_leaf, verify_auth_path};
use crate::error::{CudaError, Result};
use crate::extension::GoldilocksExt2;
use crate::field::GoldilocksField;
use crate::memory::DeviceBuffer;
use crate::merkle::DeviceMerkleTree;
use crate::poseidon2::Poseidon2Hash;
use crate::{ffi, GOLDILOCKS_PRIME};

const BLOCK_SIZE: usize = 256;

/// Field-aware equality for Ext2 elements (handles non-canonical representations like p ≡ 0)
#[inline]
fn ext2_field_eq(a: GoldilocksExt2, b: GoldilocksExt2) -> bool {
    let p = GOLDILOCKS_PRIME;
    (a.c0.0 % p == b.c0.0 % p) && (a.c1.0 % p == b.c1.0 % p)
}

/// Compute eq(r, x) on GPU, returning DeviceBuffer<u64> (not typed GoldilocksField).
fn eq_dp_all_u64(point: &[GoldilocksField], log_n: usize) -> Result<DeviceBuffer<u64>> {
    let raw: Vec<u64> = point.iter().map(|p| p.0).collect();
    let d_r = DeviceBuffer::from_slice(&raw)?;
    let n = 1usize << log_n;
    let mut d_buf_a = DeviceBuffer::<u64>::new(n)?;
    let mut d_buf_b = DeviceBuffer::<u64>::new(n)?;
    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::eq_dp_all_ffi(
            d_r.as_ptr(),
            d_buf_a.as_mut_ptr(),
            d_buf_b.as_mut_ptr(),
            log_n as std::os::raw::c_int,
            &mut result_ptr,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    if result_ptr == d_buf_a.as_ptr() as *mut u64 {
        Ok(d_buf_a)
    } else {
        Ok(d_buf_b)
    }
}

/// Compute ext2 eq(r, x) on GPU, returning DeviceBuffer<u64>.
pub fn ext2_eq_dp_all_u64(point: &[GoldilocksExt2], log_n: usize) -> Result<DeviceBuffer<u64>> {
    let raw: Vec<u64> = point.iter().flat_map(|p| [p.c0.0, p.c1.0]).collect();
    let d_r = DeviceBuffer::from_slice(&raw)?;
    let n = 1usize << log_n;
    let mut d_buf_a = DeviceBuffer::<u64>::new(n * 2)?; // ext2 = 2 u64 each
    let mut d_buf_b = DeviceBuffer::<u64>::new(n * 2)?;
    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::ext2_eq_dp_all_ffi(
            d_r.as_ptr(),
            d_buf_a.as_mut_ptr(),
            d_buf_b.as_mut_ptr(),
            log_n as std::os::raw::c_int,
            &mut result_ptr,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    if result_ptr == d_buf_a.as_ptr() as *mut u64 {
        Ok(d_buf_a)
    } else {
        Ok(d_buf_b)
    }
}

// ============================================================================
// Data Structures
// ============================================================================

/// A folding table entry: stores a folding point and precomputed weight.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FoldingEntry {
    pub point: GoldilocksField,
    pub weight: GoldilocksField,
}

/// One round's degree-2 sum-check polynomial: p(X) = c0 + c1·X + c2·X²
#[derive(Clone, Debug)]
pub struct SumcheckOracle<F: Copy> {
    pub c0: F,
    pub c1: F,
    pub c2: F,
}

/// Per-query authentication data across all folding rounds.
#[derive(Clone, Debug)]
pub struct QueryProof<F: Copy> {
    pub index: usize,
    pub values: Vec<(F, F)>,
    pub merkle_paths: Vec<Vec<Poseidon2Hash>>,
}

/// Base-field opening proof.
#[derive(Clone, Debug)]
pub struct BasefoldProof {
    pub eval: GoldilocksField,
    pub sumcheck_oracles: Vec<SumcheckOracle<GoldilocksField>>,
    pub folded_roots: Vec<Poseidon2Hash>,
    pub final_codeword: Vec<GoldilocksField>,
    pub query_proofs: Vec<QueryProof<GoldilocksField>>,
}

/// Extension-field opening proof.
#[derive(Clone, Debug)]
pub struct BasefoldProofExt2 {
    pub eval: GoldilocksExt2,
    pub sumcheck_oracles: Vec<SumcheckOracle<GoldilocksExt2>>,
    pub folded_roots: Vec<Poseidon2Hash>,
    pub final_codeword: Vec<GoldilocksExt2>,
    pub query_proofs: Vec<QueryProof<GoldilocksExt2>>,
}

/// One claimed evaluation: poly[eval.poly] at points[eval.point] = eval.value.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub poly: usize,
    pub point: usize,
    pub value: GoldilocksField,
}

impl Evaluation {
    pub fn new(poly: usize, point: usize, value: GoldilocksField) -> Self {
        Self { poly, point, value }
    }
}

/// One claimed ext2 evaluation: poly[eval.poly] at points[eval.point] = eval.value.
#[derive(Clone, Debug)]
pub struct EvaluationExt2 {
    pub poly: usize,
    pub point: usize,
    pub value: GoldilocksExt2,
}

impl EvaluationExt2 {
    pub fn new(poly: usize, point: usize, value: GoldilocksExt2) -> Self {
        Self { poly, point, value }
    }
}

/// Per-query authentication data for one individual commitment.
#[derive(Clone, Debug)]
pub struct IndividualQueryProof {
    pub values: (GoldilocksField, GoldilocksField),
    pub merkle_path: Vec<Poseidon2Hash>,
}

/// Batch opening proof for multiple polynomials at (possibly different) points.
#[derive(Clone, Debug)]
pub struct BatchBasefoldProof {
    /// Outer sum-check oracles (num_vars rounds).
    pub outer_sumcheck_oracles: Vec<SumcheckOracle<GoldilocksField>>,
    /// Merkle root of the combined codeword g'.
    pub combined_root: Poseidon2Hash,
    /// Claimed evaluation g'(r), bridging outer and inner sum-checks.
    pub inner_eval: GoldilocksField,
    /// Inner commit-phase sum-check oracles for g' at point r.
    pub inner_sumcheck_oracles: Vec<SumcheckOracle<GoldilocksField>>,
    /// Merkle roots of the combined codeword's folded rounds.
    pub folded_roots: Vec<Poseidon2Hash>,
    /// Final codeword after all inner folding rounds.
    pub final_codeword: Vec<GoldilocksField>,
    /// Query proofs for the combined codeword.
    pub combined_query_proofs: Vec<QueryProof<GoldilocksField>>,
    /// Per-query, per-evaluation individual commitment queries.
    pub individual_query_proofs: Vec<Vec<IndividualQueryProof>>,
}

/// Batch opening proof for multiple polynomials at (possibly different) ext2 points.
#[derive(Clone, Debug)]
pub struct BatchBasefoldProofExt2 {
    /// Outer sum-check oracles (num_vars rounds, all ext2).
    pub outer_sumcheck_oracles: Vec<SumcheckOracle<GoldilocksExt2>>,
    /// Merkle root of the combined ext2 codeword.
    pub combined_root: Poseidon2Hash,
    /// Claimed evaluation bridging outer and inner sum-checks (ext2).
    pub inner_eval: GoldilocksExt2,
    /// Inner commit-phase sum-check oracles (ext2).
    pub inner_sumcheck_oracles: Vec<SumcheckOracle<GoldilocksExt2>>,
    /// Merkle roots of the combined codeword's folded rounds.
    pub folded_roots: Vec<Poseidon2Hash>,
    /// Final codeword after all inner folding rounds.
    pub final_codeword: Vec<GoldilocksExt2>,
    /// Query proofs for the combined ext2 codeword.
    pub combined_query_proofs: Vec<QueryProof<GoldilocksExt2>>,
    /// Per-query, per-evaluation individual commitment queries (still base-field pairs).
    pub individual_query_proofs: Vec<Vec<IndividualQueryProof>>,
}

// ============================================================================
// Transcript trait (Fiat-Shamir)
// ============================================================================

/// Trait for Fiat-Shamir challenge generation during basefold open/verify.
pub trait BasefoldTranscript {
    fn observe_field(&mut self, value: GoldilocksField);
    fn observe_ext2(&mut self, value: GoldilocksExt2);
    fn observe_hash(&mut self, hash: &Poseidon2Hash);
    fn sample_challenge(&mut self) -> GoldilocksField;
    fn sample_challenge_ext2(&mut self) -> GoldilocksExt2;
}

/// Deterministic transcript for testing (xorshift-based, NOT cryptographic).
pub struct TestTranscript {
    state: u64,
}

impl TestTranscript {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

impl BasefoldTranscript for TestTranscript {
    fn observe_field(&mut self, value: GoldilocksField) {
        self.state = self.state.wrapping_add(value.0).max(1);
        self.next_u64();
    }
    fn observe_ext2(&mut self, value: GoldilocksExt2) {
        self.observe_field(value.c0);
        self.observe_field(value.c1);
    }
    fn observe_hash(&mut self, hash: &Poseidon2Hash) {
        for e in &hash.elements {
            self.observe_field(*e);
        }
    }
    fn sample_challenge(&mut self) -> GoldilocksField {
        GoldilocksField(self.next_u64() % GOLDILOCKS_PRIME)
    }
    fn sample_challenge_ext2(&mut self) -> GoldilocksExt2 {
        let c0 = self.sample_challenge();
        let c1 = self.sample_challenge();
        GoldilocksExt2::new(c0, c1)
    }
}

// ============================================================================
// Folding Table
// ============================================================================

/// Folding table for basefold codeword folding.
pub struct BasefoldTable {
    pub entries: Vec<FoldingEntry>,
    pub level_offsets: Vec<usize>,
    pub level_sizes: Vec<usize>,
    pub num_rounds: usize,
    d_entries: Option<DeviceBuffer<u64>>,
}

impl BasefoldTable {
    /// Generate a random folding table using a simple PRNG.
    pub fn generate(num_vars: usize, log_rate: usize, num_rounds: usize, seed: u64) -> Self {
        let mut rng_state = seed;
        let mut rand_u64 = || -> u64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state % GOLDILOCKS_PRIME
        };

        let mut entries = Vec::new();
        let mut level_offsets = Vec::new();
        let mut level_sizes = Vec::new();

        for i in 0..num_rounds {
            let level_size = 1usize << (num_vars + log_rate - i - 1);
            level_offsets.push(entries.len());
            level_sizes.push(level_size);

            for _ in 0..level_size {
                let x0 = rand_u64();
                let mut x1 = rand_u64();
                while x1 == x0 {
                    x1 = rand_u64();
                }
                let diff = gl_sub_host(x1, x0);
                let weight = gl_inv_host(diff);
                entries.push(FoldingEntry {
                    point: GoldilocksField(x0),
                    weight: GoldilocksField(weight),
                });
            }
        }

        BasefoldTable {
            entries,
            level_offsets,
            level_sizes,
            num_rounds,
            d_entries: None,
        }
    }

    /// Upload the table to GPU device memory.
    pub fn upload(&mut self) -> Result<()> {
        let flat: Vec<u64> = self
            .entries
            .iter()
            .flat_map(|e| [e.point.0, e.weight.0])
            .collect();
        self.d_entries = Some(DeviceBuffer::from_slice(&flat)?);
        Ok(())
    }

    /// Clone CPU data and upload to the current GPU device.
    /// Avoids re-running the expensive `generate()` (modular inverses).
    pub fn clone_to_current_device(&self) -> Result<Self> {
        let mut t = BasefoldTable {
            entries: self.entries.clone(),
            level_offsets: self.level_offsets.clone(),
            level_sizes: self.level_sizes.clone(),
            num_rounds: self.num_rounds,
            d_entries: None,
        };
        t.upload()?;
        Ok(t)
    }

    /// Free GPU memory held by this table.
    pub fn free_gpu(&mut self) {
        self.d_entries = None;
    }

    /// Get device pointer for a specific level of the table.
    pub fn device_level_ptr(&self, level: usize) -> Result<*const u64> {
        let buf = self
            .d_entries
            .as_ref()
            .ok_or(CudaError::AllocationFailed)?;
        let offset = self.level_offsets[level] * 2; // 2 u64 per entry
        Ok(unsafe { buf.as_ptr().add(offset) })
    }
}

// ============================================================================
// BasefoldCommitment
// ============================================================================

/// Commitment to a multilinear polynomial. Heavy data lives on GPU.
pub struct BasefoldCommitment {
    pub root: Poseidon2Hash,
    d_codeword: DeviceBuffer<u64>,
    d_bh_evals: DeviceBuffer<u64>,
    merkle_tree: DeviceMerkleTree,
    pub num_vars: usize,
    pub log_rate: usize,
}

impl BasefoldCommitment {
    /// Commit from evaluations already on GPU.
    ///
    /// Transfers: 0 H→D, 32 bytes D→H (merkle root).
    pub fn commit_device(
        d_evals: &DeviceBuffer<u64>,
        num_vars: usize,
        log_rate: usize,
    ) -> Result<Self> {
        let n = 1usize << num_vars;
        let cw_len = 1usize << (num_vars + log_rate);

        let mut d_coeffs = DeviceBuffer::<u64>::new(n)?;
        let mut d_bh_evals = DeviceBuffer::<u64>::new(n)?;
        let mut d_codeword = DeviceBuffer::<u64>::new(cw_len)?;

        // BHC interpolation (GPU only)
        BasefoldBatch::bhc_interpolate(d_evals, &mut d_coeffs, &mut d_bh_evals, num_vars)?;

        // Encode (GPU only)
        BasefoldBatch::encode(&d_coeffs, &mut d_codeword, num_vars, log_rate)?;
        // d_coeffs no longer needed
        drop(d_coeffs);

        // Build merkle tree (GPU only)
        let merkle_tree = DeviceMerkleTree::build_from_gl_codeword(&d_codeword, cw_len)?;
        let root = merkle_tree.root()?; // 32 bytes D→H

        Ok(Self {
            root,
            d_codeword,
            d_bh_evals,
            merkle_tree,
            num_vars,
            log_rate,
        })
    }

    /// Commit from host evaluations.
    ///
    /// Transfers: 2^num_vars × 8 bytes H→D (one-time), 32 bytes D→H (root).
    pub fn commit(
        evals: &[GoldilocksField],
        num_vars: usize,
        log_rate: usize,
    ) -> Result<Self> {
        let raw: Vec<u64> = evals.iter().map(|e| e.0).collect();
        let d_evals = DeviceBuffer::from_slice(&raw)?;
        Self::commit_device(&d_evals, num_vars, log_rate)
    }

    /// Download codeword + bh_evals to host. Call after commit, before free.
    pub fn to_host_cache(&self) -> Result<HostCommitmentCache> {
        let codeword = self.d_codeword.to_vec()?;
        let bh_evals = self.d_bh_evals.to_vec()?;
        Ok(HostCommitmentCache {
            root: self.root,
            codeword,
            bh_evals,
            num_vars: self.num_vars,
            log_rate: self.log_rate,
        })
    }

    /// Generate base-field opening proof.
    pub fn open(
        &self,
        point: &[GoldilocksField],
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
        num_queries: usize,
    ) -> Result<BasefoldProof> {
        let num_vars = self.num_vars;
        let log_rate = self.log_rate;
        let num_rounds = num_vars;
        let n = 1usize << num_vars;

        // Observe commitment
        transcript.observe_hash(&self.root);
        for p in point {
            transcript.observe_field(*p);
        }

        // 1. eq polynomial on GPU (returns DeviceBuffer<u64>)
        let mut d_eq = eq_dp_all_u64(point, num_vars)?;

        // 2. Bit-reverse bh_evals copy to Type2 (GPU→GPU)
        let mut d_bh_work = self.d_bh_evals.clone_on_device()?;
        BasefoldBatch::bit_reverse_gl(&mut d_bh_work, num_vars)?;

        // 3. Dot product for eval (GPU, then ~2 KB D→H for partial sums)
        let dp_blocks = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
        let mut d_dp_partial = DeviceBuffer::<u64>::new(dp_blocks)?;
        BasefoldBatch::dot_product_gl(&d_bh_work, &d_eq, &mut d_dp_partial, n, dp_blocks)?;
        let eval = reduce_dot_product_gl(&d_dp_partial, dp_blocks)?;
        drop(d_dp_partial);

        // 4. First sum-check round
        let mut pair_count = n / 2;
        BasefoldBatch::sumcheck_interp_gl(&mut d_eq, pair_count)?;
        BasefoldBatch::sumcheck_interp_gl(&mut d_bh_work, pair_count)?;
        let oracle0 = sumcheck_product_and_reduce_gl(&d_eq, &d_bh_work, pair_count)?;

        let mut sumcheck_oracles = vec![oracle0.clone()];
        transcript.observe_field(oracle0.c0);
        transcript.observe_field(oracle0.c1);
        transcript.observe_field(oracle0.c2);

        // 5. Folding loop
        let mut folded_roots = Vec::new();
        let mut folded_codewords: Vec<DeviceBuffer<u64>> = Vec::new();
        let mut folded_trees: Vec<DeviceMerkleTree> = Vec::new();
        let mut d_codeword_cur = self.d_codeword.clone_on_device()?;
        let mut cw_pair_count = (1usize << (num_vars + log_rate)) / 2;

        for round in 0..num_rounds - 1 {
            let challenge = transcript.sample_challenge();

            // Sum-check eval at challenge (GPU only)
            let mut d_eq_half = DeviceBuffer::<u64>::new(pair_count)?;
            let mut d_bh_half = DeviceBuffer::<u64>::new(pair_count)?;
            BasefoldBatch::sumcheck_eval_gl(&d_eq, challenge, &mut d_eq_half, pair_count)?;
            BasefoldBatch::sumcheck_eval_gl(&d_bh_work, challenge, &mut d_bh_half, pair_count)?;

            pair_count /= 2;
            d_eq = d_eq_half;
            d_bh_work = d_bh_half;

            // Interp + product for next oracle
            BasefoldBatch::sumcheck_interp_gl(&mut d_eq, pair_count)?;
            BasefoldBatch::sumcheck_interp_gl(&mut d_bh_work, pair_count)?;
            let oracle =
                sumcheck_product_and_reduce_gl(&d_eq, &d_bh_work, pair_count)?;

            sumcheck_oracles.push(oracle.clone());
            transcript.observe_field(oracle.c0);
            transcript.observe_field(oracle.c1);
            transcript.observe_field(oracle.c2);

            // Fold codeword (GPU only)
            let table_ptr = table.device_level_ptr(round)?;
            let mut d_folded = DeviceBuffer::<u64>::new(cw_pair_count)?;
            BasefoldBatch::fold_gl(
                &d_codeword_cur,
                table_ptr,
                challenge,
                &mut d_folded,
                cw_pair_count,
            )?;

            // Build merkle tree on folded codeword (GPU only)
            let tree = DeviceMerkleTree::build_from_gl_codeword(&d_folded, cw_pair_count)?;
            let root = tree.root()?;
            transcript.observe_hash(&root);
            folded_roots.push(root);

            folded_codewords.push(d_codeword_cur);
            folded_trees.push(tree);
            d_codeword_cur = d_folded;
            cw_pair_count /= 2;
        }

        // Last challenge for final fold
        let last_challenge = transcript.sample_challenge();
        let table_ptr = table.device_level_ptr(num_rounds - 1)?;
        let mut d_final = DeviceBuffer::<u64>::new(cw_pair_count)?;
        BasefoldBatch::fold_gl(
            &d_codeword_cur,
            table_ptr,
            last_challenge,
            &mut d_final,
            cw_pair_count,
        )?;

        // Download final codeword (small, D→H)
        let final_raw = d_final.to_vec()?;
        let final_codeword: Vec<GoldilocksField> = final_raw.into_iter().map(GoldilocksField).collect();

        // Query phase
        let query_proofs = extract_gl_queries(
            &self.d_codeword,
            &self.merkle_tree,
            &folded_codewords,
            &folded_trees,
            &d_codeword_cur,
            num_queries,
            transcript,
            1usize << (num_vars + log_rate),
        )?;

        Ok(BasefoldProof {
            eval,
            sumcheck_oracles,
            folded_roots,
            final_codeword,
            query_proofs,
        })
    }

    /// Generate extension-field opening proof.
    pub fn open_ext2(
        &self,
        point: &[GoldilocksExt2],
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
        num_queries: usize,
    ) -> Result<BasefoldProofExt2> {
        let num_vars = self.num_vars;
        let log_rate = self.log_rate;
        let num_rounds = num_vars;
        let n = 1usize << num_vars;
        let t_open = std::time::Instant::now();
        let debug_timing = std::env::var("ZK_OPEN_TIMING").is_ok();

        // Observe commitment
        transcript.observe_hash(&self.root);
        for p in point {
            transcript.observe_ext2(*p);
        }

        // 1. ext2 eq polynomial on GPU (returns DeviceBuffer<u64>)
        let d_eq_init = ext2_eq_dp_all_u64(point, num_vars)?;

        // 2. Bit-reverse bh_evals copy to Type2 (GPU→GPU)
        let mut d_bh_fp = self.d_bh_evals.clone_on_device()?;
        BasefoldBatch::bit_reverse_gl(&mut d_bh_fp, num_vars)?;

        // 3. Mixed dot product for eval
        let dp_blocks = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
        let mut d_dp_partial = DeviceBuffer::<u64>::new(dp_blocks * 2)?;
        BasefoldBatch::dot_product_mixed(&d_bh_fp, &d_eq_init, &mut d_dp_partial, n, dp_blocks)?;
        let eval = reduce_dot_product_ext2(&d_dp_partial, dp_blocks)?;
        drop(d_dp_partial);

        if debug_timing {
            crate::synchronize().ok();
            eprintln!("  [open n={}] setup+eq+dot: {:.3}ms", num_vars, t_open.elapsed().as_secs_f64() * 1e3);
        }

        // === Pre-allocate reusable buffers (Opts 1-2) ===
        // Sumcheck partial reduction buffers (Opt 1): 3 small buffers reused every round
        let max_blocks = 256usize;
        let mut pc0 = DeviceBuffer::<u64>::new(max_blocks * 2)?;
        let mut pc1 = DeviceBuffer::<u64>::new(max_blocks * 2)?;
        let mut pc2 = DeviceBuffer::<u64>::new(max_blocks * 2)?;

        // Double-buffer for eq and bh (Opt 2): two buffers each, swap every round
        // Initial eq size: n ext2 = n*2 u64. After round 0: n/2 ext2.
        // We allocate at initial (max) size.
        let mut d_eq_a = d_eq_init; // reuse initial allocation
        let mut d_eq_b = DeviceBuffer::<u64>::new(n * 2)?;
        let mut d_bh_a = DeviceBuffer::<u64>::new(n * 2)?; // for ext2 bh after transition
        let mut d_bh_b = DeviceBuffer::<u64>::new(n * 2)?;
        // Track which buffer is "current" (true = A, false = B)
        let mut eq_is_a = true;
        let mut bh_is_a = true;

        // 4. First sum-check round (mixed: bh in F_p, eq in F_{p^2})
        let mut pair_count = n / 2;
        BasefoldBatch::sumcheck_interp_gl(&mut d_bh_fp, pair_count)?;
        BasefoldBatch::sumcheck_interp_ext2(&mut d_eq_a, pair_count)?;
        let oracle0 = sumcheck_product_and_reduce_mixed(&d_eq_a, &d_bh_fp, pair_count)?;

        let mut sumcheck_oracles = vec![oracle0.clone()];
        transcript.observe_ext2(oracle0.c0);
        transcript.observe_ext2(oracle0.c1);
        transcript.observe_ext2(oracle0.c2);

        let t_sc = std::time::Instant::now();
        // 5. Round 0: mixed → ext2 transition
        let challenge0 = transcript.sample_challenge_ext2();

        // bh: F_p → F_{p^2} transition, write into d_bh_a
        BasefoldBatch::sumcheck_eval_mixed(&d_bh_fp, challenge0, &mut d_bh_a, pair_count)?;
        drop(d_bh_fp);
        bh_is_a = true;

        // eq: pure ext2 eval, write into d_eq_b
        BasefoldBatch::sumcheck_eval_ext2(&d_eq_a, challenge0, &mut d_eq_b, pair_count)?;
        pair_count /= 2;
        eq_is_a = false; // current eq is in d_eq_b

        // Interp + product for oracle[1], using pre-allocated pc buffers
        {
            let (eq_cur, bh_cur) = if eq_is_a { (&mut d_eq_a, &mut d_bh_a) } else { (&mut d_eq_b, &mut d_bh_a) };
            BasefoldBatch::sumcheck_interp_ext2(eq_cur, pair_count)?;
            BasefoldBatch::sumcheck_interp_ext2(bh_cur, pair_count)?;
            let oracle1 = sumcheck_product_and_reduce_ext2_reuse(eq_cur, bh_cur, pair_count, &mut pc0, &mut pc1, &mut pc2)?;
            sumcheck_oracles.push(oracle1.clone());
            transcript.observe_ext2(oracle1.c0);
            transcript.observe_ext2(oracle1.c1);
            transcript.observe_ext2(oracle1.c2);
        }

        // Fold codeword: mixed fold (F_p → F_{p^2})
        let mut cw_pair_count = (1usize << (num_vars + log_rate)) / 2;
        let table_ptr = table.device_level_ptr(0)?;

        // Pre-allocate ALL fold output buffers upfront (Opt 4)
        // Round 0 fold: cw_pair_count ext2 pairs. Rounds 1..num_rounds-2: halving each time.
        let mut fold_buffers: Vec<DeviceBuffer<u64>> = Vec::with_capacity(num_rounds);
        {
            let mut cw_sz = cw_pair_count;
            fold_buffers.push(DeviceBuffer::<u64>::new(cw_sz * 2)?); // round 0
            for _ in 1..num_rounds - 1 {
                cw_sz /= 2;
                fold_buffers.push(DeviceBuffer::<u64>::new(cw_sz * 2)?);
            }
            // Last fold (final codeword)
            cw_sz /= 2;
            fold_buffers.push(DeviceBuffer::<u64>::new(cw_sz * 2)?);
        }

        // Pre-allocate ALL Merkle tree buffers upfront (Opt 5)
        // Tree i has num_leaves = cw_pair_count / 2^i, needs (2*num_leaves - 1) * DIGEST_SIZE u64
        let mut tree_buffers: Vec<DeviceBuffer<u64>> = Vec::with_capacity(num_rounds - 1);
        {
            let mut tree_leaves = cw_pair_count / 2;
            for _ in 0..num_rounds - 1 {
                let tree_nodes = 2 * tree_leaves - 1;
                tree_buffers.push(DeviceBuffer::<u64>::new(tree_nodes * crate::poseidon2::POSEIDON2_DIGEST_SIZE)?);
                tree_leaves /= 2;
            }
        }

        BasefoldBatch::fold_mixed(
            &self.d_codeword,
            table_ptr,
            challenge0,
            &mut fold_buffers[0],
            cw_pair_count,
        )?;

        // Merkle tree on ext2 codeword — use pre-allocated buffer
        let tree_buf0 = tree_buffers.remove(0);
        let tree0 = DeviceMerkleTree::build_from_ext2_codeword_into(&fold_buffers[0], cw_pair_count, tree_buf0)?;
        let root0 = tree0.root()?;
        transcript.observe_hash(&root0);
        let mut folded_roots = vec![root0];
        let mut folded_trees: Vec<DeviceMerkleTree> = Vec::new();
        folded_trees.push(tree0);
        cw_pair_count /= 2;

        if debug_timing {
            crate::synchronize().ok();
            eprintln!("  [open n={}] round0 (sc+fold+merkle): {:.3}ms", num_vars, t_sc.elapsed().as_secs_f64() * 1e3);
        }
        let t_loop = std::time::Instant::now();
        let mut t_fold_total = 0.0f64;
        let mut t_merkle_total = 0.0f64;
        let mut t_sumcheck_total = 0.0f64;

        // 6. Remaining rounds: fused eval+interp+product kernel
        for round in 1..num_rounds - 1 {
            let challenge = transcript.sample_challenge_ext2();

            let t_sc_r = std::time::Instant::now();

            // Fused kernel: reads 4*product_pairs from current buffers,
            // writes 2*product_pairs to other buffers + block partials.
            // pair_count is the EVAL pair count (= current element count / 2).
            // product_pairs = pair_count / 2 = current element count / 4.
            let product_pairs = pair_count / 2;

            let oracle = match (eq_is_a, bh_is_a) {
                (true, true) => fused_sumcheck_round_ext2_reuse(
                    &d_eq_a, &d_bh_a, challenge, &mut d_eq_b, &mut d_bh_b,
                    product_pairs, &mut pc0, &mut pc1, &mut pc2,
                )?,
                (true, false) => fused_sumcheck_round_ext2_reuse(
                    &d_eq_a, &d_bh_b, challenge, &mut d_eq_b, &mut d_bh_a,
                    product_pairs, &mut pc0, &mut pc1, &mut pc2,
                )?,
                (false, true) => fused_sumcheck_round_ext2_reuse(
                    &d_eq_b, &d_bh_a, challenge, &mut d_eq_a, &mut d_bh_b,
                    product_pairs, &mut pc0, &mut pc1, &mut pc2,
                )?,
                (false, false) => fused_sumcheck_round_ext2_reuse(
                    &d_eq_b, &d_bh_b, challenge, &mut d_eq_a, &mut d_bh_a,
                    product_pairs, &mut pc0, &mut pc1, &mut pc2,
                )?,
            };
            eq_is_a = !eq_is_a;
            bh_is_a = !bh_is_a;
            pair_count /= 2;

            sumcheck_oracles.push(oracle.clone());
            transcript.observe_ext2(oracle.c0);
            transcript.observe_ext2(oracle.c1);
            transcript.observe_ext2(oracle.c2);
            if debug_timing { crate::synchronize().ok(); t_sumcheck_total += t_sc_r.elapsed().as_secs_f64(); }

            // Fold codeword using pre-allocated buffer (Opt 4)
            let t_fold_r = std::time::Instant::now();
            let table_ptr = table.device_level_ptr(round)?;
            // Read from fold_buffers[round-1], write to fold_buffers[round]
            // Since fold_buffers is a Vec, we need split borrows
            let (prev_slice, cur_slice) = fold_buffers.split_at_mut(round);
            let d_cw_input = &prev_slice[round - 1];
            let d_cw_output = &mut cur_slice[0];
            BasefoldBatch::fold_ext2(
                d_cw_input,
                table_ptr,
                challenge,
                d_cw_output,
                cw_pair_count,
            )?;
            if debug_timing { crate::synchronize().ok(); t_fold_total += t_fold_r.elapsed().as_secs_f64(); }

            // Merkle tree on folded codeword — use pre-allocated buffer
            let t_merkle_r = std::time::Instant::now();
            let tree_buf = tree_buffers.remove(0);
            let tree = DeviceMerkleTree::build_from_ext2_codeword_into(&fold_buffers[round], cw_pair_count, tree_buf)?;
            let root = tree.root()?;
            transcript.observe_hash(&root);
            folded_roots.push(root);
            folded_trees.push(tree);
            if debug_timing { crate::synchronize().ok(); t_merkle_total += t_merkle_r.elapsed().as_secs_f64(); }

            cw_pair_count /= 2;
        }

        // Last fold using pre-allocated buffer
        let last_challenge = transcript.sample_challenge_ext2();
        let table_ptr = table.device_level_ptr(num_rounds - 1)?;
        {
            let last_idx = num_rounds - 1;
            let (prev_slice, cur_slice) = fold_buffers.split_at_mut(last_idx);
            let d_cw_input = &prev_slice[last_idx - 1];
            let d_cw_output = &mut cur_slice[0];
            BasefoldBatch::fold_ext2(
                d_cw_input,
                table_ptr,
                last_challenge,
                d_cw_output,
                cw_pair_count,
            )?;
        }

        // Download final codeword
        let final_raw = fold_buffers[num_rounds - 1].to_vec()?;
        let final_codeword: Vec<GoldilocksExt2> = final_raw
            .chunks_exact(2)
            .map(|c| GoldilocksExt2::new(GoldilocksField(c[0]), GoldilocksField(c[1])))
            .collect();

        if debug_timing {
            crate::synchronize().ok();
            eprintln!("  [open n={}] loop ({} rounds): sumcheck {:.1}ms, fold {:.1}ms, merkle {:.1}ms, total {:.1}ms",
                num_vars, num_rounds - 2,
                t_sumcheck_total * 1e3, t_fold_total * 1e3, t_merkle_total * 1e3,
                t_loop.elapsed().as_secs_f64() * 1e3);
        }

        // Query proofs — use fold_buffers directly (Opt 3: no clone_on_device needed)
        // fold_buffers[0] = round 0 folded codeword, fold_buffers[i] = round i folded codeword
        // folded_trees[i] = Merkle tree for fold_buffers[i]
        // d_codeword_cur for extract = fold_buffers[num_rounds-2] (last non-final folded cw)
        let t_query = std::time::Instant::now();
        let query_proofs = extract_ext2_queries(
            &self.d_codeword,
            &self.merkle_tree,
            &fold_buffers[0..num_rounds - 1],
            &folded_trees,
            &fold_buffers[num_rounds - 2],
            num_queries,
            transcript,
            1usize << (num_vars + log_rate),
        )?;

        if debug_timing {
            crate::synchronize().ok();
            eprintln!("  [open n={}] queries ({} queries): {:.1}ms, TOTAL: {:.1}ms",
                num_vars, num_queries,
                t_query.elapsed().as_secs_f64() * 1e3,
                t_open.elapsed().as_secs_f64() * 1e3);
        }

        Ok(BasefoldProofExt2 {
            eval,
            sumcheck_oracles,
            folded_roots,
            final_codeword,
            query_proofs,
        })
    }
}

// ============================================================================
// HostCommitmentCache
// ============================================================================

/// Host-cached commitment data for fast re-upload during opening proofs.
/// Stores the pre-computed codeword and BHC evaluations, avoiding
/// expensive BHC interpolation + RS encoding during opening.
pub struct HostCommitmentCache {
    pub root: Poseidon2Hash,
    pub codeword: Vec<u64>,       // len = 2^(num_vars + log_rate)
    pub bh_evals: Vec<u64>,       // len = 2^num_vars
    pub num_vars: usize,
    pub log_rate: usize,
}

impl HostCommitmentCache {
    /// Upload to GPU: codeword + bh_evals (H→D), then rebuild Merkle tree (GPU).
    /// Skips BHC interpolation and RS encoding.
    pub fn to_device(&self) -> Result<BasefoldCommitment> {
        let d_codeword = DeviceBuffer::from_slice(&self.codeword)?;
        let d_bh_evals = DeviceBuffer::from_slice(&self.bh_evals)?;
        let cw_len = 1usize << (self.num_vars + self.log_rate);
        let merkle_tree = DeviceMerkleTree::build_from_gl_codeword(&d_codeword, cw_len)?;
        let root = merkle_tree.root()?;
        assert_eq!(root, self.root, "Rebuilt Merkle root differs from cached root");
        Ok(BasefoldCommitment {
            root: self.root,
            d_codeword,
            d_bh_evals,
            merkle_tree,
            num_vars: self.num_vars,
            log_rate: self.log_rate,
        })
    }
}

// ============================================================================
// Batch open
// ============================================================================

/// Batch opening: open multiple polynomials at (possibly different) base-field points.
///
/// All polynomials must share the same `num_vars` and `log_rate`.
///
/// Algorithm:
/// 1. Random batching weights from `eq(x, t)` where t ← transcript.
/// 2. Outer sum-check: proves `sum_i weight_i * poly_i(point_i) = claimed_sum`.
/// 3. Combined polynomial `g' = sum_i scalar_i * poly_i` and its codeword.
/// 4. Inner commit phase: sum-check + FRI folding on `g'` at point `r`.
/// 5. Query proofs for individual + combined codewords.
pub fn batch_open(
    comms: &[&BasefoldCommitment],
    points: &[&[GoldilocksField]],
    evals: &[Evaluation],
    table: &BasefoldTable,
    transcript: &mut impl BasefoldTranscript,
    num_queries: usize,
) -> Result<BatchBasefoldProof> {
    assert!(!comms.is_empty());
    assert!(!evals.is_empty());
    let num_vars = comms[0].num_vars;
    let log_rate = comms[0].log_rate;
    for c in comms {
        assert_eq!(c.num_vars, num_vars);
        assert_eq!(c.log_rate, log_rate);
    }
    let n = 1usize << num_vars;
    let num_points = points.len();

    // ── 1. Transcript: observe commitments, points, values ──
    for comm in comms {
        transcript.observe_hash(&comm.root);
    }
    for point in points {
        for p in *point {
            transcript.observe_field(*p);
        }
    }
    for e in evals {
        transcript.observe_field(e.value);
    }

    // ── 2. Random batching weights ──
    let num_evals = evals.len();
    let ell = if num_evals <= 1 {
        1
    } else {
        num_evals.next_power_of_two().trailing_zeros() as usize
    };
    let t: Vec<GoldilocksField> = (0..ell)
        .map(|_| transcript.sample_challenge())
        .collect();
    let eq_xt = eq_poly_host(&t);

    // ── 3. Merge bh_evals per point on GPU ──
    let mut bh_cache: Vec<Option<DeviceBuffer<u64>>> = (0..comms.len()).map(|_| None).collect();
    let mut d_g_bufs: Vec<DeviceBuffer<u64>> = (0..num_points)
        .map(|_| gpu_zero_buffer(n))
        .collect::<Result<Vec<_>>>()?;
    let mut d_tmp = DeviceBuffer::<u64>::new(n)?;
    for (i, eval) in evals.iter().enumerate() {
        if bh_cache[eval.poly].is_none() {
            let mut br = comms[eval.poly].d_bh_evals.clone_on_device()?;
            BasefoldBatch::bit_reverse_gl(&mut br, num_vars)?;
            bh_cache[eval.poly] = Some(br);
        }
        gpu_accumulate_scaled(
            &mut d_g_bufs[eval.point],
            bh_cache[eval.poly].as_ref().unwrap(),
            &mut d_tmp,
            eq_xt[i],
            n,
        )?;
    }
    drop(d_tmp);
    drop(bh_cache);

    // ── 4. Eq polynomials for each evaluation point on GPU ──
    let mut d_eq_bufs: Vec<DeviceBuffer<u64>> = points
        .iter()
        .map(|pt| eq_dp_all_u64(pt, num_vars))
        .collect::<Result<Vec<_>>>()?;

    // ── 5. Outer sum-check (num_vars rounds) ──
    let mut pair_count = n / 2;
    for p in 0..num_points {
        BasefoldBatch::sumcheck_interp_gl(&mut d_eq_bufs[p], pair_count)?;
        BasefoldBatch::sumcheck_interp_gl(&mut d_g_bufs[p], pair_count)?;
    }
    let oracle0 = multi_point_sumcheck_product_gl(&d_eq_bufs, &d_g_bufs, pair_count)?;
    let mut outer_oracles = vec![oracle0.clone()];
    transcript.observe_field(oracle0.c0);
    transcript.observe_field(oracle0.c1);
    transcript.observe_field(oracle0.c2);

    let mut outer_challenges = Vec::with_capacity(num_vars);
    for _round in 0..num_vars - 1 {
        let ch = transcript.sample_challenge();
        outer_challenges.push(ch);
        for p in 0..num_points {
            let mut d_eq_h = DeviceBuffer::<u64>::new(pair_count)?;
            let mut d_g_h = DeviceBuffer::<u64>::new(pair_count)?;
            BasefoldBatch::sumcheck_eval_gl(&d_eq_bufs[p], ch, &mut d_eq_h, pair_count)?;
            BasefoldBatch::sumcheck_eval_gl(&d_g_bufs[p], ch, &mut d_g_h, pair_count)?;
            d_eq_bufs[p] = d_eq_h;
            d_g_bufs[p] = d_g_h;
        }
        pair_count /= 2;
        for p in 0..num_points {
            BasefoldBatch::sumcheck_interp_gl(&mut d_eq_bufs[p], pair_count)?;
            BasefoldBatch::sumcheck_interp_gl(&mut d_g_bufs[p], pair_count)?;
        }
        let oracle = multi_point_sumcheck_product_gl(&d_eq_bufs, &d_g_bufs, pair_count)?;
        outer_oracles.push(oracle.clone());
        transcript.observe_field(oracle.c0);
        transcript.observe_field(oracle.c1);
        transcript.observe_field(oracle.c2);
    }
    let last_ch = transcript.sample_challenge();
    outer_challenges.push(last_ch);
    let r = outer_challenges;
    drop(d_eq_bufs);
    drop(d_g_bufs);

    // ── 6. Combined codeword + bh_evals on GPU ──
    let eq_r_pts: Vec<GoldilocksField> = points
        .iter()
        .map(|pt| eq_eval_host(&r, pt))
        .collect();
    let cw_len = 1usize << (num_vars + log_rate);
    let mut d_combined_cw = gpu_zero_buffer(cw_len)?;
    let mut d_combined_bh = gpu_zero_buffer(n)?;
    let mut d_tmp_cw = DeviceBuffer::<u64>::new(cw_len)?;
    let mut d_tmp_bh = DeviceBuffer::<u64>::new(n)?;
    for (i, eval) in evals.iter().enumerate() {
        let scalar = GoldilocksField(gl_mul_host(eq_r_pts[eval.point].0, eq_xt[i].0));
        gpu_accumulate_scaled(
            &mut d_combined_cw,
            &comms[eval.poly].d_codeword,
            &mut d_tmp_cw,
            scalar,
            cw_len,
        )?;
        gpu_accumulate_scaled(
            &mut d_combined_bh,
            &comms[eval.poly].d_bh_evals,
            &mut d_tmp_bh,
            scalar,
            n,
        )?;
    }
    drop(d_tmp_cw);
    drop(d_tmp_bh);

    // ── 7. Combined Merkle tree ──
    let combined_tree = DeviceMerkleTree::build_from_gl_codeword(&d_combined_cw, cw_len)?;
    let combined_root = combined_tree.root()?;
    transcript.observe_hash(&combined_root);

    // ── 8. Inner commit phase (sum-check + folding on g' at point r) ──
    let mut d_inner_eq = eq_dp_all_u64(&r, num_vars)?;
    BasefoldBatch::bit_reverse_gl(&mut d_combined_bh, num_vars)?;

    let dp_blocks = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    let mut d_dp = DeviceBuffer::<u64>::new(dp_blocks)?;
    BasefoldBatch::dot_product_gl(&d_combined_bh, &d_inner_eq, &mut d_dp, n, dp_blocks)?;
    let inner_eval = reduce_dot_product_gl(&d_dp, dp_blocks)?;
    drop(d_dp);

    let mut inner_pc = n / 2;
    BasefoldBatch::sumcheck_interp_gl(&mut d_inner_eq, inner_pc)?;
    BasefoldBatch::sumcheck_interp_gl(&mut d_combined_bh, inner_pc)?;
    let inner_o0 = sumcheck_product_and_reduce_gl(&d_inner_eq, &d_combined_bh, inner_pc)?;
    let mut inner_oracles = vec![inner_o0.clone()];
    transcript.observe_field(inner_o0.c0);
    transcript.observe_field(inner_o0.c1);
    transcript.observe_field(inner_o0.c2);

    let mut folded_roots = Vec::new();
    let mut folded_cws: Vec<DeviceBuffer<u64>> = Vec::new();
    let mut folded_trees_inner: Vec<DeviceMerkleTree> = Vec::new();
    let mut d_cw_cur = d_combined_cw.clone_on_device()?;
    let mut cw_pc = cw_len / 2;

    for round in 0..num_vars - 1 {
        let ch = transcript.sample_challenge();
        let mut d_eq_h = DeviceBuffer::<u64>::new(inner_pc)?;
        let mut d_bh_h = DeviceBuffer::<u64>::new(inner_pc)?;
        BasefoldBatch::sumcheck_eval_gl(&d_inner_eq, ch, &mut d_eq_h, inner_pc)?;
        BasefoldBatch::sumcheck_eval_gl(&d_combined_bh, ch, &mut d_bh_h, inner_pc)?;
        inner_pc /= 2;
        d_inner_eq = d_eq_h;
        d_combined_bh = d_bh_h;

        BasefoldBatch::sumcheck_interp_gl(&mut d_inner_eq, inner_pc)?;
        BasefoldBatch::sumcheck_interp_gl(&mut d_combined_bh, inner_pc)?;
        let oracle = sumcheck_product_and_reduce_gl(&d_inner_eq, &d_combined_bh, inner_pc)?;
        inner_oracles.push(oracle.clone());
        transcript.observe_field(oracle.c0);
        transcript.observe_field(oracle.c1);
        transcript.observe_field(oracle.c2);

        let table_ptr = table.device_level_ptr(round)?;
        let mut d_folded = DeviceBuffer::<u64>::new(cw_pc)?;
        BasefoldBatch::fold_gl(&d_cw_cur, table_ptr, ch, &mut d_folded, cw_pc)?;

        let tree = DeviceMerkleTree::build_from_gl_codeword(&d_folded, cw_pc)?;
        let root = tree.root()?;
        transcript.observe_hash(&root);
        folded_roots.push(root);

        folded_cws.push(d_cw_cur);
        folded_trees_inner.push(tree);
        d_cw_cur = d_folded;
        cw_pc /= 2;
    }

    // Last fold
    let last_inner_ch = transcript.sample_challenge();
    let table_ptr = table.device_level_ptr(num_vars - 1)?;
    let mut d_final = DeviceBuffer::<u64>::new(cw_pc)?;
    BasefoldBatch::fold_gl(&d_cw_cur, table_ptr, last_inner_ch, &mut d_final, cw_pc)?;

    let final_raw = d_final.to_vec()?;
    let final_codeword: Vec<GoldilocksField> = final_raw.into_iter().map(GoldilocksField).collect();

    // ── 9. Query proofs ──
    let mut query_indices = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let idx_raw = transcript.sample_challenge().0 as usize;
        query_indices.push(idx_raw % (cw_len / 2));
    }

    // Combined codeword queries
    let mut combined_query_proofs = Vec::with_capacity(num_queries);
    for &leaf_idx in &query_indices {
        let pair = d_combined_cw.read_slice(leaf_idx * 2, 2)?;
        let mut values = vec![(GoldilocksField(pair[0]), GoldilocksField(pair[1]))];
        let mut paths = vec![combined_tree.auth_path(leaf_idx)?];
        let mut idx = leaf_idx / 2;
        for i in 0..folded_cws.len() {
            let pair_off = idx * 2;
            if pair_off + 1 < folded_cws[i].len() {
                let p = folded_cws[i].read_slice(pair_off, 2)?;
                values.push((GoldilocksField(p[0]), GoldilocksField(p[1])));
                if i < folded_trees_inner.len() {
                    paths.push(folded_trees_inner[i].auth_path(idx)?);
                }
            }
            idx /= 2;
        }
        combined_query_proofs.push(QueryProof {
            index: leaf_idx,
            values,
            merkle_paths: paths,
        });
    }

    // Individual commitment queries
    let mut individual_query_proofs = Vec::with_capacity(num_queries);
    for &leaf_idx in &query_indices {
        let mut eval_proofs = Vec::with_capacity(evals.len());
        for eval in evals {
            let comm = comms[eval.poly];
            let pair = comm.d_codeword.read_slice(leaf_idx * 2, 2)?;
            let path = comm.merkle_tree.auth_path(leaf_idx)?;
            eval_proofs.push(IndividualQueryProof {
                values: (GoldilocksField(pair[0]), GoldilocksField(pair[1])),
                merkle_path: path,
            });
        }
        individual_query_proofs.push(eval_proofs);
    }

    Ok(BatchBasefoldProof {
        outer_sumcheck_oracles: outer_oracles,
        combined_root,
        inner_eval,
        inner_sumcheck_oracles: inner_oracles,
        folded_roots,
        final_codeword,
        combined_query_proofs,
        individual_query_proofs,
    })
}

/// Batch opening: open multiple base-field polynomials at (possibly different) ext2 points.
///
/// All polynomials must share the same `num_vars` and `log_rate`.
///
/// Algorithm:
/// 1. Random batching weights from `eq(x, t)` where t ← transcript (base field).
/// 2. Outer sum-check: round 0 mixed (base-field g × ext2 eq), rounds 1+ pure ext2.
/// 3. Combined ext2 codeword via decomposed base-field accumulation + interleave.
/// 4. Inner commit phase: ext2 sum-check + ext2 FRI folding.
/// 5. Query proofs: combined = ext2 pairs, individual = base-field pairs.
pub fn batch_open_ext2(
    comms: &[&BasefoldCommitment],
    points: &[&[GoldilocksExt2]],
    evals: &[EvaluationExt2],
    table: &BasefoldTable,
    transcript: &mut impl BasefoldTranscript,
    num_queries: usize,
) -> Result<BatchBasefoldProofExt2> {
    assert!(!comms.is_empty());
    assert!(!evals.is_empty());
    let num_vars = comms[0].num_vars;
    let log_rate = comms[0].log_rate;
    for c in comms {
        assert_eq!(c.num_vars, num_vars);
        assert_eq!(c.log_rate, log_rate);
    }
    let n = 1usize << num_vars;
    let num_points = points.len();

    // ── 1. Transcript: observe commitments, ext2 points, ext2 values ──
    for comm in comms {
        transcript.observe_hash(&comm.root);
    }
    for point in points {
        for p in *point {
            transcript.observe_ext2(*p);
        }
    }
    for e in evals {
        transcript.observe_ext2(e.value);
    }

    // ── 2. Random batching weights (base field) ──
    let num_evals = evals.len();
    let ell = if num_evals <= 1 {
        1
    } else {
        num_evals.next_power_of_two().trailing_zeros() as usize
    };
    let t: Vec<GoldilocksField> = (0..ell)
        .map(|_| transcript.sample_challenge())
        .collect();
    let eq_xt = eq_poly_host(&t);

    // ── 3. Merge bh_evals per point on GPU (base field) ──
    let mut bh_cache: Vec<Option<DeviceBuffer<u64>>> = (0..comms.len()).map(|_| None).collect();
    let mut d_g_bufs: Vec<DeviceBuffer<u64>> = (0..num_points)
        .map(|_| gpu_zero_buffer(n))
        .collect::<Result<Vec<_>>>()?;
    let mut d_tmp = DeviceBuffer::<u64>::new(n)?;
    for (i, eval) in evals.iter().enumerate() {
        if bh_cache[eval.poly].is_none() {
            let mut br = comms[eval.poly].d_bh_evals.clone_on_device()?;
            BasefoldBatch::bit_reverse_gl(&mut br, num_vars)?;
            bh_cache[eval.poly] = Some(br);
        }
        gpu_accumulate_scaled(
            &mut d_g_bufs[eval.point],
            bh_cache[eval.poly].as_ref().unwrap(),
            &mut d_tmp,
            eq_xt[i],
            n,
        )?;
    }
    drop(d_tmp);
    drop(bh_cache);

    // ── 4. Ext2 eq polynomials for each evaluation point on GPU ──
    let mut d_eq_bufs: Vec<DeviceBuffer<u64>> = points
        .iter()
        .map(|pt| ext2_eq_dp_all_u64(pt, num_vars))
        .collect::<Result<Vec<_>>>()?;

    // ── 5. Outer sum-check round 0 (mixed: g in F_p, eq in F_{p^2}) ──
    let mut pair_count = n / 2;
    for p in 0..num_points {
        BasefoldBatch::sumcheck_interp_gl(&mut d_g_bufs[p], pair_count)?;
        BasefoldBatch::sumcheck_interp_ext2(&mut d_eq_bufs[p], pair_count)?;
    }
    let oracle0 = multi_point_sumcheck_product_mixed(&d_eq_bufs, &d_g_bufs, pair_count)?;
    let mut outer_oracles = vec![oracle0.clone()];
    transcript.observe_ext2(oracle0.c0);
    transcript.observe_ext2(oracle0.c1);
    transcript.observe_ext2(oracle0.c2);

    // ── 6. Round 0 transition: mixed → ext2 ──
    let ch0 = transcript.sample_challenge_ext2();
    let mut outer_challenges: Vec<GoldilocksExt2> = vec![ch0];

    // Transition g (base field) → ext2
    let mut d_g_ext2_bufs: Vec<DeviceBuffer<u64>> = Vec::with_capacity(num_points);
    for p in 0..num_points {
        let mut d_g_ext2 = DeviceBuffer::<u64>::new(pair_count * 2)?;
        BasefoldBatch::sumcheck_eval_mixed(&d_g_bufs[p], ch0, &mut d_g_ext2, pair_count)?;
        d_g_ext2_bufs.push(d_g_ext2);
    }
    drop(d_g_bufs);

    // Transition eq (ext2) → ext2
    for p in 0..num_points {
        let mut d_eq_h = DeviceBuffer::<u64>::new(pair_count * 2)?;
        BasefoldBatch::sumcheck_eval_ext2(&d_eq_bufs[p], ch0, &mut d_eq_h, pair_count)?;
        d_eq_bufs[p] = d_eq_h;
    }
    pair_count /= 2;

    // Interp + product for oracle[1]
    for p in 0..num_points {
        BasefoldBatch::sumcheck_interp_ext2(&mut d_eq_bufs[p], pair_count)?;
        BasefoldBatch::sumcheck_interp_ext2(&mut d_g_ext2_bufs[p], pair_count)?;
    }
    let oracle1 = multi_point_sumcheck_product_ext2(&d_eq_bufs, &d_g_ext2_bufs, pair_count)?;
    outer_oracles.push(oracle1.clone());
    transcript.observe_ext2(oracle1.c0);
    transcript.observe_ext2(oracle1.c1);
    transcript.observe_ext2(oracle1.c2);

    // ── 7. Remaining outer rounds (pure ext2) ──
    for _round in 1..num_vars - 1 {
        let ch = transcript.sample_challenge_ext2();
        outer_challenges.push(ch);
        for p in 0..num_points {
            let mut d_eq_h = DeviceBuffer::<u64>::new(pair_count * 2)?;
            let mut d_g_h = DeviceBuffer::<u64>::new(pair_count * 2)?;
            BasefoldBatch::sumcheck_eval_ext2(&d_eq_bufs[p], ch, &mut d_eq_h, pair_count)?;
            BasefoldBatch::sumcheck_eval_ext2(&d_g_ext2_bufs[p], ch, &mut d_g_h, pair_count)?;
            d_eq_bufs[p] = d_eq_h;
            d_g_ext2_bufs[p] = d_g_h;
        }
        pair_count /= 2;
        for p in 0..num_points {
            BasefoldBatch::sumcheck_interp_ext2(&mut d_eq_bufs[p], pair_count)?;
            BasefoldBatch::sumcheck_interp_ext2(&mut d_g_ext2_bufs[p], pair_count)?;
        }
        let oracle = multi_point_sumcheck_product_ext2(&d_eq_bufs, &d_g_ext2_bufs, pair_count)?;
        outer_oracles.push(oracle.clone());
        transcript.observe_ext2(oracle.c0);
        transcript.observe_ext2(oracle.c1);
        transcript.observe_ext2(oracle.c2);
    }
    let last_ch = transcript.sample_challenge_ext2();
    outer_challenges.push(last_ch);
    let r = outer_challenges; // ext2 challenge vector
    drop(d_eq_bufs);
    drop(d_g_ext2_bufs);

    // ── 8. Combined ext2 codeword + bh_evals (decomposed base-field accumulation) ──
    let eq_r_pts: Vec<GoldilocksExt2> = points
        .iter()
        .map(|pt| eq_eval_host_ext2(&r, pt))
        .collect();
    let cw_len = 1usize << (num_vars + log_rate);
    let mut d_cw_c0 = gpu_zero_buffer(cw_len)?;
    let mut d_cw_c1 = gpu_zero_buffer(cw_len)?;
    let mut d_bh_c0 = gpu_zero_buffer(n)?;
    let mut d_bh_c1 = gpu_zero_buffer(n)?;
    let mut d_tmp_cw = DeviceBuffer::<u64>::new(cw_len)?;
    let mut d_tmp_bh = DeviceBuffer::<u64>::new(n)?;
    for (i, eval) in evals.iter().enumerate() {
        // scalar = eq_r_pts[eval.point] * eq_xt[i] (ext2 * base_field = ext2)
        let eq_xt_ext2 = GoldilocksExt2::new(eq_xt[i], GoldilocksField(0));
        let scalar = ext2_mul_host(eq_r_pts[eval.point], eq_xt_ext2);
        gpu_accumulate_scaled_ext2_from_gl(
            &mut d_cw_c0, &mut d_cw_c1,
            &comms[eval.poly].d_codeword,
            &mut d_tmp_cw, scalar, cw_len,
        )?;
        gpu_accumulate_scaled_ext2_from_gl(
            &mut d_bh_c0, &mut d_bh_c1,
            &comms[eval.poly].d_bh_evals,
            &mut d_tmp_bh, scalar, n,
        )?;
    }
    drop(d_tmp_cw);
    drop(d_tmp_bh);

    // Interleave to ext2 layout
    let d_combined_cw = interleave_to_ext2(&d_cw_c0, &d_cw_c1, cw_len)?;
    let mut d_combined_bh = interleave_to_ext2(&d_bh_c0, &d_bh_c1, n)?;
    drop(d_cw_c0);
    drop(d_cw_c1);
    drop(d_bh_c0);
    drop(d_bh_c1);

    // ── 9. Combined Merkle tree (ext2) ──
    let combined_tree = DeviceMerkleTree::build_from_ext2_codeword(&d_combined_cw, cw_len)?;
    let combined_root = combined_tree.root()?;
    transcript.observe_hash(&combined_root);

    // ── 10. Inner commit phase (ext2 sum-check + ext2 folding) ──
    let mut d_inner_eq = ext2_eq_dp_all_u64(&r, num_vars)?;
    BasefoldBatch::bit_reverse_ext2(&mut d_combined_bh, num_vars)?;

    let mut inner_pc = n / 2;
    BasefoldBatch::sumcheck_interp_ext2(&mut d_inner_eq, inner_pc)?;
    BasefoldBatch::sumcheck_interp_ext2(&mut d_combined_bh, inner_pc)?;
    let inner_o0 = sumcheck_product_and_reduce_ext2(&d_inner_eq, &d_combined_bh, inner_pc)?;

    // inner_eval from first oracle: p(0) + p(1) (avoids needing ext2 dot product kernel)
    let p_at_0 = inner_o0.c0;
    let p_at_1 = ext2_add_host(inner_o0.c0, ext2_add_host(inner_o0.c1, inner_o0.c2));
    let inner_eval = ext2_add_host(p_at_0, p_at_1);

    let mut inner_oracles = vec![inner_o0.clone()];
    transcript.observe_ext2(inner_o0.c0);
    transcript.observe_ext2(inner_o0.c1);
    transcript.observe_ext2(inner_o0.c2);

    let mut folded_roots = Vec::new();
    let mut folded_cws: Vec<DeviceBuffer<u64>> = Vec::new();
    let mut folded_trees_inner: Vec<DeviceMerkleTree> = Vec::new();
    let mut d_cw_cur = d_combined_cw.clone_on_device()?;
    let mut cw_pc = cw_len / 2;

    for round in 0..num_vars - 1 {
        let ch = transcript.sample_challenge_ext2();
        let mut d_eq_h = DeviceBuffer::<u64>::new(inner_pc * 2)?;
        let mut d_bh_h = DeviceBuffer::<u64>::new(inner_pc * 2)?;
        BasefoldBatch::sumcheck_eval_ext2(&d_inner_eq, ch, &mut d_eq_h, inner_pc)?;
        BasefoldBatch::sumcheck_eval_ext2(&d_combined_bh, ch, &mut d_bh_h, inner_pc)?;
        inner_pc /= 2;
        d_inner_eq = d_eq_h;
        d_combined_bh = d_bh_h;

        BasefoldBatch::sumcheck_interp_ext2(&mut d_inner_eq, inner_pc)?;
        BasefoldBatch::sumcheck_interp_ext2(&mut d_combined_bh, inner_pc)?;
        let oracle = sumcheck_product_and_reduce_ext2(&d_inner_eq, &d_combined_bh, inner_pc)?;
        inner_oracles.push(oracle.clone());
        transcript.observe_ext2(oracle.c0);
        transcript.observe_ext2(oracle.c1);
        transcript.observe_ext2(oracle.c2);

        let table_ptr = table.device_level_ptr(round)?;
        let mut d_folded = DeviceBuffer::<u64>::new(cw_pc * 2)?; // ext2
        BasefoldBatch::fold_ext2(&d_cw_cur, table_ptr, ch, &mut d_folded, cw_pc)?;

        let tree = DeviceMerkleTree::build_from_ext2_codeword(&d_folded, cw_pc)?;
        let root = tree.root()?;
        transcript.observe_hash(&root);
        folded_roots.push(root);

        // Push fold RESULT (not input) for query extraction — must match folded_trees
        folded_cws.push(d_folded.clone_on_device()?);
        folded_trees_inner.push(tree);
        drop(d_cw_cur);
        d_cw_cur = d_folded;
        cw_pc /= 2;
    }

    // Last fold
    let last_inner_ch = transcript.sample_challenge_ext2();
    let table_ptr = table.device_level_ptr(num_vars - 1)?;
    let mut d_final = DeviceBuffer::<u64>::new(cw_pc * 2)?;
    BasefoldBatch::fold_ext2(&d_cw_cur, table_ptr, last_inner_ch, &mut d_final, cw_pc)?;

    let final_raw = d_final.to_vec()?;
    let final_codeword: Vec<GoldilocksExt2> = final_raw
        .chunks_exact(2)
        .map(|c| GoldilocksExt2::new(GoldilocksField(c[0]), GoldilocksField(c[1])))
        .collect();

    // ── 11. Query proofs ──
    let mut query_indices = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let idx_raw = transcript.sample_challenge().0 as usize;
        query_indices.push(idx_raw % (cw_len / 2));
    }

    // Combined codeword queries (ext2 pairs)
    let mut combined_query_proofs = Vec::with_capacity(num_queries);
    for &leaf_idx in &query_indices {
        let pair_off = leaf_idx * 4; // 2 ext2 elements × 2 u64 each
        let p = d_combined_cw.read_slice(pair_off, 4)?;
        let mut values = vec![(
            GoldilocksExt2::new(GoldilocksField(p[0]), GoldilocksField(p[1])),
            GoldilocksExt2::new(GoldilocksField(p[2]), GoldilocksField(p[3])),
        )];
        let mut paths = vec![combined_tree.auth_path(leaf_idx)?];
        let mut idx = leaf_idx / 2;
        for i in 0..folded_cws.len() {
            let pair_off = idx * 4;
            if pair_off + 3 < folded_cws[i].len() {
                let p = folded_cws[i].read_slice(pair_off, 4)?;
                values.push((
                    GoldilocksExt2::new(GoldilocksField(p[0]), GoldilocksField(p[1])),
                    GoldilocksExt2::new(GoldilocksField(p[2]), GoldilocksField(p[3])),
                ));
                if i < folded_trees_inner.len() {
                    paths.push(folded_trees_inner[i].auth_path(idx)?);
                }
            }
            idx /= 2;
        }
        combined_query_proofs.push(QueryProof {
            index: leaf_idx,
            values,
            merkle_paths: paths,
        });
    }

    // Individual commitment queries (base-field pairs)
    let mut individual_query_proofs = Vec::with_capacity(num_queries);
    for &leaf_idx in &query_indices {
        let mut eval_proofs = Vec::with_capacity(evals.len());
        for eval in evals {
            let comm = comms[eval.poly];
            let pair = comm.d_codeword.read_slice(leaf_idx * 2, 2)?;
            let path = comm.merkle_tree.auth_path(leaf_idx)?;
            eval_proofs.push(IndividualQueryProof {
                values: (GoldilocksField(pair[0]), GoldilocksField(pair[1])),
                merkle_path: path,
            });
        }
        individual_query_proofs.push(eval_proofs);
    }

    Ok(BatchBasefoldProofExt2 {
        outer_sumcheck_oracles: outer_oracles,
        combined_root,
        inner_eval,
        inner_sumcheck_oracles: inner_oracles,
        folded_roots,
        final_codeword,
        combined_query_proofs,
        individual_query_proofs,
    })
}

// ============================================================================
// Verifier
// ============================================================================

pub struct BasefoldVerifier;

impl BasefoldVerifier {
    /// Verify a base-field opening proof. All CPU.
    pub fn verify(
        root: &Poseidon2Hash,
        point: &[GoldilocksField],
        proof: &BasefoldProof,
        _table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
    ) -> Result<bool> {
        let num_vars = point.len();

        // Re-derive transcript state
        transcript.observe_hash(root);
        for p in point {
            transcript.observe_field(*p);
        }

        // Check: oracle[0] evaluated at 0 and 1 should sum to eval
        let o0 = &proof.sumcheck_oracles[0];
        // p(0) = c0, p(1) = c0 + c1 + c2, so p(0) + p(1) = 2*c0 + c1 + c2 should = eval
        let p_at_0 = o0.c0.0;
        let p_at_1 = gl_add_host(o0.c0.0, gl_add_host(o0.c1.0, o0.c2.0));
        let sum = gl_add_host(p_at_0, p_at_1);
        if sum != proof.eval.0 {
            return Ok(false);
        }

        // Observe first oracle
        transcript.observe_field(o0.c0);
        transcript.observe_field(o0.c1);
        transcript.observe_field(o0.c2);

        // Check sum-check transitions
        let mut challenges = Vec::with_capacity(num_vars);
        for round in 0..num_vars - 1 {
            let challenge = transcript.sample_challenge();
            challenges.push(challenge);

            // oracle[round] evaluated at challenge
            let o = &proof.sumcheck_oracles[round];
            let val_at_challenge = gl_add_host(
                o.c0.0,
                gl_add_host(
                    gl_mul_host(o.c1.0, challenge.0),
                    gl_mul_host(o.c2.0, gl_mul_host(challenge.0, challenge.0)),
                ),
            );

            // Should equal oracle[round+1].p(0) + oracle[round+1].p(1)
            let o_next = &proof.sumcheck_oracles[round + 1];
            let next_p0 = o_next.c0.0;
            let next_p1 = gl_add_host(o_next.c0.0, gl_add_host(o_next.c1.0, o_next.c2.0));
            let next_sum = gl_add_host(next_p0, next_p1);
            if val_at_challenge != next_sum {
                return Ok(false);
            }

            // Observe next oracle + root
            transcript.observe_field(o_next.c0);
            transcript.observe_field(o_next.c1);
            transcript.observe_field(o_next.c2);

            if round < proof.folded_roots.len() {
                transcript.observe_hash(&proof.folded_roots[round]);
            }
        }

        Ok(true)
    }

    /// Verify a batch opening proof. All CPU.
    ///
    /// Checks:
    /// 1. Outer sum-check: `sum_i eq_xt[i] * eval_i` equals claimed sum.
    /// 2. Outer oracle transitions are consistent.
    /// 3. Inner eval equals the outer sum-check's final evaluated value.
    /// 4. Inner sum-check: `inner_eval` equals the inner oracle initial sum.
    /// 5. Inner oracle transitions are consistent.
    pub fn batch_verify(
        roots: &[Poseidon2Hash],
        points: &[&[GoldilocksField]],
        evals: &[Evaluation],
        proof: &BatchBasefoldProof,
        _table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
    ) -> Result<bool> {
        let num_vars = points[0].len();

        // ── Re-derive transcript state ──
        for root in roots {
            transcript.observe_hash(root);
        }
        for point in points {
            for p in *point {
                transcript.observe_field(*p);
            }
        }
        for e in evals {
            transcript.observe_field(e.value);
        }

        // ── Random weights ──
        let num_evals = evals.len();
        let ell = if num_evals <= 1 {
            1
        } else {
            num_evals.next_power_of_two().trailing_zeros() as usize
        };
        let t: Vec<GoldilocksField> = (0..ell)
            .map(|_| transcript.sample_challenge())
            .collect();
        let eq_xt = eq_poly_host(&t);

        // ── Claimed sum ──
        let mut claimed_sum = 0u64;
        for (i, eval) in evals.iter().enumerate() {
            claimed_sum = gl_add_host(claimed_sum, gl_mul_host(eq_xt[i].0, eval.value.0));
        }

        // ── Outer sum-check ──
        let o0 = &proof.outer_sumcheck_oracles[0];
        let p0 = o0.c0.0;
        let p1 = gl_add_host(o0.c0.0, gl_add_host(o0.c1.0, o0.c2.0));
        if gl_add_host(p0, p1) != claimed_sum {
            return Ok(false);
        }

        transcript.observe_field(o0.c0);
        transcript.observe_field(o0.c1);
        transcript.observe_field(o0.c2);

        let mut outer_challenges = Vec::with_capacity(num_vars);
        for round in 0..num_vars - 1 {
            let ch = transcript.sample_challenge();
            outer_challenges.push(ch);

            let o = &proof.outer_sumcheck_oracles[round];
            let val = gl_add_host(
                o.c0.0,
                gl_add_host(
                    gl_mul_host(o.c1.0, ch.0),
                    gl_mul_host(o.c2.0, gl_mul_host(ch.0, ch.0)),
                ),
            );

            let o_next = &proof.outer_sumcheck_oracles[round + 1];
            let np0 = o_next.c0.0;
            let np1 = gl_add_host(o_next.c0.0, gl_add_host(o_next.c1.0, o_next.c2.0));
            if val != gl_add_host(np0, np1) {
                return Ok(false);
            }

            transcript.observe_field(o_next.c0);
            transcript.observe_field(o_next.c1);
            transcript.observe_field(o_next.c2);
        }
        let last_ch = transcript.sample_challenge();
        outer_challenges.push(last_ch);

        // ── Outer final value should equal inner_eval ──
        let last_outer = &proof.outer_sumcheck_oracles[num_vars - 1];
        let outer_final = gl_add_host(
            last_outer.c0.0,
            gl_add_host(
                gl_mul_host(last_outer.c1.0, last_ch.0),
                gl_mul_host(last_outer.c2.0, gl_mul_host(last_ch.0, last_ch.0)),
            ),
        );
        if outer_final != proof.inner_eval.0 {
            return Ok(false);
        }

        // ── Observe combined root ──
        transcript.observe_hash(&proof.combined_root);

        // ── Inner sum-check ──
        let io0 = &proof.inner_sumcheck_oracles[0];
        let ip0 = io0.c0.0;
        let ip1 = gl_add_host(io0.c0.0, gl_add_host(io0.c1.0, io0.c2.0));
        if gl_add_host(ip0, ip1) != proof.inner_eval.0 {
            return Ok(false);
        }

        transcript.observe_field(io0.c0);
        transcript.observe_field(io0.c1);
        transcript.observe_field(io0.c2);

        for round in 0..num_vars - 1 {
            let ch = transcript.sample_challenge();

            let o = &proof.inner_sumcheck_oracles[round];
            let val = gl_add_host(
                o.c0.0,
                gl_add_host(
                    gl_mul_host(o.c1.0, ch.0),
                    gl_mul_host(o.c2.0, gl_mul_host(ch.0, ch.0)),
                ),
            );

            let o_next = &proof.inner_sumcheck_oracles[round + 1];
            let np0 = o_next.c0.0;
            let np1 = gl_add_host(o_next.c0.0, gl_add_host(o_next.c1.0, o_next.c2.0));
            if val != gl_add_host(np0, np1) {
                return Ok(false);
            }

            transcript.observe_field(o_next.c0);
            transcript.observe_field(o_next.c1);
            transcript.observe_field(o_next.c2);

            if round < proof.folded_roots.len() {
                transcript.observe_hash(&proof.folded_roots[round]);
            }
        }

        Ok(true)
    }

    /// Verify an ext2 opening proof. All CPU.
    ///
    /// Checks:
    /// 1. Sumcheck oracle consistency (P(0)+P(1)=eval, round transitions)
    /// 2. If query_proofs are present: Merkle auth paths, codeword fold consistency, final codeword
    pub fn verify_ext2(
        root: &Poseidon2Hash,
        point: &[GoldilocksExt2],
        proof: &BasefoldProofExt2,
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
    ) -> Result<bool> {
        let num_vars = point.len();

        // Re-derive transcript state
        transcript.observe_hash(root);
        for p in point {
            transcript.observe_ext2(*p);
        }

        // Check: oracle[0] p(0) + p(1) should sum to eval
        let o0 = &proof.sumcheck_oracles[0];
        let p_at_0 = o0.c0;
        let p_at_1 = ext2_add_host(o0.c0, ext2_add_host(o0.c1, o0.c2));
        let sum = ext2_add_host(p_at_0, p_at_1);
        if !ext2_field_eq(sum, proof.eval) {
            eprintln!("[verify_ext2] FAIL: oracle[0] sum check. sum={:?} eval={:?}", sum, proof.eval);
            return Ok(false);
        }

        // Observe first oracle
        transcript.observe_ext2(o0.c0);
        transcript.observe_ext2(o0.c1);
        transcript.observe_ext2(o0.c2);

        // Collect challenges for fold verification
        let mut challenges: Vec<GoldilocksExt2> = Vec::with_capacity(num_vars);

        // Check sum-check transitions
        for round in 0..num_vars - 1 {
            let challenge = transcript.sample_challenge_ext2();
            challenges.push(challenge);

            // oracle[round] evaluated at challenge
            let o = &proof.sumcheck_oracles[round];
            let ch_sq = ext2_mul_host(challenge, challenge);
            let val_at_challenge = ext2_add_host(
                o.c0,
                ext2_add_host(
                    ext2_mul_host(o.c1, challenge),
                    ext2_mul_host(o.c2, ch_sq),
                ),
            );

            // Should equal oracle[round+1].p(0) + oracle[round+1].p(1)
            let o_next = &proof.sumcheck_oracles[round + 1];
            let next_p0 = o_next.c0;
            let next_p1 = ext2_add_host(o_next.c0, ext2_add_host(o_next.c1, o_next.c2));
            let next_sum = ext2_add_host(next_p0, next_p1);
            if !ext2_field_eq(val_at_challenge, next_sum) {
                eprintln!("[verify_ext2] FAIL: sumcheck transition round {}. val_at_challenge={:?} next_sum={:?}", round, val_at_challenge, next_sum);
                return Ok(false);
            }

            // Observe next oracle + root
            transcript.observe_ext2(o_next.c0);
            transcript.observe_ext2(o_next.c1);
            transcript.observe_ext2(o_next.c2);

            if round < proof.folded_roots.len() {
                transcript.observe_hash(&proof.folded_roots[round]);
            }
        }

        // Last challenge (for final fold)
        let last_challenge = transcript.sample_challenge_ext2();
        challenges.push(last_challenge);

        // ── Query proof verification ──
        // Skip if no query proofs (backward compat with old sumcheck-only proofs)
        if !proof.query_proofs.is_empty() {
            let num_queries = proof.query_proofs.len();
            let has_final_cw = !proof.final_codeword.is_empty();

            for q_idx in 0..num_queries {
                let query = &proof.query_proofs[q_idx];
                let _query_challenge = transcript.sample_challenge();

                // 1. Verify initial codeword auth path against commitment root
                // values[0] is a GL pair lifted to ext2 (c1 should be 0)
                let (v0, v1) = query.values[0];
                let leaf_hash = hash_gl_leaf(v0.c0, v1.c0);
                if !verify_auth_path(&leaf_hash, &query.merkle_paths[0], query.index, root) {
                    eprintln!("[verify_ext2] FAIL: initial auth path q={} index={}", q_idx, query.index);
                    return Ok(false);
                }

                // 2. Verify fold consistency across rounds
                let mut fold_idx = query.index;
                for r in 0..num_vars {
                    let (val0, val1) = query.values[r];
                    let table_offset = table.level_offsets[r];
                    let entry = &table.entries[table_offset + fold_idx];
                    let x0 = entry.point;
                    let w = entry.weight;

                    // Fold formula: result = val0 + (challenge - x0) * (val1 - val0) * w
                    let challenge_r = challenges[r];
                    let fold_result = if r == 0 {
                        // Round 0: mixed fold (val0, val1 are GL lifted to ext2)
                        let diff = ext2_sub_host(val1, val0);
                        let cx = ext2_sub_host(challenge_r, GoldilocksExt2::from_base(x0));
                        let w_ext2 = GoldilocksExt2::from_base(w);
                        ext2_add_host(val0, ext2_mul_host(ext2_mul_host(cx, diff), w_ext2))
                    } else {
                        // Rounds 1+: pure ext2 fold
                        let diff = ext2_sub_host(val1, val0);
                        let cx = ext2_sub_host(challenge_r, GoldilocksExt2::from_base(x0));
                        let w_ext2 = GoldilocksExt2::from_base(w);
                        ext2_add_host(val0, ext2_mul_host(ext2_mul_host(cx, diff), w_ext2))
                    };

                    if r < num_vars - 1 {
                        // Check fold result appears correctly in the next round's values
                        let (next_v0, next_v1) = query.values[r + 1];
                        let expected = if fold_idx & 1 == 0 { next_v0 } else { next_v1 };
                        if !ext2_field_eq(fold_result, expected) {
                            eprintln!("[verify_ext2] FAIL: fold consistency q={} r={} fold_idx={} n={}", q_idx, r, fold_idx, num_vars);
                            eprintln!("  challenge_r={:?}", challenge_r);
                            eprintln!("  val0={:?} val1={:?}", val0, val1);
                            eprintln!("  table entry: x0={:?} w={:?}", x0, w);
                            eprintln!("  fold_result={:?}", fold_result);
                            eprintln!("  expected={:?} (fold_idx&1={})", expected, fold_idx & 1);
                            return Ok(false);
                        }

                        // Verify auth path for folded codeword against folded root
                        // merkle_paths[r+1] corresponds to folded_roots[r]
                        if r + 1 < query.merkle_paths.len() && r < proof.folded_roots.len() {
                            let next_leaf_hash = hash_ext2_leaf(next_v0, next_v1);
                            let next_idx = fold_idx / 2;
                            if !verify_auth_path(&next_leaf_hash, &query.merkle_paths[r + 1], next_idx, &proof.folded_roots[r]) {
                                eprintln!("[verify_ext2] FAIL: folded auth path q={} r={} next_idx={}", q_idx, r, next_idx);
                                return Ok(false);
                            }
                        }
                    } else if has_final_cw {
                        // Last round: check fold result against final codeword
                        let final_idx = fold_idx / 2;
                        let expected = if fold_idx & 1 == 0 {
                            proof.final_codeword[final_idx * 2]
                        } else {
                            proof.final_codeword[final_idx * 2 + 1]
                        };
                        if !ext2_field_eq(fold_result, expected) {
                            eprintln!("[verify_ext2] FAIL: final cw q={} fold_idx={} final_idx={}", q_idx, fold_idx, final_idx);
                            return Ok(false);
                        }
                    }

                    fold_idx /= 2;
                }
            }
        } else {
            // No query proofs — just consume transcript challenges to stay in sync
            for _ in 0..0 {
                // Old proofs don't have query phase — transcript was already consumed by caller
            }
        }

        Ok(true)
    }

    /// Verify a batch ext2 opening proof. All CPU.
    ///
    /// Checks outer sumcheck, inner sumcheck, combined query proofs (Merkle + fold),
    /// individual query proofs (Merkle), and linear combination consistency.
    pub fn batch_verify_ext2(
        roots: &[Poseidon2Hash],
        points: &[&[GoldilocksExt2]],
        evals: &[EvaluationExt2],
        proof: &BatchBasefoldProofExt2,
        table: &BasefoldTable,
        transcript: &mut impl BasefoldTranscript,
        log_rate: usize,
    ) -> Result<bool> {
        let num_vars = points[0].len();

        // ── Re-derive transcript state ──
        for root in roots {
            transcript.observe_hash(root);
        }
        for point in points {
            for p in *point {
                transcript.observe_ext2(*p);
            }
        }
        for e in evals {
            transcript.observe_ext2(e.value);
        }

        // ── Random weights (base field) ──
        let num_evals = evals.len();
        let ell = if num_evals <= 1 {
            1
        } else {
            num_evals.next_power_of_two().trailing_zeros() as usize
        };
        let t: Vec<GoldilocksField> = (0..ell)
            .map(|_| transcript.sample_challenge())
            .collect();
        let eq_xt = eq_poly_host(&t);

        // ── Claimed sum (base_field * ext2 = ext2) ──
        let zero = GoldilocksExt2::new(GoldilocksField(0), GoldilocksField(0));
        let mut claimed_sum = zero;
        for (i, eval) in evals.iter().enumerate() {
            let w = GoldilocksExt2::new(eq_xt[i], GoldilocksField(0));
            claimed_sum = ext2_add_host(claimed_sum, ext2_mul_host(w, eval.value));
        }

        // ── Outer sum-check ──
        let o0 = &proof.outer_sumcheck_oracles[0];
        let p0 = o0.c0;
        let p1 = ext2_add_host(o0.c0, ext2_add_host(o0.c1, o0.c2));
        let sum = ext2_add_host(p0, p1);
        if !ext2_field_eq(sum, claimed_sum) {
            eprintln!("[batch_verify_ext2] FAIL: outer sumcheck initial sum");
            return Ok(false);
        }

        transcript.observe_ext2(o0.c0);
        transcript.observe_ext2(o0.c1);
        transcript.observe_ext2(o0.c2);

        let mut outer_challenges: Vec<GoldilocksExt2> = Vec::with_capacity(num_vars);
        for round in 0..num_vars - 1 {
            let ch = transcript.sample_challenge_ext2();
            outer_challenges.push(ch);

            let o = &proof.outer_sumcheck_oracles[round];
            let ch_sq = ext2_mul_host(ch, ch);
            let val = ext2_add_host(
                o.c0,
                ext2_add_host(
                    ext2_mul_host(o.c1, ch),
                    ext2_mul_host(o.c2, ch_sq),
                ),
            );

            let o_next = &proof.outer_sumcheck_oracles[round + 1];
            let np0 = o_next.c0;
            let np1 = ext2_add_host(o_next.c0, ext2_add_host(o_next.c1, o_next.c2));
            let nsum = ext2_add_host(np0, np1);
            if !ext2_field_eq(val, nsum) {
                eprintln!("[batch_verify_ext2] FAIL: outer sumcheck transition round {}", round);
                return Ok(false);
            }

            transcript.observe_ext2(o_next.c0);
            transcript.observe_ext2(o_next.c1);
            transcript.observe_ext2(o_next.c2);
        }
        let last_ch = transcript.sample_challenge_ext2();
        outer_challenges.push(last_ch);
        let r = &outer_challenges; // ext2 challenge vector

        // ── Outer final value should equal inner_eval ──
        let last_outer = &proof.outer_sumcheck_oracles[num_vars - 1];
        let last_ch_sq = ext2_mul_host(last_ch, last_ch);
        let outer_final = ext2_add_host(
            last_outer.c0,
            ext2_add_host(
                ext2_mul_host(last_outer.c1, last_ch),
                ext2_mul_host(last_outer.c2, last_ch_sq),
            ),
        );
        if !ext2_field_eq(outer_final, proof.inner_eval) {
            eprintln!("[batch_verify_ext2] FAIL: outer final != inner_eval");
            return Ok(false);
        }

        // ── Observe combined root ──
        transcript.observe_hash(&proof.combined_root);

        // ── Inner sum-check (ext2) ──
        let io0 = &proof.inner_sumcheck_oracles[0];
        let ip0 = io0.c0;
        let ip1 = ext2_add_host(io0.c0, ext2_add_host(io0.c1, io0.c2));
        let isum = ext2_add_host(ip0, ip1);
        if !ext2_field_eq(isum, proof.inner_eval) {
            eprintln!("[batch_verify_ext2] FAIL: inner sumcheck initial sum");
            return Ok(false);
        }

        transcript.observe_ext2(io0.c0);
        transcript.observe_ext2(io0.c1);
        transcript.observe_ext2(io0.c2);

        let mut inner_challenges: Vec<GoldilocksExt2> = Vec::with_capacity(num_vars);
        for round in 0..num_vars - 1 {
            let ch = transcript.sample_challenge_ext2();
            inner_challenges.push(ch);

            let o = &proof.inner_sumcheck_oracles[round];
            let ch_sq = ext2_mul_host(ch, ch);
            let val = ext2_add_host(
                o.c0,
                ext2_add_host(
                    ext2_mul_host(o.c1, ch),
                    ext2_mul_host(o.c2, ch_sq),
                ),
            );

            let o_next = &proof.inner_sumcheck_oracles[round + 1];
            let np0 = o_next.c0;
            let np1 = ext2_add_host(o_next.c0, ext2_add_host(o_next.c1, o_next.c2));
            let nsum = ext2_add_host(np0, np1);
            if !ext2_field_eq(val, nsum) {
                eprintln!("[batch_verify_ext2] FAIL: inner sumcheck transition round {}", round);
                return Ok(false);
            }

            transcript.observe_ext2(o_next.c0);
            transcript.observe_ext2(o_next.c1);
            transcript.observe_ext2(o_next.c2);

            if round < proof.folded_roots.len() {
                transcript.observe_hash(&proof.folded_roots[round]);
            }
        }
        // Last inner challenge (for final fold)
        let last_inner_ch = transcript.sample_challenge_ext2();
        inner_challenges.push(last_inner_ch);

        // ── Query proof verification ──
        if proof.combined_query_proofs.is_empty() {
            return Ok(true);
        }

        let cw_len = 1usize << (num_vars + log_rate);
        let num_queries = proof.combined_query_proofs.len();
        let has_final_cw = !proof.final_codeword.is_empty();

        // Recompute scalars for linear combination check: scalar_i = eq_xt[i] * eq(r, point_i)
        let eq_r_pts: Vec<GoldilocksExt2> = points
            .iter()
            .map(|pt| eq_eval_host_ext2(r, pt))
            .collect();
        let scalars: Vec<GoldilocksExt2> = evals
            .iter()
            .enumerate()
            .map(|(i, eval)| {
                let eq_xt_ext2 = GoldilocksExt2::new(eq_xt[i], GoldilocksField(0));
                ext2_mul_host(eq_r_pts[eval.point], eq_xt_ext2)
            })
            .collect();

        // Sample query indices (must match prover)
        let mut query_indices = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let idx_raw = transcript.sample_challenge().0 as usize;
            query_indices.push(idx_raw % (cw_len / 2));
        }

        for q_idx in 0..num_queries {
            let query = &proof.combined_query_proofs[q_idx];
            let leaf_idx = query_indices[q_idx];

            if query.index != leaf_idx {
                eprintln!("[batch_verify_ext2] FAIL: query index mismatch q={}", q_idx);
                return Ok(false);
            }

            // 1. Verify initial combined codeword auth path (ext2 leaves)
            let (v0, v1) = query.values[0];
            let leaf_hash = hash_ext2_leaf(v0, v1);
            if !verify_auth_path(&leaf_hash, &query.merkle_paths[0], leaf_idx, &proof.combined_root) {
                eprintln!("[batch_verify_ext2] FAIL: combined initial auth path q={}", q_idx);
                return Ok(false);
            }

            // 2. Verify fold consistency across inner rounds (all ext2)
            let mut fold_idx = leaf_idx;
            for round in 0..num_vars {
                let (val0, val1) = query.values[round];
                let table_offset = table.level_offsets[round];
                let entry = &table.entries[table_offset + fold_idx];
                let x0 = entry.point;
                let w = entry.weight;

                let challenge_r = inner_challenges[round];
                let diff = ext2_sub_host(val1, val0);
                let cx = ext2_sub_host(challenge_r, GoldilocksExt2::from_base(x0));
                let w_ext2 = GoldilocksExt2::from_base(w);
                let fold_result = ext2_add_host(val0, ext2_mul_host(ext2_mul_host(cx, diff), w_ext2));

                if round < num_vars - 1 {
                    let (next_v0, next_v1) = query.values[round + 1];
                    let expected = if fold_idx & 1 == 0 { next_v0 } else { next_v1 };
                    if !ext2_field_eq(fold_result, expected) {
                        eprintln!("[batch_verify_ext2] FAIL: combined fold consistency q={} round={}", q_idx, round);
                        return Ok(false);
                    }

                    // Verify auth path for folded round against folded_roots[round]
                    if round + 1 < query.merkle_paths.len() && round < proof.folded_roots.len() {
                        let next_leaf_hash = hash_ext2_leaf(next_v0, next_v1);
                        let next_idx = fold_idx / 2;
                        if !verify_auth_path(&next_leaf_hash, &query.merkle_paths[round + 1], next_idx, &proof.folded_roots[round]) {
                            eprintln!("[batch_verify_ext2] FAIL: combined folded auth path q={} round={}", q_idx, round);
                            return Ok(false);
                        }
                    }
                } else if has_final_cw {
                    let final_idx = fold_idx / 2;
                    let expected = if fold_idx & 1 == 0 {
                        proof.final_codeword[final_idx * 2]
                    } else {
                        proof.final_codeword[final_idx * 2 + 1]
                    };
                    if !ext2_field_eq(fold_result, expected) {
                        eprintln!("[batch_verify_ext2] FAIL: combined final cw q={}", q_idx);
                        return Ok(false);
                    }
                }

                fold_idx /= 2;
            }

            // 3. Verify individual query proofs and linear combination consistency
            let indiv_proofs = &proof.individual_query_proofs[q_idx];
            let mut combined_v0 = GoldilocksExt2::new(GoldilocksField(0), GoldilocksField(0));
            let mut combined_v1 = GoldilocksExt2::new(GoldilocksField(0), GoldilocksField(0));

            for (i, indiv) in indiv_proofs.iter().enumerate() {
                // Verify individual auth path against commitment root (GL leaves)
                let indiv_leaf_hash = hash_gl_leaf(indiv.values.0, indiv.values.1);
                if !verify_auth_path(&indiv_leaf_hash, &indiv.merkle_path, leaf_idx, &roots[evals[i].poly]) {
                    eprintln!("[batch_verify_ext2] FAIL: individual auth path q={} eval={}", q_idx, i);
                    return Ok(false);
                }

                // Accumulate: combined = Σ scalar_i * individual_pair_i
                let iv0 = GoldilocksExt2::from_base(indiv.values.0);
                let iv1 = GoldilocksExt2::from_base(indiv.values.1);
                combined_v0 = ext2_add_host(combined_v0, ext2_mul_host(scalars[i], iv0));
                combined_v1 = ext2_add_host(combined_v1, ext2_mul_host(scalars[i], iv1));
            }

            // Check linear combination: combined pair must equal the combined codeword pair
            let (comb_v0, comb_v1) = query.values[0];
            if !ext2_field_eq(combined_v0, comb_v0) || !ext2_field_eq(combined_v1, comb_v1) {
                eprintln!("[batch_verify_ext2] FAIL: linear combination check q={}", q_idx);
                eprintln!("  expected: ({:?}, {:?})", combined_v0, combined_v1);
                eprintln!("  got:      ({:?}, {:?})", comb_v0, comb_v1);
                return Ok(false);
            }
        }

        Ok(true)
    }
}

// ============================================================================
// Query extraction helpers
// ============================================================================

fn extract_gl_queries(
    d_initial_cw: &DeviceBuffer<u64>,
    initial_tree: &DeviceMerkleTree,
    folded_codewords: &[DeviceBuffer<u64>],
    folded_trees: &[DeviceMerkleTree],
    _d_last_cw: &DeviceBuffer<u64>,
    num_queries: usize,
    transcript: &mut impl BasefoldTranscript,
    initial_cw_len: usize,
) -> Result<Vec<QueryProof<GoldilocksField>>> {
    // Sample all query indices first.
    let mut leaf_indices = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let idx_raw = transcript.sample_challenge().0 as usize;
        leaf_indices.push(idx_raw % (initial_cw_len / 2));
    }

    // Batch-extract auth paths from all trees.
    let initial_paths = initial_tree.batch_auth_paths(&leaf_indices)?;
    let mut cascaded: Vec<Vec<usize>> = Vec::with_capacity(folded_trees.len());
    {
        let mut cur_indices: Vec<usize> = leaf_indices.iter().map(|&i| i / 2).collect();
        for _ in 0..folded_trees.len() {
            cascaded.push(cur_indices.clone());
            cur_indices = cur_indices.iter().map(|&i| i / 2).collect();
        }
    }
    let folded_paths: Vec<Vec<Vec<Poseidon2Hash>>> = folded_trees.iter().enumerate()
        .map(|(i, tree)| tree.batch_auth_paths(&cascaded[i]))
        .collect::<crate::error::Result<Vec<_>>>()?;

    // Bulk-download small codewords.
    let h_initial_cw: Option<Vec<u64>> = if d_initial_cw.len() <= BULK_DOWNLOAD_THRESHOLD {
        Some(d_initial_cw.to_vec()?)
    } else { None };
    let h_folded_cws: Vec<Option<Vec<u64>>> = folded_codewords.iter()
        .map(|d| if d.len() <= BULK_DOWNLOAD_THRESHOLD {
            Ok(Some(d.to_vec()?))
        } else { Ok(None) })
        .collect::<Result<Vec<_>>>()?;

    let mut proofs = Vec::with_capacity(num_queries);
    for q in 0..num_queries {
        let leaf_idx = leaf_indices[q];

        let pair_off = leaf_idx * 2;
        let (v0, v1) = if let Some(ref cw) = h_initial_cw {
            (cw[pair_off], cw[pair_off + 1])
        } else {
            let p = d_initial_cw.read_slice(pair_off, 2)?;
            (p[0], p[1])
        };
        let mut values = vec![(GoldilocksField(v0), GoldilocksField(v1))];
        let mut paths = vec![initial_paths[q].clone()];

        let mut idx = leaf_idx / 2;
        for i in 0..folded_codewords.len() {
            if i < folded_trees.len() {
                let cw_len = folded_codewords[i].len();
                let pair_off = idx * 2;
                if pair_off + 1 < cw_len {
                    let (a, b) = if let Some(ref cw) = h_folded_cws[i] {
                        (cw[pair_off], cw[pair_off + 1])
                    } else {
                        let p = folded_codewords[i].read_slice(pair_off, 2)?;
                        (p[0], p[1])
                    };
                    values.push((GoldilocksField(a), GoldilocksField(b)));
                    paths.push(folded_paths[i][q].clone());
                }
            }
            idx /= 2;
        }

        proofs.push(QueryProof {
            index: leaf_idx,
            values,
            merkle_paths: paths,
        });
    }

    Ok(proofs)
}

/// Threshold in u64 elements: codewords smaller than this are bulk-downloaded to host.
/// 512K u64 = 4MB.
const BULK_DOWNLOAD_THRESHOLD: usize = 512 * 1024;

fn extract_ext2_queries(
    d_initial_cw: &DeviceBuffer<u64>,
    initial_tree: &DeviceMerkleTree,
    folded_codewords: &[DeviceBuffer<u64>],
    folded_trees: &[DeviceMerkleTree],
    _d_last_cw: &DeviceBuffer<u64>,
    num_queries: usize,
    transcript: &mut impl BasefoldTranscript,
    initial_cw_len: usize,
) -> Result<Vec<QueryProof<GoldilocksExt2>>> {
    // Sample all query indices first (needed for batch_auth_paths).
    let mut leaf_indices = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let idx_raw = transcript.sample_challenge().0 as usize;
        leaf_indices.push(idx_raw % (initial_cw_len / 2));
    }

    // Batch-extract auth paths from all trees (level-based bulk download for large trees).
    let initial_paths = initial_tree.batch_auth_paths(&leaf_indices)?;
    // Compute cascaded indices for each folded tree level
    let mut cascaded: Vec<Vec<usize>> = Vec::with_capacity(folded_trees.len());
    {
        let mut cur_indices: Vec<usize> = leaf_indices.iter().map(|&i| i / 2).collect();
        for _ in 0..folded_trees.len() {
            cascaded.push(cur_indices.clone());
            cur_indices = cur_indices.iter().map(|&i| i / 2).collect();
        }
    }
    let folded_paths: Vec<Vec<Vec<Poseidon2Hash>>> = folded_trees.iter().enumerate()
        .map(|(i, tree)| tree.batch_auth_paths(&cascaded[i]))
        .collect::<crate::error::Result<Vec<_>>>()?;

    // Bulk-download small codewords, selective read for large ones.
    let h_initial_cw: Option<Vec<u64>> = if d_initial_cw.len() <= BULK_DOWNLOAD_THRESHOLD {
        Some(d_initial_cw.to_vec()?)
    } else { None };
    let h_folded_cws: Vec<Option<Vec<u64>>> = folded_codewords.iter()
        .map(|d| if d.len() <= BULK_DOWNLOAD_THRESHOLD {
            Ok(Some(d.to_vec()?))
        } else { Ok(None) })
        .collect::<Result<Vec<_>>>()?;

    // Assemble proofs from pre-fetched data.
    let mut proofs = Vec::with_capacity(num_queries);
    for q in 0..num_queries {
        let leaf_idx = leaf_indices[q];

        // Initial codeword: base field pair
        let pair_off = leaf_idx * 2;
        let (v0, v1) = if let Some(ref cw) = h_initial_cw {
            (cw[pair_off], cw[pair_off + 1])
        } else {
            let p = d_initial_cw.read_slice(pair_off, 2)?;
            (p[0], p[1])
        };
        let mut values = vec![(
            GoldilocksExt2::new(GoldilocksField(v0), GoldilocksField(0)),
            GoldilocksExt2::new(GoldilocksField(v1), GoldilocksField(0)),
        )];
        let mut paths = vec![initial_paths[q].clone()];

        // Subsequent rounds: ext2 codewords
        let mut idx = leaf_idx / 2;
        for i in 0..folded_codewords.len() {
            let pair_off = idx * 4;
            let cw_len = folded_codewords[i].len();
            if pair_off + 3 < cw_len {
                let (a, b, c, d) = if let Some(ref cw) = h_folded_cws[i] {
                    (cw[pair_off], cw[pair_off+1], cw[pair_off+2], cw[pair_off+3])
                } else {
                    let p = folded_codewords[i].read_slice(pair_off, 4)?;
                    (p[0], p[1], p[2], p[3])
                };
                values.push((
                    GoldilocksExt2::new(GoldilocksField(a), GoldilocksField(b)),
                    GoldilocksExt2::new(GoldilocksField(c), GoldilocksField(d)),
                ));
                if i < folded_paths.len() {
                    paths.push(folded_paths[i][q].clone());
                }
            }
            idx /= 2;
        }

        proofs.push(QueryProof {
            index: leaf_idx,
            values,
            merkle_paths: paths,
        });
    }

    Ok(proofs)
}

// ============================================================================
// Batch-level kernel wrappers (operate on DeviceBuffer, zero host copies)
// ============================================================================

/// Low-level basefold kernel wrappers operating directly on device buffers.
pub struct BasefoldBatch;

impl BasefoldBatch {
    pub fn bit_reverse_gl(data: &mut DeviceBuffer<u64>, log_n: usize) -> Result<()> {
        let ret =
            unsafe { ffi::basefold_bit_reverse_gl_ffi(data.as_mut_ptr(), log_n as c_int) };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn bit_reverse_ext2(data: &mut DeviceBuffer<u64>, log_n: usize) -> Result<()> {
        let ret =
            unsafe { ffi::basefold_bit_reverse_ext2_ffi(data.as_mut_ptr(), log_n as c_int) };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn bhc_interpolate(
        evals: &DeviceBuffer<u64>,
        coeffs: &mut DeviceBuffer<u64>,
        bh_evals: &mut DeviceBuffer<u64>,
        num_vars: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_bhc_interpolate_ffi(
                evals.as_ptr(),
                coeffs.as_mut_ptr(),
                bh_evals.as_mut_ptr(),
                num_vars as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn encode(
        coeffs: &DeviceBuffer<u64>,
        codeword: &mut DeviceBuffer<u64>,
        num_vars: usize,
        log_rate: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_encode_ffi(
                coeffs.as_ptr(),
                codeword.as_mut_ptr(),
                num_vars as c_int,
                log_rate as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn fold_gl(
        codeword: &DeviceBuffer<u64>,
        table_ptr: *const u64,
        challenge: GoldilocksField,
        output: &mut DeviceBuffer<u64>,
        pair_count: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_fold_gl_ffi(
                codeword.as_ptr(),
                table_ptr,
                challenge.0,
                output.as_mut_ptr(),
                pair_count as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn fold_mixed(
        codeword: &DeviceBuffer<u64>,
        table_ptr: *const u64,
        challenge: GoldilocksExt2,
        output: &mut DeviceBuffer<u64>,
        pair_count: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_fold_mixed_ffi(
                codeword.as_ptr(),
                table_ptr,
                challenge.c0.0,
                challenge.c1.0,
                output.as_mut_ptr(),
                pair_count as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn fold_ext2(
        codeword: &DeviceBuffer<u64>,
        table_ptr: *const u64,
        challenge: GoldilocksExt2,
        output: &mut DeviceBuffer<u64>,
        pair_count: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_fold_ext2_ffi(
                codeword.as_ptr(),
                table_ptr,
                challenge.c0.0,
                challenge.c1.0,
                output.as_mut_ptr(),
                pair_count as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_interp_gl(data: &mut DeviceBuffer<u64>, pair_count: usize) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_interp_gl_ffi(data.as_mut_ptr(), pair_count as c_int)
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_interp_ext2(data: &mut DeviceBuffer<u64>, pair_count: usize) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_interp_ext2_ffi(data.as_mut_ptr(), pair_count as c_int)
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_product_gl(
        eq: &DeviceBuffer<u64>,
        bh: &DeviceBuffer<u64>,
        partial_c0: &mut DeviceBuffer<u64>,
        partial_c1: &mut DeviceBuffer<u64>,
        partial_c2: &mut DeviceBuffer<u64>,
        pair_count: usize,
        num_blocks: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_product_gl_ffi(
                eq.as_ptr(),
                bh.as_ptr(),
                partial_c0.as_mut_ptr(),
                partial_c1.as_mut_ptr(),
                partial_c2.as_mut_ptr(),
                pair_count as c_int,
                num_blocks as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_product_mixed(
        eq: &DeviceBuffer<u64>,
        bh: &DeviceBuffer<u64>,
        partial_c0: &mut DeviceBuffer<u64>,
        partial_c1: &mut DeviceBuffer<u64>,
        partial_c2: &mut DeviceBuffer<u64>,
        pair_count: usize,
        num_blocks: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_product_mixed_ffi(
                eq.as_ptr(),
                bh.as_ptr(),
                partial_c0.as_mut_ptr(),
                partial_c1.as_mut_ptr(),
                partial_c2.as_mut_ptr(),
                pair_count as c_int,
                num_blocks as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_product_ext2(
        eq: &DeviceBuffer<u64>,
        bh: &DeviceBuffer<u64>,
        partial_c0: &mut DeviceBuffer<u64>,
        partial_c1: &mut DeviceBuffer<u64>,
        partial_c2: &mut DeviceBuffer<u64>,
        pair_count: usize,
        num_blocks: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_product_ext2_ffi(
                eq.as_ptr(),
                bh.as_ptr(),
                partial_c0.as_mut_ptr(),
                partial_c1.as_mut_ptr(),
                partial_c2.as_mut_ptr(),
                pair_count as c_int,
                num_blocks as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_eval_gl(
        data: &DeviceBuffer<u64>,
        challenge: GoldilocksField,
        output: &mut DeviceBuffer<u64>,
        pair_count: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_eval_gl_ffi(
                data.as_ptr(),
                challenge.0,
                output.as_mut_ptr(),
                pair_count as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_eval_mixed(
        data: &DeviceBuffer<u64>,
        challenge: GoldilocksExt2,
        output: &mut DeviceBuffer<u64>,
        pair_count: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_eval_mixed_ffi(
                data.as_ptr(),
                challenge.c0.0,
                challenge.c1.0,
                output.as_mut_ptr(),
                pair_count as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn sumcheck_eval_ext2(
        data: &DeviceBuffer<u64>,
        challenge: GoldilocksExt2,
        output: &mut DeviceBuffer<u64>,
        pair_count: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_sumcheck_eval_ext2_ffi(
                data.as_ptr(),
                challenge.c0.0,
                challenge.c1.0,
                output.as_mut_ptr(),
                pair_count as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn dot_product_gl(
        a: &DeviceBuffer<u64>,
        b: &DeviceBuffer<u64>,
        partial: &mut DeviceBuffer<u64>,
        n: usize,
        num_blocks: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_dot_product_gl_ffi(
                a.as_ptr(),
                b.as_ptr(),
                partial.as_mut_ptr(),
                n as c_int,
                num_blocks as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    pub fn dot_product_mixed(
        a: &DeviceBuffer<u64>,
        b: &DeviceBuffer<u64>,
        partial: &mut DeviceBuffer<u64>,
        n: usize,
        num_blocks: usize,
    ) -> Result<()> {
        let ret = unsafe {
            ffi::basefold_dot_product_mixed_ffi(
                a.as_ptr(),
                b.as_ptr(),
                partial.as_mut_ptr(),
                n as c_int,
                num_blocks as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }
}

// ============================================================================
// Partial-sum reduction helpers (small D→H transfers, piggybacking on Fiat-Shamir)
// ============================================================================

fn reduce_dot_product_gl(partial: &DeviceBuffer<u64>, num_blocks: usize) -> Result<GoldilocksField> {
    let vals = partial.read_slice(0, num_blocks)?;
    let mut sum = 0u64;
    for v in vals {
        sum = gl_add_host(sum, v);
    }
    Ok(GoldilocksField(sum))
}

pub fn reduce_dot_product_ext2(partial: &DeviceBuffer<u64>, num_blocks: usize) -> Result<GoldilocksExt2> {
    let vals = partial.read_slice(0, num_blocks * 2)?;
    let mut c0 = 0u64;
    let mut c1 = 0u64;
    for i in 0..num_blocks {
        c0 = gl_add_host(c0, vals[i * 2]);
        c1 = gl_add_host(c1, vals[i * 2 + 1]);
    }
    Ok(GoldilocksExt2::new(GoldilocksField(c0), GoldilocksField(c1)))
}

fn sumcheck_product_and_reduce_gl(
    eq: &DeviceBuffer<u64>,
    bh: &DeviceBuffer<u64>,
    pair_count: usize,
) -> Result<SumcheckOracle<GoldilocksField>> {
    let num_blocks = ((pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    let mut pc0 = DeviceBuffer::<u64>::new(num_blocks)?;
    let mut pc1 = DeviceBuffer::<u64>::new(num_blocks)?;
    let mut pc2 = DeviceBuffer::<u64>::new(num_blocks)?;
    BasefoldBatch::sumcheck_product_gl(eq, bh, &mut pc0, &mut pc1, &mut pc2, pair_count, num_blocks)?;

    let v0 = pc0.read_slice(0, num_blocks)?;
    let v1 = pc1.read_slice(0, num_blocks)?;
    let v2 = pc2.read_slice(0, num_blocks)?;

    let mut c0 = 0u64;
    let mut c1 = 0u64;
    let mut c2 = 0u64;
    for i in 0..num_blocks {
        c0 = gl_add_host(c0, v0[i]);
        c1 = gl_add_host(c1, v1[i]);
        c2 = gl_add_host(c2, v2[i]);
    }

    Ok(SumcheckOracle {
        c0: GoldilocksField(c0),
        c1: GoldilocksField(c1),
        c2: GoldilocksField(c2),
    })
}

pub fn sumcheck_product_and_reduce_mixed(
    eq: &DeviceBuffer<u64>,
    bh: &DeviceBuffer<u64>,
    pair_count: usize,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let num_blocks = ((pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    let mut pc0 = DeviceBuffer::<u64>::new(num_blocks * 2)?;
    let mut pc1 = DeviceBuffer::<u64>::new(num_blocks * 2)?;
    let mut pc2 = DeviceBuffer::<u64>::new(num_blocks * 2)?;
    BasefoldBatch::sumcheck_product_mixed(eq, bh, &mut pc0, &mut pc1, &mut pc2, pair_count, num_blocks)?;

    let v0 = pc0.read_slice(0, num_blocks * 2)?;
    let v1 = pc1.read_slice(0, num_blocks * 2)?;
    let v2 = pc2.read_slice(0, num_blocks * 2)?;

    let reduce_ext2 = |v: &[u64]| -> GoldilocksExt2 {
        let mut c0 = 0u64;
        let mut c1 = 0u64;
        for i in 0..num_blocks {
            c0 = gl_add_host(c0, v[i * 2]);
            c1 = gl_add_host(c1, v[i * 2 + 1]);
        }
        GoldilocksExt2::new(GoldilocksField(c0), GoldilocksField(c1))
    };

    Ok(SumcheckOracle {
        c0: reduce_ext2(&v0),
        c1: reduce_ext2(&v1),
        c2: reduce_ext2(&v2),
    })
}

pub fn sumcheck_product_and_reduce_ext2(
    eq: &DeviceBuffer<u64>,
    bh: &DeviceBuffer<u64>,
    pair_count: usize,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let num_blocks = ((pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    let mut pc0 = DeviceBuffer::<u64>::new(num_blocks * 2)?;
    let mut pc1 = DeviceBuffer::<u64>::new(num_blocks * 2)?;
    let mut pc2 = DeviceBuffer::<u64>::new(num_blocks * 2)?;
    BasefoldBatch::sumcheck_product_ext2(eq, bh, &mut pc0, &mut pc1, &mut pc2, pair_count, num_blocks)?;

    let v0 = pc0.read_slice(0, num_blocks * 2)?;
    let v1 = pc1.read_slice(0, num_blocks * 2)?;
    let v2 = pc2.read_slice(0, num_blocks * 2)?;

    let reduce_ext2 = |v: &[u64]| -> GoldilocksExt2 {
        let mut c0 = 0u64;
        let mut c1 = 0u64;
        for i in 0..num_blocks {
            c0 = gl_add_host(c0, v[i * 2]);
            c1 = gl_add_host(c1, v[i * 2 + 1]);
        }
        GoldilocksExt2::new(GoldilocksField(c0), GoldilocksField(c1))
    };

    Ok(SumcheckOracle {
        c0: reduce_ext2(&v0),
        c1: reduce_ext2(&v1),
        c2: reduce_ext2(&v2),
    })
}

/// Like `sumcheck_product_and_reduce_ext2` but reuses pre-allocated partial buffers.
pub fn sumcheck_product_and_reduce_ext2_reuse(
    eq: &DeviceBuffer<u64>,
    bh: &DeviceBuffer<u64>,
    pair_count: usize,
    pc0: &mut DeviceBuffer<u64>,
    pc1: &mut DeviceBuffer<u64>,
    pc2: &mut DeviceBuffer<u64>,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let num_blocks = ((pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    BasefoldBatch::sumcheck_product_ext2(eq, bh, pc0, pc1, pc2, pair_count, num_blocks)?;

    let v0 = pc0.read_slice(0, num_blocks * 2)?;
    let v1 = pc1.read_slice(0, num_blocks * 2)?;
    let v2 = pc2.read_slice(0, num_blocks * 2)?;

    let reduce_ext2 = |v: &[u64]| -> GoldilocksExt2 {
        let mut c0 = 0u64;
        let mut c1 = 0u64;
        for i in 0..num_blocks {
            c0 = gl_add_host(c0, v[i * 2]);
            c1 = gl_add_host(c1, v[i * 2 + 1]);
        }
        GoldilocksExt2::new(GoldilocksField(c0), GoldilocksField(c1))
    };

    Ok(SumcheckOracle {
        c0: reduce_ext2(&v0),
        c1: reduce_ext2(&v1),
        c2: reduce_ext2(&v2),
    })
}

/// Fused eval+interp+product for one ext2 sumcheck round.
/// Replaces 5 separate kernel launches (2×eval + 2×interp + product) with a single kernel.
/// `pair_count` = number of product pairs = (input element count) / 4.
/// Reads 4*pair_count Ext2 from eq_in/bh_in, writes 2*pair_count to eq_out/bh_out.
pub fn fused_sumcheck_round_ext2_reuse(
    eq_in: &DeviceBuffer<u64>,
    bh_in: &DeviceBuffer<u64>,
    challenge: GoldilocksExt2,
    eq_out: &mut DeviceBuffer<u64>,
    bh_out: &mut DeviceBuffer<u64>,
    pair_count: usize,
    pc0: &mut DeviceBuffer<u64>,
    pc1: &mut DeviceBuffer<u64>,
    pc2: &mut DeviceBuffer<u64>,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let num_blocks = ((pair_count + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);

    let ret = unsafe {
        ffi::fused_sumcheck_round_ext2_ffi(
            eq_in.as_ptr(),
            bh_in.as_ptr(),
            challenge.c0.0,
            challenge.c1.0,
            eq_out.as_mut_ptr(),
            bh_out.as_mut_ptr(),
            pc0.as_mut_ptr(),
            pc1.as_mut_ptr(),
            pc2.as_mut_ptr(),
            pair_count as c_int,
            num_blocks as c_int,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    let v0 = pc0.read_slice(0, num_blocks * 2)?;
    let v1 = pc1.read_slice(0, num_blocks * 2)?;
    let v2 = pc2.read_slice(0, num_blocks * 2)?;

    let reduce_ext2 = |v: &[u64]| -> GoldilocksExt2 {
        let mut c0 = 0u64;
        let mut c1 = 0u64;
        for i in 0..num_blocks {
            c0 = gl_add_host(c0, v[i * 2]);
            c1 = gl_add_host(c1, v[i * 2 + 1]);
        }
        GoldilocksExt2::new(GoldilocksField(c0), GoldilocksField(c1))
    };

    Ok(SumcheckOracle {
        c0: reduce_ext2(&v0),
        c1: reduce_ext2(&v1),
        c2: reduce_ext2(&v2),
    })
}

// ============================================================================
// Batch-open / batch-verify helpers
// ============================================================================

/// GPU: acc += scalar * src.  Uses `tmp` as scratch (same size as acc/src).
/// `n` is the number of base-field elements.
fn gpu_accumulate_scaled(
    acc: &mut DeviceBuffer<u64>,
    src: &DeviceBuffer<u64>,
    tmp: &mut DeviceBuffer<u64>,
    scalar: GoldilocksField,
    n: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::gl_batch_mul_scalar(scalar.0, src.as_ptr(), tmp.as_mut_ptr(), n as c_int)
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    // acc += tmp  (in-place on acc is safe for gl_batch_add)
    let ret = unsafe {
        ffi::gl_batch_add(acc.as_ptr(), tmp.as_ptr(), acc.as_mut_ptr(), n as c_int)
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

/// Create a zero-filled DeviceBuffer of `n` u64 elements.
fn gpu_zero_buffer(n: usize) -> Result<DeviceBuffer<u64>> {
    let zeros = vec![0u64; n];
    DeviceBuffer::from_slice(&zeros)
}

/// CPU: eq(r, point) = prod_i(r_i * point_i + (1 - r_i) * (1 - point_i))
fn eq_eval_host(r: &[GoldilocksField], point: &[GoldilocksField]) -> GoldilocksField {
    assert_eq!(r.len(), point.len());
    let mut result = 1u64;
    for i in 0..r.len() {
        let ri = r[i].0;
        let pi = point[i].0;
        // r_i * p_i + (1 - r_i) * (1 - p_i)
        let rp = gl_mul_host(ri, pi);
        let one_minus_r = gl_sub_host(1, ri);
        let one_minus_p = gl_sub_host(1, pi);
        let term = gl_add_host(rp, gl_mul_host(one_minus_r, one_minus_p));
        result = gl_mul_host(result, term);
    }
    GoldilocksField(result)
}

/// CPU: compute all 2^ell evaluations of eq(x, t) for t ∈ {0,1}^ell mapped to point coords.
/// Returns vec of length 2^ell where entry i = eq(point, binary(i)).
///
/// Used to generate random linear-combination weights from a Fiat-Shamir random point.
fn eq_poly_host(point: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let ell = point.len();
    let size = 1usize << ell;
    let mut result = vec![GoldilocksField(0); size];
    result[0] = GoldilocksField(1);
    for i in 0..ell {
        let half = 1usize << i;
        let ri = point[i].0;
        let one_minus_ri = gl_sub_host(1, ri);
        // process in reverse to avoid overwriting
        for j in (0..half).rev() {
            let val = result[j].0;
            result[2 * j] = GoldilocksField(gl_mul_host(val, one_minus_ri));
            result[2 * j + 1] = GoldilocksField(gl_mul_host(val, ri));
        }
    }
    result
}

/// Sum `sumcheck_product_and_reduce_gl` across multiple (eq, bh) pairs.
/// Returns one oracle = sum of individual oracles.
fn multi_point_sumcheck_product_gl(
    eq_bufs: &[DeviceBuffer<u64>],
    bh_bufs: &[DeviceBuffer<u64>],
    pair_count: usize,
) -> Result<SumcheckOracle<GoldilocksField>> {
    let mut total_c0 = 0u64;
    let mut total_c1 = 0u64;
    let mut total_c2 = 0u64;
    for (eq, bh) in eq_bufs.iter().zip(bh_bufs.iter()) {
        let oracle = sumcheck_product_and_reduce_gl(eq, bh, pair_count)?;
        total_c0 = gl_add_host(total_c0, oracle.c0.0);
        total_c1 = gl_add_host(total_c1, oracle.c1.0);
        total_c2 = gl_add_host(total_c2, oracle.c2.0);
    }
    Ok(SumcheckOracle {
        c0: GoldilocksField(total_c0),
        c1: GoldilocksField(total_c1),
        c2: GoldilocksField(total_c2),
    })
}

/// Sum `sumcheck_product_and_reduce_mixed` across multiple (eq_ext2, bh_gl) pairs.
fn multi_point_sumcheck_product_mixed(
    eq_bufs: &[DeviceBuffer<u64>],
    bh_bufs: &[DeviceBuffer<u64>],
    pair_count: usize,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let zero = GoldilocksExt2::new(GoldilocksField(0), GoldilocksField(0));
    let mut total = SumcheckOracle { c0: zero, c1: zero, c2: zero };
    for (eq, bh) in eq_bufs.iter().zip(bh_bufs.iter()) {
        let oracle = sumcheck_product_and_reduce_mixed(eq, bh, pair_count)?;
        total.c0 = ext2_add_host(total.c0, oracle.c0);
        total.c1 = ext2_add_host(total.c1, oracle.c1);
        total.c2 = ext2_add_host(total.c2, oracle.c2);
    }
    Ok(total)
}

/// Sum `sumcheck_product_and_reduce_ext2` across multiple (eq, bh) pairs.
fn multi_point_sumcheck_product_ext2(
    eq_bufs: &[DeviceBuffer<u64>],
    bh_bufs: &[DeviceBuffer<u64>],
    pair_count: usize,
) -> Result<SumcheckOracle<GoldilocksExt2>> {
    let zero = GoldilocksExt2::new(GoldilocksField(0), GoldilocksField(0));
    let mut total = SumcheckOracle { c0: zero, c1: zero, c2: zero };
    for (eq, bh) in eq_bufs.iter().zip(bh_bufs.iter()) {
        let oracle = sumcheck_product_and_reduce_ext2(eq, bh, pair_count)?;
        total.c0 = ext2_add_host(total.c0, oracle.c0);
        total.c1 = ext2_add_host(total.c1, oracle.c1);
        total.c2 = ext2_add_host(total.c2, oracle.c2);
    }
    Ok(total)
}

/// GPU: (acc_c0, acc_c1) += ext2_scalar * gl_src.
/// Decomposes ext2 multiplication into two base-field scalar-multiply + accumulate steps.
fn gpu_accumulate_scaled_ext2_from_gl(
    acc_c0: &mut DeviceBuffer<u64>,
    acc_c1: &mut DeviceBuffer<u64>,
    src: &DeviceBuffer<u64>,
    tmp: &mut DeviceBuffer<u64>,
    scalar: GoldilocksExt2,
    n: usize,
) -> Result<()> {
    // acc_c0 += scalar.c0 * src
    let ret = unsafe {
        ffi::gl_batch_mul_scalar(scalar.c0.0, src.as_ptr(), tmp.as_mut_ptr(), n as c_int)
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    let ret = unsafe {
        ffi::gl_batch_add(acc_c0.as_ptr(), tmp.as_ptr(), acc_c0.as_mut_ptr(), n as c_int)
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    // acc_c1 += scalar.c1 * src
    let ret = unsafe {
        ffi::gl_batch_mul_scalar(scalar.c1.0, src.as_ptr(), tmp.as_mut_ptr(), n as c_int)
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    let ret = unsafe {
        ffi::gl_batch_add(acc_c1.as_ptr(), tmp.as_ptr(), acc_c1.as_mut_ptr(), n as c_int)
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

/// Merge two base-field device buffers (c0, c1) into ext2 interleaved layout.
/// Downloads, interleaves on CPU, re-uploads.
fn interleave_to_ext2(
    d_c0: &DeviceBuffer<u64>,
    d_c1: &DeviceBuffer<u64>,
    n: usize,
) -> Result<DeviceBuffer<u64>> {
    let c0_vals = d_c0.read_slice(0, n)?;
    let c1_vals = d_c1.read_slice(0, n)?;
    let mut interleaved = Vec::with_capacity(n * 2);
    for i in 0..n {
        interleaved.push(c0_vals[i]);
        interleaved.push(c1_vals[i]);
    }
    DeviceBuffer::from_slice(&interleaved)
}

// ============================================================================
// Host-side field arithmetic
// ============================================================================

pub(crate) fn gl_add_host(a: u64, b: u64) -> u64 {
    // Normalize inputs to [0, p) to handle non-canonical GPU representations
    let a = a % GOLDILOCKS_PRIME;
    let b = b % GOLDILOCKS_PRIME;
    let sum = a.wrapping_add(b);
    if sum < a || sum >= GOLDILOCKS_PRIME {
        sum.wrapping_sub(GOLDILOCKS_PRIME)
    } else {
        sum
    }
}

fn gl_sub_host(a: u64, b: u64) -> u64 {
    // Normalize inputs to [0, p) to handle non-canonical GPU representations
    let a = a % GOLDILOCKS_PRIME;
    let b = b % GOLDILOCKS_PRIME;
    if a >= b {
        a - b
    } else {
        a.wrapping_add(GOLDILOCKS_PRIME).wrapping_sub(b)
    }
}

pub(crate) fn gl_mul_host(a: u64, b: u64) -> u64 {
    let full = (a as u128) * (b as u128);
    (full % GOLDILOCKS_PRIME as u128) as u64
}

fn gl_inv_host(a: u64) -> u64 {
    let mut result: u64 = 1;
    let mut base = a;
    let mut exp = GOLDILOCKS_PRIME - 2;
    while exp > 0 {
        if exp & 1 == 1 {
            result = gl_mul_host(result, base);
        }
        base = gl_mul_host(base, base);
        exp >>= 1;
    }
    result
}

// ============================================================================
// Extension-field host arithmetic
// ============================================================================

fn ext2_add_host(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    GoldilocksExt2::new(
        GoldilocksField(gl_add_host(a.c0.0, b.c0.0)),
        GoldilocksField(gl_add_host(a.c1.0, b.c1.0)),
    )
}

fn ext2_sub_host(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    GoldilocksExt2::new(
        GoldilocksField(gl_sub_host(a.c0.0, b.c0.0)),
        GoldilocksField(gl_sub_host(a.c1.0, b.c1.0)),
    )
}

fn ext2_mul_host(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    const W: u64 = 7; // EXT2_W for X^2 - 7
    let c0 = gl_add_host(
        gl_mul_host(a.c0.0, b.c0.0),
        gl_mul_host(gl_mul_host(a.c1.0, b.c1.0), W),
    );
    let c1 = gl_add_host(
        gl_mul_host(a.c0.0, b.c1.0),
        gl_mul_host(a.c1.0, b.c0.0),
    );
    GoldilocksExt2::new(GoldilocksField(c0), GoldilocksField(c1))
}

/// CPU: eq(r, point) for ext2 vectors.
fn eq_eval_host_ext2(r: &[GoldilocksExt2], point: &[GoldilocksExt2]) -> GoldilocksExt2 {
    assert_eq!(r.len(), point.len());
    let one = GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(0));
    let mut result = one;
    for i in 0..r.len() {
        let rp = ext2_mul_host(r[i], point[i]);
        let one_minus_r = ext2_sub_host(one, r[i]);
        let one_minus_p = ext2_sub_host(one, point[i]);
        let term = ext2_add_host(rp, ext2_mul_host(one_minus_r, one_minus_p));
        result = ext2_mul_host(result, term);
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::synchronize;

    #[test]
    fn test_basefold_bit_reverse() {
        crate::init().unwrap();
        let n = 16usize;
        let log_n = 4;
        let data: Vec<u64> = (0..n as u64).collect();
        let mut d_data = DeviceBuffer::from_slice(&data).unwrap();
        BasefoldBatch::bit_reverse_gl(&mut d_data, log_n).unwrap();
        synchronize().unwrap();
        let result = d_data.to_vec().unwrap();
        for i in 0..n {
            let mut rev = 0usize;
            for b in 0..log_n {
                rev = (rev << 1) | ((i >> b) & 1);
            }
            assert_eq!(result[i], rev as u64, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_basefold_bhc_interpolation() {
        crate::init().unwrap();
        let num_vars = 3;
        let n = 1 << num_vars;
        let evals: Vec<u64> = (0..n)
            .map(|i: usize| {
                let mut val = 1u64;
                if i & 1 != 0 { val += 1; }
                if i & 2 != 0 { val += 2; }
                if i & 4 != 0 { val += 3; }
                val
            })
            .collect();
        let d_evals = DeviceBuffer::from_slice(&evals).unwrap();
        let mut d_coeffs = DeviceBuffer::<u64>::new(n).unwrap();
        let mut d_bh_evals = DeviceBuffer::<u64>::new(n).unwrap();
        BasefoldBatch::bhc_interpolate(&d_evals, &mut d_coeffs, &mut d_bh_evals, num_vars).unwrap();
        synchronize().unwrap();
        let coeffs = d_coeffs.to_vec().unwrap();
        for x in 0..n {
            let mut eval = 0u64;
            for s in 0..n {
                if (x & s) == s {
                    eval = gl_add_host(eval, coeffs[s]);
                }
            }
            let expected = {
                let mut val = 1u64;
                if x & 1 != 0 { val += 1; }
                if x & 2 != 0 { val += 2; }
                if x & 4 != 0 { val += 3; }
                val
            };
            assert_eq!(eval % GOLDILOCKS_PRIME, expected, "Re-eval mismatch at x={}", x);
        }
    }

    #[test]
    fn test_basefold_encode() {
        crate::init().unwrap();
        let num_vars = 3;
        let log_rate = 1;
        let n = 1 << num_vars;
        let cw_len = 1 << (num_vars + log_rate);
        let coeffs: Vec<u64> = (1..=n as u64).collect();
        let d_coeffs = DeviceBuffer::from_slice(&coeffs).unwrap();
        let mut d_codeword = DeviceBuffer::<u64>::new(cw_len).unwrap();
        BasefoldBatch::encode(&d_coeffs, &mut d_codeword, num_vars, log_rate).unwrap();
        synchronize().unwrap();
        let cw = d_codeword.to_vec().unwrap();
        assert_eq!(cw.len(), cw_len);
        assert!(cw.iter().any(|&v| v != 0), "Codeword should be non-trivial");
    }

    #[test]
    fn test_basefold_table_generation() {
        let mut table = BasefoldTable::generate(4, 1, 4, 12345);
        assert_eq!(table.num_rounds, 4);
        assert!(!table.entries.is_empty());
        for entry in &table.entries {
            assert_ne!(entry.weight.0, 0, "Weight should be non-zero");
        }
        crate::init().unwrap();
        table.upload().unwrap();
    }

    #[test]
    fn test_basefold_commit() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        assert!(comm.root.elements.iter().any(|e| e.0 != 0), "Root should be non-zero");
        assert_eq!(comm.num_vars, num_vars);
        assert_eq!(comm.log_rate, log_rate);
    }

    #[test]
    fn test_basefold_commit_open_verify() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME))
            .collect();

        let mut prover_transcript = TestTranscript::new(999);
        let proof = comm.open(&point, &table, &mut prover_transcript, 4).unwrap();
        synchronize().unwrap();

        // Verify eval by CPU multilinear evaluation
        let cpu_eval = cpu_multilinear_eval(&evals, &point);
        assert_eq!(
            proof.eval.0, cpu_eval,
            "GPU eval {} != CPU eval {}",
            proof.eval.0, cpu_eval
        );

        // Verify sum-check
        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::verify(
            &comm.root,
            &point,
            &proof,
            &table,
            &mut verifier_transcript,
        ).unwrap();
        assert!(valid, "Verification should succeed");
    }

    #[test]
    fn test_basefold_ext2_open() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let mut transcript = TestTranscript::new(999);
        let proof = comm.open_ext2(&point, &table, &mut transcript, 4).unwrap();
        synchronize().unwrap();

        // Verify eval by CPU multilinear evaluation over ext2
        let cpu_eval = cpu_multilinear_eval_ext2(&evals, &point);
        assert_eq!(
            proof.eval.c0.0, cpu_eval.c0.0,
            "GPU eval c0 {} != CPU eval c0 {}",
            proof.eval.c0.0, cpu_eval.c0.0
        );
        assert_eq!(
            proof.eval.c1.0, cpu_eval.c1.0,
            "GPU eval c1 {} != CPU eval c1 {}",
            proof.eval.c1.0, cpu_eval.c1.0
        );
    }

    // CPU reference: multilinear evaluation
    fn cpu_multilinear_eval(evals: &[GoldilocksField], point: &[GoldilocksField]) -> u64 {
        let n = evals.len();
        let num_vars = point.len();
        let mut sum = 0u64;
        for x in 0..n {
            let mut eq_val = 1u64;
            for i in 0..num_vars {
                let xi = ((x >> i) & 1) as u64;
                if xi == 1 {
                    eq_val = gl_mul_host(eq_val, point[i].0);
                } else {
                    eq_val = gl_mul_host(eq_val, gl_sub_host(1, point[i].0));
                }
            }
            sum = gl_add_host(sum, gl_mul_host(evals[x].0, eq_val));
        }
        sum
    }

    fn cpu_multilinear_eval_ext2(
        evals: &[GoldilocksField],
        point: &[GoldilocksExt2],
    ) -> GoldilocksExt2 {
        let n = evals.len();
        let num_vars = point.len();
        let mut sum = GoldilocksExt2::new(GoldilocksField(0), GoldilocksField(0));
        for x in 0..n {
            let mut eq_val = GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(0));
            for i in 0..num_vars {
                let xi = ((x >> i) & 1) as u64;
                if xi == 1 {
                    eq_val = ext2_mul_host(eq_val, point[i]);
                } else {
                    let one = GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(0));
                    eq_val = ext2_mul_host(eq_val, ext2_sub_host(one, point[i]));
                }
            }
            // evals[x] is base field, promote to ext2
            let ev_ext2 = GoldilocksExt2::new(evals[x], GoldilocksField(0));
            sum = ext2_add_host(sum, ext2_mul_host(ev_ext2, eq_val));
        }
        sum
    }

    // ================================================================
    // Batch open / verify tests
    // ================================================================

    #[test]
    fn test_batch_open_single_poly() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME))
            .collect();

        let cpu_eval = cpu_multilinear_eval(&evals, &point);

        let eval_claim = Evaluation::new(0, 0, GoldilocksField(cpu_eval));
        let points_refs: Vec<&[GoldilocksField]> = vec![&point];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm];

        let mut prover_transcript = TestTranscript::new(999);
        let proof = batch_open(
            &comms_refs,
            &points_refs,
            &[eval_claim],
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::batch_verify(
            &[comm.root.clone()],
            &points_refs,
            &[Evaluation::new(0, 0, GoldilocksField(cpu_eval))],
            &proof,
            &table,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(valid, "Batch verify (single poly) should succeed");
    }

    #[test]
    fn test_batch_open_two_polys_same_point() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals_a: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let evals_b: Vec<GoldilocksField> = (100..100 + n as u64).map(GoldilocksField).collect();
        let comm_a = BasefoldCommitment::commit(&evals_a, num_vars, log_rate).unwrap();
        let comm_b = BasefoldCommitment::commit(&evals_b, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME))
            .collect();

        let eval_a = cpu_multilinear_eval(&evals_a, &point);
        let eval_b = cpu_multilinear_eval(&evals_b, &point);

        let claims = vec![
            Evaluation::new(0, 0, GoldilocksField(eval_a)),
            Evaluation::new(1, 0, GoldilocksField(eval_b)),
        ];
        let points_refs: Vec<&[GoldilocksField]> = vec![&point];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm_a, &comm_b];

        let mut prover_transcript = TestTranscript::new(888);
        let proof = batch_open(
            &comms_refs,
            &points_refs,
            &claims,
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        let mut verifier_transcript = TestTranscript::new(888);
        let valid = BasefoldVerifier::batch_verify(
            &[comm_a.root.clone(), comm_b.root.clone()],
            &points_refs,
            &claims,
            &proof,
            &table,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(valid, "Batch verify (two polys, same point) should succeed");
    }

    #[test]
    fn test_batch_open_two_polys_different_points() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals_a: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let evals_b: Vec<GoldilocksField> = (100..100 + n as u64).map(GoldilocksField).collect();
        let comm_a = BasefoldCommitment::commit(&evals_a, num_vars, log_rate).unwrap();
        let comm_b = BasefoldCommitment::commit(&evals_b, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point_a: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME))
            .collect();
        let point_b: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 333) % GOLDILOCKS_PRIME))
            .collect();

        let eval_a = cpu_multilinear_eval(&evals_a, &point_a);
        let eval_b = cpu_multilinear_eval(&evals_b, &point_b);

        let claims = vec![
            Evaluation::new(0, 0, GoldilocksField(eval_a)),
            Evaluation::new(1, 1, GoldilocksField(eval_b)),
        ];
        let points_refs: Vec<&[GoldilocksField]> = vec![&point_a, &point_b];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm_a, &comm_b];

        let mut prover_transcript = TestTranscript::new(777);
        let proof = batch_open(
            &comms_refs,
            &points_refs,
            &claims,
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        let mut verifier_transcript = TestTranscript::new(777);
        let valid = BasefoldVerifier::batch_verify(
            &[comm_a.root.clone(), comm_b.root.clone()],
            &points_refs,
            &claims,
            &proof,
            &table,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(
            valid,
            "Batch verify (two polys, different points) should succeed"
        );
    }

    #[test]
    fn test_batch_verify_reject_bad_eval() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksField> = (0..num_vars)
            .map(|i| GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME))
            .collect();

        let cpu_eval = cpu_multilinear_eval(&evals, &point);

        // Correct proof
        let claims = vec![Evaluation::new(0, 0, GoldilocksField(cpu_eval))];
        let points_refs: Vec<&[GoldilocksField]> = vec![&point];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm];

        let mut prover_transcript = TestTranscript::new(999);
        let proof = batch_open(
            &comms_refs,
            &points_refs,
            &claims,
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        // Tamper: wrong eval value
        let bad_claims = vec![Evaluation::new(0, 0, GoldilocksField(cpu_eval + 1))];
        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::batch_verify(
            &[comm.root.clone()],
            &points_refs,
            &bad_claims,
            &proof,
            &table,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(!valid, "Batch verify should reject tampered eval");
    }

    // ================================================================
    // Ext2 batch open / verify tests
    // ================================================================

    #[test]
    fn test_verify_ext2() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        // === TRACE Step 1: Download codeword, rebuild tree on CPU, compare roots ===
        {
            let cw = comm.d_codeword.to_vec().unwrap();
            let cw_len = cw.len();
            let num_leaves = cw_len / 2;
            eprintln!("[TRACE] codeword len={}, num_leaves={}", cw_len, num_leaves);
            eprintln!("[TRACE] GPU root: {:?}", comm.root.to_raw());

            // CPU: hash all leaves
            let cpu_leaves: Vec<Poseidon2Hash> = (0..num_leaves)
                .map(|i| hash_gl_leaf(GoldilocksField(cw[2 * i]), GoldilocksField(cw[2 * i + 1])))
                .collect();

            // GPU: read all leaf digests
            let mut leaf_mismatches = 0;
            for i in 0..num_leaves {
                let gpu_leaf = comm.merkle_tree.leaf_digest(i).unwrap();
                if gpu_leaf != cpu_leaves[i] {
                    eprintln!("[TRACE] leaf {} MISMATCH: cw[{}]={} cw[{}]={}",
                        i, 2*i, cw[2*i], 2*i+1, cw[2*i+1]);
                    eprintln!("[TRACE]   GPU={:?}", gpu_leaf.to_raw());
                    eprintln!("[TRACE]   CPU={:?}", cpu_leaves[i].to_raw());
                    leaf_mismatches += 1;
                }
            }
            eprintln!("[TRACE] leaf mismatches: {}/{}", leaf_mismatches, num_leaves);

            // CPU: build tree bottom-up
            let mut layer = cpu_leaves.clone();
            let mut cpu_layers: Vec<Vec<Poseidon2Hash>> = vec![layer.clone()];
            while layer.len() > 1 {
                let next: Vec<Poseidon2Hash> = (0..layer.len() / 2)
                    .map(|i| {
                        #[cfg(feature = "monolith")]
                        { crate::cpu_monolith::monolith_compress(&layer[2*i], &layer[2*i+1]) }
                        #[cfg(not(feature = "monolith"))]
                        { crate::cpu_poseidon2::poseidon2_compress(&layer[2*i], &layer[2*i+1]) }
                    })
                    .collect();
                layer = next;
                cpu_layers.push(layer.clone());
            }
            let cpu_root = layer[0];
            eprintln!("[TRACE] CPU root: {:?} match_GPU={}", cpu_root.to_raw(), cpu_root == comm.root);

            // Trace auth path for leaf 0
            let leaf_idx = 0usize;
            let leaf_hash = cpu_leaves[leaf_idx];
            let gpu_path = comm.merkle_tree.auth_path(leaf_idx).unwrap();
            eprintln!("[TRACE] auth path for leaf {}: {} siblings", leaf_idx, gpu_path.len());
            let mut current = leaf_hash;
            let mut idx = leaf_idx;
            for (level, sibling) in gpu_path.iter().enumerate() {
                // Expected sibling from CPU tree
                let sibling_idx = idx ^ 1;
                let cpu_sibling = &cpu_layers[level][sibling_idx];
                eprintln!("[TRACE]   level {}: idx={} sibling_idx={} GPU_sibling={:?} CPU_sibling={:?} match={}",
                    level, idx, sibling_idx, sibling.to_raw(), cpu_sibling.to_raw(), sibling == cpu_sibling);

                if idx & 1 == 0 {
                    #[cfg(feature = "monolith")]
                    { current = crate::cpu_monolith::monolith_compress(&current, sibling); }
                    #[cfg(not(feature = "monolith"))]
                    { current = crate::cpu_poseidon2::poseidon2_compress(&current, sibling); }
                } else {
                    #[cfg(feature = "monolith")]
                    { current = crate::cpu_monolith::monolith_compress(sibling, &current); }
                    #[cfg(not(feature = "monolith"))]
                    { current = crate::cpu_poseidon2::poseidon2_compress(sibling, &current); }
                }

                // Expected parent from CPU tree
                let parent_idx = idx / 2;
                let cpu_parent = &cpu_layers[level + 1][parent_idx];
                eprintln!("[TRACE]   -> parent at [{}][{}]: computed={:?} CPU_expected={:?} match={}",
                    level + 1, parent_idx, current.to_raw(), cpu_parent.to_raw(), current == *cpu_parent);

                idx /= 2;
            }
            eprintln!("[TRACE] final hash: {:?} root={:?} match={}", current.to_raw(), comm.root.to_raw(), current == comm.root);
        }

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let mut prover_transcript = TestTranscript::new(999);
        let proof = comm.open_ext2(&point, &table, &mut prover_transcript, 4).unwrap();
        synchronize().unwrap();

        // Verify eval by CPU multilinear evaluation over ext2
        let cpu_eval = cpu_multilinear_eval_ext2(&evals, &point);
        assert_eq!(proof.eval.c0.0, cpu_eval.c0.0);
        assert_eq!(proof.eval.c1.0, cpu_eval.c1.0);

        // Verify sum-check
        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::verify_ext2(
            &comm.root,
            &point,
            &proof,
            &table,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(valid, "verify_ext2 should succeed");
    }

    #[test]
    fn test_verify_ext2_reject_tampered_query() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let mut prover_transcript = TestTranscript::new(999);
        let mut proof = comm.open_ext2(&point, &table, &mut prover_transcript, 4).unwrap();
        synchronize().unwrap();

        // Tamper with a query proof value
        if !proof.query_proofs.is_empty() {
            proof.query_proofs[0].values[0].0.c0 = GoldilocksField(12345);
        }

        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::verify_ext2(
            &comm.root,
            &point,
            &proof,
            &table,
            &mut verifier_transcript,
        )
        .unwrap();
        assert!(!valid, "verify_ext2 should reject tampered query proof");
    }

    #[test]
    fn test_verify_ext2_small_poly_high_rate() {
        // Test with num_vars=2, log_rate=3 — matches DAG test conditions
        crate::init().unwrap();
        let num_vars = 2;
        let log_rate = 3;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = vec![
            GoldilocksField(1), GoldilocksField(2),
            GoldilocksField(3), GoldilocksField(4),
        ];
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| GoldilocksExt2::new(
                GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
            ))
            .collect();

        let mut prover_transcript = TestTranscript::new(999);
        let proof = comm.open_ext2(&point, &table, &mut prover_transcript, 10).unwrap();
        synchronize().unwrap();

        let cpu_eval = cpu_multilinear_eval_ext2(&evals, &point);
        assert_eq!(proof.eval.c0.0, cpu_eval.c0.0);
        assert_eq!(proof.eval.c1.0, cpu_eval.c1.0);

        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::verify_ext2(
            &comm.root, &point, &proof, &table, &mut verifier_transcript,
        ).unwrap();
        assert!(valid, "verify_ext2 should succeed for small poly with high rate");
    }

    #[test]
    fn test_batch_open_ext2_single_poly() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let cpu_eval = cpu_multilinear_eval_ext2(&evals, &point);
        let eval_claim = EvaluationExt2::new(0, 0, cpu_eval);
        let points_refs: Vec<&[GoldilocksExt2]> = vec![&point];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm];

        let mut prover_transcript = TestTranscript::new(999);
        let proof = batch_open_ext2(
            &comms_refs,
            &points_refs,
            &[eval_claim],
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::batch_verify_ext2(
            &[comm.root.clone()],
            &points_refs,
            &[EvaluationExt2::new(0, 0, cpu_eval)],
            &proof,
            &table,
            &mut verifier_transcript,
            log_rate,
        )
        .unwrap();
        assert!(valid, "Batch verify ext2 (single poly) should succeed");
    }

    #[test]
    fn test_batch_open_ext2_two_polys_same_point() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals_a: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let evals_b: Vec<GoldilocksField> = (100..100 + n as u64).map(GoldilocksField).collect();
        let comm_a = BasefoldCommitment::commit(&evals_a, num_vars, log_rate).unwrap();
        let comm_b = BasefoldCommitment::commit(&evals_b, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let eval_a = cpu_multilinear_eval_ext2(&evals_a, &point);
        let eval_b = cpu_multilinear_eval_ext2(&evals_b, &point);

        let claims = vec![
            EvaluationExt2::new(0, 0, eval_a),
            EvaluationExt2::new(1, 0, eval_b),
        ];
        let points_refs: Vec<&[GoldilocksExt2]> = vec![&point];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm_a, &comm_b];

        let mut prover_transcript = TestTranscript::new(888);
        let proof = batch_open_ext2(
            &comms_refs,
            &points_refs,
            &claims,
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        let mut verifier_transcript = TestTranscript::new(888);
        let valid = BasefoldVerifier::batch_verify_ext2(
            &[comm_a.root.clone(), comm_b.root.clone()],
            &points_refs,
            &claims,
            &proof,
            &table,
            &mut verifier_transcript,
            log_rate,
        )
        .unwrap();
        assert!(
            valid,
            "Batch verify ext2 (two polys, same point) should succeed"
        );
    }

    #[test]
    fn test_batch_open_ext2_two_polys_different_points() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals_a: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let evals_b: Vec<GoldilocksField> = (100..100 + n as u64).map(GoldilocksField).collect();
        let comm_a = BasefoldCommitment::commit(&evals_a, num_vars, log_rate).unwrap();
        let comm_b = BasefoldCommitment::commit(&evals_b, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point_a: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();
        let point_b: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 333) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 444) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let eval_a = cpu_multilinear_eval_ext2(&evals_a, &point_a);
        let eval_b = cpu_multilinear_eval_ext2(&evals_b, &point_b);

        let claims = vec![
            EvaluationExt2::new(0, 0, eval_a),
            EvaluationExt2::new(1, 1, eval_b),
        ];
        let points_refs: Vec<&[GoldilocksExt2]> = vec![&point_a, &point_b];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm_a, &comm_b];

        let mut prover_transcript = TestTranscript::new(777);
        let proof = batch_open_ext2(
            &comms_refs,
            &points_refs,
            &claims,
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        let mut verifier_transcript = TestTranscript::new(777);
        let valid = BasefoldVerifier::batch_verify_ext2(
            &[comm_a.root.clone(), comm_b.root.clone()],
            &points_refs,
            &claims,
            &proof,
            &table,
            &mut verifier_transcript,
            log_rate,
        )
        .unwrap();
        assert!(
            valid,
            "Batch verify ext2 (two polys, different points) should succeed"
        );
    }

    #[test]
    fn test_batch_verify_ext2_reject_bad_eval() {
        crate::init().unwrap();
        let num_vars = 4;
        let log_rate = 1;
        let n = 1usize << num_vars;

        let evals: Vec<GoldilocksField> = (1..=n as u64).map(GoldilocksField).collect();
        let comm = BasefoldCommitment::commit(&evals, num_vars, log_rate).unwrap();
        synchronize().unwrap();

        let mut table = BasefoldTable::generate(num_vars, log_rate, num_vars, 42);
        table.upload().unwrap();

        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| {
                GoldilocksExt2::new(
                    GoldilocksField(((i as u64 + 1) * 111) % GOLDILOCKS_PRIME),
                    GoldilocksField(((i as u64 + 1) * 222) % GOLDILOCKS_PRIME),
                )
            })
            .collect();

        let cpu_eval = cpu_multilinear_eval_ext2(&evals, &point);

        // Correct proof
        let claims = vec![EvaluationExt2::new(0, 0, cpu_eval)];
        let points_refs: Vec<&[GoldilocksExt2]> = vec![&point];
        let comms_refs: Vec<&BasefoldCommitment> = vec![&comm];

        let mut prover_transcript = TestTranscript::new(999);
        let proof = batch_open_ext2(
            &comms_refs,
            &points_refs,
            &claims,
            &table,
            &mut prover_transcript,
            4,
        )
        .unwrap();
        synchronize().unwrap();

        // Tamper: wrong eval value
        let bad_eval = GoldilocksExt2::new(
            GoldilocksField(cpu_eval.c0.0 + 1),
            cpu_eval.c1,
        );
        let bad_claims = vec![EvaluationExt2::new(0, 0, bad_eval)];
        let mut verifier_transcript = TestTranscript::new(999);
        let valid = BasefoldVerifier::batch_verify_ext2(
            &[comm.root.clone()],
            &points_refs,
            &bad_claims,
            &proof,
            &table,
            &mut verifier_transcript,
            log_rate,
        )
        .unwrap();
        assert!(!valid, "Batch verify ext2 should reject tampered eval");
    }
}
