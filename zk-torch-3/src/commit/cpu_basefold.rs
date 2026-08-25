use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use goldilocks_cuda::basefold::{
    BasefoldCommitment, BasefoldProofExt2, BasefoldTable, FoldingEntry,
    QueryProof, SumcheckOracle,
};
#[cfg(feature = "monolith")]
use goldilocks_cuda::cpu_monolith::{hash_gl_leaf, hash_ext2_leaf, monolith_compress as hash_compress};
#[cfg(not(feature = "monolith"))]
use goldilocks_cuda::cpu_poseidon2::{hash_gl_leaf, hash_ext2_leaf, poseidon2_compress as hash_compress};
use goldilocks_cuda::poseidon2::Poseidon2Hash;
use crate::commit::basefold::BasefoldOpeningProof;
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_sub, ext2_mul};

use goldilocks_cuda::basefold::BasefoldTranscript;

// ============================================================================
// CPU Merkle tree
// ============================================================================

struct CpuMerkleTree {
    layers: Vec<Vec<Poseidon2Hash>>,
}

impl CpuMerkleTree {
    /// Build Merkle tree from base-field (GL) codeword pairs.
    /// codeword has 2N u64 elements → N leaf hashes.
    fn build_from_gl_pairs(codeword: &[u64]) -> Self {
        let num_leaves = codeword.len() / 2;
        let leaf_hashes: Vec<Poseidon2Hash> = (0..num_leaves)
            .map(|i| {
                hash_gl_leaf(
                    GoldilocksField(codeword[2 * i]),
                    GoldilocksField(codeword[2 * i + 1]),
                )
            })
            .collect();
        Self::build_from_leaves(leaf_hashes)
    }

    /// Build Merkle tree from extension-field (Ext2) codeword pairs.
    /// codeword has 2N Ext2 elements → N leaf hashes.
    fn build_from_ext2_pairs(codeword: &[GoldilocksExt2]) -> Self {
        let num_leaves = codeword.len() / 2;
        let leaf_hashes: Vec<Poseidon2Hash> = (0..num_leaves)
            .map(|i| hash_ext2_leaf(codeword[2 * i], codeword[2 * i + 1]))
            .collect();
        Self::build_from_leaves(leaf_hashes)
    }

    fn build_from_leaves(leaf_hashes: Vec<Poseidon2Hash>) -> Self {
        let mut layers = vec![leaf_hashes];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let next: Vec<Poseidon2Hash> = (0..prev.len() / 2)
                .map(|i| hash_compress(&prev[2 * i], &prev[2 * i + 1]))
                .collect();
            layers.push(next);
        }
        Self { layers }
    }

    fn root(&self) -> Poseidon2Hash {
        self.layers.last().unwrap()[0]
    }

    fn auth_path(&self, leaf_index: usize) -> Vec<Poseidon2Hash> {
        let mut path = Vec::new();
        let mut idx = leaf_index;
        for layer in &self.layers[..self.layers.len() - 1] {
            let sibling = idx ^ 1;
            path.push(layer[sibling]);
            idx /= 2;
        }
        path
    }
}

// ============================================================================
// CPU codeword fold
// ============================================================================

/// Fold a base-field codeword with an Ext2 challenge.
/// result[i] = val0 + (challenge - x0) * (val1 - val0) * w
/// where val0 = cw[2i], val1 = cw[2i+1], (x0, w) from table entries.
fn cpu_fold_mixed(
    cw: &[u64],
    entries: &[FoldingEntry],
    challenge: GoldilocksExt2,
) -> Vec<GoldilocksExt2> {
    let p = goldilocks_cuda::GOLDILOCKS_PRIME;
    let pair_count = cw.len() / 2;
    (0..pair_count)
        .map(|i| {
            // Normalize codeword values to canonical [0, p) range.
            // GPU may produce non-canonical values which break trait-based arithmetic.
            let val0 = GoldilocksExt2::from_base(GoldilocksField(cw[2 * i] % p));
            let val1 = GoldilocksExt2::from_base(GoldilocksField(cw[2 * i + 1] % p));
            let x0 = GoldilocksExt2::from_base(entries[i].point);
            let w = GoldilocksExt2::from_base(entries[i].weight);
            let diff = ext2_sub(val1, val0);
            let diff_w = ext2_mul(diff, w);
            let cx = ext2_sub(challenge, x0);
            ext2_add(val0, ext2_mul(cx, diff_w))
        })
        .collect()
}

/// Fold an Ext2 codeword with an Ext2 challenge.
fn cpu_fold_ext2(
    cw: &[GoldilocksExt2],
    entries: &[FoldingEntry],
    challenge: GoldilocksExt2,
) -> Vec<GoldilocksExt2> {
    let pair_count = cw.len() / 2;
    (0..pair_count)
        .map(|i| {
            let val0 = cw[2 * i];
            let val1 = cw[2 * i + 1];
            let x0 = GoldilocksExt2::from_base(entries[i].point);
            let w = GoldilocksExt2::from_base(entries[i].weight);
            let diff = ext2_sub(val1, val0);
            let diff_w = ext2_mul(diff, w);
            let cx = ext2_sub(challenge, x0);
            ext2_add(val0, ext2_mul(cx, diff_w))
        })
        .collect()
}

// ============================================================================
// Bit-reverse
// ============================================================================

fn bit_reverse(x: usize, bits: usize) -> usize {
    let mut y = 0;
    let mut x = x;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn bit_reverse_vec(v: &[u64], bits: usize) -> Vec<u64> {
    let mut out = vec![0u64; v.len()];
    for i in 0..v.len() {
        out[bit_reverse(i, bits)] = v[i];
    }
    out
}

// ============================================================================
// Sumcheck oracle computation
// ============================================================================

const INV2: u64 = 9223372034707292161u64;

/// Mixed oracle: bh in F_p, eq in F_{p^2}.
/// Computes P(0), P(1), P(2) from pairs then converts to c0, c1, c2.
fn compute_oracle_mixed(
    bh: &[u64],
    eq: &[GoldilocksExt2],
    size: usize,
) -> SumcheckOracle<GoldilocksExt2> {
    let two = GoldilocksExt2::from_base(GoldilocksField(2));
    let inv2 = GoldilocksExt2::from_base(GoldilocksField(INV2));
    let half = size / 2;

    let mut p0 = GoldilocksExt2::zero();
    let mut p1 = GoldilocksExt2::zero();
    let mut p2 = GoldilocksExt2::zero();

    for j in 0..half {
        let bh0 = GoldilocksExt2::from_base(GoldilocksField(bh[2 * j]));
        let bh1 = GoldilocksExt2::from_base(GoldilocksField(bh[2 * j + 1]));
        let eq0 = eq[2 * j];
        let eq1 = eq[2 * j + 1];

        p0 = ext2_add(p0, ext2_mul(eq0, bh0));
        p1 = ext2_add(p1, ext2_mul(eq1, bh1));

        let eq_interp = ext2_sub(ext2_mul(two, eq1), eq0);
        let bh_interp = ext2_sub(ext2_mul(two, bh1), bh0);
        p2 = ext2_add(p2, ext2_mul(eq_interp, bh_interp));
    }

    let c0 = p0;
    let c2 = ext2_mul(ext2_add(ext2_sub(p0, ext2_mul(two, p1)), p2), inv2);
    let c1 = ext2_sub(ext2_sub(p1, p0), c2);
    SumcheckOracle { c0, c1, c2 }
}

/// Pure Ext2 oracle: both bh and eq in F_{p^2}.
fn compute_oracle_ext2(
    bh: &[GoldilocksExt2],
    eq: &[GoldilocksExt2],
    size: usize,
) -> SumcheckOracle<GoldilocksExt2> {
    let two = GoldilocksExt2::from_base(GoldilocksField(2));
    let inv2 = GoldilocksExt2::from_base(GoldilocksField(INV2));
    let half = size / 2;

    let mut p0 = GoldilocksExt2::zero();
    let mut p1 = GoldilocksExt2::zero();
    let mut p2 = GoldilocksExt2::zero();

    for j in 0..half {
        p0 = ext2_add(p0, ext2_mul(eq[2 * j], bh[2 * j]));
        p1 = ext2_add(p1, ext2_mul(eq[2 * j + 1], bh[2 * j + 1]));

        let eq_interp = ext2_sub(ext2_mul(two, eq[2 * j + 1]), eq[2 * j]);
        let bh_interp = ext2_sub(ext2_mul(two, bh[2 * j + 1]), bh[2 * j]);
        p2 = ext2_add(p2, ext2_mul(eq_interp, bh_interp));
    }

    let c0 = p0;
    let c2 = ext2_mul(ext2_add(ext2_sub(p0, ext2_mul(two, p1)), p2), inv2);
    let c1 = ext2_sub(ext2_sub(p1, p0), c2);
    SumcheckOracle { c0, c1, c2 }
}

// ============================================================================
// Full CPU basefold opening proof
// ============================================================================

/// CPU-based full basefold opening proof.
///
/// Downloads the commitment data (codeword + bh_evals) from GPU, then performs
/// the complete basefold protocol on CPU: sumcheck oracles, codeword folding,
/// Merkle tree construction, and query proof extraction.
///
/// Suitable for small polynomials (n ≤ ~16) where GPU kernel launch overhead
/// exceeds the actual computation cost.
pub fn cpu_full_open_ext2(
    commitment: &BasefoldCommitment,
    point: &[GoldilocksExt2],
    table: &BasefoldTable,
    transcript: &mut Transcript,
    num_queries: usize,
) -> BasefoldOpeningProof {
    let num_vars = commitment.num_vars;
    let log_rate = commitment.log_rate;
    let n = 1usize << num_vars;
    assert_eq!(point.len(), num_vars);

    // Download commitment data from GPU
    let cache = commitment.to_host_cache().expect("to_host_cache failed");
    let codeword = cache.codeword; // GL field, len = 2^(num_vars + log_rate)
    let bh_raw = cache.bh_evals; // GL field, len = 2^num_vars

    // Bit-reverse bh_evals (matches GPU BasefoldBatch::bit_reverse_gl)
    let bh_raw_br = bit_reverse_vec(&bh_raw, num_vars);

    // Normalize bh to canonical range [0, p) for correct field arithmetic.
    // GPU may store non-canonical values (>= p), which cause incorrect results
    // with the trait-based Add/Sub that assume canonical inputs.
    let p = goldilocks_cuda::GOLDILOCKS_PRIME;
    let bh: Vec<u64> = bh_raw_br.iter().map(|&v| v % p).collect();

    // 1. Transcript preamble
    transcript.observe_hash(&cache.root);
    for p in point {
        transcript.observe_ext2(*p);
    }

    // 2. Build eq polynomial
    let eq = evaluate_lagrange_basis_ext2(point);

    // 3. Compute eval = Σ bh[i] * eq[i] (mixed: bh in F_p, eq in F_{p^2})
    let mut eval = GoldilocksExt2::zero();
    for i in 0..n {
        let bh_ext2 = GoldilocksExt2::from_base(GoldilocksField(bh[i]));
        eval = ext2_add(eval, ext2_mul(bh_ext2, eq[i]));
    }

    // 4. First sumcheck oracle (mixed mode)
    let oracle0 = compute_oracle_mixed(&bh, &eq, n);
    let mut sumcheck_oracles = vec![oracle0.clone()];
    transcript.observe_ext2(oracle0.c0);
    transcript.observe_ext2(oracle0.c1);
    transcript.observe_ext2(oracle0.c2);

    // 5. Round 0: mixed → ext2 transition
    let challenge0 = transcript.sample_challenge_ext2();

    // Fold bh (mixed → ext2) and eq at challenge0
    let half = n / 2;
    let mut bh_ext2: Vec<GoldilocksExt2> = (0..half)
        .map(|j| {
            let lo = GoldilocksExt2::from_base(GoldilocksField(bh[2 * j]));
            let hi = GoldilocksExt2::from_base(GoldilocksField(bh[2 * j + 1]));
            ext2_add(lo, ext2_mul(challenge0, ext2_sub(hi, lo)))
        })
        .collect();
    let mut eq_work: Vec<GoldilocksExt2> = (0..half)
        .map(|j| {
            let lo = eq[2 * j];
            let hi = eq[2 * j + 1];
            ext2_add(lo, ext2_mul(challenge0, ext2_sub(hi, lo)))
        })
        .collect();
    let mut sc_size = half; // current sumcheck polynomial size

    // Oracle[1]: first pure ext2 round
    let oracle1 = compute_oracle_ext2(&bh_ext2, &eq_work, sc_size);
    sumcheck_oracles.push(oracle1.clone());
    transcript.observe_ext2(oracle1.c0);
    transcript.observe_ext2(oracle1.c1);
    transcript.observe_ext2(oracle1.c2);

    // Fold codeword: mixed fold (GL → Ext2)
    let level_entries = &table.entries
        [table.level_offsets[0]..table.level_offsets[0] + table.level_sizes[0]];
    let mut cw_ext2 = cpu_fold_mixed(&codeword, level_entries, challenge0);

    // Build initial Merkle tree from folded ext2 codeword
    let tree0 = CpuMerkleTree::build_from_ext2_pairs(&cw_ext2);
    let root0 = tree0.root();
    transcript.observe_hash(&root0);

    let mut folded_roots = vec![root0];
    let mut folded_codewords: Vec<Vec<GoldilocksExt2>> = vec![cw_ext2.clone()];
    let mut folded_trees: Vec<CpuMerkleTree> = vec![tree0];

    // 6. Remaining rounds: all pure ext2
    for round in 1..num_vars - 1 {
        let challenge = transcript.sample_challenge_ext2();

        // Fold eq and bh at challenge
        let half = sc_size / 2;
        for j in 0..half {
            let lo = eq_work[2 * j];
            let hi = eq_work[2 * j + 1];
            eq_work[j] = ext2_add(lo, ext2_mul(challenge, ext2_sub(hi, lo)));

            let blo = bh_ext2[2 * j];
            let bhi = bh_ext2[2 * j + 1];
            bh_ext2[j] = ext2_add(blo, ext2_mul(challenge, ext2_sub(bhi, blo)));
        }
        sc_size = half;

        // Compute oracle
        let oracle = compute_oracle_ext2(&bh_ext2, &eq_work, sc_size);
        sumcheck_oracles.push(oracle.clone());
        transcript.observe_ext2(oracle.c0);
        transcript.observe_ext2(oracle.c1);
        transcript.observe_ext2(oracle.c2);

        // Fold codeword
        let level_entries = &table.entries
            [table.level_offsets[round]..table.level_offsets[round] + table.level_sizes[round]];
        let folded_cw = cpu_fold_ext2(&cw_ext2, level_entries, challenge);

        // Build Merkle tree
        let tree = CpuMerkleTree::build_from_ext2_pairs(&folded_cw);
        let root = tree.root();
        transcript.observe_hash(&root);
        folded_roots.push(root);

        folded_codewords.push(folded_cw.clone());
        folded_trees.push(tree);
        cw_ext2 = folded_cw;
    }

    // 7. Last fold (no tree)
    let last_challenge = transcript.sample_challenge_ext2();
    let level_entries = &table.entries[table.level_offsets[num_vars - 1]
        ..table.level_offsets[num_vars - 1] + table.level_sizes[num_vars - 1]];
    let final_cw = cpu_fold_ext2(&cw_ext2, level_entries, last_challenge);

    // 8. Extract query proofs
    let initial_cw_len = 1usize << (num_vars + log_rate);
    let mut query_proofs = Vec::with_capacity(num_queries);

    // Build initial Merkle tree from GL codeword for query extraction
    let initial_tree = CpuMerkleTree::build_from_gl_pairs(&codeword);

    for _ in 0..num_queries {
        let idx_raw = transcript.sample_challenge().0 as usize;
        let leaf_idx = idx_raw % (initial_cw_len / 2);

        // Initial codeword: base field pair
        let mut values = vec![(
            GoldilocksExt2::new(
                GoldilocksField(codeword[2 * leaf_idx]),
                GoldilocksField(0),
            ),
            GoldilocksExt2::new(
                GoldilocksField(codeword[2 * leaf_idx + 1]),
                GoldilocksField(0),
            ),
        )];
        let mut paths = vec![initial_tree.auth_path(leaf_idx)];

        // Cascade through folded codewords
        let mut idx = leaf_idx / 2;
        for i in 0..folded_codewords.len() {
            let fc = &folded_codewords[i];
            let pair_idx = idx;
            if pair_idx * 2 + 1 < fc.len() {
                values.push((fc[pair_idx * 2], fc[pair_idx * 2 + 1]));
                paths.push(folded_trees[i].auth_path(pair_idx));
            }
            idx /= 2;
        }

        query_proofs.push(QueryProof {
            index: leaf_idx,
            values,
            merkle_paths: paths,
        });
    }

    BasefoldOpeningProof {
        eval,
        gpu_proof: BasefoldProofExt2 {
            eval,
            sumcheck_oracles,
            folded_roots,
            final_codeword: final_cw,
            query_proofs,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goldilocks_cuda::basefold::{BasefoldCommitment, BasefoldTable, BasefoldVerifier};

    #[test]
    fn test_cpu_vs_gpu_opening() {
        goldilocks_cuda::init().expect("CUDA init");

        // Create polynomial (n=6, 64 elements) with table generated for larger max_num_vars=9
        let num_vars = 6;
        let max_num_vars = 9; // table generated for larger size
        let log_rate = 3;
        let seed = 42;
        let num_queries = 10;
        let p = goldilocks_cuda::GOLDILOCKS_PRIME;

        let evals: Vec<u64> = (0..64).map(|i| (p - 1 - i * 1000 % p) % p).collect();
        let commitment = BasefoldCommitment::commit(&evals.iter().map(|&x| GoldilocksField(x)).collect::<Vec<_>>(), num_vars, log_rate)
            .expect("commit failed");

        // Table with max_num_vars > num_vars (like DAG test)
        let mut table = BasefoldTable::generate(max_num_vars, log_rate, max_num_vars, seed);
        table.upload().expect("upload");

        // Point
        let point: Vec<GoldilocksExt2> = (0..num_vars)
            .map(|i| GoldilocksExt2::from_base(GoldilocksField(100 + i as u64)))
            .collect();

        // GPU opening
        let mut t_gpu = Transcript::new(b"test-open");
        t_gpu.append_u64(b"", 42);
        let gpu_proof = commitment.open_ext2(&point, &table, &mut t_gpu, num_queries)
            .expect("GPU open failed");

        // CPU opening
        let mut t_cpu = Transcript::new(b"test-open");
        t_cpu.append_u64(b"", 42);
        let cpu_proof = cpu_full_open_ext2(&commitment, &point, &table, &mut t_cpu, num_queries);

        // Compare eval
        eprintln!("GPU eval: {:?}", gpu_proof.eval);
        eprintln!("CPU eval: {:?}", cpu_proof.gpu_proof.eval);
        assert_eq!(gpu_proof.eval.c0.0, cpu_proof.gpu_proof.eval.c0.0, "eval mismatch");

        // Compare oracles
        for (i, (g, c)) in gpu_proof.sumcheck_oracles.iter().zip(cpu_proof.gpu_proof.sumcheck_oracles.iter()).enumerate() {
            let p = goldilocks_cuda::GOLDILOCKS_PRIME;
            let c0_match = g.c0.c0.0 % p == c.c0.c0.0 % p && g.c0.c1.0 % p == c.c0.c1.0 % p;
            let c1_match = g.c1.c0.0 % p == c.c1.c0.0 % p && g.c1.c1.0 % p == c.c1.c1.0 % p;
            let c2_match = g.c2.c0.0 % p == c.c2.c0.0 % p && g.c2.c1.0 % p == c.c2.c1.0 % p;
            if !c0_match || !c1_match || !c2_match {
                eprintln!("Oracle {} MISMATCH:", i);
                eprintln!("  GPU: c0={:?} c1={:?} c2={:?}", g.c0, g.c1, g.c2);
                eprintln!("  CPU: c0={:?} c1={:?} c2={:?}", c.c0, c.c1, c.c2);
            }
        }

        // Compare folded roots
        for (i, (g, c)) in gpu_proof.folded_roots.iter().zip(cpu_proof.gpu_proof.folded_roots.iter()).enumerate() {
            if g != c {
                eprintln!("Folded root {} MISMATCH: GPU={:?} CPU={:?}", i, g, c);
            }
        }

        // Verify GPU proof
        let mut t_vg = Transcript::new(b"test-open");
        t_vg.append_u64(b"", 42);
        let gpu_ok = BasefoldVerifier::verify_ext2(&commitment.root, &point, &gpu_proof, &table, &mut t_vg)
            .expect("GPU verify");
        eprintln!("GPU verify: {}", gpu_ok);

        // Verify CPU proof
        let mut t_vc = Transcript::new(b"test-open");
        t_vc.append_u64(b"", 42);
        let cpu_ok = BasefoldVerifier::verify_ext2(&commitment.root, &point, &cpu_proof.gpu_proof, &table, &mut t_vc)
            .expect("CPU verify");
        eprintln!("CPU verify: {}", cpu_ok);

        assert!(gpu_ok, "GPU proof should verify");
        assert!(cpu_ok, "CPU proof should verify");
    }
}
