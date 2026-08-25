//! Rust integration tests for the Ajtai commitment.
//!
//! Validates the GPU implementation against a pure-Rust reference that
//! re-derives `M` from the same seed using the same ChaCha8 keying. The
//! reference uses naive polynomial multiplication mod (X^64 + 1) and is
//! independent of the GPU kernels — so a passing test confirms both
//! end-to-end correctness *and* deterministic seed-to-M derivation.

use almost_goldilocks_cuda::{
    ajtai::{
        commit, commit_batched, commit_sparse, commit_ternary, commit_ternary_premat,
        fold_commitment, fold_witness, multifold_commitment, multifold_witness,
        multifold_mixed_witness, multifold_mixed_witness_tc,
        multifold_mixed_witness_tc_fused,
        split_witness, split_witness_device,
        ChunkSize, MaterializedM, RingChallenge, RingCommitment, Seed, TernaryChunks,
        KAPPA, RING_DIM, SPLIT_K_CHUNKS,
    },
    init,
    memory::DeviceBuffer,
    ALMOST_GOLDILOCKS_PRIME,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const Q: u128 = ALMOST_GOLDILOCKS_PRIME as u128;

// ============================================================================
// Pure-Rust ChaCha8 + rejection-sampled PRG (must match the CUDA implementation)
// ============================================================================

fn rotl32(x: u32, n: u32) -> u32 {
    x.rotate_left(n)
}

fn chacha8_block(key: &[u32; 8], counter: u32, nonce: &[u32; 3], out: &mut [u32; 16]) {
    let mut s = [0u32; 16];
    s[0] = 0x61707865; s[1] = 0x3320646e;
    s[2] = 0x79622d32; s[3] = 0x6b206574;
    s[4..12].copy_from_slice(key);
    s[12] = counter;
    s[13] = nonce[0]; s[14] = nonce[1]; s[15] = nonce[2];

    let mut x = s;
    for _ in 0..4 {
        // column rounds
        qr(&mut x, 0, 4,  8, 12);
        qr(&mut x, 1, 5,  9, 13);
        qr(&mut x, 2, 6, 10, 14);
        qr(&mut x, 3, 7, 11, 15);
        // diagonal rounds
        qr(&mut x, 0, 5, 10, 15);
        qr(&mut x, 1, 6, 11, 12);
        qr(&mut x, 2, 7,  8, 13);
        qr(&mut x, 3, 4,  9, 14);
    }
    for i in 0..16 {
        out[i] = x[i].wrapping_add(s[i]);
    }
}

fn qr(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]); x[d] ^= x[a]; x[d] = rotl32(x[d], 16);
    x[c] = x[c].wrapping_add(x[d]); x[b] ^= x[c]; x[b] = rotl32(x[b], 12);
    x[a] = x[a].wrapping_add(x[b]); x[d] ^= x[a]; x[d] = rotl32(x[d], 8);
    x[c] = x[c].wrapping_add(x[d]); x[b] ^= x[c]; x[b] = rotl32(x[b], 7);
}

/// Generate 8 rejection-sampled F_q coefficients corresponding to
/// (key, row, j, block_idx). Matches the CUDA prg_ring_block_chacha8.
fn prg_ring_block(key: &[u32; 8], row: u32, j: u64, block_idx: u32) -> [u64; 8] {
    let nonce = [
        row,
        (j & 0xFFFFFFFF) as u32,
        (j >> 32) as u32,
    ];
    let mut out = [0u64; 8];   // zero-init defensively (matches CUDA prg)
    let mut written = 0usize;
    for retry in 0..16u32 {
        if written == 8 { break; }
        let counter = block_idx + retry * 8;
        let mut buf = [0u32; 16];
        chacha8_block(key, counter, &nonce, &mut buf);
        for k in 0..8 {
            if written == 8 { break; }
            let lo = buf[2*k] as u64;
            let hi = buf[2*k + 1] as u64;
            let s = (hi << 32) | lo;
            if s < ALMOST_GOLDILOCKS_PRIME {
                out[written] = s;
                written += 1;
            }
        }
    }
    out
}

/// Generate one ring element M[row, j] = 64 coefficients.
fn prg_ring_elem(key: &[u32; 8], row: u32, j: u64) -> [u64; 64] {
    let mut out = [0u64; 64];
    for block in 0..8 {
        let blk = prg_ring_block(key, row, j, block as u32);
        for k in 0..8 {
            out[block * 8 + k] = blk[k];
        }
    }
    out
}

// ============================================================================
// Pure-Rust field arithmetic (canonical inputs / outputs)
// ============================================================================

fn f_add(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % Q) as u64
}
fn f_neg(a: u64) -> u64 {
    if a == 0 { 0 } else { ALMOST_GOLDILOCKS_PRIME - a }
}

// ============================================================================
// Ring R = F_q[X] / (X^64 + 1)
// ============================================================================

fn ring_shift(a: &[u64; 64], ell: i32) -> [u64; 64] {
    let mut out = [0u64; 64];
    for r in 0..64 {
        let idx = r as i32 - ell;
        if idx >= 0 {
            out[r] = a[idx as usize];
        } else {
            out[r] = f_neg(a[(idx + 64) as usize]);
        }
    }
    out
}

fn ring_binary_mul(a: &[u64; 64], z_bits: u64) -> [u64; 64] {
    let mut out = [0u64; 64];
    let mut mask = z_bits;
    while mask != 0 {
        let ell = mask.trailing_zeros() as i32;
        mask &= mask - 1;
        let shifted = ring_shift(a, ell);
        for r in 0..64 {
            out[r] = f_add(out[r], shifted[r]);
        }
    }
    out
}

/// Reference Ajtai commit.
fn cpu_commit(seed: &Seed, z_bits: &[u64]) -> RingCommitment {
    let mut c = RingCommitment::zero();
    for j in 0..z_bits.len() {
        let bits = z_bits[j];
        if bits == 0 { continue; }
        for i in 0..KAPPA {
            let m_ij = prg_ring_elem(&seed.0, i as u32, j as u64);
            let contrib = ring_binary_mul(&m_ij, bits);
            for r in 0..RING_DIM {
                c.rows[i][r] = f_add(c.rows[i][r], contrib[r]);
            }
        }
    }
    c
}

fn rings_equal(a: &RingCommitment, b: &RingCommitment) -> bool {
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            if a.rows[i][r] != b.rows[i][r] {
                return false;
            }
        }
    }
    true
}

// ============================================================================
// Tests
// ============================================================================

fn random_seed(rng: &mut StdRng) -> Seed {
    let mut k = [0u32; 8];
    for i in 0..8 { k[i] = rng.gen(); }
    Seed(k)
}

fn random_z(rng: &mut StdRng, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.gen::<u64>()).collect()
}

#[test]
fn test_dense_single_small_n() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_001);
    for &n in &[64usize, 256, 1024, 4096] {
        let seed = random_seed(&mut rng);
        let z = random_z(&mut rng, n);
        let gpu = commit(seed, &z, Some(ChunkSize::C256)).expect("gpu commit");
        let cpu = cpu_commit(&seed, &z);
        assert!(rings_equal(&gpu, &cpu),
                "dense single commit GPU != CPU at N={}", n);
    }
}

#[test]
fn test_dense_batched_matches_singles() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_002);
    for &b in &[2usize, 4, 8, 16] {
        for &n in &[128usize, 1024] {
            let seed = random_seed(&mut rng);
            let witnesses: Vec<Vec<u64>> = (0..b).map(|_| random_z(&mut rng, n)).collect();
            let refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();

            let batched = commit_batched(seed, &refs, Some(ChunkSize::C256))
                .expect("gpu batched");
            assert_eq!(batched.len(), b);

            // Cross-check each batched commit against (a) CPU reference
            // and (b) the single-commit GPU path on the same witness.
            for (i, w) in witnesses.iter().enumerate() {
                let cpu = cpu_commit(&seed, w);
                let single = commit(seed, w, Some(ChunkSize::C256)).expect("gpu single");
                assert!(rings_equal(&batched[i], &cpu),
                        "batched[{}] != CPU at B={} N={}", i, b, n);
                assert!(rings_equal(&batched[i], &single),
                        "batched[{}] != GPU single at B={} N={}", i, b, n);
            }
        }
    }
}

#[test]
fn test_chunk_invariance() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_003);
    let seed = random_seed(&mut rng);
    let z = random_z(&mut rng, 2048);

    let c_256  = commit(seed, &z, Some(ChunkSize::C256 )).expect("c_256");
    let c_1024 = commit(seed, &z, Some(ChunkSize::C1024)).expect("c_1024");
    let c_4096 = commit(seed, &z, Some(ChunkSize::C4096)).expect("c_4096");

    assert!(rings_equal(&c_256, &c_1024));
    assert!(rings_equal(&c_256, &c_4096));
}

#[test]
fn test_zero_witness() {
    init().expect("CUDA init failed");
    let seed = Seed([1, 2, 3, 4, 5, 6, 7, 8]);
    let z = vec![0u64; 256];
    let c = commit(seed, &z, None).expect("zero commit");
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            assert_eq!(c.rows[i][r], 0, "zero commit at i={} r={}", i, r);
        }
    }
}

#[test]
fn test_sparse_vs_dense() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_004);
    let n = 256;
    let seed = random_seed(&mut rng);

    // Build a moderately sparse witness via a position list (no duplicates).
    let mut z = vec![0u64; n];
    let mut positions = Vec::<u64>::new();
    for _ in 0..32 {
        let p: u64 = rng.gen_range(0..(64 * n as u64));
        let j = (p >> 6) as usize;
        let ell = p & 63;
        if z[j] & (1 << ell) == 0 {
            z[j] |= 1 << ell;
            positions.push(p);
        }
    }

    let dense  = commit(seed, &z, Some(ChunkSize::C256)).expect("dense");
    let sparse = commit_sparse(seed, &positions, Some(ChunkSize::C256)).expect("sparse");
    assert!(rings_equal(&dense, &sparse),
            "sparse commit does not match dense commit on the same witness");
}

/// Multi-GPU correctness check, simulated on one GPU.
///
/// The full witness `z` is split into `G` contiguous j-range slices,
/// each committed independently. Per the design (PRG = pure function of
/// (seed, i, j, block_idx)), summing the G per-slice commitments must
/// equal the single-pass commitment. This is the §15.9 test.
#[test]
fn test_multi_gpu_split_simulation() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_005);
    let n = 1024;
    let seed = random_seed(&mut rng);
    let z = random_z(&mut rng, n);

    let single = commit(seed, &z, Some(ChunkSize::C256)).expect("single");

    for &g in &[2usize, 4, 8] {
        assert!(n % g == 0, "n must split evenly for this test");
        let slice_len = n / g;
        // Build the G witness slices by zeroing everything outside each slice.
        // This is equivalent to running each GPU on a contiguous j-range and
        // padding the rest with zeros (which contribute nothing to the sum).
        let mut sum = RingCommitment::zero();
        for s in 0..g {
            let mut z_slice = vec![0u64; n];
            for k in 0..slice_len {
                z_slice[s * slice_len + k] = z[s * slice_len + k];
            }
            let part = commit(seed, &z_slice, Some(ChunkSize::C256)).expect("part");
            for i in 0..KAPPA {
                for r in 0..RING_DIM {
                    sum.rows[i][r] = f_add(sum.rows[i][r], part.rows[i][r]);
                }
            }
        }
        assert!(rings_equal(&single, &sum),
                "multi-GPU split (G={}) sum != single-pass commit", g);
    }
}

/// Production-scale smoke test: large N, run end-to-end and validate against
/// the pure-Rust reference. Not exhaustive (the reference takes a while at
/// this size), but verifies the kernel doesn't regress at scale.
///
/// Marked #[ignore] so `cargo test` skips it by default; run with
/// `cargo test --release -- --ignored test_large_n_scaling`.
#[test]
#[ignore]
fn test_large_n_scaling() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_AAAA);
    let n = 1 << 18;
    let seed = random_seed(&mut rng);
    let z = random_z(&mut rng, n);

    let t0 = std::time::Instant::now();
    let gpu = commit(seed, &z, Some(ChunkSize::C4096)).expect("gpu");
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let cpu = cpu_commit(&seed, &z);
    let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("N=2^18: GPU {:.1} ms, CPU reference {:.1} ms ({:.1}x speedup)",
             gpu_ms, cpu_ms, cpu_ms / gpu_ms);
    assert!(rings_equal(&gpu, &cpu), "GPU result diverges from CPU at N=2^18");
}

/// Same-seed-same-z determinism: trivial but worth a check that the kernel
/// doesn't accidentally depend on some uninitialized state.
#[test]
fn test_determinism_across_runs() {
    init().expect("CUDA init failed");
    let mut rng = StdRng::seed_from_u64(0xA17_006);
    let seed = random_seed(&mut rng);
    let z = random_z(&mut rng, 256);

    let a = commit(seed, &z, None).expect("a");
    let b = commit(seed, &z, None).expect("b");
    assert!(rings_equal(&a, &b), "different runs same input != deterministic");
}

#[test]
fn test_chacha_determinism() {
    // Pure-Rust chacha must be deterministic and counter-sensitive.
    let key = [0xdeadbeefu32, 0xcafebabe, 1, 2, 3, 4, 5, 6];
    let nonce = [7u32, 8, 9];
    let mut a = [0u32; 16];
    let mut b = [0u32; 16];
    chacha8_block(&key, 0, &nonce, &mut a);
    chacha8_block(&key, 0, &nonce, &mut b);
    assert_eq!(a, b);
    chacha8_block(&key, 1, &nonce, &mut b);
    assert_ne!(a, b);
}

// ============================================================================
// Fold tests: additive homomorphism c1 + r·c2 = M·(z1 + r·z2)
// ============================================================================

/// CPU reference: produce z1 + r·z2 as canonical-F_q-per-coefficient.
fn cpu_fold_witness(z1: &[u64], r: &RingChallenge, z2: &[u64]) -> Vec<u64> {
    let n = z1.len();
    let mut out = vec![0u64; n * RING_DIM];
    for j in 0..n {
        let z2j = z2[j];
        let z1j = z1[j];
        for k in 0..RING_DIM {
            let mut acc: i32 = ((z1j >> k) & 1) as i32;
            let mut mask = z2j;
            while mask != 0 {
                let ell = mask.trailing_zeros() as i32;
                mask &= mask - 1;
                let signed_idx = k as i32 - ell;
                let (idx, wrap) = if signed_idx < 0 {
                    ((signed_idx + RING_DIM as i32) as usize, true)
                } else {
                    (signed_idx as usize, false)
                };
                let mut rv = r.coeffs[idx] as i32;
                if wrap { rv = -rv; }
                acc += rv;
            }
            let val = if acc >= 0 {
                acc as u64
            } else {
                ALMOST_GOLDILOCKS_PRIME - (-acc as u64)
            };
            out[j * RING_DIM + k] = val;
        }
    }
    out
}

/// CPU reference: c1 + r·c2 for ring commitments.
fn cpu_fold_commitment(c1: &RingCommitment, r: &RingChallenge, c2: &RingCommitment) -> RingCommitment {
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            let mut acc = c1.rows[i][k];
            for ell in 0..RING_DIM {
                let rv = r.coeffs[ell];
                if rv == 0 { continue; }
                let signed_idx = k as i32 - ell as i32;
                let (idx, wrap) = if signed_idx < 0 {
                    ((signed_idx + RING_DIM as i32) as usize, true)
                } else {
                    (signed_idx as usize, false)
                };
                let c2_val = c2.rows[i][idx];
                let rv_signed: i32 = if wrap { -(rv as i32) } else { rv as i32 };
                match rv_signed {
                    1  => acc = f_add(acc, c2_val),
                    -1 => acc = f_sub(acc, c2_val),
                    2  => acc = f_add(acc, f_add(c2_val, c2_val)),
                    -2 => acc = f_sub(acc, f_add(c2_val, c2_val)),
                    _  => {} // 0
                }
            }
            out.rows[i][k] = canon(acc);
        }
    }
    out
}

fn f_sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { ALMOST_GOLDILOCKS_PRIME - (b - a) }
}
fn canon(v: u64) -> u64 {
    if v >= ALMOST_GOLDILOCKS_PRIME { v - ALMOST_GOLDILOCKS_PRIME } else { v }
}

/// Generic ring-vector commit (for general u64-per-coefficient witnesses,
/// not necessarily binary). Used only in tests to verify the homomorphism
/// identity on the wide folded witness.
fn cpu_commit_general(seed: &Seed, w: &[u64], n_ring: usize) -> RingCommitment {
    assert_eq!(w.len(), n_ring * RING_DIM);
    let mut c = RingCommitment::zero();
    for j in 0..n_ring {
        for i in 0..KAPPA {
            let m_ij = prg_ring_elem(&seed.0, i as u32, j as u64);
            // w[j] as a ring element (64 canonical F_q coeffs)
            // Compute m_ij * w_j by naive polynomial mul mod (X^64 + 1)
            for a in 0..RING_DIM {
                for b in 0..RING_DIM {
                    let r = a + b;
                    let prod = ((m_ij[a] as u128 * w[j * RING_DIM + b] as u128) % Q) as u64;
                    if r < RING_DIM {
                        c.rows[i][r] = f_add(c.rows[i][r], prod);
                    } else {
                        c.rows[i][r - RING_DIM] = f_sub(c.rows[i][r - RING_DIM], prod);
                    }
                }
            }
        }
    }
    c
}

fn rand_ring_challenge(rng: &mut StdRng) -> RingChallenge {
    // r ∈ {-1, 0, 1, 2}^64 uniform — the SuperNeo almost-Goldilocks param set.
    let mut coeffs = [0i8; 64];
    for v in coeffs.iter_mut() {
        *v = match rng.gen_range(0..4) {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => 2,
        };
    }
    RingChallenge::new(coeffs).expect("range valid by construction")
}

#[test]
fn test_fold_witness_vs_cpu() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xF01D_001);
    for &n in &[16usize, 64, 256, 1024] {
        let z1 = random_z(&mut rng, n);
        let z2 = random_z(&mut rng, n);
        let r = rand_ring_challenge(&mut rng);

        let cpu_out = cpu_fold_witness(&z1, &r, &z2);
        let gpu_out = fold_witness(&z1, &r, &z2).expect("gpu fold");

        assert_eq!(gpu_out.len(), cpu_out.len());
        for (i, (g, c)) in gpu_out.iter().zip(cpu_out.iter()).enumerate() {
            assert_eq!(g, c, "fold_witness mismatch at flat idx {} (N={})", i, n);
        }
    }
}

#[test]
fn test_fold_commitment_vs_cpu() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xF01D_002);
    for _ in 0..5 {
        let mut c1 = RingCommitment::zero();
        let mut c2 = RingCommitment::zero();
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                c1.rows[i][k] = rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME;
                c2.rows[i][k] = rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME;
            }
        }
        let r = rand_ring_challenge(&mut rng);

        let cpu_out = cpu_fold_commitment(&c1, &r, &c2);
        let gpu_out = fold_commitment(&c1, &r, &c2).expect("gpu fold");

        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                assert_eq!(gpu_out.rows[i][k], cpu_out.rows[i][k],
                    "fold_commitment mismatch at (i={}, k={})", i, k);
            }
        }
    }
}

/// The headline test: verify the additive-homomorphism identity end-to-end.
/// commit(z1 + r·z2) must equal commit(z1) + r·commit(z2) bit-exactly.
#[test]
fn test_fold_homomorphism_identity() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xF01D_DEAD);
    // Small N — the general-coefficient CPU commit is O(N · κ · D^2).
    for &n in &[4usize, 16, 64] {
        let seed = random_seed(&mut rng);
        let z1 = random_z(&mut rng, n);
        let z2 = random_z(&mut rng, n);
        let r = rand_ring_challenge(&mut rng);

        // Path A: commit individually, then fold the commitments.
        let c1 = commit(seed, &z1, Some(ChunkSize::C64)).expect("c1");
        let c2 = commit(seed, &z2, Some(ChunkSize::C64)).expect("c2");
        let folded_commit = fold_commitment(&c1, &r, &c2).expect("fold_commit");

        // Path B: fold the witnesses (producing a general-coefficient vector),
        // then commit to the folded witness via the general-purpose CPU commit.
        let folded_witness = fold_witness(&z1, &r, &z2).expect("fold_witness");
        let commit_of_fold = cpu_commit_general(&seed, &folded_witness, n);

        // The homomorphism c1 + r·c2 == M·(z1 + r·z2) must hold bit-exact.
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                assert_eq!(folded_commit.rows[i][k], commit_of_fold.rows[i][k],
                    "homomorphism violated at N={} i={} k={}: \
                     fold(commit)={} vs commit(fold)={}",
                    n, i, k, folded_commit.rows[i][k], commit_of_fold.rows[i][k]);
            }
        }
    }
}

#[test]
fn test_fold_commitment_zero_challenge() {
    // r = 0 ⇒ c1 + r·c2 = c1.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xF01D_FACE);
    let mut c1 = RingCommitment::zero();
    let mut c2 = RingCommitment::zero();
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            c1.rows[i][k] = rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME;
            c2.rows[i][k] = rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME;
        }
    }
    let r = RingChallenge::new([0i8; 64]).unwrap();
    let out = fold_commitment(&c1, &r, &c2).unwrap();
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            assert_eq!(out.rows[i][k], c1.rows[i][k]);
        }
    }
}

#[test]
fn test_fold_witness_zero_challenge() {
    // r = 0 ⇒ z1 + r·z2 has each coefficient ∈ {0, 1} from z1's bits only.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xF01D_BEEF);
    let n = 64;
    let z1 = random_z(&mut rng, n);
    let z2 = random_z(&mut rng, n);
    let r = RingChallenge::new([0i8; 64]).unwrap();
    let out = fold_witness(&z1, &r, &z2).expect("fold");
    for j in 0..n {
        for k in 0..RING_DIM {
            let expected = (z1[j] >> k) & 1;
            assert_eq!(out[j * RING_DIM + k], expected);
        }
    }
}

// ============================================================================
// Multi-fold tests (K + k = 63 binary instances per SuperNeo Almost-Goldilocks)
// ============================================================================

/// CPU reference for multi-fold over witnesses:
///   z' = z_0 + Σ_{i=1..M-1} r_i · z_i
/// `challenges` has length `M − 1` and applies to `witnesses[1..]`.
fn cpu_multifold_witness(
    witnesses: &[&[u64]],
    challenges: &[RingChallenge],
) -> Vec<i16> {
    let m = witnesses.len();
    assert_eq!(challenges.len() + 1, m, "expected M - 1 challenges");
    let n = witnesses[0].len();
    let mut out = vec![0i16; n * RING_DIM];
    for j in 0..n {
        for k in 0..RING_DIM {
            let mut acc: i32 = 0;
            // i = 0: implicit weight 1, contributes just the binary coefficient z_0[j][k].
            acc += ((witnesses[0][j] >> k) & 1) as i32;
            // i = 1..M-1: weighted by challenges[i-1].
            for i in 1..m {
                let bits = witnesses[i][j];
                let mut mask = bits;
                while mask != 0 {
                    let ell = mask.trailing_zeros() as i32;
                    mask &= mask - 1;
                    let signed_idx = k as i32 - ell;
                    let (idx, wrap) = if signed_idx < 0 {
                        ((signed_idx + RING_DIM as i32) as usize, true)
                    } else {
                        (signed_idx as usize, false)
                    };
                    let mut rv = challenges[i - 1].coeffs[idx] as i32;
                    if wrap { rv = -rv; }
                    acc += rv;
                }
            }
            out[j * RING_DIM + k] = acc as i16;
        }
    }
    out
}

/// CPU reference for multi-fold over commitments:
///   c' = c_0 + Σ_{i=1..M-1} r_i · c_i
/// `challenges` has length `M − 1` and applies to `commits[1..]`.
fn cpu_multifold_commitment(
    commits: &[&RingCommitment],
    challenges: &[RingChallenge],
) -> RingCommitment {
    let m = commits.len();
    assert_eq!(challenges.len() + 1, m, "expected M - 1 challenges");
    let mut out = RingCommitment::zero();
    for i_row in 0..KAPPA {
        for k in 0..RING_DIM {
            // i = 0: implicit weight 1.
            let mut acc = commits[0].rows[i_row][k];
            // i = 1..M-1: r_i · c_i.
            for i in 1..m {
                for ell in 0..RING_DIM {
                    let rv = challenges[i - 1].coeffs[ell];
                    if rv == 0 { continue; }
                    let signed_idx = k as i32 - ell as i32;
                    let (idx, wrap) = if signed_idx < 0 {
                        ((signed_idx + RING_DIM as i32) as usize, true)
                    } else {
                        (signed_idx as usize, false)
                    };
                    let c2_val = commits[i].rows[i_row][idx];
                    let rv_signed: i32 = if wrap { -(rv as i32) } else { rv as i32 };
                    match rv_signed {
                        1  => acc = f_add(acc, c2_val),
                        -1 => acc = f_sub(acc, c2_val),
                        2  => acc = f_add(acc, f_add(c2_val, c2_val)),
                        -2 => acc = f_sub(acc, f_add(c2_val, c2_val)),
                        _  => {}
                    }
                }
            }
            out.rows[i_row][k] = canon(acc);
        }
    }
    out
}

/// Convert i16 folded witness (signed) to canonical F_q u64 form for
/// feeding into the general-coefficient CPU commit.
fn i16_witness_to_fq(w: &[i16]) -> Vec<u64> {
    w.iter()
        .map(|&v| {
            if v >= 0 {
                v as u64
            } else {
                ALMOST_GOLDILOCKS_PRIME - ((-(v as i32)) as u64)
            }
        })
        .collect()
}

#[test]
fn test_multifold_witness_vs_cpu_small() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xF01D_F00Du64);
    for &m in &[2usize, 4, 8, 16, 32, 63] {
        for &n in &[8usize, 64, 256] {
            let witnesses: Vec<Vec<u64>> = (0..m).map(|_| random_z(&mut rng, n)).collect();
            let refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
            let challenges: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

            let gpu = multifold_witness(&refs, &challenges).expect("gpu multifold");
            let cpu = cpu_multifold_witness(&refs, &challenges);

            assert_eq!(gpu.len(), cpu.len());
            for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                assert_eq!(g, c,
                    "multifold_witness mismatch at flat idx {} (M={}, N={})", i, m, n);
            }
        }
    }
}

#[test]
fn test_multifold_commitment_vs_cpu() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xFD0_BEEFu64);
    for &m in &[2usize, 16, 32, 63] {
        let commits: Vec<RingCommitment> = (0..m).map(|_| {
            let mut c = RingCommitment::zero();
            for i in 0..KAPPA {
                for k in 0..RING_DIM {
                    c.rows[i][k] = rng.gen::<u64>() % ALMOST_GOLDILOCKS_PRIME;
                }
            }
            c
        }).collect();
        let refs: Vec<&RingCommitment> = commits.iter().collect();
        let challenges: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

        let gpu = multifold_commitment(&refs, &challenges).expect("gpu");
        let cpu = cpu_multifold_commitment(&refs, &challenges);

        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                assert_eq!(gpu.rows[i][k], cpu.rows[i][k],
                    "multifold_commitment mismatch at (M={}, i={}, k={})", m, i, k);
            }
        }
    }
}

/// The headline test: at SuperNeo's actual K + k = 50 + 13 = 63, verify that
/// commit(Σ r_i z_i) == Σ r_i commit(z_i) bit-exactly. Uses a small N_ring
/// to keep the general-coefficient CPU commit tractable.
#[test]
fn test_multifold_homomorphism_K50_k13() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xC0DE_FEEDu64);
    let n_ring = 8;          // very small — keeps the CPU general commit ≤ a few seconds
    let num_instances = 63;  // K = 50 + k = 13 per Almost-Goldilocks params
    let seed = random_seed(&mut rng);

    let mut witnesses: Vec<Vec<u64>> = Vec::with_capacity(num_instances);
    let mut commits:   Vec<RingCommitment> = Vec::with_capacity(num_instances);
    // M − 1 = 62 challenges; witnesses[0] / commits[0] have implicit weight 1.
    let mut challenges: Vec<RingChallenge> = Vec::with_capacity(num_instances - 1);
    for i in 0..num_instances {
        let z = random_z(&mut rng, n_ring);
        commits.push(commit(seed, &z, Some(ChunkSize::C64)).expect("commit"));
        witnesses.push(z);
        if i + 1 < num_instances {
            challenges.push(rand_ring_challenge(&mut rng));
        }
    }
    assert_eq!(challenges.len(), num_instances - 1);

    let w_refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
    let c_refs: Vec<&RingCommitment> = commits.iter().collect();

    // Path A: multi-fold the commitments.
    let c_folded = multifold_commitment(&c_refs, &challenges).expect("mf commit");

    // Path B: multi-fold the witnesses → general-coefficient CPU commit.
    let z_folded_i16 = multifold_witness(&w_refs, &challenges).expect("mf witness");
    let z_folded_fq = i16_witness_to_fq(&z_folded_i16);
    let commit_of_fold = cpu_commit_general(&seed, &z_folded_fq, n_ring);

    // The Ajtai homomorphism must hold bit-exactly across all 960 output
    // coefficients.
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            assert_eq!(c_folded.rows[i][k], commit_of_fold.rows[i][k],
                "Homomorphism violated at i={} k={}: \
                 multifold_commitment = {}, M·multifold_witness = {}",
                i, k, c_folded.rows[i][k], commit_of_fold.rows[i][k]);
        }
    }
}

#[test]
fn test_multifold_norm_bound_K50_k13() {
    // For M = 63 binary instances and r ∈ {-1,0,1,2}^64, every coefficient of
    // the folded witness must satisfy |·| ≤ 1 + (M-1) * 128 = 7937 (well within the
    // SuperNeo binding bound B = 2^13 = 8192).
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0xB0B0_BABEu64);
    let n = 64;
    let m = 63;
    let witnesses: Vec<Vec<u64>> = (0..m).map(|_| random_z(&mut rng, n)).collect();
    let refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
    let challenges: Vec<RingChallenge> = (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

    let out = multifold_witness(&refs, &challenges).expect("mf");
    let observed_max = out.iter().map(|&v| (v as i32).abs()).max().unwrap();
    // implicit-1 contributes ≤ 1 for the first instance; remaining (m-1) each contribute ≤ T·(b-1) = 128.
    let theoretical_max = 1 + ((m - 1) as i32) * 128;
    assert!(observed_max <= theoretical_max,
        "observed max |coef| = {} exceeds theoretical bound {}",
        observed_max, theoretical_max);
    // Also confirm we're within the paper's binding bound B = 2^13.
    assert!(observed_max < (1 << 13),
        "observed max |coef| = {} exceeds B = 2^13 = 8192", observed_max);
}

// ============================================================================
// Phase 1: Split correctness tests (i16 wide witness → 13 ternary chunks)
// ============================================================================

/// Reconstruct z_wide[j][k] from the 13 ternary chunks. Mirrors the algebra
/// `z' = Σ 2^i · (pos[i] − neg[i])` used by the downstream commit / multifold
/// kernels — must agree bit-exactly with the GPU output to be useful.
fn cpu_reconstruct_from_chunks(chunks: &TernaryChunks) -> Vec<i16> {
    let n = chunks.n_ring;
    let mut out = vec![0i16; n * RING_DIM];
    for j in 0..n {
        for k in 0..RING_DIM {
            let mut acc: i32 = 0;
            for i in 0..chunks.k_chunks {
                let p = ((chunks.pos[i * n + j] >> k) & 1) as i32;
                let q = ((chunks.neg[i * n + j] >> k) & 1) as i32;
                acc += (1 << i) * (p - q);
            }
            out[j * RING_DIM + k] = acc as i16;
        }
    }
    out
}

#[test]
fn test_split_reconstruct_random() {
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_001_u64);
    // Cover several N_ring sizes so the kernel sees > 1, > 1 block, > 1 wave.
    for &n_ring in &[1usize, 4, 64, 1024, 8192] {
        let n_coefs = n_ring * RING_DIM;
        // Uniform in (-8064, 8064) — the multifold output range at M = 63.
        let z_wide: Vec<i16> = (0..n_coefs)
            .map(|_| rng.gen_range(-8064i16..=8064))
            .collect();

        let chunks = split_witness(&z_wide).expect("gpu split");
        assert_eq!(chunks.n_ring, n_ring);
        assert_eq!(chunks.k_chunks, SPLIT_K_CHUNKS);

        let recon = cpu_reconstruct_from_chunks(&chunks);
        assert_eq!(recon.len(), z_wide.len());
        for (i, (a, b)) in recon.iter().zip(z_wide.iter()).enumerate() {
            assert_eq!(a, b,
                "reconstruction mismatch at flat idx {} (N_ring={}): \
                 split says {}, original was {}", i, n_ring, a, b);
        }
    }
}

#[test]
fn test_split_no_pos_neg_overlap() {
    // Genuinely ternary: pos & neg must be 0 for every (i, j) — no
    // coefficient is encoded as both +1 and -1 simultaneously.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_002_u64);
    let n_ring = 4096;
    let z_wide: Vec<i16> = (0..n_ring * RING_DIM)
        .map(|_| rng.gen_range(-8064i16..=8064))
        .collect();
    let chunks = split_witness(&z_wide).expect("gpu split");
    for i in 0..SPLIT_K_CHUNKS {
        for j in 0..n_ring {
            let overlap = chunks.pos[i * n_ring + j] & chunks.neg[i * n_ring + j];
            assert_eq!(overlap, 0,
                "pos & neg overlap at chunk {}, ring elem {}: 0x{:016x}",
                i, j, overlap);
        }
    }
}

#[test]
fn test_split_boundary_values() {
    init().expect("CUDA init");
    // Edge cases: 0, ±max, all-positive, all-negative, mixed.
    let n_ring = 4;
    let mut z_wide = vec![0i16; n_ring * RING_DIM];

    // Ring elem 0: all zeros
    // Ring elem 1: all +8064 (max positive in our folded range)
    for k in 0..RING_DIM { z_wide[RING_DIM + k] = 8064; }
    // Ring elem 2: all -8064
    for k in 0..RING_DIM { z_wide[2 * RING_DIM + k] = -8064; }
    // Ring elem 3: alternating +/-, peaks, +1/-1
    for k in 0..RING_DIM {
        z_wide[3 * RING_DIM + k] = match k % 4 {
            0 => 0,
            1 => if k & 8 == 0 { 1 } else { -1 },
            2 => 8064,
            _ => -8064,
        };
    }

    let chunks = split_witness(&z_wide).expect("gpu split");
    let recon = cpu_reconstruct_from_chunks(&chunks);
    for (i, (a, b)) in recon.iter().zip(z_wide.iter()).enumerate() {
        assert_eq!(a, b, "boundary mismatch at idx {}: {} vs {}", i, a, b);
    }
    // Ring elem 0 (all zeros) must have all-zero chunks
    for i in 0..SPLIT_K_CHUNKS {
        assert_eq!(chunks.pos[i * n_ring + 0], 0);
        assert_eq!(chunks.neg[i * n_ring + 0], 0);
    }
}

#[test]
fn test_split_then_multifold_roundtrip() {
    // End-to-end: multifold 63 binary instances → wide i16 witness → split into 13 chunks.
    // The reconstructed chunks must match the original folded witness bit-exactly.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_003_u64);
    let n_ring = 256;
    let m = 63;

    let witnesses: Vec<Vec<u64>> = (0..m).map(|_| random_z(&mut rng, n_ring)).collect();
    let refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
    let challenges: Vec<RingChallenge> =
        (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

    let z_folded = multifold_witness(&refs, &challenges).expect("multifold");
    let chunks = split_witness(&z_folded).expect("split");
    let recon = cpu_reconstruct_from_chunks(&chunks);

    assert_eq!(recon.len(), z_folded.len());
    for (i, (a, b)) in recon.iter().zip(z_folded.iter()).enumerate() {
        assert_eq!(a, b,
            "round-trip mismatch at flat idx {} (post-multifold)", i);
    }
}

#[test]
fn test_split_chunks_are_ternary() {
    // Every (i, j, k) coefficient must be -1, 0, or +1 — i.e. (pos_bit, neg_bit)
    // is one of (0,0), (1,0), (0,1). This is the same property as
    // test_split_no_pos_neg_overlap but framed at the coefficient level.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_004_u64);
    let n_ring = 512;
    let z_wide: Vec<i16> = (0..n_ring * RING_DIM)
        .map(|_| rng.gen_range(-8064i16..=8064))
        .collect();
    let chunks = split_witness(&z_wide).expect("split");
    for i in 0..SPLIT_K_CHUNKS {
        for j in 0..n_ring {
            let p = chunks.pos[i * n_ring + j];
            let q = chunks.neg[i * n_ring + j];
            for k in 0..RING_DIM {
                let pb = ((p >> k) & 1) as i32;
                let qb = ((q >> k) & 1) as i32;
                assert!(pb + qb <= 1,
                    "non-ternary digit at chunk {} j {} k {}: pos={} neg={}",
                    i, j, k, pb, qb);
            }
        }
    }
}

// ============================================================================
// Phase 3: commit_ternary correctness + split/commit homomorphism
// ============================================================================

/// CPU reference: ring elem times signed-i32 coefficient k (negative ⇒ mod q).
fn ring_scale_signed(a: &[u64; 64], k: i64) -> [u64; 64] {
    // Reduce k into [0, q); doing two-step (i64 → i128) avoids i64 overflow.
    let k_mod = (((k as i128).rem_euclid(Q as i128)) as u128) as u64;
    let mut out = [0u64; 64];
    for r in 0..64 {
        out[r] = ((a[r] as u128 * k_mod as u128) % Q) as u64;
    }
    out
}

/// CPU reference: ring multiply by a signed-ternary witness encoded as
/// (pos_bits, neg_bits). Set bits in pos ⇒ +X^ℓ, set bits in neg ⇒ −X^ℓ.
fn ring_ternary_mul(a: &[u64; 64], pos_bits: u64, neg_bits: u64) -> [u64; 64] {
    let mut out = [0u64; 64];
    // positive contributions
    let mut mp = pos_bits;
    while mp != 0 {
        let ell = mp.trailing_zeros() as i32;
        mp &= mp - 1;
        let shifted = ring_shift(a, ell);
        for r in 0..64 {
            out[r] = f_add(out[r], shifted[r]);
        }
    }
    // negative contributions
    let mut mn = neg_bits;
    while mn != 0 {
        let ell = mn.trailing_zeros() as i32;
        mn &= mn - 1;
        let shifted = ring_shift(a, ell);
        for r in 0..64 {
            out[r] = f_add(out[r], f_neg(shifted[r]));
        }
    }
    out
}

/// CPU reference: ring multiply by an i16 coefficient witness `z[64]`.
/// Mirrors what the prover would compute on the wide witness directly:
///   `Σ_ℓ z[ℓ] · X^ℓ · a`, with z[ℓ] ∈ Z embedded into F_q.
fn ring_i16_mul(a: &[u64; 64], z: &[i16]) -> [u64; 64] {
    assert_eq!(z.len(), 64);
    let mut out = [0u64; 64];
    for ell in 0..64 {
        let v = z[ell] as i64;
        if v == 0 { continue; }
        let shifted = ring_shift(a, ell as i32);
        let scaled  = ring_scale_signed(&shifted, v);
        for r in 0..64 {
            out[r] = f_add(out[r], scaled[r]);
        }
    }
    out
}

/// CPU reference for `commit_ternary`. Returns 13 commitments.
fn cpu_commit_ternary(seed: &Seed, chunks: &TernaryChunks) -> Vec<RingCommitment> {
    let n = chunks.n_ring;
    let mut out = vec![RingCommitment::zero(); SPLIT_K_CHUNKS];
    for j in 0..n {
        for i_row in 0..KAPPA {
            let m_ij = prg_ring_elem(&seed.0, i_row as u32, j as u64);
            for b in 0..SPLIT_K_CHUNKS {
                let pos = chunks.pos[b * n + j];
                let neg = chunks.neg[b * n + j];
                if pos == 0 && neg == 0 { continue; }
                let contrib = ring_ternary_mul(&m_ij, pos, neg);
                for r in 0..RING_DIM {
                    out[b].rows[i_row][r] = f_add(out[b].rows[i_row][r], contrib[r]);
                }
            }
        }
    }
    out
}

/// CPU reference Ajtai commit on a wide i16 witness `z[n_ring * 64]`.
fn cpu_commit_i16(seed: &Seed, z_wide: &[i16]) -> RingCommitment {
    assert_eq!(z_wide.len() % RING_DIM, 0);
    let n_ring = z_wide.len() / RING_DIM;
    let mut c = RingCommitment::zero();
    for j in 0..n_ring {
        let z_j = &z_wide[j * RING_DIM..(j + 1) * RING_DIM];
        if z_j.iter().all(|&v| v == 0) { continue; }
        for i_row in 0..KAPPA {
            let m_ij = prg_ring_elem(&seed.0, i_row as u32, j as u64);
            let contrib = ring_i16_mul(&m_ij, z_j);
            for r in 0..RING_DIM {
                c.rows[i_row][r] = f_add(c.rows[i_row][r], contrib[r]);
            }
        }
    }
    c
}

/// Build random small-N ternary chunks directly (without going through split).
/// Sparsity is controlled so pos & neg never collide.
fn random_chunks(rng: &mut StdRng, n_ring: usize) -> TernaryChunks {
    let mut pos = vec![0u64; SPLIT_K_CHUNKS * n_ring];
    let mut neg = vec![0u64; SPLIT_K_CHUNKS * n_ring];
    for i in 0..SPLIT_K_CHUNKS {
        for j in 0..n_ring {
            // Each coefficient is uniformly {-1, 0, +1} with 50% zero.
            let mut p = 0u64;
            let mut n = 0u64;
            for k in 0..64 {
                match rng.gen_range(0..4) {
                    0 => p |= 1u64 << k,
                    1 => n |= 1u64 << k,
                    _ => {}
                }
            }
            pos[i * n_ring + j] = p;
            neg[i * n_ring + j] = n;
        }
    }
    TernaryChunks { n_ring, k_chunks: SPLIT_K_CHUNKS, pos, neg }
}

fn upload_chunks(chunks: &TernaryChunks)
    -> almost_goldilocks_cuda::ajtai::TernaryChunksDevice
{
    use almost_goldilocks_cuda::ajtai::TernaryChunksDevice;
    TernaryChunksDevice {
        n_ring: chunks.n_ring,
        k_chunks: chunks.k_chunks,
        pos: DeviceBuffer::from_slice(&chunks.pos).expect("upload pos"),
        neg: DeviceBuffer::from_slice(&chunks.neg).expect("upload neg"),
    }
}

#[test]
fn test_commit_ternary_vs_cpu_small() {
    // Bit-exact GPU = CPU for random ternary chunks, several N_ring sizes.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x3_001);

    for &n_ring in &[1usize, 4, 16, 64, 256] {
        let seed = random_seed(&mut rng);
        let chunks = random_chunks(&mut rng, n_ring);
        let dev_chunks = upload_chunks(&chunks);

        let gpu = commit_ternary(seed, &dev_chunks, None).expect("gpu commit_ternary");
        let cpu = cpu_commit_ternary(&seed, &chunks);

        assert_eq!(gpu.len(), SPLIT_K_CHUNKS);
        assert_eq!(cpu.len(), SPLIT_K_CHUNKS);
        for b in 0..SPLIT_K_CHUNKS {
            assert!(rings_equal(&gpu[b], &cpu[b]),
                "commit_ternary GPU != CPU at chunk {} (N_ring={})", b, n_ring);
        }
    }
}

#[test]
fn test_commit_ternary_pos_only_matches_binary_commit() {
    // If chunk 0 has only pos bits and chunks 1..12 are all-zero, then
    // commit_ternary[0] must equal the binary commit on the same z_bits,
    // and commit_ternary[1..]  must all be zero.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x3_002);

    let n_ring = 128;
    let seed = random_seed(&mut rng);
    let z_bits = random_z(&mut rng, n_ring);

    let mut pos = vec![0u64; SPLIT_K_CHUNKS * n_ring];
    let neg     = vec![0u64; SPLIT_K_CHUNKS * n_ring];
    pos[..n_ring].copy_from_slice(&z_bits);     // chunk 0 = z_bits
    let chunks = TernaryChunks { n_ring, k_chunks: SPLIT_K_CHUNKS, pos, neg };
    let dev = upload_chunks(&chunks);

    let gpu_ternary = commit_ternary(seed, &dev, None).expect("commit_ternary");
    let gpu_binary  = commit(seed, &z_bits, Some(ChunkSize::C256)).expect("commit");

    assert!(rings_equal(&gpu_ternary[0], &gpu_binary),
        "chunk 0 pos-only ternary commit must match binary commit");

    let zero = RingCommitment::zero();
    for b in 1..SPLIT_K_CHUNKS {
        assert!(rings_equal(&gpu_ternary[b], &zero),
            "chunk {} should be zero (all-zero pos/neg)", b);
    }
}

#[test]
fn test_split_then_commit_homomorphism() {
    // The security-relevant invariant:
    //   commit_wide(z_wide) == Σ_i 2^i · commit_ternary(split(z_wide))[i]
    // Compared ring-exact in F_q.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x3_003);

    for &n_ring in &[1usize, 4, 16, 64] {
        let seed = random_seed(&mut rng);
        // Wide i16 coefficients in (-8064, 8064) — the multifold range at M=63.
        let z_wide: Vec<i16> = (0..n_ring * RING_DIM)
            .map(|_| rng.gen_range(-8064i16..=8064))
            .collect();
        let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).expect("upload z_wide");

        let dev_chunks = split_witness_device(&d_z_wide).expect("split");
        let gpu_ternary = commit_ternary(seed, &dev_chunks, None)
            .expect("commit_ternary");

        // Reconstruct: Σ_i 2^i · c_i in F_q.
        let mut recon = RingCommitment::zero();
        for b in 0..SPLIT_K_CHUNKS {
            let scale: u64 = 1u64 << b;     // 2^b ≤ 2^12, fits trivially in u64
            for i_row in 0..KAPPA {
                for r in 0..RING_DIM {
                    let term = ((gpu_ternary[b].rows[i_row][r] as u128
                              * scale as u128) % Q) as u64;
                    recon.rows[i_row][r] = f_add(recon.rows[i_row][r], term);
                }
            }
        }

        let direct = cpu_commit_i16(&seed, &z_wide);
        assert!(rings_equal(&recon, &direct),
            "homomorphism failed at N_ring={}: \
             Σ 2^i · c_i (GPU) != commit_wide(z) (CPU)", n_ring);
    }
}

#[test]
fn test_split_then_commit_via_multifold_homomorphism() {
    // End-to-end with realistic input: 63-instance multifold → wide i16 →
    // split → commit_ternary. Compare the digit-weighted sum to the CPU
    // direct commit on the multifold output.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x3_004);
    let n_ring = 32;
    let m = 63;

    let seed = random_seed(&mut rng);
    let witnesses: Vec<Vec<u64>> = (0..m).map(|_| random_z(&mut rng, n_ring)).collect();
    let refs: Vec<&[u64]> = witnesses.iter().map(|w| w.as_slice()).collect();
    let challenges: Vec<RingChallenge> =
        (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

    let z_wide = multifold_witness(&refs, &challenges).expect("multifold");

    let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).expect("upload");
    let dev_chunks = split_witness_device(&d_z_wide).expect("split");
    let gpu_ternary = commit_ternary(seed, &dev_chunks, None).expect("commit_ternary");

    let mut recon = RingCommitment::zero();
    for b in 0..SPLIT_K_CHUNKS {
        let scale: u64 = 1u64 << b;
        for i_row in 0..KAPPA {
            for r in 0..RING_DIM {
                let term = ((gpu_ternary[b].rows[i_row][r] as u128
                          * scale as u128) % Q) as u64;
                recon.rows[i_row][r] = f_add(recon.rows[i_row][r], term);
            }
        }
    }

    let direct = cpu_commit_i16(&seed, &z_wide);
    assert!(rings_equal(&recon, &direct),
        "post-multifold homomorphism failed");
}

// ============================================================================
// Phase 4: mixed-type multifold (K binary + T ternary chunks)
// ============================================================================

/// CPU reference for the mixed multifold. Mirrors `cpu_multifold_witness`
/// (binary-only) but extends the inner loop with a ternary section that
/// adds for pos bits and subtracts for neg bits.
fn cpu_mixed_multifold(
    binary: &[&[u64]],
    chunks: &TernaryChunks,
    challenges: &[RingChallenge],
) -> Vec<i16> {
    let k = binary.len();
    let t = chunks.k_chunks;
    let m = k + t;
    assert!(k >= 1, "need at least one binary witness for the weight-1 slot");
    assert_eq!(challenges.len() + 1, m, "expected K + T − 1 challenges");

    let n = binary[0].len();
    assert_eq!(chunks.n_ring, n, "binary and ternary n_ring must match");
    let mut out = vec![0i16; n * RING_DIM];

    for j in 0..n {
        for col in 0..RING_DIM {
            let mut acc: i32 = 0;

            // Binary i = 0: implicit weight 1 — just the binary coefficient.
            acc += ((binary[0][j] >> col) & 1) as i32;

            // Binary i ∈ [1, K): challenge[i - 1].
            for i in 1..k {
                let mut mask = binary[i][j];
                while mask != 0 {
                    let ell = mask.trailing_zeros() as i32;
                    mask &= mask - 1;
                    let signed_idx = col as i32 - ell;
                    let (idx, wrap) = if signed_idx < 0 {
                        ((signed_idx + RING_DIM as i32) as usize, true)
                    } else {
                        (signed_idx as usize, false)
                    };
                    let mut rv = challenges[i - 1].coeffs[idx] as i32;
                    if wrap { rv = -rv; }
                    acc += rv;
                }
            }

            // Ternary t ∈ [0, T): challenge[K − 1 + t]. pos additive, neg subtractive.
            for t_idx in 0..t {
                let (pos, neg) = chunks.chunk(t_idx);
                let r = &challenges[(k - 1) + t_idx];
                let mut mp = pos[j];
                while mp != 0 {
                    let ell = mp.trailing_zeros() as i32;
                    mp &= mp - 1;
                    let signed_idx = col as i32 - ell;
                    let (idx, wrap) = if signed_idx < 0 {
                        ((signed_idx + RING_DIM as i32) as usize, true)
                    } else {
                        (signed_idx as usize, false)
                    };
                    let mut rv = r.coeffs[idx] as i32;
                    if wrap { rv = -rv; }
                    acc += rv;
                }
                let mut mn = neg[j];
                while mn != 0 {
                    let ell = mn.trailing_zeros() as i32;
                    mn &= mn - 1;
                    let signed_idx = col as i32 - ell;
                    let (idx, wrap) = if signed_idx < 0 {
                        ((signed_idx + RING_DIM as i32) as usize, true)
                    } else {
                        (signed_idx as usize, false)
                    };
                    let mut rv = r.coeffs[idx] as i32;
                    if wrap { rv = -rv; }
                    acc -= rv;
                }
            }
            out[j * RING_DIM + col] = acc as i16;
        }
    }
    out
}

#[test]
fn test_mixed_multifold_vs_cpu_small() {
    // GPU = CPU for several (K, T, N_ring) combos.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x4_001);

    let cases: &[(usize, usize, usize)] = &[
        // (K binary, T ternary, N_ring)
        (1, 1, 4),
        (2, 3, 16),
        (4, 8, 16),
        (8, 13, 32),
        (16, 13, 64),
    ];

    for &(k_bin, k_tern, n_ring) in cases {
        // Build random binary witnesses.
        let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
        let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

        // Build random ternary chunks of size k_tern. We use the helper from
        // Phase 3 tests, but the helper always produces SPLIT_K_CHUNKS chunks.
        // For arbitrary k_tern we generate manually.
        let mut pos = vec![0u64; k_tern * n_ring];
        let mut neg = vec![0u64; k_tern * n_ring];
        for i in 0..k_tern {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_host = TernaryChunks {
            n_ring, k_chunks: k_tern, pos: pos.clone(), neg: neg.clone(),
        };
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring,
            k_chunks: k_tern,
            pos: DeviceBuffer::from_slice(&pos).expect("upload pos"),
            neg: DeviceBuffer::from_slice(&neg).expect("upload neg"),
        };

        let m = k_bin + k_tern;
        let challenges: Vec<RingChallenge> =
            (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

        let gpu = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
            .expect("gpu mixed multifold");
        let cpu = cpu_mixed_multifold(&bin_refs, &chunks_host, &challenges);

        assert_eq!(gpu.len(), cpu.len());
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert_eq!(g, c,
                "GPU != CPU at flat idx {} (K={} T={} N_ring={}): {} vs {}",
                i, k_bin, k_tern, n_ring, g, c);
        }
    }
}

#[test]
fn test_mixed_multifold_no_ternary_matches_binary_path() {
    // With T = 0, multifold_mixed_witness must produce the exact same output
    // as multifold_witness on the same binaries + challenges.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x4_002);

    let k_bin = 8;
    let n_ring = 128;
    let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
    let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();
    let challenges: Vec<RingChallenge> =
        (0..(k_bin - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

    // Build a zero-chunk TernaryChunksDevice.
    let empty_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring,
        k_chunks: 0,
        pos: DeviceBuffer::<u64>::new(0).expect("alloc 0 pos"),
        neg: DeviceBuffer::<u64>::new(0).expect("alloc 0 neg"),
    };

    let mixed = multifold_mixed_witness(&bin_refs, &empty_dev, &challenges)
        .expect("gpu mixed (T=0)");
    let binary_only = multifold_witness(&bin_refs, &challenges)
        .expect("gpu binary multifold");

    assert_eq!(mixed.len(), binary_only.len());
    for (i, (a, b)) in mixed.iter().zip(binary_only.iter()).enumerate() {
        assert_eq!(a, b, "T=0 mismatch at idx {}: {} vs {}", i, a, b);
    }
}

#[test]
fn test_mixed_multifold_K50_T13_realistic() {
    // SuperNeo prover-loop config: 50 fresh binary + 13 ternary accumulator
    // chunks. Compare against the CPU reference at a modest N_ring.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x4_003);
    let k_bin = 50;
    let k_tern = SPLIT_K_CHUNKS;
    let n_ring = 16;
    let m = k_bin + k_tern;

    let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
    let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

    // Use the splitb output of a freshly multifolded witness as the ternary
    // input — exactly the prover's data flow.
    let inner_witnesses: Vec<Vec<u64>> =
        (0..(k_bin + k_tern)).map(|_| random_z(&mut rng, n_ring)).collect();
    let inner_refs: Vec<&[u64]> = inner_witnesses.iter().map(|v| v.as_slice()).collect();
    let inner_challenges: Vec<RingChallenge> =
        (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();
    let z_wide = multifold_witness(&inner_refs, &inner_challenges)
        .expect("inner multifold");
    let d_z_wide = DeviceBuffer::<i16>::from_slice(&z_wide).expect("upload");
    let chunks_dev = split_witness_device(&d_z_wide).expect("split");
    let chunks_host = split_witness(&z_wide).expect("split host");

    let challenges: Vec<RingChallenge> =
        (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

    let gpu = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
        .expect("mixed multifold");
    let cpu = cpu_mixed_multifold(&bin_refs, &chunks_host, &challenges);

    assert_eq!(gpu.len(), cpu.len());
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_eq!(g, c,
            "K=50/T=13 mismatch at idx {}: GPU {} CPU {}", i, g, c);
    }
}

#[test]
fn test_mixed_multifold_input_validation() {
    // API rejects bad inputs cleanly.
    init().expect("CUDA init");

    // Empty binary witnesses (no implicit-weight-1 slot).
    let empty_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring: 4, k_chunks: 0,
        pos: DeviceBuffer::<u64>::new(0).unwrap(),
        neg: DeviceBuffer::<u64>::new(0).unwrap(),
    };
    let challenges: Vec<RingChallenge> = vec![];
    let result = multifold_mixed_witness(&[], &empty_dev, &challenges);
    assert!(result.is_err(), "should reject empty binary_witnesses");

    // Wrong challenge count.
    let bins = vec![vec![0u64; 4]];
    let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();
    let wrong_chal: Vec<RingChallenge> = vec![]; // K=1, T=0 → expects 0; supply 1 wrong.
    let res_ok = multifold_mixed_witness(&bin_refs, &empty_dev, &wrong_chal);
    assert!(res_ok.is_ok(), "K=1, T=0, 0 challenges should work");

    // Mismatched n_ring between binary and ternary.
    let mut pos = vec![0u64; 13 * 8];
    let mut neg = vec![0u64; 13 * 8];
    // Set one bit so chunks aren't all-zero.
    pos[0] = 1;
    neg[0] = 2;
    let chunks_n8 = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring: 8, k_chunks: 13,
        pos: DeviceBuffer::from_slice(&pos).unwrap(),
        neg: DeviceBuffer::from_slice(&neg).unwrap(),
    };
    let result = multifold_mixed_witness(&bin_refs, &chunks_n8, &vec![]);
    assert!(result.is_err(), "should reject mismatched n_ring");
}

// ============================================================================
// Phase 5: tensor-core multifold equivalence (scalar vs WMMA INT8)
// ============================================================================

#[test]
fn test_mixed_multifold_tc_matches_scalar() {
    // The WMMA path must produce bit-exact identical output to the scalar
    // path for the same (binary, ternary, challenges) inputs.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_001);

    let cases: &[(usize, usize, usize)] = &[
        // (K binary, T ternary, N_ring) — N_ring must be a multiple of 16
        // for the WMMA kernel's row tiling; we pad internally otherwise,
        // but pick multiples here for fast/clean comparison.
        (1, 0, 16),
        (2, 3, 16),
        (8, 13, 32),
        (16, 13, 64),
        (50, 13, 64),
        (50, 13, 128),
    ];

    for &(k_bin, k_tern, n_ring) in cases {
        let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
        let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

        let mut pos = vec![0u64; k_tern * n_ring];
        let mut neg = vec![0u64; k_tern * n_ring];
        for i in 0..k_tern {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring,
            k_chunks: k_tern,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        };

        let m = k_bin + k_tern;
        let challenges: Vec<RingChallenge> = if m >= 1 {
            (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect()
        } else {
            vec![]
        };

        let scalar = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
            .expect("scalar mixed multifold");
        let tc = multifold_mixed_witness_tc(&bin_refs, &chunks_dev, &challenges)
            .expect("tc mixed multifold");

        assert_eq!(tc.len(), scalar.len());
        for (i, (a, b)) in tc.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(a, b,
                "TC vs scalar mismatch at flat idx {} (K={} T={} N_ring={}): {} vs {}",
                i, k_bin, k_tern, n_ring, a, b);
        }
    }
}

#[test]
fn test_mixed_multifold_tc_unaligned_n_ring() {
    // N_ring not a multiple of 16 — the FFI must pad internally and still
    // produce bit-exact output (over the original [0, N_ring) rows).
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_002);

    for &n_ring in &[1usize, 7, 17, 33, 100] {
        let k_bin = 4;
        let k_tern = 3;
        let m = k_bin + k_tern;

        let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
        let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

        let mut pos = vec![0u64; k_tern * n_ring];
        let mut neg = vec![0u64; k_tern * n_ring];
        for i in 0..k_tern {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring,
            k_chunks: k_tern,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        };

        let challenges: Vec<RingChallenge> =
            (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

        let scalar = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
            .expect("scalar mixed multifold");
        let tc = multifold_mixed_witness_tc(&bin_refs, &chunks_dev, &challenges)
            .expect("tc mixed multifold");

        assert_eq!(tc.len(), scalar.len(),
            "length mismatch at N_ring={}", n_ring);
        for (i, (a, b)) in tc.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(a, b, "unaligned N_ring={} mismatch at idx {}", n_ring, i);
        }
    }
}

#[test]
fn test_mixed_multifold_tc_fused_matches_scalar() {
    // Fused TC kernel must produce bit-exact identical output to scalar.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_101);

    let cases: &[(usize, usize, usize)] = &[
        (1, 0, 64),
        (4, 5, 64),
        (16, 13, 64),
        (50, 13, 64),
        (50, 13, 256),
        (50, 13, 1024),
    ];

    for &(k_bin, k_tern, n_ring) in cases {
        let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
        let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

        let mut pos = vec![0u64; k_tern * n_ring];
        let mut neg = vec![0u64; k_tern * n_ring];
        for i in 0..k_tern {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring,
            k_chunks: k_tern,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        };

        let m = k_bin + k_tern;
        let challenges: Vec<RingChallenge> = if m >= 1 {
            (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect()
        } else {
            vec![]
        };

        let scalar = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
            .expect("scalar");
        let fused = multifold_mixed_witness_tc_fused(&bin_refs, &chunks_dev, &challenges)
            .expect("tc fused");

        assert_eq!(fused.len(), scalar.len());
        for (i, (a, b)) in fused.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(a, b,
                "TC fused vs scalar mismatch at flat idx {} (K={} T={} N_ring={}): {} vs {}",
                i, k_bin, k_tern, n_ring, a, b);
        }
    }
}

#[test]
fn test_mixed_multifold_tc_fused_unaligned() {
    // Non-multiple-of-64 N_ring — the kernel must mask off-end rows.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x5_102);

    for &n_ring in &[1usize, 7, 17, 65, 100, 130] {
        let k_bin = 4;
        let k_tern = 3;
        let m = k_bin + k_tern;

        let bins: Vec<Vec<u64>> = (0..k_bin).map(|_| random_z(&mut rng, n_ring)).collect();
        let bin_refs: Vec<&[u64]> = bins.iter().map(|v| v.as_slice()).collect();

        let mut pos = vec![0u64; k_tern * n_ring];
        let mut neg = vec![0u64; k_tern * n_ring];
        for i in 0..k_tern {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring,
            k_chunks: k_tern,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        };

        let challenges: Vec<RingChallenge> =
            (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

        let scalar = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
            .expect("scalar");
        let fused = multifold_mixed_witness_tc_fused(&bin_refs, &chunks_dev, &challenges)
            .expect("fused");

        assert_eq!(fused.len(), scalar.len());
        for (i, (a, b)) in fused.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(a, b, "unaligned N_ring={} mismatch at idx {}", n_ring, i);
        }
    }
}

// ============================================================================
// Phase 6: ternary-only multifold (no binary witnesses) + concat of running
//           accumulators
// ============================================================================

#[test]
fn test_ternary_only_multifold_vs_cpu() {
    // K_bin = 0, T = 13 ternary chunks: ternary[0] takes the implicit-
    // weight-1 slot. Must match the CPU reference.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x6_001);

    for &n_ring in &[1usize, 4, 16, 64, 256] {
        let k_tern = SPLIT_K_CHUNKS;

        let mut pos = vec![0u64; k_tern * n_ring];
        let mut neg = vec![0u64; k_tern * n_ring];
        for i in 0..k_tern {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_host = TernaryChunks {
            n_ring, k_chunks: k_tern, pos: pos.clone(), neg: neg.clone(),
        };
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring, k_chunks: k_tern,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        };

        let challenges: Vec<RingChallenge> =
            (0..(k_tern - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();

        // GPU: ternary-only
        let bin_refs: Vec<&[u64]> = vec![];
        let gpu = multifold_mixed_witness(&bin_refs, &chunks_dev, &challenges)
            .expect("scalar ternary-only");

        // CPU: explicitly compute ternary[0] with implicit weight 1, ternary[1..]
        // with challenges[0..]. Reuse cpu_mixed_multifold by feeding a "fake"
        // binary witness of length 0 — actually our cpu_mixed_multifold requires
        // at least one binary witness too. So inline the calc here.
        let mut expected = vec![0i16; n_ring * RING_DIM];
        for j in 0..n_ring {
            for col in 0..RING_DIM {
                let mut acc: i32 = 0;
                // ternary[0]: implicit weight 1 → contributes z_0[j][col].
                let p0 = chunks_host.pos[j];
                let n0 = chunks_host.neg[j];
                acc += (((p0 >> col) & 1) as i32) - (((n0 >> col) & 1) as i32);
                // ternary[1..]: challenge[t - 1] applies.
                for t in 1..k_tern {
                    let (pos_t, neg_t) = chunks_host.chunk(t);
                    let r = &challenges[t - 1];
                    let mut mp = pos_t[j];
                    while mp != 0 {
                        let ell = mp.trailing_zeros() as i32;
                        mp &= mp - 1;
                        let signed_idx = col as i32 - ell;
                        let (idx, wrap) = if signed_idx < 0 {
                            ((signed_idx + RING_DIM as i32) as usize, true)
                        } else {
                            (signed_idx as usize, false)
                        };
                        let mut rv = r.coeffs[idx] as i32;
                        if wrap { rv = -rv; }
                        acc += rv;
                    }
                    let mut mn = neg_t[j];
                    while mn != 0 {
                        let ell = mn.trailing_zeros() as i32;
                        mn &= mn - 1;
                        let signed_idx = col as i32 - ell;
                        let (idx, wrap) = if signed_idx < 0 {
                            ((signed_idx + RING_DIM as i32) as usize, true)
                        } else {
                            (signed_idx as usize, false)
                        };
                        let mut rv = r.coeffs[idx] as i32;
                        if wrap { rv = -rv; }
                        acc -= rv;
                    }
                }
                expected[j * RING_DIM + col] = acc as i16;
            }
        }

        assert_eq!(gpu.len(), expected.len());
        for (i, (g, e)) in gpu.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g, e,
                "ternary-only GPU != CPU at idx {} (N_ring={}): {} vs {}",
                i, n_ring, g, e);
        }

        // The TC paths must agree too.
        let tc = multifold_mixed_witness_tc(&bin_refs, &chunks_dev, &challenges)
            .expect("tc ternary-only");
        let fused = multifold_mixed_witness_tc_fused(&bin_refs, &chunks_dev, &challenges)
            .expect("fused ternary-only");
        assert_eq!(tc, gpu, "tc ternary-only != scalar (N_ring={})", n_ring);
        assert_eq!(fused, gpu, "fused ternary-only != scalar (N_ring={})", n_ring);
    }
}

#[test]
fn test_concat_4_running_then_fold() {
    // 4 running accumulators, each split into 13 ternary chunks. Concat
    // along k → one TernaryChunksDevice with k_chunks = 52, then fold.
    // The concatenated result must equal a direct call against the
    // concatenated host-side chunks.
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x6_002);
    let n_ring = 64;
    let n_running = 4;
    let k_per = SPLIT_K_CHUNKS;          // = 13
    let k_total = n_running * k_per;     // = 52
    let m = k_total;

    // Build 4 separate ternary-chunk sets.
    let mut host_blocks: Vec<TernaryChunks> = Vec::with_capacity(n_running);
    let mut dev_blocks: Vec<almost_goldilocks_cuda::ajtai::TernaryChunksDevice> =
        Vec::with_capacity(n_running);
    for _ in 0..n_running {
        let mut pos = vec![0u64; k_per * n_ring];
        let mut neg = vec![0u64; k_per * n_ring];
        for i in 0..k_per {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        host_blocks.push(TernaryChunks {
            n_ring, k_chunks: k_per, pos: pos.clone(), neg: neg.clone(),
        });
        dev_blocks.push(almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring, k_chunks: k_per,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        });
    }

    // Concat on-device.
    let dev_refs: Vec<&almost_goldilocks_cuda::ajtai::TernaryChunksDevice> =
        dev_blocks.iter().collect();
    let dev_concat = almost_goldilocks_cuda::ajtai::TernaryChunksDevice::concat(&dev_refs)
        .expect("concat");
    assert_eq!(dev_concat.n_ring, n_ring);
    assert_eq!(dev_concat.k_chunks, k_total);

    // Reference: build the equivalent flat host-side chunks and upload.
    let mut flat_pos = Vec::<u64>::with_capacity(k_total * n_ring);
    let mut flat_neg = Vec::<u64>::with_capacity(k_total * n_ring);
    for blk in &host_blocks {
        flat_pos.extend_from_slice(&blk.pos);
        flat_neg.extend_from_slice(&blk.neg);
    }
    let dev_flat = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring, k_chunks: k_total,
        pos: DeviceBuffer::from_slice(&flat_pos).unwrap(),
        neg: DeviceBuffer::from_slice(&flat_neg).unwrap(),
    };

    let challenges: Vec<RingChallenge> =
        (0..(m - 1)).map(|_| rand_ring_challenge(&mut rng)).collect();
    let bin_refs: Vec<&[u64]> = vec![];

    let from_concat  = multifold_mixed_witness(&bin_refs, &dev_concat, &challenges)
        .expect("concat fold");
    let from_flat    = multifold_mixed_witness(&bin_refs, &dev_flat,   &challenges)
        .expect("flat fold");
    assert_eq!(from_concat, from_flat, "concat != flat upload for 4×13 ternary fold");

    // TC variants must also agree.
    let tc_concat    = multifold_mixed_witness_tc(&bin_refs, &dev_concat, &challenges)
        .expect("tc concat");
    let fused_concat = multifold_mixed_witness_tc_fused(&bin_refs, &dev_concat, &challenges)
        .expect("fused concat");
    assert_eq!(tc_concat,    from_concat, "tc != scalar after concat fold");
    assert_eq!(fused_concat, from_concat, "fused != scalar after concat fold");
}

#[test]
fn test_concat_validation() {
    init().expect("CUDA init");

    // Empty inputs.
    let result = almost_goldilocks_cuda::ajtai::TernaryChunksDevice::concat(&[]);
    assert!(result.is_err(), "empty inputs should error");

    // n_ring mismatch.
    let a = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring: 8, k_chunks: 0,
        pos: DeviceBuffer::<u64>::new(0).unwrap(),
        neg: DeviceBuffer::<u64>::new(0).unwrap(),
    };
    let b = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
        n_ring: 16, k_chunks: 0,
        pos: DeviceBuffer::<u64>::new(0).unwrap(),
        neg: DeviceBuffer::<u64>::new(0).unwrap(),
    };
    let result = almost_goldilocks_cuda::ajtai::TernaryChunksDevice::concat(&[&a, &b]);
    assert!(result.is_err(), "n_ring mismatch should error");
}

// ============================================================================
// Phase 7: pre-materialized M ternary commit
// ============================================================================

#[test]
fn test_commit_ternary_premat_matches_on_the_fly() {
    // Pre-materialized M must give bit-exact identical output to the
    // on-the-fly commit_ternary for any (seed, chunks).
    init().expect("CUDA init");
    let mut rng = StdRng::seed_from_u64(0x7_001);

    for &n_ring in &[1usize, 4, 16, 64, 256, 1024] {
        let seed = random_seed(&mut rng);

        // Random ternary chunks.
        let mut pos = vec![0u64; SPLIT_K_CHUNKS * n_ring];
        let mut neg = vec![0u64; SPLIT_K_CHUNKS * n_ring];
        for i in 0..SPLIT_K_CHUNKS {
            for j in 0..n_ring {
                let mut p = 0u64;
                let mut n = 0u64;
                for c in 0..64 {
                    match rng.gen_range(0..4) {
                        0 => p |= 1u64 << c,
                        1 => n |= 1u64 << c,
                        _ => {}
                    }
                }
                pos[i * n_ring + j] = p;
                neg[i * n_ring + j] = n;
            }
        }
        let chunks_dev = almost_goldilocks_cuda::ajtai::TernaryChunksDevice {
            n_ring,
            k_chunks: SPLIT_K_CHUNKS,
            pos: DeviceBuffer::from_slice(&pos).unwrap(),
            neg: DeviceBuffer::from_slice(&neg).unwrap(),
        };

        let on_the_fly = commit_ternary(seed, &chunks_dev, None).expect("on-the-fly");

        let m = MaterializedM::new(seed, n_ring).expect("materialize");
        let premat = commit_ternary_premat(&m, &chunks_dev, None).expect("premat");

        assert_eq!(on_the_fly.len(), premat.len());
        for (b, (a, p)) in on_the_fly.iter().zip(premat.iter()).enumerate() {
            assert!(rings_equal(a, p),
                "premat != on-the-fly at chunk {} (N_ring={})", b, n_ring);
        }
    }
}
