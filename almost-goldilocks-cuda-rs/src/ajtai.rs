//! Ajtai commitment over `R = F_q[X] / (X^64 + 1)`, almost-Goldilocks.
//!
//! Computes `c = M · z` where:
//!
//! - `q = 2^64 - 2^32 - 31` (almost-Goldilocks)
//! - `z` is a binary vector packed as `u64` bitmasks (64 binary coefficients
//!   per `u64`).
//! - `M` is a `15 × N` ring matrix derived deterministically from a 32-byte
//!   public seed via ChaCha8; never materialized.
//! - `c ∈ R^15`: 15 ring elements, 64 field coefficients each.
//!
//! See `cuda_almost_goldilocks/ajtai.md` for the full design.
//!
//! ## API
//!
//! - [`commit_batched`]: dense binary witness, `B` commitments sharing one `M`.
//!   `B ∈ {1, 2, 4, 8, 16}`. Recommended path for high throughput.
//! - [`commit`]: convenience wrapper for `B = 1`.
//! - [`commit_sparse`]: when within-block density is very low; iterates over
//!   a position list instead of scanning `z_bits`. Single witness only.
//!
//! The output of every variant is `RingCommitment` (15 ring elements per
//! commitment, 64 canonical `u64` coefficients each).

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;

/// Ring dimension `d` (= 64 in production).
pub const RING_DIM: usize = 64;

/// Number of output rows `κ` (= 15 in production).
pub const KAPPA: usize = 42;

/// One Ajtai commitment: 15 ring elements, each 64 canonical `u64` coefficients.
#[derive(Clone, Debug)]
pub struct RingCommitment {
    /// `rows[i][r]` = coefficient `r` of output row `i`, canonical in `[0, q)`.
    pub rows: [[u64; RING_DIM]; KAPPA],
}

impl RingCommitment {
    pub fn zero() -> Self {
        Self { rows: [[0u64; RING_DIM]; KAPPA] }
    }
}

/// 256-bit ChaCha8 key. Caller is responsible for deriving this from the
/// public seed via a cryptographic hash (e.g., SHA-256 of `seed || domain_tag`).
#[derive(Clone, Copy, Debug)]
pub struct Seed(pub [u32; 8]);

impl Seed {
    /// Derive a key from 32 raw bytes (little-endian per `u32`).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut k = [0u32; 8];
        for i in 0..8 {
            k[i] = u32::from_le_bytes([
                bytes[4*i],
                bytes[4*i + 1],
                bytes[4*i + 2],
                bytes[4*i + 3],
            ]);
        }
        Self(k)
    }
}

/// CHUNK_SIZE values supported by the kernel.
///
/// Smaller CHUNK ⇒ more blocks per stage-1 launch ⇒ better SM utilization at
/// small N. Larger CHUNK ⇒ less per-block overhead and smaller partial buffer
/// at the cost of less parallelism. The default ([`commit`] / `chunk = None`)
/// auto-selects based on `N` to keep `num_chunks` in the range
/// [~num_SMs, ~4 × num_SMs] on A100.
#[derive(Clone, Copy, Debug)]
pub enum ChunkSize {
    /// Tiny: best parallelism at log_n ∈ [12, 16].
    C64   = 64,
    /// Small: good at log_n ∈ [16, 20].
    C128  = 128,
    /// Mid-small.
    C256  = 256,
    /// Mid-range.
    C1024 = 1024,
    /// Large: minimizes partial-buffer memory, recommended at full `N = 2^27`.
    C4096 = 4096,
}

impl ChunkSize {
    fn as_int(self) -> c_int { self as c_int }
}

/// B-aware heuristic.
///
/// Different `B` configurations achieve different occupancies on A100:
/// - `B = 1` (64 threads/block, ~76 regs/thread): ≈ 11 blocks/SM, so we
///   want `num_chunks` ≳ 11 · num_SMs = ~1200 to saturate.
/// - `B ≥ 4` (≥ 256 threads/block, same regs/thread): ≈ 1 block/SM, so
///   we want `num_chunks` ≳ num_SMs = ~108, but going much past that
///   adds stage-2 reduce work without parallelism benefit.
///
/// We pick the smallest supported CHUNK that yields the target chunk count
/// for the given `b`. At very large `N`, we cap at C4096 to keep the
/// partial buffer bounded.
fn pick_default_chunk(n: u64, b: usize) -> ChunkSize {
    // A100 has 108 SMs. Multipliers chosen to land just above saturation
    // without flooding stage-2 reduce.
    let target_chunks: u64 = if b <= 2 { 1200 } else { 200 };
    // Smallest CHUNK such that num_chunks ≤ target_chunks; equivalently:
    // CHUNK ≥ ceil(n / target_chunks).
    let needed = (n + target_chunks - 1) / target_chunks;
    if      needed <=   64 { ChunkSize::C64   }
    else if needed <=  128 { ChunkSize::C128  }
    else if needed <=  256 { ChunkSize::C256  }
    else if needed <= 1024 { ChunkSize::C1024 }
    else                   { ChunkSize::C4096 }
}

fn supported_b(b: usize) -> Result<()> {
    match b {
        1 | 2 | 4 | 8 | 16 => Ok(()),
        _ => Err(CudaError::InvalidArgument(format!(
            "batch size {} is not supported (must be in {{1, 2, 4, 8, 16}})", b
        ))),
    }
}

/// Batched dense commit: `B` binary witnesses share the same matrix `M`.
///
/// Each entry in `witnesses` is one packed witness of length `N` `u64`s
/// (bit `ℓ` of `witnesses[b][j]` is `z[b][64*j + ℓ]`). All witnesses must
/// have the same length.
///
/// Returns `B` commitments. PRG cost is amortized across the batch — this
/// is the high-throughput path.
pub fn commit_batched(
    seed: Seed,
    witnesses: &[&[u64]],
    chunk: Option<ChunkSize>,
) -> Result<Vec<RingCommitment>> {
    let b = witnesses.len();
    supported_b(b)?;
    if b == 0 {
        return Err(CudaError::InvalidArgument("witnesses must not be empty".into()));
    }
    let n = witnesses[0].len() as u64;
    if n == 0 {
        return Err(CudaError::InvalidArgument("witness length must be > 0".into()));
    }
    for (i, w) in witnesses.iter().enumerate() {
        if w.len() as u64 != n {
            return Err(CudaError::InvalidArgument(format!(
                "witness {} has length {}, expected {}", i, w.len(), n
            )));
        }
    }
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(n, b));

    // Pack witnesses to a single contiguous [B * N] u64 host buffer, then upload.
    let mut packed = Vec::<u64>::with_capacity(b * (n as usize));
    for w in witnesses {
        packed.extend_from_slice(w);
    }
    let d_z = DeviceBuffer::<u64>::from_slice(&packed)?;
    let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
    let mut d_out = DeviceBuffer::<u64>::new(b * KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_commit_dense_batched_ffi(
            d_key.as_ptr(),
            d_z.as_ptr(),
            n,
            b as c_int,
            chunk.as_int(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut out = Vec::with_capacity(b);
    for bi in 0..b {
        let mut c = RingCommitment::zero();
        for i in 0..KAPPA {
            for r in 0..RING_DIM {
                c.rows[i][r] = flat[((bi * KAPPA) + i) * RING_DIM + r];
            }
        }
        out.push(c);
    }
    Ok(out)
}

/// [`commit_batched`] against an arbitrary column window of `M_max`.
///
/// `col_offset` is in ring elements. This is the binary-witness counterpart of
/// [`commit_wide`]'s offset argument: packing several leaves into one
/// commitment means committing each at its own window and ring-summing, which
/// the Ajtai map's linearity makes exact.
pub fn commit_batched_at(
    seed: Seed,
    witnesses: &[&[u64]],
    col_offset: u64,
    chunk: Option<ChunkSize>,
) -> Result<Vec<RingCommitment>> {
    let b = witnesses.len();
    supported_b(b)?;
    if b == 0 {
        return Err(CudaError::InvalidArgument("witnesses must not be empty".into()));
    }
    let n = witnesses[0].len() as u64;
    if n == 0 {
        return Err(CudaError::InvalidArgument("witness length must be > 0".into()));
    }
    for (i, w) in witnesses.iter().enumerate() {
        if w.len() as u64 != n {
            return Err(CudaError::InvalidArgument(format!(
                "witness {} has length {}, expected {}", i, w.len(), n
            )));
        }
    }
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(n, b));

    let mut packed = Vec::<u64>::with_capacity(b * (n as usize));
    for w in witnesses {
        packed.extend_from_slice(w);
    }
    let d_z = DeviceBuffer::<u64>::from_slice(&packed)?;
    let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
    let mut d_out = DeviceBuffer::<u64>::new(b * KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_commit_dense_batched_at_ffi(
            d_key.as_ptr(),
            d_z.as_ptr(),
            n,
            b as c_int,
            chunk.as_int(),
            col_offset,
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut out = Vec::with_capacity(b);
    for bi in 0..b {
        let mut c = RingCommitment::zero();
        for i in 0..KAPPA {
            for r in 0..RING_DIM {
                c.rows[i][r] = flat[((bi * KAPPA) + i) * RING_DIM + r];
            }
        }
        out.push(c);
    }
    Ok(out)
}

/// Single-witness dense commit. Convenience wrapper for `commit_batched` with
/// `B = 1`. For multiple commits sharing the same `seed`, prefer
/// [`commit_batched`] — PRG cost is amortized.
pub fn commit(
    seed: Seed,
    z_bits: &[u64],
    chunk: Option<ChunkSize>,
) -> Result<RingCommitment> {
    let mut out = commit_batched(seed, &[z_bits], chunk)?;
    Ok(out.pop().unwrap())
}

// ============================================================================
// Wide commit (full-width coefficients) with column offset
// ============================================================================

/// Commit a witness whose coefficients are arbitrary field elements, against
/// the column window `[col_offset, col_offset + n_ring)` of `M_max`.
///
/// This is the mask-commitment path for the masked-RLC opening: `U_l` is a
/// discrete Gaussian with ~36-bit coefficients, and committing it as 36 binary
/// planes via [`commit_batched`] re-runs the ChaCha8 matrix PRG 36 times over
/// the same columns. Here the PRG runs once and the inner loop is a modular
/// multiply.
///
/// `z_wide` must have length `n_ring * RING_DIM`, in column-major
/// `[j][coefficient]` order, each entry a canonical field element in `[0, q)`.
/// Signed values must be reduced into that range by the caller.
///
/// `col_offset` is in ring elements, not coefficients. `col_offset = 0`
/// reproduces the same matrix columns [`commit_batched`] uses, so a packed
/// commitment equals the ring-sum of its blocks committed at their own offsets.
pub fn commit_wide(
    seed: Seed,
    z_wide: &[u64],
    col_offset: u64,
    chunk: Option<ChunkSize>,
) -> Result<RingCommitment> {
    if z_wide.is_empty() {
        return Err(CudaError::InvalidArgument("z_wide must not be empty".into()));
    }
    if z_wide.len() % RING_DIM != 0 {
        return Err(CudaError::InvalidArgument(format!(
            "z_wide.len() {} not divisible by RING_DIM = {}", z_wide.len(), RING_DIM
        )));
    }
    let n_ring = (z_wide.len() / RING_DIM) as u64;
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(n_ring, 1));

    let d_z = DeviceBuffer::<u64>::from_slice(z_wide)?;
    let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
    let mut d_out = DeviceBuffer::<u64>::new(KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_commit_wide_ffi(
            d_key.as_ptr(),
            d_z.as_ptr(),
            n_ring,
            col_offset,
            chunk.as_int(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut c = RingCommitment::zero();
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            c.rows[i][r] = flat[i * RING_DIM + r];
        }
    }
    Ok(c)
}

/// Device-resident variant of [`commit_wide`]: the witness never leaves the
/// GPU. Used by the masked-RLC prover, where `U_l` is sampled on-device.
pub fn commit_wide_device(
    seed: Seed,
    d_z_wide: &DeviceBuffer<u64>,
    col_offset: u64,
    chunk: Option<ChunkSize>,
) -> Result<RingCommitment> {
    if d_z_wide.len() % RING_DIM != 0 || d_z_wide.is_empty() {
        return Err(CudaError::InvalidArgument(format!(
            "d_z_wide.len() {} must be a nonzero multiple of {}", d_z_wide.len(), RING_DIM
        )));
    }
    let n_ring = (d_z_wide.len() / RING_DIM) as u64;
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(n_ring, 1));

    let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
    let mut d_out = DeviceBuffer::<u64>::new(KAPPA * RING_DIM)?;
    let ret = unsafe {
        ffi::ajtai_commit_wide_ffi(
            d_key.as_ptr(),
            d_z_wide.as_ptr(),
            n_ring,
            col_offset,
            chunk.as_int(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut c = RingCommitment::zero();
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            c.rows[i][r] = flat[i * RING_DIM + r];
        }
    }
    Ok(c)
}

// ============================================================================
// Folding: additive homomorphism c1 + r·c2 = M·(z1 + r·z2)
// ============================================================================

/// A small-coefficient ring element used as a folding challenge.
///
/// Each coefficient must be in `{-1, 0, 1, 2}` (the SuperNeo almost-Goldilocks
/// parameter set). 64 coefficients total — one ring element of `R`.
#[derive(Clone, Copy, Debug)]
pub struct RingChallenge {
    pub coeffs: [i8; 64],
}

impl RingChallenge {
    /// Build from a 64-coefficient array, validating the range.
    pub fn new(coeffs: [i8; 64]) -> Result<Self> {
        for (i, &c) in coeffs.iter().enumerate() {
            if !(-1..=2).contains(&c) {
                return Err(CudaError::InvalidArgument(format!(
                    "RingChallenge.coeffs[{}] = {} is outside the {{-1, 0, 1, 2}} range",
                    i, c
                )));
            }
        }
        Ok(Self { coeffs })
    }

    /// Construct without checking (useful when coefficients are known-valid).
    pub fn from_coeffs_unchecked(coeffs: [i8; 64]) -> Self {
        Self { coeffs }
    }
}

/// Witness fold: produce `z1 + r · z2` as a 64-coefficient-per-element F_q
/// vector on the GPU.
///
/// `z1` and `z2` are binary witnesses (one `u64` packs 64 binary coefficients),
/// each of length `N_ring`. The output `dst` must have capacity
/// `N_ring * 64` and is written with canonical `F_q` values. Each output
/// coefficient has small absolute value (≤ 65) but is stored as a full `u64`
/// in canonical form so it can feed into subsequent field-arithmetic kernels.
///
/// To compute just `r · z2` (no `z1`), pass a zero-filled `z1`. The memory
/// cost is identical (this kernel is memory-bound on the output write).
pub fn fold_witness_device(
    z1: &DeviceBuffer<u64>,
    r: &RingChallenge,
    z2: &DeviceBuffer<u64>,
    dst: &mut DeviceBuffer<u64>,
) -> Result<()> {
    let n_ring = z1.len() as u64;
    if z2.len() as u64 != n_ring {
        return Err(CudaError::InvalidArgument(
            "z1 and z2 must have the same length".into(),
        ));
    }
    if dst.len() as u64 != n_ring * RING_DIM as u64 {
        return Err(CudaError::InvalidArgument(format!(
            "dst length {} != N_ring * 64 = {}",
            dst.len(),
            n_ring * RING_DIM as u64
        )));
    }
    let chunk = pick_default_chunk(n_ring, 1).as_int();
    let ret = unsafe {
        ffi::ajtai_fold_witness_ffi(
            z1.as_ptr(),
            z2.as_ptr(),
            r.coeffs.as_ptr(),
            n_ring,
            chunk,
            dst.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

/// Host-side convenience: uploads `z1`, `z2`, computes `z1 + r·z2`, downloads
/// the full F_q output. Allocates `N_ring * 64 * 8` bytes on the device and
/// the host — at large `N_ring` this is large (e.g., `N_ring = 2^20` is 512 MiB).
/// For repeated use, prefer [`fold_witness_device`].
pub fn fold_witness(
    z1: &[u64],
    r: &RingChallenge,
    z2: &[u64],
) -> Result<Vec<u64>> {
    let n_ring = z1.len();
    if z2.len() != n_ring {
        return Err(CudaError::InvalidArgument(
            "z1 and z2 must have the same length".into(),
        ));
    }
    let d_z1 = DeviceBuffer::<u64>::from_slice(z1)?;
    let d_z2 = DeviceBuffer::<u64>::from_slice(z2)?;
    let mut d_out = DeviceBuffer::<u64>::new(n_ring * RING_DIM)?;
    fold_witness_device(&d_z1, r, &d_z2, &mut d_out)?;
    crate::memory::synchronize()?;
    d_out.to_vec()
}

/// Commitment fold: compute `c1 + r · c2` ∈ R^15. Tiny (~15 × 64 field
/// elements) — runs in microseconds on A100.
pub fn fold_commitment(
    c1: &RingCommitment,
    r: &RingChallenge,
    c2: &RingCommitment,
) -> Result<RingCommitment> {
    // Flatten c1, c2 to row-major u64 arrays of length KAPPA * RING_DIM.
    let mut c1_flat = vec![0u64; KAPPA * RING_DIM];
    let mut c2_flat = vec![0u64; KAPPA * RING_DIM];
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            c1_flat[i * RING_DIM + k] = c1.rows[i][k];
            c2_flat[i * RING_DIM + k] = c2.rows[i][k];
        }
    }
    let d_c1 = DeviceBuffer::<u64>::from_slice(&c1_flat)?;
    let d_c2 = DeviceBuffer::<u64>::from_slice(&c2_flat)?;
    let mut d_out = DeviceBuffer::<u64>::new(KAPPA * RING_DIM)?;
    let ret = unsafe {
        ffi::ajtai_fold_commitment_ffi(
            d_c1.as_ptr(),
            d_c2.as_ptr(),
            r.coeffs.as_ptr(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;
    let flat = d_out.to_vec()?;
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            out.rows[i][k] = flat[i * RING_DIM + k];
        }
    }
    Ok(out)
}

// ============================================================================
// Multi-fold (K + k binary instances → 1 wide witness / 1 commitment)
//
// Per SuperNeo's Almost-Goldilocks parameters (Appendix B.1), K = 50 fresh
// CCS instances + k = 13 accumulator chunks are folded together each round.
// All M = K + k inputs are binary witnesses (one u64 packs 64 binary coefs).
//
// Folding form:
//
//     z'  =  z_0  +  Σ_{i=1..M-1}  r_i · z_i
//
// The first instance is anchored with implicit coefficient 1 (the constant
// ring element [1, 0, 0, ..., 0] ∈ R), so the caller supplies M − 1 random
// challenges, NOT M. The kernel sees a synthesized "constant-1" challenge
// in slot 0 internally.
//
// Output bound:  ||z'||_∞  ≤  1 + (M-1) · T · (b-1)  ≤  M · T · (b-1).
// At M = 63: |·| ≤ 8064, fits in i16 (range [-8192, 8191]).
// ============================================================================

/// The constant-1 ring element. `r_0 · z = z` regardless of `z`.
fn constant_one_challenge() -> RingChallenge {
    let mut c = [0i8; 64];
    c[0] = 1;
    RingChallenge::from_coeffs_unchecked(c)
}

/// Multi-fold witness:  `z' = z_0 + Σ_{i=1..M-1} r_i · z_i`  for `M` binary
/// witnesses and `M − 1` independent challenges.
///
/// `witnesses[i]` is a packed binary witness of length `N_ring` (one u64
/// packs 64 binary coefficients). `challenges[i]` is the matching challenge
/// for `witnesses[i + 1]` (i.e., the first instance `witnesses[0]` has
/// implicit weight 1).
///
/// `challenges` must have length `witnesses.len() − 1`. Returns a flat
/// `Vec<i16>` of length `N_ring * 64` with the folded witness coefficients.
///
/// For `M ≤ 511` the output is guaranteed to fit in `i16`.
pub fn multifold_witness(
    witnesses: &[&[u64]],
    challenges: &[RingChallenge],
) -> Result<Vec<i16>> {
    let m = witnesses.len();
    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "witnesses must be non-empty".into(),
        ));
    }
    if challenges.len() + 1 != m {
        return Err(CudaError::InvalidArgument(format!(
            "expected exactly witnesses.len() − 1 = {} challenges, got {} \
             (witnesses[0] has implicit weight 1)",
            m - 1, challenges.len()
        )));
    }
    if m > 511 {
        return Err(CudaError::InvalidArgument(format!(
            "num_instances {} would overflow i16 (max supported 511)", m
        )));
    }
    let n_ring = witnesses[0].len();
    for (i, w) in witnesses.iter().enumerate() {
        if w.len() != n_ring {
            return Err(CudaError::InvalidArgument(format!(
                "witnesses[{}] has length {}, expected {}", i, w.len(), n_ring
            )));
        }
    }
    if n_ring == 0 {
        return Ok(Vec::new());
    }

    // Pack witnesses as a single contiguous [M * N_ring] u64.
    let mut z_packed = Vec::<u64>::with_capacity(m * n_ring);
    for w in witnesses {
        z_packed.extend_from_slice(w);
    }
    let d_z = DeviceBuffer::<u64>::from_slice(&z_packed)?;

    // Pack challenges: synthesized constant-1 at slot 0, caller's challenges
    // at slots 1..M.
    let mut r_packed = Vec::<i8>::with_capacity(m * 64);
    let constant_one = constant_one_challenge();
    r_packed.extend_from_slice(&constant_one.coeffs);
    for r in challenges {
        r_packed.extend_from_slice(&r.coeffs);
    }
    let d_r = DeviceBuffer::<i8>::from_slice(&r_packed)?;

    let mut d_out = DeviceBuffer::<i16>::new(n_ring * RING_DIM)?;

    // Many small chunks for small N (more SM parallelism), bigger chunks at
    // larger N (amortize per-block overhead). The inner work per j is
    // O(M · popcount(z_i[j])) ≈ M · 32, which is heavy enough that we don't
    // need very large chunks to saturate.
    let chunk_size: u64 = if n_ring <= 64 {
        1
    } else if n_ring <= 4096 {
        4
    } else {
        16
    };

    let ret = unsafe {
        ffi::ajtai_multifold_witness_ffi(
            d_z.as_ptr(),
            d_r.as_ptr(),
            d_out.as_mut_ptr(),
            m as c_int,
            n_ring as u64,
            chunk_size,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;
    d_out.to_vec()
}

/// Mixed-type multi-fold:  fold `K` binary witnesses **plus** `T` ternary
/// chunks (typically the previous round's split-decomposed accumulator) in
/// one kernel.  Output is identical to feeding `K + T` total instances to
/// `multifold_witness` after manually unpacking the ternary chunks into
/// `i16`-coefficient witnesses, but stays entirely on-GPU and skips a
/// round-trip through host memory.
///
/// Convention:
///   * `binary_witnesses[0]` (or `ternary_chunks.chunk(0)` if `K = 0`) has
///     implicit weight 1 (synthesized constant-1 challenge prepended).
///   * The remaining `K + T − 1` instances each get one entry from
///     `challenges`, applied in order: binary first, ternary second.
///   * `challenges.len()` must equal `K + T − 1`.
///
/// Returns a `Vec<i16>` of length `n_ring · 64`. For `K + T ≤ 511` the
/// output fits in `i16`.
///
/// **Norm-growth invariant — the caller's responsibility.** The fold's
/// output norm is bounded by `1 + (M − 1) · T_chal · max|z_coef|`. To
/// stay inside a parameter set that decomposes the output back into `k`
/// chunks of base `b`, the caller must respect `(M − 1) · T_chal · (b − 1) < b^k`.
/// For SuperNeo Almost-Goldilocks (`T_chal = 128`, `b = 2`, `k = 13`,
/// `B = 8192`), this means `M ≤ 64` (and the standard `K = 50` + `k = 13`
/// fresh-plus-running configuration sits at `M = 63`). This API does
/// **not** enforce that — it only checks `M ≤ 511` so the i16 output
/// can't overflow.
pub fn multifold_mixed_witness(
    binary_witnesses: &[&[u64]],
    ternary_chunks: &TernaryChunksDevice,
    challenges: &[RingChallenge],
) -> Result<Vec<i16>> {
    let k_bin = binary_witnesses.len();
    let k_tern = ternary_chunks.k_chunks;
    let m = k_bin + k_tern;

    // Either binary[0] or (when k_bin == 0) ternary[0] holds the implicit-
    // weight-1 slot — same constant-1 challenge in r_all[0..64] either way.
    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "need at least one instance (binary or ternary)".into(),
        ));
    }
    if challenges.len() + 1 != m {
        return Err(CudaError::InvalidArgument(format!(
            "expected K + T − 1 = {} challenges, got {} \
             (K = {} binary, T = {} ternary; instance[0] has implicit weight 1)",
            m - 1, challenges.len(), k_bin, k_tern
        )));
    }
    if m > 511 {
        return Err(CudaError::InvalidArgument(format!(
            "num_instances {} would overflow i16 (max supported 511)", m
        )));
    }

    let n_ring = if k_bin > 0 { binary_witnesses[0].len() } else { ternary_chunks.n_ring };
    for (i, w) in binary_witnesses.iter().enumerate() {
        if w.len() != n_ring {
            return Err(CudaError::InvalidArgument(format!(
                "binary_witnesses[{}] has length {}, expected {}", i, w.len(), n_ring
            )));
        }
    }
    if k_bin > 0 && ternary_chunks.n_ring != n_ring {
        return Err(CudaError::InvalidArgument(format!(
            "ternary_chunks.n_ring = {} mismatches binary witnesses' n_ring = {}",
            ternary_chunks.n_ring, n_ring
        )));
    }
    if n_ring == 0 {
        return Ok(Vec::new());
    }

    // Pack binary witnesses (possibly empty) to a contiguous [K * N_ring] u64.
    let mut z_packed = Vec::<u64>::with_capacity(k_bin * n_ring);
    for w in binary_witnesses {
        z_packed.extend_from_slice(w);
    }
    let d_z_bin = DeviceBuffer::<u64>::from_slice(&z_packed)?;

    // Challenges: constant-1 for binary[0], then K + T − 1 caller challenges.
    let mut r_packed = Vec::<i8>::with_capacity(m * 64);
    r_packed.extend_from_slice(&constant_one_challenge().coeffs);
    for r in challenges {
        r_packed.extend_from_slice(&r.coeffs);
    }
    let d_r = DeviceBuffer::<i8>::from_slice(&r_packed)?;

    let mut d_out = DeviceBuffer::<i16>::new(n_ring * RING_DIM)?;

    // Inner work per j: K · 32 + T · 64 ≈ 50·32 + 13·64 = 2432 ops at typical K=50,
    // similar to binary multifold. Same chunk_size heuristic.
    let chunk_size: u64 = if n_ring <= 64 {
        1
    } else if n_ring <= 4096 {
        4
    } else {
        16
    };

    let ret = unsafe {
        ffi::ajtai_multifold_mixed_witness_ffi(
            d_z_bin.as_ptr(),
            ternary_chunks.pos.as_ptr(),
            ternary_chunks.neg.as_ptr(),
            d_r.as_ptr(),
            d_out.as_mut_ptr(),
            k_bin as c_int,
            k_tern as c_int,
            n_ring as u64,
            chunk_size,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;
    d_out.to_vec()
}

/// Tensor-core variant of [`multifold_mixed_witness`].
///
/// Reformulates the fold as a single INT8 matrix multiply
/// `out = z_mat @ R_mat` of shape `[N_ring, M·64] × [M·64, 64]` and
/// dispatches to A100 `mma.sync.m16n16k16.s8.s8.s32` tensor cores via
/// the CUDA WMMA API. Output is bit-exact identical to the scalar path.
///
/// At large `N_ring` this is the recommended path (~order-of-magnitude
/// speedup over the scalar `multifold_mixed_witness`). The scalar path is
/// retained for A/B comparison and small problems where the WMMA prep
/// cost (building `R` + expanding `z`) dominates.
///
/// Same convention as [`multifold_mixed_witness`]: `binary_witnesses[0]`
/// has implicit weight 1, and `challenges` has length
/// `binary_witnesses.len() + ternary_chunks.k_chunks − 1`.
///
/// Memory: this routine allocates a transient `[N_ring_padded, M·64]`
/// int8 device buffer (e.g. ~1 GB at `M = 63`, `N_ring = 2^18`). For
/// `N_ring` so large that this won't fit (e.g. `≥ 2^20`), call this in
/// `N_ring` chunks at the host level.
pub fn multifold_mixed_witness_tc(
    binary_witnesses: &[&[u64]],
    ternary_chunks: &TernaryChunksDevice,
    challenges: &[RingChallenge],
) -> Result<Vec<i16>> {
    let k_bin  = binary_witnesses.len();
    let k_tern = ternary_chunks.k_chunks;
    let m      = k_bin + k_tern;

    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "need at least one instance (binary or ternary)".into(),
        ));
    }
    if challenges.len() + 1 != m {
        return Err(CudaError::InvalidArgument(format!(
            "expected K + T − 1 = {} challenges, got {} \
             (K = {} binary, T = {} ternary; instance[0] has implicit weight 1)",
            m - 1, challenges.len(), k_bin, k_tern
        )));
    }
    if m > 511 {
        return Err(CudaError::InvalidArgument(format!(
            "num_instances {} would overflow i16 (max supported 511)", m
        )));
    }

    let n_ring = if k_bin > 0 { binary_witnesses[0].len() } else { ternary_chunks.n_ring };
    for (i, w) in binary_witnesses.iter().enumerate() {
        if w.len() != n_ring {
            return Err(CudaError::InvalidArgument(format!(
                "binary_witnesses[{}] has length {}, expected {}", i, w.len(), n_ring
            )));
        }
    }
    if k_bin > 0 && ternary_chunks.n_ring != n_ring {
        return Err(CudaError::InvalidArgument(format!(
            "ternary_chunks.n_ring = {} mismatches binary witnesses' n_ring = {}",
            ternary_chunks.n_ring, n_ring
        )));
    }
    if n_ring == 0 {
        return Ok(Vec::new());
    }

    let mut z_packed = Vec::<u64>::with_capacity(k_bin * n_ring);
    for w in binary_witnesses {
        z_packed.extend_from_slice(w);
    }
    let d_z_bin = DeviceBuffer::<u64>::from_slice(&z_packed)?;

    let mut r_packed = Vec::<i8>::with_capacity(m * 64);
    r_packed.extend_from_slice(&constant_one_challenge().coeffs);
    for r in challenges {
        r_packed.extend_from_slice(&r.coeffs);
    }
    let d_r = DeviceBuffer::<i8>::from_slice(&r_packed)?;

    let mut d_out = DeviceBuffer::<i16>::new(n_ring * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_multifold_mixed_witness_tc_ffi(
            d_z_bin.as_ptr(),
            ternary_chunks.pos.as_ptr(),
            ternary_chunks.neg.as_ptr(),
            d_r.as_ptr(),
            d_out.as_mut_ptr(),
            k_bin as c_int,
            k_tern as c_int,
            n_ring as u64,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;
    d_out.to_vec()
}

/// Fused tensor-core multifold: same WMMA matmul as
/// [`multifold_mixed_witness_tc`] but with `z` unpacked on-the-fly inside
/// the kernel instead of materialized into a dense `[N_ring, M·64]` int8
/// buffer. Eliminates the ~66 MB transient buffer at `log_n = 20` and the
/// associated `expand_z_kernel` launch.
///
/// Output is bit-exact identical to both the scalar
/// [`multifold_mixed_witness`] and the materialized
/// [`multifold_mixed_witness_tc`].
///
/// Recommended path at large `N_ring`. The materialized variant is
/// retained for direct A/B comparison and is sometimes useful when the
/// dense `z_mat` is needed for a downstream consumer.
pub fn multifold_mixed_witness_tc_fused(
    binary_witnesses: &[&[u64]],
    ternary_chunks: &TernaryChunksDevice,
    challenges: &[RingChallenge],
) -> Result<Vec<i16>> {
    let k_bin  = binary_witnesses.len();
    let k_tern = ternary_chunks.k_chunks;
    let m      = k_bin + k_tern;

    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "need at least one instance (binary or ternary)".into(),
        ));
    }
    if challenges.len() + 1 != m {
        return Err(CudaError::InvalidArgument(format!(
            "expected K + T − 1 = {} challenges, got {}",
            m - 1, challenges.len()
        )));
    }
    if m > 511 {
        return Err(CudaError::InvalidArgument(format!(
            "num_instances {} would overflow i16 (max supported 511)", m
        )));
    }

    let n_ring = if k_bin > 0 { binary_witnesses[0].len() } else { ternary_chunks.n_ring };
    for (i, w) in binary_witnesses.iter().enumerate() {
        if w.len() != n_ring {
            return Err(CudaError::InvalidArgument(format!(
                "binary_witnesses[{}] has length {}, expected {}", i, w.len(), n_ring
            )));
        }
    }
    if k_bin > 0 && ternary_chunks.n_ring != n_ring {
        return Err(CudaError::InvalidArgument(format!(
            "ternary_chunks.n_ring = {} mismatches binary witnesses' n_ring = {}",
            ternary_chunks.n_ring, n_ring
        )));
    }
    if n_ring == 0 {
        return Ok(Vec::new());
    }

    let timing = std::env::var("ZK4_TIMING_MF").is_ok();
    let t0 = std::time::Instant::now();
    // Pool-backed upload: per-witness direct writes into a pooled device
    // buffer — no host-side concat copy, no fresh cudaMalloc + synchronizing
    // cudaFree per group (same fix as the same-point packed-bits upload).
    let mut d_z_bin = crate::sumcheck_prover::pool_take((k_bin * n_ring).max(1))?;
    for (i, w) in binary_witnesses.iter().enumerate() {
        d_z_bin.write_slice_at(i * n_ring, w)?;
    }
    let t1 = std::time::Instant::now();

    let mut r_packed = Vec::<i8>::with_capacity(m * 64);
    r_packed.extend_from_slice(&constant_one_challenge().coeffs);
    for r in challenges {
        r_packed.extend_from_slice(&r.coeffs);
    }
    let d_r = DeviceBuffer::<i8>::from_slice(&r_packed)?;

    let mut d_out = DeviceBuffer::<i16>::new(n_ring * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_multifold_mixed_witness_tc_fused_ffi(
            d_z_bin.as_ptr(),
            ternary_chunks.pos.as_ptr(),
            ternary_chunks.neg.as_ptr(),
            d_r.as_ptr(),
            d_out.as_mut_ptr(),
            k_bin as c_int,
            k_tern as c_int,
            n_ring as u64,
        )
    };
    crate::sumcheck_prover::pool_return(d_z_bin);
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;
    let t2 = std::time::Instant::now();
    let out = d_out.to_vec();
    if timing {
        eprintln!("[mf_fused n_ring={} k_bin={} k_tern={}] upload={:?} kernel+sync={:?} dl={:?}",
            n_ring, k_bin, k_tern, t1 - t0, t2 - t1, t2.elapsed());
    }
    out
}

/// Device-resident variant of [`multifold_mixed_witness_tc_fused`]: inputs
/// are already on the CURRENT device (a binary concat buffer and/or a
/// ternary pos/neg concat pair) and the wide i16 output STAYS on device —
/// no download, no host encode. Feed the result to
/// [`wide_to_ternary_device`] + [`commit_ternary`] to complete the
/// multifold → split chain without touching the host.
pub fn multifold_mixed_witness_tc_fused_dev(
    d_z_bin: Option<&DeviceBuffer<u64>>,
    k_bin: usize,
    d_ternary: Option<(&DeviceBuffer<u64>, &DeviceBuffer<u64>)>,
    k_tern: usize,
    n_ring: usize,
    challenges: &[RingChallenge],
) -> Result<DeviceBuffer<i16>> {
    let m = k_bin + k_tern;
    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "need at least one instance (binary or ternary)".into(),
        ));
    }
    if challenges.len() + 1 != m {
        return Err(CudaError::InvalidArgument(format!(
            "expected K + T − 1 = {} challenges, got {}", m - 1, challenges.len()
        )));
    }
    if m > 511 {
        return Err(CudaError::InvalidArgument(format!(
            "num_instances {} would overflow i16 (max supported 511)", m
        )));
    }

    let mut r_packed = Vec::<i8>::with_capacity(m * 64);
    r_packed.extend_from_slice(&constant_one_challenge().coeffs);
    for r in challenges {
        r_packed.extend_from_slice(&r.coeffs);
    }
    let d_r = DeviceBuffer::<i8>::from_slice(&r_packed)?;

    let mut d_out = DeviceBuffer::<i16>::new(n_ring * RING_DIM)?;
    let null_u64 = std::ptr::null::<u64>();
    let (pos_ptr, neg_ptr) = match d_ternary {
        Some((p, n)) => (p.as_ptr(), n.as_ptr()),
        None => (null_u64, null_u64),
    };
    let ret = unsafe {
        ffi::ajtai_multifold_mixed_witness_tc_fused_ffi(
            d_z_bin.map(|b| b.as_ptr()).unwrap_or(null_u64),
            pos_ptr,
            neg_ptr,
            d_r.as_ptr(),
            d_out.as_mut_ptr(),
            k_bin as c_int,
            k_tern as c_int,
            n_ring as u64,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(d_out)
}

/// Device-side split decomposition: wide i16 multifold output →
/// `SPLIT_K_CHUNKS` signed-binary chunks (pos/neg bit-planes), all on the
/// current device. Bit-exactly replicates the host `encode_wide` path
/// (v < 0 → |v| bits in neg planes). Returns the chunk plane buffers,
/// ready for [`commit_ternary`] / next-level reuse. Errors if any |coef|
/// ≥ 2^SPLIT_K_CHUNKS (norm-bound violation).
pub fn wide_to_ternary_device(
    d_wide: &DeviceBuffer<i16>,
    n_ring: usize,
) -> Result<(DeviceBuffer<u64>, DeviceBuffer<u64>)> {
    assert_eq!(d_wide.len(), n_ring * RING_DIM, "wide buffer size mismatch");
    let k = SPLIT_K_CHUNKS;
    let mut d_pos = DeviceBuffer::<u64>::new(k * n_ring)?;
    let mut d_neg = DeviceBuffer::<u64>::new(k * n_ring)?;
    let mut d_err = DeviceBuffer::<i32>::new(1)?;
    d_err.zero()?;
    let ret = unsafe {
        ffi::aext2_wide_to_ternary_ffi(
            d_wide.as_ptr(),
            d_pos.as_mut_ptr(),
            d_neg.as_mut_ptr(),
            d_err.as_mut_ptr(),
            n_ring,
            k as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    let err = d_err.read_slice(0, 1)?;
    if err[0] != 0 {
        return Err(CudaError::InvalidArgument(
            "fold output |coef| >= 2^13 = 8192 (norm bound violated)".into(),
        ));
    }
    Ok((d_pos, d_neg))
}

/// Multi-fold commitment:  `c' = c_0 + Σ_{i=1..M-1} r_i · c_i`
///
/// `commitments[0]` is anchored with implicit weight 1; `challenges[i]`
/// applies to `commitments[i + 1]`. So `challenges` must have length
/// `commitments.len() − 1`.
pub fn multifold_commitment(
    commitments: &[&RingCommitment],
    challenges: &[RingChallenge],
) -> Result<RingCommitment> {
    let m = commitments.len();
    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "commitments must be non-empty".into(),
        ));
    }
    if challenges.len() + 1 != m {
        return Err(CudaError::InvalidArgument(format!(
            "expected exactly commitments.len() − 1 = {} challenges, got {} \
             (commitments[0] has implicit weight 1)",
            m - 1, challenges.len()
        )));
    }

    // Pack commitments: [M * KAPPA * D] flat u64.
    let mut c_packed = Vec::<u64>::with_capacity(m * KAPPA * RING_DIM);
    for c in commitments {
        for i in 0..KAPPA {
            for k in 0..RING_DIM {
                c_packed.push(c.rows[i][k]);
            }
        }
    }
    let d_c = DeviceBuffer::<u64>::from_slice(&c_packed)?;

    // Pack challenges with synthesized constant-1 at slot 0.
    let mut r_packed = Vec::<i8>::with_capacity(m * 64);
    let constant_one = constant_one_challenge();
    r_packed.extend_from_slice(&constant_one.coeffs);
    for r in challenges {
        r_packed.extend_from_slice(&r.coeffs);
    }
    let d_r = DeviceBuffer::<i8>::from_slice(&r_packed)?;

    let mut d_out = DeviceBuffer::<u64>::new(KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_multifold_commitment_ffi(
            d_c.as_ptr(),
            d_r.as_ptr(),
            m as c_int,
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            out.rows[i][k] = flat[i * RING_DIM + k];
        }
    }
    Ok(out)
}

// ============================================================================
// Split: i16 wide witness → 13 ternary chunks (pos/neg bitmask pairs)
//
// Per SuperNeo's Almost-Goldilocks parameters: b = 2, k = 13 so each chunk
// has coefficients in {-1, 0, 1} and Σ 2^i · z_i = z_wide exactly.
// Stored as two u64 bitmasks per ring element per chunk (same packing as
// the binary witness format, doubled along the sign axis).
// ============================================================================

/// Number of chunks produced by [`split_witness`] (matches the paper's `k = 13`).
pub const SPLIT_K_CHUNKS: usize = 13;

/// 13 ternary chunks of a wide folded witness.
///
/// `pos[i * n_ring + j]` is the `u64` bitmask where bit `k` is set iff the
/// `i`-th chunk's `(j, k)` coefficient is `+1`. Likewise for `neg` and `-1`.
/// By construction `pos[i*n_ring + j] & neg[i*n_ring + j] == 0` for every
/// `(i, j)` — chunks are truly ternary, never both bits set.
#[derive(Clone, Debug)]
pub struct TernaryChunks {
    pub n_ring: usize,
    pub k_chunks: usize,
    pub pos: Vec<u64>,
    pub neg: Vec<u64>,
}

impl TernaryChunks {
    /// Returns `(pos_mask, neg_mask)` slices for the i-th chunk, each of
    /// length `n_ring`.
    pub fn chunk(&self, i: usize) -> (&[u64], &[u64]) {
        let s = i * self.n_ring;
        let e = s + self.n_ring;
        (&self.pos[s..e], &self.neg[s..e])
    }
}

/// Device-resident counterpart of [`TernaryChunks`]. Use this in pipelines
/// that feed straight into a downstream commit / multifold kernel without
/// round-tripping through host memory.
pub struct TernaryChunksDevice {
    pub n_ring: usize,
    pub k_chunks: usize,
    pub pos: DeviceBuffer<u64>,
    pub neg: DeviceBuffer<u64>,
}

impl TernaryChunksDevice {
    /// Concatenate `N` ternary-chunk sets along the k-axis into a single
    /// device buffer. All inputs must share `n_ring`. The result has
    /// `k_chunks = Σ inputs[i].k_chunks` and matches the byte layout the
    /// kernels expect (chunks laid out consecutively in pos/neg).
    ///
    /// Use case: fold `N` running accumulators (each splitb-decomposed into
    /// 13 ternary chunks) together — concat them into one [`TernaryChunksDevice`]
    /// of `k_chunks = 13·N`, then pass to [`multifold_mixed_witness`] et al.
    /// Copies are on-device (`cudaMemcpy DtoD`), no host round-trip.
    pub fn concat(inputs: &[&TernaryChunksDevice]) -> Result<TernaryChunksDevice> {
        if inputs.is_empty() {
            return Err(CudaError::InvalidArgument(
                "concat() requires at least one input".into(),
            ));
        }
        let n_ring = inputs[0].n_ring;
        let mut total_k = 0usize;
        for (i, c) in inputs.iter().enumerate() {
            if c.n_ring != n_ring {
                return Err(CudaError::InvalidArgument(format!(
                    "inputs[{}].n_ring = {} mismatches inputs[0].n_ring = {}",
                    i, c.n_ring, n_ring
                )));
            }
            total_k += c.k_chunks;
        }
        if n_ring == 0 || total_k == 0 {
            return Ok(TernaryChunksDevice {
                n_ring,
                k_chunks: total_k,
                pos: DeviceBuffer::<u64>::new(0)?,
                neg: DeviceBuffer::<u64>::new(0)?,
            });
        }

        let mut pos = DeviceBuffer::<u64>::new(total_k * n_ring)?;
        let mut neg = DeviceBuffer::<u64>::new(total_k * n_ring)?;

        let mut offset_u64 = 0usize;
        for c in inputs {
            if c.k_chunks == 0 { continue; }
            let bytes = c.k_chunks * n_ring * std::mem::size_of::<u64>();
            unsafe {
                crate::memory::memcpy_dtod(
                    pos.as_mut_ptr().add(offset_u64) as *mut std::os::raw::c_void,
                    c.pos.as_ptr() as *const std::os::raw::c_void,
                    bytes,
                )?;
                crate::memory::memcpy_dtod(
                    neg.as_mut_ptr().add(offset_u64) as *mut std::os::raw::c_void,
                    c.neg.as_ptr() as *const std::os::raw::c_void,
                    bytes,
                )?;
            }
            offset_u64 += c.k_chunks * n_ring;
        }
        Ok(TernaryChunksDevice {
            n_ring,
            k_chunks: total_k,
            pos,
            neg,
        })
    }
}

/// Split the wide folded witness `z_wide` (`N_ring * 64` `i16`s) into 13
/// ternary chunks. Each coefficient `v` of `z_wide` is decomposed as
/// `v = Σ_{i=0..12} 2^i · (b_i_pos − b_i_neg)` with `b_i_pos, b_i_neg ∈ {0,1}`
/// and `b_i_pos · b_i_neg == 0`. `v` must satisfy `|v| < 2^13`.
///
/// Internally uploads to device, runs the kernel, and downloads the chunks.
/// For large `N_ring` (e.g. `≥ 2^16`) where the chunks will be re-uploaded
/// immediately for a downstream commit kernel, prefer [`split_witness_device`].
pub fn split_witness(z_wide: &[i16]) -> Result<TernaryChunks> {
    let total = z_wide.len();
    if total % RING_DIM != 0 {
        return Err(CudaError::InvalidArgument(format!(
            "z_wide length {} not divisible by RING_DIM = {}", total, RING_DIM
        )));
    }
    let n_ring = total / RING_DIM;
    if n_ring == 0 {
        return Ok(TernaryChunks {
            n_ring: 0, k_chunks: SPLIT_K_CHUNKS,
            pos: Vec::new(), neg: Vec::new(),
        });
    }

    let d_z = DeviceBuffer::<i16>::from_slice(z_wide)?;
    let mut d_pos = DeviceBuffer::<u64>::new(SPLIT_K_CHUNKS * n_ring)?;
    let mut d_neg = DeviceBuffer::<u64>::new(SPLIT_K_CHUNKS * n_ring)?;

    let ret = unsafe {
        ffi::ajtai_split_witness_ffi(
            d_z.as_ptr(),
            d_pos.as_mut_ptr(),
            d_neg.as_mut_ptr(),
            n_ring as u64,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    Ok(TernaryChunks {
        n_ring,
        k_chunks: SPLIT_K_CHUNKS,
        pos: d_pos.to_vec()?,
        neg: d_neg.to_vec()?,
    })
}

/// Device-resident variant: takes an `i16` device buffer of length
/// `n_ring * 64` and produces the chunks directly on the GPU. No host transfers.
pub fn split_witness_device(z_wide: &DeviceBuffer<i16>) -> Result<TernaryChunksDevice> {
    let total = z_wide.len();
    if total % RING_DIM != 0 {
        return Err(CudaError::InvalidArgument(format!(
            "z_wide.len() {} not divisible by RING_DIM = {}", total, RING_DIM
        )));
    }
    let n_ring = total / RING_DIM;
    if n_ring == 0 {
        return Ok(TernaryChunksDevice {
            n_ring: 0, k_chunks: SPLIT_K_CHUNKS,
            pos: DeviceBuffer::<u64>::new(0)?,
            neg: DeviceBuffer::<u64>::new(0)?,
        });
    }

    let mut pos = DeviceBuffer::<u64>::new(SPLIT_K_CHUNKS * n_ring)?;
    let mut neg = DeviceBuffer::<u64>::new(SPLIT_K_CHUNKS * n_ring)?;

    let ret = unsafe {
        ffi::ajtai_split_witness_ffi(
            z_wide.as_ptr(),
            pos.as_mut_ptr(),
            neg.as_mut_ptr(),
            n_ring as u64,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(TernaryChunksDevice { n_ring, k_chunks: SPLIT_K_CHUNKS, pos, neg })
}

/// Ajtai commit over 13 ternary chunks at once.
///
/// Returns `[c_0, c_1, …, c_12]` where `c_i = Σ_j M[*,j] · z_i[j]` and
/// `z_i ∈ {-1, 0, +1}^{64·N_ring}` is the `i`-th digit of the wide folded
/// witness (the output of [`split_witness_device`]). All 13 commitments
/// share the same matrix `M` derived from `seed`, so the ChaCha8 PRG cost
/// is amortized — this is **the** path you want after `split_witness`.
///
/// `chunk` controls the j-axis CHUNK template parameter (same meaning as
/// in [`commit_batched`]). If `None`, picks a default based on `N_ring`.
///
/// Homomorphism: if `z_wide` is the i16 wide witness fed to `split_witness`,
/// then `commit_ternary(seed, split_witness(z_wide))[i]` summed with weight
/// `2^i` equals the direct commit of `z_wide` as a ring element with the
/// same 15×N_ring matrix. This is the security-relevant invariant.
pub fn commit_ternary(
    seed: Seed,
    chunks: &TernaryChunksDevice,
    chunk: Option<ChunkSize>,
) -> Result<Vec<RingCommitment>> {
    if chunks.k_chunks != SPLIT_K_CHUNKS {
        return Err(CudaError::InvalidArgument(format!(
            "expected {} ternary chunks, got {}", SPLIT_K_CHUNKS, chunks.k_chunks
        )));
    }
    let n_ring = chunks.n_ring;
    if n_ring == 0 {
        return Err(CudaError::InvalidArgument("n_ring must be > 0".into()));
    }
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(n_ring as u64, SPLIT_K_CHUNKS));

    let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
    let mut d_out = DeviceBuffer::<u64>::new(SPLIT_K_CHUNKS * KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_commit_ternary_ffi(
            d_key.as_ptr(),
            chunks.pos.as_ptr(),
            chunks.neg.as_ptr(),
            n_ring as u64,
            chunk.as_int(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut out = Vec::with_capacity(SPLIT_K_CHUNKS);
    for bi in 0..SPLIT_K_CHUNKS {
        let mut c = RingCommitment::zero();
        for i in 0..KAPPA {
            for r in 0..RING_DIM {
                c.rows[i][r] = flat[((bi * KAPPA) + i) * RING_DIM + r];
            }
        }
        out.push(c);
    }
    Ok(out)
}

/// Pre-materialized Ajtai matrix `M[j][i][r]` of shape `[N, KAPPA, D]`
/// stored as a device buffer. Generated once from a seed via ChaCha8; the
/// caller can then re-use it across many [`commit_ternary_premat`] calls
/// (and any future `_premat` variants) without paying the per-commit PRG
/// cost.
///
/// Memory: `N · KAPPA · D · 8 bytes  =  7680 · N` bytes.
/// At `N = 2^20` that's 7.5 GB; at `N = 2^22`, 30 GB; at `N = 2^27`,
/// 960 GB (won't fit on a single A100).
pub struct MaterializedM {
    pub n: usize,
    pub buf: DeviceBuffer<u64>,
}

impl MaterializedM {
    /// Generate `M[0..N][0..KAPPA][0..D]` from `seed` via one ChaCha8 pass.
    pub fn new(seed: Seed, n: usize) -> Result<Self> {
        if n == 0 {
            return Err(CudaError::InvalidArgument("n must be > 0".into()));
        }
        let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
        let mut buf = DeviceBuffer::<u64>::new(n * KAPPA * RING_DIM)?;
        let ret = unsafe {
            ffi::ajtai_materialize_m_ffi(d_key.as_ptr(), buf.as_mut_ptr(), n as u64)
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        crate::memory::synchronize()?;
        Ok(Self { n, buf })
    }
}

/// Ternary commit against a pre-materialized `M` (no PRG inside the commit
/// kernel). Bit-exact identical to [`commit_ternary`] for the same
/// `(seed, chunks)` — the only difference is that the matrix is read from
/// HBM instead of regenerated via ChaCha8 each call.
///
/// At `N` where M fits on-device, this is ~2× faster than [`commit_ternary`]
/// in the per-commit kernel (HBM read at 1.5 TB/s replaces ChaCha8 output
/// at ~770 GB/s). The materialization cost is one-time per seed and is
/// the equivalent of a single on-the-fly commit's PRG work.
pub fn commit_ternary_premat(
    m: &MaterializedM,
    chunks: &TernaryChunksDevice,
    chunk: Option<ChunkSize>,
) -> Result<Vec<RingCommitment>> {
    if chunks.k_chunks != SPLIT_K_CHUNKS {
        return Err(CudaError::InvalidArgument(format!(
            "expected {} ternary chunks, got {}", SPLIT_K_CHUNKS, chunks.k_chunks
        )));
    }
    let n_ring = chunks.n_ring;
    if n_ring == 0 {
        return Err(CudaError::InvalidArgument("n_ring must be > 0".into()));
    }
    if m.n != n_ring {
        return Err(CudaError::InvalidArgument(format!(
            "MaterializedM.n = {} mismatches chunks.n_ring = {}", m.n, n_ring
        )));
    }
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(n_ring as u64, SPLIT_K_CHUNKS));

    let mut d_out = DeviceBuffer::<u64>::new(SPLIT_K_CHUNKS * KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_commit_ternary_premat_ffi(
            m.buf.as_ptr(),
            chunks.pos.as_ptr(),
            chunks.neg.as_ptr(),
            n_ring as u64,
            chunk.as_int(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut out = Vec::with_capacity(SPLIT_K_CHUNKS);
    for bi in 0..SPLIT_K_CHUNKS {
        let mut c = RingCommitment::zero();
        for i in 0..KAPPA {
            for r in 0..RING_DIM {
                c.rows[i][r] = flat[((bi * KAPPA) + i) * RING_DIM + r];
            }
        }
        out.push(c);
    }
    Ok(out)
}

/// Sparse commit: iterate over a position list `positions[K]` where each
/// `positions[k] ∈ [0, 64·N)` encodes `(j, ℓ) = (p >> 6, p & 63)`.
/// Best used when within-non-zero-block density is very low — at typical
/// random densities, `commit` (dense) is ~16× cheaper. Single witness only.
pub fn commit_sparse(
    seed: Seed,
    positions: &[u64],
    chunk: Option<ChunkSize>,
) -> Result<RingCommitment> {
    if positions.is_empty() {
        return Ok(RingCommitment::zero());
    }
    // Sparse is always single-witness (b = 1).
    let chunk = chunk.unwrap_or_else(|| pick_default_chunk(positions.len() as u64, 1));
    let d_pos = DeviceBuffer::<u64>::from_slice(positions)?;
    let d_key = DeviceBuffer::<u32>::from_slice(&seed.0)?;
    let mut d_out = DeviceBuffer::<u64>::new(KAPPA * RING_DIM)?;

    let ret = unsafe {
        ffi::ajtai_commit_sparse_ffi(
            d_key.as_ptr(),
            d_pos.as_ptr(),
            positions.len() as u64,
            chunk.as_int(),
            d_out.as_mut_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    crate::memory::synchronize()?;

    let flat = d_out.to_vec()?;
    let mut c = RingCommitment::zero();
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            c.rows[i][r] = flat[i * RING_DIM + r];
        }
    }
    Ok(c)
}
