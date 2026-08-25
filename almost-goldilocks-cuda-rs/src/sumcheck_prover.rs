//! GPU-accelerated sumcheck prover state for the almost-Goldilocks field.
//!
//! Holds `d` polynomials on GPU and supports round-message computation and
//! folding with double buffering for race-free updates. Mirrors the API of
//! `goldilocks-cuda-rs::sumcheck_prover`.

use crate::error::{CudaError, Result};
use crate::extension::AlmostGoldilocksExt2;
use crate::ffi;
use crate::field::AlmostGoldilocksField;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;

const BLOCK_SIZE: usize = 256;

// ============================================================================
// Thread-local GPU buffer pool — amortizes the cost of (re)allocating the
// large `d_polys`/`d_scratch` buffers (≈ 16 GB each at arity 22 / 63 leaves)
// across the serially-dispatched fold-tree groups within a bucket.
//
// Measured: a fresh 16 GB cudaMalloc costs ~5 ms when the allocator has a
// cached block but ~240 ms once it must commit new pages — and the fold tree
// does this ~40 times per arity-22 bucket. The pool keeps freed buffers alive
// (never cudaFree's them during proving) so subsequent same-size requests are
// instant.
//
// Keyed by (device, size). Groups in a bucket are dispatched serially on one
// rayon worker and pinned to one device, so the per-thread pool is reused
// cleanly; the device key guards against a worker that later serves a
// different bucket pinned to another GPU.
// ============================================================================
fn current_device() -> i32 {
    let mut d = 0i32;
    unsafe { let _ = ffi::cuda_get_device(&mut d as *mut i32); }
    d
}

thread_local! {
    static SP_POOL: std::cell::RefCell<std::collections::HashMap<(i32, usize), Vec<DeviceBuffer<u64>>>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Take a pooled u64 device buffer of AT LEAST `size` elements (best-fit may
/// return up to 4*size - callers must derive all kernel offsets from their
/// own computed sizes, never `buf.len()`). Public so other GPU-heavy modules
/// (ajtai multifold, zk-torch fold tree) can share the per-thread pool
/// instead of paying fresh cudaMalloc + synchronizing cudaFree per call.
pub fn pool_take(size: usize) -> Result<DeviceBuffer<u64>> {
    let dev = current_device();
    let cached = SP_POOL.with(|p| {
        let mut pool = p.borrow_mut();
        // Exact-size slot first.
        if let Some(b) = pool.get_mut(&(dev, size)).and_then(|v| v.pop()) {
            return Some(b);
        }
        // Best-fit: reuse the smallest cached buffer whose length is in
        // [size, 4·size]. Callers only touch the first `size` u64s (all
        // kernel offsets derive from the computed total, never buf.len()),
        // so an oversized buffer is correct. This avoids the clear+realloc
        // churn when consecutive fold-tree groups in a bucket have slightly
        // different leaf counts (e.g. arity-24 M=31 then M=22/M=26): without
        // it each size-mismatched group pays a multi-GB cudaFree+cudaMalloc.
        // The 4× cap stops a tiny request (e.g. d_partial) from claiming a
        // multi-GB d_polys buffer.
        let best = pool.iter()
            .filter(|(k, v)| k.0 == dev && k.1 >= size && k.1 <= size.saturating_mul(4) && !v.is_empty())
            .map(|(k, _)| *k)
            .min_by_key(|k| k.1);
        if let Some(bk) = best {
            return pool.get_mut(&bk).and_then(|v| v.pop());
        }
        None
    });
    if let Some(buf) = cached { return Ok(buf); }
    // Fresh allocation. If it OOMs, the pool may be holding large stale
    // buffers from a previous (different-arity) bucket — evict them and
    // retry. This is the cross-bucket case: e.g. the arity-22 bucket
    // pooled 32 GB buffers, then the arity-24 bucket needs 64 GB and the
    // stale pool tips it over the device limit.
    match DeviceBuffer::<u64>::new(size) {
        Ok(b) => Ok(b),
        Err(_) => {
            // A failed cudaMalloc leaves cudaErrorMemoryAllocation as the
            // sticky "last error"; consume it here so the next kernel's
            // post-launch cudaGetLastError() check doesn't misattribute it
            // as a KernelFailed. Then evict stale buffers and retry.
            let _ = crate::memory::get_last_error();
            SP_POOL.with(|p| p.borrow_mut().clear());
            DeviceBuffer::<u64>::new(size)
        }
    }
}

/// Return a buffer taken via [`pool_take`] to the per-thread pool. Safe to
/// call right after an async kernel launch that reads the buffer: all
/// subsequent reuse on this thread serializes behind it on the default
/// stream.
pub fn pool_return(buf: DeviceBuffer<u64>) {
    let dev = current_device();
    let size = buf.len();
    SP_POOL.with(|p| p.borrow_mut().entry((dev, size)).or_default().push(buf));
}

/// Clear this thread's SP_POOL (forcing all cached buffers to drop +
/// cudaFree). Used by long-running streams (e.g. streaming-inference
/// bench) to release pool memory between iterations. Without this, the
/// pool accumulates buffers across different (size) slots from groups
/// of varying leaf-count/arity — measured ~10 GB / iter growth on
/// Llama-2 streaming, eventually OOMing the GPU.
///
/// MUST be called from the same thread that populated the pool. To
/// clear pools across all rayon workers, use `rayon::broadcast`:
///
/// ```ignore
/// rayon::broadcast(|_| almost_goldilocks_cuda::clear_thread_sp_pool());
/// ```
pub fn clear_thread_sp_pool() {
    SP_POOL.with(|p| p.borrow_mut().clear());
}

// Host-side base-field add (canonicalize inputs first; reuses the field's host arithmetic).
#[inline]
fn agl_add_host(a: u64, b: u64) -> u64 {
    (AlmostGoldilocksField(a) + AlmostGoldilocksField(b)).reduce().0
}

pub struct GpuSumcheckState {
    d_polys: DeviceBuffer<u64>,
    d_scratch: DeviceBuffer<u64>,
    d_partial: DeviceBuffer<u64>,
    num_polys: usize,
    original_size: usize,
    current_round: usize,
    num_vars: usize,
}

impl GpuSumcheckState {
    pub fn new(polys: &[&[AlmostGoldilocksField]]) -> Result<Self> {
        assert!(!polys.is_empty(), "Must have at least one polynomial");
        let original_size = polys[0].len();
        assert!(original_size.is_power_of_two(), "Polynomial size must be power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        let num_polys = polys.len();
        for (i, p) in polys.iter().enumerate() {
            assert_eq!(p.len(), original_size, "Polynomial {} size mismatch", i);
        }

        let total = num_polys * original_size;
        let mut packed = Vec::with_capacity(total);
        for p in polys {
            for &v in *p { packed.push(v.0); }
        }
        let d_polys = DeviceBuffer::from_slice(&packed)?;
        let d_scratch = DeviceBuffer::new(total)?;

        let max_blocks = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks = max_blocks.min(256);
        let d_partial = DeviceBuffer::new(max_blocks * (num_polys + 1))?;

        Ok(Self {
            d_polys, d_scratch, d_partial,
            num_polys, original_size, current_round: 0, num_vars,
        })
    }

    pub fn compute_round_message(&mut self) -> Result<Vec<AlmostGoldilocksField>> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds available".to_string()));
        }
        let num_blocks = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256) as c_int;
        let dp1 = self.num_polys + 1;

        let ret = unsafe {
            ffi::agl_sumcheck_round_message_ffi(
                self.d_polys.as_ptr(),
                self.d_partial.as_mut_ptr(),
                self.num_polys as c_int,
                self.original_size,
                half,
                num_blocks,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let partials = self.d_partial.read_slice(0, num_blocks as usize * dp1)?;
        let mut result = vec![AlmostGoldilocksField(0); dp1];
        for block in 0..num_blocks as usize {
            for c in 0..dp1 {
                result[c] = AlmostGoldilocksField(agl_add_host(
                    result[c].0,
                    partials[block * dp1 + c],
                ));
            }
        }
        Ok(result)
    }

    pub fn fold(&mut self, challenge: AlmostGoldilocksField) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds to fold".to_string()));
        }
        let ret = unsafe {
            ffi::agl_sumcheck_fold_ffi(
                self.d_polys.as_ptr(),
                self.d_scratch.as_mut_ptr(),
                challenge.0,
                self.num_polys as c_int,
                self.original_size,
                half,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        std::mem::swap(&mut self.d_polys, &mut self.d_scratch);
        self.current_round += 1;
        Ok(())
    }

    pub fn final_evaluations(&self) -> Result<Vec<AlmostGoldilocksField>> {
        assert_eq!(self.current_round, self.num_vars,
                   "Must complete all rounds before getting final evaluations");
        let mut result = Vec::with_capacity(self.num_polys);
        for i in 0..self.num_polys {
            let vals = self.d_polys.read_slice(i * self.original_size, 1)?;
            result.push(AlmostGoldilocksField(vals[0]));
        }
        Ok(result)
    }

    pub fn final_eval(&self, poly_idx: usize) -> Result<AlmostGoldilocksField> {
        assert!(poly_idx < self.num_polys);
        let vals = self.d_polys.read_slice(poly_idx * self.original_size, 1)?;
        Ok(AlmostGoldilocksField(vals[0]))
    }

    pub fn num_vars(&self) -> usize { self.num_vars }
    pub fn num_polys(&self) -> usize { self.num_polys }
    pub fn current_round(&self) -> usize { self.current_round }
}

// ============================================================================
// Ext2 variant
// ============================================================================

#[inline]
fn aext2_add_host(a: AlmostGoldilocksExt2, b: AlmostGoldilocksExt2) -> AlmostGoldilocksExt2 {
    a + b
}

pub struct GpuSumcheckStateExt2 {
    d_polys: DeviceBuffer<u64>,
    d_scratch: DeviceBuffer<u64>,
    d_partial: DeviceBuffer<u64>,
    num_polys: usize,
    original_size: usize,
    current_round: usize,
    num_vars: usize,
}

impl GpuSumcheckStateExt2 {
    pub fn new(polys: &[&[AlmostGoldilocksExt2]]) -> Result<Self> {
        assert!(!polys.is_empty(), "Must have at least one polynomial");
        let original_size = polys[0].len();
        assert!(original_size.is_power_of_two(), "Polynomial size must be power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        let num_polys = polys.len();
        for (i, p) in polys.iter().enumerate() {
            assert_eq!(p.len(), original_size, "Polynomial {} size mismatch", i);
        }

        let total_u64s = num_polys * original_size * 2;
        let mut packed = Vec::with_capacity(total_u64s);
        for p in polys {
            for &v in *p { packed.push(v.c0.0); packed.push(v.c1.0); }
        }
        let d_polys = DeviceBuffer::from_slice(&packed)?;
        let d_scratch = pool_take(total_u64s)?;

        let max_blocks = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks = max_blocks.min(256);
        let d_partial = DeviceBuffer::new(max_blocks * (num_polys + 1) * 2)?;

        Ok(Self {
            d_polys, d_scratch, d_partial,
            num_polys, original_size, current_round: 0, num_vars,
        })
    }

    pub fn new_from_base(polys: &[&[AlmostGoldilocksField]]) -> Result<Self> {
        let ext_polys: Vec<Vec<AlmostGoldilocksExt2>> = polys.iter()
            .map(|p| p.iter().map(|&v| AlmostGoldilocksExt2::from_base(v)).collect())
            .collect();
        let refs: Vec<&[AlmostGoldilocksExt2]> = ext_polys.iter().map(|v| v.as_slice()).collect();
        Self::new(&refs)
    }

    /// Build from GPU-resident Ext2 buffers (D2D copy, no host round-trip).
    ///
    /// Each buffer holds `original_size` Ext2 elements = `original_size * 2`
    /// `u64`s in interleaved `[c0, c1, c0, c1, ...]` order — matching the
    /// packing used by [`Self::new`]. Buffers are concatenated contiguously
    /// via `cudaMemcpy DtoD`.
    ///
    /// Use this from prover code that already has Ext2 polys on-device
    /// (e.g. the Reducer's per-claim eq tables) to avoid an unnecessary
    /// download + re-upload.
    pub fn from_device_buffers(
        buffers: &[&DeviceBuffer<u64>],
        original_size: usize,
    ) -> Result<Self> {
        assert!(!buffers.is_empty(), "from_device_buffers: at least one buffer");
        assert!(original_size.is_power_of_two(), "original_size must be a power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        let num_polys = buffers.len();
        let stride_u64 = original_size * 2;
        for (i, buf) in buffers.iter().enumerate() {
            assert_eq!(
                buf.len(), stride_u64,
                "buffer {} has {} u64s, expected {} (original_size = {})",
                i, buf.len(), stride_u64, original_size,
            );
        }

        let total_u64s = num_polys * stride_u64;
        let mut d_polys = pool_take(total_u64s)?;
        for (i, buf) in buffers.iter().enumerate() {
            let offset_u64 = i * stride_u64;
            let bytes = stride_u64 * std::mem::size_of::<u64>();
            unsafe {
                crate::memory::memcpy_dtod(
                    d_polys.as_mut_ptr().add(offset_u64) as *mut std::os::raw::c_void,
                    buf.as_ptr() as *const std::os::raw::c_void,
                    bytes,
                )?;
            }
        }
        let d_scratch = pool_take(total_u64s)?;
        let max_blocks = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks = max_blocks.min(256);
        let d_partial = DeviceBuffer::new(max_blocks * (num_polys + 1) * 2)?;

        Ok(Self {
            d_polys, d_scratch, d_partial,
            num_polys, original_size, current_round: 0, num_vars,
        })
    }

    pub fn compute_round_message(&mut self) -> Result<Vec<AlmostGoldilocksExt2>> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds available".to_string()));
        }
        let num_blocks = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256) as c_int;
        let dp1 = self.num_polys + 1;

        let ret = unsafe {
            ffi::aext2_sumcheck_round_message_ffi(
                self.d_polys.as_ptr(),
                self.d_partial.as_mut_ptr(),
                self.num_polys as c_int,
                self.original_size,
                half,
                num_blocks,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let partials = self.d_partial.read_slice(0, num_blocks as usize * dp1 * 2)?;
        let mut result = vec![AlmostGoldilocksExt2::zero(); dp1];
        for block in 0..num_blocks as usize {
            for c in 0..dp1 {
                let off = (block * dp1 + c) * 2;
                let p = AlmostGoldilocksExt2::new(
                    AlmostGoldilocksField(partials[off]),
                    AlmostGoldilocksField(partials[off + 1]),
                );
                result[c] = aext2_add_host(result[c], p);
            }
        }
        Ok(result)
    }

    pub fn fold(&mut self, challenge: AlmostGoldilocksExt2) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds to fold".to_string()));
        }
        let ret = unsafe {
            ffi::aext2_sumcheck_fold_ffi(
                self.d_polys.as_ptr(),
                self.d_scratch.as_mut_ptr(),
                challenge.c0.0,
                challenge.c1.0,
                self.num_polys as c_int,
                self.original_size,
                half,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        std::mem::swap(&mut self.d_polys, &mut self.d_scratch);
        self.current_round += 1;
        Ok(())
    }

    pub fn final_evaluations(&self) -> Result<Vec<AlmostGoldilocksExt2>> {
        assert_eq!(self.current_round, self.num_vars,
                   "Must complete all rounds before getting final evaluations");
        let mut result = Vec::with_capacity(self.num_polys);
        for i in 0..self.num_polys {
            let off = i * self.original_size * 2;
            let vals = self.d_polys.read_slice(off, 2)?;
            result.push(AlmostGoldilocksExt2::new(
                AlmostGoldilocksField(vals[0]),
                AlmostGoldilocksField(vals[1]),
            ));
        }
        Ok(result)
    }

    pub fn num_vars(&self) -> usize { self.num_vars }
    pub fn num_polys(&self) -> usize { self.num_polys }
}

impl Drop for GpuSumcheckStateExt2 {
    fn drop(&mut self) {
        let z = || DeviceBuffer::<u64>::new(0).expect("0-size buffer");
        pool_return(std::mem::replace(&mut self.d_polys, z()));
        pool_return(std::mem::replace(&mut self.d_scratch, z()));
        pool_return(std::mem::replace(&mut self.d_partial, z()));
    }
}

/// Batched per-leaf same-point sumcheck state. Holds K leaves' (eq, f)
/// Ext2 polynomials on-device in a single contiguous buffer.
/// One kernel launch per round handles ALL leaves — avoids the
/// per-leaf-launch overhead that dominated the unbatched `prove_same_point_gpu`.
///
/// Memory layout: `[leaf_0_eq | leaf_0_f | leaf_1_eq | leaf_1_f | ...]`
/// — `2 * num_leaves` polys of `original_size` Ext2 values each.
/// Deduplicate per-leaf claim points. Returns `(unique_pts, leaf_to_unique)`
/// where `unique_pts.len() ≤ claim_pts.len()` and `leaf_to_unique[i]`
/// indexes into `unique_pts` for leaf `i`.
///
/// Identifies leaves that share an eq-table — all bit planes of one
/// committed edge in zk-torch-4 share `extended_point`, so a fold-tree
/// group of 63 leaves drawn from ~3 edges has only ~3 unique points.
fn dedup_claim_pts(claim_pts: &[Vec<AlmostGoldilocksExt2>]) -> (Vec<&[AlmostGoldilocksExt2]>, Vec<usize>) {
    let mut unique: Vec<&[AlmostGoldilocksExt2]> = Vec::new();
    let mut map: Vec<usize> = Vec::with_capacity(claim_pts.len());
    for pt in claim_pts {
        if let Some(idx) = unique.iter().position(|u| {
            u.len() == pt.len() && u.iter().zip(pt.iter()).all(|(a, b)| a == b)
        }) {
            map.push(idx);
        } else {
            map.push(unique.len());
            unique.push(pt.as_slice());
        }
    }
    (unique, map)
}

pub struct GpuBatchedSamePointState {
    d_polys: DeviceBuffer<u64>,
    d_scratch: DeviceBuffer<u64>,
    d_partial: DeviceBuffer<u64>,
    num_leaves: usize,
    original_size: usize,
    current_round: usize,
    num_vars: usize,
    /// When `Some`, round 0 runs the binary fused path: the f-poly is still
    /// packed bits here (the eq-slot of `d_polys` is built; the f-slot is
    /// NOT lifted). The round-0 message + fold read these bits directly,
    /// avoiding the `2^arity` Ext2 lift. Dropped after round 0.
    d_packed: Option<DeviceBuffer<u64>>,
    packed_size_u64: usize,
}

impl GpuBatchedSamePointState {
    /// Build from per-leaf `claim_pt` (small — `arity` Ext2 values each)
    /// and host-lifted `f` Ext2 tables. Eq tables are built ON DEVICE
    /// via [`crate::eq_lagrange::ext2_eq_dp_all_device`] — avoiding the
    /// ~2.7 GB of host→device upload that the all-host
    /// [`Self::new`] constructor incurs at GPT-2 arity 22.
    ///
    /// `claim_pts.len() == fs.len()` must hold; all `f`s must have the
    /// same length `2^arity`.
    pub fn new_device_eq(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        fs: &[Vec<AlmostGoldilocksExt2>],
    ) -> Result<Self> {
        use crate::eq_lagrange::ext2_eq_dp_all_device;
        assert_eq!(claim_pts.len(), fs.len(), "leaf count mismatch");
        assert!(!claim_pts.is_empty(), "empty input");
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        for (i, pt) in claim_pts.iter().enumerate() {
            assert_eq!(pt.len(), arity, "leaf {} claim_pt arity mismatch", i);
        }
        for (i, f) in fs.iter().enumerate() {
            assert_eq!(f.len(), original_size, "leaf {} f size mismatch", i);
        }

        let num_leaves = claim_pts.len();
        let poly_u64s = original_size * 2;
        let leaf_u64s = 2 * poly_u64s;
        let total_u64s = num_leaves * leaf_u64s;

        // Allocate the full batched buffer.
        let mut d_polys = pool_take(total_u64s)?;

        // For each leaf: upload claim_pt, build eq on device, copy eq into
        // the leaf's eq slot, copy host-built f into the leaf's f slot.
        for (leaf, (pt, f)) in claim_pts.iter().zip(fs.iter()).enumerate() {
            // 1) eq on device.
            let d_pt = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(pt)?;
            let (d_a, d_b, in_a) = ext2_eq_dp_all_device(&d_pt, arity)?;
            let src_eq_buf = if in_a { &d_a } else { &d_b };
            let dst_eq_offset_u64 = leaf * leaf_u64s;
            let eq_bytes = poly_u64s * std::mem::size_of::<u64>();
            unsafe {
                crate::memory::memcpy_dtod(
                    d_polys.as_mut_ptr().add(dst_eq_offset_u64) as *mut std::os::raw::c_void,
                    src_eq_buf.as_ptr() as *const std::os::raw::c_void,
                    eq_bytes,
                )?;
            }
            // 2) f on device — upload host vec directly into its slot.
            let mut f_packed = Vec::with_capacity(poly_u64s);
            for v in f { f_packed.push(v.c0.0); f_packed.push(v.c1.0); }
            let dst_f_offset_u64 = leaf * leaf_u64s + poly_u64s;
            // Upload to a transient buffer then D2D copy (DeviceBuffer
            // doesn't expose an "upload at offset" API).
            let d_f = DeviceBuffer::<u64>::from_slice(&f_packed)?;
            let f_bytes = poly_u64s * std::mem::size_of::<u64>();
            unsafe {
                crate::memory::memcpy_dtod(
                    d_polys.as_mut_ptr().add(dst_f_offset_u64) as *mut std::os::raw::c_void,
                    d_f.as_ptr() as *const std::os::raw::c_void,
                    f_bytes,
                )?;
            }
        }

        let d_scratch = pool_take(total_u64s)?;
        let max_blocks_x = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks_x = max_blocks_x.min(256);
        let d_partial = pool_take(max_blocks_x * num_leaves * 3 * 2)?;
        Ok(Self {
            d_polys, d_scratch, d_partial,
            num_leaves, original_size,
            current_round: 0,
            num_vars: arity,
            d_packed: None,
            packed_size_u64: 0,
        })
    }

    /// Maximum host→device upload reduction: eq tables built on device
    /// from `claim_pts`, binary `f` lifted from packed bits on device.
    /// Upload size per leaf is ~`arity` Ext2 (claim_pt) + `n/64` u64s
    /// (packed witness) — at arity 22, ~64 KB vs ~128 MB for the
    /// host-lifted path.
    pub fn new_device_eq_packed_f(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        packed_fs: &[&[u64]],
    ) -> Result<Self> {
        use crate::eq_lagrange::ext2_eq_dp_all_device;
        assert_eq!(claim_pts.len(), packed_fs.len(), "leaf count mismatch");
        assert!(!claim_pts.is_empty(), "empty input");
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        let expected_packed = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
        for (i, pt) in claim_pts.iter().enumerate() {
            assert_eq!(pt.len(), arity, "leaf {} claim_pt arity mismatch", i);
        }
        for (i, p) in packed_fs.iter().enumerate() {
            assert_eq!(p.len(), expected_packed, "leaf {} packed f size mismatch", i);
        }

        let num_leaves = claim_pts.len();
        let poly_u64s = original_size * 2;
        let leaf_u64s = 2 * poly_u64s;
        let total_u64s = num_leaves * leaf_u64s;
        let _dbg = std::env::var("ZK4_TIMING_SETUP").is_ok();
        let _ta = std::time::Instant::now();
        let mut d_polys = pool_take(total_u64s)?;
        // Allocate d_scratch up-front so the batched-eq path can use it
        // as the ping-pong buffer (later reused by the sumcheck fold).
        let mut d_scratch = pool_take(total_u64s)?;
        let _dt_alloc = _ta.elapsed();
        let _teq = std::time::Instant::now();

        // Fast path: if all claim_pts are identical (the common case at
        // level 1+ of any fold-tree bucket — every leaf in a group
        // shares the previous level's `shared_r`), build the eq table
        // ONCE on device and D2D-broadcast it into every leaf's slot.
        // Saves `(num_leaves - 1) × per-leaf-eq-build` time — ~1.8 s
        // per group at arity 22 with 63 leaves.
        let shared_eq = num_leaves > 1
            && claim_pts.iter().skip(1).all(|p| p == &claim_pts[0]);

        if shared_eq {
            let d_pt = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&claim_pts[0])?;
            let (d_a, d_b, in_a) = ext2_eq_dp_all_device(&d_pt, arity)?;
            let src = if in_a { &d_a } else { &d_b };
            let eq_bytes = poly_u64s * std::mem::size_of::<u64>();
            for leaf in 0..num_leaves {
                let dst_off = leaf * leaf_u64s;
                unsafe {
                    crate::memory::memcpy_dtod(
                        d_polys.as_mut_ptr().add(dst_off) as *mut std::os::raw::c_void,
                        src.as_ptr() as *const std::os::raw::c_void,
                        eq_bytes,
                    )?;
                }
            }
        } else {
            // Cluster leaves by claim_pt: bit planes of the same edge
            // all share `extended_point`, so a group of 63 leaves from
            // ~3 edges has only ~3 unique claim_pts. Build eq ONCE per
            // unique point on device, then D2D-broadcast to every leaf
            // sharing it. Cuts eq-build work by `num_leaves / num_unique`
            // — typically ~21× for fold-tree groups in Llama-2-7B.
            let (unique_pts, leaf_to_unique) = dedup_claim_pts(claim_pts);
            let num_unique = unique_pts.len();
            if std::env::var("ZK4_TIMING_DEDUP").is_ok() {
                eprintln!("[dedup arity={} M={}] unique_pts={} (savings {:.1}x)",
                    arity, num_leaves, num_unique, num_leaves as f64 / num_unique as f64);
            }
            let leaf_stride_ext2 = leaf_u64s / 2;
            let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(num_unique * arity);
            for pt in &unique_pts { r_concat.extend_from_slice(pt); }
            let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
            // Build `num_unique` eq tables in `d_scratch[0..num_unique]`,
            // packed contiguously (stride = original_size Ext2 per unique).
            let total_unique_u64s = num_unique * original_size * 2;
            let mut d_eq_unique = DeviceBuffer::<u64>::new(total_unique_u64s)?;
            let mut d_eq_scratch = DeviceBuffer::<u64>::new(total_unique_u64s)?;
            let mut result_ptr: *mut u64 = std::ptr::null_mut();
            let ret = unsafe {
                ffi::aext2_eq_dp_all_batched_ffi(
                    d_r_all.as_ptr() as *const u64,
                    d_eq_unique.as_mut_ptr(),
                    d_eq_scratch.as_mut_ptr(),
                    arity as c_int,
                    num_unique as c_int,
                    original_size,  // contiguous packing — stride = N Ext2s
                    &mut result_ptr,
                    std::ptr::null_mut(),
                )
            };
            if ret != 0 { return Err(CudaError::KernelFailed); }
            // result_ptr points into either d_eq_unique or d_eq_scratch.
            let eq_bytes = poly_u64s * std::mem::size_of::<u64>();
            for leaf in 0..num_leaves {
                let unique_idx = leaf_to_unique[leaf];
                let src_off_u64 = unique_idx * original_size * 2;
                let dst_off_u64 = leaf * leaf_u64s;
                unsafe {
                    crate::memory::memcpy_dtod(
                        d_polys.as_mut_ptr().add(dst_off_u64) as *mut std::os::raw::c_void,
                        (result_ptr as *const u64).add(src_off_u64) as *const std::os::raw::c_void,
                        eq_bytes,
                    )?;
                }
            }
            crate::memory::synchronize()?;
            drop(d_eq_unique);
            drop(d_eq_scratch);
        }
        let _dt_eq = _teq.elapsed();
        let _tlift = std::time::Instant::now();

        // 2) Concat all packed bits, upload once.
        let total_packed = num_leaves * expected_packed;
        let mut packed_concat = Vec::with_capacity(total_packed);
        for p in packed_fs { packed_concat.extend_from_slice(p); }
        let d_packed = DeviceBuffer::<u64>::from_slice(&packed_concat)?;

        // Binary round-0 fused path: keep the packed bits and skip the
        // `2^arity` Ext2 lift — round 0's message + fold read the bits
        // directly. Net win only at large arity (measured: arity-22 sp
        // 2.70s→2.23s; but arity-20 GPT-2 is ~5% slower — the branchy
        // selective-add diverges and the lift it saves is cheap at small
        // size). Gate at arity ≥ 22 by default (ZK4_BINARY_ROUND0_MIN_ARITY);
        // ZK4_BINARY_ROUND0=0 forces the standard lift path (A/B testing).
        let min_arity = std::env::var("ZK4_BINARY_ROUND0_MIN_ARITY").ok()
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(22);
        let binary_round0 = std::env::var("ZK4_BINARY_ROUND0").as_deref() != Ok("0")
            && arity >= min_arity;
        let kept_packed = if binary_round0 {
            Some(d_packed)
        } else {
            let ret = unsafe {
                ffi::aext2_batched_lift_binary_ffi(
                    d_packed.as_ptr(),
                    d_polys.as_mut_ptr(),
                    original_size,
                    num_leaves as c_int,
                    expected_packed,
                )
            };
            if ret != 0 { return Err(CudaError::KernelFailed); }
            None
        };

        let max_blocks_x = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks_x = max_blocks_x.min(256);
        let d_partial = pool_take(max_blocks_x * num_leaves * 3 * 2)?;
        if _dbg {
            eprintln!("[sp_setup arity={} M={}] alloc={:?} eq={:?} lift={:?}",
                arity, num_leaves, _dt_alloc, _dt_eq, _tlift.elapsed());
        }
        Ok(Self {
            d_polys, d_scratch, d_partial,
            num_leaves, original_size,
            current_round: 0, num_vars: arity,
            d_packed: kept_packed,
            packed_size_u64: expected_packed,
        })
    }

    /// Single-chunk ternary witness equivalent of [`Self::new_device_eq_packed_f`].
    /// Per leaf: claim_pt (small), `pos`/`neg` packed bitmasks (small).
    /// Eq built on device, f lifted on device — upload stays ~MB regardless of arity.
    pub fn new_device_eq_packed_ternary(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        pos_per_leaf: &[&[u64]],
        neg_per_leaf: &[&[u64]],
    ) -> Result<Self> {
        use crate::eq_lagrange::ext2_eq_dp_all_device;
        assert_eq!(claim_pts.len(), pos_per_leaf.len());
        assert_eq!(claim_pts.len(), neg_per_leaf.len());
        assert!(!claim_pts.is_empty());
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        let expected_packed = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
        for (i, p) in pos_per_leaf.iter().enumerate() {
            assert_eq!(p.len(), expected_packed, "leaf {} pos size mismatch", i);
        }
        for (i, p) in neg_per_leaf.iter().enumerate() {
            assert_eq!(p.len(), expected_packed, "leaf {} neg size mismatch", i);
        }
        let num_leaves = claim_pts.len();
        let poly_u64s = original_size * 2;
        let leaf_u64s = 2 * poly_u64s;
        let total_u64s = num_leaves * leaf_u64s;
        let mut d_polys = pool_take(total_u64s)?;
        let mut d_scratch = pool_take(total_u64s)?;

        // Same shared-eq fast path as the binary ctor.
        let shared_eq = num_leaves > 1
            && claim_pts.iter().skip(1).all(|p| p == &claim_pts[0]);
        if shared_eq {
            let d_pt = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&claim_pts[0])?;
            let (d_a, d_b, in_a) = ext2_eq_dp_all_device(&d_pt, arity)?;
            let src = if in_a { &d_a } else { &d_b };
            let eq_bytes = poly_u64s * std::mem::size_of::<u64>();
            for leaf in 0..num_leaves {
                let dst_off = leaf * leaf_u64s;
                unsafe {
                    crate::memory::memcpy_dtod(
                        d_polys.as_mut_ptr().add(dst_off) as *mut std::os::raw::c_void,
                        src.as_ptr() as *const std::os::raw::c_void,
                        eq_bytes,
                    )?;
                }
            }
        } else {
            // Dedup claim_pts and build eq once per unique point.
            let (unique_pts, leaf_to_unique) = dedup_claim_pts(claim_pts);
            let num_unique = unique_pts.len();
            let _ = leaf_u64s;
            let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(num_unique * arity);
            for pt in &unique_pts { r_concat.extend_from_slice(pt); }
            let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
            let total_unique_u64s = num_unique * original_size * 2;
            let mut d_eq_unique = DeviceBuffer::<u64>::new(total_unique_u64s)?;
            let mut d_eq_scratch = DeviceBuffer::<u64>::new(total_unique_u64s)?;
            let mut result_ptr: *mut u64 = std::ptr::null_mut();
            let ret = unsafe {
                ffi::aext2_eq_dp_all_batched_ffi(
                    d_r_all.as_ptr() as *const u64,
                    d_eq_unique.as_mut_ptr(),
                    d_eq_scratch.as_mut_ptr(),
                    arity as c_int,
                    num_unique as c_int,
                    original_size,
                    &mut result_ptr,
                    std::ptr::null_mut(),
                )
            };
            if ret != 0 { return Err(CudaError::KernelFailed); }
            let eq_bytes = poly_u64s * std::mem::size_of::<u64>();
            for leaf in 0..num_leaves {
                let unique_idx = leaf_to_unique[leaf];
                let src_off_u64 = unique_idx * original_size * 2;
                let dst_off_u64 = leaf * leaf_u64s;
                unsafe {
                    crate::memory::memcpy_dtod(
                        d_polys.as_mut_ptr().add(dst_off_u64) as *mut std::os::raw::c_void,
                        (result_ptr as *const u64).add(src_off_u64) as *const std::os::raw::c_void,
                        eq_bytes,
                    )?;
                }
            }
            drop(d_eq_unique);
            drop(d_eq_scratch);
        }

        // 2) Concat pos / neg, upload, on-device lift.
        let total_packed = num_leaves * expected_packed;
        let mut pos_concat = Vec::with_capacity(total_packed);
        let mut neg_concat = Vec::with_capacity(total_packed);
        for p in pos_per_leaf { pos_concat.extend_from_slice(p); }
        for p in neg_per_leaf { neg_concat.extend_from_slice(p); }
        let d_pos = DeviceBuffer::<u64>::from_slice(&pos_concat)?;
        let d_neg = DeviceBuffer::<u64>::from_slice(&neg_concat)?;
        let ret = unsafe {
            ffi::aext2_batched_lift_ternary_single_ffi(
                d_pos.as_ptr(),
                d_neg.as_ptr(),
                d_polys.as_mut_ptr(),
                original_size,
                num_leaves as c_int,
                expected_packed,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let max_blocks_x = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks_x = max_blocks_x.min(256);
        let d_partial = pool_take(max_blocks_x * num_leaves * 3 * 2)?;
        Ok(Self {
            d_polys, d_scratch, d_partial,
            num_leaves, original_size,
            current_round: 0, num_vars: arity,
            d_packed: None,
            packed_size_u64: 0,
        })
    }

    /// Build from per-leaf (eq, f) pairs. All must share `original_size`.
    pub fn new(per_leaf: &[(Vec<AlmostGoldilocksExt2>, Vec<AlmostGoldilocksExt2>)]) -> Result<Self> {
        assert!(!per_leaf.is_empty(), "GpuBatchedSamePointState: empty input");
        let original_size = per_leaf[0].0.len();
        assert!(original_size.is_power_of_two(), "size must be power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        for (i, (eq, f)) in per_leaf.iter().enumerate() {
            assert_eq!(eq.len(), original_size, "leaf {} eq size mismatch", i);
            assert_eq!(f.len(), original_size, "leaf {} f size mismatch", i);
        }
        let num_leaves = per_leaf.len();
        let poly_u64s = original_size * 2;
        let leaf_u64s = 2 * poly_u64s;
        let total_u64s = num_leaves * leaf_u64s;

        // Pack: per leaf, eq then f (each as c0,c1 interleaved per element).
        let mut packed = Vec::with_capacity(total_u64s);
        for (eq, f) in per_leaf {
            for &v in eq { packed.push(v.c0.0); packed.push(v.c1.0); }
            for &v in f  { packed.push(v.c0.0); packed.push(v.c1.0); }
        }
        let d_polys = DeviceBuffer::from_slice(&packed)?;
        let d_scratch = pool_take(total_u64s)?;

        let max_blocks_x = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks_x = max_blocks_x.min(256);
        let d_partial = pool_take(max_blocks_x * num_leaves * 3 * 2)?;

        Ok(Self { d_polys, d_scratch, d_partial, num_leaves, original_size, current_round: 0, num_vars, d_packed: None, packed_size_u64: 0 })
    }

    /// Compute the degree-2 round message per leaf at the current round.
    /// Returns `num_leaves` triples `[T_k(0), T_k(1), T_k(2)]` packed
    /// as a flat `Vec<AlmostGoldilocksExt2>` of length `3 * num_leaves`.
    pub fn compute_round_messages(&mut self) -> Result<Vec<AlmostGoldilocksExt2>> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds".to_string()));
        }
        let num_blocks_x = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256) as c_int;

        let ret = if self.current_round == 0 && self.d_packed.is_some() {
            // Binary round-0 fused message: read packed bits + eq directly.
            let d_packed = self.d_packed.as_ref().unwrap();
            unsafe {
                ffi::aext2_sumcheck_batched_round0_binary_msg_ffi(
                    self.d_polys.as_ptr(),
                    d_packed.as_ptr(),
                    self.d_partial.as_mut_ptr(),
                    self.original_size,
                    half,
                    self.num_leaves as c_int,
                    num_blocks_x,
                    self.packed_size_u64,
                )
            }
        } else {
            unsafe {
                ffi::aext2_sumcheck_batched_round_message_ffi(
                    self.d_polys.as_ptr(),
                    self.d_partial.as_mut_ptr(),
                    self.original_size,
                    half,
                    self.num_leaves as c_int,
                    num_blocks_x,
                )
            }
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        // Per-block layout: [block_x][leaf][c][c0, c1]. Reduce across blocks.
        let n_partials = (num_blocks_x as usize) * self.num_leaves * 3;
        let partials = self.d_partial.read_slice(0, n_partials * 2)?;
        let mut result = vec![AlmostGoldilocksExt2::zero(); self.num_leaves * 3];
        for b in 0..(num_blocks_x as usize) {
            for leaf in 0..self.num_leaves {
                for c in 0..3 {
                    let off = ((b * self.num_leaves + leaf) * 3 + c) * 2;
                    let p = AlmostGoldilocksExt2::new(
                        AlmostGoldilocksField(partials[off]),
                        AlmostGoldilocksField(partials[off + 1]),
                    );
                    result[leaf * 3 + c] = aext2_add_host(result[leaf * 3 + c], p);
                }
            }
        }
        Ok(result)
    }

    /// In-place fold of all leaves' (eq, f) by the same challenge `r`.
    pub fn fold(&mut self, challenge: AlmostGoldilocksExt2) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds".to_string()));
        }
        let ret = if self.current_round == 0 && self.d_packed.is_some() {
            // Binary round-0 fused fold: read packed bits + eq, write the
            // Ext2 round-1 eq'/f' into d_scratch. After this the f-poly is
            // dense Ext2, so subsequent rounds use the standard kernel.
            let d_packed = self.d_packed.as_ref().unwrap();
            unsafe {
                ffi::aext2_sumcheck_batched_round0_binary_fold_ffi(
                    self.d_polys.as_ptr(),
                    d_packed.as_ptr(),
                    self.d_scratch.as_mut_ptr(),
                    challenge.c0.0,
                    challenge.c1.0,
                    self.original_size,
                    half,
                    self.num_leaves as c_int,
                    self.packed_size_u64,
                )
            }
        } else {
            unsafe {
                ffi::aext2_sumcheck_batched_fold_ffi(
                    self.d_polys.as_ptr(),
                    self.d_scratch.as_mut_ptr(),
                    challenge.c0.0,
                    challenge.c1.0,
                    self.original_size,
                    half,
                    self.num_leaves as c_int,
                )
            }
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        std::mem::swap(&mut self.d_polys, &mut self.d_scratch);
        self.current_round += 1;
        // The packed bits are only needed for round 0; free them now.
        self.d_packed = None;
        Ok(())
    }

    /// `f_i(R)` for each leaf — read after `current_round == num_vars`.
    pub fn final_f_evals(&self) -> Result<Vec<AlmostGoldilocksExt2>> {
        assert_eq!(self.current_round, self.num_vars,
            "final_f_evals before all rounds done");
        let poly_u64s = self.original_size * 2;
        let leaf_u64s = 2 * poly_u64s;
        let mut out = Vec::with_capacity(self.num_leaves);
        for leaf in 0..self.num_leaves {
            // f is the second poly per leaf — offset by poly_u64s.
            let off = leaf * leaf_u64s + poly_u64s;
            let vals = self.d_polys.read_slice(off, 2)?;
            out.push(AlmostGoldilocksExt2::new(
                AlmostGoldilocksField(vals[0]),
                AlmostGoldilocksField(vals[1]),
            ));
        }
        Ok(out)
    }

    pub fn num_leaves(&self) -> usize { self.num_leaves }
    pub fn num_vars(&self) -> usize { self.num_vars }
}

impl Drop for GpuBatchedSamePointState {
    fn drop(&mut self) {
        // Return the large buffers to the thread-local pool instead of
        // cudaFree'ing — the next fold-tree group reuses them instantly.
        let z = || DeviceBuffer::<u64>::new(0).expect("0-size buffer");
        pool_return(std::mem::replace(&mut self.d_polys, z()));
        pool_return(std::mem::replace(&mut self.d_scratch, z()));
        pool_return(std::mem::replace(&mut self.d_partial, z()));
    }
}

/// F_u shared-eq same-point sumcheck. Pre-combines a group's leaves by
/// claim_pt: `F_u[x] = Σ_{leaf : unique[leaf]==u} α_leaf · f_leaf[x]`. The
/// sumcheck then runs only `num_unique`-wide on the `(eq_u, F_u)` pairs; the
/// combined round message is `Σ_u` of the per-unique `eq_u·F_u` messages (the
/// same-point α-weights are already folded into `F_u`). vs the interleaved
/// [`GpuBatchedSamePointState`]'s `num_leaves`-wide message this cuts the
/// (bandwidth-bound) message work by `num_leaves / num_unique` — ≈21× at
/// level 0 (21 bit-planes of an edge share one eq) and ≈63× at level 1+ (all
/// leaves share `shared_r`). Per-leaf `f_i(R)` for the proof is computed by
/// the caller as a binary MLE eval at the sumcheck challenges.
pub struct GpuSharedEqState {
    /// All eq SUFFIX stages per unique point (factored-eq / Gruen backend):
    /// eq is never materialized at full size and never folded. The folded eq
    /// table at round t equals `prefix[u] · eqsuf_t[y]`, where the prefix is
    /// a host scalar updated on `fold` and eqsuf_t is the challenge-
    /// independent stage at element offset `2^(n-t) - 1` (length `2^(n-t)`)
    /// within each unique's `original_size`-Ext2 stride.
    d_eqsuf: DeviceBuffer<u64>,
    d_fu: DeviceBuffer<u64>,        // num_unique combined-f tables
    d_fu_scratch: DeviceBuffer<u64>,
    d_partial: DeviceBuffer<u64>,
    /// Host copy of the deduped claim points (R_u), for per-round R_{u,t}.
    unique_pts: Vec<Vec<AlmostGoldilocksExt2>>,
    /// p_u = Π_{i<t} eq1(R_{u,i}, r_i); multiplied per round in `fold`.
    prefix: Vec<AlmostGoldilocksExt2>,
    num_unique: usize,
    original_size: usize,
    current_round: usize,
    num_vars: usize,
}

#[inline]
fn aext2_one_host() -> AlmostGoldilocksExt2 {
    AlmostGoldilocksExt2::new(AlmostGoldilocksField(1), AlmostGoldilocksField(0))
}

/// Leaf indices grouped by unique claim-pt + per-unique prefix offsets, the
/// layout `aext2_build_fu_*_kernel` wants: leaves of unique `u` are
/// `sorted[offsets[u] .. offsets[u+1]]`. Lets each (x, u) GPU thread loop
/// only its own unique's leaves instead of scanning all leaves with a
/// divergent branch.
fn sort_leaves_by_unique(leaf_to_unique: &[usize], num_unique: usize) -> (Vec<i32>, Vec<i32>) {
    let mut sorted: Vec<i32> = (0..leaf_to_unique.len() as i32).collect();
    sorted.sort_by_key(|&i| leaf_to_unique[i as usize]);
    let mut offsets = vec![0i32; num_unique + 1];
    for &u in leaf_to_unique {
        offsets[u + 1] += 1;
    }
    for u in 0..num_unique {
        offsets[u + 1] += offsets[u];
    }
    (sorted, offsets)
}

impl GpuSharedEqState {
    /// Build from per-leaf binary `claim_pt`, packed `f` bits, and same-point
    /// `alphas` (α^i per leaf). Dedups claim_pts → `num_unique` eq tables, and
    /// builds the α-combined `F_u` tables on device.
    pub fn new_binary_packed_f(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        packed_fs: &[&[u64]],
        alphas: &[AlmostGoldilocksExt2],
    ) -> Result<Self> {
        assert_eq!(claim_pts.len(), packed_fs.len(), "leaf count mismatch");
        assert_eq!(claim_pts.len(), alphas.len(), "alpha count mismatch");
        assert!(!claim_pts.is_empty(), "empty input");
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        let expected_packed = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
        for (i, p) in packed_fs.iter().enumerate() {
            assert_eq!(p.len(), expected_packed, "leaf {} packed f size mismatch", i);
        }
        let num_leaves = claim_pts.len();
        let poly_u64s = original_size * 2;

        // 1) Dedup claim_pts, build ALL eq suffix stages (factored eq — the
        // full eq table is never materialized; total 2^n − 1 elements per
        // unique vs 2 · 2^n for the old table + fold scratch).
        let (unique_pts, leaf_to_unique) = dedup_claim_pts(claim_pts);
        let num_unique = unique_pts.len();
        let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(num_unique * arity);
        for pt in &unique_pts { r_concat.extend_from_slice(pt); }
        let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
        let unique_u64s = num_unique * poly_u64s;
        let mut d_eqsuf = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_eq_suffix_dp_ffi(
                d_r_all.as_ptr() as *const u64,
                d_eqsuf.as_mut_ptr(),
                arity as c_int,
                num_unique as c_int,
                poly_u64s,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        // 2) Upload packed bits + alphas + map; build F_u (num_unique tables).
        // The upload buffer comes from the SP_POOL (a fresh 126 MB
        // cudaMalloc + synchronizing cudaFree per group costs ~10-20 ms at
        // arity 24), and each leaf's slice uploads directly — no host-side
        // concat copy. Returned to the pool right after the build_fu launch:
        // safe because all subsequent work on this thread serializes behind
        // it on the default stream.
        let total_packed = num_leaves * expected_packed;
        let mut d_packed = pool_take(total_packed)?;
        for (i, p) in packed_fs.iter().enumerate() {
            d_packed.write_slice_at(i * expected_packed, p)?;
        }
        let mut alpha_packed = Vec::with_capacity(num_leaves * 2);
        for a in alphas { alpha_packed.push(a.c0.0); alpha_packed.push(a.c1.0); }
        let d_alphas = DeviceBuffer::<u64>::from_slice(&alpha_packed)?;
        let (sorted_idx, offsets) = sort_leaves_by_unique(&leaf_to_unique, num_unique);
        let d_sorted = DeviceBuffer::<i32>::from_slice(&sorted_idx)?;
        let d_offsets = DeviceBuffer::<i32>::from_slice(&offsets)?;
        let mut d_fu = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_build_fu_ffi(
                d_packed.as_ptr(),
                d_alphas.as_ptr(),
                d_sorted.as_ptr(),
                d_offsets.as_ptr(),
                d_fu.as_mut_ptr(),
                original_size,
                num_unique as c_int,
                expected_packed,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        pool_return(d_packed);
        let d_fu_scratch = pool_take(unique_u64s)?;

        // 1024-block cap (was 256): the factored msg kernel is the only
        // remaining full-table pass per round; 256 blocks undersubscribe an
        // A100 ~3x at large arity. Partial readback stays tiny (≤ 96 KB).
        let max_blocks_x = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks_x = max_blocks_x.min(1024).max(1);
        let d_partial = pool_take(max_blocks_x * num_unique * 2 * 2)?;

        Ok(Self {
            d_eqsuf, d_fu, d_fu_scratch, d_partial,
            prefix: vec![aext2_one_host(); num_unique],
            unique_pts: unique_pts.iter().map(|p| p.to_vec()).collect(),
            num_unique, original_size, current_round: 0, num_vars: arity,
        })
    }

    /// Device-input variant of [`Self::new_binary_packed_f`]: the per-leaf
    /// packed bits are ALREADY on the current device as one concat buffer
    /// (leaf i at element offset `i · packed_size`). Skips the host upload
    /// entirely — the device-resident fold-tree path assembles this buffer
    /// once per group and shares it with the multifold kernel.
    pub fn new_binary_packed_f_dev(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        d_packed: &DeviceBuffer<u64>,
        alphas: &[AlmostGoldilocksExt2],
    ) -> Result<Self> {
        assert_eq!(claim_pts.len(), alphas.len(), "alpha count mismatch");
        assert!(!claim_pts.is_empty(), "empty input");
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        let expected_packed = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
        let num_leaves = claim_pts.len();
        assert!(d_packed.len() >= num_leaves * expected_packed,
            "device packed buffer too small: {} < {}",
            d_packed.len(), num_leaves * expected_packed);
        let poly_u64s = original_size * 2;

        let (unique_pts, leaf_to_unique) = dedup_claim_pts(claim_pts);
        let num_unique = unique_pts.len();
        let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(num_unique * arity);
        for pt in &unique_pts { r_concat.extend_from_slice(pt); }
        let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
        let unique_u64s = num_unique * poly_u64s;
        let mut d_eqsuf = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_eq_suffix_dp_ffi(
                d_r_all.as_ptr() as *const u64,
                d_eqsuf.as_mut_ptr(),
                arity as c_int, num_unique as c_int, poly_u64s,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let mut alpha_packed = Vec::with_capacity(num_leaves * 2);
        for a in alphas { alpha_packed.push(a.c0.0); alpha_packed.push(a.c1.0); }
        let d_alphas = DeviceBuffer::<u64>::from_slice(&alpha_packed)?;
        let (sorted_idx, offsets) = sort_leaves_by_unique(&leaf_to_unique, num_unique);
        let d_sorted = DeviceBuffer::<i32>::from_slice(&sorted_idx)?;
        let d_offsets = DeviceBuffer::<i32>::from_slice(&offsets)?;
        let mut d_fu = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_build_fu_ffi(
                d_packed.as_ptr(),
                d_alphas.as_ptr(),
                d_sorted.as_ptr(),
                d_offsets.as_ptr(),
                d_fu.as_mut_ptr(),
                original_size,
                num_unique as c_int,
                expected_packed,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        let d_fu_scratch = pool_take(unique_u64s)?;
        let max_blocks_x = (((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE)
            .min(1024).max(1);
        let d_partial = pool_take(max_blocks_x * num_unique * 2 * 2)?;
        Ok(Self {
            d_eqsuf, d_fu, d_fu_scratch, d_partial,
            prefix: vec![aext2_one_host(); num_unique],
            unique_pts: unique_pts.iter().map(|p| p.to_vec()).collect(),
            num_unique, original_size, current_round: 0, num_vars: arity,
        })
    }

    /// Device-input variant of [`Self::new_ternary_packed`]: pos/neg chunk
    /// planes already on the current device as concat buffers (leaf i at
    /// element offset `i · packed_size`).
    pub fn new_ternary_packed_dev(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        d_pos: &DeviceBuffer<u64>,
        d_neg: &DeviceBuffer<u64>,
        alphas: &[AlmostGoldilocksExt2],
    ) -> Result<Self> {
        assert_eq!(claim_pts.len(), alphas.len(), "alpha count mismatch");
        assert!(!claim_pts.is_empty(), "empty input");
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        let expected_packed = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
        let num_leaves = claim_pts.len();
        assert!(d_pos.len() >= num_leaves * expected_packed
             && d_neg.len() >= num_leaves * expected_packed,
            "device pos/neg buffers too small");
        let poly_u64s = original_size * 2;

        let (unique_pts, leaf_to_unique) = dedup_claim_pts(claim_pts);
        let num_unique = unique_pts.len();
        let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(num_unique * arity);
        for pt in &unique_pts { r_concat.extend_from_slice(pt); }
        let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
        let unique_u64s = num_unique * poly_u64s;
        let mut d_eqsuf = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_eq_suffix_dp_ffi(
                d_r_all.as_ptr() as *const u64,
                d_eqsuf.as_mut_ptr(),
                arity as c_int, num_unique as c_int, poly_u64s,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let mut alpha_packed = Vec::with_capacity(num_leaves * 2);
        for a in alphas { alpha_packed.push(a.c0.0); alpha_packed.push(a.c1.0); }
        let d_alphas = DeviceBuffer::<u64>::from_slice(&alpha_packed)?;
        let (sorted_idx, offsets) = sort_leaves_by_unique(&leaf_to_unique, num_unique);
        let d_sorted = DeviceBuffer::<i32>::from_slice(&sorted_idx)?;
        let d_offsets = DeviceBuffer::<i32>::from_slice(&offsets)?;
        let mut d_fu = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_build_fu_ternary_ffi(
                d_pos.as_ptr(), d_neg.as_ptr(), d_alphas.as_ptr(),
                d_sorted.as_ptr(), d_offsets.as_ptr(),
                d_fu.as_mut_ptr(), original_size,
                num_unique as c_int, expected_packed,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        let d_fu_scratch = pool_take(unique_u64s)?;
        let max_blocks_x = (((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE)
            .min(1024).max(1);
        let d_partial = pool_take(max_blocks_x * num_unique * 2 * 2)?;
        Ok(Self {
            d_eqsuf, d_fu, d_fu_scratch, d_partial,
            prefix: vec![aext2_one_host(); num_unique],
            unique_pts: unique_pts.iter().map(|p| p.to_vec()).collect(),
            num_unique, original_size, current_round: 0, num_vars: arity,
        })
    }

    /// As [`Self::new_binary_packed_f`] but for single-chunk ternary leaves:
    /// `F_u[x] = Σ_{leaf∈u} α_leaf·(pos_leaf[x] - neg_leaf[x])`. This is the
    /// level-1+ fold-tree case where every leaf shares `shared_r`
    /// (num_unique = 1), so the sumcheck collapses 63-wide → 1-wide.
    pub fn new_ternary_packed(
        claim_pts: &[Vec<AlmostGoldilocksExt2>],
        pos_per_leaf: &[&[u64]],
        neg_per_leaf: &[&[u64]],
        alphas: &[AlmostGoldilocksExt2],
    ) -> Result<Self> {
        assert_eq!(claim_pts.len(), pos_per_leaf.len(), "leaf count mismatch");
        assert_eq!(claim_pts.len(), neg_per_leaf.len(), "leaf count mismatch");
        assert_eq!(claim_pts.len(), alphas.len(), "alpha count mismatch");
        assert!(!claim_pts.is_empty(), "empty input");
        let arity = claim_pts[0].len();
        let original_size = 1usize << arity;
        let expected_packed = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
        let num_leaves = claim_pts.len();
        let poly_u64s = original_size * 2;

        // 1) Dedup + eq suffix stages (factored eq — see binary constructor).
        let (unique_pts, leaf_to_unique) = dedup_claim_pts(claim_pts);
        let num_unique = unique_pts.len();
        let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(num_unique * arity);
        for pt in &unique_pts { r_concat.extend_from_slice(pt); }
        let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
        let unique_u64s = num_unique * poly_u64s;
        let mut d_eqsuf = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_eq_suffix_dp_ffi(
                d_r_all.as_ptr() as *const u64,
                d_eqsuf.as_mut_ptr(),
                arity as c_int, num_unique as c_int, poly_u64s,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        // 2) Upload pos/neg + alphas + map; build F_u. Pool-backed direct
        // per-leaf uploads — see the binary constructor for rationale.
        let total_packed = num_leaves * expected_packed;
        let mut d_pos = pool_take(total_packed)?;
        let mut d_neg = pool_take(total_packed)?;
        for (i, p) in pos_per_leaf.iter().enumerate() {
            d_pos.write_slice_at(i * expected_packed, p)?;
        }
        for (i, p) in neg_per_leaf.iter().enumerate() {
            d_neg.write_slice_at(i * expected_packed, p)?;
        }
        let mut alpha_packed = Vec::with_capacity(num_leaves * 2);
        for a in alphas { alpha_packed.push(a.c0.0); alpha_packed.push(a.c1.0); }
        let d_alphas = DeviceBuffer::<u64>::from_slice(&alpha_packed)?;
        let (sorted_idx, offsets) = sort_leaves_by_unique(&leaf_to_unique, num_unique);
        let d_sorted = DeviceBuffer::<i32>::from_slice(&sorted_idx)?;
        let d_offsets = DeviceBuffer::<i32>::from_slice(&offsets)?;
        let mut d_fu = pool_take(unique_u64s)?;
        let ret = unsafe {
            ffi::aext2_build_fu_ternary_ffi(
                d_pos.as_ptr(), d_neg.as_ptr(), d_alphas.as_ptr(),
                d_sorted.as_ptr(), d_offsets.as_ptr(),
                d_fu.as_mut_ptr(), original_size,
                num_unique as c_int, expected_packed,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        pool_return(d_pos);
        pool_return(d_neg);
        let d_fu_scratch = pool_take(unique_u64s)?;

        let max_blocks_x = (((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE)
            .min(1024).max(1);
        let d_partial = pool_take(max_blocks_x * num_unique * 2 * 2)?;

        Ok(Self {
            d_eqsuf, d_fu, d_fu_scratch, d_partial,
            prefix: vec![aext2_one_host(); num_unique],
            unique_pts: unique_pts.iter().map(|p| p.to_vec()).collect(),
            num_unique, original_size, current_round: 0, num_vars: arity,
        })
    }

    /// Per-unique degree-2 round message `[T(0), T(1), T(2)]`, flat length
    /// `3 * num_unique`. The caller's combined message is `Σ_u` of these
    /// (α already folded into F_u).
    ///
    /// Factored-eq path: the GPU computes only A_u = Σ_y eqsuf[y]·F_u[2y]
    /// and B_u = Σ_y eqsuf[y]·F_u[2y+1]; the host assembles
    ///   T(0) = p_u·(1−R_t)·A,  T(1) = p_u·R_t·B,  T(2) = p_u·(3R_t−1)·(2B−A)
    /// which equals the fold-based message exactly (the folded eq table is
    /// p_u·eqsuf_t by induction), so the transcript is unchanged.
    pub fn compute_round_messages(&mut self) -> Result<Vec<AlmostGoldilocksExt2>> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds".to_string()));
        }
        let num_blocks_x = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(1024).max(1) as c_int;
        let t = self.current_round;
        // eqsuf_{t+1} lives at element offset 2^(n−t−1) − 1, length = half.
        let eqsuf_off = (1usize << (self.num_vars - t - 1)) - 1;
        let poly_stride = self.original_size * 2;
        let ret = unsafe {
            ffi::aext2_sharedeq_factored_msg_ffi(
                self.d_eqsuf.as_ptr(),
                self.d_fu.as_ptr(),
                self.d_partial.as_mut_ptr(),
                eqsuf_off,
                poly_stride,
                poly_stride,
                half,
                self.num_unique as c_int,
                num_blocks_x,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let n_partials = (num_blocks_x as usize) * self.num_unique * 2;
        let partials = self.d_partial.read_slice(0, n_partials * 2)?;
        let zero = AlmostGoldilocksExt2::zero();
        let mut sums = vec![zero; self.num_unique * 2]; // (A_u, B_u)
        for b in 0..(num_blocks_x as usize) {
            for u in 0..self.num_unique {
                for c in 0..2 {
                    let off = ((b * self.num_unique + u) * 2 + c) * 2;
                    let p = AlmostGoldilocksExt2::new(
                        AlmostGoldilocksField(partials[off]),
                        AlmostGoldilocksField(partials[off + 1]),
                    );
                    sums[u * 2 + c] = aext2_add_host(sums[u * 2 + c], p);
                }
            }
        }
        let one = aext2_one_host();
        let mut result = vec![zero; self.num_unique * 3];
        for u in 0..self.num_unique {
            let a = sums[u * 2];
            let b = sums[u * 2 + 1];
            let r_t = self.unique_pts[u][t];
            let p = self.prefix[u];
            result[u * 3]     = p * (one - r_t) * a;
            result[u * 3 + 1] = p * r_t * b;
            result[u * 3 + 2] = p * (r_t + r_t + r_t - one) * (b + b - a);
        }
        Ok(result)
    }

    /// Fold F_u by `challenge` and fold the eq prefix scalar analytically:
    /// p_u ← p_u · eq1(R_{u,t}, r) with eq1(R, r) = (1−R)(1−r) + R·r. The
    /// eq tables themselves are never folded (factored-eq backend).
    pub fn fold(&mut self, challenge: AlmostGoldilocksExt2) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds".to_string()));
        }
        let ret = unsafe {
            ffi::aext2_fold_single_ffi(
                self.d_fu.as_ptr(), self.d_fu_scratch.as_mut_ptr(),
                challenge.c0.0, challenge.c1.0,
                self.original_size, half, self.num_unique as c_int,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        std::mem::swap(&mut self.d_fu, &mut self.d_fu_scratch);
        let one = aext2_one_host();
        let t = self.current_round;
        for u in 0..self.num_unique {
            let r_t = self.unique_pts[u][t];
            let e = (one - r_t) * (one - challenge) + r_t * challenge;
            self.prefix[u] = self.prefix[u] * e;
        }
        self.current_round += 1;
        Ok(())
    }

    pub fn num_unique(&self) -> usize { self.num_unique }
    pub fn num_vars(&self) -> usize { self.num_vars }
}

impl Drop for GpuSharedEqState {
    fn drop(&mut self) {
        let z = || DeviceBuffer::<u64>::new(0).expect("0-size buffer");
        pool_return(std::mem::replace(&mut self.d_eqsuf, z()));
        pool_return(std::mem::replace(&mut self.d_fu, z()));
        pool_return(std::mem::replace(&mut self.d_fu_scratch, z()));
        pool_return(std::mem::replace(&mut self.d_partial, z()));
    }
}

/// Factored-eq (Gruen) degree-2 sumcheck for the streaming reducer:
/// `Σ_x f(x)·(eq(r0,x) + α·eq(r1,x))`, reducing two claims about one
/// dense Ext2 witness `f` (= the accumulated weight) to a single claim.
///
/// Unlike [`GpuSharedEqState`] (binary/ternary `f` built from packed
/// planes), here `f` is the cached dense Ext2 witness, and there are TWO
/// eq sources over ONE shared `f`. The eq tables are NEVER materialized
/// or folded — all suffix stages are precomputed once (factored), and the
/// shared `f` is folded once per round. The two eq sources reuse the
/// same factored-msg kernel in one launch via `poly_stride_u64 = 0` (both
/// units read the same `f`); the host combines the per-source
/// `[T0,T1,T2]` with weights `[1, α]`.
///
/// Round messages and the final `(f(r_new), eq_combined(r_new))` are
/// identical to the materialized 2-poly sumcheck, so the proof wire
/// format and the verifier are unchanged.
pub struct GpuReducerFactoredState {
    d_eqsuf: DeviceBuffer<u64>,   // 2 suffix stacks (r0, r1), stride poly_u64s
    d_f: DeviceBuffer<u64>,        // dense Ext2 witness, folded each round
    d_f_scratch: DeviceBuffer<u64>,
    d_partial: DeviceBuffer<u64>,
    r0: Vec<AlmostGoldilocksExt2>,
    r1: Vec<AlmostGoldilocksExt2>,
    alpha: AlmostGoldilocksExt2,
    p0: AlmostGoldilocksExt2,
    p1: AlmostGoldilocksExt2,
    original_size: usize,
    current_round: usize,
    num_vars: usize,
}

impl GpuReducerFactoredState {
    /// `d_x_ext2` is the (cached, NOT mutated) dense Ext2 witness — copied
    /// into a fold scratch. `r0`/`r1` are the two claim points (length n).
    pub fn new(
        d_x_ext2: &DeviceBuffer<u64>,
        r0: &[AlmostGoldilocksExt2],
        r1: &[AlmostGoldilocksExt2],
        alpha: AlmostGoldilocksExt2,
    ) -> Result<Self> {
        let n = r0.len();
        assert_eq!(r1.len(), n, "r0/r1 length mismatch");
        let original_size = 1usize << n;
        let poly_u64s = original_size * 2;
        assert_eq!(d_x_ext2.len(), poly_u64s, "witness buffer size mismatch");

        // Suffix stages for both points in one dp (num_unique = 2).
        let mut r_concat: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(2 * n);
        r_concat.extend_from_slice(r0);
        r_concat.extend_from_slice(r1);
        let d_r_all = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&r_concat)?;
        let mut d_eqsuf = pool_take(2 * poly_u64s)?;
        let ret = unsafe {
            ffi::aext2_eq_suffix_dp_ffi(
                d_r_all.as_ptr() as *const u64,
                d_eqsuf.as_mut_ptr(),
                n as c_int, 2, poly_u64s,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        // Copy witness into the fold buffer (cache must stay intact).
        let mut d_f = pool_take(poly_u64s)?;
        d_f.copy_range_from_device(0, d_x_ext2, 0, poly_u64s)?;
        let d_f_scratch = pool_take(poly_u64s)?;

        let max_blocks_x = (((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE)
            .min(1024).max(1);
        let d_partial = pool_take(max_blocks_x * 2 * 2 * 2)?;

        Ok(Self {
            d_eqsuf, d_f, d_f_scratch, d_partial,
            r0: r0.to_vec(), r1: r1.to_vec(), alpha,
            p0: aext2_one_host(), p1: aext2_one_host(),
            original_size, current_round: 0, num_vars: n,
        })
    }

    /// Degree-2 round message `[T(0), T(1), T(2)]` for the current round.
    pub fn compute_round_message(&mut self) -> Result<[AlmostGoldilocksExt2; 3]> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds".to_string()));
        }
        let num_blocks_x = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(1024).max(1) as c_int;
        let t = self.current_round;
        // eqsuf_{t+1} stage offset (same as same-point factored path).
        let eqsuf_off = (1usize << (self.num_vars - t - 1)) - 1;
        let poly_stride = self.original_size * 2;
        let ret = unsafe {
            ffi::aext2_sharedeq_factored_msg_ffi(
                self.d_eqsuf.as_ptr(),
                self.d_f.as_ptr(),
                self.d_partial.as_mut_ptr(),
                eqsuf_off,
                poly_stride,   // eqsuf stride: 2 stacks poly_u64s apart
                0,             // f stride 0 → both eq sources share one f
                half,
                2,             // num_unique = 2 eq sources
                num_blocks_x,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }

        let n_partials = (num_blocks_x as usize) * 2 * 2;
        let partials = self.d_partial.read_slice(0, n_partials * 2)?;
        let zero = AlmostGoldilocksExt2::zero();
        let mut sums = [zero; 4]; // A0, B0, A1, B1
        for b in 0..(num_blocks_x as usize) {
            for u in 0..2 {
                for c in 0..2 {
                    let off = ((b * 2 + u) * 2 + c) * 2;
                    let p = AlmostGoldilocksExt2::new(
                        AlmostGoldilocksField(partials[off]),
                        AlmostGoldilocksField(partials[off + 1]),
                    );
                    sums[u * 2 + c] = aext2_add_host(sums[u * 2 + c], p);
                }
            }
        }
        let one = aext2_one_host();
        let two = one + one;
        let three = two + one;
        let assemble = |p: AlmostGoldilocksExt2, r: AlmostGoldilocksExt2,
                        a: AlmostGoldilocksExt2, b: AlmostGoldilocksExt2| {
            [
                p * (one - r) * a,
                p * r * b,
                p * (three * r - one) * (two * b - a),
            ]
        };
        let s0 = assemble(self.p0, self.r0[t], sums[0], sums[1]);
        let s1 = assemble(self.p1, self.r1[t], sums[2], sums[3]);
        Ok([
            s0[0] + self.alpha * s1[0],
            s0[1] + self.alpha * s1[1],
            s0[2] + self.alpha * s1[2],
        ])
    }

    /// Fold the shared `f` once and update both eq prefix scalars.
    pub fn fold(&mut self, challenge: AlmostGoldilocksExt2) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds".to_string()));
        }
        let ret = unsafe {
            ffi::aext2_fold_single_ffi(
                self.d_f.as_ptr(), self.d_f_scratch.as_mut_ptr(),
                challenge.c0.0, challenge.c1.0,
                self.original_size, half, 1,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        std::mem::swap(&mut self.d_f, &mut self.d_f_scratch);
        let one = aext2_one_host();
        let t = self.current_round;
        let e0 = (one - self.r0[t]) * (one - challenge) + self.r0[t] * challenge;
        let e1 = (one - self.r1[t]) * (one - challenge) + self.r1[t] * challenge;
        self.p0 = self.p0 * e0;
        self.p1 = self.p1 * e1;
        self.current_round += 1;
        Ok(())
    }

    /// `f(r_new)` — the fully-folded witness scalar (after `n` folds).
    pub fn f_final(&self) -> Result<AlmostGoldilocksExt2> {
        let v = self.d_f.read_slice(0, 2)?;
        Ok(AlmostGoldilocksExt2::new(
            AlmostGoldilocksField(v[0]), AlmostGoldilocksField(v[1])))
    }

    /// `eq_combined(r_new) = eq(r0,c) + α·eq(r1,c) = p0 + α·p1` (after folds).
    pub fn eq_combined_final(&self) -> AlmostGoldilocksExt2 {
        self.p0 + self.alpha * self.p1
    }
}

impl Drop for GpuReducerFactoredState {
    fn drop(&mut self) {
        let z = || DeviceBuffer::<u64>::new(0).expect("0-size buffer");
        pool_return(std::mem::replace(&mut self.d_eqsuf, z()));
        pool_return(std::mem::replace(&mut self.d_f, z()));
        pool_return(std::mem::replace(&mut self.d_f_scratch, z()));
        pool_return(std::mem::replace(&mut self.d_partial, z()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(agl(v))
    }

    /// On-device eq construction must produce a bit-identical state to
    /// the all-host construction. Run a few sumcheck rounds and check
    /// round messages match.
    #[test]
    fn batched_same_point_device_eq_matches_host() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 8;
        let size = 1usize << n_var;
        let pt0: Vec<AlmostGoldilocksExt2> = (0..n_var as u64).map(|i| lift(i * 3 + 1)).collect();
        let pt1: Vec<AlmostGoldilocksExt2> = (0..n_var as u64).map(|i| lift(i * 7 + 11)).collect();
        let f0:  Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i * 13 + 5)).collect();
        let f1:  Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i * 17 + 19)).collect();
        let eq0 = crate::eq_lagrange::ext2_eq_dp_all(&pt0).expect("eq host 0");
        let eq1 = crate::eq_lagrange::ext2_eq_dp_all(&pt1).expect("eq host 1");

        let per_leaf_host = vec![(eq0.clone(), f0.clone()), (eq1.clone(), f1.clone())];
        let mut st_host = GpuBatchedSamePointState::new(&per_leaf_host).expect("host ctor");

        let claim_pts = vec![pt0, pt1];
        let fs = vec![f0, f1];
        let mut st_dev = GpuBatchedSamePointState::new_device_eq(&claim_pts, &fs).expect("device-eq ctor");

        // Run all rounds, comparing round messages + folding by the
        // same challenge so subsequent rounds stay in sync.
        for round in 0..n_var {
            let m_h = st_host.compute_round_messages().expect("host msg");
            let m_d = st_dev.compute_round_messages().expect("dev msg");
            for c in 0..(2 * 3) {
                assert_eq!(m_h[c].c0.0, m_d[c].c0.0, "round {} c={} c0", round, c);
                assert_eq!(m_h[c].c1.0, m_d[c].c1.0, "round {} c={} c1", round, c);
            }
            let r = lift((round as u64 + 1) * 23);
            st_host.fold(r).expect("host fold");
            st_dev.fold(r).expect("dev fold");
        }
    }

    /// On-device eq + on-device binary lift must match the all-host
    /// construction over all rounds.
    #[test]
    fn batched_same_point_device_eq_packed_f_matches_host() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 8;
        let size = 1usize << n_var;
        let pt0: Vec<AlmostGoldilocksExt2> = (0..n_var as u64).map(|i| lift(i * 5 + 3)).collect();
        let pt1: Vec<AlmostGoldilocksExt2> = (0..n_var as u64).map(|i| lift(i * 9 + 7)).collect();
        // Packed binary witnesses (n_var=8 → 1 u64 each since size=256).
        let packed0: Vec<u64> = vec![0xCAFEBABE_DEADBEEFu64; size / 64];
        let packed1: Vec<u64> = vec![0xA5A5A5A5_5A5A5A5Au64; size / 64];

        // Reference: lift on host.
        let lift = |packed: &[u64]| -> Vec<AlmostGoldilocksExt2> {
            let mut out = Vec::with_capacity(size);
            for w in packed {
                for k in 0..64 {
                    let bit = ((w >> k) & 1) as u64;
                    out.push(AlmostGoldilocksExt2::new(
                        AlmostGoldilocksField(bit),
                        AlmostGoldilocksField(0),
                    ));
                }
            }
            out.truncate(size);
            out
        };
        let f0 = lift(&packed0);
        let f1 = lift(&packed1);
        let eq0 = crate::eq_lagrange::ext2_eq_dp_all(&pt0).expect("eq host 0");
        let eq1 = crate::eq_lagrange::ext2_eq_dp_all(&pt1).expect("eq host 1");
        let per_leaf_host = vec![(eq0, f0), (eq1, f1)];
        let mut st_host = GpuBatchedSamePointState::new(&per_leaf_host).expect("host ctor");

        let claim_pts = vec![pt0, pt1];
        let packed_refs: Vec<&[u64]> = vec![packed0.as_slice(), packed1.as_slice()];
        let mut st_dev = GpuBatchedSamePointState::new_device_eq_packed_f(&claim_pts, &packed_refs)
            .expect("device-eq packed-f ctor");

        for round in 0..n_var {
            let m_h = st_host.compute_round_messages().expect("host msg");
            let m_d = st_dev.compute_round_messages().expect("dev msg");
            for c in 0..(2 * 3) {
                assert_eq!(m_h[c].c0.0, m_d[c].c0.0, "round {} c={} c0", round, c);
                assert_eq!(m_h[c].c1.0, m_d[c].c1.0, "round {} c={} c1", round, c);
            }
            let r = lift_ext2((round as u64 + 1) * 29);
            st_host.fold(r).expect("host fold");
            st_dev.fold(r).expect("dev fold");
        }
    }

    fn lift_ext2(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    /// Ternary on-device lift parity: build via host-lift (pos-neg → Ext2)
    /// vs new_device_eq_packed_ternary should produce equivalent round
    /// messages across all rounds.
    #[test]
    fn batched_same_point_device_eq_packed_ternary_matches_host() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 8;
        let size = 1usize << n_var;
        let pt0: Vec<AlmostGoldilocksExt2> = (0..n_var as u64).map(|i| lift(i * 5 + 3)).collect();
        let pt1: Vec<AlmostGoldilocksExt2> = (0..n_var as u64).map(|i| lift(i * 9 + 7)).collect();
        // Disjoint pos/neg masks per leaf.
        let pos0: Vec<u64> = vec![0xAA00AA00AA00AA00u64; size / 64];
        let neg0: Vec<u64> = vec![0x0055005500550055u64; size / 64];
        let pos1: Vec<u64> = vec![0x12345678ABCDEF00u64; size / 64];
        let neg1: Vec<u64> = vec![0x0000000054321000u64; size / 64];
        // Ensure pos & neg disjoint.
        let conflict0 = pos0[0] & neg0[0];
        let conflict1 = pos1[0] & neg1[0];
        assert_eq!(conflict0, 0); assert_eq!(conflict1, 0);

        // Host reference: lift to Ext2.
        let lift_t = |pos: &[u64], neg: &[u64]| -> Vec<AlmostGoldilocksExt2> {
            let mut out = Vec::with_capacity(size);
            for (i, (&pw, &nw)) in pos.iter().zip(neg.iter()).enumerate() {
                for k in 0..64 {
                    let pos_bit = (pw >> k) & 1;
                    let neg_bit = (nw >> k) & 1;
                    let val = if pos_bit == 1 {
                        AlmostGoldilocksExt2::new(AlmostGoldilocksField(1), AlmostGoldilocksField(0))
                    } else if neg_bit == 1 {
                        AlmostGoldilocksExt2::new(
                            AlmostGoldilocksField(crate::field::ALMOST_GOLDILOCKS_PRIME - 1),
                            AlmostGoldilocksField(0),
                        )
                    } else {
                        AlmostGoldilocksExt2::zero()
                    };
                    out.push(val);
                    let _ = i;
                }
            }
            out.truncate(size);
            out
        };
        let f0 = lift_t(&pos0, &neg0);
        let f1 = lift_t(&pos1, &neg1);
        let eq0 = crate::eq_lagrange::ext2_eq_dp_all(&pt0).expect("eq0");
        let eq1 = crate::eq_lagrange::ext2_eq_dp_all(&pt1).expect("eq1");
        let per_leaf_host = vec![(eq0, f0), (eq1, f1)];
        let mut st_host = GpuBatchedSamePointState::new(&per_leaf_host).expect("host ctor");

        let claim_pts = vec![pt0, pt1];
        let pos_refs: Vec<&[u64]> = vec![pos0.as_slice(), pos1.as_slice()];
        let neg_refs: Vec<&[u64]> = vec![neg0.as_slice(), neg1.as_slice()];
        let mut st_dev = GpuBatchedSamePointState::new_device_eq_packed_ternary(
            &claim_pts, &pos_refs, &neg_refs,
        ).expect("dev ternary ctor");

        for round in 0..n_var {
            let m_h = st_host.compute_round_messages().expect("host msg");
            let m_d = st_dev.compute_round_messages().expect("dev msg");
            for c in 0..(2 * 3) {
                assert_eq!(m_h[c].c0.0, m_d[c].c0.0, "round {} c={}", round, c);
                assert_eq!(m_h[c].c1.0, m_d[c].c1.0);
            }
            let r = lift_ext2((round as u64 + 1) * 31);
            st_host.fold(r).expect("host fold");
            st_dev.fold(r).expect("dev fold");
        }
    }

    /// Single-leaf at arity 18 — matches the exp_pipeline ExpHelper aux size.
    /// Compare against an unbatched per-leaf state.
    #[test]
    fn batched_same_point_arity18_single_leaf() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 18;
        let size = 1usize << n_var;
        // Deterministic pseudo-random fill.
        let eq: Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift((i.wrapping_mul(7)) ^ 0xA5A5)).collect();
        let f:  Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift((i.wrapping_mul(11)) ^ 0x5A5A)).collect();
        let per_leaf = vec![(eq.clone(), f.clone())];
        let mut st_batched = GpuBatchedSamePointState::new(&per_leaf).expect("batched new");
        // Reference: unbatched single-leaf state.
        let refs: Vec<&[AlmostGoldilocksExt2]> = vec![&eq, &f];
        let mut st_ref = GpuSumcheckStateExt2::new(&refs).expect("ref new");

        for round in 0..n_var {
            let m_b = st_batched.compute_round_messages().expect("batched msg");
            let m_r = st_ref.compute_round_message().expect("ref msg");
            assert_eq!(m_b.len(), 3, "round {} batched len", round);
            assert_eq!(m_r.len(), 3, "round {} ref len", round);
            for c in 0..3 {
                assert_eq!(m_b[c].c0.0, m_r[c].c0.0,
                    "round {} c={} c0 mismatch", round, c);
                assert_eq!(m_b[c].c1.0, m_r[c].c1.0,
                    "round {} c={} c1 mismatch", round, c);
            }
            let r = lift((round as u64 + 1) * 17);
            st_batched.fold(r).expect("batched fold");
            st_ref.fold(r).expect("ref fold");
        }
        let final_b = st_batched.final_f_evals().expect("batched final");
        let final_r = st_ref.final_evaluations().expect("ref final");
        assert_eq!(final_b[0].c0.0, final_r[1].c0.0, "final c0");
        assert_eq!(final_b[0].c1.0, final_r[1].c1.0, "final c1");
    }

    /// Single-leaf batched same-point — exercises the K=1 code path.
    #[test]
    fn batched_same_point_single_leaf() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 3;
        let size = 1usize << n_var;
        let eq: Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i + 1)).collect();
        let f:  Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(2*i + 3)).collect();
        let per_leaf = vec![(eq.clone(), f.clone())];
        let mut st = GpuBatchedSamePointState::new(&per_leaf).expect("new");

        let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
        let half = size / 2;
        let mut t = [AlmostGoldilocksExt2::zero(); 3];
        for j in 0..half {
            let e0 = eq[2*j]; let e1 = eq[2*j+1];
            let f0 = f[2*j];  let f1 = f[2*j+1];
            let e2 = (two * e1) - e0;
            let f2 = (two * f1) - f0;
            t[0] = aext2_add_host(t[0], e0 * f0);
            t[1] = aext2_add_host(t[1], e1 * f1);
            t[2] = aext2_add_host(t[2], e2 * f2);
        }
        let gpu = st.compute_round_messages().expect("gpu");
        assert_eq!(gpu.len(), 3);
        for c in 0..3 {
            assert_eq!(gpu[c].c0.0, t[c].c0.0);
            assert_eq!(gpu[c].c1.0, t[c].c1.0);
        }
    }

    /// Batched same-point: K=2 leaves, each with (eq, f) of size 8.
    /// Compare against CPU degree-2 round message.
    #[test]
    fn batched_same_point_round0_matches_cpu() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 3;
        let size = 1usize << n_var;
        let eq0: Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i + 1)).collect();
        let f0:  Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(2*i + 3)).collect();
        let eq1: Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i * 5 + 7)).collect();
        let f1:  Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i * 11 + 13)).collect();

        let per_leaf = vec![(eq0.clone(), f0.clone()), (eq1.clone(), f1.clone())];
        let mut st = GpuBatchedSamePointState::new(&per_leaf).expect("batched new");

        // CPU reference: T_k(c) = Σ_j (eq[2j] + c*(eq[2j+1]-eq[2j])) · (f[2j] + c*(f[2j+1]-f[2j])).
        let cpu_msg = |eq: &[AlmostGoldilocksExt2], f: &[AlmostGoldilocksExt2]| -> [AlmostGoldilocksExt2; 3] {
            let half = eq.len() / 2;
            let mut t = [AlmostGoldilocksExt2::zero(); 3];
            let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
            for j in 0..half {
                let e0 = eq[2*j]; let e1 = eq[2*j+1];
                let f0 = f[2*j];  let f1 = f[2*j+1];
                let e2 = (two * e1) - e0;
                let f2 = (two * f1) - f0;
                t[0] = aext2_add_host(t[0], e0 * f0);
                t[1] = aext2_add_host(t[1], e1 * f1);
                t[2] = aext2_add_host(t[2], e2 * f2);
            }
            t
        };
        let exp0 = cpu_msg(&eq0, &f0);
        let exp1 = cpu_msg(&eq1, &f1);

        let gpu_msg = st.compute_round_messages().expect("gpu compute");
        assert_eq!(gpu_msg.len(), 6);
        for c in 0..3 {
            assert_eq!(gpu_msg[0 * 3 + c].c0.0, exp0[c].c0.0, "leaf0 c={}", c);
            assert_eq!(gpu_msg[0 * 3 + c].c1.0, exp0[c].c1.0);
            assert_eq!(gpu_msg[1 * 3 + c].c0.0, exp1[c].c0.0, "leaf1 c={}", c);
            assert_eq!(gpu_msg[1 * 3 + c].c1.0, exp1[c].c1.0);
        }

        // Now fold both leaves by r = 7 and check round 1.
        let r = lift(7);
        st.fold(r).expect("fold");
        let cpu_fold = |eq: &[AlmostGoldilocksExt2], f: &[AlmostGoldilocksExt2]| -> (Vec<AlmostGoldilocksExt2>, Vec<AlmostGoldilocksExt2>) {
            let half = eq.len() / 2;
            let mut neq = Vec::with_capacity(half);
            let mut nf  = Vec::with_capacity(half);
            for j in 0..half {
                let a = eq[2*j]; let b = eq[2*j+1];
                neq.push(aext2_add_host(a, r * (b - a)));
                let af = f[2*j]; let bf = f[2*j+1];
                nf.push(aext2_add_host(af, r * (bf - af)));
            }
            (neq, nf)
        };
        let (neq0, nf0) = cpu_fold(&eq0, &f0);
        let (neq1, nf1) = cpu_fold(&eq1, &f1);
        let exp0_r1 = cpu_msg(&neq0, &nf0);
        let exp1_r1 = cpu_msg(&neq1, &nf1);
        let gpu_msg1 = st.compute_round_messages().expect("gpu compute r1");
        for c in 0..3 {
            assert_eq!(gpu_msg1[0 * 3 + c].c0.0, exp0_r1[c].c0.0, "r1 leaf0 c={}", c);
            assert_eq!(gpu_msg1[1 * 3 + c].c0.0, exp1_r1[c].c0.0, "r1 leaf1 c={}", c);
        }

        // Fold + round 2.
        let r2 = lift(11);
        st.fold(r2).expect("fold2");
        let (neq0_2, nf0_2) = (|eq: &[AlmostGoldilocksExt2], f: &[AlmostGoldilocksExt2]| {
            let half = eq.len() / 2;
            let mut ne = Vec::with_capacity(half);
            let mut nf = Vec::with_capacity(half);
            for j in 0..half {
                ne.push(aext2_add_host(eq[2*j], r2 * (eq[2*j+1] - eq[2*j])));
                nf.push(aext2_add_host(f[2*j],  r2 * (f[2*j+1]  - f[2*j])));
            }
            (ne, nf)
        })(&neq0, &nf0);
        let (neq1_2, nf1_2) = (|eq: &[AlmostGoldilocksExt2], f: &[AlmostGoldilocksExt2]| {
            let half = eq.len() / 2;
            let mut ne = Vec::with_capacity(half);
            let mut nf = Vec::with_capacity(half);
            for j in 0..half {
                ne.push(aext2_add_host(eq[2*j], r2 * (eq[2*j+1] - eq[2*j])));
                nf.push(aext2_add_host(f[2*j],  r2 * (f[2*j+1]  - f[2*j])));
            }
            (ne, nf)
        })(&neq1, &nf1);
        let exp0_r2 = cpu_msg(&neq0_2, &nf0_2);
        let exp1_r2 = cpu_msg(&neq1_2, &nf1_2);
        let gpu_msg2 = st.compute_round_messages().expect("gpu r2");
        for c in 0..3 {
            assert_eq!(gpu_msg2[0 * 3 + c].c0.0, exp0_r2[c].c0.0, "r2 leaf0 c={}", c);
            assert_eq!(gpu_msg2[1 * 3 + c].c0.0, exp1_r2[c].c0.0, "r2 leaf1 c={}", c);
        }

        // Final: fold round 2 and read f_final per leaf.
        let r3 = lift(13);
        st.fold(r3).expect("fold3");
        let final_f = st.final_f_evals().expect("final");
        // CPU expected: fold (neq0_2, nf0_2) by r3 → final (1 entry each).
        let cpu_final = |f: &[AlmostGoldilocksExt2]| -> AlmostGoldilocksExt2 {
            aext2_add_host(f[0], r3 * (f[1] - f[0]))
        };
        assert_eq!(final_f[0].c0.0, cpu_final(&nf0_2).c0.0, "leaf0 final f");
        assert_eq!(final_f[1].c0.0, cpu_final(&nf1_2).c0.0, "leaf1 final f");
    }

    /// `from_device_buffers` must produce bit-identical state to `new` when
    /// given the same logical input. Validate by running one round of sumcheck
    /// on each and checking the round message matches.
    #[test]
    fn ext2_from_device_buffers_matches_new() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let n_var = 3;
        let size = 1usize << n_var;
        // Two polys of size 8.
        let p1: Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i * 3 + 1)).collect();
        let p2: Vec<AlmostGoldilocksExt2> = (0..size as u64).map(|i| lift(i * 5 + 7)).collect();

        // Build via new().
        let refs: Vec<&[AlmostGoldilocksExt2]> = vec![&p1, &p2];
        let mut state_new = GpuSumcheckStateExt2::new(&refs).expect("new");
        let msg_new = state_new.compute_round_message().expect("rm_new");

        // Build via from_device_buffers: pack each poly into a u64 buffer
        // [c0, c1, c0, c1, ...].
        let pack = |p: &[AlmostGoldilocksExt2]| -> Vec<u64> {
            let mut v = Vec::with_capacity(p.len() * 2);
            for e in p { v.push(e.c0.0); v.push(e.c1.0); }
            v
        };
        let p1_u64 = pack(&p1);
        let p2_u64 = pack(&p2);
        let d1 = DeviceBuffer::<u64>::from_slice(&p1_u64).expect("d1");
        let d2 = DeviceBuffer::<u64>::from_slice(&p2_u64).expect("d2");
        let buffers: Vec<&DeviceBuffer<u64>> = vec![&d1, &d2];
        let mut state_dev = GpuSumcheckStateExt2::from_device_buffers(&buffers, size)
            .expect("from_device_buffers");
        let msg_dev = state_dev.compute_round_message().expect("rm_dev");

        assert_eq!(msg_new.len(), msg_dev.len());
        for i in 0..msg_new.len() {
            assert_eq!(
                msg_new[i].c0.reduce(),
                msg_dev[i].c0.reduce(),
                "c0 mismatch at slot {}", i,
            );
            assert_eq!(
                msg_new[i].c1.reduce(),
                msg_dev[i].c1.reduce(),
                "c1 mismatch at slot {}", i,
            );
        }
    }
}

// ===========================================================================
// Sparse boolean-check sumcheck, device resident.
// ===========================================================================

/// One device-resident boolean-check sumcheck over sparse selection terms
/// sharing a dense eq table.  Mirrors zk-torch-4's
/// `SparseBoolSumcheckProverExt2::prove` dense branch exactly: same pair walk,
/// same degree-3 round message, same fold.  Only the round messages cross the
/// PCIe bus (8 u64 per round), so the 7 GB of support data stays on device.
pub struct BoolSumcheckGpu {
    d_idx: crate::memory::DeviceBuffer<u32>,
    d_val: crate::memory::DeviceBuffer<u64>,
    d_off: crate::memory::DeviceBuffer<u32>,
    d_w: crate::memory::DeviceBuffer<u64>,
    d_eq: crate::memory::DeviceBuffer<u64>,
    // scratch
    d_flags: crate::memory::DeviceBuffer<u32>,
    d_gid: crate::memory::DeviceBuffer<u32>,
    d_partial: crate::memory::DeviceBuffer<u64>,
    d_scan: crate::memory::DeviceBuffer<u32>,
    d_oidx: crate::memory::DeviceBuffer<u32>,
    d_oval: crate::memory::DeviceBuffer<u64>,
    d_ooff: crate::memory::DeviceBuffer<u32>,
    n: usize,
    n_terms: usize,
    grid_x: i32,
    scan_len: usize,
}

const BOOL_BLOCK: usize = 256;

impl BoolSumcheckGpu {
    /// `positions[t]` must be sorted ascending, as the CPU prover sorts them.
    /// `eq` is the dense `2^arity` table in the same order the CPU builds it.
    pub fn new(
        weights: &[crate::extension::AlmostGoldilocksExt2],
        positions: &[Vec<usize>],
        eq: &[crate::extension::AlmostGoldilocksExt2],
    ) -> crate::error::Result<Self> {
        let n_terms = weights.len();
        // Build the u32 index array DIRECTLY and sort each term's slice in
        // place. The caller's positions come from a HashMap so they are
        // unordered and must be sorted, but doing that by deep-cloning
        // Vec<Vec<usize>> first cost 363 MB per sub-group at arity 22 before a
        // single byte reached the device.
        let mut idx: Vec<u32> = Vec::with_capacity(positions.iter().map(|p| p.len()).sum());
        let mut off: Vec<u32> = Vec::with_capacity(n_terms + 1);
        // COUNTING sort, not a comparison sort. A selection polynomial's cube
        // index is `input_index + table_index << input_n` (poly/sparse.rs), and
        // table_index < 2^table_commit_log, which is 64 in the shipped configs.
        // So sorted order is "bucket by the high bits, stable within a bucket",
        // and the number of buckets is tiny. sort_unstable on 45M u32 per
        // sub-group was the single largest item in setup.
        //
        // The bucket count is derived from the data (max index >> input_n),
        // not assumed: if a caller ever passes something whose high part is
        // wide, this falls back to sorting rather than allocating a huge
        // histogram.
        let mut scratch: Vec<u32> = Vec::new();
        let mut hist: Vec<u32> = Vec::new();
        for p in positions {
            let start = idx.len();
            off.push(start as u32);
            idx.extend(p.iter().map(|&x| x as u32));
            let seg = &mut idx[start..];
            if seg.len() < 2 { continue; }
            let maxv = seg.iter().copied().max().unwrap_or(0);
            // low bits = input index; its width is ceil(log2(len)) for a
            // selection poly (exactly one nonzero per input row).
            let input_bits = usize::BITS as usize - (seg.len() as u32 - 1).leading_zeros() as usize;
            let buckets = (maxv as usize >> input_bits) + 1;
            if buckets > 4096 || input_bits >= 32 { seg.sort_unstable(); continue; }
            hist.clear(); hist.resize(buckets + 1, 0);
            for &v in seg.iter() { hist[(v as usize >> input_bits) + 1] += 1; }
            for b in 1..=buckets { hist[b] += hist[b - 1]; }
            scratch.clear(); scratch.resize(seg.len(), 0);
            for &v in seg.iter() {
                let b = v as usize >> input_bits;
                scratch[hist[b] as usize] = v;
                hist[b] += 1;
            }
            // Within a bucket the low bits are the input index, which the
            // caller supplies in arbitrary (HashMap) order, so each bucket
            // still needs ordering -- but over 2^input_bits/buckets items.
            let mut lo = 0usize;
            for b in 0..buckets {
                let hi = hist[b] as usize;
                if hi > lo + 1 { scratch[lo..hi].sort_unstable(); }
                lo = hi;
            }
            seg.copy_from_slice(&scratch);
        }
        off.push(idx.len() as u32);
        let n = idx.len();
        let mut w: Vec<u64> = Vec::with_capacity(2 * n_terms);
        for x in weights { w.push(x.c0.0); w.push(x.c1.0); }
        let mut eqf: Vec<u64> = Vec::with_capacity(2 * eq.len());
        for x in eq { eqf.push(x.c0.0); eqf.push(x.c1.0); }

        // One partial per (block, term) rather than one per 256 entries, so the
        // host-side fold is over a few thousand values instead of 65535.
        let per_term = if n_terms > 0 { (n + n_terms - 1) / n_terms } else { n };
        let grid_x = ((per_term + BOOL_BLOCK - 1) / BOOL_BLOCK).clamp(1, 32);
        let partial_blocks = grid_x * n_terms.max(1);
        // Scan scratch: block sums plus the recursion's own block sums.
        // Scan scratch: each level needs nb for its block sums plus nb for the
        // level below's output, and the levels shrink by 256x. 3*nb + slack
        // covers the whole recursion with room to spare.
        let nb = (n + BOOL_BLOCK - 1) / BOOL_BLOCK;
        let scan_len = 3 * nb + 4 * BOOL_BLOCK;
        Ok(Self {
            d_idx: crate::memory::DeviceBuffer::from_slice(&idx)?,
            d_val: {
                // Filled on device: every entry is the constant Ext2(1, 0), so
                // uploading it was 726 MB of PCIe per sub-group to transfer 1.
                let mut b: crate::memory::DeviceBuffer<u64> =
                    crate::memory::DeviceBuffer::new(2 * n.max(1))?;
                let rc = unsafe { crate::ffi::agl_bool_init_val_ffi(b.as_mut_ptr(), n) };
                if rc != 0 {
                    return Err(crate::error::CudaError::InvalidArgument(
                        format!("bool init val: {}", rc)));
                }
                b
            },
            d_off: crate::memory::DeviceBuffer::from_slice(&off)?,
            d_w: crate::memory::DeviceBuffer::from_slice(&w)?,
            d_eq: crate::memory::DeviceBuffer::from_slice(&eqf)?,
            d_flags: crate::memory::DeviceBuffer::new(n.max(1))?,
            d_gid: crate::memory::DeviceBuffer::new(n.max(1))?,
            d_partial: crate::memory::DeviceBuffer::new(partial_blocks * 8)?,
            d_scan: crate::memory::DeviceBuffer::new(scan_len)?,
            d_oidx: crate::memory::DeviceBuffer::new(n.max(1))?,
            d_oval: crate::memory::DeviceBuffer::new(2 * n.max(1))?,
            d_ooff: crate::memory::DeviceBuffer::new(n_terms + 1)?,
            n, n_terms, grid_x: grid_x as i32, scan_len,
        })
    }

    /// Degree-3 round message as 4 Ext2 values.
    pub fn round_message(&mut self) -> crate::error::Result<([crate::extension::AlmostGoldilocksExt2; 4], u32)> {
        let mut msg = [0u64; 8];
        let mut total: u32 = 0;
        let rc = unsafe {
            crate::ffi::agl_bool_round_msg_ffi(
                self.d_idx.as_ptr(), self.d_val.as_ptr(), self.d_off.as_ptr(),
                self.d_w.as_ptr(), self.d_eq.as_ptr(),
                self.n_terms as std::os::raw::c_int, self.n,
                self.d_flags.as_mut_ptr(), self.d_gid.as_mut_ptr(),
                self.d_partial.as_mut_ptr(), self.grid_x,
                self.d_scan.as_mut_ptr(), self.scan_len,
                msg.as_mut_ptr(), &mut total as *mut u32,
            )
        };
        if rc != 0 { return Err(crate::error::CudaError::InvalidArgument(format!("bool round msg: {}", rc))); }
        let e = |k: usize| crate::extension::AlmostGoldilocksExt2::new(
            crate::field::AlmostGoldilocksField(msg[2 * k]),
            crate::field::AlmostGoldilocksField(msg[2 * k + 1]));
        Ok(([e(0), e(1), e(2), e(3)], total))
    }

    /// Fold by the challenge; `total` comes from the preceding round_message.
    pub fn fold(&mut self, r: crate::extension::AlmostGoldilocksExt2, total: u32)
        -> crate::error::Result<()>
    {
        let rc = unsafe {
            crate::ffi::agl_bool_fold_ffi(
                self.d_idx.as_ptr(), self.d_val.as_ptr(), self.d_flags.as_ptr(),
                self.d_gid.as_ptr(), self.d_off.as_ptr(),
                self.n_terms as std::os::raw::c_int, self.n, r.c0.0, r.c1.0, total, self.grid_x,
                self.d_oidx.as_mut_ptr(), self.d_oval.as_mut_ptr(), self.d_ooff.as_mut_ptr(),
            )
        };
        if rc != 0 { return Err(crate::error::CudaError::InvalidArgument(format!("bool fold: {}", rc))); }
        if std::env::var("ZK4_GPU_BOOL_DBG").is_ok() {
            let off = self.d_ooff.to_vec().unwrap_or_default();
            eprintln!("[bool] after fold: total={} n_terms={} new_off={:?} (buf lens idx={} val={} off={})",
                total, self.n_terms, off, self.d_oidx.len(), self.d_oval.len(), self.d_ooff.len());
        }
        std::mem::swap(&mut self.d_idx, &mut self.d_oidx);
        std::mem::swap(&mut self.d_val, &mut self.d_oval);
        std::mem::swap(&mut self.d_off, &mut self.d_ooff);
        self.n = total as usize;
        // eq halves with adjacent pairing, which is exactly aext2_fold_single.
        let half = self.d_eq.len() / 4;
        if half > 0 {
            let mut out: crate::memory::DeviceBuffer<u64> = crate::memory::DeviceBuffer::new(half * 2)?;
            let rc = unsafe {
                crate::ffi::aext2_fold_single_ffi(
                    self.d_eq.as_ptr(), out.as_mut_ptr(), r.c0.0, r.c1.0,
                    self.d_eq.len() / 2, half, 1)
            };
            if rc != 0 { return Err(crate::error::CudaError::InvalidArgument(format!("eq fold: {}", rc))); }
            // aext2_fold_single_ffi does not synchronize, so an illegal access
            // here would only surface at the next sync, in an unrelated kernel.
            if std::env::var("ZK4_GPU_BOOL_DBG").is_ok() {
                crate::memory::synchronize().map_err(|e| crate::error::CudaError::InvalidArgument(
                    format!("eq fold sync (len={} half={}): {:?}", self.d_eq.len(), half, e)))?;
            }
            self.d_eq = out;
        }
        Ok(())
    }

    /// `(term values at position 0, eq[0])` for the final evaluation.
    pub fn finish(&self) -> crate::error::Result<(Vec<crate::extension::AlmostGoldilocksExt2>, crate::extension::AlmostGoldilocksExt2)> {
        // Gather on device and download 2 u64 per term. to_vec() here pulled
        // the FULL original idx and val buffers -- ~900 MB per sub-group at
        // arity 22 -- to read one value per term.
        let mut d_out: crate::memory::DeviceBuffer<u64> =
            crate::memory::DeviceBuffer::new(2 * self.n_terms.max(1))?;
        let rc = unsafe {
            crate::ffi::agl_bool_finish_ffi(
                self.d_idx.as_ptr(), self.d_val.as_ptr(), self.d_off.as_ptr(),
                self.n_terms as std::os::raw::c_int, self.n, d_out.as_mut_ptr())
        };
        if rc != 0 {
            return Err(crate::error::CudaError::InvalidArgument(format!("bool finish: {}", rc)));
        }
        let flat = d_out.to_vec()?;
        let mk = |c0: u64, c1: u64| crate::extension::AlmostGoldilocksExt2::new(
            crate::field::AlmostGoldilocksField(c0), crate::field::AlmostGoldilocksField(c1));
        let zero = mk(0, 0);
        let out: Vec<_> = (0..self.n_terms).map(|t| mk(flat[2 * t], flat[2 * t + 1])).collect();
        // eq has been folded down to a single element by the last round.
        let eq = self.d_eq.to_vec()?;
        Ok((out, if eq.len() >= 2 { mk(eq[0], eq[1]) } else { zero }))
    }
}
