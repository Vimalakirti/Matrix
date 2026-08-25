//! GPU eq(r, x) over the Boolean hypercube for the almost-Goldilocks field.
//!
//! `eq(r, x) = Π_i (r_i x_i + (1 - r_i)(1 - x_i))` for `x ∈ {0,1}^n`.
//! Backed by the DP kernel in `cuda_almost_goldilocks/almost_eq_lagrange.cuh`.

use crate::error::{CudaError, Result};
use crate::extension::AlmostGoldilocksExt2;
use crate::ffi;
use crate::field::AlmostGoldilocksField;
use crate::memory::DeviceBuffer;

/// Compute `eq(r, x)` for all `x ∈ {0,1}^log_n` (base field).
pub fn eq_dp_all(r: &[AlmostGoldilocksField]) -> Result<Vec<AlmostGoldilocksField>> {
    let log_n = r.len();
    if log_n == 0 {
        return Ok(vec![AlmostGoldilocksField::new(1)]);
    }
    let n = 1usize << log_n;

    let d_r = DeviceBuffer::from_slice(r)?;
    let mut d_a: DeviceBuffer<AlmostGoldilocksField> = DeviceBuffer::new(n)?;
    let mut d_b: DeviceBuffer<AlmostGoldilocksField> = DeviceBuffer::new(n)?;

    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::agl_eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_a.as_mut_ptr() as *mut u64,
            d_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }

    let result = if result_ptr == d_a.as_ptr() as *mut u64 {
        d_a.to_vec()?
    } else {
        d_b.to_vec()?
    };
    Ok(result)
}

/// Compute `eq(r, x)` for all `x` (Ext2 challenge vector).
pub fn ext2_eq_dp_all(r: &[AlmostGoldilocksExt2]) -> Result<Vec<AlmostGoldilocksExt2>> {
    let log_n = r.len();
    if log_n == 0 {
        return Ok(vec![AlmostGoldilocksExt2::one()]);
    }
    let n = 1usize << log_n;

    let d_r = DeviceBuffer::from_slice(r)?;
    let mut d_a: DeviceBuffer<AlmostGoldilocksExt2> = DeviceBuffer::new(n)?;
    let mut d_b: DeviceBuffer<AlmostGoldilocksExt2> = DeviceBuffer::new(n)?;

    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::aext2_eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_a.as_mut_ptr() as *mut u64,
            d_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }

    let result = if result_ptr == d_a.as_ptr() as *mut u64 {
        d_a.to_vec()?
    } else {
        d_b.to_vec()?
    };
    Ok(result)
}

/// Device-resident base-field variant. Returns both buffers and a flag
/// indicating which one contains the final result.
pub fn eq_dp_all_device(
    d_r: &DeviceBuffer<AlmostGoldilocksField>,
    log_n: usize,
) -> Result<(DeviceBuffer<AlmostGoldilocksField>, DeviceBuffer<AlmostGoldilocksField>, bool)> {
    let n = 1usize << log_n;
    let mut d_a: DeviceBuffer<AlmostGoldilocksField> = DeviceBuffer::new(n)?;
    let mut d_b: DeviceBuffer<AlmostGoldilocksField> = DeviceBuffer::new(n)?;

    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::agl_eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_a.as_mut_ptr() as *mut u64,
            d_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }

    let result_in_a = result_ptr == d_a.as_ptr() as *mut u64;
    Ok((d_a, d_b, result_in_a))
}

/// Device-resident Ext2 variant.
pub fn ext2_eq_dp_all_device(
    d_r: &DeviceBuffer<AlmostGoldilocksExt2>,
    log_n: usize,
) -> Result<(DeviceBuffer<AlmostGoldilocksExt2>, DeviceBuffer<AlmostGoldilocksExt2>, bool)> {
    let n = 1usize << log_n;
    let mut d_a: DeviceBuffer<AlmostGoldilocksExt2> = DeviceBuffer::new(n)?;
    let mut d_b: DeviceBuffer<AlmostGoldilocksExt2> = DeviceBuffer::new(n)?;

    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::aext2_eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_a.as_mut_ptr() as *mut u64,
            d_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }

    let result_in_a = result_ptr == d_a.as_ptr() as *mut u64;
    Ok((d_a, d_b, result_in_a))
}

/// Build the eq table on device from `claim_pt`, then evaluate each
/// binary plane via `eval_p = Σ_{i : plane_p[i] = 1} eq[i]`. Returns
/// one Ext2 per plane. All planes share the same eq table; one kernel
/// launch handles all `n_planes` at once.
///
/// Layout:
/// - `claim_pt`: `log_n` Ext2 challenge coordinates.
/// - `packed_planes`: each entry is the binary plane packed as `u64`s
///   of length `1 << (log_n - 6)` (caller must pad sub-word arities).
///
/// Bandwidth-bound on large `log_n`; outperforms the CPU loop
/// (build eq + selective add) by ~5× at `log_n ≥ 16` once host
/// allocation and upload costs are amortized across many planes
/// per call.
/// Device-input variant of [`eval_binary_planes_device`]: the plane data is
/// ALREADY on the current device as one or more concat buffers (each with
/// `n_planes_i` consecutive planes of `1 << (log_n − 6)` u64s). Builds eq
/// once and evaluates every buffer's planes against it. Returns the evals
/// in buffer order, planes-within-buffer order.
pub fn eval_binary_planes_from_dev(
    claim_pt: &[AlmostGoldilocksExt2],
    bufs: &[(&crate::memory::DeviceBuffer<u64>, usize)],
) -> Result<Vec<AlmostGoldilocksExt2>> {
    use crate::memory::DeviceBuffer;
    let log_n = claim_pt.len();
    let d_pt = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(claim_pt)?;
    let (d_a, d_b, in_a) = ext2_eq_dp_all_device(&d_pt, log_n)?;
    let d_eq = if in_a { d_a } else { d_b };
    eval_binary_planes_with_eq_dev(&d_eq, log_n, bufs)
}

/// As [`eval_binary_planes_from_dev`], but against a PREBUILT device eq
/// table (e.g. shared between the same-point f_evals recovery and the
/// chunk evals of one fold-tree group — both evaluate at the same
/// shared challenge point, so the ~2^n-element eq dp only needs to run
/// once per group).
pub fn eval_binary_planes_with_eq_dev(
    d_eq: &crate::memory::DeviceBuffer<AlmostGoldilocksExt2>,
    log_n: usize,
    bufs: &[(&crate::memory::DeviceBuffer<u64>, usize)],
) -> Result<Vec<AlmostGoldilocksExt2>> {
    use crate::field::AlmostGoldilocksField;
    use crate::memory::DeviceBuffer;
    use crate::ffi;
    use std::os::raw::c_int;
    const BLOCK_SIZE: usize = 256;

    let total = 1usize << log_n;
    let expected_packed = if log_n >= 6 { 1usize << (log_n - 6) } else { 1 };
    let total_planes: usize = bufs.iter().map(|&(_, n)| n).sum();
    if total_planes == 0 { return Ok(Vec::new()); }
    for &(b, n) in bufs {
        assert!(b.len() >= n * expected_packed, "device plane buffer too small");
    }
    assert!(d_eq.len() >= total, "eq table too small for log_n");

    let num_blocks_x = ((total + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    let mut out = Vec::with_capacity(total_planes);
    let mut d_partial = DeviceBuffer::<u64>::new(
        num_blocks_x * bufs.iter().map(|&(_, n)| n).max().unwrap_or(1) * 2)?;
    for &(buf, n_planes) in bufs {
        if n_planes == 0 { continue; }
        let ret = unsafe {
            ffi::aext2_selective_add_batched_planes_ffi(
                d_eq.as_ptr() as *const u64,
                buf.as_ptr(),
                d_partial.as_mut_ptr(),
                total,
                n_planes as c_int,
                expected_packed,
                num_blocks_x as c_int,
            )
        };
        if ret != 0 { return Err(crate::error::CudaError::KernelFailed); }
        let partials = d_partial.read_slice(0, num_blocks_x * n_planes * 2)?;
        for plane in 0..n_planes {
            let mut acc = AlmostGoldilocksExt2::zero();
            for b in 0..num_blocks_x {
                let off = (b * n_planes + plane) * 2;
                acc = acc + AlmostGoldilocksExt2::new(
                    AlmostGoldilocksField(partials[off]),
                    AlmostGoldilocksField(partials[off + 1]),
                );
            }
            out.push(acc);
        }
    }
    Ok(out)
}

pub fn eval_binary_planes_device(
    claim_pt: &[AlmostGoldilocksExt2],
    packed_planes: &[&[u64]],
) -> Result<Vec<AlmostGoldilocksExt2>> {
    use crate::field::AlmostGoldilocksField;
    use crate::memory::DeviceBuffer;
    use crate::ffi;
    use std::os::raw::c_int;
    const BLOCK_SIZE: usize = 256;

    let log_n = claim_pt.len();
    let total = 1usize << log_n;
    let expected_packed = if log_n >= 6 { 1usize << (log_n - 6) } else { 1 };
    for (i, p) in packed_planes.iter().enumerate() {
        assert_eq!(p.len(), expected_packed, "plane {} length mismatch", i);
    }
    let n_planes = packed_planes.len();
    if n_planes == 0 { return Ok(Vec::new()); }

    // 1) eq on device.
    let d_pt = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(claim_pt)?;
    let (d_a, d_b, in_a) = ext2_eq_dp_all_device(&d_pt, log_n)?;
    let d_eq = if in_a { d_a } else { d_b };

    // 2) Concat all packed planes, upload.
    let mut packed_concat = Vec::with_capacity(n_planes * expected_packed);
    for p in packed_planes { packed_concat.extend_from_slice(p); }
    let d_packed = DeviceBuffer::<u64>::from_slice(&packed_concat)?;

    // 3) Launch the batched selective-add kernel.
    let num_blocks_x = ((total + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256);
    let mut d_partial = DeviceBuffer::<u64>::new(num_blocks_x * n_planes * 2)?;
    let ret = unsafe {
        ffi::aext2_selective_add_batched_planes_ffi(
            d_eq.as_ptr() as *const u64,
            d_packed.as_ptr(),
            d_partial.as_mut_ptr(),
            total,
            n_planes as c_int,
            expected_packed,
            num_blocks_x as c_int,
        )
    };
    if ret != 0 { return Err(crate::error::CudaError::KernelFailed); }

    // 4) Download partials, reduce on host.
    let partials = d_partial.read_slice(0, num_blocks_x * n_planes * 2)?;
    let mut out = vec![AlmostGoldilocksExt2::zero(); n_planes];
    for b in 0..num_blocks_x {
        for plane in 0..n_planes {
            let off = (b * n_planes + plane) * 2;
            let v = AlmostGoldilocksExt2::new(
                AlmostGoldilocksField(partials[off]),
                AlmostGoldilocksField(partials[off + 1]),
            );
            out[plane] = out[plane] + v;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::AlmostGoldilocksField;

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    /// Round-trip: build eq on host, do selective add on host; compare
    /// against `eval_binary_planes_device`.
    #[test]
    fn eval_binary_planes_device_matches_cpu() {
        if crate::init().is_err() { eprintln!("skipping: no CUDA"); return; }
        let log_n = 10;
        let total = 1usize << log_n;
        let pt: Vec<AlmostGoldilocksExt2> = (0..log_n as u64).map(|i| lift(i * 7 + 11)).collect();
        // 5 binary planes, each total bits = 1024 = 16 u64s.
        let planes: Vec<Vec<u64>> = (0..5u64).map(|p| {
            (0..total / 64).map(|i| {
                ((i as u64 + p).wrapping_mul(0x9E3779B97F4A7C15)) ^ (p << 16)
            }).collect()
        }).collect();
        let plane_refs: Vec<&[u64]> = planes.iter().map(|p| p.as_slice()).collect();

        // CPU reference.
        let eq_cpu = ext2_eq_dp_all(&pt).expect("eq cpu");
        let cpu_evals: Vec<AlmostGoldilocksExt2> = planes.iter().map(|plane| {
            let mut acc = AlmostGoldilocksExt2::zero();
            for (wi, &w) in plane.iter().enumerate() {
                if w == 0 { continue; }
                let base = wi * 64;
                for k in 0..64 {
                    if (w >> k) & 1 == 1 {
                        let idx = base + k;
                        if idx < total { acc = acc + eq_cpu[idx]; }
                    }
                }
            }
            acc
        }).collect();

        let gpu_evals = eval_binary_planes_device(&pt, &plane_refs).expect("gpu");
        assert_eq!(gpu_evals.len(), 5);
        for (p, (g, c)) in gpu_evals.iter().zip(cpu_evals.iter()).enumerate() {
            assert_eq!(g.c0.0, c.c0.0, "plane {} c0 mismatch", p);
            assert_eq!(g.c1.0, c.c1.0, "plane {} c1 mismatch", p);
        }
    }
}
