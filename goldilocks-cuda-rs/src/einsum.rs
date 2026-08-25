//! GPU einsum (base-field) — generic 2-input or 1-input contraction over the
//! Goldilocks base field. One thread per output element loops over the
//! summation indices and accumulates A * B (or just A for 1-input).
//!
//! Indexing follows the little-endian convention used in zk-torch-3 (first
//! shape dim has stride 1).

use std::os::raw::c_int;

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;

/// Maximum number of output / summation dimensions per call. Mirrors the
/// `EINSUM_MAX_NDIM` constant on the CUDA side.
pub const EINSUM_MAX_NDIM: usize = 8;

/// 2-input GPU einsum: C[out_idx] = Σ_s A[..]*B[..]
///
/// `out_dims[d]` is the (padded) extent of output dim `d`, in iteration order
/// (d=0 has stride 1 in the output's flat layout). `out_strides_a[d]` is the
/// stride within `A`'s flat buffer when output dim `d` advances; 0 means dim
/// `d` is not in A. Similarly for B and for sum dims.
pub fn einsum2(
    a: &DeviceBuffer<u64>,
    b: &DeviceBuffer<u64>,
    c: &mut DeviceBuffer<u64>,
    out_size: usize,
    sum_size: usize,
    out_dims: &[i32],
    out_strides_a: &[i32],
    out_strides_b: &[i32],
    sum_dims: &[i32],
    sum_strides_a: &[i32],
    sum_strides_b: &[i32],
) -> Result<()> {
    if out_dims.len() > EINSUM_MAX_NDIM || sum_dims.len() > EINSUM_MAX_NDIM {
        return Err(CudaError::InvalidArgument(format!(
            "einsum2: ndim must be <= {} (got out={}, sum={})",
            EINSUM_MAX_NDIM,
            out_dims.len(),
            sum_dims.len()
        )));
    }
    if out_strides_a.len() != out_dims.len()
        || out_strides_b.len() != out_dims.len()
        || sum_strides_a.len() != sum_dims.len()
        || sum_strides_b.len() != sum_dims.len()
    {
        return Err(CudaError::InvalidArgument(
            "einsum2: stride/dim length mismatch".to_string(),
        ));
    }
    if c.len() < out_size {
        return Err(CudaError::InvalidArgument(format!(
            "einsum2: output buffer too small ({} < {})",
            c.len(),
            out_size
        )));
    }

    let ret = unsafe {
        ffi::gl_einsum2(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            out_size as c_int,
            sum_size as c_int,
            out_dims.len() as c_int,
            out_dims.as_ptr(),
            out_strides_a.as_ptr(),
            out_strides_b.as_ptr(),
            sum_dims.len() as c_int,
            sum_dims.as_ptr(),
            sum_strides_a.as_ptr(),
            sum_strides_b.as_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

/// 1-input GPU einsum: C[out_idx] = Σ_s A[..]
pub fn einsum1(
    a: &DeviceBuffer<u64>,
    c: &mut DeviceBuffer<u64>,
    out_size: usize,
    sum_size: usize,
    out_dims: &[i32],
    out_strides_a: &[i32],
    sum_dims: &[i32],
    sum_strides_a: &[i32],
) -> Result<()> {
    if out_dims.len() > EINSUM_MAX_NDIM || sum_dims.len() > EINSUM_MAX_NDIM {
        return Err(CudaError::InvalidArgument(format!(
            "einsum1: ndim must be <= {}",
            EINSUM_MAX_NDIM
        )));
    }
    if out_strides_a.len() != out_dims.len() || sum_strides_a.len() != sum_dims.len() {
        return Err(CudaError::InvalidArgument(
            "einsum1: stride/dim length mismatch".to_string(),
        ));
    }
    if c.len() < out_size {
        return Err(CudaError::InvalidArgument(
            "einsum1: output buffer too small".to_string(),
        ));
    }

    let ret = unsafe {
        ffi::gl_einsum1(
            a.as_ptr(),
            c.as_mut_ptr(),
            out_size as c_int,
            sum_size as c_int,
            out_dims.len() as c_int,
            out_dims.as_ptr(),
            out_strides_a.as_ptr(),
            sum_dims.len() as c_int,
            sum_dims.as_ptr(),
            sum_strides_a.as_ptr(),
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}
