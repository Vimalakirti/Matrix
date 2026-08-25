//! Bit-decomposition + scale kernels for ScaleDown / ScaleUp / NonNegative.
//!
//! `scale_down`: per-element integer division (signed semantics matching
//! `f_to_int`/`int_to_f`) plus a 32-bit decomposition of the remainder.
//! `scale_up`: modular multiply by a small scalar.
//! `decompose_bits32`: split each input's low 32 bits into a `[n × 32]` bit
//! polynomial, all-zero for values >= 2^32 (mirrors `NonNegative::run`).
//!
//! Bit layout in all cases is little-endian: `bits[i + bit * n]`.

use std::os::raw::c_int;

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;

pub fn scale_down(
    input: &DeviceBuffer<u64>,
    quotients: &mut DeviceBuffer<u64>,
    bits: &mut DeviceBuffer<u64>,
    n: usize,
    sf: u64,
) -> Result<()> {
    if quotients.len() < n || bits.len() < n * 32 {
        return Err(CudaError::InvalidArgument("scale_down: output buffer too small".into()));
    }
    let ret = unsafe {
        ffi::gl_scale_down(input.as_ptr(), quotients.as_mut_ptr(), bits.as_mut_ptr(), n as c_int, sf)
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

pub fn scale_up(
    input: &DeviceBuffer<u64>,
    output: &mut DeviceBuffer<u64>,
    n: usize,
    sf: u64,
) -> Result<()> {
    if output.len() < n {
        return Err(CudaError::InvalidArgument("scale_up: output buffer too small".into()));
    }
    let ret = unsafe { ffi::gl_scale_up(input.as_ptr(), output.as_mut_ptr(), n as c_int, sf) };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

pub fn decompose_bits32(
    input: &DeviceBuffer<u64>,
    bits: &mut DeviceBuffer<u64>,
    n: usize,
) -> Result<()> {
    if bits.len() < n * 32 {
        return Err(CudaError::InvalidArgument("decompose_bits32: output buffer too small".into()));
    }
    let ret = unsafe { ffi::gl_decompose_bits32(input.as_ptr(), bits.as_mut_ptr(), n as c_int) };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

pub fn memset_zero(buf: &mut DeviceBuffer<u64>, n_u64: usize) -> Result<()> {
    let ret = unsafe { ffi::gl_memset_zero(buf.as_mut_ptr(), n_u64) };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}
