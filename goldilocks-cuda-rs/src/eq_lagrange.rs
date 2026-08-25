//! Eq Lagrange polynomial computation on GPU.
//!
//! This module provides GPU-accelerated computation of the eq(r, x) polynomial
//! over the Boolean hypercube {0,1}^n.
//!
//! For a point r = (r_0, ..., r_{n-1}) in F^n:
//!   eq(r, x) = ∏_{i=0}^{n-1} (r_i * x_i + (1 - r_i) * (1 - x_i))
//!
//! The result is a vector of 2^n field elements, one for each x in {0,1}^n.

use crate::error::{CudaError, Result};
use crate::extension::GoldilocksExt2;
use crate::field::GoldilocksField;
use crate::ffi;
use crate::memory::DeviceBuffer;

/// Compute eq(r, x) for all x in {0,1}^log_n using the DP algorithm.
///
/// # Arguments
/// * `r` - The evaluation point r = (r_0, ..., r_{log_n-1})
///
/// # Returns
/// A vector of 2^log_n field elements where result[x] = eq(r, x)
///
/// # Example
/// ```ignore
/// use goldilocks_cuda::prelude::*;
/// use goldilocks_cuda::eq_lagrange;
///
/// goldilocks_cuda::init().unwrap();
///
/// // Compute eq(r, x) for r = (0.5, 0.25) over {0,1}^2
/// let r = vec![
///     GoldilocksField::new(1 << 63),  // ~0.5 in the field
///     GoldilocksField::new(1 << 62),  // ~0.25 in the field
/// ];
/// let result = eq_lagrange::eq_dp_all(&r).unwrap();
/// assert_eq!(result.len(), 4);  // 2^2 = 4
/// ```
pub fn eq_dp_all(r: &[GoldilocksField]) -> Result<Vec<GoldilocksField>> {
    let log_n = r.len();
    if log_n == 0 {
        return Ok(vec![GoldilocksField::new(1)]);
    }

    let n = 1usize << log_n;

    // Allocate device buffers
    let d_r = DeviceBuffer::from_slice(r)?;
    let mut d_buf_a: DeviceBuffer<GoldilocksField> = DeviceBuffer::new(n)?;
    let mut d_buf_b: DeviceBuffer<GoldilocksField> = DeviceBuffer::new(n)?;

    // Call the CUDA kernel
    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_buf_a.as_mut_ptr() as *mut u64,
            d_buf_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    // Determine which buffer has the result and copy back
    let result = if result_ptr == d_buf_a.as_ptr() as *mut u64 {
        d_buf_a.to_vec()?
    } else {
        d_buf_b.to_vec()?
    };

    Ok(result)
}

/// Compute eq(r, x) for all x in {0,1}^log_n using the DP algorithm (Ext2 version).
///
/// # Arguments
/// * `r` - The evaluation point r = (r_0, ..., r_{log_n-1}) in the quadratic extension field
///
/// # Returns
/// A vector of 2^log_n Ext2 elements where result[x] = eq(r, x)
///
/// # Example
/// ```ignore
/// use goldilocks_cuda::prelude::*;
/// use goldilocks_cuda::eq_lagrange;
///
/// goldilocks_cuda::init().unwrap();
///
/// let r = vec![
///     GoldilocksExt2::new(GoldilocksField::new(123), GoldilocksField::new(456)),
///     GoldilocksExt2::new(GoldilocksField::new(789), GoldilocksField::new(101)),
/// ];
/// let result = eq_lagrange::ext2_eq_dp_all(&r).unwrap();
/// assert_eq!(result.len(), 4);  // 2^2 = 4
/// ```
pub fn ext2_eq_dp_all(r: &[GoldilocksExt2]) -> Result<Vec<GoldilocksExt2>> {
    let log_n = r.len();
    if log_n == 0 {
        return Ok(vec![GoldilocksExt2::one()]);
    }

    let n = 1usize << log_n;

    // Allocate device buffers
    let d_r = DeviceBuffer::from_slice(r)?;
    let mut d_buf_a: DeviceBuffer<GoldilocksExt2> = DeviceBuffer::new(n)?;
    let mut d_buf_b: DeviceBuffer<GoldilocksExt2> = DeviceBuffer::new(n)?;

    // Call the CUDA kernel
    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::ext2_eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_buf_a.as_mut_ptr() as *mut u64,
            d_buf_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    // Determine which buffer has the result and copy back
    let result = if result_ptr == d_buf_a.as_ptr() as *mut u64 {
        d_buf_a.to_vec()?
    } else {
        d_buf_b.to_vec()?
    };

    Ok(result)
}

/// Batch version: Compute eq(r, x) keeping data on GPU.
///
/// Returns a DeviceBuffer containing the result, avoiding host-device copies
/// when chaining multiple GPU operations.
pub fn eq_dp_all_device(
    d_r: &DeviceBuffer<GoldilocksField>,
    log_n: usize,
) -> Result<(DeviceBuffer<GoldilocksField>, DeviceBuffer<GoldilocksField>, bool)> {
    let n = 1usize << log_n;

    // Allocate device buffers
    let mut d_buf_a: DeviceBuffer<GoldilocksField> = DeviceBuffer::new(n)?;
    let mut d_buf_b: DeviceBuffer<GoldilocksField> = DeviceBuffer::new(n)?;

    // Call the CUDA kernel
    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_buf_a.as_mut_ptr() as *mut u64,
            d_buf_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    // Return both buffers and indicate which one has the result
    let result_in_a = result_ptr == d_buf_a.as_ptr() as *mut u64;
    Ok((d_buf_a, d_buf_b, result_in_a))
}

/// Batch version: Compute eq(r, x) for Ext2 keeping data on GPU.
///
/// Returns a DeviceBuffer containing the result, avoiding host-device copies
/// when chaining multiple GPU operations.
pub fn ext2_eq_dp_all_device(
    d_r: &DeviceBuffer<GoldilocksExt2>,
    log_n: usize,
) -> Result<(DeviceBuffer<GoldilocksExt2>, DeviceBuffer<GoldilocksExt2>, bool)> {
    let n = 1usize << log_n;

    // Allocate device buffers
    let mut d_buf_a: DeviceBuffer<GoldilocksExt2> = DeviceBuffer::new(n)?;
    let mut d_buf_b: DeviceBuffer<GoldilocksExt2> = DeviceBuffer::new(n)?;

    // Call the CUDA kernel
    let mut result_ptr: *mut u64 = std::ptr::null_mut();
    let ret = unsafe {
        ffi::ext2_eq_dp_all_ffi(
            d_r.as_ptr() as *const u64,
            d_buf_a.as_mut_ptr() as *mut u64,
            d_buf_b.as_mut_ptr() as *mut u64,
            log_n as i32,
            &mut result_ptr,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    // Return both buffers and indicate which one has the result
    let result_in_a = result_ptr == d_buf_a.as_ptr() as *mut u64;
    Ok((d_buf_a, d_buf_b, result_in_a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::EXT2_W;
    use crate::field::GOLDILOCKS_PRIME;

    // ========================================================================
    // CPU Reference Implementation for Goldilocks Field
    // ========================================================================

    /// Goldilocks field multiplication using u128 arithmetic
    fn cpu_gl_mul(a: u64, b: u64) -> u64 {
        let prod = (a as u128) * (b as u128);
        // Reduce mod p = 2^64 - 2^32 + 1
        // Using: x mod p = x_lo - x_hi + x_hi * 2^32 (mod p)
        let x_lo = prod as u64;
        let x_hi = (prod >> 64) as u64;

        // Compute in u128 to avoid overflow
        let result = (x_lo as u128)
            .wrapping_add((x_hi as u128) << 32)
            .wrapping_sub(x_hi as u128);

        // Reduce to u64 range
        let mut r = (result % (GOLDILOCKS_PRIME as u128)) as u64;
        if r >= GOLDILOCKS_PRIME {
            r -= GOLDILOCKS_PRIME;
        }
        r
    }

    fn cpu_gl_sub(a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a.wrapping_sub(b).wrapping_add(GOLDILOCKS_PRIME)
        }
    }

    fn cpu_gl_add(a: u64, b: u64) -> u64 {
        let sum = a.wrapping_add(b);
        if sum >= GOLDILOCKS_PRIME || sum < a {
            sum.wrapping_sub(GOLDILOCKS_PRIME)
        } else {
            sum
        }
    }

    fn canonicalize(v: u64) -> u64 {
        if v >= GOLDILOCKS_PRIME {
            v - GOLDILOCKS_PRIME
        } else {
            v
        }
    }

    /// CPU reference: eq(r, x) = ∏_{i=0}^{n-1} (r_i * x_i + (1 - r_i) * (1 - x_i))
    fn cpu_eq_reference(r: &[GoldilocksField]) -> Vec<GoldilocksField> {
        let log_n = r.len();
        let n = 1usize << log_n;
        let mut result = Vec::with_capacity(n);

        for x in 0..n {
            let mut acc = 1u64;
            for i in 0..log_n {
                let x_i = (x >> i) & 1;
                if x_i == 1 {
                    acc = cpu_gl_mul(acc, r[i].0);
                } else {
                    acc = cpu_gl_mul(acc, cpu_gl_sub(1, r[i].0));
                }
            }
            result.push(GoldilocksField::new(acc));
        }

        result
    }

    // ========================================================================
    // CPU Reference Implementation for Ext2
    // ========================================================================

    fn cpu_ext2_mul(a: &GoldilocksExt2, b: &GoldilocksExt2) -> GoldilocksExt2 {
        // (a0 + a1*X) * (b0 + b1*X) = a0*b0 + a1*b1*W + (a0*b1 + a1*b0)*X
        let b1_w = cpu_gl_mul(b.c1.0, EXT2_W);
        let c0 = cpu_gl_add(cpu_gl_mul(a.c0.0, b.c0.0), cpu_gl_mul(a.c1.0, b1_w));
        let c1 = cpu_gl_add(cpu_gl_mul(a.c0.0, b.c1.0), cpu_gl_mul(a.c1.0, b.c0.0));
        GoldilocksExt2::new(GoldilocksField::new(c0), GoldilocksField::new(c1))
    }

    fn cpu_ext2_sub(a: &GoldilocksExt2, b: &GoldilocksExt2) -> GoldilocksExt2 {
        GoldilocksExt2::new(
            GoldilocksField::new(cpu_gl_sub(a.c0.0, b.c0.0)),
            GoldilocksField::new(cpu_gl_sub(a.c1.0, b.c1.0)),
        )
    }

    fn cpu_ext2_eq_reference(r: &[GoldilocksExt2]) -> Vec<GoldilocksExt2> {
        let log_n = r.len();
        let n = 1usize << log_n;
        let mut result = Vec::with_capacity(n);
        let one = GoldilocksExt2::one();

        for x in 0..n {
            let mut acc = one;
            for i in 0..log_n {
                let x_i = (x >> i) & 1;
                if x_i == 1 {
                    acc = cpu_ext2_mul(&acc, &r[i]);
                } else {
                    acc = cpu_ext2_mul(&acc, &cpu_ext2_sub(&one, &r[i]));
                }
            }
            result.push(acc);
        }

        result
    }

    // ========================================================================
    // Correctness Tests
    // ========================================================================

    #[test]
    fn test_eq_dp_all_correctness() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        // Test with various sizes
        for log_n in [1, 2, 4, 8, 12] {
            let r: Vec<GoldilocksField> = (0..log_n)
                .map(|i| {
                    GoldilocksField::new(
                        ((i as u64 + 1) * 12345678901234567u64) % GOLDILOCKS_PRIME,
                    )
                })
                .collect();

            let gpu_result = eq_dp_all(&r).expect("GPU computation failed");
            let cpu_result = cpu_eq_reference(&r);

            assert_eq!(gpu_result.len(), cpu_result.len());
            for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
                assert_eq!(
                    canonicalize(gpu.0),
                    canonicalize(cpu.0),
                    "Mismatch at index {} for log_n={}: GPU={}, CPU={}",
                    i,
                    log_n,
                    gpu.0,
                    cpu.0
                );
            }
        }
    }

    #[test]
    fn test_ext2_eq_dp_all_correctness() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        // Test with various sizes
        for log_n in [1, 2, 4, 8] {
            let r: Vec<GoldilocksExt2> = (0..log_n)
                .map(|i| {
                    GoldilocksExt2::new(
                        GoldilocksField::new(
                            ((i as u64 + 1) * 12345678901234567u64) % GOLDILOCKS_PRIME,
                        ),
                        GoldilocksField::new(
                            ((i as u64 + 1) * 98765432109876543u64) % GOLDILOCKS_PRIME,
                        ),
                    )
                })
                .collect();

            let gpu_result = ext2_eq_dp_all(&r).expect("GPU computation failed");
            let cpu_result = cpu_ext2_eq_reference(&r);

            assert_eq!(gpu_result.len(), cpu_result.len());
            for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
                assert_eq!(
                    canonicalize(gpu.c0.0),
                    canonicalize(cpu.c0.0),
                    "c0 mismatch at index {} for log_n={}: GPU={}, CPU={}",
                    i,
                    log_n,
                    gpu.c0.0,
                    cpu.c0.0
                );
                assert_eq!(
                    canonicalize(gpu.c1.0),
                    canonicalize(cpu.c1.0),
                    "c1 mismatch at index {} for log_n={}: GPU={}, CPU={}",
                    i,
                    log_n,
                    gpu.c1.0,
                    cpu.c1.0
                );
            }
        }
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    #[test]
    fn test_eq_dp_all_edge_cases() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        // log_n = 0: should return [1]
        let result = eq_dp_all(&[]).expect("GPU computation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(canonicalize(result[0].0), 1);

        // log_n = 1 with r = [0]: should return [1, 0]
        let r = vec![GoldilocksField::new(0)];
        let result = eq_dp_all(&r).expect("GPU computation failed");
        assert_eq!(result.len(), 2);
        assert_eq!(canonicalize(result[0].0), 1); // eq(0, 0) = 1
        assert_eq!(canonicalize(result[1].0), 0); // eq(0, 1) = 0

        // log_n = 1 with r = [1]: should return [0, 1]
        let r = vec![GoldilocksField::new(1)];
        let result = eq_dp_all(&r).expect("GPU computation failed");
        assert_eq!(result.len(), 2);
        assert_eq!(canonicalize(result[0].0), 0); // eq(1, 0) = 0
        assert_eq!(canonicalize(result[1].0), 1); // eq(1, 1) = 1
    }

    #[test]
    fn test_ext2_eq_dp_all_edge_cases() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        // log_n = 0: should return [1]
        let result = ext2_eq_dp_all(&[]).expect("GPU computation failed");
        assert_eq!(result.len(), 1);
        assert_eq!(canonicalize(result[0].c0.0), 1);
        assert_eq!(canonicalize(result[0].c1.0), 0);

        // log_n = 1 with r = [(0, 0)]: should return [(1,0), (0,0)]
        let r = vec![GoldilocksExt2::new(
            GoldilocksField::new(0),
            GoldilocksField::new(0),
        )];
        let result = ext2_eq_dp_all(&r).expect("GPU computation failed");
        assert_eq!(result.len(), 2);
        assert_eq!(canonicalize(result[0].c0.0), 1);
        assert_eq!(canonicalize(result[0].c1.0), 0);
        assert_eq!(canonicalize(result[1].c0.0), 0);
        assert_eq!(canonicalize(result[1].c1.0), 0);
    }
}
