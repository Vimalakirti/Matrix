//! GPU einsum kernels (`einsum1` unary, `einsum2` binary).
//!
//! Direct field-swap port of `goldilocks-cuda::einsum`. The contract is
//! identical: caller hands in `EinsumDimSpec`s for the output and sum
//! dimensions with per-input strides, plus the actual device buffers.
//!
//! All arithmetic uses the on-device `agl_add` / `agl_mul`. Output is a flat
//! `DeviceBuffer<u64>` of length `out_size`.

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;

/// Maximum number of dimensions accepted by the GPU kernel (matches
/// `AGL_EINSUM_MAX_NDIM` in the CUDA wrapper).
pub const EINSUM_MAX_NDIM: usize = 8;

/// Two-input einsum: `C[out_idx] = Σ_{sum_idx} A[base_a(...)] · B[base_b(...)]`.
///
/// `out_dims` lists the output-axis sizes; `out_strides_a` / `out_strides_b`
/// are how a unit step along each output axis advances the input pointers.
/// `sum_dims` / `sum_strides_*` describe the summation axes the same way.
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
            "einsum: ndim exceeds {} (out: {}, sum: {})",
            EINSUM_MAX_NDIM,
            out_dims.len(),
            sum_dims.len()
        )));
    }
    if out_strides_a.len() != out_dims.len() || out_strides_b.len() != out_dims.len() {
        return Err(CudaError::InvalidArgument(
            "einsum: out_strides_a/b length must equal out_dims".to_string(),
        ));
    }
    if sum_strides_a.len() != sum_dims.len() || sum_strides_b.len() != sum_dims.len() {
        return Err(CudaError::InvalidArgument(
            "einsum: sum_strides_a/b length must equal sum_dims".to_string(),
        ));
    }
    if c.len() < out_size {
        return Err(CudaError::InvalidArgument(format!(
            "einsum2: output buffer {} < out_size {}",
            c.len(),
            out_size
        )));
    }
    let ret = unsafe {
        ffi::agl_einsum2_ffi(
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

/// One-input einsum: `C[out_idx] = Σ_{sum_idx} A[base_a(...)]`.
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
            "einsum: ndim exceeds {} (out: {}, sum: {})",
            EINSUM_MAX_NDIM,
            out_dims.len(),
            sum_dims.len()
        )));
    }
    if out_strides_a.len() != out_dims.len() {
        return Err(CudaError::InvalidArgument(
            "einsum: out_strides_a length must equal out_dims".to_string(),
        ));
    }
    if sum_strides_a.len() != sum_dims.len() {
        return Err(CudaError::InvalidArgument(
            "einsum: sum_strides_a length must equal sum_dims".to_string(),
        ));
    }
    if c.len() < out_size {
        return Err(CudaError::InvalidArgument(format!(
            "einsum1: output buffer {} < out_size {}",
            c.len(),
            out_size
        )));
    }
    let ret = unsafe {
        ffi::agl_einsum1_ffi(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::AlmostGoldilocksField;

    fn cuda_ready() -> bool {
        crate::init().is_ok()
    }

    /// `C[m, n] = Σ_k A[m, k] · B[k, n]`: classic matmul. Tensors use the
    /// little-endian (first-dim-varies-fastest) flat layout the kernel
    /// expects — so `A[m, k] = a_buf[m + k·M]`, `B[k, n] = b_buf[k + n·K]`,
    /// `C[m, n] = c_buf[m + n·M]`.
    #[test]
    fn einsum2_matmul_matches_cpu() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        let m = 4usize;
        let k_ = 3usize;
        let n = 5usize;

        // Build A in little-endian (M, K) layout.
        let mut a = vec![0u64; m * k_];
        for mm in 0..m {
            for kk in 0..k_ {
                a[mm + kk * m] = (mm + kk * m) as u64 + 1;
            }
        }
        // Build B in little-endian (K, N) layout.
        let mut b = vec![0u64; k_ * n];
        for kk in 0..k_ {
            for nn in 0..n {
                b[kk + nn * k_] = (kk + nn * k_) as u64 + 1;
            }
        }

        let d_a = DeviceBuffer::<u64>::from_slice(&a).expect("a");
        let d_b = DeviceBuffer::<u64>::from_slice(&b).expect("b");
        let mut d_c = DeviceBuffer::<u64>::new(m * n).expect("c");

        // Output (M, N) — m has stride 1 in A and 0 in B; n has stride 0 in A
        // and K in B. Sum dim k has stride M in A and 1 in B.
        let out_dims = [m as i32, n as i32];
        let out_strides_a = [1, 0];
        let out_strides_b = [0, k_ as i32];
        let sum_dims = [k_ as i32];
        let sum_strides_a = [m as i32];
        let sum_strides_b = [1];
        einsum2(
            &d_a, &d_b, &mut d_c, m * n, k_,
            &out_dims, &out_strides_a, &out_strides_b,
            &sum_dims, &sum_strides_a, &sum_strides_b,
        )
        .expect("einsum2");
        let gpu = d_c.to_vec().expect("download");

        // CPU reference using the same little-endian layout.
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = AlmostGoldilocksField(0);
                for kk in 0..k_ {
                    acc = acc
                        + AlmostGoldilocksField(a[mm + kk * m])
                            * AlmostGoldilocksField(b[kk + nn * k_]);
                }
                let got = AlmostGoldilocksField(gpu[mm + nn * m]).reduce();
                assert_eq!(got, acc.reduce(), "(m, n) = ({}, {})", mm, nn);
            }
        }
    }

    /// `C[n] = Σ_m A[m, n]`: column sum. Little-endian layout — A[m, n] at
    /// `a[m + n·M]`.
    #[test]
    fn einsum1_column_sum_matches_cpu() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        let m = 4usize;
        let n = 5usize;
        let mut a = vec![0u64; m * n];
        for mm in 0..m {
            for nn in 0..n {
                a[mm + nn * m] = (mm + nn * m) as u64 + 1;
            }
        }
        let d_a = DeviceBuffer::<u64>::from_slice(&a).expect("a");
        let mut d_c = DeviceBuffer::<u64>::new(n).expect("c");

        // Output dim n with stride M (each step in n advances M elements in A);
        // sum dim m with stride 1.
        let out_dims = [n as i32];
        let out_strides_a = [m as i32];
        let sum_dims = [m as i32];
        let sum_strides_a = [1];
        einsum1(&d_a, &mut d_c, n, m, &out_dims, &out_strides_a, &sum_dims, &sum_strides_a)
            .expect("einsum1");
        let gpu = d_c.to_vec().expect("download");

        for nn in 0..n {
            let mut acc = AlmostGoldilocksField(0);
            for mm in 0..m {
                acc = acc + AlmostGoldilocksField(a[mm + nn * m]);
            }
            assert_eq!(
                AlmostGoldilocksField(gpu[nn]).reduce(),
                acc.reduce(),
                "n = {}",
                nn
            );
        }
    }
}
