//! GPU partial evaluation of multilinear polynomials over the almost-Goldilocks field.

use crate::error::{CudaError, Result};
use crate::extension::AlmostGoldilocksExt2;
use crate::ffi;
use crate::field::AlmostGoldilocksField;
use crate::memory::DeviceBuffer;

/// Partial evaluate at base-field `r`.
pub fn partial_eval(
    poly_evals: &[AlmostGoldilocksField],
    r: &[AlmostGoldilocksField],
) -> Result<Vec<AlmostGoldilocksField>> {
    let n = poly_evals.len();
    let m = r.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(CudaError::InvalidArgument(
            "poly_evals length must be a positive power of two".to_string(),
        ));
    }
    let log_n = n.trailing_zeros() as usize;
    if m > log_n {
        return Err(CudaError::InvalidArgument(format!(
            "r length {} exceeds log_n {}",
            m, log_n
        )));
    }
    if m == 0 { return Ok(poly_evals.to_vec()); }

    let mut d_data = DeviceBuffer::from_slice(poly_evals)?;
    let d_r = DeviceBuffer::from_slice(r)?;

    let ret = unsafe {
        ffi::agl_partial_eval_ffi(
            d_data.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    crate::memory::synchronize()?;

    let result_len = 1usize << (log_n - m);
    d_data.read_slice(0, result_len)
}

/// Partial evaluate a base-field polynomial at Ext2 `r`.
pub fn partial_eval_ext2(
    poly_evals: &[AlmostGoldilocksField],
    r: &[AlmostGoldilocksExt2],
) -> Result<Vec<AlmostGoldilocksExt2>> {
    let n = poly_evals.len();
    let m = r.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(CudaError::InvalidArgument(
            "poly_evals length must be a positive power of two".to_string(),
        ));
    }
    let log_n = n.trailing_zeros() as usize;
    if m > log_n {
        return Err(CudaError::InvalidArgument(format!(
            "r length {} exceeds log_n {}",
            m, log_n
        )));
    }
    if m == 0 {
        return Ok(poly_evals
            .iter()
            .map(|&v| AlmostGoldilocksExt2::from_base(v))
            .collect());
    }

    let d_input = DeviceBuffer::from_slice(poly_evals)?;
    let d_r = DeviceBuffer::from_slice(r)?;
    let output_len = 1usize << (log_n - 1);
    let mut d_output: DeviceBuffer<AlmostGoldilocksExt2> = DeviceBuffer::new(output_len)?;

    let ret = unsafe {
        ffi::agl_partial_eval_ext2_from_base_ffi(
            d_input.as_ptr() as *const u64,
            d_output.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    crate::memory::synchronize()?;

    let result_len = 1usize << (log_n - m);
    d_output.read_slice(0, result_len)
}

/// In-place device-side base-field partial eval.
pub fn partial_eval_device(
    d_data: &mut DeviceBuffer<AlmostGoldilocksField>,
    d_r: &DeviceBuffer<AlmostGoldilocksField>,
    log_n: usize,
    m: usize,
) -> Result<()> {
    if m == 0 { return Ok(()); }
    let ret = unsafe {
        ffi::agl_partial_eval_ffi(
            d_data.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// `u64`-typed shim around [`partial_eval_ext2_device`]. Mirrors the
/// `goldilocks-cuda::partial_eval::partial_eval_ext2_device_u64` API that
/// downstream crates already depend on — the input buffer is treated as
/// `AlmostGoldilocksField` (transparent over `u64`), the output as
/// interleaved `[c0, c1, c0, c1, ...]` `u64` pairs.
pub fn partial_eval_ext2_device_u64(
    d_input: &DeviceBuffer<u64>,
    d_output: &mut DeviceBuffer<AlmostGoldilocksExt2>,
    d_r: &DeviceBuffer<AlmostGoldilocksExt2>,
    log_n: usize,
    m: usize,
) -> Result<()> {
    if m == 0 {
        return Err(CudaError::InvalidArgument("m must be >= 1 for ext2 partial eval".to_string()));
    }
    let ret = unsafe {
        ffi::agl_partial_eval_ext2_from_base_ffi(
            d_input.as_ptr(),
            d_output.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// Device-side base→Ext2 partial eval. Output buffer needs `2^{log_n - 1}` Ext2 elements.
pub fn partial_eval_ext2_device(
    d_input: &DeviceBuffer<AlmostGoldilocksField>,
    d_output: &mut DeviceBuffer<AlmostGoldilocksExt2>,
    d_r: &DeviceBuffer<AlmostGoldilocksExt2>,
    log_n: usize,
    m: usize,
) -> Result<()> {
    if m == 0 {
        return Err(CudaError::InvalidArgument("m must be >= 1 for ext2 partial eval".to_string()));
    }
    let ret = unsafe {
        ffi::agl_partial_eval_ext2_from_base_ffi(
            d_input.as_ptr() as *const u64,
            d_output.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// Fused permute + partial-evaluation kernel.
///
///   `output[j] = Σ_{b=0..2^m-1} evals[perm(b + j*2^m)] * eq(r, b)`
///
/// where `perm()` is built from split LUTs derived from `permute_ranges`.
pub fn fused_permute_partial_eval(
    evals: &[AlmostGoldilocksField],
    challenges: &[AlmostGoldilocksExt2],
    permute_ranges: &[(usize, usize)],
    n: usize,
) -> Result<Vec<AlmostGoldilocksExt2>> {
    let m = challenges.len();
    assert_eq!(evals.len(), 1 << n);
    assert!(m <= n);

    if m == 0 {
        return Ok(evals.iter().map(|&v| AlmostGoldilocksExt2::from_base(v)).collect());
    }

    let half = n / 2;
    let (lo_lut, hi_lut) = build_split_luts(n, permute_ranges, half);

    let d_evals = DeviceBuffer::from_slice(evals)?;
    let d_lo_lut = DeviceBuffer::<u32>::from_slice(&lo_lut)?;
    let d_hi_lut = DeviceBuffer::<u32>::from_slice(&hi_lut)?;

    let d_challenges = DeviceBuffer::from_slice(challenges)?;
    let (d_buf_a, d_buf_b, result_in_a) =
        crate::eq_lagrange::ext2_eq_dp_all_device(&d_challenges, m)?;
    let d_eq = if result_in_a { &d_buf_a } else { &d_buf_b };

    let output_size = 1usize << (n - m);
    let mut d_output = DeviceBuffer::<u64>::new(output_size * 2)?;

    let lo_size = 1usize << half;
    let hi_size = 1usize << (n - half);
    let lut_bytes = (lo_size + hi_size) * std::mem::size_of::<u32>();
    let aligned_lut = (lut_bytes + 7) & !7;
    let num_warps = 256 / 32;
    let warp_bytes = num_warps * 2 * std::mem::size_of::<u64>();
    let smem_bytes = aligned_lut + warp_bytes;

    let ret = unsafe {
        ffi::agl_fused_permute_partial_eval_ffi(
            d_evals.as_ptr() as *const u64,
            d_output.as_mut_ptr(),
            d_eq.as_ptr() as *const u64,
            d_lo_lut.as_ptr(),
            d_hi_lut.as_ptr(),
            n as i32,
            m as i32,
            half as i32,
            output_size as i32,
            smem_bytes as i32,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    crate::memory::synchronize()?;

    let raw: Vec<u64> = d_output.read_slice(0, output_size * 2)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| AlmostGoldilocksExt2::new(AlmostGoldilocksField(c[0]), AlmostGoldilocksField(c[1])))
        .collect())
}

fn build_split_luts(
    n: usize,
    permute_ranges: &[(usize, usize)],
    half: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut new_var_order = Vec::with_capacity(n);
    for &(start, end) in permute_ranges {
        for v in start..end { new_var_order.push(v); }
    }
    assert_eq!(new_var_order.len(), n);

    let mut perm_map = vec![0usize; n];
    for (new_pos, &old_var) in new_var_order.iter().enumerate() {
        perm_map[old_var] = new_pos;
    }
    let mut inv_perm = vec![0usize; n];
    for old_var in 0..n { inv_perm[perm_map[old_var]] = old_var; }

    let lo_size = 1usize << half;
    let mut lo_lut = vec![0u32; lo_size];
    for lo_bits in 0..lo_size {
        let mut old_idx = 0u32;
        for bit in 0..half {
            if lo_bits & (1 << bit) != 0 { old_idx |= 1 << inv_perm[bit]; }
        }
        lo_lut[lo_bits] = old_idx;
    }

    let hi_size = 1usize << (n - half);
    let mut hi_lut = vec![0u32; hi_size];
    for hi_bits in 0..hi_size {
        let mut old_idx = 0u32;
        for bit in 0..(n - half) {
            if hi_bits & (1 << bit) != 0 { old_idx |= 1 << inv_perm[half + bit]; }
        }
        hi_lut[hi_bits] = old_idx;
    }
    (lo_lut, hi_lut)
}
