//! GPU-accelerated partial evaluation of multilinear polynomials.
//!
//! Given a multilinear polynomial f(x_1,...,x_N) represented by its 2^N
//! evaluations over the Boolean hypercube, and a partial assignment
//! r = (r_1,...,r_m), compute g(x_{m+1},...,x_N) = f(r, x_{m+1},...,x_N)
//! yielding 2^{N-m} evaluations.
//!
//! Two variants:
//! - **GL r**: r is base field, f is base field -> result is base field
//! - **Ext2 r**: r is ext2, f is base field -> result is ext2

use crate::error::{CudaError, Result};
use crate::extension::GoldilocksExt2;
use crate::field::GoldilocksField;
use crate::ffi;
use crate::memory::DeviceBuffer;

/// Partial evaluate a multilinear polynomial at r (base field).
///
/// Given f with 2^N evaluations and r of length m, returns g with
/// 2^{N-m} base field evaluations.
///
/// # Arguments
/// * `poly_evals` - The 2^N evaluations of f. Length must be a power of two.
/// * `r` - The partial assignment point of length m (m <= N).
///
/// # Returns
/// A vector of 2^{N-m} base field elements.
pub fn partial_eval_gl(
    poly_evals: &[GoldilocksField],
    r: &[GoldilocksField],
) -> Result<Vec<GoldilocksField>> {
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
        return Ok(poly_evals.to_vec());
    }

    let mut d_data = DeviceBuffer::from_slice(poly_evals)?;
    let d_r = DeviceBuffer::from_slice(r)?;

    let ret = unsafe {
        ffi::partial_eval_gl_ffi(
            d_data.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    crate::memory::synchronize()?;

    let result_len = 1usize << (log_n - m);
    d_data.read_slice(0, result_len)
}

/// Partial evaluate a GL multilinear polynomial at ext2 r.
///
/// Given f (base field) with 2^N evaluations and r (ext2) of length m,
/// returns g with 2^{N-m} ext2 evaluations.
///
/// # Arguments
/// * `poly_evals` - The 2^N base field evaluations of f. Length must be a power of two.
/// * `r` - The ext2 partial assignment point of length m (1 <= m <= N).
///
/// # Returns
/// A vector of 2^{N-m} ext2 elements.
pub fn partial_eval_ext2(
    poly_evals: &[GoldilocksField],
    r: &[GoldilocksExt2],
) -> Result<Vec<GoldilocksExt2>> {
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
            .map(|&v| GoldilocksExt2::from_base(v))
            .collect());
    }

    let d_input = DeviceBuffer::from_slice(poly_evals)?;
    let d_r = DeviceBuffer::from_slice(r)?;
    let output_len = 1usize << (log_n - 1); // first round halves
    let mut d_output: DeviceBuffer<GoldilocksExt2> = DeviceBuffer::new(output_len)?;

    let ret = unsafe {
        ffi::partial_eval_ext2_from_gl_ffi(
            d_input.as_ptr() as *const u64,
            d_output.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    crate::memory::synchronize()?;

    let result_len = 1usize << (log_n - m);
    d_output.read_slice(0, result_len)
}

/// In-place partial eval on a device buffer (base field).
///
/// After this call, the first 2^{log_n - m} positions of `d_data` hold the result.
///
/// # Arguments
/// * `d_data` - Device buffer with at least 2^log_n GL elements.
/// * `d_r` - Device buffer with m GL elements.
/// * `log_n` - Log of the input polynomial size.
/// * `m` - Number of variables to evaluate.
pub fn partial_eval_gl_device(
    d_data: &mut DeviceBuffer<GoldilocksField>,
    d_r: &DeviceBuffer<GoldilocksField>,
    log_n: usize,
    m: usize,
) -> Result<()> {
    if m == 0 {
        return Ok(());
    }

    let ret = unsafe {
        ffi::partial_eval_gl_ffi(
            d_data.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    Ok(())
}

/// Partial eval GL -> ext2 on device buffers.
///
/// Caller provides the output buffer (needs at least 2^{log_n - 1} ext2 elements).
/// After this call, the first 2^{log_n - m} positions of `d_output` hold the result.
///
/// # Arguments
/// * `d_input` - Device buffer with 2^log_n GL elements (read-only).
/// * `d_output` - Device buffer for ext2 output (at least 2^{log_n - 1} elements).
/// * `d_r` - Device buffer with m ext2 elements.
/// * `log_n` - Log of the input polynomial size.
/// * `m` - Number of variables to evaluate (m >= 1).
pub fn partial_eval_ext2_device(
    d_input: &DeviceBuffer<GoldilocksField>,
    d_output: &mut DeviceBuffer<GoldilocksExt2>,
    d_r: &DeviceBuffer<GoldilocksExt2>,
    log_n: usize,
    m: usize,
) -> Result<()> {
    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "m must be >= 1 for ext2 partial eval".to_string(),
        ));
    }

    let ret = unsafe {
        ffi::partial_eval_ext2_from_gl_ffi(
            d_input.as_ptr() as *const u64,
            d_output.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    Ok(())
}

/// Same as `partial_eval_ext2_device` but accepts `DeviceBuffer<u64>` input
/// (for when data is already on GPU as raw u64s, e.g., after GPU permutation).
/// GoldilocksField is repr(transparent) over u64, so the memory layout is identical.
pub fn partial_eval_ext2_device_u64(
    d_input: &DeviceBuffer<u64>,
    d_output: &mut DeviceBuffer<GoldilocksExt2>,
    d_r: &DeviceBuffer<GoldilocksExt2>,
    log_n: usize,
    m: usize,
) -> Result<()> {
    if m == 0 {
        return Err(CudaError::InvalidArgument(
            "m must be >= 1 for ext2 partial eval".to_string(),
        ));
    }

    let ret = unsafe {
        ffi::partial_eval_ext2_from_gl_ffi(
            d_input.as_ptr(),
            d_output.as_mut_ptr() as *mut u64,
            d_r.as_ptr() as *const u64,
            log_n as i32,
            m as i32,
        )
    };

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    Ok(())
}

/// Fused permute + partial evaluation on GPU.
///
/// Combines bit-permutation and partial evaluation into a single GPU kernel,
/// eliminating the intermediate permuted array. Computes:
///
///   output[j] = sum_{b=0}^{2^m-1} evals[perm(b + j*2^m)] * eq(r, b)
///
/// where perm() uses split-LUTs for O(1) permutation lookup.
///
/// # Arguments
/// * `evals` - Base-field polynomial evaluations, size 2^n.
/// * `challenges` - m Ext2 challenge points to fold (partial eval).
/// * `permute_ranges` - Bit-range permutation spec: list of (start, end) ranges.
/// * `n` - Log of the polynomial size.
///
/// # Returns
/// Ext2 polynomial of size 2^{n-m}.
pub fn fused_permute_partial_eval(
    evals: &[GoldilocksField],
    challenges: &[GoldilocksExt2],
    permute_ranges: &[(usize, usize)],
    n: usize,
) -> Result<Vec<GoldilocksExt2>> {
    let m = challenges.len();
    assert_eq!(evals.len(), 1 << n);
    assert!(m <= n);

    if m == 0 {
        // No partial eval needed; just permute (but this shouldn't happen in practice)
        return Ok(evals.iter().map(|&v| GoldilocksExt2::from_base(v)).collect());
    }

    // Compute inverse permutation and split LUTs
    let half = n / 2;
    let (lo_lut, hi_lut) = build_split_luts(n, permute_ranges, half);

    // Upload data to GPU
    let d_evals = DeviceBuffer::from_slice(evals)?;
    let d_lo_lut = DeviceBuffer::<u32>::from_slice(&lo_lut)?;
    let d_hi_lut = DeviceBuffer::<u32>::from_slice(&hi_lut)?;

    // Compute eq table on GPU
    let d_challenges = DeviceBuffer::from_slice(challenges)?;
    let (d_buf_a, d_buf_b, result_in_a) =
        crate::eq_lagrange::ext2_eq_dp_all_device(&d_challenges, m)?;
    let d_eq = if result_in_a { &d_buf_a } else { &d_buf_b };

    // Allocate output
    let output_size = 1usize << (n - m);
    let mut d_output = DeviceBuffer::<u64>::new(output_size * 2)?;

    // Compute shared memory size
    let lo_size = 1usize << half;
    let hi_size = 1usize << (n - half);
    let lut_bytes = (lo_size + hi_size) * std::mem::size_of::<u32>();
    let aligned_lut = (lut_bytes + 7) & !7;
    let num_warps = 256 / 32; // FUSED_BLOCK_SIZE / warp_size
    let warp_bytes = num_warps * 2 * std::mem::size_of::<u64>();
    let smem_bytes = aligned_lut + warp_bytes;

    // Launch fused kernel
    let ret = unsafe {
        ffi::fused_permute_partial_eval_ffi(
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

    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }

    crate::memory::synchronize()?;

    // Read output as Ext2 elements
    // d_output is [c0, c1, c0, c1, ...] layout, same as GoldilocksExt2
    let raw: Vec<u64> = d_output.read_slice(0, output_size * 2)?;
    let result: Vec<GoldilocksExt2> = raw
        .chunks_exact(2)
        .map(|c| GoldilocksExt2::new(GoldilocksField(c[0]), GoldilocksField(c[1])))
        .collect();
    Ok(result)
}

/// Build split LUTs for the fused permute kernel.
///
/// Returns (lo_lut, hi_lut) where:
/// - lo_lut[new_lo_bits] = old_index contribution from the low `half` new bits
/// - hi_lut[new_hi_bits] = old_index contribution from the high `n-half` new bits
/// - Full old_index = lo_lut[idx & lo_mask] | hi_lut[idx >> half]
fn build_split_luts(
    n: usize,
    permute_ranges: &[(usize, usize)],
    half: usize,
) -> (Vec<u32>, Vec<u32>) {
    // Compute perm_map: perm_map[old_var] = new_var_position
    let mut new_var_order = Vec::with_capacity(n);
    for &(start, end) in permute_ranges {
        for v in start..end {
            new_var_order.push(v);
        }
    }
    assert_eq!(new_var_order.len(), n);

    let mut perm_map = vec![0usize; n];
    for (new_pos, &old_var) in new_var_order.iter().enumerate() {
        perm_map[old_var] = new_pos;
    }

    // Compute inverse: inv_perm[new_pos] = old_var
    let mut inv_perm = vec![0usize; n];
    for old_var in 0..n {
        inv_perm[perm_map[old_var]] = old_var;
    }

    // Build lo_lut: for each combination of the low `half` new-index bits,
    // compute the contribution to the old index
    let lo_size = 1usize << half;
    let mut lo_lut = vec![0u32; lo_size];
    for lo_bits in 0..lo_size {
        let mut old_idx = 0u32;
        for bit in 0..half {
            if lo_bits & (1 << bit) != 0 {
                old_idx |= 1 << inv_perm[bit];
            }
        }
        lo_lut[lo_bits] = old_idx;
    }

    // Build hi_lut: for each combination of the high `n-half` new-index bits
    let hi_size = 1usize << (n - half);
    let mut hi_lut = vec![0u32; hi_size];
    for hi_bits in 0..hi_size {
        let mut old_idx = 0u32;
        for bit in 0..(n - half) {
            if hi_bits & (1 << bit) != 0 {
                old_idx |= 1 << inv_perm[half + bit];
            }
        }
        hi_lut[hi_bits] = old_idx;
    }

    (lo_lut, hi_lut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::EXT2_W;
    use crate::field::GOLDILOCKS_PRIME;

    // CPU reference helpers

    fn cpu_gl_mul(a: u64, b: u64) -> u64 {
        let prod = (a as u128) * (b as u128);
        let x_lo = prod as u64;
        let x_hi = (prod >> 64) as u64;
        let result = (x_lo as u128)
            .wrapping_add((x_hi as u128) << 32)
            .wrapping_sub(x_hi as u128);
        let mut r = (result % (GOLDILOCKS_PRIME as u128)) as u64;
        if r >= GOLDILOCKS_PRIME {
            r -= GOLDILOCKS_PRIME;
        }
        r
    }

    fn cpu_gl_add(a: u64, b: u64) -> u64 {
        let sum = a.wrapping_add(b);
        if sum >= GOLDILOCKS_PRIME || sum < a {
            sum.wrapping_sub(GOLDILOCKS_PRIME)
        } else {
            sum
        }
    }

    fn cpu_gl_sub(a: u64, b: u64) -> u64 {
        if a >= b {
            a - b
        } else {
            a.wrapping_sub(b).wrapping_add(GOLDILOCKS_PRIME)
        }
    }

    fn canonicalize(v: u64) -> u64 {
        if v >= GOLDILOCKS_PRIME {
            v - GOLDILOCKS_PRIME
        } else {
            v
        }
    }

    fn cpu_partial_eval_gl(evals: &[u64], r: &[u64]) -> Vec<u64> {
        let mut data = evals.to_vec();
        let mut size = data.len();
        for &ri in r {
            let half = size / 2;
            for j in 0..half {
                let a = data[2 * j];
                let b = data[2 * j + 1];
                let diff = cpu_gl_sub(b, a);
                data[j] = cpu_gl_add(a, cpu_gl_mul(ri, diff));
            }
            size = half;
        }
        data[..size].to_vec()
    }

    // ext2 CPU reference

    fn cpu_ext2_add(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
        (cpu_gl_add(a.0, b.0), cpu_gl_add(a.1, b.1))
    }

    fn cpu_ext2_sub(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
        (cpu_gl_sub(a.0, b.0), cpu_gl_sub(a.1, b.1))
    }

    fn cpu_ext2_mul(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
        let b1_w = cpu_gl_mul(b.1, EXT2_W);
        let c0 = cpu_gl_add(cpu_gl_mul(a.0, b.0), cpu_gl_mul(a.1, b1_w));
        let c1 = cpu_gl_add(cpu_gl_mul(a.0, b.1), cpu_gl_mul(a.1, b.0));
        (c0, c1)
    }

    fn cpu_partial_eval_ext2(evals: &[u64], r: &[(u64, u64)]) -> Vec<(u64, u64)> {
        if r.is_empty() {
            return evals.iter().map(|&v| (v, 0)).collect();
        }

        // Round 0: mixed
        let half = evals.len() / 2;
        let r0 = r[0];
        let mut data: Vec<(u64, u64)> = Vec::with_capacity(half);
        for j in 0..half {
            let a = evals[2 * j];
            let b = evals[2 * j + 1];
            let diff = cpu_gl_sub(b, a);
            let c0 = cpu_gl_add(a, cpu_gl_mul(r0.0, diff));
            let c1 = cpu_gl_mul(r0.1, diff);
            data.push((c0, c1));
        }
        let mut size = half;

        // Rounds 1+: ext2
        for &ri in &r[1..] {
            let half = size / 2;
            for j in 0..half {
                let a = data[2 * j];
                let b = data[2 * j + 1];
                let diff = cpu_ext2_sub(b, a);
                data[j] = cpu_ext2_add(a, cpu_ext2_mul(ri, diff));
            }
            size = half;
        }

        data[..size].to_vec()
    }

    // ========================================================================
    // Tests
    // ========================================================================

    #[test]
    fn test_partial_eval_gl_correctness() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        for log_n in [4, 8, 12, 16] {
            let n = 1usize << log_n;
            let evals: Vec<GoldilocksField> = (0..n)
                .map(|i| {
                    GoldilocksField::new(
                        ((i as u64 + 1).wrapping_mul(0xDEADBEEFCAFEBABE)) % GOLDILOCKS_PRIME,
                    )
                })
                .collect();
            let evals_raw: Vec<u64> = evals.iter().map(|f| f.0).collect();

            for m in [1, log_n / 2, log_n] {
                let r: Vec<GoldilocksField> = (0..m)
                    .map(|i| {
                        GoldilocksField::new(
                            ((i as u64 + 1).wrapping_mul(12345678901234567)) % GOLDILOCKS_PRIME,
                        )
                    })
                    .collect();
                let r_raw: Vec<u64> = r.iter().map(|f| f.0).collect();

                let gpu_result = partial_eval_gl(&evals, &r)
                    .unwrap_or_else(|e| panic!("GPU failed for log_n={}, m={}: {:?}", log_n, m, e));
                let cpu_result = cpu_partial_eval_gl(&evals_raw, &r_raw);

                assert_eq!(
                    gpu_result.len(),
                    cpu_result.len(),
                    "Length mismatch for log_n={}, m={}",
                    log_n,
                    m
                );

                for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
                    assert_eq!(
                        canonicalize(gpu.0),
                        canonicalize(*cpu),
                        "Mismatch at index {} for log_n={}, m={}: GPU={}, CPU={}",
                        i,
                        log_n,
                        m,
                        gpu.0,
                        *cpu
                    );
                }
            }
        }
    }

    #[test]
    fn test_partial_eval_ext2_correctness() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        for log_n in [4, 8, 12, 16] {
            let n = 1usize << log_n;
            let evals: Vec<GoldilocksField> = (0..n)
                .map(|i| {
                    GoldilocksField::new(
                        ((i as u64 + 1).wrapping_mul(0xDEADBEEFCAFEBABE)) % GOLDILOCKS_PRIME,
                    )
                })
                .collect();
            let evals_raw: Vec<u64> = evals.iter().map(|f| f.0).collect();

            for m in [1, log_n / 2, log_n] {
                if m == 0 {
                    continue;
                }
                let r: Vec<GoldilocksExt2> = (0..m)
                    .map(|i| {
                        GoldilocksExt2::new(
                            GoldilocksField::new(
                                ((i as u64 + 1).wrapping_mul(12345678901234567)) % GOLDILOCKS_PRIME,
                            ),
                            GoldilocksField::new(
                                ((i as u64 + 1).wrapping_mul(98765432109876543)) % GOLDILOCKS_PRIME,
                            ),
                        )
                    })
                    .collect();
                let r_raw: Vec<(u64, u64)> = r.iter().map(|e| (e.c0.0, e.c1.0)).collect();

                let gpu_result = partial_eval_ext2(&evals, &r).unwrap_or_else(|e| {
                    panic!("GPU failed for log_n={}, m={}: {:?}", log_n, m, e)
                });
                let cpu_result = cpu_partial_eval_ext2(&evals_raw, &r_raw);

                assert_eq!(
                    gpu_result.len(),
                    cpu_result.len(),
                    "Length mismatch for log_n={}, m={}",
                    log_n,
                    m
                );

                for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
                    assert_eq!(
                        canonicalize(gpu.c0.0),
                        canonicalize(cpu.0),
                        "c0 mismatch at index {} for log_n={}, m={}: GPU={}, CPU={}",
                        i,
                        log_n,
                        m,
                        gpu.c0.0,
                        cpu.0
                    );
                    assert_eq!(
                        canonicalize(gpu.c1.0),
                        canonicalize(cpu.1),
                        "c1 mismatch at index {} for log_n={}, m={}: GPU={}, CPU={}",
                        i,
                        log_n,
                        m,
                        gpu.c1.0,
                        cpu.1
                    );
                }
            }
        }
    }

    #[test]
    fn test_partial_eval_gl_edge_cases() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        // m=0: identity
        let evals = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ];
        let result = partial_eval_gl(&evals, &[]).unwrap();
        assert_eq!(result.len(), 4);
        for (g, e) in result.iter().zip(evals.iter()) {
            assert_eq!(canonicalize(g.0), canonicalize(e.0));
        }

        // m=log_n: full eval to single element
        let r = vec![GoldilocksField::new(0), GoldilocksField::new(0)];
        let result = partial_eval_gl(&evals, &r).unwrap();
        assert_eq!(result.len(), 1);
        // f(0,0) = evals[0] = 1
        assert_eq!(canonicalize(result[0].0), 1);

        let r = vec![GoldilocksField::new(1), GoldilocksField::new(1)];
        let result = partial_eval_gl(&evals, &r).unwrap();
        assert_eq!(result.len(), 1);
        // f(1,1) = evals[3] = 4
        assert_eq!(canonicalize(result[0].0), 4);
    }

    #[test]
    fn test_partial_eval_ext2_edge_cases() {
        if crate::init().is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }

        let evals = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ];

        // m=0: identity (returns GL values embedded as ext2)
        let result = partial_eval_ext2(&evals, &[]).unwrap();
        assert_eq!(result.len(), 4);
        for (g, e) in result.iter().zip(evals.iter()) {
            assert_eq!(canonicalize(g.c0.0), canonicalize(e.0));
            assert_eq!(canonicalize(g.c1.0), 0);
        }

        // m=1 with base-field r (c1=0): should match GL result
        let r_ext2 = vec![GoldilocksExt2::new(
            GoldilocksField::new(7),
            GoldilocksField::new(0),
        )];
        let r_gl = vec![GoldilocksField::new(7)];
        let ext2_result = partial_eval_ext2(&evals, &r_ext2).unwrap();
        let gl_result = partial_eval_gl(&evals, &r_gl).unwrap();
        assert_eq!(ext2_result.len(), gl_result.len());
        for (e, g) in ext2_result.iter().zip(gl_result.iter()) {
            assert_eq!(canonicalize(e.c0.0), canonicalize(g.0));
            assert_eq!(canonicalize(e.c1.0), 0);
        }

        // m=log_n: full eval to single element
        let r = vec![
            GoldilocksExt2::new(GoldilocksField::new(0), GoldilocksField::new(0)),
            GoldilocksExt2::new(GoldilocksField::new(0), GoldilocksField::new(0)),
        ];
        let result = partial_eval_ext2(&evals, &r).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(canonicalize(result[0].c0.0), 1);
        assert_eq!(canonicalize(result[0].c1.0), 0);
    }
}
