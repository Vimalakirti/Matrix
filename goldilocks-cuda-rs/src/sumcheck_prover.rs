//! GPU-accelerated sumcheck prover state.
//!
//! Manages d polynomials on GPU, supports round message computation and folding.
//! Uses double buffering to avoid cross-warp race conditions during fold.

use crate::basefold::gl_add_host;
use crate::error::{CudaError, Result};
use crate::extension::GoldilocksExt2;
use crate::field::GoldilocksField;
use crate::ffi;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;

const BLOCK_SIZE: usize = 256;

/// GPU state for the linear sumcheck protocol.
///
/// Holds d polynomials packed contiguously on GPU with stride = original_size.
/// Uses double buffering (d_polys / d_scratch) for race-free folding.
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
    /// Create a new GPU sumcheck state from polynomial evaluation slices.
    ///
    /// All polynomials must have the same length (= 2^num_vars).
    pub fn new(polys: &[&[GoldilocksField]]) -> Result<Self> {
        assert!(!polys.is_empty(), "Must have at least one polynomial");
        let original_size = polys[0].len();
        assert!(original_size.is_power_of_two(), "Polynomial size must be power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        let num_polys = polys.len();

        for (i, p) in polys.iter().enumerate() {
            assert_eq!(
                p.len(),
                original_size,
                "Polynomial {} has size {} but expected {}",
                i,
                p.len(),
                original_size
            );
        }

        // Pack all polynomials contiguously
        let total = num_polys * original_size;
        let mut packed = Vec::with_capacity(total);
        for p in polys {
            for &v in *p {
                packed.push(v.0);
            }
        }

        let d_polys = DeviceBuffer::from_slice(&packed)?;
        let d_scratch = DeviceBuffer::new(total)?;

        // Allocate partial sums buffer: max_blocks * (num_polys + 1)
        let max_blocks = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks = max_blocks.min(256);
        let d_partial = DeviceBuffer::new(max_blocks * (num_polys + 1))?;

        Ok(Self {
            d_polys,
            d_scratch,
            d_partial,
            num_polys,
            original_size,
            current_round: 0,
            num_vars,
        })
    }

    /// Compute the round message for the current round.
    ///
    /// Returns evaluations g(c) for c in {0, 1, ..., num_polys}.
    pub fn compute_round_message(&mut self) -> Result<Vec<GoldilocksField>> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument(
                "No more rounds available".to_string(),
            ));
        }

        let num_blocks = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256) as c_int;
        let dp1 = self.num_polys + 1;

        let ret = unsafe {
            ffi::sumcheck_round_message_ffi(
                self.d_polys.as_ptr(),
                self.d_partial.as_mut_ptr(),
                self.num_polys as c_int,
                self.original_size,
                half,
                num_blocks,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        // Read partial sums and reduce on host
        let partials = self.d_partial.read_slice(0, num_blocks as usize * dp1)?;

        let mut result = vec![GoldilocksField(0); dp1];
        for block in 0..num_blocks as usize {
            for c in 0..dp1 {
                result[c] = GoldilocksField(gl_add_host(
                    result[c].0,
                    partials[block * dp1 + c],
                ));
            }
        }

        Ok(result)
    }

    /// Fold all polynomials at the given challenge, advancing to the next round.
    /// Uses double buffering: reads from d_polys, writes to d_scratch, then swaps.
    pub fn fold(&mut self, challenge: GoldilocksField) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument(
                "No more rounds to fold".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::sumcheck_fold_ffi(
                self.d_polys.as_ptr(),
                self.d_scratch.as_mut_ptr(),
                challenge.0,
                self.num_polys as c_int,
                self.original_size,
                half,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        // Swap buffers: scratch becomes the active buffer
        std::mem::swap(&mut self.d_polys, &mut self.d_scratch);

        self.current_round += 1;
        Ok(())
    }

    /// Get the final evaluations after all rounds (one per polynomial).
    pub fn final_evaluations(&self) -> Result<Vec<GoldilocksField>> {
        assert_eq!(
            self.current_round, self.num_vars,
            "Must complete all rounds before getting final evaluations"
        );

        let mut result = Vec::with_capacity(self.num_polys);
        for i in 0..self.num_polys {
            let vals = self.d_polys.read_slice(i * self.original_size, 1)?;
            result.push(GoldilocksField(vals[0]));
        }
        Ok(result)
    }

    /// Get a single final evaluation for polynomial at index `poly_idx`.
    pub fn final_eval(&self, poly_idx: usize) -> Result<GoldilocksField> {
        assert!(poly_idx < self.num_polys);
        let vals = self.d_polys.read_slice(poly_idx * self.original_size, 1)?;
        Ok(GoldilocksField(vals[0]))
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    pub fn current_round(&self) -> usize {
        self.current_round
    }
}

// ============================================================================
// Ext2 GPU Sumcheck State
// ============================================================================

fn ext2_add_host(a: GoldilocksExt2, b: GoldilocksExt2) -> GoldilocksExt2 {
    a + b
}

/// GPU state for the linear sumcheck protocol over Ext2.
///
/// Holds d polynomials packed contiguously on GPU as interleaved Ext2 (2 u64s per element).
/// Stride = original_size * 2 u64s per polynomial.
/// Uses double buffering for race-free folding.
pub struct GpuSumcheckStateExt2 {
    d_polys: DeviceBuffer<u64>,
    d_scratch: DeviceBuffer<u64>,
    d_partial: DeviceBuffer<u64>,
    num_polys: usize,
    original_size: usize, // number of Ext2 elements per poly
    current_round: usize,
    num_vars: usize,
}

impl GpuSumcheckStateExt2 {
    /// Create from Ext2 polynomial slices.
    /// All polynomials must have the same length (= 2^num_vars).
    pub fn new(polys: &[&[GoldilocksExt2]]) -> Result<Self> {
        assert!(!polys.is_empty(), "Must have at least one polynomial");
        let original_size = polys[0].len();
        assert!(original_size.is_power_of_two(), "Polynomial size must be power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        let num_polys = polys.len();

        for (i, p) in polys.iter().enumerate() {
            assert_eq!(
                p.len(), original_size,
                "Polynomial {} has size {} but expected {}", i, p.len(), original_size
            );
        }

        // Pack all polynomials as interleaved [c0, c1, c0, c1, ...]
        let total_u64s = num_polys * original_size * 2;
        let mut packed = Vec::with_capacity(total_u64s);
        for p in polys {
            for &v in *p {
                packed.push(v.c0.0);
                packed.push(v.c1.0);
            }
        }

        let d_polys = DeviceBuffer::from_slice(&packed)?;
        let d_scratch = DeviceBuffer::new(total_u64s)?;

        // Partial sums buffer: max_blocks * (num_polys + 1) * 2 u64s (Ext2)
        let max_blocks = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks = max_blocks.min(256);
        let d_partial = DeviceBuffer::new(max_blocks * (num_polys + 1) * 2)?;

        Ok(Self {
            d_polys,
            d_scratch,
            d_partial,
            num_polys,
            original_size,
            current_round: 0,
            num_vars,
        })
    }

    /// Create from base-field polynomial slices, converting to Ext2 on upload.
    pub fn new_from_base(polys: &[&[GoldilocksField]]) -> Result<Self> {
        let ext2_polys: Vec<Vec<GoldilocksExt2>> = polys
            .iter()
            .map(|p| p.iter().map(|&v| GoldilocksExt2::from_base(v)).collect())
            .collect();
        let ext2_refs: Vec<&[GoldilocksExt2]> = ext2_polys.iter().map(|v| v.as_slice()).collect();
        Self::new(&ext2_refs)
    }

    /// Create from GPU-resident Ext2 buffers (device-to-device, no host round-trip).
    ///
    /// Each buffer contains `original_size` Ext2 elements = `original_size * 2` u64s.
    /// Buffers are packed contiguously via device-to-device copy.
    pub fn from_device_buffers(
        buffers: &[&DeviceBuffer<u64>],
        original_size: usize,
    ) -> Result<Self> {
        assert!(!buffers.is_empty(), "Must have at least one buffer");
        assert!(original_size.is_power_of_two(), "original_size must be power of 2");
        let num_vars = original_size.trailing_zeros() as usize;
        let num_polys = buffers.len();
        let stride_u64 = original_size * 2; // u64s per polynomial

        for (i, buf) in buffers.iter().enumerate() {
            assert_eq!(
                buf.len(), stride_u64,
                "Buffer {} has {} u64s but expected {} (original_size={})",
                i, buf.len(), stride_u64, original_size
            );
        }

        let total_u64s = num_polys * stride_u64;
        let mut d_polys = DeviceBuffer::<u64>::new(total_u64s)?;

        // Pack each buffer contiguously via D2D copy
        for (i, buf) in buffers.iter().enumerate() {
            let offset = i * stride_u64;
            let size_bytes = stride_u64 * std::mem::size_of::<u64>();
            let ret = unsafe {
                ffi::cuda_memcpy_dtod(
                    d_polys.as_mut_ptr().add(offset) as *mut std::os::raw::c_void,
                    buf.as_ptr() as *const std::os::raw::c_void,
                    size_bytes,
                )
            };
            if ret != 0 {
                return Err(CudaError::MemcpyFailed);
            }
        }

        let d_scratch = DeviceBuffer::new(total_u64s)?;

        let max_blocks = ((original_size / 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let max_blocks = max_blocks.min(256);
        let d_partial = DeviceBuffer::new(max_blocks * (num_polys + 1) * 2)?;

        Ok(Self {
            d_polys,
            d_scratch,
            d_partial,
            num_polys,
            original_size,
            current_round: 0,
            num_vars,
        })
    }

    /// Compute round message for current round. Returns (d+1) Ext2 values.
    pub fn compute_round_message(&mut self) -> Result<Vec<GoldilocksExt2>> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds available".to_string()));
        }

        let num_blocks = ((half + BLOCK_SIZE - 1) / BLOCK_SIZE).min(256) as c_int;
        let dp1 = self.num_polys + 1;

        let ret = unsafe {
            ffi::sumcheck_round_message_ext2_ffi(
                self.d_polys.as_ptr(),
                self.d_partial.as_mut_ptr(),
                self.num_polys as c_int,
                self.original_size,
                half,
                num_blocks,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        // Read partial sums (Ext2 interleaved) and reduce on host
        let partials = self.d_partial.read_slice(0, num_blocks as usize * dp1 * 2)?;

        let mut result = vec![GoldilocksExt2::zero(); dp1];
        for block in 0..num_blocks as usize {
            for c in 0..dp1 {
                let off = (block * dp1 + c) * 2;
                let partial = GoldilocksExt2::new(
                    GoldilocksField(partials[off]),
                    GoldilocksField(partials[off + 1]),
                );
                result[c] = ext2_add_host(result[c], partial);
            }
        }

        Ok(result)
    }

    /// Fold all polynomials at the given Ext2 challenge.
    pub fn fold(&mut self, challenge: GoldilocksExt2) -> Result<()> {
        let current_size = self.original_size >> self.current_round;
        let half = current_size / 2;
        if half == 0 {
            return Err(CudaError::InvalidArgument("No more rounds to fold".to_string()));
        }

        let ret = unsafe {
            ffi::sumcheck_fold_ext2_ffi(
                self.d_polys.as_ptr(),
                self.d_scratch.as_mut_ptr(),
                challenge.c0.0,
                challenge.c1.0,
                self.num_polys as c_int,
                self.original_size,
                half,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        std::mem::swap(&mut self.d_polys, &mut self.d_scratch);
        self.current_round += 1;
        Ok(())
    }

    /// Get the final evaluations after all rounds (one Ext2 per polynomial).
    pub fn final_evaluations(&self) -> Result<Vec<GoldilocksExt2>> {
        assert_eq!(
            self.current_round, self.num_vars,
            "Must complete all rounds before getting final evaluations"
        );

        let mut result = Vec::with_capacity(self.num_polys);
        for i in 0..self.num_polys {
            let off = i * self.original_size * 2;
            let vals = self.d_polys.read_slice(off, 2)?;
            result.push(GoldilocksExt2::new(GoldilocksField(vals[0]), GoldilocksField(vals[1])));
        }
        Ok(result)
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_polys(&self) -> usize {
        self.num_polys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gl_add(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
        GoldilocksField(gl_add_host(a.0, b.0))
    }

    fn gl_mul(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
        use crate::basefold::gl_mul_host;
        GoldilocksField(gl_mul_host(a.0, b.0))
    }

    fn gl_sub(a: GoldilocksField, b: GoldilocksField) -> GoldilocksField {
        if a.0 >= b.0 {
            GoldilocksField(a.0 - b.0)
        } else {
            GoldilocksField(a.0.wrapping_add(crate::GOLDILOCKS_PRIME).wrapping_sub(b.0))
        }
    }

    /// CPU reference: compute round message for linear sumcheck
    fn cpu_round_message(polys: &[Vec<GoldilocksField>]) -> Vec<GoldilocksField> {
        let d = polys.len();
        let half = polys[0].len() / 2;
        let dp1 = d + 1;

        let mut result = vec![GoldilocksField(0); dp1];
        for c in 0..dp1 {
            let c_val = GoldilocksField(c as u64);
            for y in 0..half {
                let mut product = GoldilocksField(1);
                for poly in polys {
                    let a = poly[2 * y];
                    let b = poly[2 * y + 1];
                    let val = gl_add(a, gl_mul(c_val, gl_sub(b, a)));
                    product = gl_mul(product, val);
                }
                result[c] = gl_add(result[c], product);
            }
        }
        result
    }

    /// CPU reference: fold polynomials
    fn cpu_fold(polys: &mut [Vec<GoldilocksField>], challenge: GoldilocksField) {
        for poly in polys.iter_mut() {
            let half = poly.len() / 2;
            let mut new = Vec::with_capacity(half);
            for j in 0..half {
                let a = poly[2 * j];
                let b = poly[2 * j + 1];
                new.push(gl_add(a, gl_mul(challenge, gl_sub(b, a))));
            }
            *poly = new;
        }
    }

    #[test]
    fn test_gpu_sumcheck_vs_cpu_small() {
        crate::init().expect("CUDA init failed");

        // 2 polynomials, 4 variables (16 elements each)
        let n = 16;
        let p1: Vec<GoldilocksField> = (1..=n).map(|i| GoldilocksField(i as u64)).collect();
        let p2: Vec<GoldilocksField> = (100..100 + n).map(|i| GoldilocksField(i as u64)).collect();

        let mut gpu_state =
            GpuSumcheckState::new(&[&p1, &p2]).expect("GPU state creation failed");

        let mut cpu_polys = vec![p1.clone(), p2.clone()];

        for round in 0..4 {
            let gpu_msg = gpu_state
                .compute_round_message()
                .expect("GPU round message failed");
            let cpu_msg = cpu_round_message(&cpu_polys);

            assert_eq!(gpu_msg.len(), cpu_msg.len(), "Round {} message length mismatch", round);
            for (i, (g, c)) in gpu_msg.iter().zip(cpu_msg.iter()).enumerate() {
                assert_eq!(g, c, "Round {} eval point {} mismatch: GPU={} CPU={}", round, i, g.0, c.0);
            }

            let challenge = GoldilocksField(round as u64 * 7 + 3);
            gpu_state.fold(challenge).expect("GPU fold failed");
            cpu_fold(&mut cpu_polys, challenge);
        }

        let gpu_finals = gpu_state.final_evaluations().expect("GPU final eval failed");
        assert_eq!(gpu_finals.len(), 2);
        assert_eq!(gpu_finals[0], cpu_polys[0][0]);
        assert_eq!(gpu_finals[1], cpu_polys[1][0]);
    }

    #[test]
    fn test_gpu_sumcheck_vs_cpu_large() {
        crate::init().expect("CUDA init failed");

        // 3 polynomials, 12 variables (4096 elements each) — exercises cross-warp fold
        let num_vars = 12;
        let n = 1usize << num_vars;
        let p1: Vec<GoldilocksField> = (0..n).map(|i| GoldilocksField((i as u64 * 37 + 1) % crate::GOLDILOCKS_PRIME)).collect();
        let p2: Vec<GoldilocksField> = (0..n).map(|i| GoldilocksField((i as u64 * 53 + 7) % crate::GOLDILOCKS_PRIME)).collect();
        let p3: Vec<GoldilocksField> = (0..n).map(|i| GoldilocksField((i as u64 * 71 + 13) % crate::GOLDILOCKS_PRIME)).collect();

        let mut gpu_state =
            GpuSumcheckState::new(&[&p1, &p2, &p3]).expect("GPU state creation failed");

        let mut cpu_polys = vec![p1, p2, p3];

        for round in 0..num_vars {
            let gpu_msg = gpu_state.compute_round_message().unwrap();
            let cpu_msg = cpu_round_message(&cpu_polys);

            for (i, (g, c)) in gpu_msg.iter().zip(cpu_msg.iter()).enumerate() {
                assert_eq!(g, c, "Round {} eval point {} mismatch: GPU={} CPU={}", round, i, g.0, c.0);
            }

            let challenge = GoldilocksField(round as u64 * 11 + 5);
            gpu_state.fold(challenge).unwrap();
            cpu_fold(&mut cpu_polys, challenge);
        }

        let gpu_finals = gpu_state.final_evaluations().unwrap();
        for (i, (g, c)) in gpu_finals.iter().zip(cpu_polys.iter()).enumerate() {
            assert_eq!(g, &c[0], "Final eval {} mismatch", i);
        }
    }

    #[test]
    fn test_gpu_sumcheck_3_polys() {
        crate::init().expect("CUDA init failed");

        // 3 polynomials (degree d=3, so d+1=4 eval points), 3 variables (8 elements)
        let n = 8;
        let p1: Vec<GoldilocksField> = (1..=n).map(|i| GoldilocksField(i as u64)).collect();
        let p2: Vec<GoldilocksField> = (10..10 + n).map(|i| GoldilocksField(i as u64)).collect();
        let p3: Vec<GoldilocksField> = (20..20 + n).map(|i| GoldilocksField(i as u64)).collect();

        let mut gpu_state =
            GpuSumcheckState::new(&[&p1, &p2, &p3]).expect("GPU state creation failed");

        let mut cpu_polys = vec![p1, p2, p3];

        for round in 0..3 {
            let gpu_msg = gpu_state.compute_round_message().unwrap();
            let cpu_msg = cpu_round_message(&cpu_polys);

            for (i, (g, c)) in gpu_msg.iter().zip(cpu_msg.iter()).enumerate() {
                assert_eq!(g, c, "Round {} eval point {} mismatch: GPU={} CPU={}", round, i, g.0, c.0);
            }

            let challenge = GoldilocksField(round as u64 * 11 + 5);
            gpu_state.fold(challenge).unwrap();
            cpu_fold(&mut cpu_polys, challenge);
        }

        let gpu_finals = gpu_state.final_evaluations().unwrap();
        for (i, (g, c)) in gpu_finals.iter().zip(cpu_polys.iter()).enumerate() {
            assert_eq!(g, &c[0], "Final eval {} mismatch", i);
        }
    }
}
