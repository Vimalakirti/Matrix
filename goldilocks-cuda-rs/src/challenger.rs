//! Fiat-Shamir DuplexChallenger on GPU.
//!
//! This module provides a GPU-accelerated implementation of the Fiat-Shamir
//! transform using Poseidon2 as the underlying permutation.
//!
//! # Example
//!
//! ```ignore
//! use goldilocks_cuda::prelude::*;
//! use goldilocks_cuda::challenger::DuplexChallenger;
//!
//! // Initialize CUDA
//! goldilocks_cuda::init().expect("Failed to initialize CUDA");
//!
//! // Create a challenger
//! let mut challenger = DuplexChallenger::new().unwrap();
//!
//! // Observe values (like commitments)
//! challenger.observe(GoldilocksField(123)).unwrap();
//! challenger.observe(GoldilocksField(456)).unwrap();
//!
//! // Sample challenges
//! let challenge = challenger.sample().unwrap();
//! ```

use crate::error::{CudaError, Result};
use crate::extension::GoldilocksExt2;
use crate::ffi;
use crate::field::GoldilocksField;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;
use std::ptr;

/// Challenger configuration constants.
pub const CHALLENGER_WIDTH: usize = 8;
pub const CHALLENGER_RATE: usize = 4;
pub const CHALLENGER_CAPACITY: usize = 4;

/// Size of the challenger state in bytes.
pub fn challenger_state_size() -> usize {
    unsafe { ffi::challenger_state_size() as usize }
}

/// A batch of challenger states on the GPU.
///
/// This is useful for parallel proving where multiple independent
/// Fiat-Shamir transcripts need to be maintained.
pub struct ChallengerBatch {
    ptr: *mut std::ffi::c_void,
    count: usize,
}

impl ChallengerBatch {
    /// Create a new batch of challengers on the GPU.
    pub fn new(count: usize) -> Result<Self> {
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        let ret = unsafe { ffi::challenger_alloc_states(&mut ptr, count as c_int) };
        if ret != 0 {
            return Err(CudaError::AllocationFailed);
        }

        let ret = unsafe { ffi::challenger_init_states(ptr, count as c_int) };
        if ret != 0 {
            unsafe { ffi::cuda_free(ptr) };
            return Err(CudaError::KernelFailed);
        }

        Ok(Self { ptr, count })
    }

    /// Get the number of challengers in this batch.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Observe one value per challenger.
    ///
    /// `values` must have exactly `count` elements.
    pub fn observe(&mut self, values: &[GoldilocksField]) -> Result<()> {
        if values.len() != self.count {
            return Err(CudaError::InvalidArgument(
                "Values length must match challenger count".to_string(),
            ));
        }

        let flat: Vec<u64> = values.iter().map(|f| f.0).collect();
        let d_values = DeviceBuffer::from_slice(&flat)?;

        let ret = unsafe {
            ffi::challenger_observe_ffi(self.ptr, d_values.as_ptr(), self.count as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Observe a slice of values per challenger.
    ///
    /// `values` must have exactly `count * slice_len` elements,
    /// laid out as `[challenger_0_val_0, ..., challenger_0_val_n, challenger_1_val_0, ...]`.
    pub fn observe_slice(&mut self, values: &[GoldilocksField], slice_len: usize) -> Result<()> {
        if values.len() != self.count * slice_len {
            return Err(CudaError::InvalidArgument(
                "Values length must be count * slice_len".to_string(),
            ));
        }

        let flat: Vec<u64> = values.iter().map(|f| f.0).collect();
        let d_values = DeviceBuffer::from_slice(&flat)?;

        let ret = unsafe {
            ffi::challenger_observe_slice_ffi(
                self.ptr,
                d_values.as_ptr(),
                slice_len as c_int,
                self.count as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Sample one value per challenger.
    pub fn sample(&mut self) -> Result<Vec<GoldilocksField>> {
        let mut d_outputs = DeviceBuffer::<u64>::new(self.count)?;

        let ret = unsafe {
            ffi::challenger_sample_ffi(self.ptr, d_outputs.as_mut_ptr(), self.count as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        let flat = d_outputs.to_vec()?;
        Ok(flat.into_iter().map(GoldilocksField).collect())
    }

    /// Sample multiple values per challenger.
    pub fn sample_array(&mut self, count_per_challenger: usize) -> Result<Vec<GoldilocksField>> {
        let total = self.count * count_per_challenger;
        let mut d_outputs = DeviceBuffer::<u64>::new(total)?;

        let ret = unsafe {
            ffi::challenger_sample_array_ffi(
                self.ptr,
                d_outputs.as_mut_ptr(),
                count_per_challenger as c_int,
                self.count as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        let flat = d_outputs.to_vec()?;
        Ok(flat.into_iter().map(GoldilocksField).collect())
    }

    /// Sample one GF(p^2) element per challenger.
    pub fn sample_ext2(&mut self) -> Result<Vec<GoldilocksExt2>> {
        let mut d_outputs = DeviceBuffer::<u64>::new(self.count * 2)?;

        let ret = unsafe {
            ffi::challenger_sample_ext2_ffi(self.ptr, d_outputs.as_mut_ptr(), self.count as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        let flat = d_outputs.to_vec()?;
        Ok(flat
            .chunks_exact(2)
            .map(|c| GoldilocksExt2::new(GoldilocksField(c[0]), GoldilocksField(c[1])))
            .collect())
    }

    /// Observe one GF(p^2) element per challenger.
    pub fn observe_ext2(&mut self, values: &[GoldilocksExt2]) -> Result<()> {
        if values.len() != self.count {
            return Err(CudaError::InvalidArgument(
                "Values length must match challenger count".to_string(),
            ));
        }

        let flat: Vec<u64> = values.iter().flat_map(|e| [e.c0.0, e.c1.0]).collect();
        let d_values = DeviceBuffer::from_slice(&flat)?;

        let ret = unsafe {
            ffi::challenger_observe_ext2_ffi(self.ptr, d_values.as_ptr(), self.count as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Get raw device pointer (for advanced use).
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

impl Drop for ChallengerBatch {
    fn drop(&mut self) {
        unsafe {
            ffi::cuda_free(self.ptr);
        }
    }
}

// ChallengerBatch is not thread-safe due to mutable GPU state
// but can be sent between threads
unsafe impl Send for ChallengerBatch {}

/// A single Fiat-Shamir challenger with GPU acceleration.
///
/// This is a convenience wrapper around `ChallengerBatch` with `count = 1`.
/// It provides a simpler API for the common case of a single transcript.
pub struct DuplexChallenger {
    batch: ChallengerBatch,
}

impl DuplexChallenger {
    /// Create a new challenger.
    pub fn new() -> Result<Self> {
        Ok(Self {
            batch: ChallengerBatch::new(1)?,
        })
    }

    /// Observe a field element.
    pub fn observe(&mut self, value: GoldilocksField) -> Result<()> {
        self.batch.observe(&[value])
    }

    /// Observe multiple field elements.
    pub fn observe_slice(&mut self, values: &[GoldilocksField]) -> Result<()> {
        self.batch.observe_slice(values, values.len())
    }

    /// Sample a random field element.
    pub fn sample(&mut self) -> Result<GoldilocksField> {
        let samples = self.batch.sample()?;
        Ok(samples[0])
    }

    /// Sample multiple random field elements.
    pub fn sample_array(&mut self, count: usize) -> Result<Vec<GoldilocksField>> {
        self.batch.sample_array(count)
    }

    /// Sample a random GF(p^2) element.
    pub fn sample_ext2(&mut self) -> Result<GoldilocksExt2> {
        let samples = self.batch.sample_ext2()?;
        Ok(samples[0])
    }

    /// Observe a GF(p^2) element.
    pub fn observe_ext2(&mut self, value: GoldilocksExt2) -> Result<()> {
        self.batch.observe_ext2(&[value])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init;

    #[test]
    fn test_challenger_deterministic() {
        init().unwrap();

        // Create two challengers
        let mut c1 = DuplexChallenger::new().unwrap();
        let mut c2 = DuplexChallenger::new().unwrap();

        // Observe same values
        c1.observe(GoldilocksField(123)).unwrap();
        c1.observe(GoldilocksField(456)).unwrap();

        c2.observe(GoldilocksField(123)).unwrap();
        c2.observe(GoldilocksField(456)).unwrap();

        // Sample should produce same results
        let s1 = c1.sample().unwrap();
        let s2 = c2.sample().unwrap();

        assert_eq!(s1.0, s2.0, "Challengers should be deterministic");
    }

    #[test]
    fn test_challenger_different_inputs() {
        init().unwrap();

        let mut c1 = DuplexChallenger::new().unwrap();
        let mut c2 = DuplexChallenger::new().unwrap();

        c1.observe(GoldilocksField(100)).unwrap();
        c2.observe(GoldilocksField(200)).unwrap();

        let s1 = c1.sample().unwrap();
        let s2 = c2.sample().unwrap();

        assert_ne!(s1.0, s2.0, "Different inputs should produce different outputs");
    }

    #[test]
    fn test_challenger_batch() {
        init().unwrap();

        let mut batch = ChallengerBatch::new(4).unwrap();

        // Each challenger observes a different value
        let values: Vec<GoldilocksField> = (0..4).map(|i| GoldilocksField(i as u64)).collect();
        batch.observe(&values).unwrap();

        // Sample from all
        let samples = batch.sample().unwrap();

        assert_eq!(samples.len(), 4);
        // All samples should be different (with very high probability)
        let unique: std::collections::HashSet<u64> = samples.iter().map(|s| s.0).collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_challenger_observe_slice() {
        init().unwrap();

        let mut c = DuplexChallenger::new().unwrap();

        let values: Vec<GoldilocksField> = (1..5).map(GoldilocksField).collect();
        c.observe_slice(&values).unwrap();

        let sample = c.sample().unwrap();
        assert_ne!(sample.0, 0);
    }

    #[test]
    fn test_challenger_sample_array() {
        init().unwrap();

        let mut c = DuplexChallenger::new().unwrap();
        c.observe(GoldilocksField(42)).unwrap();

        let samples = c.sample_array(5).unwrap();

        assert_eq!(samples.len(), 5);
        // Check they're not all the same
        let unique: std::collections::HashSet<u64> = samples.iter().map(|s| s.0).collect();
        assert!(unique.len() > 1);
    }

    #[test]
    fn test_challenger_ext2() {
        init().unwrap();

        let mut c = DuplexChallenger::new().unwrap();
        c.observe(GoldilocksField(12345)).unwrap();

        let ext2_sample = c.sample_ext2().unwrap();

        // Just check it doesn't crash and returns something
        assert!(ext2_sample.c0.0 != 0 || ext2_sample.c1.0 != 0);
    }
}
