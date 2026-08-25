//! Goldilocks CUDA - GPU-accelerated Goldilocks field arithmetic, extension fields, and Poseidon2 hashing.
//!
//! This crate provides CUDA-accelerated implementations of:
//! - Goldilocks prime field arithmetic (p = 2^64 - 2^32 + 1)
//! - Quadratic extension field (GF(p^2) with X^2 - 7)
//! - Poseidon2 hash function
//! - Fiat-Shamir DuplexChallenger
//!
//! # Example
//!
//! ```ignore
//! use goldilocks_cuda::prelude::*;
//!
//! // Initialize CUDA
//! goldilocks_cuda::init().expect("Failed to initialize CUDA");
//!
//! // Batch field multiplication
//! let a: Vec<GoldilocksField> = (1..1001).map(GoldilocksField::new).collect();
//! let b: Vec<GoldilocksField> = (1..1001).map(GoldilocksField::new).collect();
//! let result = GoldilocksOps::mul(&a, &b).expect("Multiplication failed");
//!
//! // Extension field operations
//! let ext_a = vec![GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(2))];
//! let ext_b = vec![GoldilocksExt2::new(GoldilocksField(3), GoldilocksField(4))];
//! let ext_result = Ext2Ops::mul(&ext_a, &ext_b).expect("Ext2 multiplication failed");
//!
//! // Poseidon2 hashing
//! let leaves: Vec<Poseidon2Hash> = (0..8)
//!     .map(|i| Poseidon2Hash::from_raw([i, 0, 0, 0]))
//!     .collect();
//! let root = Poseidon2Ops::merkle_root(&leaves).expect("Merkle tree failed");
//!
//! // Fiat-Shamir challenger
//! use goldilocks_cuda::challenger::DuplexChallenger;
//! let mut challenger = DuplexChallenger::new().unwrap();
//! challenger.observe(GoldilocksField(123)).unwrap();
//! let challenge = challenger.sample().unwrap();
//! ```

pub mod basefold;
pub mod bit_decomp;
pub mod challenger;
pub mod conv;
pub mod cpu_poseidon2;
pub mod cpu_monolith;
pub mod device;
pub mod einsum;
pub mod eq_lagrange;
pub mod error;
pub mod extension;
mod ffi;
pub mod field;
pub mod memory;
pub mod merkle;
pub mod partial_eval;
pub mod poseidon2;
pub mod sumcheck_prover;

use std::os::raw::c_int;
use std::sync::Once;

pub use device::{Ext2Device, Ext5Device, GoldilocksDevice, Poseidon2Device, ToDevice};
pub use error::{CudaError, Result};
pub use extension::{Ext2Batch, Ext2Ops, GoldilocksExt2, EXT2_W};
pub use field::{GoldilocksBatch, GoldilocksField, GoldilocksOps, GOLDILOCKS_PRIME};
pub use memory::{memcpy_dtod, synchronize, get_last_error, peek_at_last_error, mem_get_info, DeviceBuffer};
pub use partial_eval::fused_permute_partial_eval;
pub use poseidon2::{
    Poseidon2Batch, Poseidon2Hash, Poseidon2Ops, POSEIDON2_DIGEST_SIZE, POSEIDON2_RATE,
    POSEIDON2_WIDTH,
};
pub use basefold::{
    batch_open, batch_open_ext2, BasefoldBatch, BasefoldCommitment, BasefoldProof,
    BasefoldProofExt2, BasefoldTable, BasefoldTranscript, BasefoldVerifier, BatchBasefoldProof,
    BatchBasefoldProofExt2, Evaluation, EvaluationExt2, FoldingEntry, HostCommitmentCache,
    IndividualQueryProof, QueryProof, SumcheckOracle, TestTranscript,
    ext2_eq_dp_all_u64, reduce_dot_product_ext2,
    sumcheck_product_and_reduce_mixed, sumcheck_product_and_reduce_ext2,
};
pub use merkle::DeviceMerkleTree;
pub use challenger::{
    ChallengerBatch, DuplexChallenger, CHALLENGER_CAPACITY, CHALLENGER_RATE, CHALLENGER_WIDTH,
};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::basefold::{
        batch_open, batch_open_ext2, BasefoldBatch, BasefoldCommitment, BasefoldProof,
        BasefoldProofExt2, BasefoldTable, BasefoldTranscript, BasefoldVerifier,
        BatchBasefoldProof, BatchBasefoldProofExt2, Evaluation, EvaluationExt2, FoldingEntry,
        HostCommitmentCache, IndividualQueryProof, QueryProof, SumcheckOracle, TestTranscript,
    };
    pub use crate::merkle::DeviceMerkleTree;
    pub use crate::challenger::{ChallengerBatch, DuplexChallenger};
    pub use crate::device::{Ext2Device, Ext5Device, GoldilocksDevice, Poseidon2Device, ToDevice};
    pub use crate::error::{CudaError, Result};
    pub use crate::extension::{Ext2Batch, Ext2Ops, GoldilocksExt2};
    pub use crate::field::{GoldilocksBatch, GoldilocksField, GoldilocksOps};
    pub use crate::memory::{synchronize, DeviceBuffer};
    pub use crate::poseidon2::{Poseidon2Batch, Poseidon2Hash, Poseidon2Ops};
}

static INIT: Once = Once::new();
static mut INIT_RESULT: i32 = 0;

/// Initialize the CUDA library.
/// This must be called before using any GPU operations.
/// It is safe to call this multiple times; subsequent calls are no-ops.
pub fn init() -> Result<()> {
    INIT.call_once(|| {
        let ret = unsafe { ffi::goldilocks_cuda_init() };
        if ret != 0 {
            unsafe { INIT_RESULT = ret };
            return;
        }
        let ret = unsafe { ffi::poseidon2_cuda_init() };
        if ret != 0 {
            unsafe { INIT_RESULT = ret };
            return;
        }
        let ret = unsafe { ffi::monolith_cuda_init() };
        unsafe { INIT_RESULT = ret };
    });

    if unsafe { INIT_RESULT } != 0 {
        return Err(CudaError::InitializationFailed);
    }
    Ok(())
}

/// Initialize GPU constants on the CURRENT device.
/// Unlike `init()`, this can be called multiple times for different devices.
/// Call `set_device(d)` first, then `init_device()` to load hash constants on device d.
pub fn init_device() -> Result<()> {
    let ret = unsafe { ffi::poseidon2_cuda_init() };
    if ret != 0 {
        return Err(CudaError::InitializationFailed);
    }
    let ret = unsafe { ffi::monolith_cuda_init() };
    if ret != 0 {
        return Err(CudaError::InitializationFailed);
    }
    Ok(())
}

/// Set the active CUDA device for the current thread.
pub fn set_device(device: i32) -> Result<()> {
    let ret = unsafe { ffi::cuda_set_device(device as c_int) };
    if ret != 0 {
        return Err(CudaError::InitializationFailed);
    }
    Ok(())
}

/// Get the current CUDA device for the calling thread.
pub fn get_device() -> Result<i32> {
    let mut device: c_int = 0;
    let ret = unsafe { ffi::cuda_get_device(&mut device) };
    if ret != 0 {
        return Err(CudaError::NoDevice);
    }
    Ok(device)
}

/// Get the number of available CUDA devices.
pub fn device_count() -> Result<i32> {
    let mut count: c_int = 0;
    let ret = unsafe { ffi::cuda_get_device_count(&mut count) };
    if ret != 0 {
        return Err(CudaError::NoDevice);
    }
    Ok(count)
}

/// Get the name of a CUDA device.
pub fn device_name(device: i32) -> Result<String> {
    let mut name = vec![0u8; 256];
    let ret = unsafe {
        ffi::cuda_get_device_name(device, name.as_mut_ptr() as *mut i8, 256)
    };
    if ret != 0 {
        return Err(CudaError::NoDevice);
    }

    // Find null terminator
    let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
    Ok(String::from_utf8_lossy(&name[..len]).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        let result = init();
        // This will fail if no CUDA device is available, which is OK for CI
        if let Ok(count) = device_count() {
            if count > 0 {
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_device_info() {
        if let Ok(count) = device_count() {
            if count > 0 {
                let name = device_name(0).unwrap();
                println!("GPU: {}", name);
                assert!(!name.is_empty());
            }
        }
    }
}
