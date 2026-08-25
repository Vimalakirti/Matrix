//! Rust bindings for CUDA kernels over the **almost-Goldilocks** prime field.
//!
//! `P = 2^64 - 2^32 + 1 - 32 = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1`.
//!
//! This crate is a sibling of `goldilocks-cuda-rs` but targets the slightly
//! different almost-Goldilocks prime. Field arithmetic uses a 2-pass Solinas
//! reduction (with the wrap constant `c = 2^32 + 31` instead of Goldilocks'
//! `2^32 - 1`), and the Ext2 non-residue is `W = 3` instead of `7` (since
//! `Legendre(7) = +1` mod this prime — `7` is a quadratic residue here).
//!
//! Provides:
//! - base field arithmetic ([`AlmostGoldilocksField`])
//! - quadratic extension `F_p[X] / (X^2 - 3)` ([`AlmostGoldilocksExt2`])
//! - eq-Lagrange over the Boolean hypercube ([`eq_lagrange`])
//! - multilinear partial evaluation ([`partial_eval`])
//! - sumcheck round-message + fold ([`sumcheck_prover`])
//!
//! Out of scope (intentionally): Poseidon2, Monolith, Challenger, Merkle,
//! and Basefold — they embed Goldilocks-specific round constants that would
//! need to be regenerated for this prime.

pub mod ajtai;
pub mod conv;
pub mod einsum;
pub mod eq_lagrange;
pub mod error;
pub mod extension;
pub mod ffi;
pub mod field;
pub mod memory;
pub mod partial_eval;
pub mod sumcheck_prover;

use std::sync::Mutex;
use std::sync::OnceLock;

pub use error::{CudaError, Result};
pub use extension::{AlmostExt2Batch, AlmostExt2Ops, AlmostGoldilocksExt2, AEXT2_W, AEXT2_DTH_ROOT};
pub use field::{
    AlmostGoldilocksBatch, AlmostGoldilocksField, AlmostGoldilocksOps,
    ALMOST_GOLDILOCKS_PRIME, ALMOST_REDUCE_C, ALMOST_HALF_P_PLUS_ONE,
};
pub use memory::{
    get_last_error, mem_get_info, memcpy_dtod, peek_at_last_error, pool_trim, synchronize,
    DeviceBuffer,
};

/// Initialize the CUDA context and almost-Goldilocks subsystem. Safe to call
/// multiple times; the underlying setup runs exactly once.
pub fn init() -> Result<()> {
    static INIT_RESULT: OnceLock<Mutex<std::result::Result<(), CudaError>>> = OnceLock::new();
    let cell = INIT_RESULT.get_or_init(|| {
        let ret = unsafe { ffi::almost_goldilocks_cuda_init() };
        Mutex::new(if ret == 0 {
            Ok(())
        } else if ret == -1 {
            Err(CudaError::NoDevice)
        } else {
            Err(CudaError::InitializationFailed)
        })
    });
    match &*cell.lock().unwrap() {
        Ok(()) => Ok(()),
        Err(CudaError::NoDevice) => Err(CudaError::NoDevice),
        Err(_) => Err(CudaError::InitializationFailed),
    }
}

/// Set the active CUDA device for the calling thread. Subsequent
/// allocations and kernel launches from this thread will use this
/// device. Each rayon worker can target a different device — that's
/// how `zk-torch-4` distributes fold-tree buckets across multiple GPUs.
pub fn set_device(device: i32) -> Result<()> {
    let ret = unsafe { ffi::cuda_set_device(device) };
    if ret == 0 { Ok(()) } else { Err(CudaError::InitializationFailed) }
}

/// Best-effort: enable peer access from the CURRENT device to `peer`.
/// Returns false when P2P is unsupported for the pair — cross-device
/// copies still work via `cudaMemcpyPeer` (host-staged), just slower.
pub fn enable_peer_access(peer: i32) -> bool {
    unsafe { ffi::cuda_enable_peer_access(peer) == 0 }
}

/// The calling thread's current CUDA device (as set by [`set_device`]).
pub fn current_device() -> i32 {
    let mut d = 0i32;
    unsafe { let _ = ffi::cuda_get_device(&mut d as *mut i32); }
    d
}

/// Number of CUDA devices visible to this process (after
/// `CUDA_VISIBLE_DEVICES` filtering).
pub fn device_count() -> i32 {
    let mut count = 0i32;
    let ret = unsafe { ffi::cuda_get_device_count(&mut count as *mut i32) };
    if ret == 0 { count } else { 0 }
}

pub mod prelude {
    pub use crate::error::{CudaError, Result};
    pub use crate::extension::{AlmostExt2Ops, AlmostGoldilocksExt2};
    pub use crate::field::{
        AlmostGoldilocksField, AlmostGoldilocksOps, ALMOST_GOLDILOCKS_PRIME,
    };
    pub use crate::memory::DeviceBuffer;
}

/// Decode a CUDA error code returned by the FFI memcpy helpers.
pub fn cuda_error_string(code: i32) -> String {
    unsafe {
        let p = ffi::agl_cuda_error_string(code);
        if p.is_null() { return format!("code {}", code); }
        format!("{} (code {})",
                std::ffi::CStr::from_ptr(p).to_string_lossy(), code)
    }
}
