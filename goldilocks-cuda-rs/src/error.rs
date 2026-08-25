//! Error types for Goldilocks CUDA operations.

use thiserror::Error;

/// Errors that can occur during CUDA operations.
#[derive(Error, Debug)]
pub enum CudaError {
    /// CUDA initialization failed
    #[error("CUDA initialization failed")]
    InitializationFailed,

    /// Memory allocation failed
    #[error("CUDA memory allocation failed")]
    AllocationFailed,

    /// Memory copy failed
    #[error("CUDA memory copy failed")]
    MemcpyFailed,

    /// Kernel execution failed
    #[error("CUDA kernel execution failed")]
    KernelFailed,

    /// Device synchronization failed
    #[error("CUDA device synchronization failed")]
    SyncFailed,

    /// No CUDA device found
    #[error("No CUDA device found")]
    NoDevice,

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

/// Result type for CUDA operations.
pub type Result<T> = std::result::Result<T, CudaError>;
