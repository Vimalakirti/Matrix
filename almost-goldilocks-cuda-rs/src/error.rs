//! Error types for almost-Goldilocks CUDA operations.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA initialization failed")]
    InitializationFailed,
    #[error("CUDA memory allocation failed")]
    AllocationFailed,
    #[error("CUDA memory copy failed: {0}")]
    MemcpyFailed(String),
    #[error("CUDA kernel execution failed")]
    KernelFailed,
    #[error("CUDA device synchronization failed")]
    SyncFailed,
    #[error("No CUDA device found")]
    NoDevice,
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, CudaError>;
