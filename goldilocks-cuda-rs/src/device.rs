//! Device-side types that keep data on GPU.
//!
//! Use these types to avoid frequent CPU-GPU transfers. Data stays on GPU
//! until explicitly moved back to host with `to_host()`.
//!
//! # Example
//! ```ignore
//! use goldilocks_cuda::prelude::*;
//!
//! // Move data to GPU once
//! let a = GoldilocksVec::from_slice(&host_a)?.to_device()?;
//! let b = GoldilocksVec::from_slice(&host_b)?.to_device()?;
//!
//! // All operations happen on GPU - no transfers!
//! let c = (&a * &b)?;
//! let d = (&c + &a)?;
//! let e = d.inverse()?;
//!
//! // Only transfer when you need the result
//! let result = e.to_host()?;
//! ```

use crate::error::{CudaError, Result};
use crate::extension::GoldilocksExt2;
use crate::ffi;
use crate::field::GoldilocksField;
use crate::memory::DeviceBuffer;
use crate::poseidon2::{Poseidon2Hash, POSEIDON2_DIGEST_SIZE};
use std::os::raw::c_int;

// ============================================================================
// GoldilocksDevice - Goldilocks field elements on GPU
// ============================================================================

/// A vector of Goldilocks field elements stored on GPU.
#[derive(Debug)]
pub struct GoldilocksDevice {
    buffer: DeviceBuffer<u64>,
}

impl GoldilocksDevice {
    /// Create a new device buffer with uninitialized data.
    pub fn uninit(len: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::new(len)?,
        })
    }

    /// Create a device buffer from a host slice.
    pub fn from_slice(data: &[GoldilocksField]) -> Result<Self> {
        let u64_data: Vec<u64> = data.iter().map(|f| f.0).collect();
        Ok(Self {
            buffer: DeviceBuffer::from_slice(&u64_data)?,
        })
    }

    /// Create a device buffer from raw u64 slice.
    pub fn from_raw_slice(data: &[u64]) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::from_slice(data)?,
        })
    }

    /// Copy data back to host.
    pub fn to_host(&self) -> Result<Vec<GoldilocksField>> {
        let data = self.buffer.to_vec()?;
        Ok(data.into_iter().map(GoldilocksField).collect())
    }

    /// Copy data back to host as raw u64.
    pub fn to_host_raw(&self) -> Result<Vec<u64>> {
        self.buffer.to_vec()
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get raw pointer (for advanced use).
    pub fn as_ptr(&self) -> *const u64 {
        self.buffer.as_ptr()
    }

    /// Get mutable raw pointer (for advanced use).
    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.buffer.as_mut_ptr()
    }

    // ========================================================================
    // Arithmetic operations (all on GPU)
    // ========================================================================

    /// Element-wise addition: self + other
    pub fn add(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::gl_batch_add(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::gl_batch_sub(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::gl_batch_mul(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise inversion: self^(-1)
    pub fn inverse(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::gl_batch_inverse(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Scalar multiplication: scalar * self[i] for all i
    pub fn mul_scalar(&self, scalar: GoldilocksField) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::gl_batch_mul_scalar(
                scalar.0,
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Scalar multiplication with u64: scalar * self[i] for all i
    pub fn scale(&self, scalar: u64) -> Result<Self> {
        self.mul_scalar(GoldilocksField(scalar))
    }

    /// Convert to extension field (embedding): a -> (a, 0)
    pub fn to_ext2(&self) -> Result<Ext2Device> {
        let n = self.len();
        let mut result = Ext2Device::uninit(n)?;

        let ret = unsafe {
            ffi::gl_to_ext2_batch_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Convert to quintic extension field (embedding): a -> (a, 0, 0, 0, 0)
    pub fn to_ext5(&self) -> Result<Ext5Device> {
        let n = self.len();
        let mut result = Ext5Device::uninit(n)?;

        let ret = unsafe {
            ffi::gl_to_ext5_batch_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise negation: -self
    pub fn neg(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::gl_batch_neg(self.buffer.as_ptr(), result.buffer.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise squaring: self^2
    pub fn square(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::gl_batch_square(self.buffer.as_ptr(), result.buffer.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise doubling: 2 * self
    pub fn double(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::gl_batch_double(self.buffer.as_ptr(), result.buffer.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise exponentiation: self^exp
    pub fn exp(&self, exp: u64) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::gl_batch_exp(
                self.buffer.as_ptr(),
                exp,
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise division: self / other
    pub fn div(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::gl_batch_div(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }
}

// Operator overloads for ergonomic syntax
impl std::ops::Add for &GoldilocksDevice {
    type Output = Result<GoldilocksDevice>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl std::ops::Sub for &GoldilocksDevice {
    type Output = Result<GoldilocksDevice>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other)
    }
}

impl std::ops::Mul for &GoldilocksDevice {
    type Output = Result<GoldilocksDevice>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

// ============================================================================
// Ext2Device - Extension field elements on GPU
// ============================================================================

/// A vector of GF(p^2) extension field elements stored on GPU.
/// Each element is stored as [c0, c1] (2 u64 values).
#[derive(Debug)]
pub struct Ext2Device {
    buffer: DeviceBuffer<u64>,
}

impl Ext2Device {
    /// Create a new device buffer with uninitialized data.
    /// `len` is the number of Ext2 elements (buffer will have 2*len u64 values).
    pub fn uninit(len: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::new(len * 2)?,
        })
    }

    /// Create a device buffer from a host slice.
    pub fn from_slice(data: &[GoldilocksExt2]) -> Result<Self> {
        let u64_data: Vec<u64> = data.iter().flat_map(|e| [e.c0.0, e.c1.0]).collect();
        Ok(Self {
            buffer: DeviceBuffer::from_slice(&u64_data)?,
        })
    }

    /// Copy data back to host.
    pub fn to_host(&self) -> Result<Vec<GoldilocksExt2>> {
        let data = self.buffer.to_vec()?;
        Ok(data
            .chunks_exact(2)
            .map(|c| GoldilocksExt2::new(GoldilocksField(c[0]), GoldilocksField(c[1])))
            .collect())
    }

    /// Number of Ext2 elements.
    pub fn len(&self) -> usize {
        self.buffer.len() / 2
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get raw pointer (for advanced use).
    pub fn as_ptr(&self) -> *const u64 {
        self.buffer.as_ptr()
    }

    /// Get mutable raw pointer (for advanced use).
    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.buffer.as_mut_ptr()
    }

    // ========================================================================
    // Arithmetic operations (all on GPU)
    // ========================================================================

    /// Element-wise addition: self + other
    pub fn add(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::ext2_batch_add_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::ext2_batch_sub_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::ext2_batch_mul_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise inversion: self^(-1)
    pub fn inverse(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_inverse_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Scalar multiplication by base field element: scalar * self[i] for all i
    pub fn mul_scalar(&self, scalar: GoldilocksField) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_mul_scalar_ffi(
                scalar.0,
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Scalar multiplication with u64: scalar * self[i] for all i
    pub fn scale(&self, scalar: u64) -> Result<Self> {
        self.mul_scalar(GoldilocksField(scalar))
    }

    /// Extract base field component (c0) from each element.
    pub fn to_base(&self) -> Result<GoldilocksDevice> {
        let n = self.len();
        let mut result = GoldilocksDevice::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_to_gl_batch_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise negation: -self
    pub fn neg(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_neg_ffi(self.buffer.as_ptr(), result.buffer.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise squaring: self^2
    pub fn square(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_square_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Frobenius endomorphism: self^p (where p is the Goldilocks prime)
    pub fn frobenius(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_frobenius_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Conjugate: (c0, c1) -> (c0, -c1)
    pub fn conjugate(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_conjugate_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise exponentiation: self^exp
    pub fn exp(&self, exp: u64) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext2_batch_exp_ffi(
                self.buffer.as_ptr(),
                exp,
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }
}

// Operator overloads
impl std::ops::Add for &Ext2Device {
    type Output = Result<Ext2Device>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl std::ops::Sub for &Ext2Device {
    type Output = Result<Ext2Device>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other)
    }
}

impl std::ops::Mul for &Ext2Device {
    type Output = Result<Ext2Device>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

// ============================================================================
// Ext5Device - Quintic extension field elements on GPU
// ============================================================================

/// A vector of GF(p^5) quintic extension field elements stored on GPU.
/// Each element is stored as [c0, c1, c2, c3, c4] (5 u64 values).
/// The extension is defined by X^5 - 3 (W = 3).
#[derive(Debug)]
pub struct Ext5Device {
    buffer: DeviceBuffer<u64>,
}

impl Ext5Device {
    /// Create a new device buffer with uninitialized data.
    /// `len` is the number of Ext5 elements (buffer will have 5*len u64 values).
    pub fn uninit(len: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::new(len * 5)?,
        })
    }

    /// Create a device buffer from raw coefficients.
    /// Each element has 5 coefficients [c0, c1, c2, c3, c4].
    pub fn from_raw_slice(data: &[u64]) -> Result<Self> {
        if data.len() % 5 != 0 {
            return Err(CudaError::InvalidArgument(
                "Data length must be a multiple of 5".to_string(),
            ));
        }
        Ok(Self {
            buffer: DeviceBuffer::from_slice(data)?,
        })
    }

    /// Copy data back to host as raw u64 values.
    /// Returns flat array [c0, c1, c2, c3, c4, c0, c1, c2, c3, c4, ...].
    pub fn to_host_raw(&self) -> Result<Vec<u64>> {
        self.buffer.to_vec()
    }

    /// Copy data back to host as arrays of coefficients.
    pub fn to_host(&self) -> Result<Vec<[GoldilocksField; 5]>> {
        let data = self.buffer.to_vec()?;
        Ok(data
            .chunks_exact(5)
            .map(|c| {
                [
                    GoldilocksField(c[0]),
                    GoldilocksField(c[1]),
                    GoldilocksField(c[2]),
                    GoldilocksField(c[3]),
                    GoldilocksField(c[4]),
                ]
            })
            .collect())
    }

    /// Number of Ext5 elements.
    pub fn len(&self) -> usize {
        self.buffer.len() / 5
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get raw pointer (for advanced use).
    pub fn as_ptr(&self) -> *const u64 {
        self.buffer.as_ptr()
    }

    /// Get mutable raw pointer (for advanced use).
    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.buffer.as_mut_ptr()
    }

    // ========================================================================
    // Arithmetic operations (all on GPU)
    // ========================================================================

    /// Element-wise addition: self + other
    pub fn add(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::ext5_batch_add_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::ext5_batch_sub_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::ext5_batch_mul_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise inversion: self^(-1)
    pub fn inverse(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_batch_inverse_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Scalar multiplication by base field element: scalar * self[i] for all i
    pub fn mul_scalar(&self, scalar: GoldilocksField) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_batch_mul_scalar_ffi(
                scalar.0,
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Scalar multiplication with u64: scalar * self[i] for all i
    pub fn scale(&self, scalar: u64) -> Result<Self> {
        self.mul_scalar(GoldilocksField(scalar))
    }

    /// Element-wise negation: -self
    pub fn neg(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_batch_neg_ffi(self.buffer.as_ptr(), result.buffer.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise squaring: self^2
    pub fn square(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_batch_square_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Frobenius endomorphism: self^p (where p is the Goldilocks prime)
    pub fn frobenius(&self) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_batch_frobenius_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Element-wise exponentiation: self^exp
    pub fn exp(&self, exp: u64) -> Result<Self> {
        let n = self.len();
        let mut result = Self::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_batch_exp_ffi(
                self.buffer.as_ptr(),
                exp,
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Extract base field component (c0) from each element.
    pub fn to_base(&self) -> Result<GoldilocksDevice> {
        let n = self.len();
        let mut result = GoldilocksDevice::uninit(n)?;

        let ret = unsafe {
            ffi::ext5_to_gl_batch_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }
}

// Operator overloads
impl std::ops::Add for &Ext5Device {
    type Output = Result<Ext5Device>;

    fn add(self, other: Self) -> Self::Output {
        self.add(other)
    }
}

impl std::ops::Sub for &Ext5Device {
    type Output = Result<Ext5Device>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(other)
    }
}

impl std::ops::Mul for &Ext5Device {
    type Output = Result<Ext5Device>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(other)
    }
}

// ============================================================================
// Poseidon2Device - Poseidon2 hashes on GPU
// ============================================================================

/// A vector of Poseidon2 hash digests stored on GPU.
/// Each digest is 4 field elements.
#[derive(Debug)]
pub struct Poseidon2Device {
    buffer: DeviceBuffer<u64>,
}

impl Poseidon2Device {
    /// Create a new device buffer with uninitialized data.
    /// `len` is the number of hash digests.
    pub fn uninit(len: usize) -> Result<Self> {
        Ok(Self {
            buffer: DeviceBuffer::new(len * POSEIDON2_DIGEST_SIZE)?,
        })
    }

    /// Create a device buffer from a host slice.
    pub fn from_slice(data: &[Poseidon2Hash]) -> Result<Self> {
        let u64_data: Vec<u64> = data
            .iter()
            .flat_map(|h| h.elements.iter().map(|f| f.0))
            .collect();
        Ok(Self {
            buffer: DeviceBuffer::from_slice(&u64_data)?,
        })
    }

    /// Copy data back to host.
    pub fn to_host(&self) -> Result<Vec<Poseidon2Hash>> {
        let data = self.buffer.to_vec()?;
        Ok(data
            .chunks_exact(POSEIDON2_DIGEST_SIZE)
            .map(|c| Poseidon2Hash::from_raw([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Number of hash digests.
    pub fn len(&self) -> usize {
        self.buffer.len() / POSEIDON2_DIGEST_SIZE
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get raw pointer (for advanced use).
    pub fn as_ptr(&self) -> *const u64 {
        self.buffer.as_ptr()
    }

    /// Get mutable raw pointer (for advanced use).
    pub fn as_mut_ptr(&mut self) -> *mut u64 {
        self.buffer.as_mut_ptr()
    }

    // ========================================================================
    // Poseidon2 operations (all on GPU)
    // ========================================================================

    /// Compress pairs of hashes: compress(self[i], other[i])
    pub fn compress(&self, other: &Self) -> Result<Self> {
        let n = self.len();
        if other.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let mut result = Self::uninit(n)?;
        let ret = unsafe {
            ffi::poseidon2_compress_batch_ffi(
                self.buffer.as_ptr(),
                other.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Compute one layer of a Merkle tree.
    /// Input has 2*n nodes, output has n nodes.
    pub fn merkle_layer(&self) -> Result<Self> {
        let n_input = self.len();
        if n_input % 2 != 0 {
            return Err(CudaError::InvalidArgument(
                "Number of nodes must be even".to_string(),
            ));
        }

        let n_output = n_input / 2;
        let mut result = Self::uninit(n_output)?;

        let ret = unsafe {
            ffi::poseidon2_merkle_layer_ffi(
                self.buffer.as_ptr(),
                result.buffer.as_mut_ptr(),
                n_output as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(result)
    }

    /// Build a complete Merkle tree from leaves.
    /// Returns the root hash. All computation happens on GPU.
    pub fn merkle_root(&self) -> Result<Poseidon2Hash> {
        let n = self.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(CudaError::InvalidArgument(
                "Number of leaves must be a power of 2".to_string(),
            ));
        }

        let mut current = Self {
            buffer: DeviceBuffer::from_slice(&self.buffer.to_vec()?)?,
        };

        while current.len() > 1 {
            current = current.merkle_layer()?;
        }

        let result = current.to_host()?;
        Ok(result[0])
    }

    /// Build a complete Merkle tree and return all layers on GPU.
    /// layers[0] = leaves, layers[last] = root
    pub fn merkle_tree(&self) -> Result<Vec<Self>> {
        let n = self.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(CudaError::InvalidArgument(
                "Number of leaves must be a power of 2".to_string(),
            ));
        }

        let mut layers = Vec::new();

        // Clone leaves as first layer
        let leaves = Self {
            buffer: DeviceBuffer::from_slice(&self.buffer.to_vec()?)?,
        };
        layers.push(leaves);

        while layers.last().unwrap().len() > 1 {
            let next = layers.last().unwrap().merkle_layer()?;
            layers.push(next);
        }

        Ok(layers)
    }
}

// ============================================================================
// Convenience trait for to_device()
// ============================================================================

/// Trait for types that can be moved to GPU.
pub trait ToDevice {
    type DeviceType;

    /// Move data to GPU.
    fn to_device(&self) -> Result<Self::DeviceType>;
}

impl ToDevice for [GoldilocksField] {
    type DeviceType = GoldilocksDevice;

    fn to_device(&self) -> Result<Self::DeviceType> {
        GoldilocksDevice::from_slice(self)
    }
}

impl ToDevice for Vec<GoldilocksField> {
    type DeviceType = GoldilocksDevice;

    fn to_device(&self) -> Result<Self::DeviceType> {
        GoldilocksDevice::from_slice(self)
    }
}

impl ToDevice for [GoldilocksExt2] {
    type DeviceType = Ext2Device;

    fn to_device(&self) -> Result<Self::DeviceType> {
        Ext2Device::from_slice(self)
    }
}

impl ToDevice for Vec<GoldilocksExt2> {
    type DeviceType = Ext2Device;

    fn to_device(&self) -> Result<Self::DeviceType> {
        Ext2Device::from_slice(self)
    }
}

impl ToDevice for [Poseidon2Hash] {
    type DeviceType = Poseidon2Device;

    fn to_device(&self) -> Result<Self::DeviceType> {
        Poseidon2Device::from_slice(self)
    }
}

impl ToDevice for Vec<Poseidon2Hash> {
    type DeviceType = Poseidon2Device;

    fn to_device(&self) -> Result<Self::DeviceType> {
        Poseidon2Device::from_slice(self)
    }
}

impl ToDevice for [[GoldilocksField; 5]] {
    type DeviceType = Ext5Device;

    fn to_device(&self) -> Result<Self::DeviceType> {
        let raw: Vec<u64> = self.iter().flat_map(|e| e.iter().map(|f| f.0)).collect();
        Ext5Device::from_raw_slice(&raw)
    }
}

impl ToDevice for Vec<[GoldilocksField; 5]> {
    type DeviceType = Ext5Device;

    fn to_device(&self) -> Result<Self::DeviceType> {
        let raw: Vec<u64> = self.iter().flat_map(|e| e.iter().map(|f| f.0)).collect();
        Ext5Device::from_raw_slice(&raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init;

    #[test]
    fn test_goldilocks_device_ops() {
        init().unwrap();

        let a: Vec<GoldilocksField> = (1..101).map(GoldilocksField::new).collect();
        let b: Vec<GoldilocksField> = (1..101).map(GoldilocksField::new).collect();

        // Move to device
        let d_a = a.to_device().unwrap();
        let d_b = b.to_device().unwrap();

        // All ops on GPU
        let d_sum = d_a.add(&d_b).unwrap();
        let d_prod = d_a.mul(&d_b).unwrap();

        // Get results
        let sum = d_sum.to_host().unwrap();
        let prod = d_prod.to_host().unwrap();

        assert_eq!(sum[0].0, 2);  // 1 + 1
        assert_eq!(prod[0].0, 1); // 1 * 1
        assert_eq!(sum[99].0, 200); // 100 + 100
        assert_eq!(prod[99].0, 10000); // 100 * 100
    }

    #[test]
    fn test_ext2_device_ops() {
        init().unwrap();

        let a = vec![GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(2))];
        let b = vec![GoldilocksExt2::new(GoldilocksField(3), GoldilocksField(4))];

        let d_a = a.to_device().unwrap();
        let d_b = b.to_device().unwrap();

        let d_prod = d_a.mul(&d_b).unwrap();
        let prod = d_prod.to_host().unwrap();

        // (1 + 2X)(3 + 4X) = 3 + 4X + 6X + 8*7 = 59 + 10X
        assert_eq!(prod[0].c0.0, 59);
        assert_eq!(prod[0].c1.0, 10);
    }

    #[test]
    fn test_conversion_on_device() {
        init().unwrap();

        let base: Vec<GoldilocksField> = (1..101).map(GoldilocksField::new).collect();

        // Move to device and convert to ext2
        let d_base = base.to_device().unwrap();
        let d_ext2 = d_base.to_ext2().unwrap();

        // Verify
        let ext2 = d_ext2.to_host().unwrap();
        assert_eq!(ext2[0].c0.0, 1);
        assert_eq!(ext2[0].c1.0, 0);
        assert_eq!(ext2[99].c0.0, 100);
        assert_eq!(ext2[99].c1.0, 0);
    }

    #[test]
    fn test_merkle_on_device() {
        init().unwrap();

        let leaves: Vec<Poseidon2Hash> = (0..8)
            .map(|i| Poseidon2Hash::from_raw([i, 0, 0, 0]))
            .collect();

        let d_leaves = leaves.to_device().unwrap();
        let root = d_leaves.merkle_root().unwrap();

        // Just verify it completes
        println!("Root: {:?}", root.to_raw());
    }

    #[test]
    fn test_mul_scalar() {
        init().unwrap();

        // Test base field scalar multiplication
        let a: Vec<GoldilocksField> = (1..11).map(GoldilocksField::new).collect();
        let d_a = a.to_device().unwrap();

        // Multiply by 3
        let d_result = d_a.scale(3).unwrap();
        let result = d_result.to_host().unwrap();

        for i in 0..10 {
            assert_eq!(result[i].0, (i as u64 + 1) * 3);
        }

        // Test extension field scalar multiplication
        let ext_a = vec![
            GoldilocksExt2::new(GoldilocksField(2), GoldilocksField(3)),
            GoldilocksExt2::new(GoldilocksField(4), GoldilocksField(5)),
        ];
        let d_ext = ext_a.to_device().unwrap();

        // Multiply by 2: 2*(2+3X) = 4+6X, 2*(4+5X) = 8+10X
        let d_ext_result = d_ext.scale(2).unwrap();
        let ext_result = d_ext_result.to_host().unwrap();

        assert_eq!(ext_result[0].c0.0, 4);
        assert_eq!(ext_result[0].c1.0, 6);
        assert_eq!(ext_result[1].c0.0, 8);
        assert_eq!(ext_result[1].c1.0, 10);
    }

    #[test]
    fn test_goldilocks_neg_square_double() {
        init().unwrap();

        let a: Vec<GoldilocksField> = vec![
            GoldilocksField::new(5),
            GoldilocksField::new(10),
            GoldilocksField::new(100),
        ];
        let d_a = a.to_device().unwrap();

        // Test negation: -5 mod p
        let d_neg = d_a.neg().unwrap();
        let neg = d_neg.to_host().unwrap();
        // -5 mod p = p - 5
        assert_eq!(neg[0].0, crate::field::GOLDILOCKS_PRIME - 5);

        // Test square: 5^2 = 25, 10^2 = 100, 100^2 = 10000
        let d_sq = d_a.square().unwrap();
        let sq = d_sq.to_host().unwrap();
        assert_eq!(sq[0].0, 25);
        assert_eq!(sq[1].0, 100);
        assert_eq!(sq[2].0, 10000);

        // Test double: 2*5 = 10, 2*10 = 20, 2*100 = 200
        let d_dbl = d_a.double().unwrap();
        let dbl = d_dbl.to_host().unwrap();
        assert_eq!(dbl[0].0, 10);
        assert_eq!(dbl[1].0, 20);
        assert_eq!(dbl[2].0, 200);
    }

    #[test]
    fn test_goldilocks_exp_div() {
        init().unwrap();

        let a: Vec<GoldilocksField> = vec![
            GoldilocksField::new(2),
            GoldilocksField::new(3),
        ];
        let b: Vec<GoldilocksField> = vec![
            GoldilocksField::new(4),
            GoldilocksField::new(9),
        ];

        let d_a = a.to_device().unwrap();
        let d_b = b.to_device().unwrap();

        // Test exp: 2^3 = 8, 3^3 = 27
        let d_exp = d_a.exp(3).unwrap();
        let exp = d_exp.to_host().unwrap();
        assert_eq!(exp[0].0, 8);
        assert_eq!(exp[1].0, 27);

        // Test div: 4/2 = 2, 9/3 = 3
        let d_div = d_b.div(&d_a).unwrap();
        let div = d_div.to_host().unwrap();
        assert_eq!(div[0].0, 2);
        assert_eq!(div[1].0, 3);
    }

    #[test]
    fn test_goldilocks_to_ext5() {
        init().unwrap();

        let base: Vec<GoldilocksField> = vec![
            GoldilocksField::new(42),
            GoldilocksField::new(100),
        ];
        let d_base = base.to_device().unwrap();

        // Convert to ext5: a -> (a, 0, 0, 0, 0)
        let d_ext5 = d_base.to_ext5().unwrap();
        let ext5 = d_ext5.to_host().unwrap();

        assert_eq!(ext5[0][0].0, 42);
        assert_eq!(ext5[0][1].0, 0);
        assert_eq!(ext5[0][2].0, 0);
        assert_eq!(ext5[0][3].0, 0);
        assert_eq!(ext5[0][4].0, 0);

        assert_eq!(ext5[1][0].0, 100);
        assert_eq!(ext5[1][1].0, 0);
    }

    #[test]
    fn test_ext2_neg_square() {
        init().unwrap();

        let a = vec![GoldilocksExt2::new(GoldilocksField(3), GoldilocksField(4))];
        let d_a = a.to_device().unwrap();

        // Test negation: -(3 + 4X) = (-3, -4) mod p
        let d_neg = d_a.neg().unwrap();
        let neg = d_neg.to_host().unwrap();
        assert_eq!(neg[0].c0.0, crate::field::GOLDILOCKS_PRIME - 3);
        assert_eq!(neg[0].c1.0, crate::field::GOLDILOCKS_PRIME - 4);

        // Test square: (3 + 4X)^2 = 9 + 24X + 16*7 = 9 + 112 + 24X = 121 + 24X
        let d_sq = d_a.square().unwrap();
        let sq = d_sq.to_host().unwrap();
        assert_eq!(sq[0].c0.0, 121);  // 9 + 16*7 = 9 + 112 = 121
        assert_eq!(sq[0].c1.0, 24);   // 2*3*4 = 24
    }

    #[test]
    fn test_ext2_frobenius_conjugate() {
        init().unwrap();

        let a = vec![GoldilocksExt2::new(GoldilocksField(5), GoldilocksField(7))];
        let d_a = a.to_device().unwrap();

        // Test conjugate: (5, 7) -> (5, -7)
        let d_conj = d_a.conjugate().unwrap();
        let conj = d_conj.to_host().unwrap();
        assert_eq!(conj[0].c0.0, 5);
        assert_eq!(conj[0].c1.0, crate::field::GOLDILOCKS_PRIME - 7);

        // Test frobenius - should equal conjugate for quadratic extension
        let d_frob = d_a.frobenius().unwrap();
        let frob = d_frob.to_host().unwrap();
        assert_eq!(frob[0].c0.0, conj[0].c0.0);
        assert_eq!(frob[0].c1.0, conj[0].c1.0);
    }

    #[test]
    fn test_ext2_exp() {
        init().unwrap();

        // Test (1 + X)^2 = 1 + 2X + 7 = 8 + 2X
        let a = vec![GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(1))];
        let d_a = a.to_device().unwrap();

        let d_exp = d_a.exp(2).unwrap();
        let exp = d_exp.to_host().unwrap();
        assert_eq!(exp[0].c0.0, 8);  // 1 + 1*7 = 8
        assert_eq!(exp[0].c1.0, 2);  // 2*1*1 = 2
    }

    #[test]
    fn test_ext5_basic_ops() {
        init().unwrap();

        // Create two ext5 elements
        let a = vec![[
            GoldilocksField(1),
            GoldilocksField(2),
            GoldilocksField(0),
            GoldilocksField(0),
            GoldilocksField(0),
        ]];
        let b = vec![[
            GoldilocksField(3),
            GoldilocksField(4),
            GoldilocksField(0),
            GoldilocksField(0),
            GoldilocksField(0),
        ]];

        let d_a = a.to_device().unwrap();
        let d_b = b.to_device().unwrap();

        // Test add: (1 + 2X) + (3 + 4X) = 4 + 6X
        let d_sum = d_a.add(&d_b).unwrap();
        let sum = d_sum.to_host().unwrap();
        assert_eq!(sum[0][0].0, 4);
        assert_eq!(sum[0][1].0, 6);

        // Test sub: (1 + 2X) - (3 + 4X) = -2 - 2X
        let d_diff = d_a.sub(&d_b).unwrap();
        let diff = d_diff.to_host().unwrap();
        assert_eq!(diff[0][0].0, crate::field::GOLDILOCKS_PRIME - 2);
        assert_eq!(diff[0][1].0, crate::field::GOLDILOCKS_PRIME - 2);

        // Test neg
        let d_neg = d_a.neg().unwrap();
        let neg = d_neg.to_host().unwrap();
        assert_eq!(neg[0][0].0, crate::field::GOLDILOCKS_PRIME - 1);
        assert_eq!(neg[0][1].0, crate::field::GOLDILOCKS_PRIME - 2);
    }

    #[test]
    fn test_ext5_mul_inverse() {
        init().unwrap();

        // Test a * a^(-1) = 1
        let a = vec![[
            GoldilocksField(5),
            GoldilocksField(3),
            GoldilocksField(1),
            GoldilocksField(0),
            GoldilocksField(0),
        ]];

        let d_a = a.to_device().unwrap();
        let d_inv = d_a.inverse().unwrap();
        let d_prod = d_a.mul(&d_inv).unwrap();
        let prod = d_prod.to_host().unwrap();

        // Should be 1 (the multiplicative identity)
        assert_eq!(prod[0][0].0, 1);
        assert_eq!(prod[0][1].0, 0);
        assert_eq!(prod[0][2].0, 0);
        assert_eq!(prod[0][3].0, 0);
        assert_eq!(prod[0][4].0, 0);
    }

    #[test]
    fn test_ext5_scale_square() {
        init().unwrap();

        let a = vec![[
            GoldilocksField(2),
            GoldilocksField(3),
            GoldilocksField(0),
            GoldilocksField(0),
            GoldilocksField(0),
        ]];

        let d_a = a.to_device().unwrap();

        // Test scale: 5 * (2 + 3X) = 10 + 15X
        let d_scaled = d_a.scale(5).unwrap();
        let scaled = d_scaled.to_host().unwrap();
        assert_eq!(scaled[0][0].0, 10);
        assert_eq!(scaled[0][1].0, 15);

        // Test square: (2 + 3X)^2 = 4 + 12X + 9X^2
        let d_sq = d_a.square().unwrap();
        let sq = d_sq.to_host().unwrap();
        assert_eq!(sq[0][0].0, 4);   // 2^2
        assert_eq!(sq[0][1].0, 12);  // 2*2*3
        assert_eq!(sq[0][2].0, 9);   // 3^2
    }

    #[test]
    fn test_ext5_to_base() {
        init().unwrap();

        let a = vec![[
            GoldilocksField(42),
            GoldilocksField(7),
            GoldilocksField(3),
            GoldilocksField(1),
            GoldilocksField(0),
        ]];

        let d_a = a.to_device().unwrap();

        // Extract c0 component
        let d_base = d_a.to_base().unwrap();
        let base = d_base.to_host().unwrap();
        assert_eq!(base[0].0, 42);
    }

    #[test]
    fn test_ext5_exp() {
        init().unwrap();

        // Test a^1 = a
        let a = vec![[
            GoldilocksField(2),
            GoldilocksField(1),
            GoldilocksField(0),
            GoldilocksField(0),
            GoldilocksField(0),
        ]];

        let d_a = a.to_device().unwrap();

        let d_exp1 = d_a.exp(1).unwrap();
        let exp1 = d_exp1.to_host().unwrap();
        assert_eq!(exp1[0][0].0, 2);
        assert_eq!(exp1[0][1].0, 1);

        // Test a^0 = 1
        let d_exp0 = d_a.exp(0).unwrap();
        let exp0 = d_exp0.to_host().unwrap();
        assert_eq!(exp0[0][0].0, 1);
        assert_eq!(exp0[0][1].0, 0);
    }
}
