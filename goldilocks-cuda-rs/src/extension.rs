//! Goldilocks extension field operations on GPU.

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::field::GoldilocksField;
use crate::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};
use std::os::raw::c_int;

/// Extension field parameter W for Ext2: X^2 - 7
pub const EXT2_W: u64 = 7;

/// Quadratic extension field element: F_p^2 = F_p[X] / (X^2 - 7)
/// Represented as c0 + c1*X where X^2 = 7
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct GoldilocksExt2 {
    pub c0: GoldilocksField,
    pub c1: GoldilocksField,
}

impl GoldilocksExt2 {
    /// Create a new extension field element.
    pub const fn new(c0: GoldilocksField, c1: GoldilocksField) -> Self {
        Self { c0, c1 }
    }

    /// Zero element.
    pub const fn zero() -> Self {
        Self {
            c0: GoldilocksField::zero(),
            c1: GoldilocksField::zero(),
        }
    }

    /// One element.
    pub const fn one() -> Self {
        Self {
            c0: GoldilocksField::one(),
            c1: GoldilocksField::zero(),
        }
    }

    /// Create an extension element from a base field element (embedding).
    /// This is the same as Ext2(a, 0).
    pub const fn from_base(base: GoldilocksField) -> Self {
        Self {
            c0: base,
            c1: GoldilocksField::zero(),
        }
    }

    /// Check if this element is in the base field (c1 == 0).
    pub fn is_base(&self) -> bool {
        self.c1.0 == 0
    }

    /// Extract the base field component (c0).
    pub fn to_base(&self) -> GoldilocksField {
        self.c0
    }

    /// Convert to raw u64 array representation [c0, c1].
    pub fn to_raw(&self) -> [u64; 2] {
        [self.c0.0, self.c1.0]
    }

    /// Create from raw u64 array representation [c0, c1].
    pub fn from_raw(raw: [u64; 2]) -> Self {
        Self {
            c0: GoldilocksField(raw[0]),
            c1: GoldilocksField(raw[1]),
        }
    }
}

impl From<GoldilocksField> for GoldilocksExt2 {
    fn from(base: GoldilocksField) -> Self {
        Self::from_base(base)
    }
}

impl From<u64> for GoldilocksExt2 {
    fn from(value: u64) -> Self {
        Self::from_base(GoldilocksField(value))
    }
}

impl std::ops::Add for GoldilocksExt2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 + rhs.c0,
            c1: self.c1 + rhs.c1,
        }
    }
}

impl std::ops::Sub for GoldilocksExt2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            c0: self.c0 - rhs.c0,
            c1: self.c1 - rhs.c1,
        }
    }
}

impl std::ops::Mul for GoldilocksExt2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Karatsuba: 3 base muls + cheap mul_by_7 instead of 5 muls
        let m0 = self.c0 * rhs.c0;                          // a0*b0
        let m1 = self.c1 * rhs.c1;                          // a1*b1
        let m2 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1);  // (a0+a1)*(b0+b1)
        Self {
            c0: m0 + GoldilocksField(EXT2_W) * m1,          // m0 + 7*m1
            c1: m2 - m0 - m1,                                // m2 - m0 - m1
        }
    }
}

/// Batch operations on Ext2 elements (low-level, requires pre-allocated device buffers).
pub struct Ext2Batch;

impl Ext2Batch {
    /// Batch addition: result[i] = a[i] + b[i]
    /// Buffers should contain 2*n u64 values (interleaved [c0, c1, c0, c1, ...])
    pub fn add(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if b.len() != a.len() || result.len() != a.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::ext2_batch_add_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch subtraction: result[i] = a[i] - b[i]
    pub fn sub(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if b.len() != a.len() || result.len() != a.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::ext2_batch_sub_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch multiplication: result[i] = a[i] * b[i]
    pub fn mul(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if b.len() != a.len() || result.len() != a.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::ext2_batch_mul_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch inversion: result[i] = a[i]^(-1)
    pub fn inverse(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::ext2_batch_inverse_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch conversion: Goldilocks -> Ext2
    pub fn from_base(input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = input.len();
        if output.len() != n * 2 {
            return Err(CudaError::InvalidArgument(
                "Output buffer must be 2x input length".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::gl_to_ext2_batch_ffi(input.as_ptr(), output.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Scale-accumulate: acc[i] += scalar * src[i] (all Ext2)
    /// Both src and acc contain n Ext2 elements (2n u64s each).
    pub fn scale_accumulate(
        scalar: GoldilocksExt2,
        src: &DeviceBuffer<u64>,
        acc: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        let n = src.len() / 2;
        if acc.len() != src.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::ext2_scale_accumulate_ffi(
                scalar.c0.0,
                scalar.c1.0,
                src.as_ptr(),
                acc.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch conversion: Ext2 -> Goldilocks (extracts c0)
    pub fn to_base(input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = input.len() / 2;
        if output.len() != n {
            return Err(CudaError::InvalidArgument(
                "Output buffer must be half input length".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::ext2_to_gl_batch_ffi(input.as_ptr(), output.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }
}

/// High-level batch operations that handle memory transfers.
pub struct Ext2Ops;

impl Ext2Ops {
    /// Convert Ext2 elements to flat u64 buffer.
    fn to_flat(elements: &[GoldilocksExt2]) -> Vec<u64> {
        elements.iter().flat_map(|e| [e.c0.0, e.c1.0]).collect()
    }

    /// Convert flat u64 buffer to Ext2 elements.
    fn from_flat(flat: Vec<u64>) -> Vec<GoldilocksExt2> {
        flat.chunks_exact(2)
            .map(|c| GoldilocksExt2 {
                c0: GoldilocksField(c[0]),
                c1: GoldilocksField(c[1]),
            })
            .collect()
    }

    /// Batch addition with automatic memory management.
    pub fn add(a: &[GoldilocksExt2], b: &[GoldilocksExt2]) -> Result<Vec<GoldilocksExt2>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        let a_flat = Self::to_flat(a);
        let b_flat = Self::to_flat(b);

        let d_a = DeviceBuffer::from_slice(&a_flat)?;
        let d_b = DeviceBuffer::from_slice(&b_flat)?;
        let mut d_result = DeviceBuffer::<u64>::new(n * 2)?;

        Ext2Batch::add(&d_a, &d_b, &mut d_result)?;

        let result_flat = d_result.to_vec()?;
        Ok(Self::from_flat(result_flat))
    }

    /// Batch subtraction with automatic memory management.
    pub fn sub(a: &[GoldilocksExt2], b: &[GoldilocksExt2]) -> Result<Vec<GoldilocksExt2>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        let a_flat = Self::to_flat(a);
        let b_flat = Self::to_flat(b);

        let d_a = DeviceBuffer::from_slice(&a_flat)?;
        let d_b = DeviceBuffer::from_slice(&b_flat)?;
        let mut d_result = DeviceBuffer::<u64>::new(n * 2)?;

        Ext2Batch::sub(&d_a, &d_b, &mut d_result)?;

        let result_flat = d_result.to_vec()?;
        Ok(Self::from_flat(result_flat))
    }

    /// Batch multiplication with automatic memory management.
    pub fn mul(a: &[GoldilocksExt2], b: &[GoldilocksExt2]) -> Result<Vec<GoldilocksExt2>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        let a_flat = Self::to_flat(a);
        let b_flat = Self::to_flat(b);

        let d_a = DeviceBuffer::from_slice(&a_flat)?;
        let d_b = DeviceBuffer::from_slice(&b_flat)?;
        let mut d_result = DeviceBuffer::<u64>::new(n * 2)?;

        Ext2Batch::mul(&d_a, &d_b, &mut d_result)?;

        let result_flat = d_result.to_vec()?;
        Ok(Self::from_flat(result_flat))
    }

    /// Batch inversion with automatic memory management.
    pub fn inverse(a: &[GoldilocksExt2]) -> Result<Vec<GoldilocksExt2>> {
        let n = a.len();
        let a_flat = Self::to_flat(a);

        let d_a = DeviceBuffer::from_slice(&a_flat)?;
        let mut d_result = DeviceBuffer::<u64>::new(n * 2)?;

        Ext2Batch::inverse(&d_a, &mut d_result)?;

        let result_flat = d_result.to_vec()?;
        Ok(Self::from_flat(result_flat))
    }

    /// Batch conversion: Goldilocks -> Ext2
    pub fn from_base(base: &[GoldilocksField]) -> Result<Vec<GoldilocksExt2>> {
        let n = base.len();
        let base_u64: Vec<u64> = base.iter().map(|x| x.0).collect();

        let d_input = DeviceBuffer::from_slice(&base_u64)?;
        let mut d_output = DeviceBuffer::<u64>::new(n * 2)?;

        Ext2Batch::from_base(&d_input, &mut d_output)?;

        let result_flat = d_output.to_vec()?;
        Ok(Self::from_flat(result_flat))
    }

    /// Batch conversion: Ext2 -> Goldilocks (extracts c0)
    pub fn to_base(ext: &[GoldilocksExt2]) -> Result<Vec<GoldilocksField>> {
        let n = ext.len();
        let ext_flat = Self::to_flat(ext);

        let d_input = DeviceBuffer::from_slice(&ext_flat)?;
        let mut d_output = DeviceBuffer::<u64>::new(n)?;

        Ext2Batch::to_base(&d_input, &mut d_output)?;

        let result = d_output.to_vec()?;
        Ok(result.into_iter().map(GoldilocksField).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init;

    #[test]
    fn test_ext2_from_base() {
        init().unwrap();

        let base: Vec<GoldilocksField> = (1..101).map(GoldilocksField::new).collect();
        let ext2 = Ext2Ops::from_base(&base).unwrap();

        for (i, e) in ext2.iter().enumerate() {
            assert_eq!(e.c0.0, (i + 1) as u64);
            assert_eq!(e.c1.0, 0);
        }
    }

    #[test]
    fn test_ext2_mul() {
        init().unwrap();

        // (1 + 2X) * (3 + 4X) = 3 + 4X + 6X + 8X^2
        //                    = 3 + 8*7 + (4+6)X = 59 + 10X
        let a = vec![GoldilocksExt2::new(GoldilocksField(1), GoldilocksField(2))];
        let b = vec![GoldilocksExt2::new(GoldilocksField(3), GoldilocksField(4))];

        let result = Ext2Ops::mul(&a, &b).unwrap();

        assert_eq!(result[0].c0.0, 59);
        assert_eq!(result[0].c1.0, 10);
    }
}
