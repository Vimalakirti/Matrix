//! Goldilocks field operations on GPU.

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};
use std::os::raw::c_int;

/// Goldilocks prime: 2^64 - 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

/// A Goldilocks field element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct GoldilocksField(pub u64);

impl GoldilocksField {
    /// Create a new field element.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Zero element.
    pub const fn zero() -> Self {
        Self(0)
    }

    /// One element.
    pub const fn one() -> Self {
        Self(1)
    }

    /// Reduce value modulo the Goldilocks prime (on CPU).
    pub fn reduce(self) -> Self {
        let mut v = self.0;
        if v >= GOLDILOCKS_PRIME {
            v -= GOLDILOCKS_PRIME;
        }
        Self(v)
    }
}

impl From<u64> for GoldilocksField {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<GoldilocksField> for u64 {
    fn from(field: GoldilocksField) -> Self {
        field.0
    }
}

impl std::ops::Add for GoldilocksField {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (sum, carry) = self.0.overflowing_add(rhs.0);
        let (r, borrow) = sum.overflowing_sub(GOLDILOCKS_PRIME);
        // If carry || !borrow, then sum >= p, so use r; else use sum
        if carry || !borrow {
            Self(r)
        } else {
            Self(sum)
        }
    }
}

impl std::ops::Sub for GoldilocksField {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        if borrow {
            Self(diff.wrapping_add(GOLDILOCKS_PRIME))
        } else {
            Self(diff)
        }
    }
}

impl std::ops::Mul for GoldilocksField {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let full = (self.0 as u128) * (rhs.0 as u128);
        let lo = full as u64;
        let hi = (full >> 64) as u64;
        // Reduce: full mod p where p = 2^64 - 2^32 + 1
        // full = hi * 2^64 + lo
        // 2^64 ≡ 2^32 - 1 (mod p)
        // So full ≡ lo + hi * (2^32 - 1) (mod p)
        let hi_shift = (hi as u128) * ((1u128 << 32) - 1);
        let r = (lo as u128) + hi_shift;
        // r fits in ~97 bits, reduce again
        let lo2 = r as u64;
        let hi2 = (r >> 64) as u64;
        let hi2_shift = hi2 as u128 * ((1u128 << 32) - 1);
        let r2 = lo2 as u128 + hi2_shift;
        let mut result = (r2 % (GOLDILOCKS_PRIME as u128)) as u64;
        if result >= GOLDILOCKS_PRIME {
            result -= GOLDILOCKS_PRIME;
        }
        Self(result)
    }
}

/// Batch operations on Goldilocks field elements.
pub struct GoldilocksBatch;

impl GoldilocksBatch {
    /// Batch addition: result[i] = a[i] + b[i]
    pub fn add(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::gl_batch_add(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch subtraction: result[i] = a[i] - b[i]
    pub fn sub(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::gl_batch_sub(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch multiplication: result[i] = a[i] * b[i]
    pub fn mul(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::gl_batch_mul(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch inversion: result[i] = a[i]^(-1)
    pub fn inverse(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe { ffi::gl_batch_inverse(a.as_ptr(), result.as_mut_ptr(), n as c_int) };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Bit permutation of polynomial evaluations on GPU.
    /// perm_map[old_var] = new_var position. Both input and output are on device.
    /// Total elements = 2^n_bits.
    /// Uses gather pattern (good write coalescing) with inverse permutation.
    pub fn bit_permute(
        input: &DeviceBuffer<u64>,
        output: &mut DeviceBuffer<u64>,
        perm_map: &[i32],
    ) -> Result<()> {
        let n_bits = perm_map.len();
        let total = 1usize << n_bits;
        if input.len() != total || output.len() != total {
            return Err(CudaError::InvalidArgument(
                format!("Buffer length must be 2^{} = {}", n_bits, total),
            ));
        }

        // Compute inverse permutation: inv_perm[new_pos] = old_var
        let mut inv_perm = vec![0i32; n_bits];
        for (old_var, &new_pos) in perm_map.iter().enumerate() {
            inv_perm[new_pos as usize] = old_var as i32;
        }

        let d_perm = DeviceBuffer::<i32>::from_slice(&inv_perm)?;
        let ret = unsafe {
            ffi::bit_permute_gl_ffi(
                input.as_ptr(),
                output.as_mut_ptr(),
                d_perm.as_ptr() as *const c_int,
                n_bits as c_int,
                total as c_int,
            )
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }
}

/// High-level batch operations that handle memory transfers.
pub struct GoldilocksOps;

impl GoldilocksOps {
    /// Batch addition with automatic memory management.
    pub fn add(a: &[GoldilocksField], b: &[GoldilocksField]) -> Result<Vec<GoldilocksField>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        // Convert to u64 slices
        let a_u64: Vec<u64> = a.iter().map(|x| x.0).collect();
        let b_u64: Vec<u64> = b.iter().map(|x| x.0).collect();

        // Allocate device buffers
        let d_a = DeviceBuffer::from_slice(&a_u64)?;
        let d_b = DeviceBuffer::from_slice(&b_u64)?;
        let mut d_result = DeviceBuffer::<u64>::new(n)?;

        // Perform operation
        GoldilocksBatch::add(&d_a, &d_b, &mut d_result)?;

        // Copy result back
        let result_u64 = d_result.to_vec()?;
        Ok(result_u64.into_iter().map(GoldilocksField).collect())
    }

    /// Batch subtraction with automatic memory management.
    pub fn sub(a: &[GoldilocksField], b: &[GoldilocksField]) -> Result<Vec<GoldilocksField>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        let a_u64: Vec<u64> = a.iter().map(|x| x.0).collect();
        let b_u64: Vec<u64> = b.iter().map(|x| x.0).collect();

        let d_a = DeviceBuffer::from_slice(&a_u64)?;
        let d_b = DeviceBuffer::from_slice(&b_u64)?;
        let mut d_result = DeviceBuffer::<u64>::new(n)?;

        GoldilocksBatch::sub(&d_a, &d_b, &mut d_result)?;

        let result_u64 = d_result.to_vec()?;
        Ok(result_u64.into_iter().map(GoldilocksField).collect())
    }

    /// Batch multiplication with automatic memory management.
    pub fn mul(a: &[GoldilocksField], b: &[GoldilocksField]) -> Result<Vec<GoldilocksField>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        let a_u64: Vec<u64> = a.iter().map(|x| x.0).collect();
        let b_u64: Vec<u64> = b.iter().map(|x| x.0).collect();

        let d_a = DeviceBuffer::from_slice(&a_u64)?;
        let d_b = DeviceBuffer::from_slice(&b_u64)?;
        let mut d_result = DeviceBuffer::<u64>::new(n)?;

        GoldilocksBatch::mul(&d_a, &d_b, &mut d_result)?;

        let result_u64 = d_result.to_vec()?;
        Ok(result_u64.into_iter().map(GoldilocksField).collect())
    }

    /// Batch inversion with automatic memory management.
    pub fn inverse(a: &[GoldilocksField]) -> Result<Vec<GoldilocksField>> {
        let n = a.len();
        let a_u64: Vec<u64> = a.iter().map(|x| x.0).collect();

        let d_a = DeviceBuffer::from_slice(&a_u64)?;
        let mut d_result = DeviceBuffer::<u64>::new(n)?;

        GoldilocksBatch::inverse(&d_a, &mut d_result)?;

        let result_u64 = d_result.to_vec()?;
        Ok(result_u64.into_iter().map(GoldilocksField).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init;

    #[test]
    fn test_batch_add() {
        init().unwrap();

        let a: Vec<GoldilocksField> = (0..1000).map(|i| GoldilocksField::new(i)).collect();
        let b: Vec<GoldilocksField> = (0..1000).map(|i| GoldilocksField::new(i * 2)).collect();

        let result = GoldilocksOps::add(&a, &b).unwrap();

        for i in 0..1000 {
            let expected = (i + i * 2) as u64;
            assert_eq!(result[i].0, expected, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_batch_mul() {
        init().unwrap();

        let a: Vec<GoldilocksField> = (1..101).map(|i| GoldilocksField::new(i)).collect();
        let b: Vec<GoldilocksField> = (1..101).map(|i| GoldilocksField::new(i)).collect();

        let result = GoldilocksOps::mul(&a, &b).unwrap();

        for i in 0..100 {
            let expected = ((i + 1) * (i + 1)) as u64;
            assert_eq!(result[i].0, expected, "Mismatch at index {}", i);
        }
    }
}
