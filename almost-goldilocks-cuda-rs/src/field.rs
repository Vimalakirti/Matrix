//! Almost-Goldilocks base field operations.
//!
//! Prime: `P = 2^64 - 2^32 + 1 - 32 = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1`.

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};
use std::os::raw::c_int;

/// The almost-Goldilocks prime: `2^64 - 2^32 - 31`.
pub const ALMOST_GOLDILOCKS_PRIME: u64 = 0xFFFFFFFEFFFFFFE1;

/// Wrap constant `c = 2^64 mod P = 2^32 + 31` used by the Solinas reduction.
pub const ALMOST_REDUCE_C: u64 = 0x10000001F;

/// `(P + 1) / 2 = 2^(-1) mod P`. Used as the inverse-of-2 constant.
pub const ALMOST_HALF_P_PLUS_ONE: u64 = 0x7FFFFFFF7FFFFFF1;

/// An almost-Goldilocks field element.
///
/// The wrapped `u64` may hold a non-canonical representative in `[0, 2^64)`;
/// equality and the host arithmetic helpers all normalize to canonical form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct AlmostGoldilocksField(pub u64);

impl AlmostGoldilocksField {
    pub const fn new(value: u64) -> Self { Self(value) }
    pub const fn zero() -> Self { Self(0) }
    pub const fn one() -> Self { Self(1) }

    /// Reduce to canonical form in `[0, P)`. Cheap: a single conditional sub.
    pub fn reduce(self) -> Self {
        let mut v = self.0;
        if v >= ALMOST_GOLDILOCKS_PRIME {
            v -= ALMOST_GOLDILOCKS_PRIME;
        }
        Self(v)
    }
}

impl From<u64> for AlmostGoldilocksField {
    fn from(value: u64) -> Self { Self(value) }
}
impl From<AlmostGoldilocksField> for u64 {
    fn from(field: AlmostGoldilocksField) -> Self { field.0 }
}

// ============================================================================
// Host arithmetic
//
// These implementations canonicalize inputs first (so they accept the non-
// canonical representations produced by GPU kernels) and produce canonical
// outputs. They use __uint128_t-equivalent paths (u128) for multiplication.
// ============================================================================

#[inline]
fn canon(v: u64) -> u64 {
    if v >= ALMOST_GOLDILOCKS_PRIME {
        v - ALMOST_GOLDILOCKS_PRIME
    } else {
        v
    }
}

impl std::ops::Add for AlmostGoldilocksField {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let a = canon(self.0);
        let b = canon(rhs.0);
        let (sum, carry) = a.overflowing_add(b);
        if carry || sum >= ALMOST_GOLDILOCKS_PRIME {
            Self(sum.wrapping_sub(ALMOST_GOLDILOCKS_PRIME))
        } else {
            Self(sum)
        }
    }
}

impl std::ops::Sub for AlmostGoldilocksField {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let a = canon(self.0);
        let b = canon(rhs.0);
        if a >= b {
            Self(a - b)
        } else {
            Self(ALMOST_GOLDILOCKS_PRIME - (b - a))
        }
    }
}

impl std::ops::Mul for AlmostGoldilocksField {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = canon(self.0);
        let b = canon(rhs.0);
        let prod = (a as u128) * (b as u128);
        Self((prod % (ALMOST_GOLDILOCKS_PRIME as u128)) as u64)
    }
}

impl std::ops::Neg for AlmostGoldilocksField {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let v = canon(self.0);
        if v == 0 { Self(0) } else { Self(ALMOST_GOLDILOCKS_PRIME - v) }
    }
}

// ============================================================================
// Low-level batch ops on device buffers
// ============================================================================

pub struct AlmostGoldilocksBatch;

impl AlmostGoldilocksBatch {
    pub fn add(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_add_ffi")?;
        let ret = unsafe { ffi::agl_batch_add_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn sub(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_sub_ffi")?;
        let ret = unsafe { ffi::agl_batch_sub_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn mul(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_mul_ffi")?;
        let ret = unsafe { ffi::agl_batch_mul_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn neg(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_neg_ffi")?;
        let ret = unsafe { ffi::agl_batch_neg_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn square(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_square_ffi")?;
        let ret = unsafe { ffi::agl_batch_square_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn double(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_double_ffi")?;
        let ret = unsafe { ffi::agl_batch_double_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn inverse(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_inverse_ffi")?;
        let ret = unsafe { ffi::agl_batch_inverse_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn exp(a: &DeviceBuffer<u64>, exp: u64, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_exp_ffi")?;
        let ret = unsafe { ffi::agl_batch_exp_ffi(a.as_ptr(), exp, result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn mul_scalar(scalar: AlmostGoldilocksField, a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_mul_scalar_ffi")?;
        let ret = unsafe { ffi::agl_batch_mul_scalar_ffi(scalar.0, a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn div(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len();
        if b.len() != n || result.len() != n {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        crate::memory::check_elem_count(n, "agl_batch_div_ffi")?;
        let ret = unsafe { ffi::agl_batch_div_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    /// Bit permutation of polynomial evaluations on GPU.
    /// `perm_map[old_var] = new_var position`. 2^n_bits total elements.
    pub fn bit_permute(
        input: &DeviceBuffer<u64>,
        output: &mut DeviceBuffer<u64>,
        perm_map: &[i32],
    ) -> Result<()> {
        let n_bits = perm_map.len();
        let total = 1usize << n_bits;
        if input.len() != total || output.len() != total {
            return Err(CudaError::InvalidArgument(format!(
                "Buffer length must be 2^{} = {}", n_bits, total
            )));
        }
        let mut inv_perm = vec![0i32; n_bits];
        for (old_var, &new_pos) in perm_map.iter().enumerate() {
            inv_perm[new_pos as usize] = old_var as i32;
        }
        let d_perm = DeviceBuffer::<i32>::from_slice(&inv_perm)?;
        let ret = unsafe {
            ffi::agl_bit_permute_ffi(
                input.as_ptr(),
                output.as_mut_ptr(),
                d_perm.as_ptr() as *const c_int,
                n_bits as c_int,
                total as c_int,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }
}

// ============================================================================
// High-level convenience wrappers
// ============================================================================

pub struct AlmostGoldilocksOps;

impl AlmostGoldilocksOps {
    fn upload_pair(a: &[AlmostGoldilocksField], b: &[AlmostGoldilocksField]) -> Result<(DeviceBuffer<u64>, DeviceBuffer<u64>)> {
        let av: Vec<u64> = a.iter().map(|x| x.0).collect();
        let bv: Vec<u64> = b.iter().map(|x| x.0).collect();
        Ok((DeviceBuffer::from_slice(&av)?, DeviceBuffer::from_slice(&bv)?))
    }

    pub fn add(a: &[AlmostGoldilocksField], b: &[AlmostGoldilocksField]) -> Result<Vec<AlmostGoldilocksField>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument("Input lengths must match".to_string()));
        }
        let (d_a, d_b) = Self::upload_pair(a, b)?;
        let mut d_r = DeviceBuffer::<u64>::new(n)?;
        AlmostGoldilocksBatch::add(&d_a, &d_b, &mut d_r)?;
        Ok(d_r.to_vec()?.into_iter().map(AlmostGoldilocksField).collect())
    }

    pub fn sub(a: &[AlmostGoldilocksField], b: &[AlmostGoldilocksField]) -> Result<Vec<AlmostGoldilocksField>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument("Input lengths must match".to_string()));
        }
        let (d_a, d_b) = Self::upload_pair(a, b)?;
        let mut d_r = DeviceBuffer::<u64>::new(n)?;
        AlmostGoldilocksBatch::sub(&d_a, &d_b, &mut d_r)?;
        Ok(d_r.to_vec()?.into_iter().map(AlmostGoldilocksField).collect())
    }

    pub fn mul(a: &[AlmostGoldilocksField], b: &[AlmostGoldilocksField]) -> Result<Vec<AlmostGoldilocksField>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument("Input lengths must match".to_string()));
        }
        let (d_a, d_b) = Self::upload_pair(a, b)?;
        let mut d_r = DeviceBuffer::<u64>::new(n)?;
        AlmostGoldilocksBatch::mul(&d_a, &d_b, &mut d_r)?;
        Ok(d_r.to_vec()?.into_iter().map(AlmostGoldilocksField).collect())
    }

    pub fn inverse(a: &[AlmostGoldilocksField]) -> Result<Vec<AlmostGoldilocksField>> {
        let n = a.len();
        let av: Vec<u64> = a.iter().map(|x| x.0).collect();
        let d_a = DeviceBuffer::from_slice(&av)?;
        let mut d_r = DeviceBuffer::<u64>::new(n)?;
        AlmostGoldilocksBatch::inverse(&d_a, &mut d_r)?;
        Ok(d_r.to_vec()?.into_iter().map(AlmostGoldilocksField).collect())
    }
}
