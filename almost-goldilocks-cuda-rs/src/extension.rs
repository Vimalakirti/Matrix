//! Almost-Goldilocks Ext2 = F_p[X] / (X^2 - 3).
//!
//! Non-residue `W = 3` is used because `Legendre(3) = -1` mod `P`;
//! the Goldilocks choice `W = 7` is reducible here (`Legendre(7) = +1`).

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::field::AlmostGoldilocksField;
use crate::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};
use std::os::raw::c_int;

/// Quadratic non-residue. Ext2 = `F_p[X] / (X^2 - W)`.
pub const AEXT2_W: u64 = 3;

/// `W^((P-1)/2) = -1 mod P`. Used by the Frobenius automorphism.
pub const AEXT2_DTH_ROOT: u64 = 0xFFFFFFFEFFFFFFE0;

/// Quadratic-extension element `c0 + c1 * X` with `X^2 = 3`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct AlmostGoldilocksExt2 {
    pub c0: AlmostGoldilocksField,
    pub c1: AlmostGoldilocksField,
}

impl AlmostGoldilocksExt2 {
    pub const fn new(c0: AlmostGoldilocksField, c1: AlmostGoldilocksField) -> Self {
        Self { c0, c1 }
    }
    pub const fn zero() -> Self {
        Self { c0: AlmostGoldilocksField::zero(), c1: AlmostGoldilocksField::zero() }
    }
    pub const fn one() -> Self {
        Self { c0: AlmostGoldilocksField::one(), c1: AlmostGoldilocksField::zero() }
    }
    pub const fn from_base(base: AlmostGoldilocksField) -> Self {
        Self { c0: base, c1: AlmostGoldilocksField::zero() }
    }
    pub fn is_base(&self) -> bool { self.c1.0 == 0 }
    pub fn to_base(&self) -> AlmostGoldilocksField { self.c0 }
    pub fn to_raw(&self) -> [u64; 2] { [self.c0.0, self.c1.0] }
    pub fn from_raw(raw: [u64; 2]) -> Self {
        Self { c0: AlmostGoldilocksField(raw[0]), c1: AlmostGoldilocksField(raw[1]) }
    }
}

impl From<AlmostGoldilocksField> for AlmostGoldilocksExt2 {
    fn from(base: AlmostGoldilocksField) -> Self { Self::from_base(base) }
}
impl From<u64> for AlmostGoldilocksExt2 {
    fn from(value: u64) -> Self { Self::from_base(AlmostGoldilocksField(value)) }
}

impl std::ops::Add for AlmostGoldilocksExt2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { c0: self.c0 + rhs.c0, c1: self.c1 + rhs.c1 }
    }
}
impl std::ops::Sub for AlmostGoldilocksExt2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { c0: self.c0 - rhs.c0, c1: self.c1 - rhs.c1 }
    }
}
impl std::ops::Mul for AlmostGoldilocksExt2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Karatsuba: 3 base muls + cheap multiply-by-3.
        let m0 = self.c0 * rhs.c0;
        let m1 = self.c1 * rhs.c1;
        let m2 = (self.c0 + self.c1) * (rhs.c0 + rhs.c1);
        Self {
            c0: m0 + AlmostGoldilocksField(AEXT2_W) * m1,
            c1: m2 - m0 - m1,
        }
    }
}

// ============================================================================
// Batch ops (low-level)
// ============================================================================

pub struct AlmostExt2Batch;

impl AlmostExt2Batch {
    pub fn add(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if b.len() != a.len() || result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_add_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn sub(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if b.len() != a.len() || result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_sub_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn mul(a: &DeviceBuffer<u64>, b: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if b.len() != a.len() || result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_mul_ffi(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn inverse(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_inverse_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn from_base(input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = input.len();
        if output.len() != n * 2 {
            return Err(CudaError::InvalidArgument("Output buffer must be 2x input length".to_string()));
        }
        let ret = unsafe { ffi::agl_to_aext2_batch_ffi(input.as_ptr(), output.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn to_base(input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = input.len() / 2;
        if output.len() != n {
            return Err(CudaError::InvalidArgument("Output buffer must be half input length".to_string()));
        }
        let ret = unsafe { ffi::aext2_to_agl_batch_ffi(input.as_ptr(), output.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn neg(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_neg_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn square(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_square_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn frobenius(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_frobenius_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn conjugate(a: &DeviceBuffer<u64>, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_conjugate_ffi(a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn exp(a: &DeviceBuffer<u64>, exp: u64, result: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_exp_ffi(a.as_ptr(), exp, result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    pub fn mul_scalar(
        scalar: AlmostGoldilocksField,
        a: &DeviceBuffer<u64>,
        result: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        let n = a.len() / 2;
        if result.len() != a.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe { ffi::aext2_batch_mul_scalar_ffi(scalar.0, a.as_ptr(), result.as_mut_ptr(), n as c_int) };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    /// `acc[i] += scalar * src[i]`, all Ext2 (interleaved layout).
    pub fn scale_accumulate(
        scalar: AlmostGoldilocksExt2,
        src: &DeviceBuffer<u64>,
        acc: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        let n = src.len() / 2;
        if acc.len() != src.len() {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe {
            ffi::aext2_scale_accumulate_ffi(
                scalar.c0.0, scalar.c1.0,
                src.as_ptr(), acc.as_mut_ptr(), n as c_int,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }

    /// As [`Self::scale_accumulate`] but `src` is an `Ext2` device buffer
    /// (e.g. an eq table straight from `ext2_eq_dp_all_device`). `Ext2` is
    /// `#[repr(C)]` of two `u64`s, so this reinterprets the same bytes —
    /// no copy, no u64-typed scratch alloc. `acc` is `n` Ext2 = `2n` u64.
    pub fn scale_accumulate_from_ext2(
        scalar: AlmostGoldilocksExt2,
        src: &DeviceBuffer<AlmostGoldilocksExt2>,
        acc: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        let n = src.len();
        if acc.len() != n * 2 {
            return Err(CudaError::InvalidArgument("Buffer lengths must match".to_string()));
        }
        let ret = unsafe {
            ffi::aext2_scale_accumulate_ffi(
                scalar.c0.0, scalar.c1.0,
                src.as_ptr() as *const u64, acc.as_mut_ptr(), n as c_int,
            )
        };
        if ret != 0 { return Err(CudaError::KernelFailed); }
        Ok(())
    }
}

// ============================================================================
// High-level convenience wrappers
// ============================================================================

pub struct AlmostExt2Ops;

impl AlmostExt2Ops {
    fn to_flat(e: &[AlmostGoldilocksExt2]) -> Vec<u64> {
        e.iter().flat_map(|x| [x.c0.0, x.c1.0]).collect()
    }
    fn from_flat(flat: Vec<u64>) -> Vec<AlmostGoldilocksExt2> {
        flat.chunks_exact(2)
            .map(|c| AlmostGoldilocksExt2 { c0: AlmostGoldilocksField(c[0]), c1: AlmostGoldilocksField(c[1]) })
            .collect()
    }

    pub fn add(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> Result<Vec<AlmostGoldilocksExt2>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument("Input lengths must match".to_string()));
        }
        let d_a = DeviceBuffer::from_slice(&Self::to_flat(a))?;
        let d_b = DeviceBuffer::from_slice(&Self::to_flat(b))?;
        let mut d_r = DeviceBuffer::<u64>::new(n * 2)?;
        AlmostExt2Batch::add(&d_a, &d_b, &mut d_r)?;
        Ok(Self::from_flat(d_r.to_vec()?))
    }

    pub fn sub(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> Result<Vec<AlmostGoldilocksExt2>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument("Input lengths must match".to_string()));
        }
        let d_a = DeviceBuffer::from_slice(&Self::to_flat(a))?;
        let d_b = DeviceBuffer::from_slice(&Self::to_flat(b))?;
        let mut d_r = DeviceBuffer::<u64>::new(n * 2)?;
        AlmostExt2Batch::sub(&d_a, &d_b, &mut d_r)?;
        Ok(Self::from_flat(d_r.to_vec()?))
    }

    pub fn mul(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> Result<Vec<AlmostGoldilocksExt2>> {
        let n = a.len();
        if b.len() != n {
            return Err(CudaError::InvalidArgument("Input lengths must match".to_string()));
        }
        let d_a = DeviceBuffer::from_slice(&Self::to_flat(a))?;
        let d_b = DeviceBuffer::from_slice(&Self::to_flat(b))?;
        let mut d_r = DeviceBuffer::<u64>::new(n * 2)?;
        AlmostExt2Batch::mul(&d_a, &d_b, &mut d_r)?;
        Ok(Self::from_flat(d_r.to_vec()?))
    }

    pub fn inverse(a: &[AlmostGoldilocksExt2]) -> Result<Vec<AlmostGoldilocksExt2>> {
        let n = a.len();
        let d_a = DeviceBuffer::from_slice(&Self::to_flat(a))?;
        let mut d_r = DeviceBuffer::<u64>::new(n * 2)?;
        AlmostExt2Batch::inverse(&d_a, &mut d_r)?;
        Ok(Self::from_flat(d_r.to_vec()?))
    }

    pub fn from_base(base: &[AlmostGoldilocksField]) -> Result<Vec<AlmostGoldilocksExt2>> {
        let n = base.len();
        let base_u64: Vec<u64> = base.iter().map(|x| x.0).collect();
        let d_input = DeviceBuffer::from_slice(&base_u64)?;
        let mut d_output = DeviceBuffer::<u64>::new(n * 2)?;
        AlmostExt2Batch::from_base(&d_input, &mut d_output)?;
        Ok(Self::from_flat(d_output.to_vec()?))
    }

    pub fn to_base(ext: &[AlmostGoldilocksExt2]) -> Result<Vec<AlmostGoldilocksField>> {
        let n = ext.len();
        let d_input = DeviceBuffer::from_slice(&Self::to_flat(ext))?;
        let mut d_output = DeviceBuffer::<u64>::new(n)?;
        AlmostExt2Batch::to_base(&d_input, &mut d_output)?;
        Ok(d_output.to_vec()?.into_iter().map(AlmostGoldilocksField).collect())
    }
}
