//! GPU memory management utilities.

use crate::error::{CudaError, Result};

/// Reject element counts that would wrap a C `int` parameter.
///
/// Several FFI entry points still take `int n`. At 2^31 elements that wraps
/// negative and, once sign-extended to size_t, addresses ~1.8e19 bytes -- which
/// is how 3D-UNet at 128^3 produced a sticky illegal access with no bad index
/// anywhere. Buffers do reach that size: c_out_pad 64 * s_full_pad 2^25 is
/// exactly 2^31. Fail with a message naming the count instead.
pub fn check_elem_count(n: usize, what: &str) -> Result<()> {
    if n > i32::MAX as usize {
        return Err(CudaError::InvalidArgument(format!(
            "{}: {} elements exceeds INT_MAX ({}); this FFI takes a C int and \
             would wrap negative", what, n, i32::MAX)));
    }
    Ok(())
}
use crate::ffi;
use std::marker::PhantomData;
use std::os::raw::c_void;

/// A buffer of data stored on the GPU device.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl<T> DeviceBuffer<T> {
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            });
        }
        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { ffi::cuda_malloc(&mut ptr, size) };
        if ret != 0 {
            return Err(CudaError::AllocationFailed);
        }
        Ok(Self { ptr: ptr as *mut T, len, _marker: PhantomData })
    }

    pub fn from_slice(data: &[T]) -> Result<Self> {
        let buffer = Self::new(data.len())?;
        if !data.is_empty() {
            let size = data.len() * std::mem::size_of::<T>();
            let ret = unsafe {
                ffi::cuda_memcpy_htod(
                    buffer.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    size,
                )
            };
            if ret != 0 {
                return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret)));
            }
        }
        Ok(buffer)
    }

    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Clone + Default,
    {
        let mut result = vec![T::default(); self.len];
        if !result.is_empty() {
            let size = self.len * std::mem::size_of::<T>();
            let ret = unsafe {
                ffi::cuda_memcpy_dtoh(
                    result.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    size,
                )
            };
            if ret != 0 {
                return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret)));
            }
        }
        Ok(result)
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }

    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(CudaError::InvalidArgument(
                "Slice length does not match buffer length".to_string(),
            ));
        }
        if !data.is_empty() {
            let size = data.len() * std::mem::size_of::<T>();
            let ret = unsafe {
                ffi::cuda_memcpy_htod(
                    self.ptr as *mut c_void,
                    data.as_ptr() as *const c_void,
                    size,
                )
            };
            if ret != 0 {
                return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret)));
            }
        }
        Ok(())
    }

    /// Upload `data` to `self[offset_elems .. offset_elems + data.len()]`.
    /// Unlike [`Self::copy_from_slice`], the buffer may be larger than the
    /// upload — lets pooled (possibly oversized, see SP_POOL best-fit)
    /// buffers receive per-leaf slices directly, skipping the host-side
    /// concat copy.
    pub fn write_slice_at(&mut self, offset_elems: usize, data: &[T]) -> Result<()> {
        if offset_elems + data.len() > self.len {
            return Err(CudaError::InvalidArgument(
                "write_slice_at out of bounds".to_string(),
            ));
        }
        if !data.is_empty() {
            let size = data.len() * std::mem::size_of::<T>();
            let ret = unsafe {
                ffi::cuda_memcpy_htod(
                    self.ptr.add(offset_elems) as *mut c_void,
                    data.as_ptr() as *const c_void,
                    size,
                )
            };
            if ret != 0 {
                return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret)));
            }
        }
        Ok(())
    }

    /// Zero this buffer in-place via `cudaMemset` (single GPU launch,
    /// no PCIe traffic). Use this instead of `copy_from_slice(&zeros)`
    /// to skip the host-side zero-vec alloc + host→device upload.
    pub fn zero(&mut self) -> Result<()> {
        if self.len == 0 { return Ok(()); }
        let size = self.len * std::mem::size_of::<T>();
        let ret = unsafe { ffi::cuda_memset(self.ptr as *mut c_void, 0, size) };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(())
    }

    /// Copy `data` into this buffer starting at element `offset`.
    ///
    /// Lets a caller seed a few scattered entries of a large device buffer
    /// without materializing the whole thing on the host first — the difference
    /// between a multi-gigabyte host allocation and a `cudaMemset` plus a
    /// handful of small copies.
    pub fn copy_from_slice_at(&mut self, offset: usize, data: &[T]) -> Result<()> {
        if data.is_empty() { return Ok(()); }
        if offset + data.len() > self.len {
            return Err(CudaError::InvalidArgument(
                "copy_from_slice_at: range exceeds buffer length".to_string(),
            ));
        }
        let esz = std::mem::size_of::<T>();
        let dst = unsafe { (self.ptr as *mut u8).add(offset * esz) };
        let ret = unsafe {
            ffi::cuda_memcpy_htod(
                dst as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len() * esz,
            )
        };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(())
    }

    /// Read `data.len()` elements starting at element `offset`.
    ///
    /// The counterpart of [`Self::copy_from_slice_at`]: reading a handful of
    /// terminal values out of a multi-gigabyte buffer should not require
    /// downloading the buffer.
    pub fn copy_to_slice_at(&self, offset: usize, data: &mut [T]) -> Result<()> {
        if data.is_empty() { return Ok(()); }
        if offset + data.len() > self.len {
            return Err(CudaError::InvalidArgument(
                "copy_to_slice_at: range exceeds buffer length".to_string(),
            ));
        }
        let esz = std::mem::size_of::<T>();
        let src = unsafe { (self.ptr as *const u8).add(offset * esz) };
        let ret = unsafe {
            ffi::cuda_memcpy_dtoh(
                data.as_mut_ptr() as *mut c_void,
                src as *const c_void,
                data.len() * esz,
            )
        };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(())
    }

    pub fn copy_to_slice(&self, data: &mut [T]) -> Result<()> {
        if data.len() != self.len {
            return Err(CudaError::InvalidArgument(
                "Slice length does not match buffer length".to_string(),
            ));
        }
        if !data.is_empty() {
            let size = data.len() * std::mem::size_of::<T>();
            let ret = unsafe {
                ffi::cuda_memcpy_dtoh(
                    data.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    size,
                )
            };
            if ret != 0 {
                return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret)));
            }
        }
        Ok(())
    }

    pub fn clone_on_device(&self) -> Result<Self> {
        if self.len == 0 { return Self::new(0); }
        let new_buf = Self::new(self.len)?;
        let size = self.len * std::mem::size_of::<T>();
        let ret = unsafe {
            ffi::cuda_memcpy_dtod(
                new_buf.ptr as *mut c_void,
                self.ptr as *const c_void,
                size,
            )
        };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(new_buf)
    }

    pub fn copy_from_device(&mut self, src: &DeviceBuffer<T>) -> Result<()> {
        if src.len != self.len {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match for device-to-device copy".to_string(),
            ));
        }
        if self.len > 0 {
            let size = self.len * std::mem::size_of::<T>();
            let ret = unsafe {
                ffi::cuda_memcpy_dtod(
                    self.ptr as *mut c_void,
                    src.ptr as *const c_void,
                    size,
                )
            };
            if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        }
        Ok(())
    }

    /// Device-to-device ranged copy: `self[dst_off .. dst_off+len] =
    /// src[src_off .. src_off+len]`. Works across devices via unified
    /// addressing (the runtime routes peer-to-peer or stages through
    /// pinned memory — either way no pageable host round-trip).
    pub fn copy_range_from_device(
        &mut self,
        dst_off: usize,
        src: &DeviceBuffer<T>,
        src_off: usize,
        len: usize,
    ) -> Result<()> {
        if dst_off + len > self.len || src_off + len > src.len {
            return Err(CudaError::InvalidArgument(format!(
                "copy_range_from_device out of bounds: dst {}+{}/{} src {}+{}/{}",
                dst_off, len, self.len, src_off, len, src.len
            )));
        }
        if len == 0 { return Ok(()); }
        let size = len * std::mem::size_of::<T>();
        let ret = unsafe {
            ffi::cuda_memcpy_dtod(
                self.ptr.add(dst_off) as *mut c_void,
                src.ptr.add(src_off) as *const c_void,
                size,
            )
        };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(())
    }

    /// Cross-device ranged copy via `cudaMemcpyPeer` (works with or
    /// without P2P; stages through pinned host memory if needed). The
    /// caller supplies both device ids — `DeviceBuffer` doesn't track
    /// its owning device.
    pub fn copy_range_from_device_peer(
        &mut self,
        dst_off: usize,
        dst_dev: i32,
        src: &DeviceBuffer<T>,
        src_off: usize,
        src_dev: i32,
        len: usize,
    ) -> Result<()> {
        if dst_off + len > self.len || src_off + len > src.len {
            return Err(CudaError::InvalidArgument(format!(
                "copy_range_from_device_peer out of bounds: dst {}+{}/{} src {}+{}/{}",
                dst_off, len, self.len, src_off, len, src.len
            )));
        }
        if len == 0 { return Ok(()); }
        let size = len * std::mem::size_of::<T>();
        let ret = unsafe {
            ffi::cuda_memcpy_peer(
                self.ptr.add(dst_off) as *mut c_void,
                dst_dev,
                src.ptr.add(src_off) as *const c_void,
                src_dev,
                size,
            )
        };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(())
    }

    /// # Safety
    /// Caller must ensure `n <= self.len`.
    pub unsafe fn offset_ptr(&self, n: usize) -> *const T { self.ptr.add(n) }

    /// # Safety
    /// Caller must ensure `n <= self.len`.
    pub unsafe fn offset_mut_ptr(&mut self, n: usize) -> *mut T { self.ptr.add(n) }

    pub fn read_slice(&self, offset: usize, len: usize) -> Result<Vec<T>>
    where
        T: Clone + Default,
    {
        if offset + len > self.len {
            return Err(CudaError::InvalidArgument(format!(
                "read_slice out of bounds: offset={} len={} buf_len={}",
                offset, len, self.len
            )));
        }
        if len == 0 { return Ok(Vec::new()); }
        let mut result = vec![T::default(); len];
        let byte_size = len * std::mem::size_of::<T>();
        let ret = unsafe {
            ffi::cuda_memcpy_dtoh(
                result.as_mut_ptr() as *mut c_void,
                self.ptr.add(offset) as *const c_void,
                byte_size,
            )
        };
        if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
        Ok(result)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::cuda_free(self.ptr as *mut c_void); }
        }
    }
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

/// Trim the default CUDA memory pool for the current device, releasing
/// pool-cached blocks back to the OS. Useful between iterations in
/// long-running streams that allocate/free repeatedly — the default
/// caching allocator retains freed blocks in pool for reuse, which can
/// accumulate to tens of GB on large workloads (observed on Llama-2
/// streaming bench at N≥10). Pass 0 to free everything possible.
pub fn pool_trim(min_bytes_to_keep: usize) -> Result<()> {
    let ret = unsafe { ffi::cuda_pool_trim(min_bytes_to_keep) };
    if ret != 0 { return Err(CudaError::SyncFailed); }
    Ok(())
}

pub fn synchronize() -> Result<()> {
    let ret = unsafe { ffi::cuda_device_synchronize() };
    if ret != 0 { return Err(CudaError::SyncFailed); }
    Ok(())
}

pub fn get_last_error() -> i32 { unsafe { ffi::cuda_get_last_error() } }
pub fn peek_at_last_error() -> i32 { unsafe { ffi::cuda_peek_at_last_error() } }

pub fn mem_get_info() -> Result<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let ret = unsafe { ffi::cuda_mem_get_info(&mut free, &mut total) };
    if ret != 0 { return Err(CudaError::SyncFailed); }
    Ok((free, total))
}

/// # Safety
/// Caller must ensure dst and src point to valid GPU memory of at least `size` bytes.
pub unsafe fn memcpy_dtod(dst: *mut c_void, src: *const c_void, size: usize) -> Result<()> {
    let ret = ffi::cuda_memcpy_dtod(dst, src, size);
    if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
    Ok(())
}

/// CUDA stream — RAII wrapper around `cudaStream_t`. Drop destroys the stream.
/// Bind to a specific device by calling [`crate::set_device`] before construction
/// (CUDA streams are per-device).
pub struct CudaStream {
    handle: *mut c_void,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        let ret = unsafe { ffi::cuda_stream_create(&mut handle as *mut *mut c_void) };
        if ret != 0 { return Err(CudaError::InitializationFailed); }
        Ok(Self { handle })
    }
    pub fn as_ptr(&self) -> *mut c_void { self.handle }
    pub fn synchronize(&self) -> Result<()> {
        let ret = unsafe { ffi::cuda_stream_synchronize(self.handle) };
        if ret != 0 { return Err(CudaError::SyncFailed); }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { let _ = ffi::cuda_stream_destroy(self.handle); }
    }
}

/// # Safety
/// Caller must ensure dst and src point to valid GPU memory and the stream is alive.
pub unsafe fn memcpy_dtod_async(
    dst: *mut c_void, src: *const c_void, size: usize, stream: &CudaStream,
) -> Result<()> {
    let ret = ffi::cuda_memcpy_dtod_async(dst, src, size, stream.as_ptr());
    if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
    Ok(())
}

/// # Safety
/// Caller must ensure dst points to valid GPU memory and src is a valid host pointer.
pub unsafe fn memcpy_htod_async(
    dst: *mut c_void, src: *const c_void, size: usize, stream: &CudaStream,
) -> Result<()> {
    let ret = ffi::cuda_memcpy_htod_async(dst, src, size, stream.as_ptr());
    if ret != 0 { return Err(CudaError::MemcpyFailed(crate::cuda_error_string(ret))); }
    Ok(())
}
