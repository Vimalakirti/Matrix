//! GPU memory management utilities.

use crate::error::{CudaError, Result};
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
    /// Allocate a new device buffer with the given number of elements.
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

        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Create a device buffer from a host slice.
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
                return Err(CudaError::MemcpyFailed);
            }
        }

        Ok(buffer)
    }

    /// Copy device buffer contents to a host vector.
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
                return Err(CudaError::MemcpyFailed);
            }
        }

        Ok(result)
    }

    /// Get the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a raw pointer to the device memory.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get a mutable raw pointer to the device memory.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copy data from a host slice to this device buffer.
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
                return Err(CudaError::MemcpyFailed);
            }
        }

        Ok(())
    }

    /// Copy data from this device buffer to a host slice.
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
                return Err(CudaError::MemcpyFailed);
            }
        }

        Ok(())
    }

    /// Clone this buffer on device (device-to-device copy, no host involvement).
    pub fn clone_on_device(&self) -> Result<Self> {
        if self.len == 0 {
            return Self::new(0);
        }
        let new_buf = Self::new(self.len)?;
        let size = self.len * std::mem::size_of::<T>();
        let ret = unsafe {
            ffi::cuda_memcpy_dtod(
                new_buf.ptr as *mut c_void,
                self.ptr as *const c_void,
                size,
            )
        };
        if ret != 0 {
            return Err(CudaError::MemcpyFailed);
        }
        Ok(new_buf)
    }

    /// Copy contents from another device buffer (device-to-device, same size).
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
            if ret != 0 {
                return Err(CudaError::MemcpyFailed);
            }
        }
        Ok(())
    }

    /// Get a raw pointer offset by `n` elements (for accessing sub-regions).
    ///
    /// # Safety
    /// Caller must ensure `n <= self.len`.
    pub unsafe fn offset_ptr(&self, n: usize) -> *const T {
        self.ptr.add(n)
    }

    /// Get a mutable raw pointer offset by `n` elements.
    ///
    /// # Safety
    /// Caller must ensure `n <= self.len`.
    pub unsafe fn offset_mut_ptr(&mut self, n: usize) -> *mut T {
        self.ptr.add(n)
    }

    /// Read a contiguous slice of `len` elements starting at `offset` from device.
    /// Copies only the requested range (not the entire buffer).
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
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut result = vec![T::default(); len];
        let byte_size = len * std::mem::size_of::<T>();
        let ret = unsafe {
            ffi::cuda_memcpy_dtoh(
                result.as_mut_ptr() as *mut c_void,
                self.ptr.add(offset) as *const c_void,
                byte_size,
            )
        };
        if ret != 0 {
            return Err(CudaError::MemcpyFailed);
        }
        Ok(result)
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::cuda_free(self.ptr as *mut c_void);
            }
        }
    }
}

// DeviceBuffer is Send and Sync because CUDA operations are thread-safe
// when properly synchronized
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

/// Synchronize the CUDA device (wait for all operations to complete).
pub fn synchronize() -> Result<()> {
    let ret = unsafe { ffi::cuda_device_synchronize() };
    if ret != 0 {
        return Err(CudaError::SyncFailed);
    }
    Ok(())
}

/// Get the last CUDA error code and clear it.
/// Returns 0 (cudaSuccess) if no error, or a CUDA error code.
pub fn get_last_error() -> i32 {
    unsafe { ffi::cuda_get_last_error() }
}

/// Peek at the last CUDA error without clearing it.
pub fn peek_at_last_error() -> i32 {
    unsafe { ffi::cuda_peek_at_last_error() }
}

/// Get free and total GPU memory in bytes.
pub fn mem_get_info() -> Result<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let ret = unsafe { ffi::cuda_mem_get_info(&mut free, &mut total) };
    if ret != 0 {
        return Err(CudaError::SyncFailed);
    }
    Ok((free, total))
}

/// Device-to-device memcpy (raw bytes). Public wrapper for FFI.
///
/// # Safety
/// Caller must ensure dst and src point to valid GPU memory of at least `size` bytes.
pub unsafe fn memcpy_dtod(dst: *mut c_void, src: *const c_void, size: usize) -> Result<()> {
    let ret = ffi::cuda_memcpy_dtod(dst, src, size);
    if ret != 0 {
        return Err(CudaError::MemcpyFailed);
    }
    Ok(())
}
