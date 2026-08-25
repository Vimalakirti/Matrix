//! GPU base-field Conv2D + FlattenKernel kernels for ResNet/VGG witness gen.
//! Direct-conv (no im2col): one thread per output element. Output buffer is
//! sized for the next-pow-2-padded shape and must be pre-zeroed (the kernel
//! writes only the valid region).

use std::os::raw::c_int;

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;

#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &DeviceBuffer<u64>,
    w_flat: &DeviceBuffer<u64>,
    y: &mut DeviceBuffer<u64>,
    c_out: usize, h_out: usize, w_out: usize,
    c_in: usize, kernel_h: usize, kernel_w: usize,
    conv_stride_h: usize, conv_stride_w: usize,
    dilation_h: usize, dilation_w: usize,
    w_in_pad: usize, h_in_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    w_out_pad: usize, h_out_pad: usize,
    stride_w_val: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::gl_conv2d(
            x.as_ptr(),
            w_flat.as_ptr(),
            y.as_mut_ptr(),
            c_out as c_int, h_out as c_int, w_out as c_int,
            c_in as c_int, kernel_h as c_int, kernel_w as c_int,
            conv_stride_h as c_int, conv_stride_w as c_int,
            dilation_h as c_int, dilation_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int,
            stride_w_val as c_int,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn flatten_kernel2d(
    w: &DeviceBuffer<u64>,
    w_flat: &mut DeviceBuffer<u64>,
    c_out: usize, c_in: usize, kh: usize, kw: usize,
    kw_pad: usize, kh_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    dilation_h: usize, dilation_w: usize, s_w: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::gl_flatten_kernel2d(
            w.as_ptr(),
            w_flat.as_mut_ptr(),
            c_out as c_int, c_in as c_int, kh as c_int, kw as c_int,
            kw_pad as c_int, kh_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            dilation_h as c_int, dilation_w as c_int, s_w as c_int,
        )
    };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

pub fn relu_helper(x: &DeviceBuffer<u64>, neg: &mut DeviceBuffer<u64>, n: usize) -> Result<()> {
    let ret = unsafe { ffi::gl_relu_helper(x.as_ptr(), neg.as_mut_ptr(), n as c_int) };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

pub fn zero_buffer(buf: &mut DeviceBuffer<u64>, n: usize) -> Result<()> {
    let ret = unsafe { ffi::gl_zero_buffer(buf.as_mut_ptr(), n as c_int) };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d(
    x: &DeviceBuffer<u64>, w_flat: &DeviceBuffer<u64>, y: &mut DeviceBuffer<u64>,
    c_out: usize, d_out: usize, h_out: usize, w_out: usize,
    c_in: usize, kernel_d: usize, kernel_h: usize, kernel_w: usize,
    conv_stride_d: usize, conv_stride_h: usize, conv_stride_w: usize,
    w_in_pad: usize, h_in_pad: usize, d_in_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    w_out_pad: usize, h_out_pad: usize, d_out_pad: usize,
    stride_h_val: usize, stride_w_val: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::gl_conv3d(
            x.as_ptr(), w_flat.as_ptr(), y.as_mut_ptr(),
            c_out as c_int, d_out as c_int, h_out as c_int, w_out as c_int,
            c_in as c_int, kernel_d as c_int, kernel_h as c_int, kernel_w as c_int,
            conv_stride_d as c_int, conv_stride_h as c_int, conv_stride_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int, d_in_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int, d_out_pad as c_int,
            stride_h_val as c_int, stride_w_val as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn flatten_kernel3d(
    w: &DeviceBuffer<u64>, w_flat: &mut DeviceBuffer<u64>,
    c_out: usize, c_in: usize, kd: usize, kh: usize, kw: usize,
    kw_pad: usize, kh_pad: usize, kd_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    stride_h: usize, stride_w: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::gl_flatten_kernel3d(
            w.as_ptr(), w_flat.as_mut_ptr(),
            c_out as c_int, c_in as c_int, kd as c_int, kh as c_int, kw as c_int,
            kw_pad as c_int, kh_pad as c_int, kd_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            stride_h as c_int, stride_w as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d(
    x: &DeviceBuffer<u64>, w_flat: &DeviceBuffer<u64>, y: &mut DeviceBuffer<u64>,
    channels: usize, h_out: usize, w_out: usize,
    kernel_h: usize, kernel_w: usize,
    conv_stride_h: usize, conv_stride_w: usize,
    w_in_pad: usize, h_in_pad: usize,
    s_kernel_pad: usize,
    w_out_pad: usize, h_out_pad: usize,
    stride_w_val: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::gl_depthwise_conv2d(
            x.as_ptr(), w_flat.as_ptr(), y.as_mut_ptr(),
            channels as c_int, h_out as c_int, w_out as c_int,
            kernel_h as c_int, kernel_w as c_int,
            conv_stride_h as c_int, conv_stride_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int,
            s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int,
            stride_w_val as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}
