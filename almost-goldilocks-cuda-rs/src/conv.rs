//! Small GPU helpers used by the conv / relu / product-zero-check basicblocks.
//!
//! Larger convolution kernels (conv1d/2d/3d, depthwise, transpose) will land
//! here as we port each basicblock. For now this module hosts the two
//! one-liners that the ReLU and ProductZeroCheck blocks need.

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;

/// `neg[i] = (v > q/2) ? (q - v) : 0` over `n` field elements.
///
/// Used by [`crate`]'s ReLUHelper basicblock: when the input row is
/// "negative" in signed-int rep (canonical value > q/2), emit the magnitude;
/// otherwise emit zero. The ReLU output is then `y = x + neg`.
pub fn relu_helper(x: &DeviceBuffer<u64>, neg: &mut DeviceBuffer<u64>, n: usize) -> Result<()> {
    if x.len() < n || neg.len() < n {
        return Err(CudaError::InvalidArgument(format!(
            "relu_helper: buffer size mismatch (x={}, neg={}, n={})",
            x.len(),
            neg.len(),
            n
        )));
    }
    let ret = unsafe { ffi::agl_relu_helper_ffi(x.as_ptr(), neg.as_mut_ptr(), n as c_int) };
    if ret != 0 {
        return Err(CudaError::KernelFailed);
    }
    Ok(())
}

/// Zero the first `n` elements of `buf` via `cudaMemset`. Used by
/// ProductZeroCheck to materialize the all-zero certificate output, and by
/// the conv basicblocks to pre-zero the dilation-gap positions of `W_flat`.
pub fn zero_buffer(buf: &mut DeviceBuffer<u64>, n: usize) -> Result<()> {
    if buf.len() < n {
        return Err(CudaError::InvalidArgument(format!(
            "zero_buffer: buffer size {} < n {}",
            buf.len(),
            n
        )));
    }
    let ret = unsafe { ffi::agl_zero_buffer_ffi(buf.as_mut_ptr(), n) };
    if ret != 0 {
        // A memset failing with "illegal memory access" means the context was
        // already corrupted by an EARLIER launch -- an entirely different
        // diagnosis from the memset being wrong. Say which.
        return Err(CudaError::InvalidArgument(format!(
            "zero_buffer/cudaMemset: {}", crate::cuda_error_string(ret))));
    }
    Ok(())
}

/// 2D convolution. See the CUDA-side `agl_conv2d_kernel` for the exact index
/// formulation; the parameters here pass through directly.
#[allow(clippy::too_many_arguments)]
pub fn conv2d(
    x: &DeviceBuffer<u64>, w_flat: &DeviceBuffer<u64>, y: &mut DeviceBuffer<u64>,
    c_out: usize, h_out: usize, w_out: usize,
    c_in: usize, kernel_h: usize, kernel_w: usize,
    conv_stride_h: usize, conv_stride_w: usize,
    dilation_h: usize, dilation_w: usize,
    w_in_pad: usize, h_in_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    w_out_pad: usize, h_out_pad: usize,
    stride_w_val: usize,
    // Batch is the most significant dimension of X and Y; the weights are
    // shared across it. batch = 1 is exactly the unbatched call.
    batch: usize, x_stride: usize, y_stride: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::agl_conv2d_ffi(
            x.as_ptr(), w_flat.as_ptr(), y.as_mut_ptr(),
            c_out as c_int, h_out as c_int, w_out as c_int,
            c_in as c_int, kernel_h as c_int, kernel_w as c_int,
            conv_stride_h as c_int, conv_stride_w as c_int,
            dilation_h as c_int, dilation_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int,
            stride_w_val as c_int,
            batch as c_int, x_stride as c_int, y_stride as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// 2D transposed convolution (gather form). The CPU path scatters; this
/// inverts it so no atomics are needed. `y` must be pre-zeroed.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    x: &DeviceBuffer<u64>, w_flat: &DeviceBuffer<u64>, y: &mut DeviceBuffer<u64>,
    c_out: usize, h_out: usize, w_out: usize,
    c_in: usize, kernel_h: usize, kernel_w: usize,
    stride_h: usize, stride_w: usize,
    input_h: usize, input_w: usize,
    w_in_pad: usize, h_in_pad: usize,
    c_out_pad: usize, s_kernel_pad: usize,
    w_out_pad: usize, h_out_pad: usize,
    flat_stride: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::agl_conv_transpose2d_ffi(
            x.as_ptr(), w_flat.as_ptr(), y.as_mut_ptr(),
            c_out as c_int, h_out as c_int, w_out as c_int,
            c_in as c_int, kernel_h as c_int, kernel_w as c_int,
            stride_h as c_int, stride_w as c_int,
            input_h as c_int, input_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int,
            c_out_pad as c_int, s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int,
            flat_stride as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// 3D transposed convolution (gather form). Mirrors the CPU
/// `ConvTranspose3D::run` index-for-index. `y` must be pre-zeroed: the kernel
/// writes only the valid output box, leaving the padded remainder untouched.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d(
    x: &DeviceBuffer<u64>, w_flat: &DeviceBuffer<u64>, y: &mut DeviceBuffer<u64>,
    c_out: usize, d_out: usize, h_out: usize, w_out: usize,
    c_in: usize, kernel_d: usize, kernel_h: usize, kernel_w: usize,
    stride_d: usize, stride_h: usize, stride_w: usize,
    input_d: usize, input_h: usize, input_w: usize,
    w_in_pad: usize, h_in_pad: usize, d_in_pad: usize,
    c_out_pad: usize, s_kernel_pad: usize,
    w_out_pad: usize, h_out_pad: usize, d_out_pad: usize,
    flat_stride_h: usize, flat_stride_w: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::agl_conv_transpose3d_ffi(
            x.as_ptr(), w_flat.as_ptr(), y.as_mut_ptr(),
            c_out as c_int, d_out as c_int, h_out as c_int, w_out as c_int,
            c_in as c_int, kernel_d as c_int, kernel_h as c_int, kernel_w as c_int,
            stride_d as c_int, stride_h as c_int, stride_w as c_int,
            input_d as c_int, input_h as c_int, input_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int, d_in_pad as c_int,
            c_out_pad as c_int, s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int, d_out_pad as c_int,
            flat_stride_h as c_int, flat_stride_w as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// 3D convolution. Same protocol-side mapping as `conv2d` with an extra
/// depth axis.
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
    // Bounds guard, off by default. Armed with ZK4_CONV_BOUNDS=1, which makes
    // the kernel range-check every access against the REAL allocation sizes and
    // report the first offender instead of faulting. It costs a device sync per
    // launch, so it is a debugging tool, not a default -- but on a machine whose
    // driver compute-sanitizer refuses to load, it is the only way to see the
    // offending index at all.
    let armed = std::env::var("ZK4_CONV_BOUNDS").is_ok();
    let (x_len, w_len, y_len) = if armed {
        (x.len() as i64, w_flat.len() as i64, y.len() as i64)
    } else {
        (0, 0, 0)
    };
    let ret = unsafe {
        ffi::agl_conv3d_ffi(
            x.as_ptr(), w_flat.as_ptr(), y.as_mut_ptr(),
            c_out as c_int, d_out as c_int, h_out as c_int, w_out as c_int,
            c_in as c_int, kernel_d as c_int, kernel_h as c_int, kernel_w as c_int,
            conv_stride_d as c_int, conv_stride_h as c_int, conv_stride_w as c_int,
            w_in_pad as c_int, h_in_pad as c_int, d_in_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            w_out_pad as c_int, h_out_pad as c_int, d_out_pad as c_int,
            stride_h_val as c_int, stride_w_val as c_int,
            x_len, w_len, y_len,
        )
    };
    if ret == -2 {
        return Err(CudaError::InvalidArgument(
            "conv3d: kernel index out of bounds (see [conv3d OOB] on stderr)".into()));
    }
    if ret != 0 {
        return Err(CudaError::InvalidArgument(format!(
            "conv3d kernel: {}", crate::cuda_error_string(ret))));
    }
    Ok(())
}

/// Depthwise 2D convolution (one filter per channel).
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
        ffi::agl_depthwise_conv2d_ffi(
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

/// Full 1D flat convolution: the `Y_full` aux witness shared by the conv
/// basicblocks. Gather formulation over `(d, m)`, `m < s_full`:
///   `Y_full[d, m] = Σ_c Σ_taps X[c, p]·W_flat[d, c, j]`, `p = (s_in−1) − m + j`
/// with tap index `j = kd·tap_d + kh·tap_h + kw·tap_w`. See the CUDA-side
/// `agl_conv_full_kernel` for the per-variant tap-stride mapping. With
/// `depthwise` set the channel loop collapses to `c = d` and `w_flat` is
/// `[C, s_kernel_pad]`. `y_full` must be pre-zeroed ([`zero_buffer`]).
#[allow(clippy::too_many_arguments)]
pub fn conv_full(
    x: &DeviceBuffer<u64>, w_flat: &DeviceBuffer<u64>, y_full: &mut DeviceBuffer<u64>,
    c_out: usize, c_in: usize,
    kernel_d: usize, kernel_h: usize, kernel_w: usize,
    tap_d: usize, tap_h: usize, tap_w: usize,
    s_in: usize, s_full: usize, s_full_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    depthwise: bool,
    // See `conv2d`. Y_full carries the batch index in the same position.
    batch: usize, x_stride: usize, yf_stride: usize,
) -> Result<()> {
    // Highest indices the gather touches (see the kernel index formulation).
    let nb = batch.max(1);
    // See conv3d: armed by ZK4_CONV_BOUNDS=1, zero otherwise.
    let armed = std::env::var("ZK4_CONV_BOUNDS").is_ok();
    let (gx_len, gw_len, gy_len) = if armed {
        (x.len() as i64, w_flat.len() as i64, y_full.len() as i64)
    } else { (0, 0, 0) };
    let x_need = (nb - 1) * x_stride
        + if depthwise { c_out * s_in } else { c_in * s_in };
    let w_need = if depthwise {
        c_out * s_kernel_pad
    } else {
        c_out.saturating_sub(1) * s_kernel_pad * c_in_pad + c_in * s_kernel_pad
    };
    let y_need = (nb - 1) * yf_stride + c_out * s_full_pad;
    if x.len() < x_need || w_flat.len() < w_need || y_full.len() < y_need {
        return Err(CudaError::InvalidArgument(format!(
            "conv_full: buffer size mismatch (x={} need {}, w_flat={} need {}, y_full={} need {})",
            x.len(), x_need,
            w_flat.len(), w_need,
            y_full.len(), y_need
        )));
    }
    let ret = unsafe {
        ffi::agl_conv_full_ffi(
            x.as_ptr(), w_flat.as_ptr(), y_full.as_mut_ptr(),
            c_out as c_int, c_in as c_int,
            kernel_d as c_int, kernel_h as c_int, kernel_w as c_int,
            tap_d as c_int, tap_h as c_int, tap_w as c_int,
            s_in as c_int, s_full as c_int, s_full_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            depthwise as c_int,
            nb as c_int, x_stride as c_int, yf_stride as c_int,
            gx_len, gw_len, gy_len,
        )
    };
    if ret == -2 {
        return Err(CudaError::InvalidArgument(
            "conv_full: kernel index out of bounds (see [conv_full OOB] on stderr)".into()));
    }
    if ret != 0 {
        return Err(CudaError::InvalidArgument(format!(
            "conv_full kernel: {}", crate::cuda_error_string(ret))));
    }
    Ok(())
}

/// 2D kernel flatten: scatter `W[C_out, C_in, kH, kW]` into the dilated
/// `W_flat[C_out, C_in, S_pad]`. Output must be pre-zeroed.
#[allow(clippy::too_many_arguments)]
pub fn flatten_kernel2d(
    w: &DeviceBuffer<u64>, w_flat: &mut DeviceBuffer<u64>,
    c_out: usize, c_in: usize, kh: usize, kw: usize,
    kw_pad: usize, kh_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    dilation_h: usize, dilation_w: usize, s_w: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::agl_flatten_kernel2d_ffi(
            w.as_ptr(), w_flat.as_mut_ptr(),
            c_out as c_int, c_in as c_int, kh as c_int, kw as c_int,
            kw_pad as c_int, kh_pad as c_int,
            c_in_pad as c_int, s_kernel_pad as c_int,
            dilation_h as c_int, dilation_w as c_int, s_w as c_int,
        )
    };
    if ret != 0 { return Err(CudaError::KernelFailed); }
    Ok(())
}

/// 3D kernel flatten — adds the depth dim to [`flatten_kernel2d`].
#[allow(clippy::too_many_arguments)]
pub fn flatten_kernel3d(
    w: &DeviceBuffer<u64>, w_flat: &mut DeviceBuffer<u64>,
    c_out: usize, c_in: usize, kd: usize, kh: usize, kw: usize,
    kw_pad: usize, kh_pad: usize, kd_pad: usize,
    c_in_pad: usize, s_kernel_pad: usize,
    stride_h: usize, stride_w: usize,
) -> Result<()> {
    let ret = unsafe {
        ffi::agl_flatten_kernel3d_ffi(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{AlmostGoldilocksField, ALMOST_GOLDILOCKS_PRIME};

    fn cuda_ready() -> bool {
        crate::init().is_ok()
    }

    /// Sign detection: positive (canonical ≤ q/2) → 0; negative (> q/2) →
    /// `q - v` (which is the magnitude `|v_signed|`).
    #[test]
    fn relu_helper_classifies_sign() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        // Pick representatives across the signed range.
        let pos1 = 0u64;
        let pos2 = 1u64;
        let pos3 = ALMOST_GOLDILOCKS_PRIME / 2; // exactly q/2 — stays positive (strict >).
        let neg1 = ALMOST_GOLDILOCKS_PRIME - 1; // = -1
        let neg2 = ALMOST_GOLDILOCKS_PRIME - 7; // = -7
        let raw = vec![pos1, pos2, pos3, neg1, neg2];

        let d_x = DeviceBuffer::<u64>::from_slice(&raw).expect("up");
        let mut d_neg = DeviceBuffer::<u64>::new(raw.len()).expect("alloc");
        relu_helper(&d_x, &mut d_neg, raw.len()).expect("kernel");
        let out = d_neg.to_vec().expect("dl");
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], ALMOST_GOLDILOCKS_PRIME - neg1);
        assert_eq!(out[4], ALMOST_GOLDILOCKS_PRIME - neg2);
        // The recovered magnitudes are the absolute values.
        assert_eq!(AlmostGoldilocksField(out[3]), AlmostGoldilocksField(1));
        assert_eq!(AlmostGoldilocksField(out[4]), AlmostGoldilocksField(7));
    }

    /// `conv2d` kernel matches a CPU reference on a 3×3, stride 1, no
    /// dilation matmul-like instance.
    #[test]
    fn conv2d_kernel_matches_cpu_reference() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        let c_in = 2usize;
        let c_out = 3usize;
        let h_in = 4usize;
        let w_in = 4usize;
        let kh = 3usize;
        let kw = 3usize;
        let h_out = h_in - kh + 1;
        let w_out = w_in - kw + 1;
        let s_w = w_in; // unused dilation factor — kernel sees stride_w_val = w_in_pad.

        // Padded sizes (power-of-2 for the witness layout).
        let w_in_pad = w_in.next_power_of_two();
        let h_in_pad = h_in.next_power_of_two();
        let c_in_pad = c_in.next_power_of_two();
        let s_kernel_pad = (kh * s_w).next_power_of_two();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_out_pad = c_out.next_power_of_two();

        // Build X with little-endian flat layout.
        let mut x = vec![0u64; c_in_pad * h_in_pad * w_in_pad];
        for c in 0..c_in {
            for h in 0..h_in {
                for w in 0..w_in {
                    x[w + h * w_in_pad + c * w_in_pad * h_in_pad] =
                        ((c * h_in + h) * w_in + w) as u64 + 1;
                }
            }
        }
        // Build W_flat in dilated layout: position j = kh_i * s_w + kw_i.
        let mut w_flat = vec![0u64; c_out_pad * c_in_pad * s_kernel_pad];
        for d in 0..c_out {
            for c in 0..c_in {
                for kh_i in 0..kh {
                    for kw_i in 0..kw {
                        let j = kh_i * s_w + kw_i;
                        let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                        w_flat[wf_idx] = ((d * c_in + c) * kh * kw + kh_i * kw + kw_i + 1) as u64;
                    }
                }
            }
        }

        let d_x = DeviceBuffer::<u64>::from_slice(&x).expect("x up");
        let d_w = DeviceBuffer::<u64>::from_slice(&w_flat).expect("w up");
        let mut d_y = DeviceBuffer::<u64>::new(c_out_pad * h_out_pad * w_out_pad).expect("alloc y");
        let dy_len = d_y.len();
        zero_buffer(&mut d_y, dy_len).expect("zero y");
        conv2d(
            &d_x, &d_w, &mut d_y,
            c_out, h_out, w_out, c_in, kh, kw,
            1, 1, 1, 1,
            w_in_pad, h_in_pad, c_in_pad, s_kernel_pad,
            w_out_pad, h_out_pad, s_w,
            // batch=1: strides are unused, the grid covers one image.
            1, 0, 0,
        )
        .expect("conv2d");
        let y_gpu = d_y.to_vec().expect("y dl");

        // CPU reference.
        use crate::field::AlmostGoldilocksField;
        for d in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = AlmostGoldilocksField(0);
                    for c in 0..c_in {
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let ih = ho + kh_i;
                                let iw = wo + kw_i;
                                let xv = AlmostGoldilocksField(
                                    x[iw + ih * w_in_pad + c * w_in_pad * h_in_pad],
                                );
                                let j = kh_i * s_w + kw_i;
                                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                                let wv = AlmostGoldilocksField(w_flat[wf_idx]);
                                acc = acc + xv * wv;
                            }
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                    let got = AlmostGoldilocksField(y_gpu[out_idx]).reduce();
                    assert_eq!(got, acc.reduce(), "(d,ho,wo) = ({}, {}, {})", d, ho, wo);
                }
            }
        }
    }

    /// Depthwise: each channel independently.
    #[test]
    fn depthwise_conv2d_kernel_matches_cpu() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        let channels = 2usize;
        let h_in = 4usize; let w_in = 4usize;
        let kh = 2usize; let kw = 2usize;
        let h_out = h_in - kh + 1;
        let w_out = w_in - kw + 1;
        let s_w = w_in;
        let w_in_pad = w_in.next_power_of_two();
        let h_in_pad = h_in.next_power_of_two();
        let c_pad = channels.next_power_of_two();
        let s_kernel_pad = (kh * s_w).next_power_of_two();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();

        let mut x = vec![0u64; c_pad * h_in_pad * w_in_pad];
        for c in 0..channels {
            for h in 0..h_in {
                for w in 0..w_in {
                    x[w + h * w_in_pad + c * w_in_pad * h_in_pad] = (c + 1) as u64 * 100 + (h * w_in + w) as u64;
                }
            }
        }
        let mut w_flat = vec![0u64; c_pad * s_kernel_pad];
        for c in 0..channels {
            for kh_i in 0..kh {
                for kw_i in 0..kw {
                    let j = kh_i * s_w + kw_i;
                    w_flat[j + c * s_kernel_pad] = (c + 1) as u64 * 10 + (kh_i * kw + kw_i) as u64;
                }
            }
        }

        let d_x = DeviceBuffer::<u64>::from_slice(&x).expect("x");
        let d_w = DeviceBuffer::<u64>::from_slice(&w_flat).expect("w");
        let mut d_y = DeviceBuffer::<u64>::new(c_pad * h_out_pad * w_out_pad).expect("y");
        let dy_len = d_y.len();
        zero_buffer(&mut d_y, dy_len).expect("zero");
        depthwise_conv2d(
            &d_x, &d_w, &mut d_y,
            channels, h_out, w_out, kh, kw, 1, 1,
            w_in_pad, h_in_pad, s_kernel_pad,
            w_out_pad, h_out_pad, s_w,
        )
        .expect("dw");
        let y_gpu = d_y.to_vec().expect("dl");

        use crate::field::AlmostGoldilocksField;
        for c in 0..channels {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = AlmostGoldilocksField(0);
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let ih = ho + kh_i;
                            let iw = wo + kw_i;
                            let xv = AlmostGoldilocksField(x[iw + ih * w_in_pad + c * w_in_pad * h_in_pad]);
                            let j = kh_i * s_w + kw_i;
                            let wv = AlmostGoldilocksField(w_flat[j + c * s_kernel_pad]);
                            acc = acc + xv * wv;
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                    assert_eq!(
                        AlmostGoldilocksField(y_gpu[out_idx]).reduce(),
                        acc.reduce(),
                        "(c,h,w) = ({}, {}, {})", c, ho, wo,
                    );
                }
            }
        }
    }

    #[test]
    fn zero_buffer_zeros_n_elements() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        let raw = vec![0xDEAD_BEEFu64; 10];
        let mut buf = DeviceBuffer::<u64>::from_slice(&raw).expect("up");
        zero_buffer(&mut buf, 6).expect("kernel");
        let out = buf.to_vec().expect("dl");
        for i in 0..6 {
            assert_eq!(out[i], 0, "i = {}", i);
        }
        for i in 6..10 {
            assert_eq!(out[i], 0xDEAD_BEEF, "i = {} should be untouched", i);
        }
    }
}
