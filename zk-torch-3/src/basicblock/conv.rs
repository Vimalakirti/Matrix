use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksField, GoldilocksExt2};
use goldilocks_cuda::bit_decomp::memset_zero;
use goldilocks_cuda::conv::{
    conv2d as gpu_conv2d, conv3d as gpu_conv3d, depthwise_conv2d as gpu_depthwise_conv2d,
    flatten_kernel2d as gpu_flatten_kernel2d, flatten_kernel3d as gpu_flatten_kernel3d,
};
use rayon::prelude::*;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Witness, DataType, Role};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_sub, ext2_mul, log2_ceil, get_n, gl_add, gl_mul};

// ============================================================================
// Conv2D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct Conv2D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub stride_w: usize,   // S_w = W_in (flat 1D stride)
    pub s_in: usize,       // H_in * W_in (input spatial size)
    pub s_kernel: usize,   // (kH-1)*dilation_h*W_in + (kW-1)*dilation_w + 1 (kernel spatial in flat layout)
    pub h_out: usize,
    pub w_out: usize,
    pub conv_stride_h: usize, // convolution stride height (default 1)
    pub conv_stride_w: usize, // convolution stride width (default 1)
    pub dilation_h: usize, // dilation height (default 1)
    pub dilation_w: usize, // dilation width (default 1)
}

impl Conv2D {
    pub fn new(c_in: usize, c_out: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize) -> Self {
        Self::new_dilated(c_in, c_out, kernel_h, kernel_w, input_h, input_w, 1, 1, 1, 1)
    }

    pub fn new_strided(c_in: usize, c_out: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, conv_stride_h: usize, conv_stride_w: usize) -> Self {
        Self::new_dilated(c_in, c_out, kernel_h, kernel_w, input_h, input_w, conv_stride_h, conv_stride_w, 1, 1)
    }

    pub fn new_dilated(c_in: usize, c_out: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, conv_stride_h: usize, conv_stride_w: usize, dilation_h: usize, dilation_w: usize) -> Self {
        let h_out = (input_h - dilation_h * (kernel_h - 1) - 1) / conv_stride_h + 1;
        let w_out = (input_w - dilation_w * (kernel_w - 1) - 1) / conv_stride_w + 1;
        // Use w_pad as stride so 1D flat index = ih * w_pad + iw decomposes into MLE variables
        let w_pad = input_w.next_power_of_two();
        let h_pad = input_h.next_power_of_two();
        let stride_w = w_pad;
        let s_in = h_pad * w_pad;
        // s_kernel accounts for dilation gaps in the flat 1D layout
        let s_kernel = (kernel_h - 1) * dilation_h * w_pad + (kernel_w - 1) * dilation_w + 1;
        Self { c_in, c_out, kernel_h, kernel_w, input_h, input_w, stride_w, s_in, s_kernel, h_out, w_out, conv_stride_h, conv_stride_w, dilation_h, dilation_w }
    }

    /// Number of variables for channel-out dimension (padded to power of 2).
    fn l_d(&self) -> usize { log2_ceil(self.c_out.max(1)) }
    /// Number of variables for channel-in dimension.
    fn l_c(&self) -> usize { log2_ceil(self.c_in.max(1)) }
    /// Number of variables for output width.
    fn l_wo(&self) -> usize { log2_ceil(self.w_out.max(1)) }
    /// Number of variables for output height.
    fn l_ho(&self) -> usize { log2_ceil(self.h_out.max(1)) }
    /// Number of variables for output spatial (l_wo + l_ho).
    fn l_spatial_out(&self) -> usize { self.l_wo() + self.l_ho() }
    /// Number of variables for input spatial (l_wi + l_hi).
    fn l_spatial_in(&self) -> usize { log2_ceil(self.input_w.max(1)) + log2_ceil(self.input_h.max(1)) }
    /// Number of variables for kernel spatial (s_kernel padded).
    fn l_kernel(&self) -> usize { log2_ceil(self.s_kernel.max(1)) }
}

impl BasicBlock for Conv2D {
    /// run(): Standard conv2d, outputting Y[C_out, H_out, W_out] in little-endian layout.
    /// Inputs: [X, W_flat]
    ///   X shape: [C_in, H_in, W_in], little-endian layout (w_in bits lowest, then h_in, then c_in)
    ///   W_flat shape: [C_out, C_in, S_kernel_pad], little-endian layout
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_in_pad = self.c_in.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();

        let out_size = c_out_pad * h_out_pad * w_out_pad;

        // Y[d, ho, wo] = Σ_c Σ_kh Σ_kw X[c, ho+kh, wo+kw] * W[d, c, kh, kw]
        // Parallelize over output elements (d, ho, wo)
        let x_data = x.data.as_ref().unwrap();
        let w_data = w_flat.data.as_ref().unwrap();
        let c_out = self.c_out;
        let h_out = self.h_out;
        let w_out = self.w_out;
        let c_in = self.c_in;
        let kernel_h = self.kernel_h;
        let kernel_w = self.kernel_w;
        let conv_stride_h = self.conv_stride_h;
        let conv_stride_w = self.conv_stride_w;
        let dilation_h = self.dilation_h;
        let dilation_w = self.dilation_w;
        let stride_w_val = self.stride_w;

        let total_outputs = c_out * h_out * w_out;
        let mut out_data = vec![GoldilocksField(0); out_size];
        let results: Vec<(usize, GoldilocksField)> = (0..total_outputs)
            .into_par_iter()
            .map(|flat_idx| {
                let wo = flat_idx % w_out;
                let ho = (flat_idx / w_out) % h_out;
                let d = flat_idx / (w_out * h_out);
                let mut acc = GoldilocksField(0);
                for c in 0..c_in {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = ho * conv_stride_h + kh * dilation_h;
                            let iw = wo * conv_stride_w + kw * dilation_w;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let j = kh * dilation_h * stride_w_val + kw * dilation_w;
                            let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                            let x_val = x_data.index(x_idx);
                            let w_val = w_data.index(wf_idx);
                            acc = gl_add(acc, gl_mul(x_val, w_val));
                        }
                    }
                }
                let out_idx = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                (out_idx, acc)
            })
            .collect();
        for (idx, val) in results {
            out_data[idx] = val;
        }

        let out_shape = vec![self.c_out, self.h_out, self.w_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_in_pad = self.c_in.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let out_size = c_out_pad * h_out_pad * w_out_pad;

        let d_x = x.as_device_buf();
        let d_w = w_flat.as_device_buf();
        let mut d_y = DeviceBuffer::<u64>::new(out_size).expect("Conv2D: alloc out");
        // Zero pad regions outside [c_out, h_out, w_out].
        memset_zero(&mut d_y, out_size).expect("Conv2D: memset zero failed");

        gpu_conv2d(
            &d_x, &d_w, &mut d_y,
            self.c_out, self.h_out, self.w_out,
            self.c_in, self.kernel_h, self.kernel_w,
            self.conv_stride_h, self.conv_stride_w,
            self.dilation_h, self.dilation_w,
            w_in_pad, h_in_pad,
            c_in_pad, s_kernel_pad,
            w_out_pad, h_out_pad,
            self.stride_w,
        ).expect("Conv2D: gpu kernel failed");

        let out_shape = vec![self.c_out, self.h_out, self.w_out];
        vec![Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        conv2d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        conv2d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// Conv1D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct Conv1D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub input_len: usize,
    pub s_in: usize,      // input_len.next_power_of_two()
    pub s_kernel: usize,  // kernel_size
    pub l_out: usize,     // (input_len - kernel_size) / conv_stride + 1
    pub conv_stride: usize, // convolution stride (default 1)
}

impl Conv1D {
    pub fn new(c_in: usize, c_out: usize, kernel_size: usize, input_len: usize) -> Self {
        Self::new_strided(c_in, c_out, kernel_size, input_len, 1)
    }

    pub fn new_strided(c_in: usize, c_out: usize, kernel_size: usize, input_len: usize, conv_stride: usize) -> Self {
        let l_out = (input_len - kernel_size) / conv_stride + 1;
        let s_in = input_len.next_power_of_two();
        let s_kernel = kernel_size;
        Self { c_in, c_out, kernel_size, input_len, s_in, s_kernel, l_out, conv_stride }
    }

    fn l_d(&self) -> usize { log2_ceil(self.c_out.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.c_in.max(1)) }
    fn l_lo(&self) -> usize { log2_ceil(self.l_out.max(1)) }
    fn l_spatial_in(&self) -> usize { log2_ceil(self.input_len.max(1)) }
    fn l_kernel(&self) -> usize { log2_ceil(self.s_kernel.max(1)) }
}

impl BasicBlock for Conv1D {
    /// run(): Conv1D. X[C_in, L_in], W[C_out, C_in, K] → Y[C_out, L_out]
    /// Little-endian: X has l_in bits (lowest) | c_in bits.
    /// W has k bits (lowest) | c_in bits | c_out bits.
    /// Y has l_out bits (lowest) | c_out bits.
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w = inputs[1];

        let c_in_pad = self.c_in.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();
        let l_in_pad = self.input_len.next_power_of_two();
        let l_out_pad = self.l_out.next_power_of_two();
        let k_pad = self.kernel_size.next_power_of_two();

        let out_size = c_out_pad * l_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for d in 0..self.c_out {
            for lo in 0..self.l_out {
                let mut acc = GoldilocksField(0);
                for c in 0..self.c_in {
                    for k in 0..self.kernel_size {
                        let il = lo * self.conv_stride + k;
                        // X index: l_in bits (lowest) | c_in bits
                        let x_idx = il + c * l_in_pad;
                        // W index: k bits (lowest) | c_in bits | c_out bits
                        let w_idx = k + c * k_pad + d * k_pad * c_in_pad;
                        let x_val = x.data.as_ref().unwrap().index(x_idx);
                        let w_val = w.data.as_ref().unwrap().index(w_idx);
                        acc = gl_add(acc, gl_mul(x_val, w_val));
                    }
                }
                let out_idx = lo + d * l_out_pad;
                out_data[out_idx] = acc;
            }
        }

        let out_shape = vec![self.c_out, self.l_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        conv1d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        conv1d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// FlattenKernel BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct FlattenKernel {
    pub s_w: usize,    // Flat stride (= W_in)
    pub kh: usize,
    pub kw: usize,
    pub c_out: usize,
    pub c_in: usize,
    pub dilation_h: usize, // dilation height (default 1)
    pub dilation_w: usize, // dilation width (default 1)
}

impl BasicBlock for FlattenKernel {
    /// run(): Scatter W[C_out, C_in, kH, kW] → W_flat[C_out, C_in, S_kernel_pad]
    /// where S_kernel accounts for dilation: j = kh*dilation_h*W_in + kw*dilation_w.
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let w = inputs[0];
        let c_out_pad = self.c_out.next_power_of_two();
        let c_in_pad = self.c_in.next_power_of_two();
        let kh_pad = self.kh.next_power_of_two();
        let kw_pad = self.kw.next_power_of_two();
        let s_kernel = (self.kh - 1) * self.dilation_h * self.s_w + (self.kw - 1) * self.dilation_w + 1;
        let s_kernel_pad = s_kernel.next_power_of_two();

        let out_size = c_out_pad * c_in_pad * s_kernel_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for d in 0..self.c_out {
            for c in 0..self.c_in {
                for kh in 0..self.kh {
                    for kw in 0..self.kw {
                        // W little-endian: kw bits (lowest) | kh bits | c_in bits | c_out bits
                        let w_idx = kw + kh * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                        let j = kh * self.dilation_h * self.s_w + kw * self.dilation_w;
                        // W_flat little-endian: j bits (lowest) | c_in bits | c_out bits
                        let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                        out_data[wf_idx] = w.data.as_ref().unwrap().index(w_idx);
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let w = inputs[0];
        let c_out_pad = self.c_out.next_power_of_two();
        let c_in_pad = self.c_in.next_power_of_two();
        let kh_pad = self.kh.next_power_of_two();
        let kw_pad = self.kw.next_power_of_two();
        let s_kernel = (self.kh - 1) * self.dilation_h * self.s_w + (self.kw - 1) * self.dilation_w + 1;
        let s_kernel_pad = s_kernel.next_power_of_two();
        let out_size = c_out_pad * c_in_pad * s_kernel_pad;

        let d_w = w.as_device_buf();
        let mut d_wf = DeviceBuffer::<u64>::new(out_size).expect("FlattenKernel: alloc");
        memset_zero(&mut d_wf, out_size).expect("FlattenKernel: memset");

        gpu_flatten_kernel2d(
            &d_w, &mut d_wf,
            self.c_out, self.c_in, self.kh, self.kw,
            kw_pad, kh_pad,
            c_in_pad, s_kernel_pad,
            self.dilation_h, self.dilation_w, self.s_w,
        ).expect("FlattenKernel: gpu kernel failed");

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        vec![Witness::new_device(out_shape, Arc::new(d_wf), DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        flatten_kernel_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        flatten_kernel_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// Helper: compute α-table MLE
// ============================================================================

/// Compute α-power table: α_table[k] = α^k for k in 0..size.
fn alpha_power_table(alpha: GoldilocksExt2, size: usize) -> Vec<GoldilocksExt2> {
    let mut table = Vec::with_capacity(size);
    let mut pow = GoldilocksExt2::one();
    for _ in 0..size {
        table.push(pow);
        pow = ext2_mul(pow, alpha);
    }
    table
}

/// Evaluate the α-table MLE at point r.
/// α_table_mle(r) = Π_j(1 + r_j*(α^{2^j} - 1))
fn alpha_table_mle_eval(alpha: GoldilocksExt2, r: &[GoldilocksExt2]) -> GoldilocksExt2 {
    let one = GoldilocksExt2::one();
    let mut result = one;
    let mut alpha_pow = alpha; // α^{2^0} = α
    for &rj in r {
        // factor = 1 + r_j * (α^{2^j} - 1)
        let factor = ext2_add(one, ext2_mul(rj, ext2_sub(alpha_pow, one)));
        result = ext2_mul(result, factor);
        alpha_pow = ext2_mul(alpha_pow, alpha_pow); // α^{2^{j+1}} = (α^{2^j})^2
    }
    result
}

// ============================================================================
// FlattenKernel prove/verify
// ============================================================================

/// FlattenKernel prove:
/// Given claim on W_flat at (r_d, r_c, r_j), proves:
///   W_flat(r_d, r_c, r_j) = Σ_{kh,kw} W(r_d, r_c, kh, kw) · eq(r_j, kh·S_w + kw)
/// Runs a small sumcheck (l_kh + l_kw rounds) to reduce to a claim on W.
fn flatten_kernel_prove(
    fk: &FlattenKernel,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    // edge_ids: [w_edge, wf_edge]
    // witnesses: [W, W_flat]
    // out_claims: claim on W_flat output
    let out_claim = out_claims[0];
    let w_edge = edge_ids[0];

    let l_d = log2_ceil(fk.c_out.max(1));
    let l_c = log2_ceil(fk.c_in.max(1));
    let l_kh = log2_ceil(fk.kh.max(1));
    let l_kw = log2_ceil(fk.kw.max(1));
    let s_kernel = (fk.kh - 1) * fk.dilation_h * fk.s_w + (fk.kw - 1) * fk.dilation_w + 1;
    let l_j = log2_ceil(s_kernel.max(1));

    // Parse the claim point: (r_j, r_c, r_d) in little-endian order
    // W_flat has shape [C_out, C_in, S_kernel] → dims in little-endian: s_kernel bits, c_in bits, c_out bits
    let r_j = &out_claim.point[..l_j];
    let r_c = &out_claim.point[l_j..l_j + l_c];
    let r_d = &out_claim.point[l_j + l_c..l_j + l_c + l_d];
    // Build eq(r_j, ·) table for all possible j = kh*dilation_h*S_w + kw*dilation_w values
    let eq_j = evaluate_lagrange_basis_ext2(r_j);

    // Build W_partial: partially evaluate W at (r_d, r_c, kh, kw)
    // W has shape [C_out, C_in, kH, kW] → dims: kw bits, kh bits, c_in bits, c_out bits
    let w_data = witnesses[0]; // W witness
    let kh_pad = fk.kh.next_power_of_two();
    let kw_pad = fk.kw.next_power_of_two();
    let c_in_pad = fk.c_in.next_power_of_two();

    // Evaluate W's MLE at (r_d, r_c) for each (kh, kw) pair.
    // Build eq tables for channel dims
    let eq_c = evaluate_lagrange_basis_ext2(r_c);
    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // W_partial[kh, kw] = Σ_d Σ_c eq_D(d) · eq_C(c) · W[d, c, kh, kw]
    let sumcheck_size = kh_pad * kw_pad;
    let mut w_partial = vec![GoldilocksExt2::zero(); sumcheck_size];
    for d in 0..fk.c_out {
        for c in 0..fk.c_in {
            let dc_weight = ext2_mul(eq_d[d], eq_c[c]);
            for kh in 0..fk.kh {
                for kw in 0..fk.kw {
                    let w_idx = kw + kh * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                    let w_val = GoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                    let sc_idx = kw + kh * kw_pad;
                    w_partial[sc_idx] = ext2_add(w_partial[sc_idx], ext2_mul(dc_weight, w_val));
                }
            }
        }
    }

    // Build H[kh, kw] = eq(r_j, kh*dilation_h*S_w + kw*dilation_w)
    // H is indexed in little-endian: kw bits (lowest), kh bits
    let mut h_poly = vec![GoldilocksExt2::zero(); sumcheck_size];
    for kh in 0..fk.kh {
        for kw in 0..fk.kw {
            let j = kh * fk.dilation_h * fk.s_w + kw * fk.dilation_w;
            if j < eq_j.len() {
                let sc_idx = kw + kh * kw_pad;
                h_poly[sc_idx] = eq_j[j];
            }
        }
    }

    // Sumcheck: Σ_{kh,kw} H[kh,kw] · W_partial[kh,kw] = claimed_val
    let num_rounds = l_kh + l_kw;
    let mut prover = CpuLinearSumcheckProverExt2::new(num_rounds, 2, transcript);
    let proof = prover.prove(&mut [h_poly, w_partial].as_mut_slice(), transcript);

    let challenges = &prover.challenges;
    // challenges[0..l_kw] = r_kw', challenges[l_kw..l_kw+l_kh] = r_kh'
    let r_kw_new = &challenges[..l_kw];
    let r_kh_new = &challenges[l_kw..];

    // Build claim on W at point (r_kw', r_kh', r_c, r_d)
    let mut w_point = Vec::with_capacity(l_kw + l_kh + l_c + l_d);
    w_point.extend_from_slice(r_kw_new);
    w_point.extend_from_slice(r_kh_new);
    w_point.extend_from_slice(r_c);
    w_point.extend_from_slice(r_d);

    let w_eval = witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&w_point);

    let w_claim = Claim {
        edge_id: w_edge,
        sparse_id: 0,
        point: w_point,
        eval: w_eval,
    };

    (vec![proof], vec![w_claim])
}

fn flatten_kernel_verify(
    fk: &FlattenKernel,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    let out_claim = claims.last().unwrap();
    let w_claim = &claims[0];

    let l_kh = log2_ceil(fk.kh.max(1));
    let l_kw = log2_ceil(fk.kw.max(1));
    let s_kernel = (fk.kh - 1) * fk.dilation_h * fk.s_w + (fk.kw - 1) * fk.dilation_w + 1;
    let l_j = log2_ceil(s_kernel.max(1));

    let r_j = &out_claim.point[..l_j];

    let num_rounds = l_kh + l_kw;
    let (ok, challenges) = SumcheckVerifier::verify(
        sumcheck_proofs[0],
        out_claim.eval,
        num_rounds,
        2,
        transcript,
    );
    if !ok {
        println!("FlattenKernel sumcheck verification failed");
        return false;
    }

    // Verify final eval: H(r') · W_partial(r') = final_eval
    // H(r') = eq(r_j, kh'*S_w + kw') where (kw', kh') = challenges
    // But the verifier can compute this:
    // α_table-like: eq(r_j, index) evaluated at the MLE of the j-mapping
    // Actually: H is the MLE of the sparse table { (kw+kh*kw_pad) → eq_j[kh*S_w+kw] }
    // Evaluated at challenges = (r_kw', r_kh')
    // H_mle(r_kw', r_kh') = Σ_{kh,kw} eq((r_kw',r_kh'), (kw,kh)) · eq(r_j, kh*S_w+kw)
    // The verifier must compute this. Since kernel is small, direct computation is fine.

    let r_kw_new = &challenges[..l_kw];
    let r_kh_new = &challenges[l_kw..];

    let eq_kw = evaluate_lagrange_basis_ext2(r_kw_new);
    let eq_kh = evaluate_lagrange_basis_ext2(r_kh_new);
    let eq_j_table = evaluate_lagrange_basis_ext2(r_j);

    let mut h_eval = GoldilocksExt2::zero();
    for kh in 0..fk.kh {
        for kw in 0..fk.kw {
            let j = kh * fk.dilation_h * fk.s_w + kw * fk.dilation_w;
            if j < eq_j_table.len() {
                h_eval = ext2_add(h_eval, ext2_mul(ext2_mul(eq_kh[kh], eq_kw[kw]), eq_j_table[j]));
            }
        }
    }

    // W_partial_eval = final_eval / h_eval... no, check product = final_eval
    // The sumcheck final_eval = H(r') * W_partial(r')
    // We know H(r'), and W_partial(r') should equal W(r_kw', r_kh', r_c, r_d) = w_claim.eval
    let expected_final = ext2_mul(h_eval, w_claim.eval);
    if expected_final != sumcheck_proofs[0].final_eval {
        println!("FlattenKernel final eval check failed: expected {:?}, got {:?}", expected_final, sumcheck_proofs[0].final_eval);
        return false;
    }

    true
}

// ============================================================================
// Conv1D prove
// ============================================================================

fn conv1d_prove(
    conv: &Conv1D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    // edge_ids: [x_edge, w_edge, y_edge]
    // witnesses: [X, W, Y]
    let x_edge = edge_ids[0];
    let w_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_lo = conv.l_lo();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let l_out_pad = conv.l_out.next_power_of_two();
    let s_in_pad = conv.s_in;
    let k_pad = conv.kernel_size.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();

    // Parse claim point: Y shape [C_out, L_out]
    // little-endian: l_out bits (lowest) | c_out bits
    let r_lo = &out_claim.point[..l_lo];
    let r_d = &out_claim.point[l_lo..l_lo + l_d];

    // ---- Sample α ----
    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // Build eq_D table
    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d] for each output position k
    let y_data = witnesses[2];
    let mut yp = vec![GoldilocksExt2::zero(); l_out_pad];
    for d in 0..conv.c_out {
        for lo in 0..conv.l_out {
            let y_idx = lo + d * l_out_pad;
            let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
            yp[lo] = ext2_add(yp[lo], ext2_mul(eq_d[d], y_val));
        }
    }

    // ---- Sumcheck 1: eq-sumcheck to reduce output spatial ----
    let eq_lo = evaluate_lagrange_basis_ext2(r_lo);

    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_lo, 2, transcript);
    let proof1 = prover1.prove(&mut [eq_lo, yp.clone()].as_mut_slice(), transcript);
    let r_lo_new = prover1.challenges.clone();

    // Self-claim on Y
    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_lo + l_d);
    y_self_point.extend_from_slice(&r_lo_new);
    y_self_point.extend_from_slice(r_d);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Channel F×G ----
    // F[c] = Σ_i X_rev[c,i]·α^i, G[c] = Σ_k W_partial[c,k]·α^k

    // Build WP[c, k] = Σ_d W[d, c, k] · eq_D[d]
    let w_data = witnesses[1];
    let mut wp = vec![GoldilocksExt2::zero(); c_in_pad * k_pad];
    for d in 0..conv.c_out {
        for c in 0..conv.c_in {
            for k in 0..conv.kernel_size {
                let w_idx = k + c * k_pad + d * k_pad * c_in_pad;
                let w_val = GoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                wp[c * k_pad + k] = ext2_add(wp[c * k_pad + k], ext2_mul(eq_d[d], w_val));
            }
        }
    }

    // Build G[c] = Σ_k WP[c, k] · α^k
    let alpha_kernel = alpha_power_table(alpha, k_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for k in 0..conv.kernel_size {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * k_pad + k], alpha_kernel[k]));
        }
    }

    // Build F[c] = Σ_i X_rev[c, i] · α^i
    // X_rev[c, i] = X[c, s_in - 1 - i]
    let x_data = witnesses[0];
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for il in 0..conv.input_len {
            let x_idx = il + c * s_in_pad;
            let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            let rev_i = conv.s_in - 1 - il;
            f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
        }
    }

    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_in_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(f_poly[c], g_poly[c]));
    }

    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F reduction to X claim ----
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![GoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for il in 0..conv.input_len {
            let x_idx = il + c * s_in_pad;
            let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            let rev_i = conv.s_in - 1 - il;
            xp[rev_i] = ext2_add(xp[rev_i], ext2_mul(eq_c[c], x_val));
        }
    }

    let alpha_poly_in = alpha_power_table(alpha, s_in_pad);

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_poly_in, xp].as_mut_slice(), transcript);
    let r_i = prover3.challenges.clone();

    // X_rev(r_c, r_i) = X(r_c, 1 - r_i)
    let one = GoldilocksExt2::one();
    let r_spatial_x: Vec<GoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_spatial_x);
    x_point.extend_from_slice(&r_c);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G reduction to W claim ----
    let mut wpp = vec![GoldilocksExt2::zero(); k_pad];
    for c in 0..conv.c_in {
        for k in 0..conv.kernel_size {
            wpp[k] = ext2_add(wpp[k], ext2_mul(eq_c[c], wp[c * k_pad + k]));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, k_pad);

    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wpp].as_mut_slice(), transcript);
    let r_k_new = prover4.challenges.clone();

    // W point: (r_k, r_c, r_d) in little-endian order
    let mut w_point = Vec::with_capacity(l_kernel + l_c + l_d);
    w_point.extend_from_slice(&r_k_new);
    w_point.extend_from_slice(&r_c);
    w_point.extend_from_slice(r_d);

    let w_eval = w_data.data.as_ref().unwrap().evaluate_at_point_ext2(&w_point);

    let w_claim = Claim {
        edge_id: w_edge,
        sparse_id: 0,
        point: w_point,
        eval: w_eval,
    };

    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, w_claim, s_alpha_claim])
}

// ============================================================================
// Conv1D verify
// ============================================================================

fn conv1d_verify(
    conv: &Conv1D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    // claims layout: [y_self_claim, x_claim, w_claim, s_alpha_claim, out_claim]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let w_claim = claims[2];

    let l_lo = conv.l_lo();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let r_lo = &out_claim.point[..l_lo];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // ---- Verify Sumcheck 1: eq-sumcheck ----
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0],
        v,
        l_lo,
        2,
        transcript,
    );
    if !ok1 {
        println!("Conv1D sumcheck 1 verification failed");
        return false;
    }

    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_lo {
            let a = r_lo[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("Conv1D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2: F×G ----
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1],
        s_alpha_conv,
        l_c,
        2,
        transcript,
    );
    if !ok2 {
        println!("Conv1D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = ext2_add(
        sumcheck_proofs[2].round_messages[0][0],
        sumcheck_proofs[2].round_messages[0][1],
    );

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[2],
        inferred_sum_3,
        l_spatial_in,
        2,
        transcript,
    );
    if !ok3 {
        println!("Conv1D sumcheck 3 verification failed");
        return false;
    }

    let alpha_mle_3 = alpha_table_mle_eval(alpha, &challenges3);
    let expected_final_3 = ext2_mul(alpha_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[2].final_eval {
        println!("Conv1D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (kernel_size=1: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("Conv1D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3],
        inferred_sum_4,
        l_kernel,
        2,
        transcript,
    );
    if !ok4 {
        println!("Conv1D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, w_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("Conv1D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// Conv2D prove
// ============================================================================

fn conv2d_prove(
    conv: &Conv2D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    // edge_ids: [x_edge, wf_edge, y_edge]
    // witnesses: [X, W_flat, Y]
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let s_in_pad = conv.s_in.next_power_of_two();
    let s_kernel_pad = conv.s_kernel.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();
    let w_in_pad = conv.input_w.next_power_of_two();
    let h_in_pad = conv.input_h.next_power_of_two();
    let w_out_pad = conv.w_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let s_out_pad = w_out_pad * h_out_pad;

    // Parse claim point: Y shape [C_out, H_out, W_out]
    // little-endian: w_out bits (lowest) | h_out bits | c_out bits
    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];

    // ---- Sample α ----
    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // Build eq_D table
    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d] for each spatial position k
    let y_data = witnesses[2]; // Y witness
    let mut yp = vec![GoldilocksExt2::zero(); s_out_pad];
    for d in 0..conv.c_out {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let y_idx = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
                let k = wo + ho * w_out_pad; // spatial index (little-endian)
                yp[k] = ext2_add(yp[k], ext2_mul(eq_d[d], y_val));
            }
        }
    }

    // ---- Sumcheck 1: eq-sumcheck to reduce output spatial ----
    // Prove: Σ_k eq(r_spatial, k) · YP[k] = v
    let eq_spatial = evaluate_lagrange_basis_ext2(r_spatial);

    let poly1_sc1 = eq_spatial;
    let poly2_sc1 = yp.clone();

    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof1 = prover1.prove(&mut [poly1_sc1, poly2_sc1].as_mut_slice(), transcript);
    let r_spatial_new = prover1.challenges.clone();

    // Self-claim: Y(r_d, r_spatial_new) = YP(r_spatial_new) = prover1.final_eval(1)
    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_spatial_out + l_d);
    y_self_point.extend_from_slice(&r_spatial_new);
    y_self_point.extend_from_slice(r_d);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Channel F×G ----
    // F[c] = Σ_i X_rev[c,i]·α^i,  G[c] = Σ_j WP[c,j]·α^j
    // Σ_c F[c]·G[c] = s_alpha_conv (prover-computed, sent to transcript)

    // Build WP[c, j] = Σ_d W_flat[d, c, j] · eq_D[d]
    let wf_data = witnesses[1]; // W_flat witness
    let mut wp = vec![GoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for d in 0..conv.c_out {
        for c in 0..conv.c_in {
            for j in 0..conv.s_kernel {
                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                let wf_val = GoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    // Build G[c] = Σ_j WP[c, j] · α^j
    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    // Build F[c] = Σ_i X_rev[c, i] · α^i
    // X_rev[c, i] = X[c, S_in - 1 - i]
    let x_data = witnesses[0]; // X witness
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for ih in 0..conv.input_h {
            for iw in 0..conv.input_w {
                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let i_flat = ih * conv.stride_w + iw;
                let rev_i = conv.s_in - 1 - i_flat;
                f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
            }
        }
    }

    // Verify: Σ_c F[c]·G[c] should match the α-weighted convolution sum
    // The relationship: Σ_c F[c]·G[c] = Σ_m YP_conv[m] · α^m
    // where m ranges over all valid 1D conv outputs.
    // For the output values: Y[d, ho, wo] contributes to m = (S_in-1) - (ho*W_in + wo)
    // So S_α_conv = Σ_c F[c]·G[c] = Σ_{ho,wo} YP[ho,wo] · α^{(S_in-1)-(ho*W_in+wo)}
    //
    // We need to use this as the expected sum for sumcheck 2.
    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_in_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(f_poly[c], g_poly[c]));
    }

    // Append s_alpha_conv to transcript (prover-computed, verifier reads from proof)
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    let _claim_f = prover2.final_eval(0);
    let _claim_g = prover2.final_eval(1);

    // ---- Sumcheck 3: F reduction to X claim ----
    // F(r_c) = claim_f
    // F[c] = Σ_i X_rev[c, i] · α^i
    // XP[i] = Σ_c X_rev[c, i] · eq(r_c, c)
    // Prove: Σ_i α^i · XP[i] = claim_f

    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![GoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for ih in 0..conv.input_h {
            for iw in 0..conv.input_w {
                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let i_flat = ih * conv.stride_w + iw;
                let rev_i = conv.s_in - 1 - i_flat;
                xp[rev_i] = ext2_add(xp[rev_i], ext2_mul(eq_c[c], x_val));
            }
        }
    }

    let alpha_poly_in = alpha_power_table(alpha, s_in_pad);

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_poly_in, xp].as_mut_slice(), transcript);
    let r_i = prover3.challenges.clone();

    // XP(r_i) = X_rev(r_c, r_i) = X(r_c, 1 - r_i)
    // So claim on X at point (1-r_i, r_c) in terms of X's layout [C_in, H_in, W_in]
    // X's little-endian: w_in bits | h_in bits | c_in bits
    // r_i corresponds to spatial_in bits in little-endian
    // But X_rev reverses the spatial index, so:
    // X(r_c, 1-r_i) means evaluate X's MLE at spatial point (1-r_i[0], 1-r_i[1], ...)
    let one = GoldilocksExt2::one();
    let r_spatial_x: Vec<GoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_spatial_x);
    x_point.extend_from_slice(&r_c);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G reduction to W_flat claim ----
    // G(r_c) = claim_g
    // G[c] = Σ_j WP[c, j] · α^j
    // WPP[j] = Σ_c eq(r_c, c) · WP[c, j] = Σ_d Σ_c eq_D[d] · eq_C[c] · W_flat[d, c, j]
    // Prove: Σ_j α^j · WPP[j] = claim_g

    let mut wpp = vec![GoldilocksExt2::zero(); s_kernel_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            wpp[j] = ext2_add(wpp[j], ext2_mul(eq_c[c], wp[c * s_kernel_pad + j]));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, s_kernel_pad);

    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wpp].as_mut_slice(), transcript);
    let r_j_new = prover4.challenges.clone();

    // WPP(r_j) = W_flat(r_d, r_c, r_j) — direct MLE evaluation
    let mut wf_point = Vec::with_capacity(l_kernel + l_c + l_d);
    wf_point.extend_from_slice(&r_j_new);
    wf_point.extend_from_slice(&r_c);
    wf_point.extend_from_slice(r_d);

    let wf_eval = wf_data.data.as_ref().unwrap().evaluate_at_point_ext2(&wf_point);

    let wf_claim = Claim {
        edge_id: wf_edge,
        sparse_id: 0,
        point: wf_point,
        eval: wf_eval,
    };

    // Carrier claim for s_alpha_conv (verifier reads from claims[3].eval)
    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    // Return: 4 proofs, claims = [y_self_claim, x_claim, wf_claim, s_alpha_claim]
    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, wf_claim, s_alpha_claim])
}

// ============================================================================
// Conv2D verify
// ============================================================================

fn conv2d_verify(
    conv: &Conv2D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    // claims layout: [y_self_claim, x_claim, wf_claim, s_alpha_claim, out_claim]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];
    // claims[3] = s_alpha_claim (carrier for s_alpha_conv value)

    let l_spatial_out = conv.l_spatial_out();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let r_spatial = &out_claim.point[..l_spatial_out];
    let v = out_claim.eval;

    // Sample α (must match prover)
    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // ---- Verify Sumcheck 1: eq-sumcheck ----
    // Σ_k eq(r_spatial, k) · YP[k] = v
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0],
        v,
        l_spatial_out,
        2,
        transcript,
    );
    if !ok1 {
        println!("Conv2D sumcheck 1 verification failed");
        return false;
    }

    // Check final eval: eq(r_spatial, r') · YP(r') = final_eval
    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("Conv2D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2: F×G ----
    // s_alpha_conv is carried as claims[3].eval (a dummy claim used for transport)
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1],
        s_alpha_conv,
        l_c,
        2,
        transcript,
    );
    if !ok2 {
        println!("Conv2D sumcheck 2 verification failed");
        return false;
    }

    // Check final eval of sumcheck 2:
    // final_eval = F(r_c) * G(r_c) = claim_f * claim_g
    // We don't have claim_f and claim_g separately at this point.
    // But the sumcheck verifier already checked this via final_eval.
    // The individual values claim_f and claim_g are used by sumchecks 3 and 4.
    // We extract them from sumcheck 3 and 4's expected sums.

    // ---- Verify Sumcheck 3 ----
    // Σ_i α^i · XP[i] = claim_f
    // For 0-round sumcheck (degenerate case), final_eval IS the sum.
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[2].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[2].round_messages[0][0],
            sumcheck_proofs[2].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[2],
        inferred_sum_3,
        l_spatial_in,
        2,
        transcript,
    );
    if !ok3 {
        println!("Conv2D sumcheck 3 verification failed");
        return false;
    }

    // Check final eval of sumcheck 3:
    // final_eval = α_table_mle(r_i) * XP(r_i)
    let alpha_mle_3 = alpha_table_mle_eval(alpha, &challenges3);
    let expected_final_3 = ext2_mul(alpha_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[2].final_eval {
        println!("Conv2D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // Σ_j α^j · WPP[j] = claim_g
    // For 0-round sumcheck (1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    // Cross-check: claim_f * claim_g = sumcheck 2's final eval
    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("Conv2D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3],
        inferred_sum_4,
        l_kernel,
        2,
        transcript,
    );
    if !ok4 {
        println!("Conv2D sumcheck 4 verification failed");
        return false;
    }

    // Check final eval of sumcheck 4:
    // final_eval = α_table_mle(r_j) * WPP(r_j)
    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("Conv2D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// FlattenKernel3D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct FlattenKernel3D {
    pub stride_h: usize,  // H_pad * W_pad
    pub stride_w: usize,  // W_pad
    pub kd: usize,
    pub kh: usize,
    pub kw: usize,
    pub c_out: usize,
    pub c_in: usize,
}

impl BasicBlock for FlattenKernel3D {
    /// run(): Scatter W[C_out, C_in, kD, kH, kW] → W_flat[C_out, C_in, S_kernel_pad]
    /// where S_kernel = (kD-1)*stride_h + (kH-1)*stride_w + kW, j = kd*stride_h + kh*stride_w + kw.
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let w = inputs[0];
        let c_out_pad = self.c_out.next_power_of_two();
        let c_in_pad = self.c_in.next_power_of_two();
        let kd_pad = self.kd.next_power_of_two();
        let kh_pad = self.kh.next_power_of_two();
        let kw_pad = self.kw.next_power_of_two();
        let s_kernel = (self.kd - 1) * self.stride_h + (self.kh - 1) * self.stride_w + self.kw;
        let s_kernel_pad = s_kernel.next_power_of_two();

        let out_size = c_out_pad * c_in_pad * s_kernel_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for d in 0..self.c_out {
            for c in 0..self.c_in {
                for kd in 0..self.kd {
                    for kh in 0..self.kh {
                        for kw in 0..self.kw {
                            // W little-endian: kw bits | kh bits | kd bits | c_in bits | c_out bits
                            let w_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad
                                + c * kw_pad * kh_pad * kd_pad
                                + d * kw_pad * kh_pad * kd_pad * c_in_pad;
                            let j = kd * self.stride_h + kh * self.stride_w + kw;
                            // W_flat little-endian: j bits | c_in bits | c_out bits
                            let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                            out_data[wf_idx] = w.data.as_ref().unwrap().index(w_idx);
                        }
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let w = inputs[0];
        let c_out_pad = self.c_out.next_power_of_two();
        let c_in_pad = self.c_in.next_power_of_two();
        let kd_pad = self.kd.next_power_of_two();
        let kh_pad = self.kh.next_power_of_two();
        let kw_pad = self.kw.next_power_of_two();
        let s_kernel = (self.kd - 1) * self.stride_h + (self.kh - 1) * self.stride_w + self.kw;
        let s_kernel_pad = s_kernel.next_power_of_two();
        let out_size = c_out_pad * c_in_pad * s_kernel_pad;

        let d_w = w.as_device_buf();
        let mut d_wf = DeviceBuffer::<u64>::new(out_size).expect("FlattenKernel3D: alloc");
        memset_zero(&mut d_wf, out_size).expect("FlattenKernel3D: memset");

        gpu_flatten_kernel3d(
            &d_w, &mut d_wf,
            self.c_out, self.c_in, self.kd, self.kh, self.kw,
            kw_pad, kh_pad, kd_pad,
            c_in_pad, s_kernel_pad,
            self.stride_h, self.stride_w,
        ).expect("FlattenKernel3D: gpu kernel failed");

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        vec![Witness::new_device(out_shape, Arc::new(d_wf), DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        flatten_kernel_3d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        flatten_kernel_3d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// FlattenKernel3D prove/verify
// ============================================================================

fn flatten_kernel_3d_prove(
    fk: &FlattenKernel3D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    let out_claim = out_claims[0];
    let w_edge = edge_ids[0];

    let l_d = log2_ceil(fk.c_out.max(1));
    let l_c = log2_ceil(fk.c_in.max(1));
    let l_kd = log2_ceil(fk.kd.max(1));
    let l_kh = log2_ceil(fk.kh.max(1));
    let l_kw = log2_ceil(fk.kw.max(1));
    let s_kernel = (fk.kd - 1) * fk.stride_h + (fk.kh - 1) * fk.stride_w + fk.kw;
    let l_j = log2_ceil(s_kernel.max(1));

    let r_j = &out_claim.point[..l_j];
    let r_c = &out_claim.point[l_j..l_j + l_c];
    let r_d = &out_claim.point[l_j + l_c..l_j + l_c + l_d];

    let eq_j = evaluate_lagrange_basis_ext2(r_j);

    let w_data = witnesses[0];
    let kd_pad = fk.kd.next_power_of_two();
    let kh_pad = fk.kh.next_power_of_two();
    let kw_pad = fk.kw.next_power_of_two();
    let c_in_pad = fk.c_in.next_power_of_two();

    let eq_c = evaluate_lagrange_basis_ext2(r_c);
    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // W_partial[kd, kh, kw] = Σ_d Σ_c eq_D(d) · eq_C(c) · W[d, c, kd, kh, kw]
    let sumcheck_size = kd_pad * kh_pad * kw_pad;
    let mut w_partial = vec![GoldilocksExt2::zero(); sumcheck_size];
    for dd in 0..fk.c_out {
        for c in 0..fk.c_in {
            let dc_weight = ext2_mul(eq_d[dd], eq_c[c]);
            for kd in 0..fk.kd {
                for kh in 0..fk.kh {
                    for kw in 0..fk.kw {
                        let w_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad
                            + c * kw_pad * kh_pad * kd_pad
                            + dd * kw_pad * kh_pad * kd_pad * c_in_pad;
                        let w_val = GoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                        let sc_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad;
                        w_partial[sc_idx] = ext2_add(w_partial[sc_idx], ext2_mul(dc_weight, w_val));
                    }
                }
            }
        }
    }

    // H[kd, kh, kw] = eq(r_j, kd*stride_h + kh*stride_w + kw)
    let mut h_poly = vec![GoldilocksExt2::zero(); sumcheck_size];
    for kd in 0..fk.kd {
        for kh in 0..fk.kh {
            for kw in 0..fk.kw {
                let j = kd * fk.stride_h + kh * fk.stride_w + kw;
                if j < eq_j.len() {
                    let sc_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad;
                    h_poly[sc_idx] = eq_j[j];
                }
            }
        }
    }

    let num_rounds = l_kd + l_kh + l_kw;
    let mut prover = CpuLinearSumcheckProverExt2::new(num_rounds, 2, transcript);
    let proof = prover.prove(&mut [h_poly, w_partial].as_mut_slice(), transcript);

    let challenges = &prover.challenges;
    let r_kw_new = &challenges[..l_kw];
    let r_kh_new = &challenges[l_kw..l_kw + l_kh];
    let r_kd_new = &challenges[l_kw + l_kh..];

    // Build claim on W at point (r_kw', r_kh', r_kd', r_c, r_d)
    let mut w_point = Vec::with_capacity(l_kw + l_kh + l_kd + l_c + l_d);
    w_point.extend_from_slice(r_kw_new);
    w_point.extend_from_slice(r_kh_new);
    w_point.extend_from_slice(r_kd_new);
    w_point.extend_from_slice(r_c);
    w_point.extend_from_slice(r_d);

    let w_eval = witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&w_point);

    let w_claim = Claim {
        edge_id: w_edge,
        sparse_id: 0,
        point: w_point,
        eval: w_eval,
    };

    (vec![proof], vec![w_claim])
}

fn flatten_kernel_3d_verify(
    fk: &FlattenKernel3D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    let out_claim = claims.last().unwrap();
    let w_claim = &claims[0];

    let l_kd = log2_ceil(fk.kd.max(1));
    let l_kh = log2_ceil(fk.kh.max(1));
    let l_kw = log2_ceil(fk.kw.max(1));
    let s_kernel = (fk.kd - 1) * fk.stride_h + (fk.kh - 1) * fk.stride_w + fk.kw;
    let l_j = log2_ceil(s_kernel.max(1));

    let r_j = &out_claim.point[..l_j];

    let num_rounds = l_kd + l_kh + l_kw;
    let (ok, challenges) = SumcheckVerifier::verify(
        sumcheck_proofs[0],
        out_claim.eval,
        num_rounds,
        2,
        transcript,
    );
    if !ok {
        println!("FlattenKernel3D sumcheck verification failed");
        return false;
    }

    let r_kw_new = &challenges[..l_kw];
    let r_kh_new = &challenges[l_kw..l_kw + l_kh];
    let r_kd_new = &challenges[l_kw + l_kh..];

    let eq_kw = evaluate_lagrange_basis_ext2(r_kw_new);
    let eq_kh = evaluate_lagrange_basis_ext2(r_kh_new);
    let eq_kd = evaluate_lagrange_basis_ext2(r_kd_new);
    let eq_j_table = evaluate_lagrange_basis_ext2(r_j);

    let mut h_eval = GoldilocksExt2::zero();
    for kd in 0..fk.kd {
        for kh in 0..fk.kh {
            for kw in 0..fk.kw {
                let j = kd * fk.stride_h + kh * fk.stride_w + kw;
                if j < eq_j_table.len() {
                    h_eval = ext2_add(h_eval,
                        ext2_mul(ext2_mul(ext2_mul(eq_kd[kd], eq_kh[kh]), eq_kw[kw]), eq_j_table[j]));
                }
            }
        }
    }

    let expected_final = ext2_mul(h_eval, w_claim.eval);
    if expected_final != sumcheck_proofs[0].final_eval {
        println!("FlattenKernel3D final eval check failed: expected {:?}, got {:?}", expected_final, sumcheck_proofs[0].final_eval);
        return false;
    }

    true
}

// ============================================================================
// Conv3D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct Conv3D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_d: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub input_d: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub stride_h: usize,  // H_pad * W_pad
    pub stride_w: usize,  // W_pad
    pub s_in: usize,      // D_pad * H_pad * W_pad
    pub s_kernel: usize,  // (kD-1)*stride_h + (kH-1)*stride_w + kW
    pub d_out: usize,
    pub h_out: usize,
    pub w_out: usize,
    pub conv_stride_d: usize, // convolution stride depth (default 1)
    pub conv_stride_h: usize, // convolution stride height (default 1)
    pub conv_stride_w: usize, // convolution stride width (default 1)
}

impl Conv3D {
    pub fn new(c_in: usize, c_out: usize, kernel_d: usize, kernel_h: usize, kernel_w: usize,
               input_d: usize, input_h: usize, input_w: usize) -> Self {
        Self::new_strided(c_in, c_out, kernel_d, kernel_h, kernel_w, input_d, input_h, input_w, 1, 1, 1)
    }

    pub fn new_strided(c_in: usize, c_out: usize, kernel_d: usize, kernel_h: usize, kernel_w: usize,
                       input_d: usize, input_h: usize, input_w: usize,
                       conv_stride_d: usize, conv_stride_h: usize, conv_stride_w: usize) -> Self {
        let d_out = (input_d - kernel_d) / conv_stride_d + 1;
        let h_out = (input_h - kernel_h) / conv_stride_h + 1;
        let w_out = (input_w - kernel_w) / conv_stride_w + 1;
        let w_pad = input_w.next_power_of_two();
        let h_pad = input_h.next_power_of_two();
        let d_pad = input_d.next_power_of_two();
        let stride_w = w_pad;
        let stride_h = h_pad * w_pad;
        let s_in = d_pad * h_pad * w_pad;
        let s_kernel = (kernel_d - 1) * stride_h + (kernel_h - 1) * stride_w + kernel_w;
        Self { c_in, c_out, kernel_d, kernel_h, kernel_w, input_d, input_h, input_w,
               stride_h, stride_w, s_in, s_kernel, d_out, h_out, w_out,
               conv_stride_d, conv_stride_h, conv_stride_w }
    }

    fn l_d(&self) -> usize { log2_ceil(self.c_out.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.c_in.max(1)) }
    fn l_wo(&self) -> usize { log2_ceil(self.w_out.max(1)) }
    fn l_ho(&self) -> usize { log2_ceil(self.h_out.max(1)) }
    fn l_do(&self) -> usize { log2_ceil(self.d_out.max(1)) }
    fn l_spatial_out(&self) -> usize { self.l_wo() + self.l_ho() + self.l_do() }
    fn l_spatial_in(&self) -> usize {
        log2_ceil(self.input_w.max(1)) + log2_ceil(self.input_h.max(1)) + log2_ceil(self.input_d.max(1))
    }
    fn l_kernel(&self) -> usize { log2_ceil(self.s_kernel.max(1)) }
}

impl BasicBlock for Conv3D {
    /// run(): Conv3D. X[C_in, D_in, H_in, W_in], W_flat[C_out, C_in, S_kernel] → Y[C_out, D_out, H_out, W_out]
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_in_pad = self.c_in.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let d_in_pad = self.input_d.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let d_out_pad = self.d_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();

        let out_size = c_out_pad * d_out_pad * h_out_pad * w_out_pad;

        // Parallelize over output elements (d, do_, ho, wo)
        let x_data = x.data.as_ref().unwrap();
        let w_data = w_flat.data.as_ref().unwrap();
        let c_out = self.c_out;
        let d_out = self.d_out;
        let h_out = self.h_out;
        let w_out = self.w_out;
        let c_in = self.c_in;
        let kernel_d = self.kernel_d;
        let kernel_h = self.kernel_h;
        let kernel_w = self.kernel_w;
        let conv_stride_d = self.conv_stride_d;
        let conv_stride_h = self.conv_stride_h;
        let conv_stride_w = self.conv_stride_w;
        let stride_h_val = self.stride_h;
        let stride_w_val = self.stride_w;

        let total_outputs = c_out * d_out * h_out * w_out;
        let mut out_data = vec![GoldilocksField(0); out_size];
        let results: Vec<(usize, GoldilocksField)> = (0..total_outputs)
            .into_par_iter()
            .map(|flat_idx| {
                let wo = flat_idx % w_out;
                let ho = (flat_idx / w_out) % h_out;
                let do_ = (flat_idx / (w_out * h_out)) % d_out;
                let d = flat_idx / (w_out * h_out * d_out);
                let mut acc = GoldilocksField(0);
                for c in 0..c_in {
                    for kd in 0..kernel_d {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let id = do_ * conv_stride_d + kd;
                                let ih = ho * conv_stride_h + kh;
                                let iw = wo * conv_stride_w + kw;
                                let x_idx = iw + ih * w_in_pad + id * w_in_pad * h_in_pad
                                    + c * w_in_pad * h_in_pad * d_in_pad;
                                let j = kd * stride_h_val + kh * stride_w_val + kw;
                                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                                let x_val = x_data.index(x_idx);
                                let w_val = w_data.index(wf_idx);
                                acc = gl_add(acc, gl_mul(x_val, w_val));
                            }
                        }
                    }
                }
                let out_idx = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad
                    + d * w_out_pad * h_out_pad * d_out_pad;
                (out_idx, acc)
            })
            .collect();
        for (idx, val) in results {
            out_data[idx] = val;
        }

        let out_shape = vec![self.c_out, self.d_out, self.h_out, self.w_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_in_pad = self.c_in.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let d_in_pad = self.input_d.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let d_out_pad = self.d_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let out_size = c_out_pad * d_out_pad * h_out_pad * w_out_pad;

        let d_x = x.as_device_buf();
        let d_w = w_flat.as_device_buf();
        let mut d_y = DeviceBuffer::<u64>::new(out_size).expect("Conv3D: alloc out");
        memset_zero(&mut d_y, out_size).expect("Conv3D: memset zero failed");

        gpu_conv3d(
            &d_x, &d_w, &mut d_y,
            self.c_out, self.d_out, self.h_out, self.w_out,
            self.c_in, self.kernel_d, self.kernel_h, self.kernel_w,
            self.conv_stride_d, self.conv_stride_h, self.conv_stride_w,
            w_in_pad, h_in_pad, d_in_pad,
            c_in_pad, s_kernel_pad,
            w_out_pad, h_out_pad, d_out_pad,
            self.stride_h, self.stride_w,
        ).expect("Conv3D: gpu kernel failed");

        let out_shape = vec![self.c_out, self.d_out, self.h_out, self.w_out];
        vec![Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        conv3d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        conv3d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// Conv3D prove
// ============================================================================

fn conv3d_prove(
    conv: &Conv3D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    // edge_ids: [x_edge, wf_edge, y_edge]
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let w_out_pad = conv.w_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let d_out_pad = conv.d_out.next_power_of_two();
    let s_out_pad = w_out_pad * h_out_pad * d_out_pad;
    let w_in_pad = conv.input_w.next_power_of_two();
    let h_in_pad = conv.input_h.next_power_of_two();
    let d_in_pad = conv.input_d.next_power_of_two();
    let s_in_pad = conv.s_in.next_power_of_two();
    let s_kernel_pad = conv.s_kernel.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();

    // Parse claim point: Y shape [C_out, D_out, H_out, W_out]
    // little-endian: w_out | h_out | d_out | c_out
    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d]
    let y_data = witnesses[2];
    let mut yp = vec![GoldilocksExt2::zero(); s_out_pad];
    for d in 0..conv.c_out {
        for do_ in 0..conv.d_out {
            for ho in 0..conv.h_out {
                for wo in 0..conv.w_out {
                    let y_idx = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad
                        + d * w_out_pad * h_out_pad * d_out_pad;
                    let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
                    let k = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad;
                    yp[k] = ext2_add(yp[k], ext2_mul(eq_d[d], y_val));
                }
            }
        }
    }

    // ---- Sumcheck 1: eq-sumcheck ----
    let eq_spatial = evaluate_lagrange_basis_ext2(r_spatial);

    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof1 = prover1.prove(&mut [eq_spatial, yp.clone()].as_mut_slice(), transcript);
    let r_spatial_new = prover1.challenges.clone();

    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_spatial_out + l_d);
    y_self_point.extend_from_slice(&r_spatial_new);
    y_self_point.extend_from_slice(r_d);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Channel F×G ----
    let wf_data = witnesses[1];
    let mut wp = vec![GoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for d in 0..conv.c_out {
        for c in 0..conv.c_in {
            for j in 0..conv.s_kernel {
                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                let wf_val = GoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    let x_data = witnesses[0];
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for id in 0..conv.input_d {
            for ih in 0..conv.input_h {
                for iw in 0..conv.input_w {
                    let x_idx = iw + ih * w_in_pad + id * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let i_flat = id * conv.stride_h + ih * conv.stride_w + iw;
                    let rev_i = conv.s_in - 1 - i_flat;
                    f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
                }
            }
        }
    }

    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_in_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(f_poly[c], g_poly[c]));
    }

    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F → X ----
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![GoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for id in 0..conv.input_d {
            for ih in 0..conv.input_h {
                for iw in 0..conv.input_w {
                    let x_idx = iw + ih * w_in_pad + id * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let i_flat = id * conv.stride_h + ih * conv.stride_w + iw;
                    let rev_i = conv.s_in - 1 - i_flat;
                    xp[rev_i] = ext2_add(xp[rev_i], ext2_mul(eq_c[c], x_val));
                }
            }
        }
    }

    let alpha_poly_in = alpha_power_table(alpha, s_in_pad);

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_poly_in, xp].as_mut_slice(), transcript);
    let r_i = prover3.challenges.clone();

    let one = GoldilocksExt2::one();
    let r_spatial_x: Vec<GoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_spatial_x);
    x_point.extend_from_slice(&r_c);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G → W_flat ----
    let mut wpp = vec![GoldilocksExt2::zero(); s_kernel_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            wpp[j] = ext2_add(wpp[j], ext2_mul(eq_c[c], wp[c * s_kernel_pad + j]));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, s_kernel_pad);

    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wpp].as_mut_slice(), transcript);
    let r_j_new = prover4.challenges.clone();

    let mut wf_point = Vec::with_capacity(l_kernel + l_c + l_d);
    wf_point.extend_from_slice(&r_j_new);
    wf_point.extend_from_slice(&r_c);
    wf_point.extend_from_slice(r_d);

    let wf_eval = wf_data.data.as_ref().unwrap().evaluate_at_point_ext2(&wf_point);

    let wf_claim = Claim {
        edge_id: wf_edge,
        sparse_id: 0,
        point: wf_point,
        eval: wf_eval,
    };

    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, wf_claim, s_alpha_claim])
}

// ============================================================================
// Conv3D verify
// ============================================================================

fn conv3d_verify(
    conv: &Conv3D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];

    let l_spatial_out = conv.l_spatial_out();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let r_spatial = &out_claim.point[..l_spatial_out];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // ---- Verify Sumcheck 1 ----
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0], v, l_spatial_out, 2, transcript,
    );
    if !ok1 {
        println!("Conv3D sumcheck 1 verification failed");
        return false;
    }

    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("Conv3D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2 ----
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("Conv3D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = ext2_add(
        sumcheck_proofs[2].round_messages[0][0],
        sumcheck_proofs[2].round_messages[0][1],
    );

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[2], inferred_sum_3, l_spatial_in, 2, transcript,
    );
    if !ok3 {
        println!("Conv3D sumcheck 3 verification failed");
        return false;
    }

    let alpha_mle_3 = alpha_table_mle_eval(alpha, &challenges3);
    let expected_final_3 = ext2_mul(alpha_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[2].final_eval {
        println!("Conv3D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("Conv3D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("Conv3D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("Conv3D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// ConvTranspose1D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct ConvTranspose1D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_size: usize,
    pub input_len: usize,
    pub stride: usize,
    pub s_in: usize,      // input_len.next_power_of_two()
    pub l_out: usize,     // (input_len - 1) * stride + kernel_size
}

impl ConvTranspose1D {
    pub fn new(c_in: usize, c_out: usize, kernel_size: usize, input_len: usize, stride: usize) -> Self {
        let l_out = (input_len - 1) * stride + kernel_size;
        let s_in = input_len.next_power_of_two();
        Self { c_in, c_out, kernel_size, input_len, stride, s_in, l_out }
    }

    fn l_d(&self) -> usize { log2_ceil(self.c_out.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.c_in.max(1)) }
    fn l_lo(&self) -> usize { log2_ceil(self.l_out.max(1)) }
    fn l_spatial_in(&self) -> usize { log2_ceil(self.input_len.max(1)) }
    fn l_kernel(&self) -> usize { log2_ceil(self.kernel_size.max(1)) }
}

impl BasicBlock for ConvTranspose1D {
    /// run(): ConvTranspose1D. X[C_in, L_in], W[C_in, C_out, K] → Y[C_out, L_out]
    /// where L_out = (L_in - 1) * stride + K.
    /// Y[d, j*stride+k] += X[c, j] * W[c, d, k]
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w = inputs[1];

        let c_out_pad = self.c_out.next_power_of_two();
        let l_in_pad = self.input_len.next_power_of_two();
        let l_out_pad = self.l_out.next_power_of_two();
        let k_pad = self.kernel_size.next_power_of_two();

        let out_size = c_out_pad * l_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for d in 0..self.c_out {
            for j in 0..self.input_len {
                for c in 0..self.c_in {
                    for k in 0..self.kernel_size {
                        let out_pos = j * self.stride + k;
                        // X index: l_in bits (lowest) | c_in bits
                        let x_idx = j + c * l_in_pad;
                        // W index: k bits (lowest) | c_out bits | c_in bits
                        let w_idx = k + d * k_pad + c * k_pad * c_out_pad;
                        let x_val = x.data.as_ref().unwrap().index(x_idx);
                        let w_val = w.data.as_ref().unwrap().index(w_idx);
                        // Y index: l_out bits (lowest) | c_out bits
                        let out_idx = out_pos + d * l_out_pad;
                        out_data[out_idx] = gl_add(out_data[out_idx], gl_mul(x_val, w_val));
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.l_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        conv_transpose1d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        conv_transpose1d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// ConvTranspose1D prove
// ============================================================================

fn conv_transpose1d_prove(
    conv: &ConvTranspose1D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    // edge_ids: [x_edge, w_edge, y_edge]
    let x_edge = edge_ids[0];
    let w_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_lo = conv.l_lo();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let l_out_pad = conv.l_out.next_power_of_two();
    let l_in_pad = conv.input_len.next_power_of_two();
    let k_pad = conv.kernel_size.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();
    let c_out_pad = conv.c_out.next_power_of_two();

    // Parse claim point: Y shape [C_out, L_out]
    // little-endian: l_out bits (lowest) | c_out bits
    let r_lo = &out_claim.point[..l_lo];
    let r_d = &out_claim.point[l_lo..l_lo + l_d];

    // ---- Sample α ----
    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // Build eq_D table
    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d]
    let y_data = witnesses[2];
    let mut yp = vec![GoldilocksExt2::zero(); l_out_pad];
    for d in 0..conv.c_out {
        for lo in 0..conv.l_out {
            let y_idx = lo + d * l_out_pad;
            let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
            yp[lo] = ext2_add(yp[lo], ext2_mul(eq_d[d], y_val));
        }
    }

    // ---- Sumcheck 1: eq-sumcheck to reduce output spatial ----
    let eq_lo = evaluate_lagrange_basis_ext2(r_lo);

    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_lo, 2, transcript);
    let proof1 = prover1.prove(&mut [eq_lo, yp.clone()].as_mut_slice(), transcript);
    let r_lo_new = prover1.challenges.clone();

    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_lo + l_d);
    y_self_point.extend_from_slice(&r_lo_new);
    y_self_point.extend_from_slice(r_d);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Channel F×G ----
    // ConvTranspose: F[c] = Σ_j X[c,j] · β^j where β = α^stride (forward, NO reversal)
    //                G[c] = Σ_k WP[c,k] · α^k
    let beta = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..conv.stride {
            b = ext2_mul(b, alpha);
        }
        b
    }; // β = α^stride

    // Build WP[c, k] = Σ_d W[c, d, k] · eq_D[d]
    // W layout: k bits (lowest) | c_out bits | c_in bits
    let w_data = witnesses[1];
    let mut wp = vec![GoldilocksExt2::zero(); c_in_pad * k_pad];
    for c in 0..conv.c_in {
        for d in 0..conv.c_out {
            for k in 0..conv.kernel_size {
                let w_idx = k + d * k_pad + c * k_pad * c_out_pad;
                let w_val = GoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                wp[c * k_pad + k] = ext2_add(wp[c * k_pad + k], ext2_mul(eq_d[d], w_val));
            }
        }
    }

    // Build G[c] = Σ_k WP[c, k] · α^k
    let alpha_kernel = alpha_power_table(alpha, k_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for k in 0..conv.kernel_size {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * k_pad + k], alpha_kernel[k]));
        }
    }

    // Build F[c] = Σ_j X[c,j] · β^j (forward, strided — NO reversal)
    let x_data = witnesses[0];
    let beta_table = alpha_power_table(beta, l_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.input_len {
            let x_idx = j + c * l_in_pad;
            let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, beta_table[j]));
        }
    }

    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_in_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(f_poly[c], g_poly[c]));
    }

    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F reduction to X claim ----
    // F(r_c) = Σ_j β^j · XP[j]  where XP[j] = Σ_c eq(r_c, c) · X[c, j]
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![GoldilocksExt2::zero(); l_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.input_len {
            let x_idx = j + c * l_in_pad;
            let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            xp[j] = ext2_add(xp[j], ext2_mul(eq_c[c], x_val));
        }
    }

    let beta_poly = alpha_power_table(beta, l_in_pad);

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [beta_poly, xp].as_mut_slice(), transcript);
    let r_j = prover3.challenges.clone();

    // Claim on X at (r_j, r_c) — NO reversal
    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_j);
    x_point.extend_from_slice(&r_c);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G reduction to W claim ----
    // G(r_c) = Σ_k α^k · WPP[k]  where WPP[k] = Σ_c eq(r_c, c) · WP[c, k]
    let mut wpp = vec![GoldilocksExt2::zero(); k_pad];
    for c in 0..conv.c_in {
        for k in 0..conv.kernel_size {
            wpp[k] = ext2_add(wpp[k], ext2_mul(eq_c[c], wp[c * k_pad + k]));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, k_pad);

    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wpp].as_mut_slice(), transcript);
    let r_k_new = prover4.challenges.clone();

    // W point: (r_k, r_d, r_c) in little-endian order
    // W layout: k bits | c_out bits | c_in bits
    let mut w_point = Vec::with_capacity(l_kernel + l_d + l_c);
    w_point.extend_from_slice(&r_k_new);
    w_point.extend_from_slice(r_d);
    w_point.extend_from_slice(&r_c);

    let w_eval = w_data.data.as_ref().unwrap().evaluate_at_point_ext2(&w_point);

    let w_claim = Claim {
        edge_id: w_edge,
        sparse_id: 0,
        point: w_point,
        eval: w_eval,
    };

    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, w_claim, s_alpha_claim])
}

// ============================================================================
// ConvTranspose1D verify
// ============================================================================

fn conv_transpose1d_verify(
    conv: &ConvTranspose1D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let w_claim = claims[2];

    let l_lo = conv.l_lo();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let r_lo = &out_claim.point[..l_lo];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // β = α^stride
    let beta = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..conv.stride {
            b = ext2_mul(b, alpha);
        }
        b
    };

    // ---- Verify Sumcheck 1: eq-sumcheck ----
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0], v, l_lo, 2, transcript,
    );
    if !ok1 {
        println!("ConvTranspose1D sumcheck 1 verification failed");
        return false;
    }

    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_lo {
            let a = r_lo[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("ConvTranspose1D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2: F×G ----
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("ConvTranspose1D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = ext2_add(
        sumcheck_proofs[2].round_messages[0][0],
        sumcheck_proofs[2].round_messages[0][1],
    );

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[2], inferred_sum_3, l_spatial_in, 2, transcript,
    );
    if !ok3 {
        println!("ConvTranspose1D sumcheck 3 verification failed");
        return false;
    }

    // Check: β-table MLE at challenges3 * x_claim.eval = final_eval
    let beta_mle_3 = alpha_table_mle_eval(beta, &challenges3);
    let expected_final_3 = ext2_mul(beta_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[2].final_eval {
        println!("ConvTranspose1D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("ConvTranspose1D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("ConvTranspose1D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, w_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("ConvTranspose1D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// ConvTranspose2D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct ConvTranspose2D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub flat_stride: usize, // W_out_pad (1D flat stride for kernel)
    pub s_kernel: usize,    // (kH-1)*flat_stride + kW
    pub h_out: usize,
    pub w_out: usize,
}

impl ConvTranspose2D {
    pub fn new(c_in: usize, c_out: usize, kernel_h: usize, kernel_w: usize,
               input_h: usize, input_w: usize, stride_h: usize, stride_w: usize) -> Self {
        let h_out = (input_h - 1) * stride_h + kernel_h;
        let w_out = (input_w - 1) * stride_w + kernel_w;
        let w_out_pad = w_out.next_power_of_two();
        let flat_stride = w_out_pad;
        let s_kernel = (kernel_h - 1) * flat_stride + kernel_w;
        Self { c_in, c_out, kernel_h, kernel_w, input_h, input_w,
               stride_h, stride_w, flat_stride, s_kernel, h_out, w_out }
    }

    fn l_d(&self) -> usize { log2_ceil(self.c_out.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.c_in.max(1)) }
    fn l_wo(&self) -> usize { log2_ceil(self.w_out.max(1)) }
    fn l_ho(&self) -> usize { log2_ceil(self.h_out.max(1)) }
    fn l_spatial_out(&self) -> usize { self.l_wo() + self.l_ho() }
    fn l_spatial_in(&self) -> usize {
        log2_ceil(self.input_w.max(1)) + log2_ceil(self.input_h.max(1))
    }
    fn l_kernel(&self) -> usize { log2_ceil(self.s_kernel.max(1)) }
}

impl BasicBlock for ConvTranspose2D {
    /// run(): ConvTranspose2D. X[C_in, H_in, W_in], W_flat[C_in, C_out, S_kernel] → Y[C_out, H_out, W_out]
    /// Y[d, jh*stride_h+kh, jw*stride_w+kw] += X[c, jh, jw] * W[c, d, kh, kw]
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_out_pad = self.c_out.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();

        let out_size = c_out_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for d in 0..self.c_out {
            for c in 0..self.c_in {
                for jh in 0..self.input_h {
                    for jw in 0..self.input_w {
                        let x_idx = jw + jh * w_in_pad + c * w_in_pad * h_in_pad;
                        let x_val = x.data.as_ref().unwrap().index(x_idx);
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let oh = jh * self.stride_h + kh;
                                let ow = jw * self.stride_w + kw;
                                let j = kh * self.flat_stride + kw;
                                // W_flat: j bits | c_out bits | c_in bits
                                let wf_idx = j + d * s_kernel_pad + c * s_kernel_pad * c_out_pad;
                                let w_val = w_flat.data.as_ref().unwrap().index(wf_idx);
                                let out_idx = ow + oh * w_out_pad + d * w_out_pad * h_out_pad;
                                out_data[out_idx] = gl_add(out_data[out_idx], gl_mul(x_val, w_val));
                            }
                        }
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.h_out, self.w_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        conv_transpose2d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        conv_transpose2d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// ConvTranspose2D prove
// ============================================================================

fn conv_transpose2d_prove(
    conv: &ConvTranspose2D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let w_out_pad = conv.w_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let s_out_pad = w_out_pad * h_out_pad;
    let w_in_pad = conv.input_w.next_power_of_two();
    let h_in_pad = conv.input_h.next_power_of_two();
    let s_in_pad = w_in_pad * h_in_pad;
    let s_kernel_pad = conv.s_kernel.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();
    let c_out_pad = conv.c_out.next_power_of_two();

    // Parse claim point: Y shape [C_out, H_out, W_out]
    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d]
    let y_data = witnesses[2];
    let mut yp = vec![GoldilocksExt2::zero(); s_out_pad];
    for d in 0..conv.c_out {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let y_idx = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
                let k = wo + ho * w_out_pad;
                yp[k] = ext2_add(yp[k], ext2_mul(eq_d[d], y_val));
            }
        }
    }

    // ---- Sumcheck 1: eq-sumcheck ----
    let eq_spatial = evaluate_lagrange_basis_ext2(r_spatial);

    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof1 = prover1.prove(&mut [eq_spatial, yp.clone()].as_mut_slice(), transcript);
    let r_spatial_new = prover1.challenges.clone();

    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_spatial_out + l_d);
    y_self_point.extend_from_slice(&r_spatial_new);
    y_self_point.extend_from_slice(r_d);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Channel F×G ----
    // β_w = α^{stride_w}, β_h = α^{stride_h * flat_stride}
    let beta_w = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride) { b = ext2_mul(b, alpha); }
        b
    };

    // Build WP[c, j] = Σ_d W_flat[c, d, j] · eq_D[d]
    // W_flat layout: j bits | c_out bits | c_in bits
    let wf_data = witnesses[1];
    let mut wp = vec![GoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for c in 0..conv.c_in {
        for d in 0..conv.c_out {
            for j in 0..conv.s_kernel {
                let wf_idx = j + d * s_kernel_pad + c * s_kernel_pad * c_out_pad;
                let wf_val = GoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    // Build G[c] = Σ_j WP[c, j] · α^j
    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    // Build F[c] = Σ_{jh,jw} X[c,jh,jw] · β_h^{jh} · β_w^{jw}
    let x_data = witnesses[0];
    let beta_w_table = alpha_power_table(beta_w, w_in_pad);
    let beta_h_table = alpha_power_table(beta_h, h_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for jh in 0..conv.input_h {
            for jw in 0..conv.input_w {
                let x_idx = jw + jh * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let power = ext2_mul(beta_h_table[jh], beta_w_table[jw]);
                f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, power));
            }
        }
    }

    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_in_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(f_poly[c], g_poly[c]));
    }

    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F → X claim ----
    // XP[jw + jh*w_in_pad] = Σ_c eq(r_c, c) · X[c, jh, jw]
    // Power table: alpha_X[jw + jh*w_in_pad] = β_w^{jw} · β_h^{jh}
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![GoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for jh in 0..conv.input_h {
            for jw in 0..conv.input_w {
                let x_idx = jw + jh * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let flat_idx = jw + jh * w_in_pad;
                xp[flat_idx] = ext2_add(xp[flat_idx], ext2_mul(eq_c[c], x_val));
            }
        }
    }

    // Build factored power table for X spatial
    let mut alpha_x = vec![GoldilocksExt2::zero(); s_in_pad];
    for jh in 0..h_in_pad {
        for jw in 0..w_in_pad {
            alpha_x[jw + jh * w_in_pad] = ext2_mul(beta_h_table[jh], beta_w_table[jw]);
        }
    }

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_x, xp].as_mut_slice(), transcript);
    let r_j = prover3.challenges.clone();

    // Claim on X at (r_jw, r_jh, r_c) — NO reversal
    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_j);
    x_point.extend_from_slice(&r_c);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G → W_flat claim ----
    let mut wpp = vec![GoldilocksExt2::zero(); s_kernel_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            wpp[j] = ext2_add(wpp[j], ext2_mul(eq_c[c], wp[c * s_kernel_pad + j]));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, s_kernel_pad);

    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wpp].as_mut_slice(), transcript);
    let r_j_new = prover4.challenges.clone();

    // W_flat point: (r_j, r_d, r_c) — j bits | c_out bits | c_in bits
    let mut wf_point = Vec::with_capacity(l_kernel + l_d + l_c);
    wf_point.extend_from_slice(&r_j_new);
    wf_point.extend_from_slice(r_d);
    wf_point.extend_from_slice(&r_c);

    let wf_eval = wf_data.data.as_ref().unwrap().evaluate_at_point_ext2(&wf_point);

    let wf_claim = Claim {
        edge_id: wf_edge,
        sparse_id: 0,
        point: wf_point,
        eval: wf_eval,
    };

    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, wf_claim, s_alpha_claim])
}

// ============================================================================
// ConvTranspose2D verify
// ============================================================================

fn conv_transpose2d_verify(
    conv: &ConvTranspose2D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];

    let l_spatial_out = conv.l_spatial_out();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_wi = log2_ceil(conv.input_w.max(1));

    let r_spatial = &out_claim.point[..l_spatial_out];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let beta_w = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride) { b = ext2_mul(b, alpha); }
        b
    };

    // ---- Verify Sumcheck 1 ----
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0], v, l_spatial_out, 2, transcript,
    );
    if !ok1 {
        println!("ConvTranspose2D sumcheck 1 verification failed");
        return false;
    }

    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("ConvTranspose2D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2 ----
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("ConvTranspose2D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = ext2_add(
        sumcheck_proofs[2].round_messages[0][0],
        sumcheck_proofs[2].round_messages[0][1],
    );

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[2], inferred_sum_3, l_spatial_in, 2, transcript,
    );
    if !ok3 {
        println!("ConvTranspose2D sumcheck 3 verification failed");
        return false;
    }

    // Factored power table check: β_w for jw dims, β_h for jh dims
    let r_jw = &challenges3[..l_wi];
    let r_jh = &challenges3[l_wi..];
    let beta_w_mle = alpha_table_mle_eval(beta_w, r_jw);
    let beta_h_mle = alpha_table_mle_eval(beta_h, r_jh);
    let power_mle_3 = ext2_mul(beta_w_mle, beta_h_mle);
    let expected_final_3 = ext2_mul(power_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[2].final_eval {
        println!("ConvTranspose2D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("ConvTranspose2D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("ConvTranspose2D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("ConvTranspose2D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// ConvTranspose3D BasicBlock
// ============================================================================

#[derive(Clone, Debug)]
pub struct ConvTranspose3D {
    pub c_in: usize,
    pub c_out: usize,
    pub kernel_d: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub input_d: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub stride_d: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub flat_stride_h: usize, // H_out_pad * W_out_pad
    pub flat_stride_w: usize, // W_out_pad
    pub s_kernel: usize,
    pub d_out: usize,
    pub h_out: usize,
    pub w_out: usize,
}

impl ConvTranspose3D {
    pub fn new(c_in: usize, c_out: usize,
               kernel_d: usize, kernel_h: usize, kernel_w: usize,
               input_d: usize, input_h: usize, input_w: usize,
               stride_d: usize, stride_h: usize, stride_w: usize) -> Self {
        let d_out = (input_d - 1) * stride_d + kernel_d;
        let h_out = (input_h - 1) * stride_h + kernel_h;
        let w_out = (input_w - 1) * stride_w + kernel_w;
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let flat_stride_w = w_out_pad;
        let flat_stride_h = h_out_pad * w_out_pad;
        let s_kernel = (kernel_d - 1) * flat_stride_h + (kernel_h - 1) * flat_stride_w + kernel_w;
        Self { c_in, c_out, kernel_d, kernel_h, kernel_w, input_d, input_h, input_w,
               stride_d, stride_h, stride_w, flat_stride_h, flat_stride_w, s_kernel,
               d_out, h_out, w_out }
    }

    fn l_d(&self) -> usize { log2_ceil(self.c_out.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.c_in.max(1)) }
    fn l_wo(&self) -> usize { log2_ceil(self.w_out.max(1)) }
    fn l_ho(&self) -> usize { log2_ceil(self.h_out.max(1)) }
    fn l_do(&self) -> usize { log2_ceil(self.d_out.max(1)) }
    fn l_spatial_out(&self) -> usize { self.l_wo() + self.l_ho() + self.l_do() }
    fn l_spatial_in(&self) -> usize {
        log2_ceil(self.input_w.max(1)) + log2_ceil(self.input_h.max(1)) + log2_ceil(self.input_d.max(1))
    }
    fn l_kernel(&self) -> usize { log2_ceil(self.s_kernel.max(1)) }
}

impl BasicBlock for ConvTranspose3D {
    /// run(): ConvTranspose3D. X[C_in, D_in, H_in, W_in], W_flat[C_in, C_out, S_kernel] → Y[C_out, D_out, H_out, W_out]
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_out_pad = self.c_out.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let d_in_pad = self.input_d.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let d_out_pad = self.d_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();

        let out_size = c_out_pad * d_out_pad * h_out_pad * w_out_pad;

        // Parallelize by computing each output element independently (gather pattern).
        // For ConvTranspose: Y[d, od, oh, ow] = Σ_c Σ_kd Σ_kh Σ_kw X[c, jd, jh, jw] * W[c, d, kd, kh, kw]
        // where jd = (od - kd) / stride_d, jh = (oh - kh) / stride_h, jw = (ow - kw) / stride_w
        // Only contribute when (od - kd) % stride == 0 and jd in bounds.
        let x_data = x.data.as_ref().unwrap();
        let w_data = w_flat.data.as_ref().unwrap();
        let c_out = self.c_out;
        let d_out = self.d_out;
        let h_out = self.h_out;
        let w_out = self.w_out;
        let c_in = self.c_in;
        let kernel_d = self.kernel_d;
        let kernel_h = self.kernel_h;
        let kernel_w = self.kernel_w;
        let stride_d = self.stride_d;
        let stride_h = self.stride_h;
        let stride_w = self.stride_w;
        let input_d = self.input_d;
        let input_h = self.input_h;
        let input_w = self.input_w;
        let flat_stride_h = self.flat_stride_h;
        let flat_stride_w = self.flat_stride_w;

        let total_outputs = c_out * d_out * h_out * w_out;
        let mut out_data = vec![GoldilocksField(0); out_size];
        let results: Vec<(usize, GoldilocksField)> = (0..total_outputs)
            .into_par_iter()
            .map(|flat_idx| {
                let ow = flat_idx % w_out;
                let oh = (flat_idx / w_out) % h_out;
                let od = (flat_idx / (w_out * h_out)) % d_out;
                let d = flat_idx / (w_out * h_out * d_out);
                let mut acc = GoldilocksField(0);
                for c in 0..c_in {
                    for kd in 0..kernel_d {
                        if od < kd || (od - kd) % stride_d != 0 { continue; }
                        let jd = (od - kd) / stride_d;
                        if jd >= input_d { continue; }
                        for kh in 0..kernel_h {
                            if oh < kh || (oh - kh) % stride_h != 0 { continue; }
                            let jh = (oh - kh) / stride_h;
                            if jh >= input_h { continue; }
                            for kw in 0..kernel_w {
                                if ow < kw || (ow - kw) % stride_w != 0 { continue; }
                                let jw = (ow - kw) / stride_w;
                                if jw >= input_w { continue; }
                                let x_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad
                                    + c * w_in_pad * h_in_pad * d_in_pad;
                                let j = kd * flat_stride_h + kh * flat_stride_w + kw;
                                let wf_idx = j + d * s_kernel_pad + c * s_kernel_pad * c_out_pad;
                                let x_val = x_data.index(x_idx);
                                let w_val = w_data.index(wf_idx);
                                acc = gl_add(acc, gl_mul(x_val, w_val));
                            }
                        }
                    }
                }
                let out_idx = ow + oh * w_out_pad + od * w_out_pad * h_out_pad
                    + d * w_out_pad * h_out_pad * d_out_pad;
                (out_idx, acc)
            })
            .collect();
        for (idx, val) in results {
            out_data[idx] = val;
        }

        let out_shape = vec![self.c_out, self.d_out, self.h_out, self.w_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        conv_transpose3d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        conv_transpose3d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// ConvTranspose3D prove
// ============================================================================

fn conv_transpose3d_prove(
    conv: &ConvTranspose3D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let w_out_pad = conv.w_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let d_out_pad = conv.d_out.next_power_of_two();
    let s_out_pad = w_out_pad * h_out_pad * d_out_pad;
    let w_in_pad = conv.input_w.next_power_of_two();
    let h_in_pad = conv.input_h.next_power_of_two();
    let d_in_pad = conv.input_d.next_power_of_two();
    let s_in_pad = w_in_pad * h_in_pad * d_in_pad;
    let s_kernel_pad = conv.s_kernel.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();
    let c_out_pad = conv.c_out.next_power_of_two();

    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d]
    let y_data = witnesses[2];
    let mut yp = vec![GoldilocksExt2::zero(); s_out_pad];
    for d in 0..conv.c_out {
        for do_ in 0..conv.d_out {
            for ho in 0..conv.h_out {
                for wo in 0..conv.w_out {
                    let y_idx = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad
                        + d * w_out_pad * h_out_pad * d_out_pad;
                    let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
                    let k = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad;
                    yp[k] = ext2_add(yp[k], ext2_mul(eq_d[d], y_val));
                }
            }
        }
    }

    // ---- Sumcheck 1 ----
    let eq_spatial = evaluate_lagrange_basis_ext2(r_spatial);

    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof1 = prover1.prove(&mut [eq_spatial, yp.clone()].as_mut_slice(), transcript);
    let r_spatial_new = prover1.challenges.clone();

    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_spatial_out + l_d);
    y_self_point.extend_from_slice(&r_spatial_new);
    y_self_point.extend_from_slice(r_d);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Channel F×G ----
    let beta_w = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride_w) { b = ext2_mul(b, alpha); }
        b
    };
    let beta_d = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..(conv.stride_d * conv.flat_stride_h) { b = ext2_mul(b, alpha); }
        b
    };

    // Build WP[c, j] = Σ_d W_flat[c, d, j] · eq_D[d]
    let wf_data = witnesses[1];
    let mut wp = vec![GoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for c in 0..conv.c_in {
        for d in 0..conv.c_out {
            for j in 0..conv.s_kernel {
                let wf_idx = j + d * s_kernel_pad + c * s_kernel_pad * c_out_pad;
                let wf_val = GoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    // Build F[c] = Σ_{jd,jh,jw} X[c,jd,jh,jw] · β_d^{jd} · β_h^{jh} · β_w^{jw}
    let x_data = witnesses[0];
    let beta_w_table = alpha_power_table(beta_w, w_in_pad);
    let beta_h_table = alpha_power_table(beta_h, h_in_pad);
    let beta_d_table = alpha_power_table(beta_d, d_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for jd in 0..conv.input_d {
            for jh in 0..conv.input_h {
                for jw in 0..conv.input_w {
                    let x_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let power = ext2_mul(ext2_mul(beta_d_table[jd], beta_h_table[jh]), beta_w_table[jw]);
                    f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, power));
                }
            }
        }
    }

    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_in_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(f_poly[c], g_poly[c]));
    }

    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F → X ----
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![GoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for jd in 0..conv.input_d {
            for jh in 0..conv.input_h {
                for jw in 0..conv.input_w {
                    let x_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let flat_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad;
                    xp[flat_idx] = ext2_add(xp[flat_idx], ext2_mul(eq_c[c], x_val));
                }
            }
        }
    }

    let mut alpha_x = vec![GoldilocksExt2::zero(); s_in_pad];
    for jd in 0..d_in_pad {
        for jh in 0..h_in_pad {
            for jw in 0..w_in_pad {
                alpha_x[jw + jh * w_in_pad + jd * w_in_pad * h_in_pad] =
                    ext2_mul(ext2_mul(beta_d_table[jd], beta_h_table[jh]), beta_w_table[jw]);
            }
        }
    }

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_x.clone(), xp.clone()].as_mut_slice(), transcript);
    let r_j = prover3.challenges.clone();


    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_j);
    x_point.extend_from_slice(&r_c);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);


    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G → W_flat ----
    let mut wpp = vec![GoldilocksExt2::zero(); s_kernel_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            wpp[j] = ext2_add(wpp[j], ext2_mul(eq_c[c], wp[c * s_kernel_pad + j]));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, s_kernel_pad);

    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wpp].as_mut_slice(), transcript);
    let r_j_new = prover4.challenges.clone();

    let mut wf_point = Vec::with_capacity(l_kernel + l_d + l_c);
    wf_point.extend_from_slice(&r_j_new);
    wf_point.extend_from_slice(r_d);
    wf_point.extend_from_slice(&r_c);

    let wf_eval = wf_data.data.as_ref().unwrap().evaluate_at_point_ext2(&wf_point);

    let wf_claim = Claim {
        edge_id: wf_edge,
        sparse_id: 0,
        point: wf_point,
        eval: wf_eval,
    };

    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, wf_claim, s_alpha_claim])
}

// ============================================================================
// ConvTranspose3D verify
// ============================================================================

fn conv_transpose3d_verify(
    conv: &ConvTranspose3D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];

    let l_spatial_out = conv.l_spatial_out();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_wi = log2_ceil(conv.input_w.max(1));
    let l_hi = log2_ceil(conv.input_h.max(1));

    let r_spatial = &out_claim.point[..l_spatial_out];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let beta_w = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride_w) { b = ext2_mul(b, alpha); }
        b
    };
    let beta_d = {
        let mut b = GoldilocksExt2::one();
        for _ in 0..(conv.stride_d * conv.flat_stride_h) { b = ext2_mul(b, alpha); }
        b
    };

    // ---- Verify Sumcheck 1 ----
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0], v, l_spatial_out, 2, transcript,
    );
    if !ok1 {
        println!("ConvTranspose3D sumcheck 1 verification failed");
        return false;
    }

    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("ConvTranspose3D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2 ----
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("ConvTranspose3D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = if l_spatial_in == 0 {
        // Degenerate: input spatial 1×1×1, sumcheck has 0 rounds.
        // final_eval = alpha_x[0] * xp[0] = 1 * x_claim.eval
        if sumcheck_proofs[2].final_eval != x_claim.eval {
            println!("ConvTranspose3D sumcheck 3 (degenerate l=0) mismatch");
            return false;
        }
        sumcheck_proofs[2].final_eval
    } else {
        let sum3 = ext2_add(
            sumcheck_proofs[2].round_messages[0][0],
            sumcheck_proofs[2].round_messages[0][1],
        );

        let (ok3, challenges3) = SumcheckVerifier::verify(
            sumcheck_proofs[2], sum3, l_spatial_in, 2, transcript,
        );
        if !ok3 {
            println!("ConvTranspose3D sumcheck 3 verification failed");
            return false;
        }

        let r_jw = &challenges3[..l_wi];
        let r_jh = &challenges3[l_wi..l_wi + l_hi];
        let r_jd = &challenges3[l_wi + l_hi..];
        let beta_w_mle = alpha_table_mle_eval(beta_w, r_jw);
        let beta_h_mle = alpha_table_mle_eval(beta_h, r_jh);
        let beta_d_mle = alpha_table_mle_eval(beta_d, r_jd);
        let power_mle_3 = ext2_mul(ext2_mul(beta_d_mle, beta_h_mle), beta_w_mle);
        let expected_final_3 = ext2_mul(power_mle_3, x_claim.eval);
        if expected_final_3 != sumcheck_proofs[2].final_eval {
            println!("ConvTranspose3D sumcheck 3 final eval mismatch");
            return false;
        }
        sum3
    };

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("ConvTranspose3D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("ConvTranspose3D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("ConvTranspose3D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// DepthwiseConv2D BasicBlock
// ============================================================================

/// Depthwise 2D convolution: each channel is convolved independently.
/// Input X[C, H_in, W_in], Weight W_flat[C, S_kernel_pad], Output Y[C, H_out, W_out].
/// Groups = C = c_in = c_out.
#[derive(Clone, Debug)]
pub struct DepthwiseConv2D {
    pub channels: usize,    // C (= c_in = c_out = groups)
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub stride_w: usize,    // W_in_pad (flat 1D stride)
    pub s_in: usize,        // H_in_pad * W_in_pad
    pub s_kernel: usize,    // (kH-1)*W_in_pad + kW
    pub h_out: usize,
    pub w_out: usize,
    pub conv_stride_h: usize,
    pub conv_stride_w: usize,
}

impl DepthwiseConv2D {
    pub fn new(channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize) -> Self {
        Self::new_strided(channels, kernel_h, kernel_w, input_h, input_w, 1, 1)
    }

    pub fn new_strided(channels: usize, kernel_h: usize, kernel_w: usize, input_h: usize, input_w: usize, conv_stride_h: usize, conv_stride_w: usize) -> Self {
        let h_out = (input_h - kernel_h) / conv_stride_h + 1;
        let w_out = (input_w - kernel_w) / conv_stride_w + 1;
        let w_pad = input_w.next_power_of_two();
        let h_pad = input_h.next_power_of_two();
        let stride_w = w_pad;
        let s_in = h_pad * w_pad;
        let s_kernel = (kernel_h - 1) * w_pad + kernel_w;
        Self { channels, kernel_h, kernel_w, input_h, input_w, stride_w, s_in, s_kernel, h_out, w_out, conv_stride_h, conv_stride_w }
    }

    fn l_c(&self) -> usize { log2_ceil(self.channels.max(1)) }
    fn l_wo(&self) -> usize { log2_ceil(self.w_out.max(1)) }
    fn l_ho(&self) -> usize { log2_ceil(self.h_out.max(1)) }
    fn l_spatial_out(&self) -> usize { self.l_wo() + self.l_ho() }
    fn l_spatial_in(&self) -> usize { log2_ceil(self.input_w.max(1)) + log2_ceil(self.input_h.max(1)) }
    fn l_kernel(&self) -> usize { log2_ceil(self.s_kernel.max(1)) }
}

impl BasicBlock for DepthwiseConv2D {
    /// run(): Depthwise conv2d, Y[C, H_out, W_out].
    /// Inputs: [X, W_flat]
    ///   X shape: [C, H_in, W_in]
    ///   W_flat shape: [C, S_kernel_pad] (already flattened via FlattenKernel with c_in=1)
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_pad = self.channels.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        // Y[c, ho, wo] = Σ_kh Σ_kw X[c, ho*sh+kh, wo*sw+kw] * W[c, kh*stride_w+kw]
        for c in 0..self.channels {
            for ho in 0..self.h_out {
                for wo in 0..self.w_out {
                    let mut acc = GoldilocksField(0);
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let ih = ho * self.conv_stride_h + kh;
                            let iw = wo * self.conv_stride_w + kw;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let j = kh * self.stride_w + kw;
                            let wf_idx = j + c * s_kernel_pad;
                            let x_val = x.data.as_ref().unwrap().index(x_idx);
                            let w_val = w_flat.data.as_ref().unwrap().index(wf_idx);
                            acc = gl_add(acc, gl_mul(x_val, w_val));
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                    out_data[out_idx] = acc;
                }
            }
        }

        let out_shape = vec![self.channels, self.h_out, self.w_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_pad = self.channels.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;

        let d_x = x.as_device_buf();
        let d_w = w_flat.as_device_buf();
        let mut d_y = DeviceBuffer::<u64>::new(out_size).expect("DepthwiseConv2D: alloc out");
        memset_zero(&mut d_y, out_size).expect("DepthwiseConv2D: memset zero");

        gpu_depthwise_conv2d(
            &d_x, &d_w, &mut d_y,
            self.channels, self.h_out, self.w_out,
            self.kernel_h, self.kernel_w,
            self.conv_stride_h, self.conv_stride_w,
            w_in_pad, h_in_pad,
            s_kernel_pad,
            w_out_pad, h_out_pad,
            self.stride_w,
        ).expect("DepthwiseConv2D: gpu kernel failed");

        let out_shape = vec![self.channels, self.h_out, self.w_out];
        vec![Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        depthwise_conv2d_prove(self, witnesses, edge_ids, out_claims, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        depthwise_conv2d_verify(self, witnesses, claims, sumcheck_proofs, transcript)
    }
}

// ============================================================================
// DepthwiseConv2D prove
// ============================================================================

fn depthwise_conv2d_prove(
    conv: &DepthwiseConv2D,
    witnesses: &[&Witness],
    edge_ids: &[usize],
    out_claims: &[&Claim],
    transcript: &mut Transcript,
) -> (Vec<SumcheckProof>, Vec<Claim>) {
    // edge_ids: [x_edge, wf_edge, y_edge]
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let s_in_pad = conv.s_in.next_power_of_two();
    let s_kernel_pad = conv.s_kernel.next_power_of_two();
    let c_pad = conv.channels.next_power_of_two();
    let w_in_pad = conv.input_w.next_power_of_two();
    let h_in_pad = conv.input_h.next_power_of_two();
    let w_out_pad = conv.w_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let s_out_pad = w_out_pad * h_out_pad;

    // Parse claim point: Y shape [C, H_out, W_out]
    // little-endian: w_out bits (lowest) | h_out bits | c bits
    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_c = &out_claim.point[l_spatial_out..l_spatial_out + l_c];

    // ---- Sample α ----
    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // Build eq_C table
    let eq_c = evaluate_lagrange_basis_ext2(r_c);

    // Build YP[k] = Σ_c Y[c,k] · eq_C[c] for each spatial position k
    let y_data = witnesses[2];
    let mut yp = vec![GoldilocksExt2::zero(); s_out_pad];
    for c in 0..conv.channels {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let y_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                let y_val = GoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
                let k = wo + ho * w_out_pad;
                yp[k] = ext2_add(yp[k], ext2_mul(eq_c[c], y_val));
            }
        }
    }

    // ---- Sumcheck 1: eq-sumcheck to reduce output spatial ----
    let eq_spatial = evaluate_lagrange_basis_ext2(r_spatial);
    let mut prover1 = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof1 = prover1.prove(&mut [eq_spatial, yp.clone()].as_mut_slice(), transcript);
    let r_spatial_new = prover1.challenges.clone();

    // Self-claim on Y
    let yp_at_r = prover1.final_eval(1);
    let mut y_self_point = Vec::with_capacity(l_spatial_out + l_c);
    y_self_point.extend_from_slice(&r_spatial_new);
    y_self_point.extend_from_slice(r_c);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck 2: Degree-3 channel sumcheck ----
    // For depthwise conv, both F and G depend on channel c.
    // F[c] = Σ_i X_rev[c,i] · α^i
    // G[c] = Σ_j W_flat[c,j] · α^j
    // Prove: Σ_c eq_C(c) · F(c) · G(c) = s_alpha_conv

    // Build F[c] = Σ_i X_rev[c, i] · α^i
    let x_data = witnesses[0];
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![GoldilocksExt2::zero(); c_pad];
    for c in 0..conv.channels {
        for ih in 0..conv.input_h {
            for iw in 0..conv.input_w {
                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let i_flat = ih * conv.stride_w + iw;
                let rev_i = conv.s_in - 1 - i_flat;
                f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
            }
        }
    }

    // Build G[c] = Σ_j W_flat[c, j] · α^j
    let wf_data = witnesses[1];
    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![GoldilocksExt2::zero(); c_pad];
    for c in 0..conv.channels {
        for j in 0..conv.s_kernel {
            let wf_idx = j + c * s_kernel_pad;
            let wf_val = GoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wf_val, alpha_kernel[j]));
        }
    }

    // eq_C is already computed above
    let eq_c_poly = evaluate_lagrange_basis_ext2(r_c);

    // s_alpha_conv = Σ_c eq_C[c] · F[c] · G[c]
    let mut s_alpha_conv = GoldilocksExt2::zero();
    for c in 0..c_pad {
        s_alpha_conv = ext2_add(s_alpha_conv, ext2_mul(eq_c_poly[c], ext2_mul(f_poly[c], g_poly[c])));
    }
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    // Degree-3 sumcheck: 3 polynomials (eq_C, F, G)
    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 3, transcript);
    let proof2 = prover2.prove(&mut [eq_c_poly, f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c_new = prover2.challenges.clone();

    // ---- Sumcheck 3: F reduction to X claim ----
    // F(r_c_new) = prover2.final_eval(1)
    // F[c] = Σ_i X_rev[c, i] · α^i
    // XP[i] = Σ_c X_rev[c, i] · eq(r_c_new, c)
    // Prove: Σ_i α^i · XP[i] = F(r_c_new)

    let eq_c_new = evaluate_lagrange_basis_ext2(&r_c_new);
    let mut xp = vec![GoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.channels {
        for ih in 0..conv.input_h {
            for iw in 0..conv.input_w {
                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = GoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let i_flat = ih * conv.stride_w + iw;
                let rev_i = conv.s_in - 1 - i_flat;
                xp[rev_i] = ext2_add(xp[rev_i], ext2_mul(eq_c_new[c], x_val));
            }
        }
    }

    let alpha_poly_in = alpha_power_table(alpha, s_in_pad);
    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_poly_in, xp].as_mut_slice(), transcript);
    let r_i = prover3.challenges.clone();

    // X_rev(r_c, r_i) = X(r_c, 1-r_i)
    let one = GoldilocksExt2::one();
    let r_spatial_x: Vec<GoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

    let mut x_point = Vec::with_capacity(l_spatial_in + l_c);
    x_point.extend_from_slice(&r_spatial_x);
    x_point.extend_from_slice(&r_c_new);

    let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(&x_point);

    let x_claim = Claim {
        edge_id: x_edge,
        sparse_id: 0,
        point: x_point,
        eval: x_eval,
    };

    // ---- Sumcheck 4: G reduction to W_flat claim ----
    // G(r_c_new) = prover2.final_eval(2)
    // G[c] = Σ_j W_flat[c, j] · α^j
    // WP[j] = Σ_c eq(r_c_new, c) · W_flat[c, j]
    // Prove: Σ_j α^j · WP[j] = G(r_c_new)

    let mut wp = vec![GoldilocksExt2::zero(); s_kernel_pad];
    for c in 0..conv.channels {
        for j in 0..conv.s_kernel {
            let wf_idx = j + c * s_kernel_pad;
            let wf_val = GoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
            wp[j] = ext2_add(wp[j], ext2_mul(eq_c_new[c], wf_val));
        }
    }

    let alpha_poly_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut prover4 = CpuLinearSumcheckProverExt2::new(l_kernel, 2, transcript);
    let proof4 = prover4.prove(&mut [alpha_poly_kernel, wp].as_mut_slice(), transcript);
    let r_j_new = prover4.challenges.clone();

    // W_flat point: (r_j, r_c) in little-endian order. W_flat shape [C, S_kernel]
    let mut wf_point = Vec::with_capacity(l_kernel + l_c);
    wf_point.extend_from_slice(&r_j_new);
    wf_point.extend_from_slice(&r_c_new);

    let wf_eval = wf_data.data.as_ref().unwrap().evaluate_at_point_ext2(&wf_point);

    let wf_claim = Claim {
        edge_id: wf_edge,
        sparse_id: 0,
        point: wf_point,
        eval: wf_eval,
    };

    // Carrier claim for s_alpha_conv
    let s_alpha_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: vec![],
        eval: s_alpha_conv,
    };

    (vec![proof1, proof2, proof3, proof4], vec![y_self_claim, x_claim, wf_claim, s_alpha_claim])
}

// ============================================================================
// DepthwiseConv2D verify
// ============================================================================

fn depthwise_conv2d_verify(
    conv: &DepthwiseConv2D,
    _witnesses: &[&Witness],
    claims: &[&Claim],
    sumcheck_proofs: &[&SumcheckProof],
    transcript: &mut Transcript,
) -> bool {
    // claims layout: [y_self_claim, x_claim, wf_claim, s_alpha_claim, out_claim]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];

    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_c = &out_claim.point[l_spatial_out..l_spatial_out + l_c];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // ---- Verify Sumcheck 1: eq-sumcheck ----
    let (ok1, challenges1) = SumcheckVerifier::verify(
        sumcheck_proofs[0],
        v,
        l_spatial_out,
        2,
        transcript,
    );
    if !ok1 {
        println!("DepthwiseConv2D sumcheck 1 verification failed");
        return false;
    }

    let eq_sr = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("DepthwiseConv2D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 2: Degree-3 channel sumcheck ----
    let s_alpha_conv = claims[3].eval;
    transcript.append_ext2(b"s_alpha_conv", &s_alpha_conv);

    let (ok2, challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[1],
        s_alpha_conv,
        l_c,
        3, // degree 3: eq_C * F * G
        transcript,
    );
    if !ok2 {
        println!("DepthwiseConv2D sumcheck 2 verification failed");
        return false;
    }

    // Verify final eval of sumcheck 2: eq_C(r_c_new) * F(r_c_new) * G(r_c_new)
    // eq_C(r_c_new) = eq(r_c, r_c_new) — verifier can compute
    let eq_c_val = {
        let one = GoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_c {
            let a = r_c[i];
            let b = challenges2[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(GoldilocksExt2::from_base(GoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[2].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[2].round_messages[0][0],
            sumcheck_proofs[2].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[2],
        inferred_sum_3,
        l_spatial_in,
        2,
        transcript,
    );
    if !ok3 {
        println!("DepthwiseConv2D sumcheck 3 verification failed");
        return false;
    }

    let alpha_mle_3 = alpha_table_mle_eval(alpha, &challenges3);
    let expected_final_3 = ext2_mul(alpha_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[2].final_eval {
        println!("DepthwiseConv2D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    // Cross-check: eq_C(r_c_new) * F(r_c_new) * G(r_c_new) = final_eval_2
    // F(r_c_new) = inferred_sum_3, G(r_c_new) = inferred_sum_4
    let fg_product = ext2_mul(eq_c_val, ext2_mul(inferred_sum_3, inferred_sum_4));
    if fg_product != sumcheck_proofs[1].final_eval {
        println!("DepthwiseConv2D eq*F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[3],
        inferred_sum_4,
        l_kernel,
        2,
        transcript,
    );
    if !ok4 {
        println!("DepthwiseConv2D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[3].final_eval {
        println!("DepthwiseConv2D sumcheck 4 final eval mismatch");
        return false;
    }

    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Witness, DataType, Role};
    use goldilocks_cuda::GoldilocksField;

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_conv2d_run_1x1_kernel() {
        // C_in=2, C_out=2, H=2, W=2, kernel 1x1
        let conv = Conv2D::new(2, 2, 1, 1, 2, 2);
        assert_eq!(conv.h_out, 2);
        assert_eq!(conv.w_out, 2);
        assert_eq!(conv.s_kernel, 1);

        // X[C_in=2, H=2, W=2]: little-endian [w, h, c]
        // C=0: [[1,2],[3,4]] → flat: [1,2,3,4]
        // C=1: [[5,6],[7,8]] → flat: [5,6,7,8]
        let x = make_witness(vec![2, 2, 2], vec![1,2,3,4, 5,6,7,8]);

        // W_flat[C_out=2, C_in=2, S_kernel=1]:
        // d=0,c=0: [10], d=0,c=1: [20]
        // d=1,c=0: [30], d=1,c=1: [40]
        let wf = make_witness(vec![2, 2, 1], vec![10, 20, 30, 40]);

        let result = conv.run(&[&x, &wf]);
        let y = &result[0];

        // Y[d=0, h, w] = X[c=0,h,w]*10 + X[c=1,h,w]*20
        // = [1*10+5*20, 2*10+6*20, 3*10+7*20, 4*10+8*20]
        // = [110, 140, 170, 200]
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(110));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(140));
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(170));
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(200));

        // Y[d=1, h, w] = X[c=0,h,w]*30 + X[c=1,h,w]*40
        // = [1*30+5*40, 2*30+6*40, 3*30+7*40, 4*30+8*40]
        // = [230, 300, 370, 440]
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(230));
        assert_eq!(y.data.as_ref().unwrap().index(5), GoldilocksField(300));
        assert_eq!(y.data.as_ref().unwrap().index(6), GoldilocksField(370));
        assert_eq!(y.data.as_ref().unwrap().index(7), GoldilocksField(440));
    }

    #[test]
    fn test_conv2d_run_3x3_kernel() {
        // C_in=1, C_out=1, H=4, W=4, kernel 3x3
        let conv = Conv2D::new(1, 1, 3, 3, 4, 4);
        assert_eq!(conv.h_out, 2);
        assert_eq!(conv.w_out, 2);

        // X[1, 4, 4] = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        // little-endian layout: w bits (2) lowest, h bits (2) next, c bits (0)
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);

        // W[1,1,3,3], kernel values all 1 for simple sum
        // FlattenKernel scatter: kh*4 + kw for j index
        // s_kernel = (3-1)*4 + 3 = 11
        let fk = FlattenKernel { s_w: 4, kh: 3, kw: 3, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let w_raw = make_witness(vec![1, 1, 3, 3], vec![1,1,1,0, 1,1,1,0, 1,1,1,0, 0,0,0,0]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Y[0,0,0] = sum of 3x3 block starting at (0,0) = 1+2+3+5+6+7+9+10+11 = 54
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(54));
        // Y[0,0,1] = sum of 3x3 block starting at (0,1) = 2+3+4+6+7+8+10+11+12 = 63
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(63));
        // Y[0,1,0] = sum starting at (1,0) = 5+6+7+9+10+11+13+14+15 = 90
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(90));
        // Y[0,1,1] = sum starting at (1,1) = 6+7+8+10+11+12+14+15+16 = 99
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(99));
    }

    #[test]
    fn test_flatten_kernel_run() {
        let fk = FlattenKernel { s_w: 4, kh: 3, kw: 3, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        // W[1,1,3,3]: kw bits(2) lowest, kh bits(2) next
        // Values: w[kh=0,kw=0]=1, w[0,1]=2, w[0,2]=3, w[1,0]=4, w[1,1]=5, w[1,2]=6, w[2,0]=7, w[2,1]=8, w[2,2]=9
        let w = make_witness(vec![1, 1, 3, 3], vec![1,2,3,0, 4,5,6,0, 7,8,9,0, 0,0,0,0]);
        let result = fk.run(&[&w]);
        let wf = &result[0];
        // s_kernel = (3-1)*4+3 = 11, padded to 16
        // j = kh*4 + kw
        // kh=0: j=0,1,2 → vals 1,2,3
        // kh=1: j=4,5,6 → vals 4,5,6
        // kh=2: j=8,9,10 → vals 7,8,9
        assert_eq!(wf.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(wf.data.as_ref().unwrap().index(1), GoldilocksField(2));
        assert_eq!(wf.data.as_ref().unwrap().index(2), GoldilocksField(3));
        assert_eq!(wf.data.as_ref().unwrap().index(3), GoldilocksField(0));
        assert_eq!(wf.data.as_ref().unwrap().index(4), GoldilocksField(4));
        assert_eq!(wf.data.as_ref().unwrap().index(5), GoldilocksField(5));
        assert_eq!(wf.data.as_ref().unwrap().index(6), GoldilocksField(6));
        assert_eq!(wf.data.as_ref().unwrap().index(7), GoldilocksField(0));
        assert_eq!(wf.data.as_ref().unwrap().index(8), GoldilocksField(7));
        assert_eq!(wf.data.as_ref().unwrap().index(9), GoldilocksField(8));
        assert_eq!(wf.data.as_ref().unwrap().index(10), GoldilocksField(9));
    }

    #[test]
    fn test_flatten_kernel_prove_verify() {
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        // s_kernel = (2-1)*4+2 = 6, padded to 8
        let w = make_witness(vec![1, 1, 2, 2], vec![1, 2, 3, 4]);
        let wf_result = fk.run(&[&w]);
        let wf = &wf_result[0];

        // Create a claim on W_flat
        let n_wf = wf.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_fk");
        let point: Vec<GoldilocksExt2> = (0..n_wf)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = wf.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_fk_prove");
        let (proofs, claims) = fk.prove(
            &[&w, wf],
            &[0, 1],
            &[&claim],
            &mut prove_transcript,
        );

        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 1);

        // Verify
        let mut verify_transcript = Transcript::new(b"test_fk_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = fk.verify(
            &[&w, wf],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "FlattenKernel prove/verify should pass");
    }

    #[test]
    fn test_conv2d_prove_verify_small() {
        // C_in=1, C_out=1, H=4, W=4, kernel 2x2
        let conv = Conv2D::new(1, 1, 2, 2, 4, 4);
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };

        // X[1,4,4]
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);

        // W[1,1,2,2]
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        // Run conv2d
        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2, // y_edge
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_conv_prove");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        // Verify
        let mut verify_transcript = Transcript::new(b"test_conv_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv2D prove/verify should pass");
    }

    #[test]
    fn test_conv2d_prove_verify_multichannel() {
        // C_in=2, C_out=2, H=4, W=4, kernel 2x2
        let conv = Conv2D::new(2, 2, 2, 2, 4, 4);
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 2, c_in: 2, dilation_h: 1, dilation_w: 1 };

        // X[2,4,4]: two channels
        let x = make_witness(vec![2, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, // c=0
            2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,17, // c=1
        ]);

        // W[2,2,2,2]: all 1s
        let w_raw = make_witness(vec![2, 2, 2, 2], vec![
            1,1,1,1, // d=0,c=0
            1,1,1,1, // d=0,c=1
            1,1,1,1, // d=1,c=0
            1,1,1,1, // d=1,c=1
        ]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv2");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_conv2_prove");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"test_conv2_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv2D multichannel prove/verify should pass");
    }

    #[test]
    fn test_flatten_kernel_prove_verify_large_sw() {
        // Simulates VGG: s_w=64 (from 34.next_power_of_two()), kh=3, kw=3
        let fk = FlattenKernel { s_w: 64, kh: 3, kw: 3, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        // s_kernel = (3-1)*64+3 = 131, padded to 256
        // W[1,1,3,3]: kw bits(2) lowest, kh bits(2) next
        let w = make_witness(vec![1, 1, 3, 3], vec![1,2,3,0, 4,5,6,0, 7,8,9,0, 0,0,0,0]);
        let wf_result = fk.run(&[&w]);
        let wf = &wf_result[0];

        // Verify run output
        // j = kh*64 + kw
        // kh=0: j=0,1,2
        // kh=1: j=64,65,66
        // kh=2: j=128,129,130
        assert_eq!(wf.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(wf.data.as_ref().unwrap().index(1), GoldilocksField(2));
        assert_eq!(wf.data.as_ref().unwrap().index(2), GoldilocksField(3));
        assert_eq!(wf.data.as_ref().unwrap().index(64), GoldilocksField(4));
        assert_eq!(wf.data.as_ref().unwrap().index(65), GoldilocksField(5));
        assert_eq!(wf.data.as_ref().unwrap().index(128), GoldilocksField(7));

        // Create a claim on W_flat
        let n_wf = wf.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_fk_large");
        let point: Vec<GoldilocksExt2> = (0..n_wf)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = wf.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_fk_large_p");
        let (proofs, claims) = fk.prove(
            &[&w, wf],
            &[0, 1],
            &[&claim],
            &mut prove_transcript,
        );

        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 1);

        // Verify
        let mut verify_transcript = Transcript::new(b"test_fk_large_p");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = fk.verify(
            &[&w, wf],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "FlattenKernel with s_w=64 prove/verify should pass");
    }

    #[test]
    fn test_conv2d_prove_verify_non_pow2_input() {
        // Simulates VGG after ZeroPad: C_in=1, C_out=1, H=34, W=34, kernel 3x3
        // w_pad = 34.next_power_of_two() = 64
        let c_in = 1;
        let c_out = 1;
        let input_h = 34;
        let input_w = 34;
        let (kh, kw) = (3, 3);

        let conv = Conv2D::new(c_in, c_out, kh, kw, input_h, input_w);
        assert_eq!(conv.h_out, 32);
        assert_eq!(conv.w_out, 32);
        assert_eq!(conv.stride_w, 64); // w_pad

        let w_pad = input_w.next_power_of_two(); // 64
        let fk = FlattenKernel { s_w: w_pad, kh, kw, c_out, c_in, dilation_h: 1, dilation_w: 1 };

        // X[1,34,34] with some pattern (pad to power-of-2: 64*64=4096)
        let w_in_pad = 64;
        let h_in_pad = 64;
        let mut x_data = vec![0u64; h_in_pad * w_in_pad];
        for ih in 0..input_h {
            for iw in 0..input_w {
                x_data[iw + ih * w_in_pad] = ((ih * input_w + iw) % 500 + 1) as u64;
            }
        }
        let x = make_witness(vec![c_in, input_h, input_w], x_data);

        // W[1,1,3,3]: simple 3x3 kernel
        let w_raw = make_witness(vec![c_out, c_in, kh, kw], vec![1,1,1,0, 1,1,1,0, 1,1,1,0, 0,0,0,0]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        // Run conv2d
        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv_np2");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove Conv2D
        let mut prove_transcript = Transcript::new(b"test_conv_np2_p");
        let (conv_proofs, conv_claims) = conv.prove(
            &[&x, wf, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(conv_proofs.len(), 4);
        assert_eq!(conv_claims.len(), 4);

        // Verify Conv2D
        let mut verify_transcript = Transcript::new(b"test_conv_np2_p");
        let mut all_conv_claims: Vec<&Claim> = conv_claims.iter().collect();
        all_conv_claims.push(&out_claim);
        let conv_proofs_ref: Vec<&SumcheckProof> = conv_proofs.iter().collect();
        let conv_verified = conv.verify(
            &[&x, wf, y],
            &all_conv_claims,
            &conv_proofs_ref,
            &mut verify_transcript,
        );
        assert!(conv_verified, "Conv2D with 34x34 input should verify");

        // Now test FlattenKernel with the wf_claim from Conv2D
        let wf_claim = &conv_claims[2]; // wf_claim from Conv2D sumcheck 4

        // Prove FlattenKernel
        let mut fk_prove_transcript = Transcript::new(b"test_fk_np2_p");
        let (fk_proofs, fk_claims) = fk.prove(
            &[&w_raw, wf],
            &[0, 1],
            &[wf_claim],
            &mut fk_prove_transcript,
        );
        assert_eq!(fk_proofs.len(), 1);
        assert_eq!(fk_claims.len(), 1);

        // Verify FlattenKernel
        let mut fk_verify_transcript = Transcript::new(b"test_fk_np2_p");
        let mut fk_all_claims: Vec<&Claim> = fk_claims.iter().collect();
        fk_all_claims.push(wf_claim);
        let fk_proofs_ref: Vec<&SumcheckProof> = fk_proofs.iter().collect();
        let fk_verified = fk.verify(
            &[&w_raw, wf],
            &fk_all_claims,
            &fk_proofs_ref,
            &mut fk_verify_transcript,
        );
        assert!(fk_verified, "FlattenKernel with wf_claim from Conv2D 34x34 should verify");
    }

    // ================================================================
    // Conv1D tests
    // ================================================================

    #[test]
    fn test_conv1d_run() {
        // C_in=1, C_out=1, L=4, K=2
        let conv = Conv1D::new(1, 1, 2, 4);
        assert_eq!(conv.l_out, 3);

        // X[1, 4]: values [1, 2, 3, 4]
        let x = make_witness(vec![1, 4], vec![1, 2, 3, 4]);
        // W[1, 1, 2]: values [1, 1]
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        // Y[0] = X[0]*1 + X[1]*1 = 1+2 = 3
        // Y[1] = X[1]*1 + X[2]*1 = 2+3 = 5
        // Y[2] = X[2]*1 + X[3]*1 = 3+4 = 7
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(3));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(5));
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(7));
    }

    #[test]
    fn test_conv1d_prove_verify() {
        // C_in=1, C_out=1, L=4, K=2
        let conv = Conv1D::new(1, 1, 2, 4);

        let x = make_witness(vec![1, 4], vec![1, 2, 3, 4]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv1d");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_conv1d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &w, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_conv1d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv1D prove/verify should pass");
    }

    #[test]
    fn test_conv1d_multichannel_prove_verify() {
        // C_in=2, C_out=2, L=4, K=2
        let conv = Conv1D::new(2, 2, 2, 4);

        // X[2, 4]: two channels, l_in_pad=4
        let x = make_witness(vec![2, 4], vec![
            1, 2, 3, 4,  // c=0
            5, 6, 7, 8,  // c=1
        ]);
        // W[2, 2, 2]: k_pad=2
        let w = make_witness(vec![2, 2, 2], vec![
            1, 1,  // d=0, c=0
            1, 1,  // d=0, c=1
            2, 2,  // d=1, c=0
            2, 2,  // d=1, c=1
        ]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv1d_mc");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_conv1d_mc_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &w, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"test_conv1d_mc_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv1D multichannel prove/verify should pass");
    }

    #[test]
    fn test_conv1d_strided_run() {
        // C_in=1, C_out=1, L=8, K=3, stride=2
        // l_out = (8-3)/2 + 1 = 3
        let conv = Conv1D::new_strided(1, 1, 3, 8, 2);
        assert_eq!(conv.l_out, 3);

        // X[1, 8]: values [1, 2, 3, 4, 5, 6, 7, 8]
        let x = make_witness(vec![1, 8], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // W[1, 1, 3]: values [1, 1, 1, 0] (sum kernel, padded to k_pad=4)
        let w = make_witness(vec![1, 1, 3], vec![1, 1, 1, 0]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        // Y[0] = X[0]+X[1]+X[2] = 1+2+3 = 6
        // Y[1] = X[2]+X[3]+X[4] = 3+4+5 = 12
        // Y[2] = X[4]+X[5]+X[6] = 5+6+7 = 18
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(6));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(12));
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(18));
    }

    #[test]
    fn test_conv1d_strided_prove_verify() {
        // C_in=1, C_out=1, L=8, K=3, stride=2
        let conv = Conv1D::new_strided(1, 1, 3, 8, 2);

        let x = make_witness(vec![1, 8], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let w = make_witness(vec![1, 1, 3], vec![1, 1, 1, 0]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv1d_s");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_conv1d_s_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &w, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_conv1d_s_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv1D strided prove/verify should pass");
    }

    // ================================================================
    // FlattenKernel3D tests
    // ================================================================

    #[test]
    fn test_flatten_kernel_3d_run() {
        // kD=2, kH=2, kW=2, C_out=1, C_in=1
        // Using stride_h = h_pad * w_pad = 4*4 = 16, stride_w = w_pad = 4
        let fk = FlattenKernel3D {
            stride_h: 16, stride_w: 4, kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        // s_kernel = (2-1)*16 + (2-1)*4 + 2 = 22

        // W[1,1,2,2,2] kw bits(1) | kh bits(1) | kd bits(1)
        let w = make_witness(vec![1, 1, 2, 2, 2], vec![
            1, 2, 3, 4,  // kd=0: kh=0,kw=0..1 then kh=1,kw=0..1
            5, 6, 7, 8,  // kd=1: kh=0,kw=0..1 then kh=1,kw=0..1
        ]);
        let result = fk.run(&[&w]);
        let wf = &result[0];

        // j = kd*16 + kh*4 + kw
        // (0,0,0)->0: val 1, (0,0,1)->1: val 2, (0,1,0)->4: val 3, (0,1,1)->5: val 4
        // (1,0,0)->16: val 5, (1,0,1)->17: val 6, (1,1,0)->20: val 7, (1,1,1)->21: val 8
        assert_eq!(wf.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(wf.data.as_ref().unwrap().index(1), GoldilocksField(2));
        assert_eq!(wf.data.as_ref().unwrap().index(4), GoldilocksField(3));
        assert_eq!(wf.data.as_ref().unwrap().index(5), GoldilocksField(4));
        assert_eq!(wf.data.as_ref().unwrap().index(16), GoldilocksField(5));
        assert_eq!(wf.data.as_ref().unwrap().index(17), GoldilocksField(6));
        assert_eq!(wf.data.as_ref().unwrap().index(20), GoldilocksField(7));
        assert_eq!(wf.data.as_ref().unwrap().index(21), GoldilocksField(8));
    }

    #[test]
    fn test_flatten_kernel_3d_prove_verify() {
        let fk = FlattenKernel3D {
            stride_h: 4, stride_w: 2, kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        // s_kernel = (2-1)*4 + (2-1)*2 + 2 = 8

        // W[1,1,2,2,2]
        let w = make_witness(vec![1, 1, 2, 2, 2], vec![
            1, 2, 3, 4,
            5, 6, 7, 8,
        ]);
        let wf_result = fk.run(&[&w]);
        let wf = &wf_result[0];

        let n_wf = wf.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_fk3d");
        let point: Vec<GoldilocksExt2> = (0..n_wf)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = wf.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_fk3d_p");
        let (proofs, claims) = fk.prove(
            &[&w, wf],
            &[0, 1],
            &[&claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 1);

        let mut verify_transcript = Transcript::new(b"test_fk3d_p");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = fk.verify(
            &[&w, wf],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "FlattenKernel3D prove/verify should pass");
    }

    // ================================================================
    // Conv3D tests
    // ================================================================

    #[test]
    fn test_conv3d_run() {
        // C_in=1, C_out=1, D=H=W=4, kernel 2x2x2
        let conv = Conv3D::new(1, 1, 2, 2, 2, 4, 4, 4);
        assert_eq!(conv.d_out, 3);
        assert_eq!(conv.h_out, 3);
        assert_eq!(conv.w_out, 3);

        let fk = FlattenKernel3D {
            stride_h: conv.stride_h,
            stride_w: conv.stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };

        // X[1,4,4,4]: sequential values
        let x_len = 4 * 4 * 4; // d_pad * h_pad * w_pad = 4*4*4
        let x_data: Vec<u64> = (1..=x_len as u64).collect();
        let x = make_witness(vec![1, 4, 4, 4], x_data);

        // W[1,1,2,2,2]: all 1s
        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 1, 1, 1, 1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Y[0,0,0,0] = sum of 2x2x2 block starting at (0,0,0)
        // X[0,0,0]+X[0,0,1]+X[0,1,0]+X[0,1,1]+X[1,0,0]+X[1,0,1]+X[1,1,0]+X[1,1,1]
        // = 1+2+5+6+17+18+21+22 = 92
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(92));
    }

    #[test]
    fn test_conv3d_prove_verify() {
        // C_in=1, C_out=1, D=H=W=4, kernel 2x2x2
        let conv = Conv3D::new(1, 1, 2, 2, 2, 4, 4, 4);

        let fk = FlattenKernel3D {
            stride_h: conv.stride_h,
            stride_w: conv.stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };

        let x_len = 4 * 4 * 4;
        let x_data: Vec<u64> = (1..=x_len as u64).collect();
        let x = make_witness(vec![1, 4, 4, 4], x_data);

        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 1, 1, 1, 1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv3d");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_conv3d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_conv3d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv3D prove/verify should pass");
    }

    // ================================================================
    // ConvTranspose1D tests
    // ================================================================

    #[test]
    fn test_conv_transpose1d_run() {
        // C_in=1, C_out=1, L=3, K=2, stride=2
        // L_out = (3-1)*2 + 2 = 6
        let conv = ConvTranspose1D::new(1, 1, 2, 3, 2);
        assert_eq!(conv.l_out, 6);

        // X[1, 3]: [1, 2, 3] (padded to 4)
        let x = make_witness(vec![1, 3], vec![1, 2, 3, 0]);
        // W[C_in=1, C_out=1, K=2]: [1, 1]
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        // Y[d, j*2+k] += X[c, j] * W[c, d, k]
        // j=0: Y[0]+=1*1=1, Y[1]+=1*1=1
        // j=1: Y[2]+=2*1=2, Y[3]+=2*1=2
        // j=2: Y[4]+=3*1=3, Y[5]+=3*1=3
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(1));
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(2));
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(2));
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(3));
        assert_eq!(y.data.as_ref().unwrap().index(5), GoldilocksField(3));
    }

    #[test]
    fn test_conv_transpose1d_prove_verify() {
        let conv = ConvTranspose1D::new(1, 1, 2, 3, 2);

        let x = make_witness(vec![1, 3], vec![1, 2, 3, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct1d");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct1d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &w, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_ct1d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose1D prove/verify should pass");
    }

    #[test]
    fn test_conv_transpose1d_stride1_prove_verify() {
        // stride=1 is just like a correlation (not reversed conv)
        let conv = ConvTranspose1D::new(2, 2, 3, 4, 1);
        // L_out = (4-1)*1 + 3 = 6

        let x = make_witness(vec![2, 4], vec![1,2,3,4, 5,6,7,8]);
        let w = make_witness(vec![2, 2, 3], vec![
            1,1,1,0, // c=0,d=0
            1,1,1,0, // c=0,d=1
            1,1,1,0, // c=1,d=0
            1,1,1,0, // c=1,d=1
        ]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct1d_s1");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct1d_s1_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &w, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"test_ct1d_s1_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose1D stride=1 prove/verify should pass");
    }

    // ================================================================
    // ConvTranspose2D tests
    // ================================================================

    #[test]
    fn test_conv_transpose2d_run() {
        // C_in=1, C_out=1, H=2, W=2, K=2x2, stride=(2,2)
        // H_out = (2-1)*2 + 2 = 4, W_out = (2-1)*2 + 2 = 4
        let conv = ConvTranspose2D::new(1, 1, 2, 2, 2, 2, 2, 2);
        assert_eq!(conv.h_out, 4);
        assert_eq!(conv.w_out, 4);

        // Need FlattenKernel for W[C_in=1, C_out=1, kH=2, kW=2] → W_flat[1, 1, s_kernel]
        // flat_stride = w_out_pad = 4
        let fk = FlattenKernel {
            s_w: conv.flat_stride, kh: 2, kw: 2,
            c_out: 1, // c_in of ConvTranspose = c_out of FlattenKernel
            c_in: 1,  // c_out of ConvTranspose = c_in of FlattenKernel
            dilation_h: 1, dilation_w: 1,
        };

        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let x = make_witness(vec![1, 2, 2], vec![1, 2, 3, 4]);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // j=(0,0): x=1, k=(0,0)→(0,0)=1, k=(0,1)→(0,1)=1, k=(1,0)→(1,0)=1, k=(1,1)→(1,1)=1
        // j=(0,1): x=2, k→ pos (0,2),(0,3),(1,2),(1,3) all +=2
        // j=(1,0): x=3, k→ pos (2,0),(2,1),(3,0),(3,1) all +=3
        // j=(1,1): x=4, k→ pos (2,2),(2,3),(3,2),(3,3) all +=4
        // Y[0,0]=1, Y[0,1]=1, Y[0,2]=2, Y[0,3]=2
        // Y[1,0]=1, Y[1,1]=1, Y[1,2]=2, Y[1,3]=2
        // Y[2,0]=3, Y[2,1]=3, Y[2,2]=4, Y[2,3]=4
        // Y[3,0]=3, Y[3,1]=3, Y[3,2]=4, Y[3,3]=4
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(1)); // (0,0)
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(1)); // (0,1)
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(2)); // (0,2)
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(2)); // (0,3)
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(1)); // (1,0)
        assert_eq!(y.data.as_ref().unwrap().index(8), GoldilocksField(3)); // (2,0)
        assert_eq!(y.data.as_ref().unwrap().index(10), GoldilocksField(4)); // (2,2)
    }

    #[test]
    fn test_conv_transpose2d_prove_verify() {
        let conv = ConvTranspose2D::new(1, 1, 2, 2, 2, 2, 2, 2);

        let fk = FlattenKernel {
            s_w: conv.flat_stride, kh: 2, kw: 2, c_out: 1, c_in: 1,
            dilation_h: 1, dilation_w: 1,
        };
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 2, 3, 4]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let x = make_witness(vec![1, 2, 2], vec![1, 2, 3, 4]);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct2d");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct2d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_ct2d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose2D prove/verify should pass");
    }

    // ================================================================
    // ConvTranspose3D tests
    // ================================================================

    #[test]
    fn test_conv_transpose3d_run() {
        // C_in=1, C_out=1, D=H=W=2, K=2x2x2, stride=(2,2,2)
        // D_out = (2-1)*2+2 = 4, H_out = 4, W_out = 4
        let conv = ConvTranspose3D::new(1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq!(conv.d_out, 4);
        assert_eq!(conv.h_out, 4);
        assert_eq!(conv.w_out, 4);

        // FlattenKernel3D: channels swapped for transpose
        let fk = FlattenKernel3D {
            stride_h: conv.flat_stride_h,
            stride_w: conv.flat_stride_w,
            kd: 2, kh: 2, kw: 2,
            c_out: 1, // c_in of ConvTranspose
            c_in: 1,  // c_out of ConvTranspose
        };

        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 1, 1, 1, 1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        // X[1, 2, 2, 2]
        let x = make_witness(vec![1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Each input element scatters to a 2x2x2 block at stride-2 offsets
        // Y[0,0,0] should be X[0,0,0]*1 = 1
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(1));
    }

    #[test]
    fn test_conv_transpose3d_prove_verify() {
        let conv = ConvTranspose3D::new(1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);

        let fk = FlattenKernel3D {
            stride_h: conv.flat_stride_h,
            stride_w: conv.flat_stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };

        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let x = make_witness(vec![1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct3d");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct3d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_ct3d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose3D prove/verify should pass");
    }

    #[test]
    fn test_conv_transpose3d_non_pow2_channels() {
        // Reproduces the 3D UNet bug: c_in=320 (non-power-of-2), c_out=256
        let c_in = 320;
        let c_out = 256;
        let conv = ConvTranspose3D::new(c_in, c_out, 2, 2, 2, 4, 4, 4, 2, 2, 2);

        let fk = FlattenKernel3D {
            stride_h: conv.flat_stride_h,
            stride_w: conv.flat_stride_w,
            kd: 2, kh: 2, kw: 2, c_out: c_in, c_in: c_out,
        };

        // W[c_in, c_out, 2, 2, 2]
        let c_in_pad = c_in.next_power_of_two();
        let c_out_pad = c_out.next_power_of_two();
        let w_size = c_in_pad * c_out_pad * 2 * 2 * 2;
        let w_data: Vec<u64> = (0..w_size).map(|i| (i as u64 % 7) + 1).collect();
        let w_raw = make_witness(vec![c_in, c_out, 2, 2, 2], w_data);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        // X[c_in, 4, 4, 4]
        let x_size = c_in_pad * 4 * 4 * 4;
        let x_data: Vec<u64> = (0..x_size).map(|i| (i as u64 % 13) + 1).collect();
        // Zero out padding channels
        let mut x_data_gl: Vec<GoldilocksField> = x_data.iter().map(|&v| GoldilocksField(v)).collect();
        for c in c_in..c_in_pad {
            for idx in 0..(4*4*4) {
                x_data_gl[idx + c * 4 * 4 * 4] = GoldilocksField(0);
            }
        }
        let x = Witness::new(vec![c_in, 4, 4, 4], x_data_gl, DataType::Uint, 0, Role::Input);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct3d_npow2");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct3d_npow2_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_ct3d_npow2_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose3D with non-power-of-2 channels should verify");
    }

    #[test]
    fn test_depthwise_conv2d_run() {
        // C=2, H=4, W=4, kernel 2x2
        let conv = DepthwiseConv2D::new(2, 2, 2, 4, 4);
        assert_eq!(conv.h_out, 3);
        assert_eq!(conv.w_out, 3);

        // X[2, 4, 4]: two channels
        let x = make_witness(vec![2, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, // c=0
            2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,17, // c=1
        ]);

        // Build W_flat via FlattenKernel with c_in=1
        // W[2, 1, 2, 2]: channel 0 kernel = [1,1,1,1], channel 1 kernel = [1,0,0,1]
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 2, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let w_raw = make_witness(vec![2, 1, 2, 2], vec![
            1,1,1,1,  // c=0 (only c_in=0 slot used)
            1,0,0,1,  // c=1
        ]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        // Reshape W_flat from [2, 1, s_kernel] to [2, s_kernel]
        let s_kernel: usize = (2 - 1) * 4 + 2; // = 6
        let s_kernel_pad = s_kernel.next_power_of_two(); // 8
        let c_pad = 2;
        let mut wf2_data = vec![GoldilocksField(0); c_pad * s_kernel_pad];
        for c in 0..2 {
            for j in 0..s_kernel_pad {
                let src_idx = j + 0 * s_kernel_pad + c * 1 * s_kernel_pad;
                let dst_idx = j + c * s_kernel_pad;
                wf2_data[dst_idx] = wf.data.as_ref().unwrap().index(src_idx);
            }
        }
        let wf2 = Witness::new(vec![2, s_kernel], wf2_data, DataType::Uint, 0, Role::Input);

        let result = conv.run(&[&x, &wf2]);
        let y = &result[0];

        // Y[c=0, ho=0, wo=0] = X[0,0,0]*1 + X[0,0,1]*1 + X[0,1,0]*1 + X[0,1,1]*1 = 1+2+5+6 = 14
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(14));
        // Y[c=0, ho=0, wo=1] = 2+3+6+7 = 18
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(18));
    }

    #[test]
    fn test_depthwise_conv2d_prove_verify() {
        let conv = DepthwiseConv2D::new(2, 2, 2, 4, 4);
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 2, c_in: 1, dilation_h: 1, dilation_w: 1 };

        let x = make_witness(vec![2, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
            2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,17,
        ]);

        let w_raw = make_witness(vec![2, 1, 2, 2], vec![
            1,1,1,1,
            1,0,0,1,
        ]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let s_kernel: usize = (2 - 1) * 4 + 2;
        let s_kernel_pad = s_kernel.next_power_of_two();
        let c_pad = 2;
        let mut wf2_data = vec![GoldilocksField(0); c_pad * s_kernel_pad];
        for c in 0..2 {
            for j in 0..s_kernel_pad {
                let src_idx = j + c * 1 * s_kernel_pad;
                let dst_idx = j + c * s_kernel_pad;
                wf2_data[dst_idx] = wf.data.as_ref().unwrap().index(src_idx);
            }
        }
        let wf2 = Witness::new(vec![2, s_kernel], wf2_data, DataType::Uint, 0, Role::Input);

        let result = conv.run(&[&x, &wf2]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_dw");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_dw_prove");
        let (proofs, new_claims) = conv.prove(
            &[&x, &wf2, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 4);
        assert_eq!(new_claims.len(), 4);

        let mut verify_transcript = Transcript::new(b"test_dw_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &wf2, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "DepthwiseConv2D prove/verify should pass");
    }

    // ================================================================
    // Dilated Conv2D tests
    // ================================================================

    #[test]
    fn test_conv2d_dilated_run() {
        // C_in=1, C_out=1, H=7, W=7, kernel 3×3, dilation=2
        // Effective kernel size: 3 + (3-1)*1 = 5 → output = (7 - 2*(3-1) - 1)/1 + 1 = 3
        let conv = Conv2D::new_dilated(1, 1, 3, 3, 7, 7, 1, 1, 2, 2);
        assert_eq!(conv.h_out, 3);
        assert_eq!(conv.w_out, 3);

        // X[1, 7, 7]: w_pad = 8
        // Fill with incrementing values
        let w_pad = 8;
        let h_pad = 8;
        let mut x_data = vec![0u64; h_pad * w_pad];
        for ih in 0..7 {
            for iw in 0..7 {
                x_data[iw + ih * w_pad] = (ih * 7 + iw + 1) as u64;
            }
        }
        let x = make_witness(vec![1, 7, 7], x_data);

        // W[1,1,3,3]: all 1s for sum kernel
        // FlattenKernel with dilation=2: j = kh*dilation_h*s_w + kw*dilation_w
        // s_w = 8 (w_pad)
        // kh=0,kw=0: j=0; kh=0,kw=1: j=2; kh=0,kw=2: j=4
        // kh=1,kw=0: j=16; kh=1,kw=1: j=18; kh=1,kw=2: j=20
        // kh=2,kw=0: j=32; kh=2,kw=1: j=34; kh=2,kw=2: j=36
        let fk = FlattenKernel { s_w: w_pad, kh: 3, kw: 3, c_out: 1, c_in: 1, dilation_h: 2, dilation_w: 2 };
        // W raw with zero padding to 4x4
        let w_raw = make_witness(vec![1, 1, 3, 3], vec![1,1,1,0, 1,1,1,0, 1,1,1,0, 0,0,0,0]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Y[0,0] = X[0,0]+X[0,2]+X[0,4] + X[2,0]+X[2,2]+X[2,4] + X[4,0]+X[4,2]+X[4,4]
        //        = 1+3+5 + 15+17+19 + 29+31+33 = 153
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(153));

        // Y[0,1] = X[0,1]+X[0,3]+X[0,5] + X[2,1]+X[2,3]+X[2,5] + X[4,1]+X[4,3]+X[4,5]
        //        = 2+4+6 + 16+18+20 + 30+32+34 = 162
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(162));

        // Y[0,2] = X[0,2]+X[0,4]+X[0,6] + X[2,2]+X[2,4]+X[2,6] + X[4,2]+X[4,4]+X[4,6]
        //        = 3+5+7 + 17+19+21 + 31+33+35 = 171
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(171));

        // Y[1,0] at index 4 (w_pad=4): X[1,0]+X[1,2]+X[1,4]+X[3,0]+X[3,2]+X[3,4]+X[5,0]+X[5,2]+X[5,4]
        //        = 8+10+12 + 22+24+26 + 36+38+40 = 216
        assert_eq!(y.data.as_ref().unwrap().index(4), GoldilocksField(216));
    }

    #[test]
    fn test_conv2d_dilated_prove_verify() {
        // C_in=1, C_out=1, H=7, W=7, kernel 3×3, dilation=2
        let conv = Conv2D::new_dilated(1, 1, 3, 3, 7, 7, 1, 1, 2, 2);
        let w_pad = 8usize;
        let h_pad = 8usize;
        let fk = FlattenKernel { s_w: w_pad, kh: 3, kw: 3, c_out: 1, c_in: 1, dilation_h: 2, dilation_w: 2 };

        // X[1, 7, 7]
        let mut x_data = vec![0u64; h_pad * w_pad];
        for ih in 0..7 {
            for iw in 0..7 {
                x_data[iw + ih * w_pad] = (ih * 7 + iw + 1) as u64;
            }
        }
        let x = make_witness(vec![1, 7, 7], x_data);

        // W[1,1,3,3]: kernel values
        let w_raw = make_witness(vec![1, 1, 3, 3], vec![1,2,3,0, 4,5,6,0, 7,8,9,0, 0,0,0,0]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        // Run conv
        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_dilated");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove Conv2D
        let mut prove_transcript = Transcript::new(b"test_dilated_p");
        let (conv_proofs, conv_claims) = conv.prove(
            &[&x, wf, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(conv_proofs.len(), 4);
        assert_eq!(conv_claims.len(), 4);

        // Verify Conv2D
        let mut verify_transcript = Transcript::new(b"test_dilated_p");
        let mut all_conv_claims: Vec<&Claim> = conv_claims.iter().collect();
        all_conv_claims.push(&out_claim);
        let conv_proofs_ref: Vec<&SumcheckProof> = conv_proofs.iter().collect();
        let conv_verified = conv.verify(
            &[&x, wf, y],
            &all_conv_claims,
            &conv_proofs_ref,
            &mut verify_transcript,
        );
        assert!(conv_verified, "Dilated Conv2D prove/verify should pass");

        // Also test FlattenKernel with dilation
        let wf_claim = &conv_claims[2];
        let mut fk_prove_transcript = Transcript::new(b"test_dilated_fk");
        let (fk_proofs, fk_claims) = fk.prove(
            &[&w_raw, wf],
            &[0, 1],
            &[wf_claim],
            &mut fk_prove_transcript,
        );
        assert_eq!(fk_proofs.len(), 1);

        let mut fk_verify_transcript = Transcript::new(b"test_dilated_fk");
        let mut fk_all_claims: Vec<&Claim> = fk_claims.iter().collect();
        fk_all_claims.push(wf_claim);
        let fk_proofs_ref: Vec<&SumcheckProof> = fk_proofs.iter().collect();
        let fk_verified = fk.verify(
            &[&w_raw, wf],
            &fk_all_claims,
            &fk_proofs_ref,
            &mut fk_verify_transcript,
        );
        assert!(fk_verified, "Dilated FlattenKernel prove/verify should pass");
    }
}
