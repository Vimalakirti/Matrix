use std::sync::Arc;

use almost_goldilocks_cuda::{DeviceBuffer, AlmostGoldilocksField, AlmostGoldilocksExt2};
use almost_goldilocks_cuda::conv::zero_buffer;
use almost_goldilocks_cuda::conv::{
    conv2d as gpu_conv2d, conv3d as gpu_conv3d, conv_full as gpu_conv_full,
    depthwise_conv2d as gpu_depthwise_conv2d,
    conv_transpose3d as gpu_conv_transpose3d,
    conv_transpose2d as gpu_conv_transpose2d,
    flatten_kernel2d as gpu_flatten_kernel2d, flatten_kernel3d as gpu_flatten_kernel3d,
};
use rayon::prelude::*;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Witness, DataType, Role};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::grand_product::{
    beta_linear_leaf_eval, prove_grand_product, verify_grand_product, GrandProductProof,
};
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_sub, ext2_mul, ext2_field_eq, log2_ceil, get_n, agl_add, agl_mul};

// Standalone grand-product vs masked-view conv output-binding comparison
// (paper microbenchmark). Child module: reaches Conv2D's private geometry
// helpers + the private conv2d_* mask/eq helpers. Test-only; not in release.
#[cfg(test)]
#[path = "conv_gp_bind.rs"]
mod conv_gp_bind;

#[cfg(test)]
thread_local! {
    /// Per-thread override for [`Conv2D::grand_product_mode`] so unit tests can
    /// exercise the grand-product output-binding path without setting the
    /// process-global `ZK4_CONV_GRANDPRODUCT` env var (which would race the
    /// parallel test runner). `None` ⇒ fall back to the env var.
    static GP_MODE_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only RAII guard: turns grand-product mode on for the current thread and
/// restores the previous state (usually `None`) on drop, so a reused test
/// worker thread does not leak the override into the next test.
#[cfg(test)]
pub(crate) struct GpModeTestGuard(Option<bool>);

#[cfg(test)]
impl GpModeTestGuard {
    pub(crate) fn enable() -> Self {
        let prev = GP_MODE_TEST_OVERRIDE.with(|c| c.replace(Some(true)));
        GpModeTestGuard(prev)
    }
}

#[cfg(test)]
impl Drop for GpModeTestGuard {
    fn drop(&mut self) {
        GP_MODE_TEST_OVERRIDE.with(|c| c.set(self.0));
    }
}

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
    /// Number of images proven together. The weights do not depend on the
    /// batch index, so `Y(b,d,p) = Sum_{c,k} X(b,c,p*s+k)*W(d,c,k)` is linear
    /// in `X` and hence multilinear in `b` on both sides. Two multilinear
    /// polynomials agreeing on the whole hypercube are equal, so the
    /// verifier's `r_b` can be bound BEFORE the conv sumcheck instead of
    /// summed over: the prover folds each batched witness to
    /// `Sum_b eq(b, r_b) * T[b]` and runs the ordinary single-image argument
    /// on the fold. Cost is therefore ~independent of `batch`, and the leaf
    /// count does not grow -- replicating the graph per image instead makes
    /// the fold tree fragment into low-arity buckets, which is superlinear.
    ///
    /// `batch = 1` reduces to the unbatched protocol bit-identically:
    /// `l_b() == 0`, `eq_b == [1]`, and nothing is appended to any claim point.
    pub batch: usize,
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
        Self { c_in, c_out, kernel_h, kernel_w, input_h, input_w, stride_w, s_in, s_kernel, h_out, w_out, conv_stride_h, conv_stride_w, dilation_h, dilation_w, batch: 1 }
    }

    /// Prove `batch` images against one shared set of weights. See the
    /// `batch` field for why this costs almost nothing beyond the fold.
    pub fn with_batch(mut self, batch: usize) -> Self {
        assert!(batch >= 1, "batch must be >= 1");
        self.batch = batch;
        self
    }

    /// Variables carried by the batch index; zero when unbatched, which is
    /// what makes `batch = 1` bit-identical to the pre-batch protocol.
    pub fn l_b(&self) -> usize { log2_ceil(self.batch.max(1)) }

    /// Padded batch extent. A device witness must fill `1<<n` exactly, and `n`
    /// rounds the batch up to a power of two, so allocations use this rather
    /// than `batch` itself.
    pub fn b_pad(&self) -> usize { 1 << self.l_b() }

    /// Padded element count of ONE image's X / Y / Y_full. The batch index is
    /// the most significant dimension, so image `b` starts at `b * stride`.
    fn x_stride(&self) -> usize {
        self.c_in.next_power_of_two()
            * self.input_h.next_power_of_two()
            * self.input_w.next_power_of_two()
    }
    fn y_stride(&self) -> usize {
        self.c_out.next_power_of_two()
            * self.h_out.next_power_of_two()
            * self.w_out.next_power_of_two()
    }
    fn yfull_stride(&self) -> usize {
        self.c_out.next_power_of_two() * self.s_full().next_power_of_two()
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
    /// Length of the full 1D flat convolution (all α-exponents, incl. junk).
    fn s_full(&self) -> usize { self.s_in + self.s_kernel - 1 }
    /// Number of variables for the full-conv exponent dimension.
    fn l_full(&self) -> usize { log2_ceil(self.s_full().max(1)) }
    /// True for 1×1 stride-1 convs: the full conv is junk-free (s_full = s_in
    /// and FullConv[d, m] = Y[d, s_in−1−m] over the whole padded box), so
    /// sumcheck B binds s_alpha_conv directly to the committed Y and no Y_full
    /// aux edge or masked-view sumcheck is needed.
    ///
    /// `ZK4_CONV_NO_FASTPATH=1` forces the general Y_full path even for
    /// junk-free convs (ablation knob; the env var is constant per process so
    /// build/run/prove/verify stay consistent).
    pub fn junk_free(&self) -> bool {
        if std::env::var("ZK4_CONV_NO_FASTPATH").is_ok() {
            return false;
        }
        self.s_kernel == 1 && self.conv_stride_h == 1 && self.conv_stride_w == 1
    }

    /// `ZK4_CONV_GRANDPRODUCT=1` binds a non-junk-free Conv2D's output with the
    /// VerfCNN-style grand-product multiset partition (see `conv_gp_bind`)
    /// instead of the production masked-view sumcheck C — a same-PCS ablation
    /// baseline. The env var is constant per process, so the graph structure
    /// (`out_arity`/builder) and prove/verify all read it consistently within
    /// one process. Read live (never cached) so build/run/prove/verify agree.
    /// Junk-free 1×1 stride-1 convs are unaffected.
    pub fn grand_product_mode(&self) -> bool {
        #[cfg(test)]
        {
            if let Some(v) = GP_MODE_TEST_OVERRIDE.with(|c| c.get()) {
                return v;
            }
        }
        std::env::var("ZK4_CONV_GRANDPRODUCT").is_ok()
    }

    /// Length of the padded "K" leftover advice for the grand-product binding:
    /// the real Y_full coefficients no valid output lands on, padded to a power
    /// of two. `run`/`run_gpu` (which materialize the K witness) and
    /// `conv2d_prove` (which builds idxK) derive from this identical formula so
    /// they agree exactly. Mirrors `conv_gp_bind::build_bind_vectors`' `k_pad`.
    pub(crate) fn gp_k_len(&self) -> usize {
        let count = self.c_out * self.s_full() - self.c_out * self.h_out * self.w_out;
        count.max(1).next_power_of_two()
    }

    /// Full-conv exponent of output position (ho, wo):
    ///   E(ho, wo) = (s_in − 1) − (cs_h·w_pad·ho + cs_w·wo)
    /// (the input is reversed in F, so output at flat input offset t sits at
    /// exponent s_in−1−t of the polynomial product).
    fn view_exponent(&self, ho: usize, wo: usize) -> usize {
        (self.s_in - 1) - (ho * self.conv_stride_h * self.stride_w + wo * self.conv_stride_w)
    }

    /// Bit layout of the crop map E for the bit-affine view: returns
    /// (wo_shift, ho_shift, l_si). Valid only when both conv strides are
    /// powers of two, so E's subtrahend has carry-free disjoint bit fields:
    /// wo bits occupy [wo_shift, wo_shift+l_wo), ho bits occupy
    /// [ho_shift, ho_shift+l_ho), and E = bitwise complement over the low
    /// l_si bits (s_in−1 is all-ones). Asserts the disjointness invariants
    /// the soundness of the view depends on.
    fn view_bit_layout(&self) -> (usize, usize, usize) {
        assert!(
            self.conv_stride_h.is_power_of_two() && self.conv_stride_w.is_power_of_two(),
            "Conv2D bit-affine view requires power-of-two conv strides, got ({}, {})",
            self.conv_stride_h, self.conv_stride_w
        );
        let lw = log2_ceil(self.input_w.max(1)); // log2(w_pad)
        let lh = log2_ceil(self.input_h.max(1)); // log2(h_pad)
        let l_si = lw + lh; // log2(s_in)
        let wo_shift = self.conv_stride_w.trailing_zeros() as usize;
        let ho_shift = lw + self.conv_stride_h.trailing_zeros() as usize;
        // Bit fields must be disjoint and within the s_in range for E to be
        // bit-affine over the whole padded output box.
        assert!(wo_shift + self.l_wo() <= ho_shift,
            "Conv2D view: wo bit-field [{}+{}) overlaps ho bit-field at {}",
            wo_shift, self.l_wo(), ho_shift);
        assert!(ho_shift + self.l_ho() <= l_si,
            "Conv2D view: ho bit-field [{}+{}) exceeds s_in bits {}",
            ho_shift, self.l_ho(), l_si);
        (wo_shift, ho_shift, l_si)
    }

    /// σ̃: map a point over Y's output-spatial vars to the Y_full exponent
    /// point such that view(Y)(r) = Y_full(σ̃(r), r_d). Each exponent bit is
    /// either a complemented spatial coordinate or a public constant.
    fn view_point(&self, r_spatial: &[AlmostGoldilocksExt2]) -> Vec<AlmostGoldilocksExt2> {
        let (wo_shift, ho_shift, l_si) = self.view_bit_layout();
        let l_wo = self.l_wo();
        let l_ho = self.l_ho();
        assert_eq!(r_spatial.len(), l_wo + l_ho);
        let one = AlmostGoldilocksExt2::one();
        let mut point = vec![one; self.l_full()];
        // Bits ≥ l_si of E are zero (E < s_in).
        for coord in point.iter_mut().skip(l_si) {
            *coord = AlmostGoldilocksExt2::zero();
        }
        for i in 0..l_wo {
            point[wo_shift + i] = ext2_sub(one, r_spatial[i]);
        }
        for i in 0..l_ho {
            point[ho_shift + i] = ext2_sub(one, r_spatial[l_wo + i]);
        }
        point
    }

    /// Compute the aux Y_full witness: the full 1D flat convolution
    ///   Y_full[d, m] = Σ_c Σ_{i+j=m} X_rev[c,i]·W_flat[d,c,j]
    /// scattered as m = (s_in−1−p) + j over real input positions p and real
    /// kernel taps j. Challenge-independent, committed as advice
    /// (Role::Auxiliary) and bound to s_alpha_conv by sumcheck B.
    fn compute_y_full(
        &self,
        x_slice: &[AlmostGoldilocksField],
        w_slice: &[AlmostGoldilocksField],
        out_sf: usize,
    ) -> Witness {
        let c_in_pad = self.c_in.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();

        let mut yfull = vec![AlmostGoldilocksField(0); c_out_pad * s_full_pad];
        let rows: Vec<(usize, Vec<AlmostGoldilocksField>)> = (0..self.c_out)
            .into_par_iter()
            .map(|d| {
                let mut row = vec![AlmostGoldilocksField(0); s_full_pad];
                for c in 0..self.c_in {
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let j = kh * self.dilation_h * self.stride_w + kw * self.dilation_w;
                            let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                            let w_val = w_slice[wf_idx];
                            for ih in 0..self.input_h {
                                for iw in 0..self.input_w {
                                    let p = ih * self.stride_w + iw;
                                    let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                                    let m = (self.s_in - 1 - p) + j;
                                    row[m] = agl_add(row[m], agl_mul(x_slice[x_idx], w_val));
                                }
                            }
                        }
                    }
                }
                (d, row)
            })
            .collect();
        for (d, row) in rows {
            yfull[d * s_full_pad..d * s_full_pad + s_full_pad].copy_from_slice(&row);
        }
        Witness::new(
            vec![self.c_out, s_full],
            yfull,
            DataType::Uint,
            out_sf,
            Role::Auxiliary,
        )
    }

    /// `compute_y_full` for a batched X, laid out to match Y's `[batch, ...]`
    /// shape. Each image is an independent full convolution against the same
    /// weights, so this is the per-image routine run `batch` times into
    /// disjoint slices; no cross-image term exists to get wrong.
    fn compute_y_full_batched(
        &self,
        x_slice: &[AlmostGoldilocksField],
        w_slice: &[AlmostGoldilocksField],
        out_sf: usize,
    ) -> Witness {
        let batch = self.batch.max(1);
        let s_full = self.s_full();
        let stride = self.yfull_stride();
        let x_stride = self.x_stride();
        // Pad the batch axis to a power of two: the MLE lives over 1<<n slots,
        // and the GPU path allocates the same extent, so a non-pow2 batch must
        // not leave the two representations different lengths.
        let mut all = vec![AlmostGoldilocksField(0); stride * self.b_pad()];
        for b in 0..batch {
            let one = self.compute_y_full(
                &x_slice[b * x_stride..(b + 1) * x_stride],
                w_slice,
                out_sf,
            );
            let d = one.data.as_ref().unwrap().evaluations_ref();
            let n = d.len().min(stride);
            all[b * stride..b * stride + n].copy_from_slice(&d[..n]);
        }
        Witness::new(
            vec![self.b_pad() * self.c_out.next_power_of_two(), s_full],
            all,
            DataType::Uint,
            out_sf,
            Role::Auxiliary,
        )
    }

    /// Materialize the grand-product "K" leftover witness from Y_full's host
    /// evaluations. Iterates the real Y_full positions (`d` outer, `m` inner),
    /// skipping exponents any valid output lands on, collecting the rest and
    /// padding with 0 to `gp_k_len()`. This ordering MUST match the `idxK`
    /// vector `conv2d_prove` builds (both derive from the identical loop).
    fn build_k_witness(
        &self,
        yfull_ev: &[AlmostGoldilocksField],
        s_full: usize,
        s_full_pad: usize,
        out_sf: usize,
    ) -> Witness {
        let mut used = vec![false; self.c_out * s_full_pad];
        for d in 0..self.c_out {
            for ho in 0..self.h_out {
                for wo in 0..self.w_out {
                    used[d * s_full_pad + self.view_exponent(ho, wo)] = true;
                }
            }
        }
        let k_len = self.gp_k_len();
        let mut k_data = Vec::with_capacity(k_len);
        for d in 0..self.c_out {
            for m in 0..s_full {
                if !used[d * s_full_pad + m] {
                    k_data.push(yfull_ev[d * s_full_pad + m]);
                }
            }
        }
        debug_assert_eq!(
            self.c_out * self.h_out * self.w_out + k_data.len(),
            self.c_out * s_full,
            "Conv2D GP: valid+leftover must tile the real Y_full domain"
        );
        k_data.resize(k_len, AlmostGoldilocksField(0));
        Witness::new(vec![k_len], k_data, DataType::Uint, out_sf, Role::Auxiliary)
    }
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
        // Borrow host slices ONCE; the inner c_in·kh·kw loop indexed these via a
        // trait `.index()` (virtual dispatch + OnceLock host check) per element.
        let x_slice = x_data.evaluations_ref();
        let w_slice = w_data.evaluations_ref();
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

        // The batch index is the most significant dimension of X and Y, and the
        // weights are shared across it, so image `b` is an independent conv
        // between offsets `b * x_stride` and `b * y_stride`.
        let batch = self.batch.max(1);
        let x_stride = self.x_stride();
        let y_stride = self.y_stride();
        let total_outputs = c_out * h_out * w_out;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size * batch];
        let results: Vec<(usize, AlmostGoldilocksField)> = (0..batch * total_outputs)
            .into_par_iter()
            .map(|bflat| {
                let b = bflat / total_outputs;
                let flat_idx = bflat % total_outputs;
                let wo = flat_idx % w_out;
                let ho = (flat_idx / w_out) % h_out;
                let d = flat_idx / (w_out * h_out);
                let mut acc = AlmostGoldilocksField(0);
                for c in 0..c_in {
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = ho * conv_stride_h + kh * dilation_h;
                            let iw = wo * conv_stride_w + kw * dilation_w;
                            let x_idx = b * x_stride
                                + iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let j = kh * dilation_h * stride_w_val + kw * dilation_w;
                            let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                            let x_val = x_slice[x_idx];
                            let w_val = w_slice[wf_idx];
                            acc = agl_add(acc, agl_mul(x_val, w_val));
                        }
                    }
                }
                let out_idx =
                    b * y_stride + wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                (out_idx, acc)
            })
            .collect();
        for (idx, val) in results {
            out_data[idx] = val;
        }

        // Folded batch layout: [b_pad*c_out_pad, H_out, W_out]. Same MLE and
        // same variables as [B, C_out, H_out, W_out] -- b sits in the high bits
        // of the leading index -- and it is the shape every per-channel block
        // downstream already emits, so shapes stay consistent across the graph.
        let out_shape = if batch > 1 {
            vec![self.b_pad() * self.c_out.next_power_of_two(), self.h_out, self.w_out]
        } else {
            vec![self.c_out, self.h_out, self.w_out]
        };
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);

        if self.junk_free() {
            // 1×1 stride-1 fast path: junk-free, no Y_full aux (see junk_free()).
            return vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output)];
        }

        // Aux output: Y_full, the full 1D flat convolution. Sumcheck B binds
        // it to s_alpha_conv (and hence to X, W); the masked-view sumcheck C
        // binds Y to its valid coefficients.
        let y_full = if batch > 1 {
            self.compute_y_full_batched(x_slice, w_slice, out_sf)
        } else {
            self.compute_y_full(x_slice, w_slice, out_sf)
        };
        #[cfg(debug_assertions)]
        {
            // Valid coefficients of Y_full must reproduce Y exactly.
            let s_full_pad = self.s_full().next_power_of_two();
            let yf = y_full.data.as_ref().unwrap().evaluations_ref();
            for d in 0..self.c_out {
                for ho in 0..self.h_out {
                    for wo in 0..self.w_out {
                        let out_idx = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                        let e = self.view_exponent(ho, wo);
                        debug_assert_eq!(
                            out_data[out_idx], yf[d * s_full_pad + e],
                            "Conv2D: Y_full valid coefficient mismatch at (d={d}, ho={ho}, wo={wo})"
                        );
                    }
                }
            }
        }

        if self.grand_product_mode() {
            // Third aux: K, the "leftover" advice for the grand-product output
            // binding — the real Y_full coefficients no valid output lands on,
            // in a FIXED order (d outer, m inner) that must match idxK in
            // `conv2d_prove`. Skip exponents hit by a valid output.
            let s_full = self.s_full();
            let s_full_pad = s_full.next_power_of_two();
            let yf = y_full.data.as_ref().unwrap().evaluations_ref();
            let k_wit = self.build_k_witness(yf, s_full, s_full_pad, out_sf);
            return vec![
                Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output),
                y_full,
                k_wit,
            ];
        }

        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output), y_full]
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
        // One grid covers the whole batch; image b lives at b*stride in each
        // buffer. The padded regions between images must be zero, so the
        // memset covers the full batched allocation.
        let batch = self.batch.max(1);
        let x_stride = self.x_stride();
        let y_stride = self.y_stride();
        // A device witness must fill 1<<n exactly, and n rounds the batch up to
        // a power of two. Allocate the padded extent and zero it; the kernel
        // writes only b < batch, so the padding images stay zero.
        let b_pad = self.b_pad();

        let d_x = x.as_device_buf();
        let d_w = w_flat.as_device_buf();
        let mut d_y =
            DeviceBuffer::<u64>::new(out_size * b_pad).expect("Conv2D: alloc out");
        // Zero pad regions outside [c_out, h_out, w_out], for every image.
        zero_buffer(&mut d_y, out_size * b_pad).expect("Conv2D: memset zero failed");

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
            batch, x_stride, y_stride,
        ).expect("Conv2D: gpu kernel failed");

        // Folded batch layout, matching the CPU path and every per-channel
        // block downstream: [b_pad*c_out_pad, H_out, W_out].
        let out_shape = if batch > 1 {
            vec![self.b_pad() * self.c_out.next_power_of_two(), self.h_out, self.w_out]
        } else {
            vec![self.c_out, self.h_out, self.w_out]
        };
        let out_sf = inputs[0].sf + inputs[1].sf;

        if self.junk_free() {
            return vec![Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output)];
        }

        // Aux Y_full gathered on-device: one thread per (d, m) exponent, tap
        // index j = kh·dilation_h·stride_w + kw·dilation_w (see
        // `agl_conv_full_kernel`). Matches `compute_y_full` element-wise.
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let yf_stride = self.yfull_stride();
        let mut d_yf = DeviceBuffer::<u64>::new(yf_stride * b_pad)
            .expect("Conv2D: alloc Y_full");
        zero_buffer(&mut d_yf, yf_stride * b_pad)
            .expect("Conv2D: memset Y_full zero failed");
        gpu_conv_full(
            &d_x, &d_w, &mut d_yf,
            self.c_out, self.c_in,
            1, self.kernel_h, self.kernel_w,
            0, self.dilation_h * self.stride_w, self.dilation_w,
            self.s_in, s_full, s_full_pad,
            c_in_pad, s_kernel_pad,
            false,
            batch, x_stride, yf_stride,
        ).expect("Conv2D: gpu Y_full kernel failed");

        let yf_shape = if batch > 1 {
            vec![self.b_pad() * self.c_out.next_power_of_two(), s_full]
        } else {
            vec![self.c_out, s_full]
        };
        let y_full = Witness::new_device(yf_shape, Arc::new(d_yf), DataType::Uint, out_sf, Role::Auxiliary);

        if self.grand_product_mode() {
            // Third aux: K leftover advice. Read Y_full's coefficients to host
            // (lazy device download) and build K with the same fixed ordering
            // as `run` / `conv2d_prove`.
            let yf = y_full.data.as_ref().unwrap().evaluations_ref();
            let k_wit = self.build_k_witness(yf, s_full, s_full_pad, out_sf);
            return vec![
                Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output),
                y_full,
                k_wit,
            ];
        }

        vec![
            Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output),
            y_full,
        ]
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
    /// Length of the full 1D convolution (all α-exponents, incl. junk).
    fn s_full(&self) -> usize { self.s_in + self.s_kernel - 1 }
    /// Number of variables for the full-conv exponent dimension.
    fn l_full(&self) -> usize { log2_ceil(self.s_full().max(1)) }

    /// Full-conv exponent of output position lo:
    ///   E(lo) = (s_in − 1) − cs·lo
    /// (the input is reversed in F, so output at input offset t sits at
    /// exponent s_in−1−t of the polynomial product).
    fn view_exponent(&self, lo: usize) -> usize {
        (self.s_in - 1) - lo * self.conv_stride
    }

    /// Bit layout of the crop map E for the bit-affine view: returns
    /// (lo_shift, l_si). Valid only when the conv stride is a power of two,
    /// so E's subtrahend is carry-free: lo bits occupy [lo_shift,
    /// lo_shift+l_lo), and E = bitwise complement over the low l_si bits
    /// (s_in−1 is all-ones). Asserts the containment invariant the soundness
    /// of the view depends on.
    fn view_bit_layout(&self) -> (usize, usize) {
        assert!(
            self.conv_stride.is_power_of_two(),
            "Conv1D bit-affine view requires a power-of-two conv stride, got {}",
            self.conv_stride
        );
        let l_si = log2_ceil(self.input_len.max(1)); // log2(s_in)
        let lo_shift = self.conv_stride.trailing_zeros() as usize;
        // The lo bit-field must fit within the s_in range for E to be
        // bit-affine over the whole padded output box.
        assert!(lo_shift + self.l_lo() <= l_si,
            "Conv1D view: lo bit-field [{}+{}) exceeds s_in bits {}",
            lo_shift, self.l_lo(), l_si);
        (lo_shift, l_si)
    }

    /// σ̃: map a point over Y's output-spatial vars to the Y_full exponent
    /// point such that view(Y)(r) = Y_full(σ̃(r), r_d). Each exponent bit is
    /// either a complemented spatial coordinate or a public constant.
    fn view_point(&self, r_spatial: &[AlmostGoldilocksExt2]) -> Vec<AlmostGoldilocksExt2> {
        let (lo_shift, l_si) = self.view_bit_layout();
        let l_lo = self.l_lo();
        assert_eq!(r_spatial.len(), l_lo);
        let one = AlmostGoldilocksExt2::one();
        let mut point = vec![one; self.l_full()];
        // Bits ≥ l_si of E are zero (E < s_in).
        for coord in point.iter_mut().skip(l_si) {
            *coord = AlmostGoldilocksExt2::zero();
        }
        for i in 0..l_lo {
            point[lo_shift + i] = ext2_sub(one, r_spatial[i]);
        }
        point
    }

    /// Compute the aux Y_full witness: the full 1D convolution
    ///   Y_full[d, m] = Σ_c Σ_{i+j=m} X_rev[c,i]·W[d,c,j]
    /// scattered as m = (s_in−1−il) + k over real input positions il and real
    /// kernel taps k. Challenge-independent, committed as advice
    /// (Role::Auxiliary) and bound to s_alpha_conv by sumcheck B.
    fn compute_y_full(
        &self,
        x_slice: &[AlmostGoldilocksField],
        w_slice: &[AlmostGoldilocksField],
        out_sf: usize,
    ) -> Witness {
        let c_in_pad = self.c_in.next_power_of_two();
        let l_in_pad = self.input_len.next_power_of_two();
        let k_pad = self.kernel_size.next_power_of_two();
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();

        let mut yfull = vec![AlmostGoldilocksField(0); c_out_pad * s_full_pad];
        let rows: Vec<(usize, Vec<AlmostGoldilocksField>)> = (0..self.c_out)
            .into_par_iter()
            .map(|d| {
                let mut row = vec![AlmostGoldilocksField(0); s_full_pad];
                for c in 0..self.c_in {
                    for k in 0..self.kernel_size {
                        let w_idx = k + c * k_pad + d * k_pad * c_in_pad;
                        let w_val = w_slice[w_idx];
                        for il in 0..self.input_len {
                            let x_idx = il + c * l_in_pad;
                            let m = (self.s_in - 1 - il) + k;
                            row[m] = agl_add(row[m], agl_mul(x_slice[x_idx], w_val));
                        }
                    }
                }
                (d, row)
            })
            .collect();
        for (d, row) in rows {
            yfull[d * s_full_pad..d * s_full_pad + s_full_pad].copy_from_slice(&row);
        }
        Witness::new(
            vec![self.c_out, s_full],
            yfull,
            DataType::Uint,
            out_sf,
            Role::Auxiliary,
        )
    }
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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        for d in 0..self.c_out {
            for lo in 0..self.l_out {
                let mut acc = AlmostGoldilocksField(0);
                for c in 0..self.c_in {
                    for k in 0..self.kernel_size {
                        let il = lo * self.conv_stride + k;
                        // X index: l_in bits (lowest) | c_in bits
                        let x_idx = il + c * l_in_pad;
                        // W index: k bits (lowest) | c_in bits | c_out bits
                        let w_idx = k + c * k_pad + d * k_pad * c_in_pad;
                        let x_val = x.data.as_ref().unwrap().index(x_idx);
                        let w_val = w.data.as_ref().unwrap().index(w_idx);
                        acc = agl_add(acc, agl_mul(x_val, w_val));
                    }
                }
                let out_idx = lo + d * l_out_pad;
                out_data[out_idx] = acc;
            }
        }

        let out_shape = vec![self.c_out, self.l_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);

        // Aux output: Y_full, the full 1D convolution. Sumcheck B binds it to
        // s_alpha_conv (and hence to X, W); the masked-view sumcheck C binds
        // Y to its valid coefficients.
        let y_full = self.compute_y_full(
            x.data.as_ref().unwrap().evaluations_ref(),
            w.data.as_ref().unwrap().evaluations_ref(),
            out_sf,
        );
        #[cfg(debug_assertions)]
        {
            // Valid coefficients of Y_full must reproduce Y exactly.
            let s_full_pad = self.s_full().next_power_of_two();
            let yf = y_full.data.as_ref().unwrap().evaluations_ref();
            for d in 0..self.c_out {
                for lo in 0..self.l_out {
                    let out_idx = lo + d * l_out_pad;
                    let e = self.view_exponent(lo);
                    debug_assert_eq!(
                        out_data[out_idx], yf[d * s_full_pad + e],
                        "Conv1D: Y_full valid coefficient mismatch at (d={d}, lo={lo})"
                    );
                }
            }
        }

        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output), y_full]
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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        // Borrow the weight's host slice ONCE instead of a per-element trait
        // `.index()` (virtual dispatch + OnceLock host check each call) — this
        // permute touches c_out·c_in·kh·kw elements per conv. Pure rearrange,
        // byte-identical.
        let w_slice = w.data.as_ref().unwrap().evaluations_ref();
        for d in 0..self.c_out {
            for c in 0..self.c_in {
                for kh in 0..self.kh {
                    for kw in 0..self.kw {
                        // W little-endian: kw bits (lowest) | kh bits | c_in bits | c_out bits
                        let w_idx = kw + kh * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                        let j = kh * self.dilation_h * self.s_w + kw * self.dilation_w;
                        // W_flat little-endian: j bits (lowest) | c_in bits | c_out bits
                        let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                        out_data[wf_idx] = w_slice[w_idx];
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output)]
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
        zero_buffer(&mut d_wf, out_size).expect("FlattenKernel: memset");

        gpu_flatten_kernel2d(
            &d_w, &mut d_wf,
            self.c_out, self.c_in, self.kh, self.kw,
            kw_pad, kh_pad,
            c_in_pad, s_kernel_pad,
            self.dilation_h, self.dilation_w, self.s_w,
        ).expect("FlattenKernel: gpu kernel failed");

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        vec![Witness::new_device(out_shape, Arc::new(d_wf), DataType::Uint, inputs[0].sf, Role::Output)]
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
fn alpha_power_table(alpha: AlmostGoldilocksExt2, size: usize) -> Vec<AlmostGoldilocksExt2> {
    let mut table = Vec::with_capacity(size);
    let mut pow = AlmostGoldilocksExt2::one();
    for _ in 0..size {
        table.push(pow);
        pow = ext2_mul(pow, alpha);
    }
    table
}

/// Evaluate the α-table MLE at point r.
/// α_table_mle(r) = Π_j(1 + r_j*(α^{2^j} - 1))
fn alpha_table_mle_eval(alpha: AlmostGoldilocksExt2, r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let one = AlmostGoldilocksExt2::one();
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

/// Evaluate eq(a, b) = Π_i ((1−a_i)(1−b_i) + a_i·b_i) for two points.
fn eq_points_ext2(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    assert_eq!(a.len(), b.len());
    let one = AlmostGoldilocksExt2::one();
    let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
    let mut prod = one;
    for i in 0..a.len() {
        // (1−a−b) + 2ab
        let term = ext2_add(
            ext2_sub(one, ext2_add(a[i], b[i])),
            ext2_mul(two, ext2_mul(a[i], b[i])),
        );
        prod = ext2_mul(prod, term);
    }
    prod
}

/// Evaluate the MLE of the indicator [x < bound] over `r.len()` little-endian
/// bits at point r. Standard binary comparison decomposition:
///   [x < c] = Σ_{p: c_p=1} (1−x_p) · Π_{q>p} eq_bit(x_q, c_q)
/// evaluated MSB→LSB with a running prefix product. O(len) time.
fn lt_mask_mle_eval(bound: usize, r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let len = r.len();
    assert!(bound <= (1usize << len), "lt_mask_mle_eval: bound out of range");
    if bound == (1usize << len) {
        return AlmostGoldilocksExt2::one();
    }
    let one = AlmostGoldilocksExt2::one();
    let mut result = AlmostGoldilocksExt2::zero();
    let mut prefix = one; // Π over already-visited higher bits of eq_bit(r_q, c_q)
    for p in (0..len).rev() {
        let c_p = (bound >> p) & 1;
        if c_p == 1 {
            result = ext2_add(result, ext2_mul(prefix, ext2_sub(one, r[p])));
            prefix = ext2_mul(prefix, r[p]); // eq_bit(r_p, 1) = r_p
        } else {
            prefix = ext2_mul(prefix, ext2_sub(one, r[p])); // eq_bit(r_p, 0) = 1−r_p
        }
    }
    result
}

/// mask MLE for Conv2D's masked-view sumcheck: indicator of the real output
/// region [wo < w_out]·[ho < h_out]·[d < c_out], point layout
/// (wo bits | ho bits | d bits) little-endian.
fn conv2d_mask_mle_eval(conv: &Conv2D, r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let l_wo = conv.l_wo();
    let l_ho = conv.l_ho();
    let l_d = conv.l_d();
    assert_eq!(r.len(), l_wo + l_ho + l_d);
    let m_wo = lt_mask_mle_eval(conv.w_out, &r[..l_wo]);
    let m_ho = lt_mask_mle_eval(conv.h_out, &r[l_wo..l_wo + l_ho]);
    let m_d = lt_mask_mle_eval(conv.c_out, &r[l_wo + l_ho..]);
    ext2_mul(ext2_mul(m_wo, m_ho), m_d)
}

/// mask MLE for Conv1D's masked-view sumcheck: indicator of the real output
/// region [lo < l_out]·[d < c_out], point layout (lo bits | d bits)
/// little-endian.
fn conv1d_mask_mle_eval(conv: &Conv1D, r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let l_lo = conv.l_lo();
    let l_d = conv.l_d();
    assert_eq!(r.len(), l_lo + l_d);
    let m_lo = lt_mask_mle_eval(conv.l_out, &r[..l_lo]);
    let m_d = lt_mask_mle_eval(conv.c_out, &r[l_lo..]);
    ext2_mul(m_lo, m_d)
}

/// mask MLE for Conv3D's masked-view sumcheck: indicator of the real output
/// region [wo < w_out]·[ho < h_out]·[do < d_out]·[d < c_out], point layout
/// (wo bits | ho bits | do bits | d bits) little-endian.
fn conv3d_mask_mle_eval(conv: &Conv3D, r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let l_wo = conv.l_wo();
    let l_ho = conv.l_ho();
    let l_do = conv.l_do();
    let l_d = conv.l_d();
    assert_eq!(r.len(), l_wo + l_ho + l_do + l_d);
    let m_wo = lt_mask_mle_eval(conv.w_out, &r[..l_wo]);
    let m_ho = lt_mask_mle_eval(conv.h_out, &r[l_wo..l_wo + l_ho]);
    let m_do = lt_mask_mle_eval(conv.d_out, &r[l_wo + l_ho..l_wo + l_ho + l_do]);
    let m_d = lt_mask_mle_eval(conv.c_out, &r[l_wo + l_ho + l_do..]);
    ext2_mul(ext2_mul(ext2_mul(m_wo, m_ho), m_do), m_d)
}

/// mask MLE for DepthwiseConv2D's masked-view sumcheck: indicator of the real
/// output region [wo < w_out]·[ho < h_out]·[c < channels], point layout
/// (wo bits | ho bits | c bits) little-endian.
fn depthwise_conv2d_mask_mle_eval(conv: &DepthwiseConv2D, r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let l_wo = conv.l_wo();
    let l_ho = conv.l_ho();
    let l_c = conv.l_c();
    assert_eq!(r.len(), l_wo + l_ho + l_c);
    let m_wo = lt_mask_mle_eval(conv.w_out, &r[..l_wo]);
    let m_ho = lt_mask_mle_eval(conv.h_out, &r[l_wo..l_wo + l_ho]);
    let m_c = lt_mask_mle_eval(conv.channels, &r[l_wo + l_ho..]);
    ext2_mul(ext2_mul(m_wo, m_ho), m_c)
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
    let mut w_partial = vec![AlmostGoldilocksExt2::zero(); sumcheck_size];
    for d in 0..fk.c_out {
        for c in 0..fk.c_in {
            let dc_weight = ext2_mul(eq_d[d], eq_c[c]);
            for kh in 0..fk.kh {
                for kw in 0..fk.kw {
                    let w_idx = kw + kh * kw_pad + c * kw_pad * kh_pad + d * kw_pad * kh_pad * c_in_pad;
                    let w_val = AlmostGoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                    let sc_idx = kw + kh * kw_pad;
                    w_partial[sc_idx] = ext2_add(w_partial[sc_idx], ext2_mul(dc_weight, w_val));
                }
            }
        }
    }

    // Build H[kh, kw] = eq(r_j, kh*dilation_h*S_w + kw*dilation_w)
    // H is indexed in little-endian: kw bits (lowest), kh bits
    let mut h_poly = vec![AlmostGoldilocksExt2::zero(); sumcheck_size];
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

    let mut h_eval = AlmostGoldilocksExt2::zero();
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
    // edge_ids: [x_edge, w_edge, y_edge, yfull_edge]
    // witnesses: [X, W, Y, Y_full]
    let x_edge = edge_ids[0];
    let w_edge = edge_ids[1];
    let y_edge = edge_ids[2];
    let yfull_edge = edge_ids[3];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_lo = conv.l_lo();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

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
    let mut yp = vec![AlmostGoldilocksExt2::zero(); l_out_pad];
    for d in 0..conv.c_out {
        for lo in 0..conv.l_out {
            let y_idx = lo + d * l_out_pad;
            let y_val = AlmostGoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
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

    // ---- Sumcheck B: bind s_alpha_conv to committed Y_full ----
    // Σ_{d,m} eq_D[d]·α^m·Y_full[d,m] = s_alpha_conv.
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // the committed full conv rather than a free prover scalar. Together with
    // sumchecks 2/3/4 (which bind the same sum to X, W via F·G),
    // Schwartz-Zippel over α gives Y_full = FullConv(X, W) coefficient-wise.
    let yfull_data = witnesses[3];
    let s_full = conv.s_full();
    let s_full_pad = s_full.next_power_of_two();
    let yfull_slice = yfull_data.data.as_ref().unwrap().evaluations_ref();
    let mut yfp = vec![AlmostGoldilocksExt2::zero(); s_full_pad];
    for d in 0..conv.c_out {
        for m in 0..s_full {
            let v = AlmostGoldilocksExt2::from_base(yfull_slice[d * s_full_pad + m]);
            yfp[m] = ext2_add(yfp[m], ext2_mul(eq_d[d], v));
        }
    }
    let alpha_full = alpha_power_table(alpha, s_full_pad);

    let mut prover_b = CpuLinearSumcheckProverExt2::new(l_full, 2, transcript);
    let proof_b = prover_b.prove(&mut [alpha_full, yfp].as_mut_slice(), transcript);
    let r_m = prover_b.challenges.clone();

    // YFP(r_m) = Y_full(r_m, r_d)
    let mut yfull_point_b = Vec::with_capacity(l_full + l_d);
    yfull_point_b.extend_from_slice(&r_m);
    yfull_point_b.extend_from_slice(r_d);
    let yfull_claim_b = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_b,
        eval: prover_b.final_eval(1),
    };

    // ---- Sumcheck 2: Channel F×G ----
    // F[c] = Σ_i X_rev[c,i]·α^i, G[c] = Σ_k W_partial[c,k]·α^k
    // Σ_c F[c]·G[c] = s_alpha_conv (bound to Y_full by sumcheck B)

    // Build WP[c, k] = Σ_d W[d, c, k] · eq_D[d]
    let w_data = witnesses[1];
    let mut wp = vec![AlmostGoldilocksExt2::zero(); c_in_pad * k_pad];
    for d in 0..conv.c_out {
        for c in 0..conv.c_in {
            for k in 0..conv.kernel_size {
                let w_idx = k + c * k_pad + d * k_pad * c_in_pad;
                let w_val = AlmostGoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                wp[c * k_pad + k] = ext2_add(wp[c * k_pad + k], ext2_mul(eq_d[d], w_val));
            }
        }
    }

    // Build G[c] = Σ_k WP[c, k] · α^k
    let alpha_kernel = alpha_power_table(alpha, k_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for k in 0..conv.kernel_size {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * k_pad + k], alpha_kernel[k]));
        }
    }

    // Build F[c] = Σ_i X_rev[c, i] · α^i
    // X_rev[c, i] = X[c, s_in - 1 - i]
    let x_data = witnesses[0];
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for il in 0..conv.input_len {
            let x_idx = il + c * s_in_pad;
            let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            let rev_i = conv.s_in - 1 - il;
            f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
        }
    }

    // Σ_c F[c]·G[c] = Σ_m FullConv[m]·α^m — the same sum sumcheck B bound to
    // the committed Y_full. The verifier checks the two sumchecks' implied
    // sums against each other, so no scalar travels outside the transcript.

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F reduction to X claim ----
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for il in 0..conv.input_len {
            let x_idx = il + c * s_in_pad;
            let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            let rev_i = conv.s_in - 1 - il;
            xp[rev_i] = ext2_add(xp[rev_i], ext2_mul(eq_c[c], x_val));
        }
    }

    let alpha_poly_in = alpha_power_table(alpha, s_in_pad);

    let mut prover3 = CpuLinearSumcheckProverExt2::new(l_spatial_in, 2, transcript);
    let proof3 = prover3.prove(&mut [alpha_poly_in, xp].as_mut_slice(), transcript);
    let r_i = prover3.challenges.clone();

    // X_rev(r_c, r_i) = X(r_c, 1 - r_i)
    let one = AlmostGoldilocksExt2::one();
    let r_spatial_x: Vec<AlmostGoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

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
    let mut wpp = vec![AlmostGoldilocksExt2::zero(); k_pad];
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

    // ---- Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // At the (already random) y_self point r*:
    //   Y(r*) = Σ_x eq(r*, x) · mask(x) · Y_full[d(x), E(lo(x))]
    // mask = [lo<l_out]·[d<c_out] pins Y's real region to the valid full-conv
    // coefficients AND its padded region to zero. Degree-3 sumcheck; the view
    // factor exits as a Y_full claim at the bit-affine point σ̃(r').
    let r_star = &y_self_claim.point;
    let l_out_n = l_lo + l_d;
    let eq_star = evaluate_lagrange_basis_ext2(r_star);
    debug_assert_eq!(eq_star.len(), 1 << l_out_n);

    let c_out_pad = conv.c_out.next_power_of_two();
    let mut mask_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    let mut view_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    for d in 0..c_out_pad {
        for lo in 0..l_out_pad {
            let x_lin = lo + d * l_out_pad;
            let e = conv.view_exponent(lo);
            view_tab[x_lin] =
                AlmostGoldilocksExt2::from_base(yfull_slice[d * s_full_pad + e]);
            if d < conv.c_out && lo < conv.l_out {
                mask_tab[x_lin] = AlmostGoldilocksExt2::one();
            }
        }
    }

    let mut prover_c = CpuLinearSumcheckProverExt2::new(l_out_n, 3, transcript);
    let proof_c = prover_c.prove(&mut [eq_star, mask_tab, view_tab].as_mut_slice(), transcript);
    let r_prime = prover_c.challenges.clone();

    // view MLE at r' = Y_full(σ̃(r'_lo), r'_d)
    let mut yfull_point_c = conv.view_point(&r_prime[..l_lo]);
    yfull_point_c.extend_from_slice(&r_prime[l_lo..]);
    let yfull_claim_c = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_c,
        eval: prover_c.final_eval(2),
    };

    // Return: 6 proofs (transcript order), claims =
    // [y_self_claim, x_claim, w_claim, yfull_claim_b, yfull_claim_c]
    (
        vec![proof1, proof_b, proof2, proof3, proof4, proof_c],
        vec![y_self_claim, x_claim, w_claim, yfull_claim_b, yfull_claim_c],
    )
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
    // claims layout: [y_self_claim, x_claim, w_claim, yfull_claim_b, yfull_claim_c, out_claim]
    // proofs layout (transcript order): [p1, pB, p2, p3, p4, pC]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let w_claim = claims[2];
    let yfull_claim_b = claims[3];
    let yfull_claim_c = claims[4];

    let l_lo = conv.l_lo();
    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

    let r_lo = &out_claim.point[..l_lo];
    let r_d = &out_claim.point[l_lo..l_lo + l_d];
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

    let eq_sr = eq_points_ext2(r_lo, &challenges1);
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("Conv1D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y_full ----
    // Σ_{d,m} eq_D[d]·α^m·Y_full[d,m] = s_alpha_conv. The implied sum of this
    // sumcheck IS s_alpha_conv — derived from the committed Y_full instead of
    // a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_full == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1],
        s_alpha_conv,
        l_full,
        2,
        transcript,
    );
    if !ok_b {
        println!("Conv1D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YFP(r_m), where YFP(r_m) = Y_full(r_m, r_d)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, yfull_claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("Conv1D sumcheck B final eval mismatch");
        return false;
    }
    // The Y_full claim must sit exactly at (r_m, r_d).
    if yfull_claim_b.point.len() != l_full + l_d {
        println!("Conv1D sumcheck B claim point arity mismatch");
        return false;
    }
    for i in 0..l_full {
        if !crate::util::arith::ext2_field_eq(yfull_claim_b.point[i], challenges_b[i]) {
            println!("Conv1D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_d {
        if !crate::util::arith::ext2_field_eq(yfull_claim_b.point[l_full + i], r_d[i]) {
            println!("Conv1D sumcheck B claim point (r_d) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2: F×G ----
    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2],
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
    // For 0-round sumcheck (degenerate case), final_eval IS the sum.
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3],
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
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("Conv1D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (kernel_size=1: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("Conv1D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4],
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
    if expected_final_4 != sumcheck_proofs[4].final_eval {
        println!("Conv1D sumcheck 4 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // Y(r*) = Σ_x eq(r*, x)·mask(x)·Y_full[d(x), E(lo(x))], with r* the
    // y_self point. Binds Y's real region to the valid full-conv coefficients
    // and Y's padded region to zero.
    let l_out_n = l_lo + l_d;
    if y_self_claim.point.len() != l_out_n {
        println!("Conv1D sumcheck C: y_self point arity mismatch");
        return false;
    }
    let (ok_c, challenges_c) = SumcheckVerifier::verify(
        sumcheck_proofs[5],
        y_self_claim.eval,
        l_out_n,
        3,
        transcript,
    );
    if !ok_c {
        println!("Conv1D sumcheck C verification failed");
        return false;
    }

    // final_eval = eq(r*, r') · mask(r') · Y_full(σ̃(r'_lo), r'_d)
    let eq_c_final = eq_points_ext2(&y_self_claim.point, &challenges_c);
    let mask_final = conv1d_mask_mle_eval(conv, &challenges_c);
    let expected_final_c = ext2_mul(ext2_mul(eq_c_final, mask_final), yfull_claim_c.eval);
    if expected_final_c != sumcheck_proofs[5].final_eval {
        println!("Conv1D sumcheck C final eval mismatch");
        return false;
    }
    // The Y_full claim must sit exactly at the bit-affine view point.
    let mut expected_point_c = conv.view_point(&challenges_c[..l_lo]);
    expected_point_c.extend_from_slice(&challenges_c[l_lo..]);
    if yfull_claim_c.point.len() != expected_point_c.len() {
        println!("Conv1D sumcheck C claim point arity mismatch");
        return false;
    }
    for i in 0..expected_point_c.len() {
        if !crate::util::arith::ext2_field_eq(yfull_claim_c.point[i], expected_point_c[i]) {
            println!("Conv1D sumcheck C claim point mismatch");
            return false;
        }
    }

    true
}

// ============================================================================
// Conv2D grand-product output binding (ZK4_CONV_GRANDPRODUCT ablation)
// ============================================================================

/// Build the three PUBLIC index vectors of the VerfCNN multiset partition
/// `{Y@E-idx} ⊎ {K@junk-idx} = {Y_full@all-idx}` for a Conv2D. Deterministic
/// from conv geometry — both `conv2d_prove` and `conv2d_verify` call it. The
/// `value=0 ⇒ idx=1` no-op fills every padded slot so `β·0+1 = 1` drops out of
/// the product. Lifted verbatim from `conv_gp_bind::build_bind_vectors`.
fn conv2d_gp_idx_vectors(
    conv: &Conv2D,
) -> (Vec<AlmostGoldilocksField>, Vec<AlmostGoldilocksField>, Vec<AlmostGoldilocksField>) {
    let c_out_pad = conv.c_out.next_power_of_two();
    let s_full = conv.s_full();
    let s_full_pad = s_full.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let w_out_pad = conv.w_out.next_power_of_two();
    let d_domain = c_out_pad * s_full_pad;

    // Y_full leg: idx = linear position (real), else 1 (no-op padding).
    let mut yfull_idx = vec![AlmostGoldilocksField(1); d_domain];
    for d in 0..conv.c_out {
        for m in 0..s_full {
            let i = d * s_full_pad + m;
            yfull_idx[i] = AlmostGoldilocksField(i as u64);
        }
    }

    // Y leg: idx = exponent position in Y_full, else 1.
    let y_len = c_out_pad * h_out_pad * w_out_pad;
    let mut y_idx = vec![AlmostGoldilocksField(1); y_len];
    for d in 0..conv.c_out {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let lin = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                let e = conv.view_exponent(ho, wo);
                y_idx[lin] = AlmostGoldilocksField((d * s_full_pad + e) as u64);
            }
        }
    }

    // K leg: the real Y_full positions no valid output lands on (d outer,
    // m inner — MUST match `Conv2D::build_k_witness`), padded with idx=1.
    let mut used = vec![false; d_domain];
    for d in 0..conv.c_out {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                used[d * s_full_pad + conv.view_exponent(ho, wo)] = true;
            }
        }
    }
    let mut k_idx = Vec::new();
    for d in 0..conv.c_out {
        for m in 0..s_full {
            let i = d * s_full_pad + m;
            if !used[i] {
                k_idx.push(AlmostGoldilocksField(i as u64));
            }
        }
    }
    k_idx.resize(conv.gp_k_len(), AlmostGoldilocksField(1));

    (y_idx, k_idx, yfull_idx)
}

/// Element-wise equality of two Ext2 points (same length + same coordinates).
fn ext2_point_eq(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(&x, &y)| ext2_field_eq(x, y))
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
    // edge_ids: [x_edge, wf_edge, y_edge] (+ yfull_edge when not junk-free)
    // witnesses: [X, W_flat, Y] (+ Y_full)
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];
    let junk_free = conv.junk_free();

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

    let s_in_pad = conv.s_in.next_power_of_two();
    let s_kernel_pad = conv.s_kernel.next_power_of_two();
    let c_in_pad = conv.c_in.next_power_of_two();
    let w_in_pad = conv.input_w.next_power_of_two();
    let h_in_pad = conv.input_h.next_power_of_two();
    let w_out_pad = conv.w_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let s_out_pad = w_out_pad * h_out_pad;

    // Parse claim point: Y shape [B, C_out, H_out, W_out]
    // little-endian: w_out bits (lowest) | h_out bits | c_out bits | b bits
    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];

    // ---- Batch variables: bound, never summed ----
    // The weights carry no batch index, so both sides of the conv identity are
    // multilinear in b and agree on the hypercube; they are therefore the same
    // polynomial, and evaluating at the verifier's r_b is sound with no extra
    // argument. Every batched witness is read as Sum_b eq_b[b]*T[b], and every
    // claim this block emits on a batched edge gets r_b appended. The weight
    // claim does not: W is shared across the batch.
    //
    // With batch == 1, l_b == 0 and eq_b == [1], so each fold degenerates to
    // from_base(T[idx]) and no point grows -- the unbatched protocol, and its
    // transcript, are unchanged.
    let l_b = conv.l_b();
    let batch = conv.batch.max(1);
    assert_eq!(
        out_claim.point.len(),
        l_spatial_out + l_d + l_b,
        "conv2d out-claim point must carry {} spatial + {} c_out + {} batch vars",
        l_spatial_out, l_d, l_b
    );
    let r_b = &out_claim.point[l_spatial_out + l_d..l_spatial_out + l_d + l_b];
    let eq_b = evaluate_lagrange_basis_ext2(r_b);
    let y_stride = conv.y_stride();
    let x_stride = conv.x_stride();
    let yfull_stride = conv.yfull_stride();

    // ---- Sample α ----
    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // Build eq_D table
    let eq_d = evaluate_lagrange_basis_ext2(r_d);

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d] for each spatial position k
    let y_data = witnesses[2]; // Y witness
    let mut yp = vec![AlmostGoldilocksExt2::zero(); s_out_pad];
    for b in 0..batch {
        for d in 0..conv.c_out {
            let wd = ext2_mul(eq_d[d], eq_b[b]);
            for ho in 0..conv.h_out {
                for wo in 0..conv.w_out {
                    let y_idx =
                        b * y_stride + wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                    let y_val = AlmostGoldilocksExt2::from_base(
                        y_data.data.as_ref().unwrap().index(y_idx));
                    let k = wo + ho * w_out_pad; // spatial index (little-endian)
                    yp[k] = ext2_add(yp[k], ext2_mul(wd, y_val));
                }
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
    let mut y_self_point = Vec::with_capacity(l_spatial_out + l_d + l_b);
    y_self_point.extend_from_slice(&r_spatial_new);
    y_self_point.extend_from_slice(r_d);
    y_self_point.extend_from_slice(r_b);

    let y_self_claim = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_self_point,
        eval: yp_at_r,
    };

    // ---- Sumcheck B: bind s_alpha_conv to a committed polynomial ----
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // committed data rather than a free prover scalar. Together with
    // sumchecks 2/3/4 (which bind the same sum to X, W via F·G),
    // Schwartz-Zippel over α gives FullConv = conv(X, W) coefficient-wise.
    let (proof_b, claim_b) = if junk_free {
        // Junk-free 1×1 stride-1 fast path: s_full = s_in and
        // FullConv[d, m] = Y[d, s_in−1−m] over the WHOLE padded box
        // (w_out_pad = w_pad, h_out_pad = h_pad), so the α-sum binds the
        // committed Y itself: Σ_{d,m} eq_D[d]·α^m·Y[d, s_in−1−m] = s_alpha_conv.
        // Y's padded region is bound to X's padded region · W (zero for an
        // honest X). The bit-complement is a pure point transform:
        // YFP(r_m) = Y(1−r_m, r_d).
        let y_slice = y_data.data.as_ref().unwrap().evaluations_ref();
        let s_box = w_out_pad * h_out_pad;
        debug_assert_eq!(s_box, conv.s_in);
        let mut yfp = vec![AlmostGoldilocksExt2::zero(); s_box];
        for b in 0..batch {
            for d in 0..conv.c_out {
                let wd = ext2_mul(eq_d[d], eq_b[b]);
                for m in 0..s_box {
                    let v = AlmostGoldilocksExt2::from_base(
                        y_slice[b * y_stride + d * s_box + (s_box - 1 - m)]);
                    yfp[m] = ext2_add(yfp[m], ext2_mul(wd, v));
                }
            }
        }
        let alpha_full = alpha_power_table(alpha, s_box);

        let mut prover_b = CpuLinearSumcheckProverExt2::new(l_full, 2, transcript);
        let proof_b = prover_b.prove(&mut [alpha_full, yfp].as_mut_slice(), transcript);
        let r_m = prover_b.challenges.clone();

        // YFP(r_m) = Y(1−r_m, r_d)
        let one = AlmostGoldilocksExt2::one();
        let mut y_point_b = Vec::with_capacity(l_full + l_d + l_b);
        y_point_b.extend(r_m.iter().map(|&r| ext2_sub(one, r)));
        y_point_b.extend_from_slice(r_d);
        y_point_b.extend_from_slice(r_b);
        (proof_b, Claim {
            edge_id: y_edge,
            sparse_id: 0,
            point: y_point_b,
            eval: prover_b.final_eval(1),
        })
    } else {
        // General path: Σ_{d,m} eq_D[d]·α^m·Y_full[d,m] = s_alpha_conv over
        // the committed Y_full aux edge.
        let yfull_edge = edge_ids[3];
        let yfull_data = witnesses[3];
        let s_full = conv.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let yfull_slice = yfull_data.data.as_ref().unwrap().evaluations_ref();
        let mut yfp = vec![AlmostGoldilocksExt2::zero(); s_full_pad];
        for b in 0..batch {
            for d in 0..conv.c_out {
                let wd = ext2_mul(eq_d[d], eq_b[b]);
                for m in 0..s_full {
                    let v = AlmostGoldilocksExt2::from_base(
                        yfull_slice[b * yfull_stride + d * s_full_pad + m]);
                    yfp[m] = ext2_add(yfp[m], ext2_mul(wd, v));
                }
            }
        }
        let alpha_full = alpha_power_table(alpha, s_full_pad);

        let mut prover_b = CpuLinearSumcheckProverExt2::new(l_full, 2, transcript);
        let proof_b = prover_b.prove(&mut [alpha_full, yfp].as_mut_slice(), transcript);
        let r_m = prover_b.challenges.clone();

        // YFP(r_m) = Y_full(r_m, r_d)
        let mut yfull_point_b = Vec::with_capacity(l_full + l_d + l_b);
        yfull_point_b.extend_from_slice(&r_m);
        yfull_point_b.extend_from_slice(r_d);
        yfull_point_b.extend_from_slice(r_b);
        (proof_b, Claim {
            edge_id: yfull_edge,
            sparse_id: 0,
            point: yfull_point_b,
            eval: prover_b.final_eval(1),
        })
    };

    // ---- Sumcheck 2: Channel F×G ----
    // F[c] = Σ_i X_rev[c,i]·α^i,  G[c] = Σ_j WP[c,j]·α^j
    // Σ_c F[c]·G[c] = s_alpha_conv (bound to Y_full by sumcheck B)

    // Build WP[c, j] = Σ_d W_flat[d, c, j] · eq_D[d]
    let wf_data = witnesses[1]; // W_flat witness
    let mut wp = vec![AlmostGoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for d in 0..conv.c_out {
        for c in 0..conv.c_in {
            for j in 0..conv.s_kernel {
                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                let wf_val = AlmostGoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    // Build G[c] = Σ_j WP[c, j] · α^j
    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    // Build F[c] = Σ_i X_rev[c, i] · α^i
    // X_rev[c, i] = X[c, S_in - 1 - i]
    let x_data = witnesses[0]; // X witness
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for b in 0..batch {
        for c in 0..conv.c_in {
            for ih in 0..conv.input_h {
                for iw in 0..conv.input_w {
                    let x_idx =
                        b * x_stride + iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                    let x_val = AlmostGoldilocksExt2::from_base(
                        x_data.data.as_ref().unwrap().index(x_idx));
                    let i_flat = ih * conv.stride_w + iw;
                    let rev_i = conv.s_in - 1 - i_flat;
                    f_poly[c] = ext2_add(
                        f_poly[c],
                        ext2_mul(ext2_mul(x_val, eq_b[b]), alpha_in[rev_i]));
                }
            }
        }
    }

    // Σ_c F[c]·G[c] = Σ_m FullConv[m]·α^m — the same sum sumcheck B bound to
    // the committed Y_full. The verifier checks the two sumchecks' implied
    // sums against each other, so no scalar travels outside the transcript.

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
    let mut xp = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
    for b in 0..batch {
        for c in 0..conv.c_in {
            let wc = ext2_mul(eq_c[c], eq_b[b]);
            for ih in 0..conv.input_h {
                for iw in 0..conv.input_w {
                    let x_idx =
                        b * x_stride + iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                    let x_val = AlmostGoldilocksExt2::from_base(
                        x_data.data.as_ref().unwrap().index(x_idx));
                    let i_flat = ih * conv.stride_w + iw;
                    let rev_i = conv.s_in - 1 - i_flat;
                    xp[rev_i] = ext2_add(xp[rev_i], ext2_mul(wc, x_val));
                }
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
    let one = AlmostGoldilocksExt2::one();
    let r_spatial_x: Vec<AlmostGoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

    let mut x_point = Vec::with_capacity(l_spatial_in + l_c + l_b);
    x_point.extend_from_slice(&r_spatial_x);
    x_point.extend_from_slice(&r_c);
    x_point.extend_from_slice(r_b);

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

    let mut wpp = vec![AlmostGoldilocksExt2::zero(); s_kernel_pad];
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

    if junk_free {
        // Junk-free fast path: sumcheck B already bound s_alpha_conv to the
        // committed Y itself — no Y_full aux and no masked-view sumcheck.
        return (
            vec![proof1, proof_b, proof2, proof3, proof4],
            vec![y_self_claim, x_claim, wf_claim, claim_b],
        );
    }

    assert!(
        !(conv.grand_product_mode() && batch > 1),
        "ZK4_CONV_GRANDPRODUCT is a single-image ablation path with no \
         batch-folded form; run it with batch = 1"
    );
    if conv.grand_product_mode() {
        // ---- Grand-product output binding (VerfCNN multiset partition) ----
        // Replaces sumcheck C. Sumchecks 1, B, 2, 3, 4 above are UNCHANGED.
        // Prove {Y@E-idx} ⊎ {K@junk-idx} = {Y_full@all-idx} at a shared β, then
        // (in verify) check P_Y·P_K == P_Yfull. Each grand product exits with a
        // bottom claim on its committed value vector, which the fold opens.
        let yfull_edge = edge_ids[3];
        let k_edge = edge_ids[4];
        let yfull_data = witnesses[3];
        let k_data = witnesses[4];

        let (y_idx, k_idx, yfull_idx) = conv2d_gp_idx_vectors(conv);
        let y_val = witnesses[2].data.as_ref().unwrap().evaluations();
        let yfull_val = yfull_data.data.as_ref().unwrap().evaluations();
        let k_val = k_data.data.as_ref().unwrap().evaluations();
        debug_assert_eq!(
            k_val.len(), k_idx.len(),
            "Conv2D GP: K witness length must match idxK length"
        );

        let beta = transcript.challenge_ext2(b"conv_gp_beta");
        let (gp_y, cy) = prove_grand_product(&y_val, &y_idx, beta, transcript);
        let (gp_k, ck) = prove_grand_product(&k_val, &k_idx, beta, transcript);
        let (gp_yfull, cyf) = prove_grand_product(&yfull_val, &yfull_idx, beta, transcript);
        debug_assert!(
            ext2_field_eq(ext2_mul(gp_y.product, gp_k.product), gp_yfull.product),
            "Conv2D GP: P_Y·P_K must equal P_Yfull on honest data"
        );

        // Produced claims carry the TRUE opened MLE value at each bottom point
        // (what the fold opens against the commitment), NOT the bottom c.
        let y_gp = Claim {
            edge_id: y_edge,
            sparse_id: 0,
            eval: witnesses[2].data.as_ref().unwrap().evaluate_at_point_ext2(&cy.point),
            point: cy.point,
        };
        let k_gp = Claim {
            edge_id: k_edge,
            sparse_id: 0,
            eval: k_data.data.as_ref().unwrap().evaluate_at_point_ext2(&ck.point),
            point: ck.point,
        };
        let yfull_gp = Claim {
            edge_id: yfull_edge,
            sparse_id: 0,
            eval: yfull_data.data.as_ref().unwrap().evaluate_at_point_ext2(&cyf.point),
            point: cyf.point,
        };

        // The products P_Y/P_K/P_Yfull ride along as the first (product-carrier)
        // slot of each flattened grand-product proof.
        let mut proofs = vec![proof1, proof_b, proof2, proof3, proof4];
        proofs.extend(gp_y.flatten());
        proofs.extend(gp_k.flatten());
        proofs.extend(gp_yfull.flatten());
        return (
            proofs,
            vec![y_self_claim, x_claim, wf_claim, claim_b, y_gp, k_gp, yfull_gp],
        );
    }

    // ---- Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // At the (already random) y_self point r*:
    //   Y(r*) = Σ_x eq(r*, x) · mask(x) · Y_full[d(x), E(spatial(x))]
    // mask = [wo<w_out]·[ho<h_out]·[d<c_out] pins Y's real region to the valid
    // full-conv coefficients AND its padded region to zero. Degree-3 sumcheck;
    // the view factor exits as a Y_full claim at the bit-affine point σ̃(r').
    let yfull_edge = edge_ids[3];
    let yfull_data = witnesses[3];
    let s_full_pad = conv.s_full().next_power_of_two();
    let yfull_slice = yfull_data.data.as_ref().unwrap().evaluations_ref();
    // The batch variables are already bound into view_tab below, so sumcheck C
    // runs over the output variables only; r_star drops the r_b suffix that
    // y_self_claim carries.
    let l_out_n = l_spatial_out + l_d;
    let r_star = &y_self_claim.point[..l_out_n];
    let eq_star = evaluate_lagrange_basis_ext2(r_star);
    debug_assert_eq!(eq_star.len(), 1 << l_out_n);

    let c_out_pad = conv.c_out.next_power_of_two();
    let mut mask_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    let mut view_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    for d in 0..c_out_pad {
        for ho in 0..h_out_pad {
            for wo in 0..w_out_pad {
                let x_lin = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                let e = conv.view_exponent(ho, wo);
                let mut acc = AlmostGoldilocksExt2::zero();
                for b in 0..batch {
                    acc = ext2_add(acc, ext2_mul(eq_b[b],
                        AlmostGoldilocksExt2::from_base(
                            yfull_slice[b * yfull_stride + d * s_full_pad + e])));
                }
                view_tab[x_lin] = acc;
                if d < conv.c_out && ho < conv.h_out && wo < conv.w_out {
                    mask_tab[x_lin] = AlmostGoldilocksExt2::one();
                }
            }
        }
    }

    let mut prover_c = CpuLinearSumcheckProverExt2::new(l_out_n, 3, transcript);
    let proof_c = prover_c.prove(&mut [eq_star, mask_tab, view_tab].as_mut_slice(), transcript);
    let r_prime = prover_c.challenges.clone();

    // view MLE at r' = Y_full(σ̃(r'_spatial), r'_d)
    let mut yfull_point_c = conv.view_point(&r_prime[..l_spatial_out]);
    yfull_point_c.extend_from_slice(&r_prime[l_spatial_out..]);
    yfull_point_c.extend_from_slice(r_b);
    let yfull_claim_c = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_c,
        eval: prover_c.final_eval(2),
    };

    // Return: 6 proofs (transcript order), claims =
    // [y_self_claim, x_claim, wf_claim, claim_b, yfull_claim_c]
    (
        vec![proof1, proof_b, proof2, proof3, proof4, proof_c],
        vec![y_self_claim, x_claim, wf_claim, claim_b, yfull_claim_c],
    )
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
    // claims layout:
    //   general:   [y_self, x, wf, yfull_b, yfull_c, out_claim]
    //   junk-free: [y_self, x, wf, y_b, out_claim]   (1×1 stride-1 fast path)
    // proofs (transcript order): [p1, pB, p2, p3, p4] (+ pC when general)
    let junk_free = conv.junk_free();
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];
    let claim_b = claims[3];

    let l_spatial_out = conv.l_spatial_out();
    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

    let l_b = conv.l_b();
    if out_claim.point.len() != l_spatial_out + l_d + l_b {
        println!("Conv2D out-claim point arity mismatch (batch vars)");
        return false;
    }
    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];
    // Batch variables were bound, not summed. Every claim this block emits on
    // a batched edge must carry exactly this r_b, or the prover could answer a
    // different image than the one the verifier asked about.
    let r_b = &out_claim.point[l_spatial_out + l_d..l_spatial_out + l_d + l_b];
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
    let eq_sr = eq_points_ext2(r_spatial, &challenges1);
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("Conv2D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y_full ----
    // Σ_{d,m} eq_D[d]·α^m·Y_full[d,m] = s_alpha_conv. The implied sum of this
    // sumcheck IS s_alpha_conv — derived from the committed Y_full instead of
    // a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_full == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1],
        s_alpha_conv,
        l_full,
        2,
        transcript,
    );
    if !ok_b {
        println!("Conv2D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YFP(r_m), where YFP(r_m) = Y_full(r_m, r_d)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("Conv2D sumcheck B final eval mismatch");
        return false;
    }
    // The claim must sit exactly at (r_m, r_d): on Y_full for the general
    // path, or on Y itself at the bit-COMPLEMENT point (1−r_m, r_d) for the
    // junk-free fast path (FullConv[d,m] = Y[d, s_in−1−m]).
    if claim_b.point.len() != l_full + l_d + l_b {
        println!("Conv2D sumcheck B claim point arity mismatch");
        return false;
    }
    let one = AlmostGoldilocksExt2::one();
    for i in 0..l_full {
        let expected = if junk_free {
            ext2_sub(one, challenges_b[i])
        } else {
            challenges_b[i]
        };
        if !crate::util::arith::ext2_field_eq(claim_b.point[i], expected) {
            println!("Conv2D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_d {
        if !crate::util::arith::ext2_field_eq(claim_b.point[l_full + i], r_d[i]) {
            println!("Conv2D sumcheck B claim point (r_d) mismatch");
            return false;
        }
    }
    for i in 0..l_b {
        if !crate::util::arith::ext2_field_eq(claim_b.point[l_full + l_d + i], r_b[i]) {
            println!("Conv2D sumcheck B claim point (r_b) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2: F×G ----
    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2],
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
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3],
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
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("Conv2D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // Σ_j α^j · WPP[j] = claim_g
    // For 0-round sumcheck (1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    // Cross-check: claim_f * claim_g = sumcheck 2's final eval
    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("Conv2D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4],
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
    if expected_final_4 != sumcheck_proofs[4].final_eval {
        println!("Conv2D sumcheck 4 final eval mismatch");
        return false;
    }

    // ---- Grand-product output binding verify (ZK4_CONV_GRANDPRODUCT) ----
    if !junk_free && conv.grand_product_mode() {
        // claims: [y_self, x, wf, yfull_b, y_gp, k_gp, yfull_gp, out_claim].
        if claims.len() < 8 {
            println!("Conv2D GP: claim arity mismatch");
            return false;
        }
        let y_gp = claims[4];
        let k_gp = claims[5];
        let yfull_gp = claims[6];

        // β mirrored: drawn immediately after sumcheck 4 (same as prover).
        let beta = transcript.challenge_ext2(b"conv_gp_beta");

        let (y_idx, k_idx, yfull_idx) = conv2d_gp_idx_vectors(conv);
        let n_y = l_spatial_out + l_d;
        let n_k = log2_ceil(conv.gp_k_len());
        let n_yfull = l_full + l_d;

        // The flattened grand-product proofs follow the 5 base sumcheck proofs.
        let flat: Vec<SumcheckProof> =
            sumcheck_proofs[5..].iter().map(|p| (*p).clone()).collect();
        let (gp_y, used_y) = GrandProductProof::unflatten(&flat, 0, n_y);
        let (gp_k, used_k) = GrandProductProof::unflatten(&flat, used_y, n_k);
        let (gp_yfull, _) = GrandProductProof::unflatten(&flat, used_y + used_k, n_yfull);

        // Verify the three grand products in prover order (Y, K, Y_full). Each
        // returns the bottom claim (point r_0, eval c). We then check c equals
        // β·v(r_0)+idx(r_0) where v(r_0) is the produced-claim eval the fold
        // opens, and that the produced claim sits at r_0.
        let cy = match verify_grand_product(&gp_y, &y_idx, beta, gp_y.product, transcript) {
            Some(c) => c,
            None => { println!("Conv2D GP: Y grand-product verify failed"); return false; }
        };
        if !ext2_point_eq(&y_gp.point, &cy.point)
            || !ext2_field_eq(cy.eval, beta_linear_leaf_eval(beta, y_gp.eval, &y_idx, &cy.point))
        {
            println!("Conv2D GP: Y bottom β·v+idx check failed");
            return false;
        }

        let ck = match verify_grand_product(&gp_k, &k_idx, beta, gp_k.product, transcript) {
            Some(c) => c,
            None => { println!("Conv2D GP: K grand-product verify failed"); return false; }
        };
        if !ext2_point_eq(&k_gp.point, &ck.point)
            || !ext2_field_eq(ck.eval, beta_linear_leaf_eval(beta, k_gp.eval, &k_idx, &ck.point))
        {
            println!("Conv2D GP: K bottom β·v+idx check failed");
            return false;
        }

        let cyf = match verify_grand_product(&gp_yfull, &yfull_idx, beta, gp_yfull.product, transcript) {
            Some(c) => c,
            None => { println!("Conv2D GP: Y_full grand-product verify failed"); return false; }
        };
        if !ext2_point_eq(&yfull_gp.point, &cyf.point)
            || !ext2_field_eq(cyf.eval, beta_linear_leaf_eval(beta, yfull_gp.eval, &yfull_idx, &cyf.point))
        {
            println!("Conv2D GP: Y_full bottom β·v+idx check failed");
            return false;
        }

        // Multiset partition: P_Y · P_K == P_Yfull (Schwartz–Zippel over β).
        if !ext2_field_eq(ext2_mul(gp_y.product, gp_k.product), gp_yfull.product) {
            println!("Conv2D GP: product partition P_Y·P_K == P_Yfull failed");
            return false;
        }
        return true;
    }

    // ---- Verify Sumcheck C (general path only): masked-view consistency ----
    if !junk_free {
        let yfull_claim_c = claims[4];
        // Y(r*) = Σ_x eq(r*, x)·mask(x)·Y_full[d(x), E(spatial(x))], with r* the
        // y_self point. Binds Y's real region to the valid full-conv coefficients
        // and Y's padded region to zero.
        let l_out_n = l_spatial_out + l_d;
        if y_self_claim.point.len() != l_out_n + l_b {
            println!("Conv2D sumcheck C: y_self point arity mismatch");
            return false;
        }
        let (ok_c, challenges_c) = SumcheckVerifier::verify(
            sumcheck_proofs[5],
            y_self_claim.eval,
            l_out_n,
            3,
            transcript,
        );
        if !ok_c {
            println!("Conv2D sumcheck C verification failed");
            return false;
        }

        // final_eval = eq(r*, r') · mask(r') · Y_full(σ̃(r'_spatial), r'_d)
        // r_b is bound inside the view factor, so eq is taken over the output
        // variables only -- the same prefix the prover used as r_star.
        let eq_c_final = eq_points_ext2(&y_self_claim.point[..l_out_n], &challenges_c);
        let mask_final = conv2d_mask_mle_eval(conv, &challenges_c);
        let expected_final_c = ext2_mul(ext2_mul(eq_c_final, mask_final), yfull_claim_c.eval);
        if expected_final_c != sumcheck_proofs[5].final_eval {
            println!("Conv2D sumcheck C final eval mismatch");
            return false;
        }
        // The Y_full claim must sit exactly at the bit-affine view point.
        let mut expected_point_c = conv.view_point(&challenges_c[..l_spatial_out]);
        expected_point_c.extend_from_slice(&challenges_c[l_spatial_out..]);
        expected_point_c.extend_from_slice(r_b);
        if yfull_claim_c.point.len() != expected_point_c.len() {
            println!("Conv2D sumcheck C claim point arity mismatch");
            return false;
        }
        for i in 0..expected_point_c.len() {
            if !crate::util::arith::ext2_field_eq(yfull_claim_c.point[i], expected_point_c[i]) {
                println!("Conv2D sumcheck C claim point mismatch");
                return false;
            }
        }

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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        let w_slice = w.data.as_ref().unwrap().evaluations_ref();
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
                            out_data[wf_idx] = w_slice[w_idx];
                        }
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output)]
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
        zero_buffer(&mut d_wf, out_size).expect("FlattenKernel3D: memset");

        gpu_flatten_kernel3d(
            &d_w, &mut d_wf,
            self.c_out, self.c_in, self.kd, self.kh, self.kw,
            kw_pad, kh_pad, kd_pad,
            c_in_pad, s_kernel_pad,
            self.stride_h, self.stride_w,
        ).expect("FlattenKernel3D: gpu kernel failed");

        let out_shape = vec![self.c_out, self.c_in, s_kernel];
        vec![Witness::new_device(out_shape, Arc::new(d_wf), DataType::Uint, inputs[0].sf, Role::Output)]
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
    let mut w_partial = vec![AlmostGoldilocksExt2::zero(); sumcheck_size];
    for dd in 0..fk.c_out {
        for c in 0..fk.c_in {
            let dc_weight = ext2_mul(eq_d[dd], eq_c[c]);
            for kd in 0..fk.kd {
                for kh in 0..fk.kh {
                    for kw in 0..fk.kw {
                        let w_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad
                            + c * kw_pad * kh_pad * kd_pad
                            + dd * kw_pad * kh_pad * kd_pad * c_in_pad;
                        let w_val = AlmostGoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                        let sc_idx = kw + kh * kw_pad + kd * kw_pad * kh_pad;
                        w_partial[sc_idx] = ext2_add(w_partial[sc_idx], ext2_mul(dc_weight, w_val));
                    }
                }
            }
        }
    }

    // H[kd, kh, kw] = eq(r_j, kd*stride_h + kh*stride_w + kw)
    let mut h_poly = vec![AlmostGoldilocksExt2::zero(); sumcheck_size];
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

    let mut h_eval = AlmostGoldilocksExt2::zero();
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
    /// Length of the full 1D flat convolution (all α-exponents, incl. junk).
    fn s_full(&self) -> usize { self.s_in + self.s_kernel - 1 }
    /// Number of variables for the full-conv exponent dimension.
    fn l_full(&self) -> usize { log2_ceil(self.s_full().max(1)) }

    /// Full-conv exponent of output position (do, ho, wo):
    ///   E(do, ho, wo) = (s_in − 1) − (cs_d·stride_h·do + cs_h·stride_w·ho + cs_w·wo)
    /// (the input is reversed in F, so output at flat input offset t sits at
    /// exponent s_in−1−t of the polynomial product).
    fn view_exponent(&self, do_: usize, ho: usize, wo: usize) -> usize {
        (self.s_in - 1)
            - (do_ * self.conv_stride_d * self.stride_h
                + ho * self.conv_stride_h * self.stride_w
                + wo * self.conv_stride_w)
    }

    /// Bit layout of the crop map E for the bit-affine view: returns
    /// (wo_shift, ho_shift, do_shift, l_si). Valid only when all conv strides
    /// are powers of two, so E's subtrahend has carry-free disjoint bit
    /// fields: wo bits occupy [wo_shift, wo_shift+l_wo), ho bits occupy
    /// [ho_shift, ho_shift+l_ho), do bits occupy [do_shift, do_shift+l_do),
    /// and E = bitwise complement over the low l_si bits (s_in−1 is
    /// all-ones). Asserts the disjointness invariants the soundness of the
    /// view depends on.
    fn view_bit_layout(&self) -> (usize, usize, usize, usize) {
        assert!(
            self.conv_stride_d.is_power_of_two()
                && self.conv_stride_h.is_power_of_two()
                && self.conv_stride_w.is_power_of_two(),
            "Conv3D bit-affine view requires power-of-two conv strides, got ({}, {}, {})",
            self.conv_stride_d, self.conv_stride_h, self.conv_stride_w
        );
        let lw = log2_ceil(self.input_w.max(1)); // log2(w_pad)
        let lh = log2_ceil(self.input_h.max(1)); // log2(h_pad)
        let ld = log2_ceil(self.input_d.max(1)); // log2(d_pad)
        let l_si = lw + lh + ld; // log2(s_in)
        let wo_shift = self.conv_stride_w.trailing_zeros() as usize;
        let ho_shift = lw + self.conv_stride_h.trailing_zeros() as usize;
        let do_shift = lw + lh + self.conv_stride_d.trailing_zeros() as usize;
        // Bit fields must be disjoint and within the s_in range for E to be
        // bit-affine over the whole padded output box.
        assert!(wo_shift + self.l_wo() <= ho_shift,
            "Conv3D view: wo bit-field [{}+{}) overlaps ho bit-field at {}",
            wo_shift, self.l_wo(), ho_shift);
        assert!(ho_shift + self.l_ho() <= do_shift,
            "Conv3D view: ho bit-field [{}+{}) overlaps do bit-field at {}",
            ho_shift, self.l_ho(), do_shift);
        assert!(do_shift + self.l_do() <= l_si,
            "Conv3D view: do bit-field [{}+{}) exceeds s_in bits {}",
            do_shift, self.l_do(), l_si);
        (wo_shift, ho_shift, do_shift, l_si)
    }

    /// σ̃: map a point over Y's output-spatial vars to the Y_full exponent
    /// point such that view(Y)(r) = Y_full(σ̃(r), r_d). Each exponent bit is
    /// either a complemented spatial coordinate or a public constant.
    fn view_point(&self, r_spatial: &[AlmostGoldilocksExt2]) -> Vec<AlmostGoldilocksExt2> {
        let (wo_shift, ho_shift, do_shift, l_si) = self.view_bit_layout();
        let l_wo = self.l_wo();
        let l_ho = self.l_ho();
        let l_do = self.l_do();
        assert_eq!(r_spatial.len(), l_wo + l_ho + l_do);
        let one = AlmostGoldilocksExt2::one();
        let mut point = vec![one; self.l_full()];
        // Bits ≥ l_si of E are zero (E < s_in).
        for coord in point.iter_mut().skip(l_si) {
            *coord = AlmostGoldilocksExt2::zero();
        }
        for i in 0..l_wo {
            point[wo_shift + i] = ext2_sub(one, r_spatial[i]);
        }
        for i in 0..l_ho {
            point[ho_shift + i] = ext2_sub(one, r_spatial[l_wo + i]);
        }
        for i in 0..l_do {
            point[do_shift + i] = ext2_sub(one, r_spatial[l_wo + l_ho + i]);
        }
        point
    }

    /// Compute the aux Y_full witness: the full 1D flat convolution
    ///   Y_full[d, m] = Σ_c Σ_{i+j=m} X_rev[c,i]·W_flat[d,c,j]
    /// scattered as m = (s_in−1−p) + j over real input positions p and real
    /// kernel taps j. Challenge-independent, committed as advice
    /// (Role::Auxiliary) and bound to s_alpha_conv by sumcheck B.
    fn compute_y_full(
        &self,
        x_slice: &[AlmostGoldilocksField],
        w_slice: &[AlmostGoldilocksField],
        out_sf: usize,
    ) -> Witness {
        let c_in_pad = self.c_in.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let d_in_pad = self.input_d.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let c_out_pad = self.c_out.next_power_of_two();

        let mut yfull = vec![AlmostGoldilocksField(0); c_out_pad * s_full_pad];
        let rows: Vec<(usize, Vec<AlmostGoldilocksField>)> = (0..self.c_out)
            .into_par_iter()
            .map(|d| {
                let mut row = vec![AlmostGoldilocksField(0); s_full_pad];
                for c in 0..self.c_in {
                    for kd in 0..self.kernel_d {
                        for kh in 0..self.kernel_h {
                            for kw in 0..self.kernel_w {
                                let j = kd * self.stride_h + kh * self.stride_w + kw;
                                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                                let w_val = w_slice[wf_idx];
                                for id in 0..self.input_d {
                                    for ih in 0..self.input_h {
                                        for iw in 0..self.input_w {
                                            let p = id * self.stride_h + ih * self.stride_w + iw;
                                            let x_idx = iw + ih * w_in_pad + id * w_in_pad * h_in_pad
                                                + c * w_in_pad * h_in_pad * d_in_pad;
                                            let m = (self.s_in - 1 - p) + j;
                                            row[m] = agl_add(row[m], agl_mul(x_slice[x_idx], w_val));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                (d, row)
            })
            .collect();
        for (d, row) in rows {
            yfull[d * s_full_pad..d * s_full_pad + s_full_pad].copy_from_slice(&row);
        }
        Witness::new(
            vec![self.c_out, s_full],
            yfull,
            DataType::Uint,
            out_sf,
            Role::Auxiliary,
        )
    }
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

        // Hoist host slices once (vs per-element trait `.index()` in the
        // c_in·kd·kh·kw inner loop).
        let x_slice = x_data.evaluations_ref();
        let w_slice = w_data.evaluations_ref();
        let total_outputs = c_out * d_out * h_out * w_out;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let results: Vec<(usize, AlmostGoldilocksField)> = (0..total_outputs)
            .into_par_iter()
            .map(|flat_idx| {
                let wo = flat_idx % w_out;
                let ho = (flat_idx / w_out) % h_out;
                let do_ = (flat_idx / (w_out * h_out)) % d_out;
                let d = flat_idx / (w_out * h_out * d_out);
                let mut acc = AlmostGoldilocksField(0);
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
                                let x_val = x_slice[x_idx];
                                let w_val = w_slice[wf_idx];
                                acc = agl_add(acc, agl_mul(x_val, w_val));
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
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);

        // Aux output: Y_full, the full 1D flat convolution. Sumcheck B binds
        // it to s_alpha_conv (and hence to X, W); the masked-view sumcheck C
        // binds Y to its valid coefficients.
        let y_full = self.compute_y_full(x_slice, w_slice, out_sf);
        #[cfg(debug_assertions)]
        {
            // Valid coefficients of Y_full must reproduce Y exactly.
            let s_full_pad = self.s_full().next_power_of_two();
            let yf = y_full.data.as_ref().unwrap().evaluations_ref();
            for d in 0..self.c_out {
                for do_ in 0..self.d_out {
                    for ho in 0..self.h_out {
                        for wo in 0..self.w_out {
                            let out_idx = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad
                                + d * w_out_pad * h_out_pad * d_out_pad;
                            let e = self.view_exponent(do_, ho, wo);
                            debug_assert_eq!(
                                out_data[out_idx], yf[d * s_full_pad + e],
                                "Conv3D: Y_full valid coefficient mismatch at (d={d}, do={do_}, ho={ho}, wo={wo})"
                            );
                        }
                    }
                }
            }
        }

        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output), y_full]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Bisect hatch; see ConvTranspose3D::run_gpu.
        if std::env::var("ZK4_CONV3D_CPU").is_ok() {
            return self.run(inputs);
        }
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
        zero_buffer(&mut d_y, out_size).expect("Conv3D: memset zero failed");

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
        let out_sf = inputs[0].sf + inputs[1].sf;

        // Aux Y_full gathered on-device: one thread per (d, m) exponent, tap
        // index j = kd·stride_h + kh·stride_w + kw (see `agl_conv_full_kernel`).
        // Matches `compute_y_full` element-wise.
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let mut d_yf = DeviceBuffer::<u64>::new(c_out_pad * s_full_pad).expect("Conv3D: alloc Y_full");
        zero_buffer(&mut d_yf, c_out_pad * s_full_pad).expect("Conv3D: memset Y_full zero failed");
        gpu_conv_full(
            &d_x, &d_w, &mut d_yf,
            self.c_out, self.c_in,
            self.kernel_d, self.kernel_h, self.kernel_w,
            self.stride_h, self.stride_w, 1,
            self.s_in, s_full, s_full_pad,
            c_in_pad, s_kernel_pad,
            false,
            // Conv3D is not batched yet: batch = 1, so the strides are unused.
            1, 0, 0,
        ).expect("Conv3D: gpu Y_full kernel failed");

        vec![
            Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output),
            Witness::new_device(vec![self.c_out, s_full], Arc::new(d_yf), DataType::Uint, out_sf, Role::Auxiliary),
        ]
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
    // edge_ids: [x_edge, wf_edge, y_edge, yfull_edge]
    // witnesses: [X, W_flat, Y, Y_full]
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];
    let yfull_edge = edge_ids[3];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

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
    let mut yp = vec![AlmostGoldilocksExt2::zero(); s_out_pad];
    for d in 0..conv.c_out {
        for do_ in 0..conv.d_out {
            for ho in 0..conv.h_out {
                for wo in 0..conv.w_out {
                    let y_idx = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad
                        + d * w_out_pad * h_out_pad * d_out_pad;
                    let y_val = AlmostGoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
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

    // ---- Sumcheck B: bind s_alpha_conv to committed Y_full ----
    // Σ_{d,m} eq_D[d]·α^m·Y_full[d,m] = s_alpha_conv.
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // the committed full conv rather than a free prover scalar. Together with
    // sumchecks 2/3/4 (which bind the same sum to X, W via F·G),
    // Schwartz-Zippel over α gives Y_full = FullConv(X, W) coefficient-wise.
    let yfull_data = witnesses[3];
    let s_full = conv.s_full();
    let s_full_pad = s_full.next_power_of_two();
    let yfull_slice = yfull_data.data.as_ref().unwrap().evaluations_ref();
    let mut yfp = vec![AlmostGoldilocksExt2::zero(); s_full_pad];
    for d in 0..conv.c_out {
        for m in 0..s_full {
            let v = AlmostGoldilocksExt2::from_base(yfull_slice[d * s_full_pad + m]);
            yfp[m] = ext2_add(yfp[m], ext2_mul(eq_d[d], v));
        }
    }
    let alpha_full = alpha_power_table(alpha, s_full_pad);

    let mut prover_b = CpuLinearSumcheckProverExt2::new(l_full, 2, transcript);
    let proof_b = prover_b.prove(&mut [alpha_full, yfp].as_mut_slice(), transcript);
    let r_m = prover_b.challenges.clone();

    // YFP(r_m) = Y_full(r_m, r_d)
    let mut yfull_point_b = Vec::with_capacity(l_full + l_d);
    yfull_point_b.extend_from_slice(&r_m);
    yfull_point_b.extend_from_slice(r_d);
    let yfull_claim_b = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_b,
        eval: prover_b.final_eval(1),
    };

    // ---- Sumcheck 2: Channel F×G ----
    // Σ_c F[c]·G[c] = s_alpha_conv (bound to Y_full by sumcheck B)
    let wf_data = witnesses[1];
    let mut wp = vec![AlmostGoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for d in 0..conv.c_out {
        for c in 0..conv.c_in {
            for j in 0..conv.s_kernel {
                let wf_idx = j + c * s_kernel_pad + d * s_kernel_pad * c_in_pad;
                let wf_val = AlmostGoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    let x_data = witnesses[0];
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for id in 0..conv.input_d {
            for ih in 0..conv.input_h {
                for iw in 0..conv.input_w {
                    let x_idx = iw + ih * w_in_pad + id * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let i_flat = id * conv.stride_h + ih * conv.stride_w + iw;
                    let rev_i = conv.s_in - 1 - i_flat;
                    f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
                }
            }
        }
    }

    // Σ_c F[c]·G[c] = Σ_m FullConv[m]·α^m — the same sum sumcheck B bound to
    // the committed Y_full. The verifier checks the two sumchecks' implied
    // sums against each other, so no scalar travels outside the transcript.

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F → X ----
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for id in 0..conv.input_d {
            for ih in 0..conv.input_h {
                for iw in 0..conv.input_w {
                    let x_idx = iw + ih * w_in_pad + id * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
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

    let one = AlmostGoldilocksExt2::one();
    let r_spatial_x: Vec<AlmostGoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

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
    let mut wpp = vec![AlmostGoldilocksExt2::zero(); s_kernel_pad];
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

    // ---- Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // At the (already random) y_self point r*:
    //   Y(r*) = Σ_x eq(r*, x) · mask(x) · Y_full[d(x), E(spatial(x))]
    // mask = [wo<w_out]·[ho<h_out]·[do<d_out]·[d<c_out] pins Y's real region
    // to the valid full-conv coefficients AND its padded region to zero.
    // Degree-3 sumcheck; the view factor exits as a Y_full claim at the
    // bit-affine point σ̃(r').
    let r_star = &y_self_claim.point;
    let l_out_n = l_spatial_out + l_d;
    let eq_star = evaluate_lagrange_basis_ext2(r_star);
    debug_assert_eq!(eq_star.len(), 1 << l_out_n);

    let c_out_pad = conv.c_out.next_power_of_two();
    let mut mask_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    let mut view_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    for d in 0..c_out_pad {
        for do_ in 0..d_out_pad {
            for ho in 0..h_out_pad {
                for wo in 0..w_out_pad {
                    let x_lin = wo + ho * w_out_pad + do_ * w_out_pad * h_out_pad
                        + d * w_out_pad * h_out_pad * d_out_pad;
                    let e = conv.view_exponent(do_, ho, wo);
                    view_tab[x_lin] =
                        AlmostGoldilocksExt2::from_base(yfull_slice[d * s_full_pad + e]);
                    if d < conv.c_out && do_ < conv.d_out && ho < conv.h_out && wo < conv.w_out {
                        mask_tab[x_lin] = AlmostGoldilocksExt2::one();
                    }
                }
            }
        }
    }

    let mut prover_c = CpuLinearSumcheckProverExt2::new(l_out_n, 3, transcript);
    let proof_c = prover_c.prove(&mut [eq_star, mask_tab, view_tab].as_mut_slice(), transcript);
    let r_prime = prover_c.challenges.clone();

    // view MLE at r' = Y_full(σ̃(r'_spatial), r'_d)
    let mut yfull_point_c = conv.view_point(&r_prime[..l_spatial_out]);
    yfull_point_c.extend_from_slice(&r_prime[l_spatial_out..]);
    let yfull_claim_c = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_c,
        eval: prover_c.final_eval(2),
    };

    // Return: 6 proofs (transcript order), claims =
    // [y_self_claim, x_claim, wf_claim, yfull_claim_b, yfull_claim_c]
    (
        vec![proof1, proof_b, proof2, proof3, proof4, proof_c],
        vec![y_self_claim, x_claim, wf_claim, yfull_claim_b, yfull_claim_c],
    )
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
    // claims layout: [y_self_claim, x_claim, wf_claim, yfull_claim_b, yfull_claim_c, out_claim]
    // proofs layout (transcript order): [p1, pB, p2, p3, p4, pC]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];
    let yfull_claim_b = claims[3];
    let yfull_claim_c = claims[4];

    let l_spatial_out = conv.l_spatial_out();
    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];
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

    let eq_sr = eq_points_ext2(r_spatial, &challenges1);
    let expected_final_1 = ext2_mul(eq_sr, y_self_claim.eval);
    if expected_final_1 != sumcheck_proofs[0].final_eval {
        println!("Conv3D sumcheck 1 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y_full ----
    // Σ_{d,m} eq_D[d]·α^m·Y_full[d,m] = s_alpha_conv. The implied sum of this
    // sumcheck IS s_alpha_conv — derived from the committed Y_full instead of
    // a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_full == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_full, 2, transcript,
    );
    if !ok_b {
        println!("Conv3D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YFP(r_m), where YFP(r_m) = Y_full(r_m, r_d)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, yfull_claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("Conv3D sumcheck B final eval mismatch");
        return false;
    }
    // The Y_full claim must sit exactly at (r_m, r_d).
    if yfull_claim_b.point.len() != l_full + l_d {
        println!("Conv3D sumcheck B claim point arity mismatch");
        return false;
    }
    for i in 0..l_full {
        if !crate::util::arith::ext2_field_eq(yfull_claim_b.point[i], challenges_b[i]) {
            println!("Conv3D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_d {
        if !crate::util::arith::ext2_field_eq(yfull_claim_b.point[l_full + i], r_d[i]) {
            println!("Conv3D sumcheck B claim point (r_d) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2 ----
    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("Conv3D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    // For 0-round sumcheck (degenerate case), final_eval IS the sum.
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_3, l_spatial_in, 2, transcript,
    );
    if !ok3 {
        println!("Conv3D sumcheck 3 verification failed");
        return false;
    }

    let alpha_mle_3 = alpha_table_mle_eval(alpha, &challenges3);
    let expected_final_3 = ext2_mul(alpha_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("Conv3D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("Conv3D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("Conv3D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[4].final_eval {
        println!("Conv3D sumcheck 4 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // Y(r*) = Σ_x eq(r*, x)·mask(x)·Y_full[d(x), E(spatial(x))], with r* the
    // y_self point. Binds Y's real region to the valid full-conv coefficients
    // and Y's padded region to zero.
    let l_out_n = l_spatial_out + l_d;
    if y_self_claim.point.len() != l_out_n {
        println!("Conv3D sumcheck C: y_self point arity mismatch");
        return false;
    }
    let (ok_c, challenges_c) = SumcheckVerifier::verify(
        sumcheck_proofs[5], y_self_claim.eval, l_out_n, 3, transcript,
    );
    if !ok_c {
        println!("Conv3D sumcheck C verification failed");
        return false;
    }

    // final_eval = eq(r*, r') · mask(r') · Y_full(σ̃(r'_spatial), r'_d)
    let eq_c_final = eq_points_ext2(&y_self_claim.point, &challenges_c);
    let mask_final = conv3d_mask_mle_eval(conv, &challenges_c);
    let expected_final_c = ext2_mul(ext2_mul(eq_c_final, mask_final), yfull_claim_c.eval);
    if expected_final_c != sumcheck_proofs[5].final_eval {
        println!("Conv3D sumcheck C final eval mismatch");
        return false;
    }
    // The Y_full claim must sit exactly at the bit-affine view point.
    let mut expected_point_c = conv.view_point(&challenges_c[..l_spatial_out]);
    expected_point_c.extend_from_slice(&challenges_c[l_spatial_out..]);
    if yfull_claim_c.point.len() != expected_point_c.len() {
        println!("Conv3D sumcheck C claim point arity mismatch");
        return false;
    }
    for i in 0..expected_point_c.len() {
        if !crate::util::arith::ext2_field_eq(yfull_claim_c.point[i], expected_point_c[i]) {
            println!("Conv3D sumcheck C claim point mismatch");
            return false;
        }
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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

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
                        out_data[out_idx] = agl_add(out_data[out_idx], agl_mul(x_val, w_val));
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.l_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output)]
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

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d] over the FULL padded box: the same
    // fold feeds sumcheck B, which must bind Y's padded region (zero in the
    // full polynomial product) too, so garbage in padding is rejected.
    let y_data = witnesses[2];
    let y_slice = y_data.data.as_ref().unwrap().evaluations_ref();
    let mut yp = vec![AlmostGoldilocksExt2::zero(); l_out_pad];
    for d in 0..c_out_pad {
        for lo in 0..l_out_pad {
            let y_val = AlmostGoldilocksExt2::from_base(y_slice[lo + d * l_out_pad]);
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

    // ---- Sumcheck B: bind s_alpha_conv DIRECTLY to committed Y ----
    // ConvTranspose1D is junk-free: the α-exponent of each X[c,j]·W[c,d,k]
    // contribution is m = j·stride + k — exactly the output position — and
    // m ≤ (input_len−1)·stride + K−1 = l_out−1 < l_out_pad, so the full 1D
    // polynomial product coincides with the committed Y over the whole padded
    // box (unreachable and padded slots are zero on both sides). The crop map
    // E is the identity, so no aux Y_full edge, no masked-view sumcheck, and
    // no power-of-two stride requirement is needed:
    //   Σ_{d,m} eq_D[d]·α^m·Y[d,m] = s_alpha_conv.
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // the committed output rather than a free prover scalar. Together with
    // sumchecks 2/3/4 (which bind the same sum to X, W via F·G),
    // Schwartz-Zippel over α gives Y = ConvTranspose(X, W) coefficient-wise.
    let alpha_full = alpha_power_table(alpha, l_out_pad);

    let mut prover_b = CpuLinearSumcheckProverExt2::new(l_lo, 2, transcript);
    let proof_b = prover_b.prove(&mut [alpha_full, yp.clone()].as_mut_slice(), transcript);
    let r_m = prover_b.challenges.clone();

    // YP(r_m) = Y(r_m, r_d)
    let mut y_point_b = Vec::with_capacity(l_lo + l_d);
    y_point_b.extend_from_slice(&r_m);
    y_point_b.extend_from_slice(r_d);
    let y_claim_b = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_point_b,
        eval: prover_b.final_eval(1),
    };

    // ---- Sumcheck 2: Channel F×G ----
    // ConvTranspose: F[c] = Σ_j X[c,j] · β^j where β = α^stride (forward, NO reversal)
    //                G[c] = Σ_k WP[c,k] · α^k
    let beta = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..conv.stride {
            b = ext2_mul(b, alpha);
        }
        b
    }; // β = α^stride

    // Build WP[c, k] = Σ_d W[c, d, k] · eq_D[d]
    // W layout: k bits (lowest) | c_out bits | c_in bits
    let w_data = witnesses[1];
    let mut wp = vec![AlmostGoldilocksExt2::zero(); c_in_pad * k_pad];
    for c in 0..conv.c_in {
        for d in 0..conv.c_out {
            for k in 0..conv.kernel_size {
                let w_idx = k + d * k_pad + c * k_pad * c_out_pad;
                let w_val = AlmostGoldilocksExt2::from_base(w_data.data.as_ref().unwrap().index(w_idx));
                wp[c * k_pad + k] = ext2_add(wp[c * k_pad + k], ext2_mul(eq_d[d], w_val));
            }
        }
    }

    // Build G[c] = Σ_k WP[c, k] · α^k
    let alpha_kernel = alpha_power_table(alpha, k_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for k in 0..conv.kernel_size {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * k_pad + k], alpha_kernel[k]));
        }
    }

    // Build F[c] = Σ_j X[c,j] · β^j (forward, strided — NO reversal)
    let x_data = witnesses[0];
    let beta_table = alpha_power_table(beta, l_in_pad);
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.input_len {
            let x_idx = j + c * l_in_pad;
            let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
            f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, beta_table[j]));
        }
    }

    // Σ_c F[c]·G[c] = Σ_m FullConv[m]·α^m — the same sum sumcheck B bound to
    // the committed Y. The verifier checks the two sumchecks' implied sums
    // against each other, so no scalar travels outside the transcript.

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F reduction to X claim ----
    // F(r_c) = Σ_j β^j · XP[j]  where XP[j] = Σ_c eq(r_c, c) · X[c, j]
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![AlmostGoldilocksExt2::zero(); l_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.input_len {
            let x_idx = j + c * l_in_pad;
            let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
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
    let mut wpp = vec![AlmostGoldilocksExt2::zero(); k_pad];
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

    // Return: 5 proofs (transcript order), claims =
    // [y_self_claim, x_claim, w_claim, y_claim_b]
    (
        vec![proof1, proof_b, proof2, proof3, proof4],
        vec![y_self_claim, x_claim, w_claim, y_claim_b],
    )
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
    // claims layout: [y_self_claim, x_claim, w_claim, y_claim_b, out_claim]
    // proofs layout (transcript order): [p1, pB, p2, p3, p4]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let w_claim = claims[2];
    let y_claim_b = claims[3];

    let l_lo = conv.l_lo();
    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();

    let r_lo = &out_claim.point[..l_lo];
    let r_d = &out_claim.point[l_lo..l_lo + l_d];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    // β = α^stride
    let beta = {
        let mut b = AlmostGoldilocksExt2::one();
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
        let one = AlmostGoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_lo {
            let a = r_lo[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2)), ext2_mul(a, b)),
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

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y ----
    // Σ_{d,m} eq_D[d]·α^m·Y[d,m] = s_alpha_conv. ConvTranspose1D is junk-free
    // (see prover): the full polynomial product IS the padded output, so the
    // α-sum binds the committed Y directly — no aux Y_full edge. The implied
    // sum of this sumcheck IS s_alpha_conv — derived from the committed Y
    // instead of a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_lo == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_lo, 2, transcript,
    );
    if !ok_b {
        println!("ConvTranspose1D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YP(r_m), where YP(r_m) = Y(r_m, r_d)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, y_claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("ConvTranspose1D sumcheck B final eval mismatch");
        return false;
    }
    // The Y claim must sit exactly at (r_m, r_d).
    if y_claim_b.point.len() != l_lo + l_d {
        println!("ConvTranspose1D sumcheck B claim point arity mismatch");
        return false;
    }
    for i in 0..l_lo {
        if !crate::util::arith::ext2_field_eq(y_claim_b.point[i], challenges_b[i]) {
            println!("ConvTranspose1D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_d {
        if !crate::util::arith::ext2_field_eq(y_claim_b.point[l_lo + i], r_d[i]) {
            println!("ConvTranspose1D sumcheck B claim point (r_d) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2: F×G ----
    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("ConvTranspose1D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_3, l_spatial_in, 2, transcript,
    );
    if !ok3 {
        println!("ConvTranspose1D sumcheck 3 verification failed");
        return false;
    }

    // Check: β-table MLE at challenges3 * x_claim.eval = final_eval
    let beta_mle_3 = alpha_table_mle_eval(beta, &challenges3);
    let expected_final_3 = ext2_mul(beta_mle_3, x_claim.eval);
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("ConvTranspose1D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("ConvTranspose1D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("ConvTranspose1D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, w_claim.eval);
    if expected_final_4 != sumcheck_proofs[4].final_eval {
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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

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
                                out_data[out_idx] = agl_add(out_data[out_idx], agl_mul(x_val, w_val));
                            }
                        }
                    }
                }
            }
        }

        let out_shape = vec![self.c_out, self.h_out, self.w_out];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output)]
    }

    /// GPU forward. PointPillars' RPN builds four of these on 432x496 BEV
    /// feature maps, and this was the last block in that model's path without
    /// a GPU path -- ScatterToBEV, PillarMaxPool and GatherFromGrid all had
    /// one. The kernel gathers rather than scatters (see the CUDA side), so
    /// results are identical without atomics.
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let x = inputs[0];
        let w_flat = inputs[1];

        let c_out_pad = self.c_out.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.w_out.next_power_of_two();
        let h_out_pad = self.h_out.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let out_size = c_out_pad * h_out_pad * w_out_pad;

        let d_x = x.as_device_buf();
        let d_w = w_flat.as_device_buf();
        let mut d_y = DeviceBuffer::<u64>::new(out_size)
            .expect("ConvTranspose2D: alloc out");
        zero_buffer(&mut d_y, out_size).expect("ConvTranspose2D: memset zero failed");

        gpu_conv_transpose2d(
            &d_x, &d_w, &mut d_y,
            self.c_out, self.h_out, self.w_out,
            self.c_in, self.kernel_h, self.kernel_w,
            self.stride_h, self.stride_w,
            self.input_h, self.input_w,
            w_in_pad, h_in_pad,
            c_out_pad, s_kernel_pad,
            w_out_pad, h_out_pad,
            self.flat_stride,
        ).expect("ConvTranspose2D: gpu kernel failed");

        let out_shape = vec![self.c_out, self.h_out, self.w_out];
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output)]
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

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d] over the FULL padded box: the same
    // fold feeds sumcheck B, which must bind Y's padded region (zero in the
    // full polynomial product) too, so garbage in padding is rejected.
    let y_data = witnesses[2];
    let y_slice = y_data.data.as_ref().unwrap().evaluations_ref();
    let mut yp = vec![AlmostGoldilocksExt2::zero(); s_out_pad];
    for d in 0..c_out_pad {
        for k in 0..s_out_pad {
            let y_val = AlmostGoldilocksExt2::from_base(y_slice[k + d * s_out_pad]);
            yp[k] = ext2_add(yp[k], ext2_mul(eq_d[d], y_val));
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

    // ---- Sumcheck B: bind s_alpha_conv DIRECTLY to committed Y ----
    // ConvTranspose2D is junk-free: the α-exponent of each contribution is
    //   m = (jh·stride_h + kh)·w_out_pad + (jw·stride_w + kw) = oh·w_out_pad + ow
    // — exactly the flat little-endian PADDED output index (carry-free since
    // ow ≤ w_out−1 < w_out_pad and oh ≤ h_out−1 < h_out_pad). The full
    // polynomial product coincides with the committed Y over the whole padded
    // box (unreachable and padded slots are zero on both sides). The crop map
    // E is the identity, so no aux Y_full edge, no masked-view sumcheck, and
    // no power-of-two stride requirement is needed:
    //   Σ_{d,m} eq_D[d]·α^m·Y[d,m] = s_alpha_conv.
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // the committed output rather than a free prover scalar.
    let alpha_full = alpha_power_table(alpha, s_out_pad);

    let mut prover_b = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof_b = prover_b.prove(&mut [alpha_full, yp.clone()].as_mut_slice(), transcript);
    let r_m = prover_b.challenges.clone();

    // YP(r_m) = Y(r_m, r_d)
    let mut y_point_b = Vec::with_capacity(l_spatial_out + l_d);
    y_point_b.extend_from_slice(&r_m);
    y_point_b.extend_from_slice(r_d);
    let y_claim_b = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_point_b,
        eval: prover_b.final_eval(1),
    };

    // ---- Sumcheck 2: Channel F×G ----
    // β_w = α^{stride_w}, β_h = α^{stride_h * flat_stride}
    let beta_w = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride) { b = ext2_mul(b, alpha); }
        b
    };

    // Build WP[c, j] = Σ_d W_flat[c, d, j] · eq_D[d]
    // W_flat layout: j bits | c_out bits | c_in bits
    let wf_data = witnesses[1];
    let mut wp = vec![AlmostGoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for c in 0..conv.c_in {
        for d in 0..conv.c_out {
            for j in 0..conv.s_kernel {
                let wf_idx = j + d * s_kernel_pad + c * s_kernel_pad * c_out_pad;
                let wf_val = AlmostGoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    // Build G[c] = Σ_j WP[c, j] · α^j
    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for j in 0..conv.s_kernel {
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wp[c * s_kernel_pad + j], alpha_kernel[j]));
        }
    }

    // Build F[c] = Σ_{jh,jw} X[c,jh,jw] · β_h^{jh} · β_w^{jw}
    let x_data = witnesses[0];
    let beta_w_table = alpha_power_table(beta_w, w_in_pad);
    let beta_h_table = alpha_power_table(beta_h, h_in_pad);
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for jh in 0..conv.input_h {
            for jw in 0..conv.input_w {
                let x_idx = jw + jh * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let power = ext2_mul(beta_h_table[jh], beta_w_table[jw]);
                f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, power));
            }
        }
    }

    // Σ_c F[c]·G[c] = Σ_m FullConv[m]·α^m — the same sum sumcheck B bound to
    // the committed Y. The verifier checks the two sumchecks' implied sums
    // against each other, so no scalar travels outside the transcript.

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F → X claim ----
    // XP[jw + jh*w_in_pad] = Σ_c eq(r_c, c) · X[c, jh, jw]
    // Power table: alpha_X[jw + jh*w_in_pad] = β_w^{jw} · β_h^{jh}
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for jh in 0..conv.input_h {
            for jw in 0..conv.input_w {
                let x_idx = jw + jh * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let flat_idx = jw + jh * w_in_pad;
                xp[flat_idx] = ext2_add(xp[flat_idx], ext2_mul(eq_c[c], x_val));
            }
        }
    }

    // Build factored power table for X spatial
    let mut alpha_x = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
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
    let mut wpp = vec![AlmostGoldilocksExt2::zero(); s_kernel_pad];
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

    // Return: 5 proofs (transcript order), claims =
    // [y_self_claim, x_claim, wf_claim, y_claim_b]
    (
        vec![proof1, proof_b, proof2, proof3, proof4],
        vec![y_self_claim, x_claim, wf_claim, y_claim_b],
    )
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
    // claims layout: [y_self_claim, x_claim, wf_claim, y_claim_b, out_claim]
    // proofs layout (transcript order): [p1, pB, p2, p3, p4]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];
    let y_claim_b = claims[3];

    let l_spatial_out = conv.l_spatial_out();
    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_wi = log2_ceil(conv.input_w.max(1));

    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let beta_w = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = AlmostGoldilocksExt2::one();
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
        let one = AlmostGoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2)), ext2_mul(a, b)),
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

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y ----
    // Σ_{d,m} eq_D[d]·α^m·Y[d,m] = s_alpha_conv. ConvTranspose2D is junk-free
    // (see prover): the full polynomial product IS the padded output, so the
    // α-sum binds the committed Y directly — no aux Y_full edge. The implied
    // sum of this sumcheck IS s_alpha_conv — derived from the committed Y
    // instead of a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_spatial_out == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_spatial_out, 2, transcript,
    );
    if !ok_b {
        println!("ConvTranspose2D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YP(r_m), where YP(r_m) = Y(r_m, r_d)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, y_claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("ConvTranspose2D sumcheck B final eval mismatch");
        return false;
    }
    // The Y claim must sit exactly at (r_m, r_d).
    if y_claim_b.point.len() != l_spatial_out + l_d {
        println!("ConvTranspose2D sumcheck B claim point arity mismatch");
        return false;
    }
    for i in 0..l_spatial_out {
        if !crate::util::arith::ext2_field_eq(y_claim_b.point[i], challenges_b[i]) {
            println!("ConvTranspose2D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_d {
        if !crate::util::arith::ext2_field_eq(y_claim_b.point[l_spatial_out + i], r_d[i]) {
            println!("ConvTranspose2D sumcheck B claim point (r_d) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2 ----
    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("ConvTranspose2D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = ext2_add(
        sumcheck_proofs[3].round_messages[0][0],
        sumcheck_proofs[3].round_messages[0][1],
    );

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_3, l_spatial_in, 2, transcript,
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
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("ConvTranspose2D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("ConvTranspose2D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("ConvTranspose2D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[4].final_eval {
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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let results: Vec<(usize, AlmostGoldilocksField)> = (0..total_outputs)
            .into_par_iter()
            .map(|flat_idx| {
                let ow = flat_idx % w_out;
                let oh = (flat_idx / w_out) % h_out;
                let od = (flat_idx / (w_out * h_out)) % d_out;
                let d = flat_idx / (w_out * h_out * d_out);
                let mut acc = AlmostGoldilocksField(0);
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
                                acc = agl_add(acc, agl_mul(x_val, w_val));
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
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output)]
    }

    /// GPU forward. 3D-UNet's decoder has one of these per level and this was
    /// the ONLY block in that model's path without a GPU path, so a 6-level
    /// run sat at 0% GPU utilisation while five CPU cores worked: 362s per
    /// 64^3 volume. Index math mirrors `run` exactly; the gather form is used
    /// (one thread per output) because the scatter form would need atomics.
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Escape hatch for bisecting GPU-side faults: compute-sanitizer
        // needs a newer driver than this box has, so isolating which
        // kernel misbehaves means turning them off one at a time.
        if std::env::var("ZK4_CT3D_CPU").is_ok() {
            return self.run(inputs);
        }
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

        let d_x = x.as_device_buf();
        let d_w = w_flat.as_device_buf();
        let mut d_y = DeviceBuffer::<u64>::new(out_size)
            .expect("ConvTranspose3D: alloc out");
        // The kernel writes only the valid output box; the padded remainder
        // must already be zero.
        zero_buffer(&mut d_y, out_size).expect("ConvTranspose3D: memset zero failed");

        gpu_conv_transpose3d(
            &d_x, &d_w, &mut d_y,
            self.c_out, self.d_out, self.h_out, self.w_out,
            self.c_in, self.kernel_d, self.kernel_h, self.kernel_w,
            self.stride_d, self.stride_h, self.stride_w,
            self.input_d, self.input_h, self.input_w,
            w_in_pad, h_in_pad, d_in_pad,
            c_out_pad, s_kernel_pad,
            w_out_pad, h_out_pad, d_out_pad,
            self.flat_stride_h, self.flat_stride_w,
        ).expect("ConvTranspose3D: gpu kernel failed");

        let out_shape = vec![self.c_out, self.d_out, self.h_out, self.w_out];
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);
        vec![Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output)]
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

    // Build YP[k] = Σ_d Y[d,k] · eq_D[d] over the FULL padded box: the same
    // fold feeds sumcheck B, which must bind Y's padded region (zero in the
    // full polynomial product) too, so garbage in padding is rejected.
    let y_data = witnesses[2];
    let y_slice = y_data.data.as_ref().unwrap().evaluations_ref();
    let mut yp = vec![AlmostGoldilocksExt2::zero(); s_out_pad];
    for d in 0..c_out_pad {
        for k in 0..s_out_pad {
            let y_val = AlmostGoldilocksExt2::from_base(y_slice[k + d * s_out_pad]);
            yp[k] = ext2_add(yp[k], ext2_mul(eq_d[d], y_val));
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

    // ---- Sumcheck B: bind s_alpha_conv DIRECTLY to committed Y ----
    // ConvTranspose3D is junk-free: the α-exponent of each contribution is
    //   m = od·h_out_pad·w_out_pad + oh·w_out_pad + ow
    // — exactly the flat little-endian PADDED output index (carry-free since
    // ow < w_out ≤ w_out_pad, oh < h_out ≤ h_out_pad, od < d_out). The full
    // polynomial product coincides with the committed Y over the whole padded
    // box (unreachable and padded slots are zero on both sides). The crop map
    // E is the identity, so no aux Y_full edge, no masked-view sumcheck, and
    // no power-of-two stride requirement is needed:
    //   Σ_{d,m} eq_D[d]·α^m·Y[d,m] = s_alpha_conv.
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // the committed output rather than a free prover scalar.
    let alpha_full = alpha_power_table(alpha, s_out_pad);

    let mut prover_b = CpuLinearSumcheckProverExt2::new(l_spatial_out, 2, transcript);
    let proof_b = prover_b.prove(&mut [alpha_full, yp.clone()].as_mut_slice(), transcript);
    let r_m = prover_b.challenges.clone();

    // YP(r_m) = Y(r_m, r_d)
    let mut y_point_b = Vec::with_capacity(l_spatial_out + l_d);
    y_point_b.extend_from_slice(&r_m);
    y_point_b.extend_from_slice(r_d);
    let y_claim_b = Claim {
        edge_id: y_edge,
        sparse_id: 0,
        point: y_point_b,
        eval: prover_b.final_eval(1),
    };

    // ---- Sumcheck 2: Channel F×G ----
    let beta_w = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride_w) { b = ext2_mul(b, alpha); }
        b
    };
    let beta_d = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..(conv.stride_d * conv.flat_stride_h) { b = ext2_mul(b, alpha); }
        b
    };

    // Build WP[c, j] = Σ_d W_flat[c, d, j] · eq_D[d]
    let wf_data = witnesses[1];
    let mut wp = vec![AlmostGoldilocksExt2::zero(); c_in_pad * s_kernel_pad];
    for c in 0..conv.c_in {
        for d in 0..conv.c_out {
            for j in 0..conv.s_kernel {
                let wf_idx = j + d * s_kernel_pad + c * s_kernel_pad * c_out_pad;
                let wf_val = AlmostGoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
                wp[c * s_kernel_pad + j] = ext2_add(wp[c * s_kernel_pad + j], ext2_mul(eq_d[d], wf_val));
            }
        }
    }

    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
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
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_in_pad];
    for c in 0..conv.c_in {
        for jd in 0..conv.input_d {
            for jh in 0..conv.input_h {
                for jw in 0..conv.input_w {
                    let x_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let power = ext2_mul(ext2_mul(beta_d_table[jd], beta_h_table[jh]), beta_w_table[jw]);
                    f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, power));
                }
            }
        }
    }

    // Σ_c F[c]·G[c] = Σ_m FullConv[m]·α^m — the same sum sumcheck B bound to
    // the committed Y. The verifier checks the two sumchecks' implied sums
    // against each other, so no scalar travels outside the transcript.

    let mut prover2 = CpuLinearSumcheckProverExt2::new(l_c, 2, transcript);
    let proof2 = prover2.prove(&mut [f_poly.clone(), g_poly.clone()].as_mut_slice(), transcript);
    let r_c = prover2.challenges.clone();

    // ---- Sumcheck 3: F → X ----
    let eq_c = evaluate_lagrange_basis_ext2(&r_c);
    let mut xp = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.c_in {
        for jd in 0..conv.input_d {
            for jh in 0..conv.input_h {
                for jw in 0..conv.input_w {
                    let x_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad
                        + c * w_in_pad * h_in_pad * d_in_pad;
                    let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                    let flat_idx = jw + jh * w_in_pad + jd * w_in_pad * h_in_pad;
                    xp[flat_idx] = ext2_add(xp[flat_idx], ext2_mul(eq_c[c], x_val));
                }
            }
        }
    }

    let mut alpha_x = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
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
    let mut wpp = vec![AlmostGoldilocksExt2::zero(); s_kernel_pad];
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

    // Return: 5 proofs (transcript order), claims =
    // [y_self_claim, x_claim, wf_claim, y_claim_b]
    (
        vec![proof1, proof_b, proof2, proof3, proof4],
        vec![y_self_claim, x_claim, wf_claim, y_claim_b],
    )
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
    // claims layout: [y_self_claim, x_claim, wf_claim, y_claim_b, out_claim]
    // proofs layout (transcript order): [p1, pB, p2, p3, p4]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];
    let y_claim_b = claims[3];

    let l_spatial_out = conv.l_spatial_out();
    let l_d = conv.l_d();
    let l_c = conv.l_c();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_wi = log2_ceil(conv.input_w.max(1));
    let l_hi = log2_ceil(conv.input_h.max(1));

    let r_spatial = &out_claim.point[..l_spatial_out];
    let r_d = &out_claim.point[l_spatial_out..l_spatial_out + l_d];
    let v = out_claim.eval;

    let alpha = transcript.challenge_ext2(b"conv_alpha");

    let beta_w = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..conv.stride_w { b = ext2_mul(b, alpha); }
        b
    };
    let beta_h = {
        let mut b = AlmostGoldilocksExt2::one();
        for _ in 0..(conv.stride_h * conv.flat_stride_w) { b = ext2_mul(b, alpha); }
        b
    };
    let beta_d = {
        let mut b = AlmostGoldilocksExt2::one();
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
        let one = AlmostGoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2)), ext2_mul(a, b)),
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

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y ----
    // Σ_{d,m} eq_D[d]·α^m·Y[d,m] = s_alpha_conv. ConvTranspose3D is junk-free
    // (see prover): the full polynomial product IS the padded output, so the
    // α-sum binds the committed Y directly — no aux Y_full edge. The implied
    // sum of this sumcheck IS s_alpha_conv — derived from the committed Y
    // instead of a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_spatial_out == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1], s_alpha_conv, l_spatial_out, 2, transcript,
    );
    if !ok_b {
        println!("ConvTranspose3D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YP(r_m), where YP(r_m) = Y(r_m, r_d)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, y_claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("ConvTranspose3D sumcheck B final eval mismatch");
        return false;
    }
    // The Y claim must sit exactly at (r_m, r_d).
    if y_claim_b.point.len() != l_spatial_out + l_d {
        println!("ConvTranspose3D sumcheck B claim point arity mismatch");
        return false;
    }
    for i in 0..l_spatial_out {
        if !crate::util::arith::ext2_field_eq(y_claim_b.point[i], challenges_b[i]) {
            println!("ConvTranspose3D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_d {
        if !crate::util::arith::ext2_field_eq(y_claim_b.point[l_spatial_out + i], r_d[i]) {
            println!("ConvTranspose3D sumcheck B claim point (r_d) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2 ----
    let (ok2, _challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2], s_alpha_conv, l_c, 2, transcript,
    );
    if !ok2 {
        println!("ConvTranspose3D sumcheck 2 verification failed");
        return false;
    }

    // ---- Verify Sumcheck 3 ----
    // For a 0-round sumcheck (input spatial 1×1×1), final_eval IS the sum —
    // but the verifier must STILL replay SumcheckVerifier::verify so the
    // transcript stays aligned with the prover (CpuLinearSumcheckProverExt2::new
    // appends num_var/num_poly even for 0 rounds); the generic final-eval
    // check below degenerates correctly (empty-slice α-table MLE = 1).
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3], inferred_sum_3, l_spatial_in, 2, transcript,
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
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("ConvTranspose3D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    // For 0-round sumcheck (1×1×1 kernel: l_kernel=0), final_eval IS the sum.
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    let fg_product = ext2_mul(inferred_sum_3, inferred_sum_4);
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("ConvTranspose3D F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4], inferred_sum_4, l_kernel, 2, transcript,
    );
    if !ok4 {
        println!("ConvTranspose3D sumcheck 4 verification failed");
        return false;
    }

    let alpha_mle_4 = alpha_table_mle_eval(alpha, &challenges4);
    let expected_final_4 = ext2_mul(alpha_mle_4, wf_claim.eval);
    if expected_final_4 != sumcheck_proofs[4].final_eval {
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
    /// Length of the full 1D flat convolution (all α-exponents, incl. junk).
    fn s_full(&self) -> usize { self.s_in + self.s_kernel - 1 }
    /// Number of variables for the full-conv exponent dimension.
    fn l_full(&self) -> usize { log2_ceil(self.s_full().max(1)) }

    /// Full-conv exponent of output position (ho, wo):
    ///   E(ho, wo) = (s_in − 1) − (cs_h·w_pad·ho + cs_w·wo)
    /// (the input is reversed in F, so output at flat input offset t sits at
    /// exponent s_in−1−t of the polynomial product).
    fn view_exponent(&self, ho: usize, wo: usize) -> usize {
        (self.s_in - 1) - (ho * self.conv_stride_h * self.stride_w + wo * self.conv_stride_w)
    }

    /// Bit layout of the crop map E for the bit-affine view: returns
    /// (wo_shift, ho_shift, l_si). Valid only when both conv strides are
    /// powers of two, so E's subtrahend has carry-free disjoint bit fields:
    /// wo bits occupy [wo_shift, wo_shift+l_wo), ho bits occupy
    /// [ho_shift, ho_shift+l_ho), and E = bitwise complement over the low
    /// l_si bits (s_in−1 is all-ones). Asserts the disjointness invariants
    /// the soundness of the view depends on.
    fn view_bit_layout(&self) -> (usize, usize, usize) {
        assert!(
            self.conv_stride_h.is_power_of_two() && self.conv_stride_w.is_power_of_two(),
            "DepthwiseConv2D bit-affine view requires power-of-two conv strides, got ({}, {})",
            self.conv_stride_h, self.conv_stride_w
        );
        let lw = log2_ceil(self.input_w.max(1)); // log2(w_pad)
        let lh = log2_ceil(self.input_h.max(1)); // log2(h_pad)
        let l_si = lw + lh; // log2(s_in)
        let wo_shift = self.conv_stride_w.trailing_zeros() as usize;
        let ho_shift = lw + self.conv_stride_h.trailing_zeros() as usize;
        // Bit fields must be disjoint and within the s_in range for E to be
        // bit-affine over the whole padded output box.
        assert!(wo_shift + self.l_wo() <= ho_shift,
            "DepthwiseConv2D view: wo bit-field [{}+{}) overlaps ho bit-field at {}",
            wo_shift, self.l_wo(), ho_shift);
        assert!(ho_shift + self.l_ho() <= l_si,
            "DepthwiseConv2D view: ho bit-field [{}+{}) exceeds s_in bits {}",
            ho_shift, self.l_ho(), l_si);
        (wo_shift, ho_shift, l_si)
    }

    /// σ̃: map a point over Y's output-spatial vars to the Y_full exponent
    /// point such that view(Y)(r) = Y_full(σ̃(r), r_c). Each exponent bit is
    /// either a complemented spatial coordinate or a public constant.
    fn view_point(&self, r_spatial: &[AlmostGoldilocksExt2]) -> Vec<AlmostGoldilocksExt2> {
        let (wo_shift, ho_shift, l_si) = self.view_bit_layout();
        let l_wo = self.l_wo();
        let l_ho = self.l_ho();
        assert_eq!(r_spatial.len(), l_wo + l_ho);
        let one = AlmostGoldilocksExt2::one();
        let mut point = vec![one; self.l_full()];
        // Bits ≥ l_si of E are zero (E < s_in).
        for coord in point.iter_mut().skip(l_si) {
            *coord = AlmostGoldilocksExt2::zero();
        }
        for i in 0..l_wo {
            point[wo_shift + i] = ext2_sub(one, r_spatial[i]);
        }
        for i in 0..l_ho {
            point[ho_shift + i] = ext2_sub(one, r_spatial[l_wo + i]);
        }
        point
    }

    /// Compute the aux Y_full witness: the per-channel full 1D flat convolution
    ///   Y_full[c, m] = Σ_{i+j=m} X_rev[c,i]·W_flat[c,j]
    /// scattered as m = (s_in−1−p) + j over real input positions p and real
    /// kernel taps j. Challenge-independent, committed as advice
    /// (Role::Auxiliary) and bound to s_alpha_conv by sumcheck B.
    fn compute_y_full(
        &self,
        x_slice: &[AlmostGoldilocksField],
        w_slice: &[AlmostGoldilocksField],
        out_sf: usize,
    ) -> Witness {
        let h_in_pad = self.input_h.next_power_of_two();
        let w_in_pad = self.input_w.next_power_of_two();
        let s_kernel_pad = self.s_kernel.next_power_of_two();
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let mut yfull = vec![AlmostGoldilocksField(0); c_pad * s_full_pad];
        let rows: Vec<(usize, Vec<AlmostGoldilocksField>)> = (0..self.channels)
            .into_par_iter()
            .map(|c| {
                let mut row = vec![AlmostGoldilocksField(0); s_full_pad];
                for kh in 0..self.kernel_h {
                    for kw in 0..self.kernel_w {
                        let j = kh * self.stride_w + kw;
                        let wf_idx = j + c * s_kernel_pad;
                        let w_val = w_slice[wf_idx];
                        for ih in 0..self.input_h {
                            for iw in 0..self.input_w {
                                let p = ih * self.stride_w + iw;
                                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                                let m = (self.s_in - 1 - p) + j;
                                row[m] = agl_add(row[m], agl_mul(x_slice[x_idx], w_val));
                            }
                        }
                    }
                }
                (c, row)
            })
            .collect();
        for (c, row) in rows {
            yfull[c * s_full_pad..c * s_full_pad + s_full_pad].copy_from_slice(&row);
        }
        Witness::new(
            vec![self.channels, s_full],
            yfull,
            DataType::Uint,
            out_sf,
            Role::Auxiliary,
        )
    }
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
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];

        // Y[c, ho, wo] = Σ_kh Σ_kw X[c, ho*sh+kh, wo*sw+kw] * W[c, kh*stride_w+kw]
        for c in 0..self.channels {
            for ho in 0..self.h_out {
                for wo in 0..self.w_out {
                    let mut acc = AlmostGoldilocksField(0);
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let ih = ho * self.conv_stride_h + kh;
                            let iw = wo * self.conv_stride_w + kw;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let j = kh * self.stride_w + kw;
                            let wf_idx = j + c * s_kernel_pad;
                            let x_val = x.data.as_ref().unwrap().index(x_idx);
                            let w_val = w_flat.data.as_ref().unwrap().index(wf_idx);
                            acc = agl_add(acc, agl_mul(x_val, w_val));
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
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        let out_sf = inputs.get(1).map(|w| inputs[0].sf + w.sf).unwrap_or(inputs[0].sf);

        // Aux output: Y_full, the per-channel full 1D flat convolution.
        // Sumcheck B binds it to s_alpha_conv (and hence to X, W_flat); the
        // masked-view sumcheck C binds Y to its valid coefficients.
        let y_full = self.compute_y_full(
            x.data.as_ref().unwrap().evaluations_ref(),
            w_flat.data.as_ref().unwrap().evaluations_ref(),
            out_sf,
        );
        #[cfg(debug_assertions)]
        {
            // Valid coefficients of Y_full must reproduce Y exactly.
            let s_full_pad = self.s_full().next_power_of_two();
            let yf = y_full.data.as_ref().unwrap().evaluations_ref();
            for c in 0..self.channels {
                for ho in 0..self.h_out {
                    for wo in 0..self.w_out {
                        let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                        let e = self.view_exponent(ho, wo);
                        debug_assert_eq!(
                            out_data[out_idx], yf[c * s_full_pad + e],
                            "DepthwiseConv2D: Y_full valid coefficient mismatch at (c={c}, ho={ho}, wo={wo})"
                        );
                    }
                }
            }
        }

        vec![Witness::new(out_shape, out_data, DataType::Uint, out_sf, Role::Output), y_full]
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
        zero_buffer(&mut d_y, out_size).expect("DepthwiseConv2D: memset zero");

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
        let out_sf = inputs[0].sf + inputs[1].sf;

        // Aux Y_full gathered on-device: one thread per (c, m) exponent, tap
        // index j = kh·stride_w + kw, channel loop collapsed to c ≡ d
        // (see `agl_conv_full_kernel`). Matches `compute_y_full` element-wise.
        let s_full = self.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let mut d_yf = DeviceBuffer::<u64>::new(c_pad * s_full_pad).expect("DepthwiseConv2D: alloc Y_full");
        zero_buffer(&mut d_yf, c_pad * s_full_pad).expect("DepthwiseConv2D: memset Y_full zero failed");
        gpu_conv_full(
            &d_x, &d_w, &mut d_yf,
            self.channels, 1,
            1, self.kernel_h, self.kernel_w,
            0, self.stride_w, 1,
            self.s_in, s_full, s_full_pad,
            1, s_kernel_pad,
            true,
            // DepthwiseConv2D is not batched yet: batch = 1.
            1, 0, 0,
        ).expect("DepthwiseConv2D: gpu Y_full kernel failed");

        vec![
            Witness::new_device(out_shape, Arc::new(d_y), DataType::Uint, out_sf, Role::Output),
            Witness::new_device(vec![self.channels, s_full], Arc::new(d_yf), DataType::Uint, out_sf, Role::Auxiliary),
        ]
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
    // edge_ids: [x_edge, wf_edge, y_edge, yfull_edge]
    // witnesses: [X, W_flat, Y, Y_full]
    let x_edge = edge_ids[0];
    let wf_edge = edge_ids[1];
    let y_edge = edge_ids[2];
    let yfull_edge = edge_ids[3];

    let out_claim = out_claims[0];
    assert_eq!(out_claim.edge_id, y_edge);

    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

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
    let mut yp = vec![AlmostGoldilocksExt2::zero(); s_out_pad];
    for c in 0..conv.channels {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let y_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                let y_val = AlmostGoldilocksExt2::from_base(y_data.data.as_ref().unwrap().index(y_idx));
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

    // ---- Sumcheck B: bind s_alpha_conv to committed Y_full ----
    // Σ_{c,m} eq_C[c]·α^m·Y_full[c,m] = s_alpha_conv.
    // The verifier infers s_alpha_conv from this sumcheck's round-0 messages
    // and feeds it to sumcheck 2 as the expected sum, so it is DERIVED from
    // the committed full conv rather than a free prover scalar. Together with
    // sumchecks 2/3/4 (which bind the same sum to X, W_flat via eq_C·F·G),
    // Schwartz-Zippel over α gives Y_full = FullConv(X, W) coefficient-wise.
    let yfull_data = witnesses[3];
    let s_full = conv.s_full();
    let s_full_pad = s_full.next_power_of_two();
    let yfull_slice = yfull_data.data.as_ref().unwrap().evaluations_ref();
    let mut yfp = vec![AlmostGoldilocksExt2::zero(); s_full_pad];
    for c in 0..conv.channels {
        for m in 0..s_full {
            let v = AlmostGoldilocksExt2::from_base(yfull_slice[c * s_full_pad + m]);
            yfp[m] = ext2_add(yfp[m], ext2_mul(eq_c[c], v));
        }
    }
    let alpha_full = alpha_power_table(alpha, s_full_pad);

    let mut prover_b = CpuLinearSumcheckProverExt2::new(l_full, 2, transcript);
    let proof_b = prover_b.prove(&mut [alpha_full, yfp].as_mut_slice(), transcript);
    let r_m = prover_b.challenges.clone();

    // YFP(r_m) = Y_full(r_m, r_c)
    let mut yfull_point_b = Vec::with_capacity(l_full + l_c);
    yfull_point_b.extend_from_slice(&r_m);
    yfull_point_b.extend_from_slice(r_c);
    let yfull_claim_b = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_b,
        eval: prover_b.final_eval(1),
    };

    // ---- Sumcheck 2: Degree-3 channel sumcheck ----
    // For depthwise conv, both F and G depend on channel c.
    // F[c] = Σ_i X_rev[c,i] · α^i
    // G[c] = Σ_j W_flat[c,j] · α^j
    // Prove: Σ_c eq_C(c) · F(c) · G(c) = s_alpha_conv (bound to Y_full by sumcheck B)

    // Build F[c] = Σ_i X_rev[c, i] · α^i
    let x_data = witnesses[0];
    let alpha_in = alpha_power_table(alpha, s_in_pad);
    let mut f_poly = vec![AlmostGoldilocksExt2::zero(); c_pad];
    for c in 0..conv.channels {
        for ih in 0..conv.input_h {
            for iw in 0..conv.input_w {
                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
                let i_flat = ih * conv.stride_w + iw;
                let rev_i = conv.s_in - 1 - i_flat;
                f_poly[c] = ext2_add(f_poly[c], ext2_mul(x_val, alpha_in[rev_i]));
            }
        }
    }

    // Build G[c] = Σ_j W_flat[c, j] · α^j
    let wf_data = witnesses[1];
    let alpha_kernel = alpha_power_table(alpha, s_kernel_pad);
    let mut g_poly = vec![AlmostGoldilocksExt2::zero(); c_pad];
    for c in 0..conv.channels {
        for j in 0..conv.s_kernel {
            let wf_idx = j + c * s_kernel_pad;
            let wf_val = AlmostGoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
            g_poly[c] = ext2_add(g_poly[c], ext2_mul(wf_val, alpha_kernel[j]));
        }
    }

    // eq_C is already computed above
    let eq_c_poly = evaluate_lagrange_basis_ext2(r_c);

    // Σ_c eq_C[c]·F[c]·G[c] = Σ_m (Σ_c eq_C[c]·FullConv[c,m])·α^m — the same
    // sum sumcheck B bound to the committed Y_full. The verifier checks the
    // two sumchecks' implied sums against each other, so no scalar travels
    // outside the transcript.

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
    let mut xp = vec![AlmostGoldilocksExt2::zero(); s_in_pad];
    for c in 0..conv.channels {
        for ih in 0..conv.input_h {
            for iw in 0..conv.input_w {
                let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                let x_val = AlmostGoldilocksExt2::from_base(x_data.data.as_ref().unwrap().index(x_idx));
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
    let one = AlmostGoldilocksExt2::one();
    let r_spatial_x: Vec<AlmostGoldilocksExt2> = r_i.iter().map(|&ri| ext2_sub(one, ri)).collect();

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

    let mut wp = vec![AlmostGoldilocksExt2::zero(); s_kernel_pad];
    for c in 0..conv.channels {
        for j in 0..conv.s_kernel {
            let wf_idx = j + c * s_kernel_pad;
            let wf_val = AlmostGoldilocksExt2::from_base(wf_data.data.as_ref().unwrap().index(wf_idx));
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

    // ---- Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // At the (already random) y_self point r*:
    //   Y(r*) = Σ_x eq(r*, x) · mask(x) · Y_full[c(x), E(spatial(x))]
    // mask = [wo<w_out]·[ho<h_out]·[c<channels] pins Y's real region to the
    // valid full-conv coefficients AND its padded region to zero. Degree-3
    // sumcheck; the view factor exits as a Y_full claim at the bit-affine
    // point σ̃(r').
    let r_star = &y_self_claim.point;
    let l_out_n = l_spatial_out + l_c;
    let eq_star = evaluate_lagrange_basis_ext2(r_star);
    debug_assert_eq!(eq_star.len(), 1 << l_out_n);

    let mut mask_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    let mut view_tab = vec![AlmostGoldilocksExt2::zero(); 1 << l_out_n];
    for c in 0..c_pad {
        for ho in 0..h_out_pad {
            for wo in 0..w_out_pad {
                let x_lin = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                let e = conv.view_exponent(ho, wo);
                view_tab[x_lin] =
                    AlmostGoldilocksExt2::from_base(yfull_slice[c * s_full_pad + e]);
                if c < conv.channels && ho < conv.h_out && wo < conv.w_out {
                    mask_tab[x_lin] = AlmostGoldilocksExt2::one();
                }
            }
        }
    }

    let mut prover_c = CpuLinearSumcheckProverExt2::new(l_out_n, 3, transcript);
    let proof_c = prover_c.prove(&mut [eq_star, mask_tab, view_tab].as_mut_slice(), transcript);
    let r_prime = prover_c.challenges.clone();

    // view MLE at r' = Y_full(σ̃(r'_spatial), r'_c)
    let mut yfull_point_c = conv.view_point(&r_prime[..l_spatial_out]);
    yfull_point_c.extend_from_slice(&r_prime[l_spatial_out..]);
    let yfull_claim_c = Claim {
        edge_id: yfull_edge,
        sparse_id: 0,
        point: yfull_point_c,
        eval: prover_c.final_eval(2),
    };

    // Return: 6 proofs (transcript order), claims =
    // [y_self_claim, x_claim, wf_claim, yfull_claim_b, yfull_claim_c]
    (
        vec![proof1, proof_b, proof2, proof3, proof4, proof_c],
        vec![y_self_claim, x_claim, wf_claim, yfull_claim_b, yfull_claim_c],
    )
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
    // claims layout: [y_self_claim, x_claim, wf_claim, yfull_claim_b, yfull_claim_c, out_claim]
    // proofs layout (transcript order): [p1, pB, p2, p3, p4, pC]
    let out_claim = claims.last().unwrap();
    let y_self_claim = claims[0];
    let x_claim = claims[1];
    let wf_claim = claims[2];
    let yfull_claim_b = claims[3];
    let yfull_claim_c = claims[4];

    let l_c = conv.l_c();
    let l_spatial_out = conv.l_spatial_out();
    let l_spatial_in = conv.l_spatial_in();
    let l_kernel = conv.l_kernel();
    let l_full = conv.l_full();

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
        let one = AlmostGoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_spatial_out {
            let a = r_spatial[i];
            let b = challenges1[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2)), ext2_mul(a, b)),
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

    // ---- Verify Sumcheck B: s_alpha_conv ↔ committed Y_full ----
    // Σ_{c,m} eq_C[c]·α^m·Y_full[c,m] = s_alpha_conv. The implied sum of this
    // sumcheck IS s_alpha_conv — derived from the committed Y_full instead of
    // a free prover scalar — and is fed to sumcheck 2 below.
    let s_alpha_conv = if l_full == 0 {
        sumcheck_proofs[1].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[1].round_messages[0][0],
            sumcheck_proofs[1].round_messages[0][1],
        )
    };

    let (ok_b, challenges_b) = SumcheckVerifier::verify(
        sumcheck_proofs[1],
        s_alpha_conv,
        l_full,
        2,
        transcript,
    );
    if !ok_b {
        println!("DepthwiseConv2D sumcheck B verification failed");
        return false;
    }

    // final_eval = α_table_mle(r_m) · YFP(r_m), where YFP(r_m) = Y_full(r_m, r_c)
    let alpha_mle_b = alpha_table_mle_eval(alpha, &challenges_b);
    let expected_final_b = ext2_mul(alpha_mle_b, yfull_claim_b.eval);
    if expected_final_b != sumcheck_proofs[1].final_eval {
        println!("DepthwiseConv2D sumcheck B final eval mismatch");
        return false;
    }
    // The Y_full claim must sit exactly at (r_m, r_c).
    if yfull_claim_b.point.len() != l_full + l_c {
        println!("DepthwiseConv2D sumcheck B claim point arity mismatch");
        return false;
    }
    for i in 0..l_full {
        if !crate::util::arith::ext2_field_eq(yfull_claim_b.point[i], challenges_b[i]) {
            println!("DepthwiseConv2D sumcheck B claim point mismatch");
            return false;
        }
    }
    for i in 0..l_c {
        if !crate::util::arith::ext2_field_eq(yfull_claim_b.point[l_full + i], r_c[i]) {
            println!("DepthwiseConv2D sumcheck B claim point (r_c) mismatch");
            return false;
        }
    }

    // ---- Verify Sumcheck 2: Degree-3 channel sumcheck ----
    let (ok2, challenges2) = SumcheckVerifier::verify(
        sumcheck_proofs[2],
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
        let one = AlmostGoldilocksExt2::one();
        let mut prod = one;
        for i in 0..l_c {
            let a = r_c[i];
            let b = challenges2[i];
            let term = ext2_add(
                ext2_sub(one, ext2_add(a, b)),
                ext2_mul(AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2)), ext2_mul(a, b)),
            );
            prod = ext2_mul(prod, term);
        }
        prod
    };

    // ---- Verify Sumcheck 3 ----
    let inferred_sum_3 = if l_spatial_in == 0 {
        sumcheck_proofs[3].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[3].round_messages[0][0],
            sumcheck_proofs[3].round_messages[0][1],
        )
    };

    let (ok3, challenges3) = SumcheckVerifier::verify(
        sumcheck_proofs[3],
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
    if expected_final_3 != sumcheck_proofs[3].final_eval {
        println!("DepthwiseConv2D sumcheck 3 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck 4 ----
    let inferred_sum_4 = if l_kernel == 0 {
        sumcheck_proofs[4].final_eval
    } else {
        ext2_add(
            sumcheck_proofs[4].round_messages[0][0],
            sumcheck_proofs[4].round_messages[0][1],
        )
    };

    // Cross-check: eq_C(r_c_new) * F(r_c_new) * G(r_c_new) = final_eval_2
    // F(r_c_new) = inferred_sum_3, G(r_c_new) = inferred_sum_4
    let fg_product = ext2_mul(eq_c_val, ext2_mul(inferred_sum_3, inferred_sum_4));
    if fg_product != sumcheck_proofs[2].final_eval {
        println!("DepthwiseConv2D eq*F*G product check failed");
        return false;
    }

    let (ok4, challenges4) = SumcheckVerifier::verify(
        sumcheck_proofs[4],
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
    if expected_final_4 != sumcheck_proofs[4].final_eval {
        println!("DepthwiseConv2D sumcheck 4 final eval mismatch");
        return false;
    }

    // ---- Verify Sumcheck C: masked-view consistency Y ≡ mask · view(Y_full) ----
    // Y(r*) = Σ_x eq(r*, x)·mask(x)·Y_full[c(x), E(spatial(x))], with r* the
    // y_self point. Binds Y's real region to the valid full-conv coefficients
    // and Y's padded region to zero.
    let l_out_n = l_spatial_out + l_c;
    if y_self_claim.point.len() != l_out_n {
        println!("DepthwiseConv2D sumcheck C: y_self point arity mismatch");
        return false;
    }
    let (ok_c, challenges_c) = SumcheckVerifier::verify(
        sumcheck_proofs[5],
        y_self_claim.eval,
        l_out_n,
        3,
        transcript,
    );
    if !ok_c {
        println!("DepthwiseConv2D sumcheck C verification failed");
        return false;
    }

    // final_eval = eq(r*, r') · mask(r') · Y_full(σ̃(r'_spatial), r'_c)
    let eq_c_final = eq_points_ext2(&y_self_claim.point, &challenges_c);
    let mask_final = depthwise_conv2d_mask_mle_eval(conv, &challenges_c);
    let expected_final_c = ext2_mul(ext2_mul(eq_c_final, mask_final), yfull_claim_c.eval);
    if expected_final_c != sumcheck_proofs[5].final_eval {
        println!("DepthwiseConv2D sumcheck C final eval mismatch");
        return false;
    }
    // The Y_full claim must sit exactly at the bit-affine view point.
    let mut expected_point_c = conv.view_point(&challenges_c[..l_spatial_out]);
    expected_point_c.extend_from_slice(&challenges_c[l_spatial_out..]);
    if yfull_claim_c.point.len() != expected_point_c.len() {
        println!("DepthwiseConv2D sumcheck C claim point arity mismatch");
        return false;
    }
    for i in 0..expected_point_c.len() {
        if !crate::util::arith::ext2_field_eq(yfull_claim_c.point[i], expected_point_c[i]) {
            println!("DepthwiseConv2D sumcheck C claim point mismatch");
            return false;
        }
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
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<AlmostGoldilocksField> = data.into_iter().map(AlmostGoldilocksField).collect();
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(110));
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(140));
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(170));
        assert_eq!(y.data.as_ref().unwrap().index(3), AlmostGoldilocksField(200));

        // Y[d=1, h, w] = X[c=0,h,w]*30 + X[c=1,h,w]*40
        // = [1*30+5*40, 2*30+6*40, 3*30+7*40, 4*30+8*40]
        // = [230, 300, 370, 440]
        assert_eq!(y.data.as_ref().unwrap().index(4), AlmostGoldilocksField(230));
        assert_eq!(y.data.as_ref().unwrap().index(5), AlmostGoldilocksField(300));
        assert_eq!(y.data.as_ref().unwrap().index(6), AlmostGoldilocksField(370));
        assert_eq!(y.data.as_ref().unwrap().index(7), AlmostGoldilocksField(440));
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(54));
        // Y[0,0,1] = sum of 3x3 block starting at (0,1) = 2+3+4+6+7+8+10+11+12 = 63
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(63));
        // Y[0,1,0] = sum starting at (1,0) = 5+6+7+9+10+11+13+14+15 = 90
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(90));
        // Y[0,1,1] = sum starting at (1,1) = 6+7+8+10+11+12+14+15+16 = 99
        assert_eq!(y.data.as_ref().unwrap().index(3), AlmostGoldilocksField(99));
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
        assert_eq!(wf.data.as_ref().unwrap().index(0), AlmostGoldilocksField(1));
        assert_eq!(wf.data.as_ref().unwrap().index(1), AlmostGoldilocksField(2));
        assert_eq!(wf.data.as_ref().unwrap().index(2), AlmostGoldilocksField(3));
        assert_eq!(wf.data.as_ref().unwrap().index(3), AlmostGoldilocksField(0));
        assert_eq!(wf.data.as_ref().unwrap().index(4), AlmostGoldilocksField(4));
        assert_eq!(wf.data.as_ref().unwrap().index(5), AlmostGoldilocksField(5));
        assert_eq!(wf.data.as_ref().unwrap().index(6), AlmostGoldilocksField(6));
        assert_eq!(wf.data.as_ref().unwrap().index(7), AlmostGoldilocksField(0));
        assert_eq!(wf.data.as_ref().unwrap().index(8), AlmostGoldilocksField(7));
        assert_eq!(wf.data.as_ref().unwrap().index(9), AlmostGoldilocksField(8));
        assert_eq!(wf.data.as_ref().unwrap().index(10), AlmostGoldilocksField(9));
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
        let point: Vec<AlmostGoldilocksExt2> = (0..n_wf)
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
        let yfull = &result[1];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 6);
        assert_eq!(new_claims.len(), 5);

        // Verify
        let mut verify_transcript = Transcript::new(b"test_conv_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y, yfull],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv2D prove/verify should pass");

        // Y_full claims must open correctly against the actual Y_full MLE.
        for c in &[&new_claims[3], &new_claims[4]] {
            let got = yfull.data.as_ref().unwrap().evaluate_at_point_ext2(&c.point);
            assert_eq!(got, c.eval, "Y_full claim eval must match its MLE");
        }
    }

    #[test]
    fn test_conv2d_batched_matches_per_image_and_verifies() {
        // Two images against one shared set of weights. The batch index is
        // BOUND by the verifier's r_b rather than summed over, so this checks
        // both halves of that claim: the forward pass must agree image-by-image
        // with the unbatched conv, and the folded argument must verify.
        for batch in [2usize, 4] {
            let conv = Conv2D::new(1, 1, 2, 2, 4, 4).with_batch(batch);
            let single = Conv2D::new(1, 1, 2, 2, 4, 4);
            let fk = FlattenKernel {
                s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1,
                dilation_h: 1, dilation_w: 1,
            };

            // Distinct images, so a fold that ignored b (or read only image 0)
            // could not accidentally pass.
            let imgs: Vec<Vec<u64>> = (0..batch)
                .map(|b| (1..=16u64).map(|v| v * (b as u64 + 1) + b as u64).collect())
                .collect();
            let mut flat = Vec::new();
            for im in &imgs { flat.extend_from_slice(im); }
            let x = make_witness(vec![batch, 1, 4, 4], flat);

            let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
            let wf_result = fk.run(&[&w_raw]);
            let wf = &wf_result[0];

            let result = conv.run(&[&x, wf]);
            let y = &result[0];
            let yfull = &result[1];

            // ---- forward parity, image by image ----
            let y_all = y.data.as_ref().unwrap().evaluations();
            let stride = conv.y_stride();
            for (b, im) in imgs.iter().enumerate() {
                let xi = make_witness(vec![1, 4, 4], im.clone());
                let one = single.run(&[&xi, wf]);
                let yi = one[0].data.as_ref().unwrap().evaluations();
                assert_eq!(
                    &y_all[b * stride..b * stride + stride],
                    &yi[..stride],
                    "batch={} image {} must equal the unbatched conv", batch, b
                );
            }

            // ---- the folded argument must verify ----
            let n_y = y.data.as_ref().unwrap().n();
            let mut transcript = Transcript::new(b"test_conv_batched");
            let point: Vec<AlmostGoldilocksExt2> =
                (0..n_y).map(|_| transcript.challenge_ext2(b"ch")).collect();
            let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
            let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

            let mut tp = Transcript::new(b"test_conv_batched_prove");
            let (proofs, new_claims) =
                conv.prove(&[&x, wf, y, yfull], &[0, 1, 2, 3], &[&out_claim], &mut tp);

            let mut tv = Transcript::new(b"test_conv_batched_prove");
            let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
            all_claims.push(&out_claim);
            let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
            assert!(
                conv.verify(&[&x, wf, y, yfull], &all_claims, &proofs_ref, &mut tv),
                "batch={} folded conv must verify", batch
            );

            // Every emitted claim must open against the real MLE of its edge --
            // this is what catches an r_b appended in the wrong position.
            let x_claim = &new_claims[1];
            assert_eq!(
                x.data.as_ref().unwrap().evaluate_at_point_ext2(&x_claim.point),
                x_claim.eval,
                "batch={} X claim must match the batched X MLE", batch
            );
            for c in &[&new_claims[3], &new_claims[4]] {
                assert_eq!(
                    yfull.data.as_ref().unwrap().evaluate_at_point_ext2(&c.point),
                    c.eval,
                    "batch={} Y_full claim must match its MLE", batch
                );
            }
        }
    }

    #[test]
    fn test_conv_transpose2d_gpu_matches_cpu() {
        // PointPillars' RPN upsampling. The CPU path SCATTERS and the GPU path
        // GATHERS, so this is not a transcription check -- it verifies that
        // inverting the loop reproduces the same sums. Covers stride 1 and 2
        // (the RPN uses both) and non-pow2 extents.
        almost_goldilocks_cuda::init().expect("CUDA init");
        for (c_in, c_out, h, w, kh, kw, sh, sw) in [
            (1usize, 1usize, 4usize, 4usize, 2usize, 2usize, 2usize, 2usize),
            (2, 3, 4, 4, 2, 2, 2, 2),
            (3, 2, 5, 3, 2, 2, 2, 2),
            (2, 2, 4, 4, 3, 3, 1, 1),
        ] {
            let ct = ConvTranspose2D::new(c_in, c_out, kh, kw, h, w, sh, sw);
            let wp = w.next_power_of_two();
            let hp = h.next_power_of_two();
            let skp = ct.s_kernel.next_power_of_two();
            let cop = c_out.next_power_of_two();

            let mut xd = vec![0u64; c_in * hp * wp];
            let mut t = 0u64;
            for c in 0..c_in { for ih in 0..h { for iw in 0..w {
                t += 1; xd[iw + ih * wp + c * wp * hp] = (t % 13) + 1;
            }}}
            let x = make_witness(vec![c_in, h, w], xd);

            let mut wd = vec![0u64; c_in * cop * skp];
            let mut u = 0u64;
            for c in 0..c_in { for o in 0..c_out {
                for khi in 0..kh { for kwi in 0..kw {
                    u += 1;
                    wd[khi * ct.flat_stride + kwi + o * skp + c * skp * cop] = (u % 7) + 1;
                }}
            }}
            let wf = make_witness(vec![c_in, c_out, ct.s_kernel], wd);

            let cpu = ct.run(&[&x, &wf]);
            let gpu = ct.run_gpu(&[&x, &wf]);
            let av = cpu[0].data.as_ref().unwrap().evaluations();
            let bv = gpu[0].data.as_ref().unwrap().evaluations();
            assert_eq!(av.len(), bv.len(),
                "c_in={} c_out={} {}x{} k{}{} s{}{}: length differs",
                c_in, c_out, h, w, kh, kw, sh, sw);
            if let Some(k) = (0..av.len()).find(|&k| av[k] != bv[k]) {
                panic!("c_in={} c_out={} {}x{} k{}{} s{}{}: differs at {} \
                        (cpu {:?}, gpu {:?}); {} of {} slots differ",
                       c_in, c_out, h, w, kh, kw, sh, sw, k, av[k], bv[k],
                       (0..av.len()).filter(|&k| av[k] != bv[k]).count(), av.len());
            }
        }
    }

    #[test]
    fn bench_conv_transpose3d_cpu_vs_gpu() {
        // Decoder-scale timing. The gather form evaluates c_in*kd*kh*kw taps
        // per output and skips all but c_in of them when stride == kernel == 2,
        // so it does ~8x the loop iterations the CPU scatter does. Whether that
        // is offset by parallelism is a question for a clock, not an argument.
        almost_goldilocks_cuda::init().expect("CUDA init");
        let mut out = String::new();
        for (c_in, c_out, d) in [(64usize, 32usize, 32usize), (128, 64, 16)] {
            let ct = ConvTranspose3D::new(c_in, c_out, 2, 2, 2, d, d, d, 2, 2, 2);
            let dp = d.next_power_of_two();
            let skp = ct.s_kernel.next_power_of_two();
            let cop = c_out.next_power_of_two();
            let x = make_witness(vec![c_in, d, d, d],
                                 vec![1u64; c_in * dp * dp * dp]);
            let wf = make_witness(vec![c_in, c_out, ct.s_kernel],
                                  vec![1u64; c_in * cop * skp]);
            // Warm up BOTH paths first. The first CUDA launch in a freshly
            // built binary pays PTX JIT, which is seconds -- timing it as if
            // it were the kernel reported a 0.06x "regression" that was
            // entirely compilation.
            let _ = ct.run(&[&x, &wf]);
            let _ = ct.run_gpu(&[&x, &wf]);
            let t0 = std::time::Instant::now();
            let _ = ct.run(&[&x, &wf]);
            let cpu = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let _ = ct.run_gpu(&[&x, &wf]);
            let gpu = t1.elapsed().as_secs_f64();
            out += &format!("c_in={} c_out={} {}^3 -> cpu {:.3}s  gpu {:.3}s  ({:.2}x)\n",
                            c_in, c_out, d, cpu, gpu, cpu / gpu.max(1e-9));
        }
        let _ = std::fs::write("/tmp/zk4_ct3d_bench.txt", &out);
    }

    #[test]
    fn test_conv_transpose3d_gpu_matches_cpu() {
        // ConvTranspose3D is 3D-UNet's decoder block. The gather form skips
        // taps by three independent stride/bounds conditions, so an off-by-one
        // in any of them still yields a full-size Y -- just wrong in some
        // cells. Compare element for element against the CPU path, including
        // non-pow2 extents and stride > 1, which is what the decoder uses.
        almost_goldilocks_cuda::init().expect("CUDA init");
        for (c_in, c_out, d, h, w, kd, kh, kw, sd, sh, sw) in [
            (1usize, 1usize, 4usize, 4usize, 4usize, 2usize, 2usize, 2usize, 2usize, 2usize, 2usize),
            (2, 3, 4, 4, 4, 2, 2, 2, 2, 2, 2),
            (2, 2, 3, 5, 3, 2, 2, 2, 2, 2, 2),
            (3, 2, 4, 4, 4, 3, 3, 3, 1, 1, 1),
        ] {
            let ct = ConvTranspose3D::new(c_in, c_out, kd, kh, kw, d, h, w, sd, sh, sw);
            let wp = w.next_power_of_two();
            let hp = h.next_power_of_two();
            let dp = d.next_power_of_two();
            let skp = ct.s_kernel.next_power_of_two();
            let cop = c_out.next_power_of_two();

            // X in PADDED layout, zeros in the pad slots: the kernel gathers
            // over the flat extent and relies on that, exactly as conv_full does.
            let mut xd = vec![0u64; c_in * dp * hp * wp];
            let mut t = 0u64;
            for c in 0..c_in { for id in 0..d { for ih in 0..h { for iw in 0..w {
                t += 1;
                xd[iw + ih * wp + id * wp * hp + c * wp * hp * dp] = (t % 13) + 1;
            }}}}
            let x = make_witness(vec![c_in, d, h, w], xd);

            let mut wd = vec![0u64; c_in * cop * skp];
            let mut u = 0u64;
            for c in 0..c_in { for o in 0..c_out {
                for kdi in 0..kd { for khi in 0..kh { for kwi in 0..kw {
                    u += 1;
                    let j = kdi * ct.flat_stride_h + khi * ct.flat_stride_w + kwi;
                    wd[j + o * skp + c * skp * cop] = (u % 7) + 1;
                }}}
            }}
            let wf = make_witness(vec![c_in, c_out, ct.s_kernel], wd);

            let cpu = ct.run(&[&x, &wf]);
            let gpu = ct.run_gpu(&[&x, &wf]);
            let av = cpu[0].data.as_ref().unwrap().evaluations();
            let bv = gpu[0].data.as_ref().unwrap().evaluations();
            assert_eq!(av.len(), bv.len(),
                "c_in={} c_out={} {}x{}x{} k{}{}{} s{}{}{}: length differs",
                c_in, c_out, d, h, w, kd, kh, kw, sd, sh, sw);
            if let Some(k) = (0..av.len()).find(|&k| av[k] != bv[k]) {
                panic!("c_in={} c_out={} {}x{}x{} k{}{}{} s{}{}{}: differs at {} \
                        (cpu {:?}, gpu {:?}); {} of {} slots differ",
                       c_in, c_out, d, h, w, kd, kh, kw, sd, sh, sw, k, av[k], bv[k],
                       (0..av.len()).filter(|&k| av[k] != bv[k]).count(), av.len());
            }
        }
    }

    #[test]
    fn test_conv2d_batched_gpu_matches_cpu() {
        // The batched CUDA kernel indexes X and Y by b*stride. If those strides
        // were wrong it would still produce a full-size Y -- just with the
        // wrong images in it -- so this compares element for element against
        // the CPU path, for both Y and the Y_full aux.
        almost_goldilocks_cuda::init().expect("CUDA init");
        for (batch, c_in, c_out, sz) in [(2usize, 1usize, 1usize, 4usize),
                                         (4, 2, 3, 8),
                                         (3, 3, 2, 5)] {
            let conv = Conv2D::new(c_in, c_out, 2, 2, sz, sz).with_batch(batch);
            let fk = FlattenKernel {
                s_w: sz.next_power_of_two(), kh: 2, kw: 2, c_out, c_in,
                dilation_h: 1, dilation_w: 1,
            };
            // X must be built in PADDED layout: row stride w_pad, channel
            // stride w_pad*h_pad, image stride x_stride, with the pad slots
            // zero. Filling the buffer densely instead would put values where
            // an honest witness has zeros, and conv_full reads those slots.
            let hp = sz.next_power_of_two();
            let wp = sz.next_power_of_two();
            let mut xd = vec![0u64; conv.b_pad() * conv.x_stride()];
            let mut t = 0u64;
            for b in 0..batch {
                for c in 0..c_in {
                    for ih in 0..sz {
                        for iw in 0..sz {
                            t += 1;
                            xd[b * conv.x_stride() + c * wp * hp + ih * wp + iw] =
                                (t % 11) + 1;
                        }
                    }
                }
            }
            let x = make_witness(vec![batch, c_in, sz, sz], xd);
            let w_raw = make_witness(
                vec![c_out, c_in, 2, 2],
                (0..(c_out * c_in * 4) as u64).map(|v| (v % 5) + 1).collect(),
            );
            let wf_result = fk.run(&[&w_raw]);
            let wf = &wf_result[0];

            let cpu = conv.run(&[&x, wf]);
            let gpu = conv.run_gpu(&[&x, wf]);
            assert_eq!(cpu.len(), gpu.len(), "same number of outputs");
            for (i, (ca, gb)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let av = ca.data.as_ref().unwrap().evaluations();
                let bv = gb.data.as_ref().unwrap().evaluations();
                assert_eq!(
                    av.len(), bv.len(),
                    "batch={} c_in={} c_out={} sz={}: output {} length differs",
                    batch, c_in, c_out, sz, i
                );
                if let Some(k) = (0..av.len()).find(|&k| av[k] != bv[k]) {
                    panic!(
                        "batch={} c_in={} c_out={} sz={}: output {} differs at \
                         index {} (cpu {:?}, gpu {:?}); {} of {} slots differ",
                        batch, c_in, c_out, sz, i, k, av[k], bv[k],
                        (0..av.len()).filter(|&k| av[k] != bv[k]).count(), av.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_conv2d_batched_rejects_wrong_image() {
        // Tamper one value in image 1 only. The batch fold weights every image
        // by eq_b, so a corrupted image must break the argument even though
        // image 0 is untouched -- otherwise the batch index would be free.
        let batch = 2usize;
        let conv = Conv2D::new(1, 1, 2, 2, 4, 4).with_batch(batch);
        let fk = FlattenKernel {
            s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1,
        };
        let mut flat: Vec<u64> = (1..=16u64).collect();
        flat.extend((1..=16u64).map(|v| v * 2));
        let x = make_witness(vec![batch, 1, 4, 4], flat);
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let y = &result[0];
        let yfull = &result[1];

        // Corrupt a real output of image 1 (b=1 starts at y_stride).
        let stride = conv.y_stride();
        let mut ev = y.data.as_ref().unwrap().evaluations();
        ev[stride] = AlmostGoldilocksField(ev[stride].0 + 1);
        let y_bad = Witness::new(
            y.shape.clone(), ev, DataType::Uint, y.sf, Role::Output);

        let n_y = y_bad.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv_batched_tamper");
        let point: Vec<AlmostGoldilocksExt2> =
            (0..n_y).map(|_| transcript.challenge_ext2(b"ch")).collect();
        let eval = y_bad.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut tp = Transcript::new(b"tamper_prove");
        let (proofs, new_claims) =
            conv.prove(&[&x, wf, &y_bad, yfull], &[0, 1, 2, 3], &[&out_claim], &mut tp);
        let mut tv = Transcript::new(b"tamper_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(
            !conv.verify(&[&x, wf, &y_bad, yfull], &all_claims, &proofs_ref, &mut tv),
            "a corrupted image 1 must be rejected"
        );
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
        let yfull = &result[1];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv2");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"test_conv2_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y, yfull],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv2D multichannel prove/verify should pass");
    }

    /// Full prove+verify roundtrip for a Conv2D block over given witnesses.
    /// The out_claim is computed from `y` exactly as a (possibly malicious)
    /// prover would present it.
    fn conv2d_run_prove_verify(
        conv: &Conv2D,
        x: &Witness,
        wf: &Witness,
        y: &Witness,
        yfull: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip_p");
        let (proofs, new_claims) = conv.prove(
            &[x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, wf, y, yfull], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    // ---- Grand-product output-binding mode (ZK4_CONV_GRANDPRODUCT) ----

    /// Build (X, W_flat) for a Conv2D with small (%3) magnitudes and run it.
    /// Assumes grand-product mode is already enabled on this thread, so run
    /// returns `[Y, Y_full, K]`.
    fn make_conv_io_gp(conv: &Conv2D, kh: usize, kw: usize) -> (Witness, Witness, Vec<Witness>) {
        let c_in = conv.c_in;
        let c_out = conv.c_out;
        let h = conv.input_h;
        let w = conv.input_w;
        let w_pad = w.next_power_of_two();
        let h_pad = h.next_power_of_two();
        let c_in_pad = c_in.next_power_of_two();
        let kw_pad = kw.next_power_of_two();
        let kh_pad = kh.next_power_of_two();

        // X in the padded strided MLE layout (iw | ih·w_pad | c·w_pad·h_pad).
        let mut x_data = vec![0u64; c_in * h_pad * w_pad];
        for c in 0..c_in {
            for ih in 0..h {
                for iw in 0..w {
                    x_data[iw + ih * w_pad + c * w_pad * h_pad] = (c + ih + iw) as u64 % 3;
                }
            }
        }
        let x = make_witness(vec![c_in, h, w], x_data);

        // Raw W in the padded strided layout (kw | kh·kw_pad | c·… | d·…).
        let mut w_data = vec![0u64; c_out * c_in_pad * kh_pad * kw_pad];
        for d in 0..c_out {
            for c in 0..c_in {
                for khh in 0..kh {
                    for kww in 0..kw {
                        let idx = kww
                            + khh * kw_pad
                            + c * kw_pad * kh_pad
                            + d * kw_pad * kh_pad * c_in_pad;
                        w_data[idx] = (d + c + khh + kww) as u64 % 3;
                    }
                }
            }
        }
        let w_raw = make_witness(vec![c_out, c_in, kh, kw], w_data);
        let fk = FlattenKernel { s_w: w_pad, kh, kw, c_out, c_in, dilation_h: 1, dilation_w: 1 };
        let wf = fk.run(&[&w_raw]).remove(0);
        let outs = conv.run(&[&x, &wf]);
        (x, wf, outs)
    }

    #[test]
    fn conv2d_gp_mode_run_emits_k() {
        // Stage 1: with grand-product mode on, a non-junk-free Conv2D's run
        // emits a THIRD witness K (leftover Y_full coefficients) whose length
        // is gp_k_len() and whose real entries are exactly the Y_full slots no
        // valid output lands on, in (d outer, m inner) order.
        let _g = GpModeTestGuard::enable();
        let conv = Conv2D::new(2, 3, 3, 3, 5, 5);
        assert!(!conv.junk_free());
        assert!(conv.grand_product_mode());

        let (_x, _wf, outs) = make_conv_io_gp(&conv, 3, 3);
        assert_eq!(outs.len(), 3, "GP mode: run must emit [Y, Y_full, K]");
        let yfull = &outs[1];
        let k = &outs[2];

        let k_len = conv.gp_k_len();
        assert_eq!(k.shape, vec![k_len], "K shape must be [gp_k_len]");
        assert_eq!(k.data.as_ref().unwrap().evaluations().len(), k_len);

        // Reconstruct the expected K (leftover) directly and compare.
        let s_full = conv.s_full();
        let s_full_pad = s_full.next_power_of_two();
        let yf = yfull.data.as_ref().unwrap().evaluations();
        let mut used = vec![false; conv.c_out * s_full_pad];
        for d in 0..conv.c_out {
            for ho in 0..conv.h_out {
                for wo in 0..conv.w_out {
                    used[d * s_full_pad + conv.view_exponent(ho, wo)] = true;
                }
            }
        }
        let mut expected = Vec::new();
        for d in 0..conv.c_out {
            for m in 0..s_full {
                if !used[d * s_full_pad + m] {
                    expected.push(yf[d * s_full_pad + m]);
                }
            }
        }
        let k_ev = k.data.as_ref().unwrap().evaluations();
        for (i, e) in expected.iter().enumerate() {
            assert_eq!(k_ev[i], *e, "K[{i}] mismatch with leftover Y_full coefficient");
        }
        for kv in k_ev.iter().skip(expected.len()) {
            assert_eq!(*kv, AlmostGoldilocksField(0), "K padding must be zero");
        }
    }

    #[test]
    fn conv2d_gp_mode_roundtrip() {
        // Stage 2: gadget-level prove/verify of the grand-product output
        // binding. Honest ⇒ true; a tampered Y opening ⇒ false.
        let _g = GpModeTestGuard::enable();
        let conv = Conv2D::new(2, 3, 3, 3, 5, 5);
        let (x, wf, outs) = make_conv_io_gp(&conv, 3, 3);
        let (y, yfull, k) = (&outs[0], &outs[1], &outs[2]);

        let n_y = y.data.as_ref().unwrap().n();
        let mut t = Transcript::new(b"gp_rt");
        let point: Vec<AlmostGoldilocksExt2> =
            (0..n_y).map(|_| t.challenge_ext2(b"ch")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut pt = Transcript::new(b"gp_rt_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &wf, y, yfull, k],
            &[0, 1, 2, 3, 4],
            &[&out_claim],
            &mut pt,
        );
        assert_eq!(new_claims.len(), 7, "GP prove: 7 produced claims");
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();

        let mut vt = Transcript::new(b"gp_rt_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        assert!(
            conv.verify(&[&x, &wf, y, yfull, k], &all_claims, &proofs_ref, &mut vt),
            "honest grand-product conv must verify"
        );

        // Tampered Y: the fold opens the committed (tampered) Y at y_gp's point
        // to a value disagreeing with the grand-product bottom claim, so the
        // β·v+idx check must reject. Emulate by overriding the y_gp claim eval
        // (produced-claim index 4) with the tampered opening.
        let y_bad = tamper_witness(y, 0);
        let mut tampered = new_claims.clone();
        let y_gp_point = tampered[4].point.clone();
        tampered[4].eval = y_bad.data.as_ref().unwrap().evaluate_at_point_ext2(&y_gp_point);
        let mut tv = Transcript::new(b"gp_rt_p");
        let mut tampered_refs: Vec<&Claim> = tampered.iter().collect();
        tampered_refs.push(&out_claim);
        assert!(
            !conv.verify(&[&x, &wf, y, yfull, k], &tampered_refs, &proofs_ref, &mut tv),
            "tampered Y opening must NOT verify under grand-product binding"
        );
    }

    /// Copy a witness with a single evaluation bumped by +1.
    fn tamper_witness(w: &Witness, idx: usize) -> Witness {
        let mut evals = w.data.as_ref().unwrap().evaluations();
        evals[idx] = AlmostGoldilocksField(evals[idx].0 + 1);
        Witness::new(w.shape.clone(), evals, w.data_type, w.sf, w.role)
    }

    #[test]
    fn test_conv2d_tampered_y_fails() {
        // Soundness (output-binding gap): a malicious prover presents a wrong Y
        // with honest X, W, Y_full. The masked-view sumcheck C must reject —
        // Y(r*) no longer equals Σ eq·mask·view(Y_full).
        let conv = Conv2D::new(1, 1, 2, 2, 4, 4);
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv2d_run_prove_verify(&conv, &x, wf, y, yfull, b"tamper_y_honest"),
            "honest baseline must verify"
        );

        // Tamper a real output position (d=0, ho=0, wo=0).
        let y_bad = tamper_witness(y, 0);
        assert!(
            !conv2d_run_prove_verify(&conv, &x, wf, &y_bad, yfull, b"tamper_y_bad"),
            "tampered Y with honest X/W/Y_full must NOT verify"
        );
    }

    #[test]
    fn test_conv2d_tampered_y_padding_fails() {
        // Garbage in Y's padded region (wo = 3 ≥ w_out = 3). Previously
        // unconstrained; now pinned to zero by the mask factor in sumcheck C.
        let conv = Conv2D::new(1, 1, 2, 2, 4, 4);
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert_eq!(conv.w_out, 3);
        let y_bad = tamper_witness(y, 3); // (d=0, ho=0, wo=3) — padded slot
        assert!(
            !conv2d_run_prove_verify(&conv, &x, wf, &y_bad, yfull, b"tamper_pad"),
            "garbage in Y's padded region must NOT verify"
        );
    }

    #[test]
    fn test_conv2d_tampered_yfull_fails() {
        // Tamper Y_full at a junk exponent (m=0): sumcheck B's implied
        // s_alpha_conv no longer matches Σ F·G, so sumcheck 2 must reject.
        let conv = Conv2D::new(1, 1, 2, 2, 4, 4);
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        let yfull_bad = tamper_witness(yfull, 0); // m=0 is junk for k > 1×1
        assert!(
            !conv2d_run_prove_verify(&conv, &x, wf, y, &yfull_bad, b"tamper_yfull"),
            "tampered Y_full must NOT verify"
        );
    }

    /// Fast-path (junk-free) roundtrip: 3 witnesses, 5 proofs, 4 claims.
    fn conv2d_run_prove_verify_fast(
        conv: &Conv2D,
        x: &Witness,
        wf: &Witness,
        y: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip_fast_p");
        let (proofs, new_claims) = conv.prove(
            &[x, wf, y],
            &[0, 1, 2],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 5, "fast path: 5 proofs (no sumcheck C)");
        assert_eq!(new_claims.len(), 4, "fast path: 4 claims (no Y_full)");

        let mut verify_transcript = Transcript::new(b"roundtrip_fast_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    #[test]
    fn test_conv2d_pointwise_fast_path() {
        // 1×1 stride-1 conv is junk-free: no Y_full aux edge, sumcheck B binds
        // s_alpha directly to the committed Y at the bit-complement point.
        // Non-pow2 3×3 input so Y has a padded region to protect.
        let conv = Conv2D::new(2, 2, 1, 1, 3, 3);
        assert!(conv.junk_free());
        let fk = FlattenKernel { s_w: 4, kh: 1, kw: 1, c_out: 2, c_in: 2, dilation_h: 1, dilation_w: 1 };

        // X[2,3,3] laid out with w_pad = 4 row stride.
        let mut x_data = vec![0u64; 2 * 4 * 4];
        for c in 0..2 {
            for ih in 0..3 {
                for iw in 0..3 {
                    x_data[iw + ih * 4 + c * 16] = (c * 9 + ih * 3 + iw + 1) as u64;
                }
            }
        }
        let x = make_witness(vec![2, 3, 3], x_data);
        let w_raw = make_witness(vec![2, 2, 1, 1], vec![1, 2, 3, 4]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let result = conv.run(&[&x, wf]);
        assert_eq!(result.len(), 1, "junk-free conv must not emit a Y_full aux");
        let y = &result[0];

        assert!(
            conv2d_run_prove_verify_fast(&conv, &x, wf, y, b"pointwise_honest"),
            "honest 1×1 fast path must verify"
        );

        // Tamper a real output position — sumcheck B must reject.
        let y_bad = tamper_witness(y, 0);
        assert!(
            !conv2d_run_prove_verify_fast(&conv, &x, wf, &y_bad, b"pointwise_bad"),
            "tampered 1×1 Y must NOT verify"
        );

        // Tamper Y's padded region (wo = 3 ≥ w_out = 3) — the full-box α-sum
        // binds padding to X's (zero) padding · W, so this must also reject.
        let y_pad_bad = tamper_witness(y, 3);
        assert!(
            !conv2d_run_prove_verify_fast(&conv, &x, wf, &y_pad_bad, b"pointwise_pad"),
            "garbage in 1×1 Y padding must NOT verify"
        );
    }

    #[test]
    fn test_conv2d_pointwise_strided_not_fast_path() {
        // 1×1 kernel with stride 2 is NOT junk-free (unsampled positions are
        // junk) — it must stay on the general Y_full path.
        let conv = Conv2D::new_strided(1, 1, 1, 1, 4, 4, 2, 2);
        assert!(!conv.junk_free());
        let fk = FlattenKernel { s_w: 4, kh: 1, kw: 1, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);
        let w_raw = make_witness(vec![1, 1, 1, 1], vec![3]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        assert_eq!(result.len(), 2, "strided 1×1 keeps the Y_full aux");
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv2d_run_prove_verify(&conv, &x, wf, y, yfull, b"pw_strided_honest"),
            "honest strided 1×1 must verify"
        );
        let y_bad = tamper_witness(y, 0);
        assert!(
            !conv2d_run_prove_verify(&conv, &x, wf, &y_bad, yfull, b"pw_strided_bad"),
            "tampered strided 1×1 Y must NOT verify"
        );
    }

    #[test]
    fn test_conv2d_strided_prove_verify_and_tamper() {
        // Stride-2 conv exercises the bit-affine view σ̃ with shifted bit
        // fields. Honest passes; tampered Y fails.
        let conv = Conv2D::new_strided(1, 1, 2, 2, 4, 4, 2, 2);
        assert_eq!((conv.h_out, conv.w_out), (2, 2));
        let fk = FlattenKernel { s_w: 4, kh: 2, kw: 2, c_out: 1, c_in: 1, dilation_h: 1, dilation_w: 1 };
        let x = make_witness(vec![1, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
        ]);
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 2, 3, 4]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv2d_run_prove_verify(&conv, &x, wf, y, yfull, b"strided_honest"),
            "honest strided Conv2D must verify"
        );

        let y_bad = tamper_witness(y, 1); // real position (ho=0, wo=1)
        assert!(
            !conv2d_run_prove_verify(&conv, &x, wf, &y_bad, yfull, b"strided_bad"),
            "tampered strided Y must NOT verify"
        );
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
        assert_eq!(wf.data.as_ref().unwrap().index(0), AlmostGoldilocksField(1));
        assert_eq!(wf.data.as_ref().unwrap().index(1), AlmostGoldilocksField(2));
        assert_eq!(wf.data.as_ref().unwrap().index(2), AlmostGoldilocksField(3));
        assert_eq!(wf.data.as_ref().unwrap().index(64), AlmostGoldilocksField(4));
        assert_eq!(wf.data.as_ref().unwrap().index(65), AlmostGoldilocksField(5));
        assert_eq!(wf.data.as_ref().unwrap().index(128), AlmostGoldilocksField(7));

        // Create a claim on W_flat
        let n_wf = wf.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_fk_large");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_wf)
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
        let yfull = &result[1];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv_np2");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(conv_proofs.len(), 6);
        assert_eq!(conv_claims.len(), 5);

        // Verify Conv2D
        let mut verify_transcript = Transcript::new(b"test_conv_np2_p");
        let mut all_conv_claims: Vec<&Claim> = conv_claims.iter().collect();
        all_conv_claims.push(&out_claim);
        let conv_proofs_ref: Vec<&SumcheckProof> = conv_proofs.iter().collect();
        let conv_verified = conv.verify(
            &[&x, wf, y, yfull],
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(3));
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(5));
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(7));
    }

    #[test]
    fn test_conv1d_prove_verify() {
        // C_in=1, C_out=1, L=4, K=2
        let conv = Conv1D::new(1, 1, 2, 4);

        let x = make_witness(vec![1, 4], vec![1, 2, 3, 4]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];
        let yfull = &result[1];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv1d");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, &w, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 6);
        assert_eq!(new_claims.len(), 5);

        let mut verify_transcript = Transcript::new(b"test_conv1d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y, yfull],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv1D prove/verify should pass");

        // Y_full claims must open correctly against the actual Y_full MLE.
        for c in &[&new_claims[3], &new_claims[4]] {
            let got = yfull.data.as_ref().unwrap().evaluate_at_point_ext2(&c.point);
            assert_eq!(got, c.eval, "Y_full claim eval must match its MLE");
        }
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
        let yfull = &result[1];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv1d_mc");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, &w, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 6);
        assert_eq!(new_claims.len(), 5);

        let mut verify_transcript = Transcript::new(b"test_conv1d_mc_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y, yfull],
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(6));
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(12));
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(18));
    }

    #[test]
    fn test_conv1d_strided_prove_verify() {
        // C_in=1, C_out=1, L=8, K=3, stride=2
        let conv = Conv1D::new_strided(1, 1, 3, 8, 2);

        let x = make_witness(vec![1, 8], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let w = make_witness(vec![1, 1, 3], vec![1, 1, 1, 0]);

        let result = conv.run(&[&x, &w]);
        let y = &result[0];
        let yfull = &result[1];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv1d_s");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, &w, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 6);
        assert_eq!(new_claims.len(), 5);

        let mut verify_transcript = Transcript::new(b"test_conv1d_s_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &w, y, yfull],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv1D strided prove/verify should pass");
    }

    /// Full prove+verify roundtrip for a Conv1D block over given witnesses.
    /// The out_claim is computed from `y` exactly as a (possibly malicious)
    /// prover would present it.
    fn conv1d_run_prove_verify(
        conv: &Conv1D,
        x: &Witness,
        w: &Witness,
        y: &Witness,
        yfull: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip1d_p");
        let (proofs, new_claims) = conv.prove(
            &[x, w, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip1d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, w, y, yfull], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    #[test]
    fn test_conv1d_tampered_y_fails() {
        // Soundness (output-binding gap): a malicious prover presents a wrong Y
        // with honest X, W, Y_full. The masked-view sumcheck C must reject —
        // Y(r*) no longer equals Σ eq·mask·view(Y_full).
        // input_len=6, K=2 → l_out=5 (padded to 8, so padding exists).
        let conv = Conv1D::new(1, 1, 2, 6);
        assert_eq!(conv.l_out, 5);
        let x = make_witness(vec![1, 6], vec![1, 2, 3, 4, 5, 6, 0, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);
        let result = conv.run(&[&x, &w]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv1d_run_prove_verify(&conv, &x, &w, y, yfull, b"tamper1d_y_honest"),
            "honest baseline must verify"
        );

        // Tamper a real output position (d=0, lo=0).
        let y_bad = tamper_witness(y, 0);
        assert!(
            !conv1d_run_prove_verify(&conv, &x, &w, &y_bad, yfull, b"tamper1d_y_bad"),
            "tampered Y with honest X/W/Y_full must NOT verify"
        );
    }

    #[test]
    fn test_conv1d_tampered_y_padding_fails() {
        // Garbage in Y's padded region (lo = 5 ≥ l_out = 5). Previously
        // unconstrained; now pinned to zero by the mask factor in sumcheck C.
        let conv = Conv1D::new(1, 1, 2, 6);
        let x = make_witness(vec![1, 6], vec![1, 2, 3, 4, 5, 6, 0, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);
        let result = conv.run(&[&x, &w]);
        let (y, yfull) = (&result[0], &result[1]);

        assert_eq!(conv.l_out, 5);
        let y_bad = tamper_witness(y, 5); // (d=0, lo=5) — padded slot
        assert!(
            !conv1d_run_prove_verify(&conv, &x, &w, &y_bad, yfull, b"tamper1d_pad"),
            "garbage in Y's padded region must NOT verify"
        );
    }

    #[test]
    fn test_conv1d_tampered_yfull_fails() {
        // Tamper Y_full at a junk exponent (m=0): sumcheck B's implied
        // s_alpha_conv no longer matches Σ F·G, so sumcheck 2 must reject.
        let conv = Conv1D::new(1, 1, 2, 6);
        let x = make_witness(vec![1, 6], vec![1, 2, 3, 4, 5, 6, 0, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);
        let result = conv.run(&[&x, &w]);
        let (y, yfull) = (&result[0], &result[1]);

        let yfull_bad = tamper_witness(yfull, 0); // m=0 is junk (il=s_in−1 is padding)
        assert!(
            !conv1d_run_prove_verify(&conv, &x, &w, y, &yfull_bad, b"tamper1d_yfull"),
            "tampered Y_full must NOT verify"
        );
    }

    #[test]
    fn test_conv1d_strided_prove_verify_and_tamper() {
        // Stride-2 conv exercises the bit-affine view σ̃ with a shifted lo
        // bit-field. Honest passes; tampered Y fails.
        let conv = Conv1D::new_strided(1, 1, 3, 8, 2);
        assert_eq!(conv.l_out, 3);
        let x = make_witness(vec![1, 8], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let w = make_witness(vec![1, 1, 3], vec![1, 2, 3, 0]);
        let result = conv.run(&[&x, &w]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv1d_run_prove_verify(&conv, &x, &w, y, yfull, b"strided1d_honest"),
            "honest strided Conv1D must verify"
        );

        let y_bad = tamper_witness(y, 1); // real position (d=0, lo=1)
        assert!(
            !conv1d_run_prove_verify(&conv, &x, &w, &y_bad, yfull, b"strided1d_bad"),
            "tampered strided Y must NOT verify"
        );
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
        assert_eq!(wf.data.as_ref().unwrap().index(0), AlmostGoldilocksField(1));
        assert_eq!(wf.data.as_ref().unwrap().index(1), AlmostGoldilocksField(2));
        assert_eq!(wf.data.as_ref().unwrap().index(4), AlmostGoldilocksField(3));
        assert_eq!(wf.data.as_ref().unwrap().index(5), AlmostGoldilocksField(4));
        assert_eq!(wf.data.as_ref().unwrap().index(16), AlmostGoldilocksField(5));
        assert_eq!(wf.data.as_ref().unwrap().index(17), AlmostGoldilocksField(6));
        assert_eq!(wf.data.as_ref().unwrap().index(20), AlmostGoldilocksField(7));
        assert_eq!(wf.data.as_ref().unwrap().index(21), AlmostGoldilocksField(8));
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
        let point: Vec<AlmostGoldilocksExt2> = (0..n_wf)
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(92));
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
        let yfull = &result[1];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_conv3d");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 6);
        assert_eq!(new_claims.len(), 5);

        let mut verify_transcript = Transcript::new(b"test_conv3d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y, yfull],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Conv3D prove/verify should pass");

        // Y_full claims must open correctly against the actual Y_full MLE.
        for c in &[&new_claims[3], &new_claims[4]] {
            let got = yfull.data.as_ref().unwrap().evaluate_at_point_ext2(&c.point);
            assert_eq!(got, c.eval, "Y_full claim eval must match its MLE");
        }
    }

    /// Full prove+verify roundtrip for a Conv3D block over given witnesses.
    /// The out_claim is computed from `y` exactly as a (possibly malicious)
    /// prover would present it.
    fn conv3d_run_prove_verify(
        conv: &Conv3D,
        x: &Witness,
        wf: &Witness,
        y: &Witness,
        yfull: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip3d_p");
        let (proofs, new_claims) = conv.prove(
            &[x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip3d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, wf, y, yfull], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    #[test]
    fn test_conv3d_tampered_y_fails() {
        // Soundness (output-binding gap): a malicious prover presents a wrong Y
        // with honest X, W, Y_full. The masked-view sumcheck C must reject —
        // Y(r*) no longer equals Σ eq·mask·view(Y_full).
        let conv = Conv3D::new(1, 1, 2, 2, 2, 4, 4, 4);
        let fk = FlattenKernel3D {
            stride_h: conv.stride_h,
            stride_w: conv.stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        let x_data: Vec<u64> = (1..=64u64).collect();
        let x = make_witness(vec![1, 4, 4, 4], x_data);
        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 1, 1, 1, 1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv3d_run_prove_verify(&conv, &x, wf, y, yfull, b"tamper3d_y_honest"),
            "honest baseline must verify"
        );

        // Tamper a real output position (d=0, do=0, ho=0, wo=0).
        let y_bad = tamper_witness(y, 0);
        assert!(
            !conv3d_run_prove_verify(&conv, &x, wf, &y_bad, yfull, b"tamper3d_y_bad"),
            "tampered Y with honest X/W/Y_full must NOT verify"
        );

        // Garbage in Y's padded region (wo = 3 ≥ w_out = 3). Previously
        // unconstrained; now pinned to zero by the mask factor in sumcheck C.
        assert_eq!(conv.w_out, 3);
        let y_pad_bad = tamper_witness(y, 3); // (do=0, ho=0, wo=3) — padded slot
        assert!(
            !conv3d_run_prove_verify(&conv, &x, wf, &y_pad_bad, yfull, b"tamper3d_pad"),
            "garbage in Y's padded region must NOT verify"
        );
    }

    #[test]
    fn test_conv3d_tampered_yfull_fails() {
        // Tamper Y_full at a junk exponent (m=0): sumcheck B's implied
        // s_alpha_conv no longer matches Σ F·G, so sumcheck 2 must reject.
        let conv = Conv3D::new(1, 1, 2, 2, 2, 4, 4, 4);
        let fk = FlattenKernel3D {
            stride_h: conv.stride_h,
            stride_w: conv.stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        let x_data: Vec<u64> = (1..=64u64).collect();
        let x = make_witness(vec![1, 4, 4, 4], x_data);
        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 1, 1, 1, 1, 1, 1, 1]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        let yfull_bad = tamper_witness(yfull, 0); // m=0 is outside the viewed region
        assert!(
            !conv3d_run_prove_verify(&conv, &x, wf, y, &yfull_bad, b"tamper3d_yfull"),
            "tampered Y_full must NOT verify"
        );
    }

    #[test]
    fn test_conv3d_strided_prove_verify_and_tamper() {
        // Stride-2 depth conv exercises the bit-affine view σ̃ with a shifted
        // do bit-field. Honest passes; tampered Y fails.
        let conv = Conv3D::new_strided(1, 1, 2, 2, 2, 4, 4, 4, 2, 1, 1);
        assert_eq!((conv.d_out, conv.h_out, conv.w_out), (2, 3, 3));
        let fk = FlattenKernel3D {
            stride_h: conv.stride_h,
            stride_w: conv.stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        let x_data: Vec<u64> = (1..=64u64).collect();
        let x = make_witness(vec![1, 4, 4, 4], x_data);
        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let result = conv.run(&[&x, wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            conv3d_run_prove_verify(&conv, &x, wf, y, yfull, b"strided3d_honest"),
            "honest strided Conv3D must verify"
        );

        let y_bad = tamper_witness(y, 1); // real position (do=0, ho=0, wo=1)
        assert!(
            !conv3d_run_prove_verify(&conv, &x, wf, &y_bad, yfull, b"strided3d_bad"),
            "tampered strided Y must NOT verify"
        );
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(1));
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(1));
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(2));
        assert_eq!(y.data.as_ref().unwrap().index(3), AlmostGoldilocksField(2));
        assert_eq!(y.data.as_ref().unwrap().index(4), AlmostGoldilocksField(3));
        assert_eq!(y.data.as_ref().unwrap().index(5), AlmostGoldilocksField(3));
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
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct1d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, &w, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 5);
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
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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

    /// Full prove+verify roundtrip for a ConvTranspose1D block. The out_claim
    /// is computed from `y` exactly as a (possibly malicious) prover would
    /// present it. No aux Y_full: the direct α-sum binding covers the whole
    /// padded output box.
    fn conv_transpose1d_run_prove_verify(
        conv: &ConvTranspose1D,
        x: &Witness,
        w: &Witness,
        y: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip_ct1d_p");
        let (proofs, new_claims) = conv.prove(
            &[x, w, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip_ct1d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, w, y], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    #[test]
    fn test_conv_transpose1d_tampered_y_fails() {
        // Soundness (output-binding gap): a malicious prover presents a wrong
        // Y with honest X, W. Sumcheck B's implied s_alpha_conv no longer
        // matches Σ F·G, so sumcheck 2 must reject.
        let conv = ConvTranspose1D::new(1, 1, 2, 3, 2);
        let x = make_witness(vec![1, 3], vec![1, 2, 3, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);
        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        assert!(
            conv_transpose1d_run_prove_verify(&conv, &x, &w, y, b"tamper_ct1d_honest"),
            "honest baseline must verify"
        );

        let y_bad = tamper_witness(y, 0); // real output position (d=0, lo=0)
        assert!(
            !conv_transpose1d_run_prove_verify(&conv, &x, &w, &y_bad, b"tamper_ct1d_bad"),
            "tampered Y with honest X/W must NOT verify"
        );
    }

    #[test]
    fn test_conv_transpose1d_tampered_y_padding_fails() {
        // No aux Y_full (degenerate direct binding), so per the protocol the
        // padding is bound by sumcheck B itself: the full polynomial product
        // has zero coefficients at lo ≥ l_out, so garbage in Y's padded
        // region breaks the α-sum identity.
        let conv = ConvTranspose1D::new(1, 1, 2, 3, 2);
        assert_eq!(conv.l_out, 6); // padded to 8 → slots 6, 7 are padding
        let x = make_witness(vec![1, 3], vec![1, 2, 3, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);
        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        let y_bad = tamper_witness(y, 6); // (d=0, lo=6) — padded slot
        assert!(
            !conv_transpose1d_run_prove_verify(&conv, &x, &w, &y_bad, b"tamper_ct1d_pad"),
            "garbage in Y's padded region must NOT verify"
        );
    }

    #[test]
    fn test_conv_transpose1d_stride4_gap_and_non_pow2_stride() {
        // stride=4 > kernel_size=2 leaves unreachable output slots (lo ≡ 2, 3
        // mod 4) that are zero in both Y and the full polynomial product —
        // tampering one must break the α-sum binding. Also checks a non-pow2
        // stride (3): the identity crop map imposes NO pow2-stride
        // requirement, unlike the forward-conv bit-affine view.
        let conv = ConvTranspose1D::new(1, 1, 2, 3, 4);
        assert_eq!(conv.l_out, 10); // (3−1)·4 + 2
        let x = make_witness(vec![1, 3], vec![1, 2, 3, 0]);
        let w = make_witness(vec![1, 1, 2], vec![1, 1]);
        let result = conv.run(&[&x, &w]);
        let y = &result[0];

        assert!(
            conv_transpose1d_run_prove_verify(&conv, &x, &w, y, b"ct1d_s4_honest"),
            "honest stride-4 ConvTranspose1D must verify"
        );
        let y_bad = tamper_witness(y, 2); // unreachable gap slot (lo=2)
        assert!(
            !conv_transpose1d_run_prove_verify(&conv, &x, &w, &y_bad, b"ct1d_s4_bad"),
            "tampered gap slot must NOT verify"
        );

        // Non-pow2 stride roundtrip (stride=3): l_out = (3−1)·3 + 2 = 8.
        let conv3 = ConvTranspose1D::new(1, 1, 2, 3, 3);
        let result3 = conv3.run(&[&x, &w]);
        let y3 = &result3[0];
        assert!(
            conv_transpose1d_run_prove_verify(&conv3, &x, &w, y3, b"ct1d_s3_honest"),
            "honest stride-3 (non-pow2) ConvTranspose1D must verify"
        );
        let y3_bad = tamper_witness(y3, 3); // real output position
        assert!(
            !conv_transpose1d_run_prove_verify(&conv3, &x, &w, &y3_bad, b"ct1d_s3_bad"),
            "tampered stride-3 Y must NOT verify"
        );
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(1)); // (0,0)
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(1)); // (0,1)
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(2)); // (0,2)
        assert_eq!(y.data.as_ref().unwrap().index(3), AlmostGoldilocksField(2)); // (0,3)
        assert_eq!(y.data.as_ref().unwrap().index(4), AlmostGoldilocksField(1)); // (1,0)
        assert_eq!(y.data.as_ref().unwrap().index(8), AlmostGoldilocksField(3)); // (2,0)
        assert_eq!(y.data.as_ref().unwrap().index(10), AlmostGoldilocksField(4)); // (2,2)
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
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct2d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 5);
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

    /// Full prove+verify roundtrip for a ConvTranspose2D block. No aux
    /// Y_full: the direct α-sum binding covers the whole padded output box.
    fn conv_transpose2d_run_prove_verify(
        conv: &ConvTranspose2D,
        x: &Witness,
        wf: &Witness,
        y: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip_ct2d_p");
        let (proofs, new_claims) = conv.prove(
            &[x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip_ct2d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    #[test]
    fn test_conv_transpose2d_tampered_y_fails() {
        // Strided (2,2) honest + tamper: sumcheck B binds Y directly, so a
        // tampered real output breaks the α-sum against Σ F·G at sumcheck 2.
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

        assert!(
            conv_transpose2d_run_prove_verify(&conv, &x, wf, y, b"tamper_ct2d_honest"),
            "honest baseline must verify"
        );
        let y_bad = tamper_witness(y, 0); // real output position (0,0)
        assert!(
            !conv_transpose2d_run_prove_verify(&conv, &x, wf, &y_bad, b"tamper_ct2d_bad"),
            "tampered Y with honest X/W must NOT verify"
        );
    }

    #[test]
    fn test_conv_transpose2d_tampered_y_padding_fails() {
        // No aux Y_full (degenerate direct binding): Y's padded region maps to
        // zero coefficients of the full polynomial product, so garbage there
        // breaks the α-sum identity. input_w=3 → w_out=6, padded to 8.
        let conv = ConvTranspose2D::new(1, 1, 2, 2, 2, 3, 2, 2);
        assert_eq!(conv.h_out, 4);
        assert_eq!(conv.w_out, 6); // w_out_pad = 8 → wo = 6, 7 are padding
        let fk = FlattenKernel {
            s_w: conv.flat_stride, kh: 2, kw: 2, c_out: 1, c_in: 1,
            dilation_h: 1, dilation_w: 1,
        };
        let w_raw = make_witness(vec![1, 1, 2, 2], vec![1, 2, 3, 4]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        // X[1, 2, 3]: w_in padded to 4
        let x = make_witness(vec![1, 2, 3], vec![1, 2, 3, 0, 4, 5, 6, 0]);
        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        assert!(
            conv_transpose2d_run_prove_verify(&conv, &x, wf, y, b"ct2d_pad_honest"),
            "honest non-pow2-width baseline must verify"
        );
        let y_bad = tamper_witness(y, 6); // (oh=0, ow=6) — padded slot
        assert!(
            !conv_transpose2d_run_prove_verify(&conv, &x, wf, &y_bad, b"ct2d_pad_bad"),
            "garbage in Y's padded region must NOT verify"
        );
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(1));
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
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct3d_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 5);
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
    fn test_conv_transpose3d_1x1x1_input() {
        // UNet bottleneck at max depth: 1×1×1 input, k=2, s=2 → 2×2×2 output.
        let conv = ConvTranspose3D::new(1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2);
        let fk = FlattenKernel3D {
            stride_h: conv.flat_stride_h,
            stride_w: conv.flat_stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        let x = make_witness(vec![1, 1, 1, 1], vec![3]);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct3d_1cube");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct3d_1cube_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"test_ct3d_1cube_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose3D with 1x1x1 input should verify");
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
        let mut x_data_gl: Vec<AlmostGoldilocksField> = x_data.iter().map(|&v| AlmostGoldilocksField(v)).collect();
        for c in c_in..c_in_pad {
            for idx in 0..(4*4*4) {
                x_data_gl[idx + c * 4 * 4 * 4] = AlmostGoldilocksField(0);
            }
        }
        let x = Witness::new(vec![c_in, 4, 4, 4], x_data_gl, DataType::Uint, 0, Role::Input);

        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ct3d_npow2");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"test_ct3d_npow2_p");
        let (proofs, new_claims) = conv.prove(
            &[&x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 5);

        let mut verify_transcript = Transcript::new(b"test_ct3d_npow2_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript,
        );
        assert!(verified, "ConvTranspose3D with non-power-of-2 channels should verify");
    }

    /// Full prove+verify roundtrip for a ConvTranspose3D block. No aux
    /// Y_full: the direct α-sum binding covers the whole padded output box.
    fn conv_transpose3d_run_prove_verify(
        conv: &ConvTranspose3D,
        x: &Witness,
        wf: &Witness,
        y: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip_ct3d_p");
        let (proofs, new_claims) = conv.prove(
            &[x, wf, y], &[0, 1, 2], &[&out_claim], &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip_ct3d_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, wf, y], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    #[test]
    fn test_conv_transpose3d_tampered_y_fails() {
        // Strided (2,2,2) honest + tamper: sumcheck B binds Y directly, so a
        // tampered real output breaks the α-sum against Σ F·G at sumcheck 2.
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

        assert!(
            conv_transpose3d_run_prove_verify(&conv, &x, wf, y, b"tamper_ct3d_honest"),
            "honest baseline must verify"
        );
        let y_bad = tamper_witness(y, 0); // real output position (0,0,0)
        assert!(
            !conv_transpose3d_run_prove_verify(&conv, &x, wf, &y_bad, b"tamper_ct3d_bad"),
            "tampered Y with honest X/W must NOT verify"
        );
    }

    #[test]
    fn test_conv_transpose3d_tampered_y_padding_fails() {
        // No aux Y_full (degenerate direct binding): Y's padded region maps to
        // zero coefficients of the full polynomial product, so garbage there
        // breaks the α-sum identity. input_w=3 → w_out=6, padded to 8.
        let conv = ConvTranspose3D::new(1, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2);
        assert_eq!(conv.d_out, 4);
        assert_eq!(conv.h_out, 4);
        assert_eq!(conv.w_out, 6); // w_out_pad = 8 → wo = 6, 7 are padding
        let fk = FlattenKernel3D {
            stride_h: conv.flat_stride_h,
            stride_w: conv.flat_stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 1, c_in: 1,
        };
        let w_raw = make_witness(vec![1, 1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];
        // X[1, 2, 2, 3]: w_in padded to 4
        let x = make_witness(vec![1, 2, 2, 3], vec![
            1, 2, 3, 0, 4, 5, 6, 0,   // jd=0
            7, 8, 9, 0, 10, 11, 12, 0, // jd=1
        ]);
        let result = conv.run(&[&x, wf]);
        let y = &result[0];

        assert!(
            conv_transpose3d_run_prove_verify(&conv, &x, wf, y, b"ct3d_pad_honest"),
            "honest non-pow2-width baseline must verify"
        );
        let y_bad = tamper_witness(y, 6); // (od=0, oh=0, ow=6) — padded slot
        assert!(
            !conv_transpose3d_run_prove_verify(&conv, &x, wf, &y_bad, b"ct3d_pad_bad"),
            "garbage in Y's padded region must NOT verify"
        );
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
        let mut wf2_data = vec![AlmostGoldilocksField(0); c_pad * s_kernel_pad];
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(14));
        // Y[c=0, ho=0, wo=1] = 2+3+6+7 = 18
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(18));
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
        let mut wf2_data = vec![AlmostGoldilocksField(0); c_pad * s_kernel_pad];
        for c in 0..2 {
            for j in 0..s_kernel_pad {
                let src_idx = j + c * 1 * s_kernel_pad;
                let dst_idx = j + c * s_kernel_pad;
                wf2_data[dst_idx] = wf.data.as_ref().unwrap().index(src_idx);
            }
        }
        let wf2 = Witness::new(vec![2, s_kernel], wf2_data, DataType::Uint, 0, Role::Input);

        let result = conv.run(&[&x, &wf2]);
        assert_eq!(result.len(), 2);
        let y = &result[0];
        let yfull = &result[1];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_dw");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, &wf2, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 6);
        assert_eq!(new_claims.len(), 5);

        let mut verify_transcript = Transcript::new(b"test_dw_prove");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = conv.verify(
            &[&x, &wf2, y, yfull],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "DepthwiseConv2D prove/verify should pass");
    }

    /// Full prove+verify roundtrip for a DepthwiseConv2D block over given
    /// witnesses. The out_claim is computed from `y` exactly as a (possibly
    /// malicious) prover would present it.
    fn depthwise_conv2d_run_prove_verify(
        conv: &DepthwiseConv2D,
        x: &Witness,
        wf: &Witness,
        y: &Witness,
        yfull: &Witness,
        label: &'static [u8],
    ) -> bool {
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(label);
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut prove_transcript = Transcript::new(b"roundtrip_dw_p");
        let (proofs, new_claims) = conv.prove(
            &[x, wf, y, yfull], &[0, 1, 2, 3], &[&out_claim], &mut prove_transcript,
        );

        let mut verify_transcript = Transcript::new(b"roundtrip_dw_p");
        let mut all_claims: Vec<&Claim> = new_claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        conv.verify(&[x, wf, y, yfull], &all_claims, &proofs_ref, &mut verify_transcript)
    }

    /// W_flat[C, S_kernel] for a 2×2 depthwise kernel on a 4-wide input:
    /// j = kh·4 + kw, s_kernel = 6, padded to 8.
    fn make_dw_wf2(kernels: &[[u64; 4]]) -> Witness {
        let s_kernel = 6usize;
        let s_kernel_pad = s_kernel.next_power_of_two();
        let c_pad = kernels.len().next_power_of_two();
        let mut data = vec![AlmostGoldilocksField(0); c_pad * s_kernel_pad];
        for (c, k) in kernels.iter().enumerate() {
            data[c * s_kernel_pad] = AlmostGoldilocksField(k[0]);     // (kh=0, kw=0) → j=0
            data[c * s_kernel_pad + 1] = AlmostGoldilocksField(k[1]); // (kh=0, kw=1) → j=1
            data[c * s_kernel_pad + 4] = AlmostGoldilocksField(k[2]); // (kh=1, kw=0) → j=4
            data[c * s_kernel_pad + 5] = AlmostGoldilocksField(k[3]); // (kh=1, kw=1) → j=5
        }
        Witness::new(vec![kernels.len(), s_kernel], data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_depthwise_conv2d_tampered_y_fails() {
        // Soundness (output-binding gap): a malicious prover presents a wrong
        // Y with honest X, W, Y_full. The masked-view sumcheck C must reject —
        // Y(r*) no longer equals Σ eq·mask·view(Y_full).
        let conv = DepthwiseConv2D::new(2, 2, 2, 4, 4);
        let x = make_witness(vec![2, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
            2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,17,
        ]);
        let wf = make_dw_wf2(&[[1, 1, 1, 1], [1, 0, 0, 1]]);
        let result = conv.run(&[&x, &wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            depthwise_conv2d_run_prove_verify(&conv, &x, &wf, y, yfull, b"tamper_dw_honest"),
            "honest baseline must verify"
        );
        let y_bad = tamper_witness(y, 0); // real output position (c=0, ho=0, wo=0)
        assert!(
            !depthwise_conv2d_run_prove_verify(&conv, &x, &wf, &y_bad, yfull, b"tamper_dw_bad"),
            "tampered Y with honest X/W/Y_full must NOT verify"
        );
        // Y's padded region (wo = 3 ≥ w_out = 3) is pinned to zero by the
        // mask factor in sumcheck C.
        let y_pad = tamper_witness(y, 3); // (c=0, ho=0, wo=3) — padded slot
        assert!(
            !depthwise_conv2d_run_prove_verify(&conv, &x, &wf, &y_pad, yfull, b"tamper_dw_pad"),
            "garbage in Y's padded region must NOT verify"
        );
    }

    #[test]
    fn test_depthwise_conv2d_tampered_yfull_fails() {
        // Tamper Y_full at a junk exponent (m=0): sumcheck B's implied
        // s_alpha_conv no longer matches Σ eq_C·F·G, so sumcheck 2 must reject.
        let conv = DepthwiseConv2D::new(2, 2, 2, 4, 4);
        let x = make_witness(vec![2, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
            2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,17,
        ]);
        let wf = make_dw_wf2(&[[1, 1, 1, 1], [1, 0, 0, 1]]);
        let result = conv.run(&[&x, &wf]);
        let (y, yfull) = (&result[0], &result[1]);

        let yfull_bad = tamper_witness(yfull, 0); // m=0 is a junk coefficient
        assert!(
            !depthwise_conv2d_run_prove_verify(&conv, &x, &wf, y, &yfull_bad, b"tamper_dw_yfull"),
            "tampered Y_full must NOT verify"
        );
    }

    #[test]
    fn test_depthwise_conv2d_strided_prove_verify_and_tamper() {
        // Stride-2 depthwise conv exercises the bit-affine view σ̃ with
        // shifted wo/ho bit-fields. Honest passes; tampered Y fails.
        let conv = DepthwiseConv2D::new_strided(2, 2, 2, 4, 4, 2, 2);
        assert_eq!(conv.h_out, 2);
        assert_eq!(conv.w_out, 2);
        let x = make_witness(vec![2, 4, 4], vec![
            1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16,
            2,3,4,5, 6,7,8,9, 10,11,12,13, 14,15,16,17,
        ]);
        let wf = make_dw_wf2(&[[1, 2, 3, 4], [4, 3, 2, 1]]);
        let result = conv.run(&[&x, &wf]);
        let (y, yfull) = (&result[0], &result[1]);

        assert!(
            depthwise_conv2d_run_prove_verify(&conv, &x, &wf, y, yfull, b"strided_dw_honest"),
            "honest strided DepthwiseConv2D must verify"
        );
        let y_bad = tamper_witness(y, 1); // real output position (c=0, ho=0, wo=1)
        assert!(
            !depthwise_conv2d_run_prove_verify(&conv, &x, &wf, &y_bad, yfull, b"strided_dw_bad"),
            "tampered strided Y must NOT verify"
        );
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
        assert_eq!(y.data.as_ref().unwrap().index(0), AlmostGoldilocksField(153));

        // Y[0,1] = X[0,1]+X[0,3]+X[0,5] + X[2,1]+X[2,3]+X[2,5] + X[4,1]+X[4,3]+X[4,5]
        //        = 2+4+6 + 16+18+20 + 30+32+34 = 162
        assert_eq!(y.data.as_ref().unwrap().index(1), AlmostGoldilocksField(162));

        // Y[0,2] = X[0,2]+X[0,4]+X[0,6] + X[2,2]+X[2,4]+X[2,6] + X[4,2]+X[4,4]+X[4,6]
        //        = 3+5+7 + 17+19+21 + 31+33+35 = 171
        assert_eq!(y.data.as_ref().unwrap().index(2), AlmostGoldilocksField(171));

        // Y[1,0] at index 4 (w_pad=4): X[1,0]+X[1,2]+X[1,4]+X[3,0]+X[3,2]+X[3,4]+X[5,0]+X[5,2]+X[5,4]
        //        = 8+10+12 + 22+24+26 + 36+38+40 = 216
        assert_eq!(y.data.as_ref().unwrap().index(4), AlmostGoldilocksField(216));
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
        let yfull = &result[1];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_dilated");
        let point: Vec<AlmostGoldilocksExt2> = (0..n_y)
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
            &[&x, wf, y, yfull],
            &[0, 1, 2, 3],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(conv_proofs.len(), 6);
        assert_eq!(conv_claims.len(), 5);

        // Verify Conv2D
        let mut verify_transcript = Transcript::new(b"test_dilated_p");
        let mut all_conv_claims: Vec<&Claim> = conv_claims.iter().collect();
        all_conv_claims.push(&out_claim);
        let conv_proofs_ref: Vec<&SumcheckProof> = conv_proofs.iter().collect();
        let conv_verified = conv.verify(
            &[&x, wf, y, yfull],
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

    // ================================================================
    // GPU parity tests: run_gpu's device-gathered Y_full must equal
    // run()'s host-scattered Y_full element-wise.
    // ================================================================

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    fn assert_witness_evals_eq(cpu: &Witness, gpu: &Witness, label: &str) {
        let cpu_evals = cpu.data.as_ref().unwrap().evaluations();
        let gpu_evals = gpu.data.as_ref().unwrap().evaluations();
        assert_eq!(cpu_evals.len(), gpu_evals.len(), "{label}: length mismatch");
        for i in 0..cpu_evals.len() {
            assert_eq!(cpu_evals[i].reduce(), gpu_evals[i].reduce(), "{label}: mismatch at {i}");
        }
    }

    #[test]
    fn test_conv2d_run_gpu_yfull_matches_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        // 3×3 kernel, non-pow2 5×5 input (pads to 8×8), stride 2, c_in=2, c_out=3.
        let conv = Conv2D::new_strided(2, 3, 3, 3, 5, 5, 2, 2);
        assert!(!conv.junk_free());

        let w_in_pad = 8usize;
        let h_in_pad = 8usize;
        let mut x_data = vec![0u64; 2 * h_in_pad * w_in_pad];
        for c in 0..2 {
            for ih in 0..5 {
                for iw in 0..5 {
                    x_data[iw + ih * w_in_pad + c * w_in_pad * h_in_pad] =
                        (c * 100 + ih * 5 + iw + 1) as u64;
                }
            }
        }
        let x = make_witness(vec![2, 5, 5], x_data);

        // W[3,2,3,3] in padded little-endian layout [4,2,4,4], flattened via
        // FlattenKernel (s_w = w_in_pad).
        let fk = FlattenKernel { s_w: w_in_pad, kh: 3, kw: 3, c_out: 3, c_in: 2, dilation_h: 1, dilation_w: 1 };
        let mut w_data = vec![0u64; 4 * 2 * 4 * 4];
        for d in 0..3 {
            for c in 0..2 {
                for kh in 0..3 {
                    for kw in 0..3 {
                        w_data[kw + kh * 4 + c * 16 + d * 32] =
                            ((d * 2 + c) * 9 + kh * 3 + kw + 1) as u64;
                    }
                }
            }
        }
        let w_raw = make_witness(vec![3, 2, 3, 3], w_data);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let cpu = conv.run(&[&x, wf]);
        let gpu = conv.run_gpu(&[&x, wf]);
        assert_eq!(cpu.len(), 2);
        assert_eq!(gpu.len(), 2);
        assert_eq!(cpu[1].shape, gpu[1].shape);
        assert_witness_evals_eq(&cpu[0], &gpu[0], "Conv2D Y");
        assert_witness_evals_eq(&cpu[1], &gpu[1], "Conv2D Y_full");
    }

    #[test]
    fn test_conv3d_run_gpu_yfull_matches_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        // 2×2×2 kernel, 4×4×4 input, c_in=2, c_out=2.
        let conv = Conv3D::new(2, 2, 2, 2, 2, 4, 4, 4);

        // X[2,4,4,4]: sequential values.
        let x_data: Vec<u64> = (1..=(2 * 4 * 4 * 4) as u64).collect();
        let x = make_witness(vec![2, 4, 4, 4], x_data);

        let fk = FlattenKernel3D {
            stride_h: conv.stride_h,
            stride_w: conv.stride_w,
            kd: 2, kh: 2, kw: 2, c_out: 2, c_in: 2,
        };
        // W[2,2,2,2,2]: distinct values per tap.
        let w_data: Vec<u64> = (1..=32u64).collect();
        let w_raw = make_witness(vec![2, 2, 2, 2, 2], w_data);
        let wf_result = fk.run(&[&w_raw]);
        let wf = &wf_result[0];

        let cpu = conv.run(&[&x, wf]);
        let gpu = conv.run_gpu(&[&x, wf]);
        assert_eq!(cpu.len(), 2);
        assert_eq!(gpu.len(), 2);
        assert_eq!(cpu[1].shape, gpu[1].shape);
        assert_witness_evals_eq(&cpu[0], &gpu[0], "Conv3D Y");
        assert_witness_evals_eq(&cpu[1], &gpu[1], "Conv3D Y_full");
    }

    #[test]
    fn test_depthwise_conv2d_run_gpu_yfull_matches_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        // 3 channels (non-pow2), 2×2 kernel, 4×4 input, stride 2.
        let conv = DepthwiseConv2D::new_strided(3, 2, 2, 4, 4, 2, 2);

        // X[3,4,4]: distinct values per channel.
        let x_data: Vec<u64> = (0..3 * 16)
            .map(|i| ((i / 16) * 1000 + i % 16 + 1) as u64)
            .collect();
        let x = make_witness(vec![3, 4, 4], x_data);

        // W_flat[3, s_kernel=6] via the same helper the prove/verify tests use.
        let wf2 = make_dw_wf2(&[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);

        let cpu = conv.run(&[&x, &wf2]);
        let gpu = conv.run_gpu(&[&x, &wf2]);
        assert_eq!(cpu.len(), 2);
        assert_eq!(gpu.len(), 2);
        assert_eq!(cpu[1].shape, gpu[1].shape);
        assert_witness_evals_eq(&cpu[0], &gpu[0], "DepthwiseConv2D Y");
        assert_witness_evals_eq(&cpu[1], &gpu[1], "DepthwiseConv2D Y_full");
    }
}
