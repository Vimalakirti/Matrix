//! Raw FFI bindings to the CUDA library.

use std::os::raw::{c_char, c_int, c_void};

#[link(name = "goldilocks_cuda_wrapper")]
extern "C" {
    // ========================================================================
    // Initialization
    // ========================================================================

    pub fn goldilocks_cuda_init() -> c_int;
    pub fn poseidon2_cuda_init() -> c_int;

    // ========================================================================
    // Memory Management
    // ========================================================================

    pub fn cuda_malloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn cuda_free(ptr: *mut c_void) -> c_int;
    pub fn cuda_memcpy_htod(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    pub fn cuda_memcpy_dtoh(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    pub fn cuda_memcpy_dtod(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    pub fn cuda_device_synchronize() -> c_int;
    pub fn cuda_get_last_error() -> c_int;
    pub fn cuda_peek_at_last_error() -> c_int;
    pub fn cuda_mem_get_info(free: *mut usize, total: *mut usize) -> c_int;

    // ========================================================================
    // Goldilocks Field Operations
    // ========================================================================

    pub fn gl_batch_add(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn gl_batch_sub(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn gl_batch_mul(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn gl_batch_inverse(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn gl_einsum2(
        d_a: *const u64,
        d_b: *const u64,
        d_c: *mut u64,
        out_size: c_int,
        sum_size: c_int,
        out_ndim: c_int,
        out_dims: *const c_int,
        out_strides_a: *const c_int,
        out_strides_b: *const c_int,
        sum_ndim: c_int,
        sum_dims: *const c_int,
        sum_strides_a: *const c_int,
        sum_strides_b: *const c_int,
    ) -> c_int;

    pub fn gl_einsum1(
        d_a: *const u64,
        d_c: *mut u64,
        out_size: c_int,
        sum_size: c_int,
        out_ndim: c_int,
        out_dims: *const c_int,
        out_strides_a: *const c_int,
        sum_ndim: c_int,
        sum_dims: *const c_int,
        sum_strides_a: *const c_int,
    ) -> c_int;

    pub fn gl_scale_down(
        d_input: *const u64,
        d_quotients: *mut u64,
        d_bits: *mut u64,
        n: c_int,
        sf: u64,
    ) -> c_int;

    pub fn gl_scale_up(
        d_input: *const u64,
        d_output: *mut u64,
        n: c_int,
        sf: u64,
    ) -> c_int;

    pub fn gl_decompose_bits32(
        d_input: *const u64,
        d_bits: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn gl_memset_zero(d_buf: *mut u64, n_u64: usize) -> c_int;

    pub fn gl_conv2d(
        d_x: *const u64,
        d_w_flat: *const u64,
        d_y: *mut u64,
        c_out: c_int, h_out: c_int, w_out: c_int,
        c_in: c_int, kernel_h: c_int, kernel_w: c_int,
        conv_stride_h: c_int, conv_stride_w: c_int,
        dilation_h: c_int, dilation_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int,
        stride_w_val: c_int,
    ) -> c_int;

    pub fn gl_flatten_kernel2d(
        d_w: *const u64,
        d_w_flat: *mut u64,
        c_out: c_int, c_in: c_int, kh: c_int, kw: c_int,
        kw_pad: c_int, kh_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        dilation_h: c_int, dilation_w: c_int, s_w: c_int,
    ) -> c_int;

    pub fn gl_relu_helper(d_x: *const u64, d_neg: *mut u64, n: c_int) -> c_int;
    pub fn gl_zero_buffer(d_buf: *mut u64, n: c_int) -> c_int;

    pub fn gl_conv3d(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        c_out: c_int, d_out: c_int, h_out: c_int, w_out: c_int,
        c_in: c_int, kernel_d: c_int, kernel_h: c_int, kernel_w: c_int,
        conv_stride_d: c_int, conv_stride_h: c_int, conv_stride_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int, d_in_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int, d_out_pad: c_int,
        stride_h_val: c_int, stride_w_val: c_int,
    ) -> c_int;

    pub fn gl_flatten_kernel3d(
        d_w: *const u64, d_w_flat: *mut u64,
        c_out: c_int, c_in: c_int, kd: c_int, kh: c_int, kw: c_int,
        kw_pad: c_int, kh_pad: c_int, kd_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        stride_h: c_int, stride_w: c_int,
    ) -> c_int;

    pub fn gl_depthwise_conv2d(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        channels: c_int, h_out: c_int, w_out: c_int,
        kernel_h: c_int, kernel_w: c_int,
        conv_stride_h: c_int, conv_stride_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int,
        s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int,
        stride_w_val: c_int,
    ) -> c_int;

    pub fn gl_batch_mul_scalar(
        scalar: u64,
        d_a: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn gl_batch_neg(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn gl_batch_square(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn gl_batch_double(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn gl_batch_exp(d_a: *const u64, exp: u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn gl_batch_div(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    // ========================================================================
    // Extension Field (Ext2) Operations
    // ========================================================================

    pub fn ext2_batch_add_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext2_batch_sub_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext2_batch_mul_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext2_batch_inverse_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext2_batch_mul_scalar_ffi(
        scalar: u64,
        d_a: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext2_batch_neg_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext2_batch_square_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext2_batch_frobenius_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext2_batch_conjugate_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext2_batch_exp_ffi(d_a: *const u64, exp: u64, d_result: *mut u64, n: c_int) -> c_int;

    // ========================================================================
    // Extension Field (Ext5) Operations
    // ========================================================================

    pub fn ext5_batch_add_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext5_batch_sub_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext5_batch_mul_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext5_batch_inverse_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext5_batch_mul_scalar_ffi(
        scalar: u64,
        d_a: *const u64,
        d_result: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn ext5_batch_neg_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext5_batch_square_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext5_batch_frobenius_ffi(d_a: *const u64, d_result: *mut u64, n: c_int) -> c_int;

    pub fn ext5_batch_exp_ffi(d_a: *const u64, exp: u64, d_result: *mut u64, n: c_int) -> c_int;

    // ========================================================================
    // Conversion Operations
    // ========================================================================

    pub fn gl_to_ext2_batch_ffi(d_input: *const u64, d_output: *mut u64, n: c_int) -> c_int;

    pub fn ext2_to_gl_batch_ffi(d_input: *const u64, d_output: *mut u64, n: c_int) -> c_int;

    pub fn gl_to_ext5_batch_ffi(d_input: *const u64, d_output: *mut u64, n: c_int) -> c_int;

    pub fn ext5_to_gl_batch_ffi(d_input: *const u64, d_output: *mut u64, n: c_int) -> c_int;

    // ========================================================================
    // Poseidon2 Operations
    // ========================================================================

    pub fn poseidon2_hash_batch_ffi(d_input: *const u64, d_output: *mut u64, n: c_int) -> c_int;

    pub fn poseidon2_compress_batch_ffi(
        d_left: *const u64,
        d_right: *const u64,
        d_output: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn poseidon2_merkle_layer_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        n: c_int,
    ) -> c_int;

    // ========================================================================
    // Device Info
    // ========================================================================

    pub fn cuda_set_device(device: c_int) -> c_int;
    pub fn cuda_get_device(device: *mut c_int) -> c_int;
    pub fn cuda_get_device_count(count: *mut c_int) -> c_int;
    pub fn cuda_get_device_name(device: c_int, name: *mut c_char, max_len: c_int) -> c_int;

    // ========================================================================
    // Fiat-Shamir Challenger Operations
    // ========================================================================

    pub fn challenger_state_size() -> c_int;
    pub fn challenger_alloc_states(d_states: *mut *mut c_void, n: c_int) -> c_int;
    pub fn challenger_init_states(d_states: *mut c_void, n: c_int) -> c_int;

    pub fn challenger_observe_ffi(
        d_states: *mut c_void,
        d_values: *const u64,
        n: c_int,
    ) -> c_int;

    pub fn challenger_observe_slice_ffi(
        d_states: *mut c_void,
        d_values: *const u64,
        count: c_int,
        n: c_int,
    ) -> c_int;

    pub fn challenger_sample_ffi(
        d_states: *mut c_void,
        d_outputs: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn challenger_sample_array_ffi(
        d_states: *mut c_void,
        d_outputs: *mut u64,
        count: c_int,
        n: c_int,
    ) -> c_int;

    pub fn challenger_sample_ext2_ffi(
        d_states: *mut c_void,
        d_outputs: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn challenger_sample_ext5_ffi(
        d_states: *mut c_void,
        d_outputs: *mut u64,
        n: c_int,
    ) -> c_int;

    pub fn challenger_observe_ext2_ffi(
        d_states: *mut c_void,
        d_values: *const u64,
        n: c_int,
    ) -> c_int;

    pub fn challenger_copy_state_to_host(
        h_state: *mut c_void,
        d_state: *const c_void,
    ) -> c_int;

    pub fn challenger_copy_state_to_device(
        d_state: *mut c_void,
        h_state: *const c_void,
    ) -> c_int;

    // ========================================================================
    // Partial Evaluation Operations
    // ========================================================================

    pub fn partial_eval_gl_ffi(
        d_data: *mut u64,
        d_r: *const u64,
        log_n: c_int,
        m: c_int,
    ) -> c_int;

    pub fn partial_eval_ext2_from_gl_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        d_r: *const u64,
        log_n: c_int,
        m: c_int,
    ) -> c_int;

    // ========================================================================
    // Eq Lagrange Operations
    // ========================================================================

    pub fn eq_dp_all_ffi(
        d_r: *const u64,
        d_buf_a: *mut u64,
        d_buf_b: *mut u64,
        log_n: c_int,
        d_result: *mut *mut u64,
    ) -> c_int;

    pub fn ext2_eq_dp_all_ffi(
        d_r: *const u64,
        d_buf_a: *mut u64,
        d_buf_b: *mut u64,
        log_n: c_int,
        d_result: *mut *mut u64,
    ) -> c_int;

    // ========================================================================
    // Basefold Operations
    // ========================================================================

    pub fn basefold_bit_reverse_gl_ffi(d_data: *mut u64, log_n: c_int) -> c_int;
    pub fn basefold_bit_reverse_ext2_ffi(d_data: *mut u64, log_n: c_int) -> c_int;

    pub fn basefold_bhc_interpolate_ffi(
        d_evals: *const u64,
        d_coeffs: *mut u64,
        d_bh_evals: *mut u64,
        num_vars: c_int,
    ) -> c_int;

    pub fn basefold_encode_ffi(
        d_coeffs: *const u64,
        d_codeword: *mut u64,
        num_vars: c_int,
        log_rate: c_int,
    ) -> c_int;

    pub fn basefold_fold_gl_ffi(
        d_codeword: *const u64,
        d_table: *const u64,
        challenge: u64,
        d_output: *mut u64,
        pair_count: c_int,
    ) -> c_int;

    pub fn basefold_fold_mixed_ffi(
        d_codeword: *const u64,
        d_table: *const u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d_output: *mut u64,
        pair_count: c_int,
    ) -> c_int;

    pub fn basefold_fold_ext2_ffi(
        d_codeword: *const u64,
        d_table: *const u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d_output: *mut u64,
        pair_count: c_int,
    ) -> c_int;

    pub fn basefold_sumcheck_interp_gl_ffi(d_data: *mut u64, pair_count: c_int) -> c_int;
    pub fn basefold_sumcheck_interp_ext2_ffi(d_data: *mut u64, pair_count: c_int) -> c_int;

    pub fn basefold_sumcheck_product_gl_ffi(
        d_eq: *const u64,
        d_bh: *const u64,
        d_partial_c0: *mut u64,
        d_partial_c1: *mut u64,
        d_partial_c2: *mut u64,
        pair_count: c_int,
        num_blocks: c_int,
    ) -> c_int;

    pub fn basefold_sumcheck_product_mixed_ffi(
        d_eq: *const u64,
        d_bh: *const u64,
        d_partial_c0: *mut u64,
        d_partial_c1: *mut u64,
        d_partial_c2: *mut u64,
        pair_count: c_int,
        num_blocks: c_int,
    ) -> c_int;

    pub fn basefold_sumcheck_product_ext2_ffi(
        d_eq: *const u64,
        d_bh: *const u64,
        d_partial_c0: *mut u64,
        d_partial_c1: *mut u64,
        d_partial_c2: *mut u64,
        pair_count: c_int,
        num_blocks: c_int,
    ) -> c_int;

    pub fn basefold_sumcheck_eval_gl_ffi(
        d_data: *const u64,
        challenge: u64,
        d_output: *mut u64,
        pair_count: c_int,
    ) -> c_int;

    pub fn basefold_sumcheck_eval_mixed_ffi(
        d_data: *const u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d_output: *mut u64,
        pair_count: c_int,
    ) -> c_int;

    pub fn basefold_sumcheck_eval_ext2_ffi(
        d_data: *const u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d_output: *mut u64,
        pair_count: c_int,
    ) -> c_int;

    pub fn fused_sumcheck_round_ext2_ffi(
        d_eq_in: *const u64,
        d_bh_in: *const u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d_eq_out: *mut u64,
        d_bh_out: *mut u64,
        d_partial_c0: *mut u64,
        d_partial_c1: *mut u64,
        d_partial_c2: *mut u64,
        pair_count: c_int,
        num_blocks: c_int,
    ) -> c_int;

    pub fn basefold_dot_product_gl_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_partial: *mut u64,
        n: c_int,
        num_blocks: c_int,
    ) -> c_int;

    pub fn basefold_dot_product_mixed_ffi(
        d_a: *const u64,
        d_b: *const u64,
        d_partial: *mut u64,
        n: c_int,
        num_blocks: c_int,
    ) -> c_int;

    // ========================================================================
    // Merkle Tree Operations (GPU-resident)
    // ========================================================================

    pub fn poseidon2_merkle_tree_gl_ffi(
        d_codeword: *const u64,
        d_tree: *mut u64,
        num_leaves: c_int,
    ) -> c_int;

    pub fn poseidon2_merkle_tree_ext2_ffi(
        d_codeword: *const u64,
        d_tree: *mut u64,
        num_leaves: c_int,
    ) -> c_int;

    // ========================================================================
    // Monolith Hash Operations
    // ========================================================================

    pub fn monolith_cuda_init() -> c_int;

    pub fn monolith_merkle_tree_gl_ffi(
        d_codeword: *const u64,
        d_tree: *mut u64,
        num_leaves: c_int,
    ) -> c_int;

    pub fn monolith_merkle_tree_ext2_ffi(
        d_codeword: *const u64,
        d_tree: *mut u64,
        num_leaves: c_int,
    ) -> c_int;

    pub fn monolith_permute_test_ffi(
        h_input: *const u64,
        h_output: *mut u64,
    ) -> c_int;

    // ========================================================================
    // Sumcheck Prover Operations
    // ========================================================================

    pub fn sumcheck_round_message_ffi(
        d_polys: *const u64,
        d_partial: *mut u64,
        d: c_int,
        original_size: usize,
        half: usize,
        num_blocks: c_int,
    ) -> c_int;

    pub fn sumcheck_fold_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        challenge: u64,
        d: c_int,
        original_size: usize,
        half: usize,
    ) -> c_int;

    // ========================================================================
    // Sumcheck Prover Ext2 Operations
    // ========================================================================

    pub fn sumcheck_round_message_ext2_ffi(
        d_polys: *const u64,
        d_partial: *mut u64,
        d: c_int,
        original_size: usize,
        half: usize,
        num_blocks: c_int,
    ) -> c_int;

    pub fn sumcheck_fold_ext2_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d: c_int,
        original_size: usize,
        half: usize,
    ) -> c_int;

    // ========================================================================
    // Ext2 Scale-Accumulate
    // ========================================================================

    pub fn ext2_scale_accumulate_ffi(
        scalar_c0: u64,
        scalar_c1: u64,
        d_src: *const u64,
        d_acc: *mut u64,
        n: c_int,
    ) -> c_int;

    // ========================================================================
    // Bit Permutation
    // ========================================================================

    pub fn bit_permute_gl_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        d_perm_map: *const c_int,
        n_bits: c_int,
        total: c_int,
    ) -> c_int;

    // ========================================================================
    // Fused Permute + Partial Eval
    // ========================================================================

    pub fn fused_permute_partial_eval_ffi(
        d_evals: *const u64,
        d_output: *mut u64,
        d_eq_table: *const u64,
        d_lo_lut: *const u32,
        d_hi_lut: *const u32,
        n: c_int,
        m: c_int,
        half: c_int,
        output_size: c_int,
        smem_bytes: c_int,
    ) -> c_int;
}
