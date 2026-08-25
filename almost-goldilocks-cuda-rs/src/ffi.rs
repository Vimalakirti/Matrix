//! Raw FFI bindings to the C wrapper around the CUDA kernels.
//!
//! Some bindings (e.g. `cuda_set_device`) are not yet wrapped by a safe API;
//! they're kept here for completeness so downstream users can call them.

#![allow(dead_code)]

use std::os::raw::{c_int, c_void};

#[link(name = "almost_goldilocks_cuda_wrapper", kind = "static")]
extern "C" {
    // -------------------- init / memory --------------------
    pub fn almost_goldilocks_cuda_init() -> c_int;

    pub fn cuda_malloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn cuda_free(ptr: *mut c_void) -> c_int;
    pub fn cuda_memcpy_htod(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    pub fn agl_cuda_error_string(code: c_int) -> *const std::os::raw::c_char;

    pub fn cuda_memcpy_dtoh(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    pub fn cuda_memcpy_dtod(dst: *mut c_void, src: *const c_void, size: usize) -> c_int;
    pub fn cuda_memset(dst: *mut c_void, value: c_int, size: usize) -> c_int;
    pub fn cuda_pool_trim(min_bytes_to_keep: usize) -> c_int;
    pub fn cuda_device_synchronize() -> c_int;
    pub fn cuda_get_last_error() -> c_int;
    pub fn cuda_peek_at_last_error() -> c_int;
    pub fn cuda_mem_get_info(free: *mut usize, total: *mut usize) -> c_int;
    pub fn cuda_set_device(device: c_int) -> c_int;
    pub fn cuda_get_device(device: *mut c_int) -> c_int;
    pub fn cuda_get_device_count(count: *mut c_int) -> c_int;
    pub fn cuda_stream_create(stream: *mut *mut c_void) -> c_int;
    pub fn cuda_stream_destroy(stream: *mut c_void) -> c_int;
    pub fn cuda_stream_synchronize(stream: *mut c_void) -> c_int;
    pub fn cuda_memcpy_dtod_async(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void) -> c_int;
    pub fn cuda_memcpy_htod_async(dst: *mut c_void, src: *const c_void, size: usize, stream: *mut c_void) -> c_int;

    // -------------------- base field batch ops --------------------
    pub fn agl_batch_add_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_sub_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_mul_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_neg_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_square_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_double_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_inverse_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_exp_ffi(a: *const u64, exp: u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_mul_scalar_ffi(scalar: u64, a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_batch_div_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn agl_bit_permute_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        d_perm_map: *const c_int,
        n_bits: c_int,
        total: c_int,
    ) -> c_int;

    // -------------------- Ext2 batch ops --------------------
    pub fn aext2_batch_add_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_sub_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_mul_ffi(a: *const u64, b: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_inverse_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_neg_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_square_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_frobenius_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_conjugate_ffi(a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_exp_ffi(a: *const u64, exp: u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_batch_mul_scalar_ffi(scalar: u64, a: *const u64, r: *mut u64, n: c_int) -> c_int;
    pub fn aext2_scale_accumulate_ffi(
        scalar_c0: u64, scalar_c1: u64,
        src: *const u64, acc: *mut u64,
        n: c_int,
    ) -> c_int;
    pub fn agl_to_aext2_batch_ffi(input: *const u64, output: *mut u64, n: c_int) -> c_int;
    pub fn aext2_to_agl_batch_ffi(input: *const u64, output: *mut u64, n: c_int) -> c_int;

    // -------------------- sparse boolean-check sumcheck --------------------
    pub fn agl_bool_init_val_ffi(d_val: *mut u64, n: usize) -> c_int;
    pub fn agl_bool_finish_ffi(
        d_idx: *const u32, d_val: *const u64, d_term_off: *const u32,
        n_terms: c_int, n: usize, d_out: *mut u64,
    ) -> c_int;
    pub fn agl_bool_round_msg_ffi(
        d_idx: *const u32, d_val: *const u64, d_term_off: *const u32,
        d_w: *const u64, d_eq: *const u64, n_terms: c_int, n: usize,
        d_flags: *mut u32, d_gid: *mut u32, d_partial: *mut u64, grid_x: c_int,
        d_scan_scratch: *mut u32, scan_scratch_len: usize,
        h_msg: *mut u64, h_total: *mut u32,
    ) -> c_int;
    pub fn agl_bool_fold_ffi(
        d_idx: *const u32, d_val: *const u64, d_flags: *const u32,
        d_gid: *const u32, d_term_off: *const u32, n_terms: c_int, n: usize,
        r0: u64, r1: u64, total: u32, grid_x: c_int,
        d_out_idx: *mut u32, d_out_val: *mut u64, d_new_off: *mut u32,
    ) -> c_int;

    // -------------------- eq_lagrange (DP) --------------------
    pub fn agl_eq_dp_all_ffi(
        d_r: *const u64,
        d_buf_a: *mut u64,
        d_buf_b: *mut u64,
        log_n: c_int,
        d_result: *mut *mut u64,
    ) -> c_int;

    pub fn aext2_eq_dp_all_ffi(
        d_r: *const u64,
        d_buf_a: *mut u64,
        d_buf_b: *mut u64,
        log_n: c_int,
        d_result: *mut *mut u64,
    ) -> c_int;

    pub fn aext2_eq_dp_all_stream_ffi(
        d_r: *const u64,
        d_buf_a: *mut u64,
        d_buf_b: *mut u64,
        log_n: c_int,
        d_result: *mut *mut u64,
        stream: *mut c_void,
    ) -> c_int;

    pub fn aext2_eq_dp_all_batched_ffi(
        d_r_all: *const u64,
        d_buf_a_all: *mut u64,
        d_buf_b_all: *mut u64,
        log_n: c_int,
        num_leaves: c_int,
        leaf_stride: usize,
        d_result: *mut *mut u64,
        stream: *mut c_void,
    ) -> c_int;

    // -------------------- partial eval --------------------
    pub fn agl_partial_eval_ffi(
        d_data: *mut u64,
        d_r: *const u64,
        log_n: c_int,
        m: c_int,
    ) -> c_int;

    pub fn agl_partial_eval_ext2_from_base_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        d_r: *const u64,
        log_n: c_int,
        m: c_int,
    ) -> c_int;

    // -------------------- fused permute + partial eval --------------------
    pub fn agl_fused_permute_partial_eval_ffi(
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

    // -------------------- sumcheck --------------------
    pub fn agl_sumcheck_round_message_ffi(
        d_polys: *const u64,
        d_partial: *mut u64,
        d: c_int,
        original_size: usize,
        half: usize,
        num_blocks: c_int,
    ) -> c_int;

    pub fn agl_sumcheck_fold_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        challenge: u64,
        d: c_int,
        original_size: usize,
        half: usize,
    ) -> c_int;

    pub fn aext2_sumcheck_round_message_ffi(
        d_polys: *const u64,
        d_partial: *mut u64,
        d: c_int,
        original_size: usize,
        half: usize,
        num_blocks: c_int,
    ) -> c_int;

    pub fn aext2_sumcheck_fold_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        challenge_c0: u64,
        challenge_c1: u64,
        d: c_int,
        original_size: usize,
        half: usize,
    ) -> c_int;

    pub fn aext2_sumcheck_batched_round_message_ffi(
        d_polys: *const u64,
        d_partial: *mut u64,
        original_size: usize,
        half: usize,
        num_leaves: c_int,
        num_blocks_x: c_int,
    ) -> c_int;

    pub fn aext2_batched_lift_ternary_single_ffi(
        d_pos: *const u64,
        d_neg: *const u64,
        d_polys: *mut u64,
        original_size: usize,
        num_leaves: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    pub fn aext2_selective_add_batched_planes_ffi(
        d_eq: *const u64,
        d_packed_planes: *const u64,
        d_partial: *mut u64,
        total: usize,
        n_planes: c_int,
        packed_size_u64: usize,
        num_blocks_x: c_int,
    ) -> c_int;

    pub fn aext2_batched_lift_binary_ffi(
        d_packed: *const u64,
        d_polys: *mut u64,
        original_size: usize,
        num_leaves: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    pub fn aext2_lift_binary_contig_ffi(
        d_packed: *const u64,
        d_f: *mut u64,
        original_size: usize,
        num_leaves: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    pub fn aext2_sumcheck_batched_fold_ffi(
        d_input: *const u64,
        d_output: *mut u64,
        challenge_c0: u64,
        challenge_c1: u64,
        original_size: usize,
        half: usize,
        num_leaves: c_int,
    ) -> c_int;

    pub fn aext2_build_fu_ffi(
        d_packed: *const u64,
        d_alphas: *const u64,
        d_leaf_idx_sorted: *const c_int,
        d_unique_offsets: *const c_int,
        d_Fu: *mut u64,
        original_size: usize,
        num_unique: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    pub fn aext2_build_fu_ternary_ffi(
        d_pos: *const u64,
        d_neg: *const u64,
        d_alphas: *const u64,
        d_leaf_idx_sorted: *const c_int,
        d_unique_offsets: *const c_int,
        d_Fu: *mut u64,
        original_size: usize,
        num_unique: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    pub fn cuda_memcpy_peer(
        dst: *mut c_void,
        dst_dev: c_int,
        src: *const c_void,
        src_dev: c_int,
        size: usize,
    ) -> c_int;

    pub fn cuda_enable_peer_access(peer: c_int) -> c_int;

    pub fn aext2_wide_to_ternary_ffi(
        d_wide: *const i16,
        d_pos: *mut u64,
        d_neg: *mut u64,
        d_err: *mut c_int,
        n_ring: usize,
        k_chunks: c_int,
    ) -> c_int;

    pub fn aext2_eq_suffix_dp_ffi(
        d_r_all: *const u64,
        d_eqsuf: *mut u64,
        log_n: c_int,
        num_unique: c_int,
        eqsuf_stride_u64: usize,
    ) -> c_int;

    pub fn aext2_sharedeq_factored_msg_ffi(
        d_eqsuf: *const u64,
        d_fu: *const u64,
        d_partial: *mut u64,
        eqsuf_off_elems: usize,
        eqsuf_stride_u64: usize,
        poly_stride_u64: usize,
        half: usize,
        num_unique: c_int,
        num_blocks_x: c_int,
    ) -> c_int;

    pub fn aext2_sharedeq_msg_ffi(
        d_eq: *const u64,
        d_f: *const u64,
        d_leaf_to_unique: *const c_int,
        d_partial: *mut u64,
        original_size: usize,
        half: usize,
        num_leaves: c_int,
        num_blocks_x: c_int,
    ) -> c_int;

    pub fn aext2_fold_single_ffi(
        d_in: *const u64,
        d_out: *mut u64,
        challenge_c0: u64,
        challenge_c1: u64,
        original_size: usize,
        half: usize,
        count: c_int,
    ) -> c_int;

    pub fn aext2_sumcheck_batched_round0_binary_msg_ffi(
        d_polys: *const u64,
        d_packed: *const u64,
        d_partial: *mut u64,
        original_size: usize,
        half: usize,
        num_leaves: c_int,
        num_blocks_x: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    pub fn aext2_sumcheck_batched_round0_binary_fold_ffi(
        d_polys: *const u64,
        d_packed: *const u64,
        d_output: *mut u64,
        challenge_c0: u64,
        challenge_c1: u64,
        original_size: usize,
        half: usize,
        num_leaves: c_int,
        packed_size_u64: usize,
    ) -> c_int;

    // -------------------- Ajtai commitment --------------------
    pub fn ajtai_commit_dense_batched_ffi(
        d_key: *const u32,
        d_z: *const u64,
        N: u64,
        B: c_int,
        chunk: c_int,
        d_out: *mut u64,
    ) -> c_int;

    pub fn ajtai_fold_witness_ffi(
        d_z1: *const u64,
        d_z2: *const u64,
        r_coeffs: *const i8,
        N_ring: u64,
        chunk: c_int,
        d_out: *mut u64,
    ) -> c_int;

    pub fn ajtai_multifold_witness_ffi(
        d_z_packed: *const u64,
        d_r_all: *const i8,
        d_out: *mut i16,
        num_instances: c_int,
        N_ring: u64,
        chunk_size: u64,
    ) -> c_int;

    pub fn ajtai_multifold_mixed_witness_ffi(
        d_z_bin_packed: *const u64,
        d_pos_packed: *const u64,
        d_neg_packed: *const u64,
        d_r_all: *const i8,
        d_out: *mut i16,
        num_binary: c_int,
        num_ternary: c_int,
        N_ring: u64,
        chunk_size: u64,
    ) -> c_int;

    pub fn ajtai_multifold_mixed_witness_tc_ffi(
        d_z_bin_packed: *const u64,
        d_pos_packed: *const u64,
        d_neg_packed: *const u64,
        d_r_all: *const i8,
        d_out: *mut i16,
        num_binary: c_int,
        num_ternary: c_int,
        N_ring: u64,
    ) -> c_int;

    pub fn ajtai_multifold_mixed_witness_tc_fused_ffi(
        d_z_bin_packed: *const u64,
        d_pos_packed: *const u64,
        d_neg_packed: *const u64,
        d_r_all: *const i8,
        d_out: *mut i16,
        num_binary: c_int,
        num_ternary: c_int,
        N_ring: u64,
    ) -> c_int;

    pub fn ajtai_split_witness_ffi(
        d_z_wide: *const i16,
        d_pos_chunks: *mut u64,
        d_neg_chunks: *mut u64,
        N_ring: u64,
    ) -> c_int;

    /// Wide (full-width field coefficient) Ajtai commit with a column window.
    /// `d_z_wide` is `[N * 64]` canonical field elements; `d_out` is
    /// `[KAPPA * 64]`. `col_offset` selects the column window of `M_max`.
    /// Dense batched commit against the column window starting at
    /// `col_offset` (in ring elements). `ajtai_commit_dense_batched_ffi` is
    /// the `col_offset = 0` case.
    pub fn ajtai_commit_dense_batched_at_ffi(
        d_key: *const u32,
        d_z: *const u64,
        n: u64,
        b: c_int,
        chunk: c_int,
        col_offset: u64,
        d_out: *mut u64,
    ) -> c_int;

    /// Round-0 message straight off the bit-packed witness.
    pub fn link_round0_bits_ffi(
        d_bits: *const u64, d_omega: *const u64, d_eq: *const u64, d_tags: *const u64,
        stride: u64, half: u64, n_commit: u64, d_partial: *mut u64, chunks: u64,
        d_out: *mut u64,
    ) -> c_int;

    /// Interleaved-layout round message: query weights evaluated on demand.
    pub fn link_round_interleaved_ffi(
        d_bits: *const u64, d_w: *const u64, d_pts: *const u64, d_scale: *const u64,
        d_eq: *const u64, d_tags: *const u64, d_list: *const u32,
        d_list_off: *const u64, d_list_len: *const u64, w_stride: u64,
        bits_stride_words: u64, half: u64, n_commit: u64, block_mask: u32,
        block_bits: c_int, leaf_arity: c_int, round: c_int, first_round: c_int,
        d_partial: *mut u64, chunks: u64, d_out: *mut u64,
    ) -> c_int;

    /// Witness-only fold (no weight table exists in the interleaved layout).
    pub fn link_fold_w_ffi(
        d_bits: *const u64, d_w_in: *const u64, d_w_out: *mut u64,
        in_stride: u64, bits_stride_words: u64, out_stride: u64,
        half: u64, n_commit: u64, first_round: c_int, r_c0: u64, r_c1: u64,
    ) -> c_int;

    /// Round-0 message off bits, driven by the support list.
    pub fn link_round0_bits_sparse_ffi(
        d_bits: *const u64, d_omega: *const u64, d_eq: *const u64, d_tags: *const u64,
        d_list: *const u32, d_list_off: *const u64, d_list_len: *const u64,
        stride: u64, half: u64, n_commit: u64, d_partial: *mut u64, chunks: u64,
        d_out: *mut u64,
    ) -> c_int;

    /// Round-0 fold: bits in, half-size Ext2 out.
    pub fn link_fold0_bits_ffi(
        d_bits: *const u64, d_w_out: *mut u64, d_omega: *mut u64,
        stride: u64, half: u64, n_commit: u64, r_c0: u64, r_c1: u64,
    ) -> c_int;

    /// Expand a bit-packed witness into Ext2 on device.
    pub fn link_expand_bits_ffi(d_bits: *const u64, d_out: *mut u64, n_words: u64) -> c_int;

    /// One expansion level of the batched query-weight construction.
    pub fn link_omega_expand_ffi(
        d_omega: *mut u64, d_bases: *const u64, d_rs: *const u64,
        span: u64, n_active: u64,
    ) -> c_int;

    /// Gather eq(idx, alpha_suffix) at exactly the listed positions.
    pub fn link_eq_gather_ffi(
        d_alpha: *const u64, n_vars: u64, d_list: *const u32,
        d_out: *mut u64, total: u64,
    ) -> c_int;

    /// Round message driven by an explicit per-commitment support list.
    pub fn link_round_message_sparse_ffi(
        d_w: *const u64, d_omega: *const u64, d_eq_suffix: *const u64,
        d_tags: *const u64, d_list: *const u32, d_list_off: *const u64,
        d_list_len: *const u64, stride: u64, omega_stride: u64, half: u64,
        n_commit: u64, first_round: c_int, d_partial: *mut u64, chunks: u64,
        d_out: *mut u64,
    ) -> c_int;

    /// Fold driven by the same support list.
    pub fn link_fold_sparse_ffi(
        d_w: *mut u64, d_omega: *mut u64, d_list: *const u32,
        d_list_off: *const u64, d_list_len: *const u64,
        stride: u64, half: u64, n_commit: u64, r_c0: u64, r_c1: u64,
    ) -> c_int;

    /// One link-sumcheck round message: `[S(0..3), E(0..2)]` as 7 Ext2 values.
    pub fn link_round_message_ffi(
        d_w: *const u64, d_omega: *const u64, d_eq_suffix: *const u64,
        d_tags: *const u64, stride: u64, omega_stride: u64, half: u64, n_commit: u64,
        first_round: c_int, d_partial: *mut u64, chunks: u64, d_out: *mut u64,
    ) -> c_int;

    /// Fold both link tables with the round challenge.
    pub fn link_fold_ffi(
        d_w: *mut u64, d_omega: *mut u64, stride: u64, omega_stride: u64, half: u64,
        n_commit: u64, r_c0: u64, r_c1: u64,
    ) -> c_int;

    /// One variable's expansion of an eq table (doubles the live span).
    pub fn link_eq_expand_ffi(d_table: *mut u64, span: u64, r_c0: u64, r_c1: u64) -> c_int;

    pub fn ajtai_commit_wide_ffi(
        d_key: *const u32,
        d_z_wide: *const u64,
        n: u64,
        col_offset: u64,
        chunk: c_int,
        d_out: *mut u64,
    ) -> c_int;
    pub fn ajtai_commit_ternary_ffi(
        d_key: *const u32,
        d_pos: *const u64,
        d_neg: *const u64,
        N_ring: u64,
        chunk: c_int,
        d_out: *mut u64,
    ) -> c_int;

    pub fn ajtai_materialize_m_ffi(
        d_chacha_key: *const u32,
        d_M: *mut u64,
        N: u64,
    ) -> c_int;

    pub fn ajtai_commit_ternary_premat_ffi(
        d_M: *const u64,
        d_pos: *const u64,
        d_neg: *const u64,
        N_ring: u64,
        chunk: c_int,
        d_out: *mut u64,
    ) -> c_int;

    pub fn ajtai_tc_commit_probe_ffi(
        d_z_int8: *const i8,
        d_M_int8: *const i8,
        d_partial: *mut i32,
        K_total: c_int,
        num_K_chunks: c_int,
    ) -> c_int;

    pub fn ajtai_multifold_commitment_ffi(
        d_c_packed: *const u64,
        d_r_all: *const i8,
        num_instances: c_int,
        d_out: *mut u64,
    ) -> c_int;

    pub fn ajtai_fold_commitment_ffi(
        d_c1: *const u64,
        d_c2: *const u64,
        r_coeffs: *const i8,
        d_out: *mut u64,
    ) -> c_int;

    pub fn ajtai_commit_sparse_ffi(
        d_key: *const u32,
        d_positions: *const u64,
        K: u64,
        chunk: c_int,
        d_out: *mut u64,
    ) -> c_int;

    pub fn agl_einsum2_ffi(
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

    pub fn agl_einsum1_ffi(
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

    pub fn agl_relu_helper_ffi(d_x: *const u64, d_neg: *mut u64, n: c_int) -> c_int;
    pub fn agl_zero_buffer_ffi(d_buf: *mut u64, n: usize) -> c_int;

    pub fn agl_conv2d_ffi(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        c_out: c_int, h_out: c_int, w_out: c_int,
        c_in: c_int, kernel_h: c_int, kernel_w: c_int,
        conv_stride_h: c_int, conv_stride_w: c_int,
        dilation_h: c_int, dilation_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int,
        stride_w_val: c_int,
        batch: c_int, x_stride: c_int, y_stride: c_int,
    ) -> c_int;

    pub fn agl_flatten_kernel2d_ffi(
        d_w: *const u64, d_w_flat: *mut u64,
        c_out: c_int, c_in: c_int, kh: c_int, kw: c_int,
        kw_pad: c_int, kh_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        dilation_h: c_int, dilation_w: c_int, s_w: c_int,
    ) -> c_int;

    pub fn agl_conv3d_ffi(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        c_out: c_int, d_out: c_int, h_out: c_int, w_out: c_int,
        c_in: c_int, kernel_d: c_int, kernel_h: c_int, kernel_w: c_int,
        conv_stride_d: c_int, conv_stride_h: c_int, conv_stride_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int, d_in_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int, d_out_pad: c_int,
        stride_h_val: c_int, stride_w_val: c_int,
        x_len: i64, w_len: i64, y_len: i64,
    ) -> c_int;

    pub fn agl_flatten_kernel3d_ffi(
        d_w: *const u64, d_w_flat: *mut u64,
        c_out: c_int, c_in: c_int, kd: c_int, kh: c_int, kw: c_int,
        kw_pad: c_int, kh_pad: c_int, kd_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        stride_h: c_int, stride_w: c_int,
    ) -> c_int;

    pub fn agl_depthwise_conv2d_ffi(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        channels: c_int, h_out: c_int, w_out: c_int,
        kernel_h: c_int, kernel_w: c_int,
        conv_stride_h: c_int, conv_stride_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int,
        s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int,
        stride_w_val: c_int,
    ) -> c_int;

    pub fn agl_conv_transpose2d_ffi(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        c_out: c_int, h_out: c_int, w_out: c_int,
        c_in: c_int, kernel_h: c_int, kernel_w: c_int,
        stride_h: c_int, stride_w: c_int,
        input_h: c_int, input_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int,
        c_out_pad: c_int, s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int,
        flat_stride: c_int,
    ) -> c_int;

    pub fn agl_conv_transpose3d_ffi(
        d_x: *const u64, d_w_flat: *const u64, d_y: *mut u64,
        c_out: c_int, d_out: c_int, h_out: c_int, w_out: c_int,
        c_in: c_int, kernel_d: c_int, kernel_h: c_int, kernel_w: c_int,
        stride_d: c_int, stride_h: c_int, stride_w: c_int,
        input_d: c_int, input_h: c_int, input_w: c_int,
        w_in_pad: c_int, h_in_pad: c_int, d_in_pad: c_int,
        c_out_pad: c_int, s_kernel_pad: c_int,
        w_out_pad: c_int, h_out_pad: c_int, d_out_pad: c_int,
        flat_stride_h: c_int, flat_stride_w: c_int,
    ) -> c_int;

    pub fn agl_conv_full_ffi(
        d_x: *const u64, d_w_flat: *const u64, d_y_full: *mut u64,
        c_out: c_int, c_in: c_int,
        kernel_d: c_int, kernel_h: c_int, kernel_w: c_int,
        tap_d: c_int, tap_h: c_int, tap_w: c_int,
        s_in: c_int, s_full: c_int, s_full_pad: c_int,
        c_in_pad: c_int, s_kernel_pad: c_int,
        depthwise: c_int,
        batch: c_int, x_stride: c_int, yf_stride: c_int,
        x_len: i64, w_len: i64, y_len: i64,
    ) -> c_int;
}
