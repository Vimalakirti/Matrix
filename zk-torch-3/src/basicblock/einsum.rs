use goldilocks_cuda::{GoldilocksField, GoldilocksExt2, DeviceBuffer, Ext2Batch};
use goldilocks_cuda::einsum::{einsum1 as gpu_einsum1, einsum2 as gpu_einsum2, EINSUM_MAX_NDIM};
use goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;
use goldilocks_cuda::eq_lagrange::ext2_eq_dp_all_device;
use goldilocks_cuda::partial_eval::{partial_eval_ext2, partial_eval_ext2_device_u64, fused_permute_partial_eval};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use rayon::prelude::*;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, GpuLinearSumcheckProver, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{get_n, gl_add, gl_mul, log2_ceil, ext2_add, ext2_sub, ext2_mul};

/// Threshold: use GPU sumcheck only when total_rounds > this value.
/// Override with ZK_GPU_SUMCHECK_THRESHOLD env var.
fn gpu_sumcheck_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_SUMCHECK_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(14)
    })
}

/// Threshold: use GPU partial evaluation in CPU sumcheck path when n > this value.
/// Override with ZK_GPU_PARTIAL_EVAL_THRESHOLD env var.
fn gpu_partial_eval_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_PARTIAL_EVAL_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
    })
}

/// Threshold: use fused GPU permute + partial eval when n > this value.
/// Override with ZK_GPU_FUSED_THRESHOLD env var.
fn gpu_fused_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_FUSED_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
    })
}

/// Einsum block: generalized tensor contraction.
#[derive(Clone, Debug)]
pub struct Einsum {
    pub equation: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    /// Precomputed bit-range permutation vectors per input (depends only on equation + shapes).
    permute_vecs: Vec<Vec<(usize, usize)>>,
    /// Precomputed number of summation rounds.
    summation_round: usize,
}

impl Einsum {
    pub fn new(equation: &str, input_shapes: Vec<Vec<usize>>, output_shape: Vec<usize>) -> Self {
        let mut all_shapes = input_shapes.clone();
        all_shapes.push(output_shape.clone());
        let (permute_vecs, summation_round) = compute_permute_vecs(equation, &all_shapes);
        Self {
            equation: equation.to_string(),
            input_shapes,
            output_shape,
            permute_vecs,
            summation_round,
        }
    }
}

// ============================================================================
// Einsum helper functions (ported from zk-torch-2)
// ============================================================================

/// Classification of einsum indices.
#[derive(Debug, Clone)]
pub struct EinsumIndexClassification {
    pub free_once: Vec<char>,
    pub free_multi: Vec<char>,
    pub summation: Vec<char>,
}

/// Map from character to bit range (start, end) in the multilinear polynomial.
pub fn char_to_range(symbol: &[char], shape: &[usize]) -> HashMap<char, (usize, usize)> {
    let mut map = HashMap::new();
    let mut start = 0;
    for (i, c) in symbol.iter().enumerate() {
        let bits = log2_ceil(shape[i]);
        map.insert(*c, (start, start + bits));
        start += bits;
    }
    map
}

/// Classify einsum indices into free_once, free_multi, summation.
pub fn classify_einsum_indices_from_shapes(
    subscripts: &str,
) -> (EinsumIndexClassification, Vec<Vec<char>>, Vec<char>) {
    let (lhs, rhs) = subscripts
        .split_once("->")
        .expect("einsum string must contain '->'");

    let input_specs: Vec<Vec<char>> = lhs.split(',').map(|s| s.trim().chars().collect()).collect();
    let out_indices: Vec<char> = rhs.trim().chars().collect();
    let out_set: HashSet<char> = out_indices.iter().copied().collect();

    // Count occurrences across inputs
    let mut input_count: HashMap<char, usize> = HashMap::new();
    for spec in &input_specs {
        for &label in spec {
            *input_count.entry(label).or_insert(0) += 1;
        }
    }

    let mut free_once = Vec::new();
    let mut free_multi = Vec::new();
    for &label in &out_indices {
        let count = input_count.get(&label).copied().unwrap_or(0);
        assert!(count > 0, "output index '{}' does not appear in any input", label);
        if count == 1 {
            free_once.push(label);
        } else {
            free_multi.push(label);
        }
    }

    let mut summation_set: HashSet<char> = HashSet::new();
    let mut summation = Vec::new();
    for spec in &input_specs {
        for &label in spec {
            if !out_set.contains(&label) && !summation_set.contains(&label) {
                summation_set.insert(label);
                summation.push(label);
            }
        }
    }

    (
        EinsumIndexClassification {
            free_once,
            free_multi,
            summation,
        },
        input_specs,
        out_indices,
    )
}

/// Compute permute_vecs and summation_round from equation and shapes.
/// This depends only on the equation and shapes, not on challenge points.
pub fn compute_permute_vecs(
    equation: &str,
    shapes: &[Vec<usize>],
) -> (Vec<Vec<(usize, usize)>>, usize) {
    let (classification, input_specs, _out_indices) =
        classify_einsum_indices_from_shapes(equation);

    let mut all_indices = classification.free_once.clone();
    all_indices.extend(classification.free_multi.clone());
    all_indices.extend(classification.summation.clone());

    let input_num = shapes.len() - 1;
    assert_eq!(input_num, input_specs.len());

    let mut permute_vecs: Vec<Vec<(usize, usize)>> = Vec::with_capacity(input_num);
    let mut summation_set: HashSet<char> = HashSet::new();
    let mut summation_round: usize = 0;

    for i in 0..input_num {
        let shape = shapes[i].clone();
        let spec = input_specs[i].clone();
        let c_to_r = char_to_range(&spec, &shape);

        let mut permute_vec = Vec::new();
        for index in all_indices.iter() {
            if c_to_r.contains_key(index) {
                permute_vec.push(*c_to_r.get(index).unwrap());
            }
        }

        for index in classification.summation.iter() {
            if c_to_r.contains_key(index) && !summation_set.contains(index) {
                summation_set.insert(*index);
                let range = *c_to_r.get(index).unwrap();
                summation_round += range.1 - range.0;
            }
        }

        permute_vecs.push(permute_vec);
    }

    (permute_vecs, summation_round)
}

/// Compute degree-one challenges and high-degree challenge from equation, shapes, and challenge point.
pub fn compute_einsum_challenges(
    equation: &str,
    shapes: &[Vec<usize>],
    challenge_point: &[GoldilocksExt2],
) -> (Vec<Vec<GoldilocksExt2>>, Vec<GoldilocksExt2>) {
    let (classification, input_specs, out_indices) =
        classify_einsum_indices_from_shapes(equation);

    let input_num = shapes.len() - 1;
    let output_c_to_r = char_to_range(&out_indices, &shapes[shapes.len() - 1]);

    let mut degree_one_challenges: Vec<Vec<GoldilocksExt2>> = Vec::with_capacity(input_num);

    for i in 0..input_num {
        let shape = shapes[i].clone();
        let spec = input_specs[i].clone();
        let c_to_r = char_to_range(&spec, &shape);

        let mut partial_challenge = vec![];
        for index in classification.free_once.iter() {
            if c_to_r.contains_key(index) {
                let output_range = *output_c_to_r.get(index).unwrap();
                let ch = challenge_point[output_range.0..output_range.1].to_vec();
                partial_challenge.extend(ch);
            }
        }

        degree_one_challenges.push(partial_challenge);
    }

    let mut high_degree_challenge = vec![];
    for index in classification.free_multi.iter() {
        let output_range = output_c_to_r.get(index).unwrap();
        let ch = challenge_point[output_range.0..output_range.1].to_vec();
        high_degree_challenge.extend(ch);
    }

    (degree_one_challenges, high_degree_challenge)
}

/// Compute the bit permutation map from ranges.
/// perm_map[old_var] = new_var_position
fn compute_perm_map(n: usize, ranges: &[(usize, usize)]) -> Vec<i32> {
    let mut new_var_order = Vec::with_capacity(n);
    let mut seen = vec![false; n];

    for &(start, end) in ranges {
        assert!(start <= end && end <= n);
        for v in start..end {
            assert!(!seen[v], "variable {} appears in multiple ranges", v);
            seen[v] = true;
            new_var_order.push(v);
        }
    }

    assert!(new_var_order.len() == n, "ranges must cover all variables exactly once");

    let mut pos_new = vec![0i32; n];
    for (new_pos, &old_var) in new_var_order.iter().enumerate() {
        pos_new[old_var] = new_pos as i32;
    }
    pos_new
}

/// Check if a permutation is the identity (no reordering needed).
fn is_identity_perm(perm_map: &[i32]) -> bool {
    perm_map.iter().enumerate().all(|(i, &v)| v == i as i32)
}

/// Permute evaluations by ranges (variable reordering).
/// Uses a LUT-based approach for large n to reduce per-element bit ops.
pub fn permute_evals_by_ranges(
    evals: &[GoldilocksField],
    n: usize,
    ranges: &[(usize, usize)],
) -> Vec<GoldilocksField> {
    assert_eq!(evals.len(), 1usize << n);
    assert!(!ranges.is_empty());

    let pos_new = compute_perm_map(n, ranges);

    // Check for identity permutation (skip copy)
    if is_identity_perm(&pos_new) {
        return evals.to_vec();
    }

    let total = evals.len();

    // Compute inverse permutation: inv_perm[new_pos] = old_pos
    let mut inv_perm = vec![0usize; n];
    for old_var in 0..n {
        inv_perm[pos_new[old_var] as usize] = old_var;
    }

    if n <= 16 {
        // For small n, direct bit manipulation gather (sequential writes, random reads)
        let mut out = vec![GoldilocksField(0); total];
        for idx_new in 0..total {
            let mut idx_old = 0usize;
            for new_var in 0..n {
                if idx_new & (1 << new_var) != 0 {
                    idx_old |= 1 << inv_perm[new_var];
                }
            }
            out[idx_new] = evals[idx_old];
        }
        out
    } else {
        // Split into two halves and use LUTs for gather (inverse) permutation
        let half = n / 2;
        let lo_mask = (1usize << half) - 1;
        let lo_size = 1usize << half;
        let hi_size = 1usize << (n - half);

        // Precompute inverse LUTs: lo_lut[new_lo_bits] = old bits from low half
        //                           hi_lut[new_hi_bits] = old bits from high half
        let mut lo_lut = vec![0usize; lo_size];
        for lo_bits in 0..lo_size {
            let mut old_idx = 0usize;
            for bit in 0..half {
                if lo_bits & (1 << bit) != 0 {
                    old_idx |= 1 << inv_perm[bit];
                }
            }
            lo_lut[lo_bits] = old_idx;
        }

        let mut hi_lut = vec![0usize; hi_size];
        for hi_bits in 0..hi_size {
            let mut old_idx = 0usize;
            for bit in 0..(n - half) {
                if hi_bits & (1 << bit) != 0 {
                    old_idx |= 1 << inv_perm[half + bit];
                }
            }
            hi_lut[hi_bits] = old_idx;
        }

        // Parallel gather: sequential writes, random reads (cache-friendly writes)
        const PAR_THRESHOLD: usize = 1 << 18; // 256K elements
        if total >= PAR_THRESHOLD {
            (0..total).into_par_iter().map(|idx_new| {
                let lo = idx_new & lo_mask;
                let hi = idx_new >> half;
                let idx_old = lo_lut[lo] | hi_lut[hi];
                evals[idx_old]
            }).collect()
        } else {
            let mut out = vec![GoldilocksField(0); total];
            for idx_new in 0..total {
                let lo = idx_new & lo_mask;
                let hi = idx_new >> half;
                let idx_old = lo_lut[lo] | hi_lut[hi];
                out[idx_new] = evals[idx_old];
            }
            out
        }
    }
}

/// Invert a point permutation for Ext2 points.
pub fn invert_points_by_ranges(
    y: &[GoldilocksExt2],
    ranges: &[(usize, usize)],
) -> Vec<GoldilocksExt2> {
    let n = y.len();

    let mut new_order = Vec::with_capacity(n);
    for &(start, end) in ranges {
        for v in start..end {
            new_order.push(v);
        }
    }
    assert!(new_order.len() == n);

    let mut old_var_to_newpos = vec![0usize; n];
    for (new_pos, &old_var) in new_order.iter().enumerate() {
        old_var_to_newpos[old_var] = new_pos;
    }

    let mut x = Vec::with_capacity(n);
    for old_var in 0..n {
        let new_pos = old_var_to_newpos[old_var];
        x.push(y[new_pos]);
    }
    x
}

/// Broadcast Ext2 evaluations by doubling.
pub fn broadcast_evals_by_doubling_ext2(
    evals: &[GoldilocksExt2],
    add_dims: usize,
) -> Vec<GoldilocksExt2> {
    let mut out = evals.to_vec();
    for _ in 0..add_dims {
        out.extend_from_within(..);
    }
    out
}

/// Broadcast base-field evaluations by doubling.
pub fn broadcast_evals_by_doubling(
    evals: &[GoldilocksField],
    add_dims: usize,
) -> Vec<GoldilocksField> {
    let mut out = evals.to_vec();
    for _ in 0..add_dims {
        out.extend_from_within(..);
    }
    out
}

/// Einsum helper: compute permutation vectors, degree-one challenges,
/// high-degree challenge, and summation round count.
/// Now uses Ext2 challenge points.
pub fn einsum_helper(
    equation: &str,
    shapes: &[Vec<usize>],
    challenge_point: &[GoldilocksExt2],
) -> (
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<GoldilocksExt2>>,
    Vec<GoldilocksExt2>,
    usize,
) {
    let (permute_vecs, summation_round) = compute_permute_vecs(equation, shapes);
    let (degree_one_challenges, high_degree_challenge) =
        compute_einsum_challenges(equation, shapes, challenge_point);
    (permute_vecs, degree_one_challenges, high_degree_challenge, summation_round)
}

/// Compute einsum output shape from equation and input shapes (ported from zk-torch-2).
pub fn einsum_output_shape(equation: &str, input_shapes: &[Vec<usize>]) -> Vec<usize> {
    let (lhs, rhs) = equation
        .split_once("->")
        .expect("Einsum equation must have ->");

    let input_terms: Vec<&str> = lhs.split(',').collect();
    assert_eq!(
        input_terms.len(),
        input_shapes.len(),
        "Number of inputs does not match number of shapes"
    );

    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (term, shape) in input_terms.iter().zip(input_shapes) {
        let indices: Vec<char> = term.chars().collect();
        assert_eq!(
            indices.len(),
            shape.len(),
            "Rank mismatch: term '{}' vs shape {:?}",
            term,
            shape
        );
        for (&idx, &dim) in indices.iter().zip(shape) {
            dim_map.insert(idx, dim);
        }
    }

    let output_indices: Vec<char> = rhs.chars().collect();
    output_indices
        .iter()
        .map(|&c| *dim_map.get(&c).unwrap_or(&1))
        .collect()
}

/// Threshold below which we prefer the rayon CPU einsum to the naive GPU
/// one-thread-per-output kernel. Tuned for VGG-style `i,ij->j` cases where
/// output is small (≤ ~4096) and the GPU saturates poorly.
fn gpu_einsum_min_outputs() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_EINSUM_MIN_OUTPUTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16384)
    })
}

/// Try to run the einsum on GPU. Returns `None` if the shape is unsupported
/// (3+ inputs or > EINSUM_MAX_NDIM in either output or summation dims) so the
/// caller can fall back to CPU.
fn try_run_gpu(equation: &str, inputs: &[&Witness]) -> Option<Vec<Witness>> {
    if inputs.is_empty() || inputs.len() > 2 {
        return None;
    }

    let (lhs, rhs) = equation.split_once("->")?;
    let input_terms: Vec<&str> = lhs.split(',').collect();
    if input_terms.len() != inputs.len() {
        return None;
    }

    // Map each char to its (next_pow_2-padded) extent across all inputs.
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (term, w) in input_terms.iter().zip(inputs) {
        let chars: Vec<char> = term.chars().collect();
        if chars.len() != w.shape.len() {
            return None;
        }
        for (&c, &dim) in chars.iter().zip(w.shape.iter()) {
            dim_map.insert(c, dim.next_power_of_two());
        }
    }

    let output_chars: Vec<char> = rhs.chars().collect();
    let output_set: HashSet<char> = output_chars.iter().copied().collect();
    let all_chars: HashSet<char> = input_terms.iter().flat_map(|t| t.chars()).collect();
    let sum_chars: Vec<char> = all_chars.difference(&output_set).copied().collect();

    if output_chars.len() > EINSUM_MAX_NDIM || sum_chars.len() > EINSUM_MAX_NDIM {
        return None;
    }

    // Output / sum dim sizes (padded).
    let out_dims_usize: Vec<usize> = output_chars
        .iter()
        .map(|c| *dim_map.get(c).unwrap_or(&1))
        .collect();
    let out_size: usize = out_dims_usize.iter().product::<usize>().max(1);

    // Heuristic fallback: when there's a real summation but the output is
    // small, the naive one-thread-per-output kernel underutilizes the GPU.
    // Let the CPU rayon path handle it.
    let sum_size_check: usize = sum_chars
        .iter()
        .map(|c| *dim_map.get(c).unwrap_or(&1))
        .product::<usize>()
        .max(1);
    if sum_size_check > 1 && out_size < gpu_einsum_min_outputs() {
        return None;
    }
    let out_dims: Vec<i32> = out_dims_usize.iter().map(|&v| v as i32).collect();

    let sum_dims_usize: Vec<usize> = sum_chars
        .iter()
        .map(|c| *dim_map.get(c).unwrap_or(&1))
        .collect();
    let sum_size: usize = sum_dims_usize.iter().product::<usize>().max(1);
    let sum_dims: Vec<i32> = sum_dims_usize.iter().map(|&v| v as i32).collect();

    // Per-input strides for output and sum dims.
    // Within an input's flat buffer the first char has stride 1 (little-endian).
    let stride_for = |term: &str, target: char| -> i32 {
        let chars: Vec<char> = term.chars().collect();
        let mut stride: usize = 1;
        for &c in &chars {
            if c == target {
                return stride as i32;
            }
            stride *= *dim_map.get(&c).unwrap_or(&1);
        }
        0
    };

    let strides_for_input = |term: &str, axes: &[char]| -> Vec<i32> {
        axes.iter().map(|&c| stride_for(term, c)).collect()
    };

    // Build padded output shape (= output dims, in rhs char order, with first
    // char = stride 1). Same data as `out_dims_usize`, semantically the new
    // witness's `shape`.
    let output_shape: Vec<usize> = output_chars
        .iter()
        .map(|c| *dim_map.get(c).unwrap_or(&1))
        .collect();

    // Allocate output device buffer.
    let mut d_out = match DeviceBuffer::<u64>::new(out_size) {
        Ok(b) => b,
        Err(_) => return None,
    };

    let sf_sum = inputs.iter().fold(0usize, |acc, i| acc + i.sf);

    if input_terms.len() == 1 {
        let strides_a = strides_for_input(input_terms[0], &output_chars);
        let sum_strides_a = strides_for_input(input_terms[0], &sum_chars);
        let d_a = inputs[0].as_device_buf();
        if gpu_einsum1(
            &d_a, &mut d_out,
            out_size, sum_size,
            &out_dims, &strides_a,
            &sum_dims, &sum_strides_a,
        ).is_err() {
            return None;
        }
    } else {
        let out_strides_a = strides_for_input(input_terms[0], &output_chars);
        let out_strides_b = strides_for_input(input_terms[1], &output_chars);
        let sum_strides_a = strides_for_input(input_terms[0], &sum_chars);
        let sum_strides_b = strides_for_input(input_terms[1], &sum_chars);

        let d_a = inputs[0].as_device_buf();
        let d_b = inputs[1].as_device_buf();
        if gpu_einsum2(
            &d_a, &d_b, &mut d_out,
            out_size, sum_size,
            &out_dims, &out_strides_a, &out_strides_b,
            &sum_dims, &sum_strides_a, &sum_strides_b,
        ).is_err() {
            return None;
        }
    }

    Some(vec![Witness::new_device(
        output_shape,
        Arc::new(d_out),
        inputs[0].data_type,
        sf_sum,
        Role::Output,
    )])
}

/// Compute einsum output on CPU using direct evaluation.
fn einsum_compute(
    equation: &str,
    inputs: &[&[GoldilocksField]],
    input_shapes: &[Vec<usize>],
) -> Vec<GoldilocksField> {
    let (lhs, rhs) = equation.split_once("->").expect("Einsum equation must have ->");
    let input_terms: Vec<&str> = lhs.split(',').collect();
    let output_indices: Vec<char> = rhs.chars().collect();

    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (term, shape) in input_terms.iter().zip(input_shapes) {
        for (idx, &dim) in term.chars().zip(shape.iter()) {
            dim_map.insert(idx, dim.next_power_of_two());
        }
    }

    let all_input_indices: HashSet<char> = input_terms.iter().flat_map(|t| t.chars()).collect();
    let output_set: HashSet<char> = output_indices.iter().copied().collect();
    let sum_indices: Vec<char> = all_input_indices.difference(&output_set).copied().collect();

    let out_dims: Vec<usize> = output_indices
        .iter()
        .map(|&c| *dim_map.get(&c).unwrap_or(&1))
        .collect();
    let out_size: usize = out_dims.iter().product::<usize>().max(1);

    let sum_dims: Vec<usize> = sum_indices
        .iter()
        .map(|&c| *dim_map.get(&c).unwrap_or(&1))
        .collect();
    let sum_size: usize = sum_dims.iter().product::<usize>().max(1);

    // Pre-compute term chars and padded shapes to avoid repeated allocation
    let term_chars: Vec<Vec<char>> = input_terms.iter().map(|t| t.chars().collect()).collect();
    let padded_shapes: Vec<Vec<usize>> = input_shapes.iter()
        .map(|shape| shape.iter().map(|&s| s.next_power_of_two()).collect())
        .collect();

    // Little-endian indexing: first dimension has stride 1, matching char_to_range
    // and DenseMLPoly::fix_variables conventions.
    let result: Vec<GoldilocksField> = (0..out_size).into_par_iter().map(|out_idx| {
        let mut out_multi = Vec::with_capacity(out_dims.len());
        let mut remainder = out_idx;
        for &d in out_dims.iter() {
            out_multi.push(remainder % d);
            remainder /= d;
        }

        let mut index_map: HashMap<char, usize> = HashMap::new();
        for (i, &c) in output_indices.iter().enumerate() {
            index_map.insert(c, out_multi[i]);
        }

        let mut sum = GoldilocksField(0);
        for sum_idx in 0..sum_size {
            let mut s_remainder = sum_idx;
            for &c in sum_indices.iter() {
                let d = *dim_map.get(&c).unwrap_or(&1);
                index_map.insert(c, s_remainder % d);
                s_remainder /= d;
            }

            let mut product = GoldilocksField(1);
            for (t, input) in inputs.iter().enumerate() {
                let mut linear_idx = 0;
                let mut stride = 1;
                for i in 0..term_chars[t].len() {
                    let c = term_chars[t][i];
                    let idx_val = *index_map.get(&c).unwrap_or(&0) % padded_shapes[t][i];
                    linear_idx += idx_val * stride;
                    stride *= padded_shapes[t][i];
                }

                if linear_idx < input.len() {
                    product = gl_mul(product, input[linear_idx]);
                } else {
                    product = GoldilocksField(0);
                    break;
                }
            }

            sum = gl_add(sum, product);
        }

        sum
    }).collect();

    result
}

/// Broadcast an Ext2 device buffer by doubling `add_dims` times.
/// Input buffer has `n_ext2` Ext2 elements (= `n_ext2 * 2` u64s).
/// Returns a new buffer with `n_ext2 * 2^add_dims` Ext2 elements.
fn broadcast_device_buffer_ext2(
    src: &DeviceBuffer<u64>,
    n_ext2: usize,
    add_dims: usize,
) -> DeviceBuffer<u64> {
    if add_dims == 0 {
        return src.clone_on_device().expect("D2D clone failed");
    }
    let final_ext2 = n_ext2 << add_dims;
    let mut d_out = DeviceBuffer::<u64>::new(final_ext2 * 2).expect("alloc failed");

    // Copy src into the beginning
    let src_bytes = n_ext2 * 2 * std::mem::size_of::<u64>();
    unsafe {
        goldilocks_cuda::memcpy_dtod(
            d_out.as_mut_ptr() as *mut std::os::raw::c_void,
            src.as_ptr() as *const std::os::raw::c_void,
            src_bytes,
        ).expect("D2D copy failed");
    }

    // Double by copying the filled region to the next region
    let mut current_ext2 = n_ext2;
    for _ in 0..add_dims {
        let copy_bytes = current_ext2 * 2 * std::mem::size_of::<u64>();
        unsafe {
            goldilocks_cuda::memcpy_dtod(
                d_out.as_mut_ptr().add(current_ext2 * 2) as *mut std::os::raw::c_void,
                d_out.as_ptr() as *const std::os::raw::c_void,
                copy_bytes,
            ).expect("D2D copy failed");
        }
        current_ext2 *= 2;
    }
    d_out
}

/// CPU partial evaluation of base-field polynomial at Ext2 challenge points.
/// Given poly of size 2^n and m challenge points r[0..m],
/// returns Ext2 poly of size 2^(n-m).
pub fn partial_eval_ext2_cpu(
    evals: &[GoldilocksField],
    challenges: &[GoldilocksExt2],
) -> Vec<GoldilocksExt2> {
    let mut current: Vec<GoldilocksExt2> = evals.iter()
        .map(|&v| GoldilocksExt2::from_base(v))
        .collect();
    for &r in challenges {
        let half = current.len() / 2;
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            let a = current[2 * j];
            let b = current[2 * j + 1];
            // a + r * (b - a)
            next.push(ext2_add(a, ext2_mul(r, ext2_sub(b, a))));
        }
        current = next;
    }
    current
}

impl BasicBlock for Einsum {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let input_refs: Vec<&[GoldilocksField]> = inputs
            .iter()
            .map(|w| w.data.as_ref().unwrap().evaluations_ref())
            .collect();

        // Use actual witness shapes (not stored shapes) for computation and output
        let actual_shapes: Vec<Vec<usize>> = inputs.iter().map(|w| w.shape.clone()).collect();
        let result = einsum_compute(&self.equation, &input_refs, &actual_shapes);
        let output_shape = einsum_output_shape(&self.equation, &actual_shapes);

        let sf_sum = inputs.iter().fold(0usize, |acc, i| acc + i.sf);
        vec![Witness::new(
            output_shape,
            result,
            inputs[0].data_type,
            sf_sum,
            Role::Output,
        )]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Bail out to CPU for shapes the GPU kernel cannot express. Currently
        // supports 1- and 2-input einsums with both output and summation
        // dimension counts up to EINSUM_MAX_NDIM.
        match try_run_gpu(&self.equation, inputs) {
            Some(witnesses) => witnesses,
            None => self.run(inputs),
        }
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert!(out_claims.len() == 1, "Einsum expects 1 output claim");

        let challenge_point = out_claims[0].point.clone();
        let shapes: Vec<Vec<usize>> = witnesses.iter().map(|w| w.shape.clone()).collect();
        let (degree_one_challenges, high_degree_challenge) =
            compute_einsum_challenges(&self.equation, &shapes, &challenge_point);
        let permute_vecs = &self.permute_vecs;
        let num_inputs = permute_vecs.len();

        let num_first_rounds = high_degree_challenge.len();
        let num_second_rounds = self.summation_round;
        let total_rounds = num_first_rounds + num_second_rounds;
        let sumcheck_size = 1usize << total_rounds;

        /// Prepare one input polynomial: fused permute+partial_eval when beneficial,
        /// otherwise falls back to separate permute + partial_eval.
        fn prepare_input_poly(
            witness_evals: &[GoldilocksField],
            n: usize,
            m: usize,
            permute_vec: &[(usize, usize)],
            challenges: &[GoldilocksExt2],
        ) -> Vec<GoldilocksExt2> {
            let perm_map = compute_perm_map(n, permute_vec);
            let needs_permute = !is_identity_perm(&perm_map);

            // Use fused GPU kernel for large polynomials with non-identity permutation
            // Cap at n=28: shared memory overflow for n>=29 (split-LUT exceeds 164KB on A100)
            // and u32 LUT index overflow for n>32
            const FUSED_MAX_N: usize = 28;
            if needs_permute && m > 0 && n > gpu_fused_threshold() && n <= FUSED_MAX_N {
                return fused_permute_partial_eval(witness_evals, challenges, permute_vec, n)
                    .expect("fused_permute_partial_eval failed");
            }

            // Fallback: separate permute + partial_eval
            let permuted_owned;
            let permuted: &[GoldilocksField] = if needs_permute {
                permuted_owned = permute_evals_by_ranges(witness_evals, n, permute_vec);
                &permuted_owned
            } else {
                witness_evals
            };

            if m > 0 {
                if n > gpu_partial_eval_threshold() {
                    partial_eval_ext2(permuted, challenges)
                        .expect("GPU partial_eval_ext2 failed")
                } else {
                    partial_eval_ext2_cpu(permuted, challenges)
                }
            } else {
                permuted.iter().map(|&v| GoldilocksExt2::from_base(v)).collect()
            }
        }

        if total_rounds <= gpu_sumcheck_threshold() {
            // === CPU sumcheck path ===
            let num_polys = num_inputs + 1; // inputs + eq
            let mut cpu_polys: Vec<Vec<GoldilocksExt2>> = Vec::with_capacity(num_polys);

            for i in 0..num_inputs {
                let witness = witnesses[i];
                let n = get_n(&witness.shape);
                let m = degree_one_challenges[i].len();
                let witness_evals_ref = witness.data.as_ref().unwrap().evaluations_ref();

                let ext2_poly = prepare_input_poly(
                    witness_evals_ref, n, m,
                    &permute_vecs[i], &degree_one_challenges[i],
                );

                let result_len = ext2_poly.len();
                let result_n = result_len.trailing_zeros() as usize;
                if result_n < total_rounds {
                    cpu_polys.push(broadcast_evals_by_doubling_ext2(&ext2_poly, total_rounds - result_n));
                } else {
                    cpu_polys.push(ext2_poly);
                }
            }

            // Build eq polynomial on CPU
            if high_degree_challenge.is_empty() {
                let eq = vec![GoldilocksExt2::one(); sumcheck_size];
                cpu_polys.push(eq);
            } else {
                let eq = evaluate_lagrange_basis_ext2(&high_degree_challenge);
                if num_second_rounds > 0 {
                    cpu_polys.push(broadcast_evals_by_doubling_ext2(&eq, num_second_rounds));
                } else {
                    cpu_polys.push(eq);
                }
            }

            let mut cpu_prover = CpuLinearSumcheckProverExt2::new(total_rounds, num_polys, transcript);
            let sumcheck_proof = cpu_prover.prove(&mut cpu_polys, transcript);
            let challenges = cpu_prover.challenges.clone();

            let mut claims = Vec::new();
            for i in 0..num_inputs {
                let n_i = get_n(&witnesses[i].shape);
                if n_i == 0 {
                    let eval = GoldilocksExt2::from_base(witnesses[i].data.as_ref().unwrap().index(0));
                    claims.push(Claim { edge_id: edge_ids[i], sparse_id: 0, point: vec![], eval });
                } else {
                    let point_i_perm: Vec<GoldilocksExt2> = degree_one_challenges[i]
                        .iter().chain(challenges.iter()).copied().collect();
                    claims.push(Claim {
                        edge_id: edge_ids[i], sparse_id: 0,
                        point: invert_points_by_ranges(&point_i_perm, &permute_vecs[i]),
                        eval: cpu_prover.final_eval(i),
                    });
                }
            }

            return (vec![sumcheck_proof], claims);
        }

        // === GPU sumcheck path ===
        let mut gpu_buffers: Vec<DeviceBuffer<u64>> = Vec::with_capacity(num_inputs + 1);
        for i in 0..num_inputs {
            let witness = witnesses[i];
            let n = get_n(&witness.shape);
            let poly_size = 1usize << n;

            let m = degree_one_challenges[i].len();
            let perm_map = compute_perm_map(n, &permute_vecs[i]);
            let needs_permute = !is_identity_perm(&perm_map);
            let witness_evals_ref = witness.data.as_ref().unwrap().evaluations_ref();

            if needs_permute && m > 0 && n > gpu_fused_threshold() {
                // Fused GPU permute + partial eval
                let ext2_poly = fused_permute_partial_eval(
                    witness_evals_ref, &degree_one_challenges[i], &permute_vecs[i], n,
                ).expect("fused_permute_partial_eval failed");

                let result_ext2_count = ext2_poly.len();
                // Upload result to GPU as u64 pairs
                let ext2_u64: Vec<u64> = ext2_poly.iter()
                    .flat_map(|v| [v.c0.0, v.c1.0])
                    .collect();
                let d_result = DeviceBuffer::<u64>::from_slice(&ext2_u64)
                    .expect("GPU upload failed");

                // Broadcast if needed
                let fixed_n = result_ext2_count.trailing_zeros() as usize;
                if fixed_n < total_rounds {
                    let add_dims = total_rounds - fixed_n;
                    let d_broadcast = broadcast_device_buffer_ext2(&d_result, result_ext2_count, add_dims);
                    gpu_buffers.push(d_broadcast);
                } else {
                    gpu_buffers.push(d_result);
                }
            } else {
                // Original path: CPU permute + GPU partial eval
                let permuted_owned;
                let permuted: &[GoldilocksField] = if needs_permute {
                    permuted_owned = permute_evals_by_ranges(witness_evals_ref, n, &permute_vecs[i]);
                    &permuted_owned
                } else {
                    witness_evals_ref
                };
                let permuted_u64: Vec<u64> = permuted.iter().map(|v| v.0).collect();
                let d_permuted = DeviceBuffer::<u64>::from_slice(&permuted_u64)
                    .expect("GPU upload failed");

                if m > 0 {
                    let output_half = poly_size >> 1;
                    let mut d_output = DeviceBuffer::<GoldilocksExt2>::new(output_half)
                        .expect("alloc failed");
                    let d_r = DeviceBuffer::<GoldilocksExt2>::from_slice(&degree_one_challenges[i])
                        .expect("GPU upload failed");

                    partial_eval_ext2_device_u64(&d_permuted, &mut d_output, &d_r, n, m)
                        .expect("partial_eval_ext2_device failed");

                    let result_ext2_count = poly_size >> m;
                    let d_output_u64 = unsafe {
                        let mut d_u64 = DeviceBuffer::<u64>::new(result_ext2_count * 2)
                            .expect("alloc failed");
                        goldilocks_cuda::memcpy_dtod(
                            d_u64.as_mut_ptr() as *mut std::os::raw::c_void,
                            d_output.as_ptr() as *const std::os::raw::c_void,
                            result_ext2_count * 2 * std::mem::size_of::<u64>(),
                        ).expect("D2D copy failed");
                        d_u64
                    };

                    let fixed_n = result_ext2_count.trailing_zeros() as usize;
                    if fixed_n < total_rounds {
                        let add_dims = total_rounds - fixed_n;
                        let d_broadcast = broadcast_device_buffer_ext2(&d_output_u64, result_ext2_count, add_dims);
                        gpu_buffers.push(d_broadcast);
                    } else {
                        gpu_buffers.push(d_output_u64);
                    }
                } else {
                    let mut d_ext2 = DeviceBuffer::<u64>::new(poly_size * 2)
                        .expect("alloc failed");
                    Ext2Batch::from_base(&d_permuted, &mut d_ext2).expect("base→Ext2 failed");

                    let fixed_n = n;
                    if fixed_n < total_rounds {
                        let add_dims = total_rounds - fixed_n;
                        let d_broadcast = broadcast_device_buffer_ext2(&d_ext2, poly_size, add_dims);
                        gpu_buffers.push(d_broadcast);
                    } else {
                        gpu_buffers.push(d_ext2);
                    }
                }
            }
        }

        // Build eq polynomial on GPU
        if high_degree_challenge.is_empty() {
            let one_ext2 = [GoldilocksExt2::one().c0.0, GoldilocksExt2::one().c1.0];
            let d_eq_one = DeviceBuffer::<u64>::from_slice(&one_ext2).expect("upload failed");
            let d_eq = broadcast_device_buffer_ext2(&d_eq_one, 1, total_rounds);
            gpu_buffers.push(d_eq);
        } else {
            let d_r = DeviceBuffer::<GoldilocksExt2>::from_slice(&high_degree_challenge)
                .expect("GPU upload failed");

            let log_eq = high_degree_challenge.len();
            let eq_size = 1usize << log_eq;
            let (d_buf_a, d_buf_b, result_in_a) = ext2_eq_dp_all_device(&d_r, log_eq)
                .expect("ext2_eq_dp_all_device failed");

            let d_eq_result_ext2 = if result_in_a { &d_buf_a } else { &d_buf_b };
            let mut d_eq_u64 = DeviceBuffer::<u64>::new(eq_size * 2).expect("alloc failed");
            unsafe {
                goldilocks_cuda::memcpy_dtod(
                    d_eq_u64.as_mut_ptr() as *mut std::os::raw::c_void,
                    d_eq_result_ext2.as_ptr() as *const std::os::raw::c_void,
                    eq_size * 2 * std::mem::size_of::<u64>(),
                ).expect("D2D copy failed");
            }

            if num_second_rounds > 0 {
                let d_eq_broadcast = broadcast_device_buffer_ext2(&d_eq_u64, eq_size, num_second_rounds);
                gpu_buffers.push(d_eq_broadcast);
            } else {
                gpu_buffers.push(d_eq_u64);
            }
        }

        let buf_refs: Vec<&DeviceBuffer<u64>> = gpu_buffers.iter().collect();
        let gpu_state = GpuSumcheckStateExt2::from_device_buffers(&buf_refs, sumcheck_size)
            .expect("from_device_buffers failed");

        let mut gpu_prover =
            GpuLinearSumcheckProver::new(total_rounds, gpu_buffers.len(), transcript);
        let sumcheck_proof = gpu_prover.prove_gpu_resident(gpu_state, transcript);
        let challenges = gpu_prover.challenges.clone();

        let mut claims = Vec::new();
        for i in 0..num_inputs {
            let n_i = get_n(&witnesses[i].shape);

            if n_i == 0 {
                let eval = GoldilocksExt2::from_base(witnesses[i].data.as_ref().unwrap().index(0));
                claims.push(Claim {
                    edge_id: edge_ids[i],
                    sparse_id: 0,
                    point: vec![],
                    eval,
                });
            } else {
                let point_i_perm: Vec<GoldilocksExt2> = degree_one_challenges[i]
                    .iter()
                    .chain(challenges.iter())
                    .copied()
                    .collect();
                let claim_i = Claim {
                    edge_id: edge_ids[i],
                    sparse_id: 0,
                    point: invert_points_by_ranges(&point_i_perm, &permute_vecs[i]),
                    eval: gpu_prover.final_eval(i),
                };
                claims.push(claim_i);
            }
        }

        (vec![sumcheck_proof], claims)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        let shapes: Vec<Vec<usize>> = witnesses.iter().map(|w| w.shape.clone()).collect();
        let out_claim = claims[claims.len() - 1];
        let (_, _, high_degree_challenge, summation_round) =
            einsum_helper(&self.equation, &shapes, &out_claim.point);
        let num_first_rounds = high_degree_challenge.len();
        let num_second_rounds = summation_round;

        let (verified, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            out_claim.eval,
            num_first_rounds + num_second_rounds,
            claims.len(), // num_poly = num_inputs + eq
            transcript,
        );
        if !verified {
            println!("verified einsum failed: sumcheck round check");
            return false;
        }

        // Final eval check: proof.final_eval == eq_eval * Π_i claims[i].eval
        let one = GoldilocksExt2::one();
        let eq_eval = high_degree_challenge.iter()
            .zip(challenges[..num_first_rounds].iter())
            .fold(one, |acc, (hd_j, r_j)| {
                ext2_mul(acc, ext2_add(
                    ext2_mul(*r_j, *hd_j),
                    ext2_mul(ext2_sub(one, *r_j), ext2_sub(one, *hd_j)),
                ))
            });
        let product_eval = claims[..claims.len() - 1].iter()
            .fold(one, |acc, c| ext2_mul(acc, c.eval));
        let expected = ext2_mul(eq_eval, product_eval);
        if sumcheck_proofs[0].final_eval != expected {
            println!("verified einsum failed: final_eval check mismatch");
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;

    #[test]
    fn test_einsum_elementwise_mul() {
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(2), GoldilocksField(3), GoldilocksField(4), GoldilocksField(5)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b = Witness::new(
            vec![4],
            vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let einsum = Einsum::new("a,a->a", vec![vec![4], vec![4]], vec![4]);
        let result = einsum.run(&[&a, &b]);
        let evals = result[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], GoldilocksField(20));
        assert_eq!(evals[1], GoldilocksField(60));
        assert_eq!(evals[2], GoldilocksField(120));
        assert_eq!(evals[3], GoldilocksField(200));
    }

    #[test]
    fn test_einsum_matmul() {
        let a = Witness::new(
            vec![2, 2],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b = Witness::new(
            vec![2, 2],
            vec![GoldilocksField(5), GoldilocksField(6), GoldilocksField(7), GoldilocksField(8)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let einsum = Einsum::new("ij,jk->ik", vec![vec![2, 2], vec![2, 2]], vec![2, 2]);
        let result = einsum.run(&[&a, &b]);
        let evals = result[0].data.as_ref().unwrap().evaluations_ref();
        // Little-endian: i at bit 0, k at bit 1
        // a: a(0,0)=1, a(1,0)=2, a(0,1)=3, a(1,1)=4
        // b: b(0,0)=5, b(1,0)=6, b(0,1)=7, b(1,1)=8
        // out(i,k) = Σ_j a(i,j)*b(j,k)
        assert_eq!(evals[0], GoldilocksField(23));  // out(0,0) = 1*5+3*6
        assert_eq!(evals[1], GoldilocksField(34));  // out(1,0) = 2*5+4*6
        assert_eq!(evals[2], GoldilocksField(31));  // out(0,1) = 1*7+3*8
        assert_eq!(evals[3], GoldilocksField(46));  // out(1,1) = 2*7+4*8
    }

    #[test]
    fn test_einsum_dot_product() {
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b = Witness::new(
            vec![4],
            vec![GoldilocksField(5), GoldilocksField(6), GoldilocksField(7), GoldilocksField(8)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let einsum = Einsum::new("i,i->", vec![vec![4], vec![4]], vec![]);
        let result = einsum.run(&[&a, &b]);
        let evals = result[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], GoldilocksField(70));
    }
}
