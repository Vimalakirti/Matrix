//! [`Einsum`] — generalized tensor contraction.
//!
//! Direct port of the zk-torch-3 einsum. The protocol is a single linear
//! sumcheck over the summation dimensions with an eq factor over the
//! "free-multi" output dimensions. Per-input polys are prepared via
//! permute+partial-eval (fused on GPU when both n and m are large enough).
//!
//! Routes:
//! - `run` / `run_gpu`: forward einsum. GPU path uses `agl_einsum1`/`einsum2`
//!   when output and summation dims fit `EINSUM_MAX_NDIM` and at least one
//!   input is on-device. Falls back to a rayon CPU einsum for the
//!   small-output / many-input case.
//! - `prove`: CPU sumcheck for small `total_rounds`, GPU sumcheck above
//!   the configurable threshold.

use std::collections::{HashMap, HashSet};
use std::os::raw::c_void;
use std::sync::{Arc, OnceLock};

use almost_goldilocks_cuda::einsum::{einsum1 as gpu_einsum1, einsum2 as gpu_einsum2, EINSUM_MAX_NDIM};
use almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all_device;
use almost_goldilocks_cuda::extension::{AlmostExt2Batch, AlmostGoldilocksExt2};
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use almost_goldilocks_cuda::memory::{memcpy_dtod, DeviceBuffer};
use almost_goldilocks_cuda::partial_eval::{
    fused_permute_partial_eval, partial_eval_ext2, partial_eval_ext2_device_u64,
};
use almost_goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;
use rayon::prelude::*;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{
    CpuLinearSumcheckProverExt2, GpuLinearSumcheckProver, SumcheckProof, SumcheckVerifier,
};
use crate::transcript::Transcript;
use crate::util::arith::{agl_add, agl_mul, ext2_add, ext2_field_eq, ext2_mul, ext2_sub, get_n, log2_ceil};

// ============================================================================
// Tuning thresholds (configurable via env vars).
// ============================================================================

fn gpu_sumcheck_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_SUMCHECK_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(14)
    })
}

fn gpu_partial_eval_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_PARTIAL_EVAL_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
    })
}

fn gpu_fused_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_FUSED_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
    })
}

fn gpu_einsum_min_outputs() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_EINSUM_MIN_OUTPUTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16384)
    })
}

// ============================================================================
// Einsum struct
// ============================================================================

#[derive(Clone, Debug)]
pub struct Einsum {
    pub equation: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    permute_vecs: Vec<Vec<(usize, usize)>>,
    summation_round: usize,
}

impl Einsum {
    /// Public cached index schedule (never tensor payloads).
    pub(crate) fn public_schedule(&self) -> (&[Vec<(usize, usize)>], usize) {
        (&self.permute_vecs, self.summation_round)
    }

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
// Einsum helper utilities (ported verbatim from zk-torch-3, field-swapped)
// ============================================================================

#[derive(Debug, Clone)]
pub struct EinsumIndexClassification {
    pub free_once: Vec<char>,
    pub free_multi: Vec<char>,
    pub summation: Vec<char>,
}

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

pub fn classify_einsum_indices_from_shapes(
    subscripts: &str,
) -> (EinsumIndexClassification, Vec<Vec<char>>, Vec<char>) {
    let (lhs, rhs) = subscripts
        .split_once("->")
        .expect("einsum string must contain '->'");
    let input_specs: Vec<Vec<char>> =
        lhs.split(',').map(|s| s.trim().chars().collect()).collect();
    let out_indices: Vec<char> = rhs.trim().chars().collect();
    let out_set: HashSet<char> = out_indices.iter().copied().collect();
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
        assert!(count > 0, "output index '{}' missing from inputs", label);
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
        EinsumIndexClassification { free_once, free_multi, summation },
        input_specs,
        out_indices,
    )
}

pub fn compute_permute_vecs(
    equation: &str,
    shapes: &[Vec<usize>],
) -> (Vec<Vec<(usize, usize)>>, usize) {
    let (classification, input_specs, _out) = classify_einsum_indices_from_shapes(equation);
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
            if let Some(&range) = c_to_r.get(index) {
                permute_vec.push(range);
            }
        }
        for index in classification.summation.iter() {
            if let Some(&range) = c_to_r.get(index) {
                if !summation_set.contains(index) {
                    summation_set.insert(*index);
                    summation_round += range.1 - range.0;
                }
            }
        }
        permute_vecs.push(permute_vec);
    }
    (permute_vecs, summation_round)
}

pub fn compute_einsum_challenges(
    equation: &str,
    shapes: &[Vec<usize>],
    challenge_point: &[AlmostGoldilocksExt2],
) -> (Vec<Vec<AlmostGoldilocksExt2>>, Vec<AlmostGoldilocksExt2>) {
    let (classification, input_specs, out_indices) = classify_einsum_indices_from_shapes(equation);
    let input_num = shapes.len() - 1;
    let output_c_to_r = char_to_range(&out_indices, &shapes[shapes.len() - 1]);
    let mut degree_one_challenges: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(input_num);
    for i in 0..input_num {
        let shape = shapes[i].clone();
        let spec = input_specs[i].clone();
        let c_to_r = char_to_range(&spec, &shape);
        let mut partial = vec![];
        for index in classification.free_once.iter() {
            if c_to_r.contains_key(index) {
                let r = *output_c_to_r.get(index).unwrap();
                partial.extend_from_slice(&challenge_point[r.0..r.1]);
            }
        }
        degree_one_challenges.push(partial);
    }
    let mut high_degree_challenge = vec![];
    for index in classification.free_multi.iter() {
        let r = output_c_to_r.get(index).unwrap();
        high_degree_challenge.extend_from_slice(&challenge_point[r.0..r.1]);
    }
    (degree_one_challenges, high_degree_challenge)
}

fn compute_perm_map(n: usize, ranges: &[(usize, usize)]) -> Vec<i32> {
    let mut new_var_order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for &(start, end) in ranges {
        assert!(start <= end && end <= n);
        for v in start..end {
            assert!(!seen[v], "var {} duplicated in ranges", v);
            seen[v] = true;
            new_var_order.push(v);
        }
    }
    assert_eq!(new_var_order.len(), n, "ranges must cover all vars exactly once");
    let mut pos_new = vec![0i32; n];
    for (new_pos, &old_var) in new_var_order.iter().enumerate() {
        pos_new[old_var] = new_pos as i32;
    }
    pos_new
}

fn is_identity_perm(perm_map: &[i32]) -> bool {
    perm_map.iter().enumerate().all(|(i, &v)| v == i as i32)
}

/// Device copies of einsum operands that are model weights (`Role::Constant`):
/// identical across proofs, but the GPU einsum prover re-uploaded them every
/// proof (the 2^26-word GPT-2 embedding table cost ~100 ms per proof in a
/// host copy + pageable upload). Keyed by (device, host data pointer, length);
/// a strided sample of 64 words is re-checked on every hit, so a buffer whose
/// contents changed is never served (it is re-uploaded instead). Capped per
/// device by ZK4_EINSUM_WCACHE_GB (default 16); allocation failure just skips
/// caching. ZK4_EINSUM_WCACHE=0 disables.
struct WCacheEntry { sample: u64, buf: Arc<DeviceBuffer<u64>> }
static WCACHE: OnceLock<std::sync::Mutex<(HashMap<(i32, usize, usize), WCacheEntry>, HashMap<i32, usize>)>> =
    OnceLock::new();

fn evals_as_u64(evals: &[AlmostGoldilocksField]) -> &[u64] {
    // SAFETY: AlmostGoldilocksField is #[repr(transparent)] over u64.
    unsafe { std::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len()) }
}

fn sample_words(w: &[u64]) -> u64 {
    if w.is_empty() { return 0; }
    let mut h: u64 = 0x9E37_79B9_7F4A_7C15 ^ (w.len() as u64);
    let step = (w.len() / 64).max(1);
    let mut k = 0usize;
    while k < w.len() { h = (h ^ w[k]).wrapping_mul(0x100_0000_01B3).rotate_left(17); k += step; }
    h ^ w[w.len() - 1]
}

/// Device buffer holding `evals` on the current device: cached when the
/// witness is a weight, else a fresh upload (no intermediate host copy).
fn device_operand(witness: &Witness, evals: &[AlmostGoldilocksField]) -> Arc<DeviceBuffer<u64>> {
    let words = evals_as_u64(evals);
    let on = std::env::var("ZK4_EINSUM_WCACHE").ok().as_deref() != Some("0");
    if std::env::var("ZK4_EINSUM_TIMING").is_ok() && words.len() >= (1 << 24) {
        eprintln!("[wcache] words={} role={:?} ptr={:p}", words.len(), witness.role, words.as_ptr());
    }
    if !(on && witness.role == Role::Constant) {
        return Arc::new(DeviceBuffer::<u64>::from_slice(words).expect("upload"));
    }
    let dev = almost_goldilocks_cuda::current_device();
    let key = (dev, words.as_ptr() as usize, words.len());
    let sample = sample_words(words);
    let cache = WCACHE.get_or_init(|| std::sync::Mutex::new((HashMap::new(), HashMap::new())));
    if let Some(e) = cache.lock().unwrap().0.get(&key) {
        if e.sample == sample { return e.buf.clone(); }
    }
    let buf = Arc::new(DeviceBuffer::<u64>::from_slice(words).expect("upload"));
    let cap = std::env::var("ZK4_EINSUM_WCACHE_GB").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(16.0);
    let bytes = words.len() * 8;
    let mut g = cache.lock().unwrap();
    let used = *g.1.get(&dev).unwrap_or(&0);
    let replacing = g.0.get(&key).map(|e| e.buf.len() * 8).unwrap_or(0);
    if (used - replacing + bytes) as f64 <= cap * 1e9 {
        g.0.insert(key, WCacheEntry { sample, buf: buf.clone() });
        *g.1.entry(dev).or_insert(0) = used - replacing + bytes;
    }
    buf
}

pub fn permute_evals_by_ranges(
    evals: &[AlmostGoldilocksField],
    n: usize,
    ranges: &[(usize, usize)],
) -> Vec<AlmostGoldilocksField> {
    assert_eq!(evals.len(), 1usize << n);
    assert!(!ranges.is_empty());
    let pos_new = compute_perm_map(n, ranges);
    if is_identity_perm(&pos_new) {
        return evals.to_vec();
    }
    let total = evals.len();
    let mut inv_perm = vec![0usize; n];
    for old_var in 0..n {
        inv_perm[pos_new[old_var] as usize] = old_var;
    }

    if n <= 16 {
        let mut out = vec![AlmostGoldilocksField(0); total];
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
        let half = n / 2;
        let lo_mask = (1usize << half) - 1;
        let lo_size = 1usize << half;
        let hi_size = 1usize << (n - half);
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
        const PAR_THRESHOLD: usize = 1 << 18;
        if total >= PAR_THRESHOLD {
            (0..total)
                .into_par_iter()
                .map(|idx_new| {
                    let lo = idx_new & lo_mask;
                    let hi = idx_new >> half;
                    evals[lo_lut[lo] | hi_lut[hi]]
                })
                .collect()
        } else {
            let mut out = vec![AlmostGoldilocksField(0); total];
            for idx_new in 0..total {
                let lo = idx_new & lo_mask;
                let hi = idx_new >> half;
                out[idx_new] = evals[lo_lut[lo] | hi_lut[hi]];
            }
            out
        }
    }
}

pub fn invert_points_by_ranges(
    y: &[AlmostGoldilocksExt2],
    ranges: &[(usize, usize)],
) -> Vec<AlmostGoldilocksExt2> {
    let n = y.len();
    let mut new_order = Vec::with_capacity(n);
    for &(start, end) in ranges {
        for v in start..end {
            new_order.push(v);
        }
    }
    assert_eq!(new_order.len(), n);
    let mut old_var_to_newpos = vec![0usize; n];
    for (new_pos, &old_var) in new_order.iter().enumerate() {
        old_var_to_newpos[old_var] = new_pos;
    }
    (0..n).map(|old_var| y[old_var_to_newpos[old_var]]).collect()
}

pub fn broadcast_evals_by_doubling_ext2(
    evals: &[AlmostGoldilocksExt2],
    add_dims: usize,
) -> Vec<AlmostGoldilocksExt2> {
    let mut out = evals.to_vec();
    for _ in 0..add_dims {
        out.extend_from_within(..);
    }
    out
}

pub fn broadcast_evals_by_doubling(
    evals: &[AlmostGoldilocksField],
    add_dims: usize,
) -> Vec<AlmostGoldilocksField> {
    let mut out = evals.to_vec();
    for _ in 0..add_dims {
        out.extend_from_within(..);
    }
    out
}

pub fn einsum_helper(
    equation: &str,
    shapes: &[Vec<usize>],
    challenge_point: &[AlmostGoldilocksExt2],
) -> (
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<AlmostGoldilocksExt2>>,
    Vec<AlmostGoldilocksExt2>,
    usize,
) {
    let (permute_vecs, summation_round) = compute_permute_vecs(equation, shapes);
    let (degree_one_challenges, high_degree_challenge) =
        compute_einsum_challenges(equation, shapes, challenge_point);
    (permute_vecs, degree_one_challenges, high_degree_challenge, summation_round)
}

pub fn einsum_output_shape(equation: &str, input_shapes: &[Vec<usize>]) -> Vec<usize> {
    let (lhs, rhs) = equation.split_once("->").expect("Einsum equation must have ->");
    let input_terms: Vec<&str> = lhs.split(',').collect();
    assert_eq!(input_terms.len(), input_shapes.len(), "input/term count mismatch");
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (term, shape) in input_terms.iter().zip(input_shapes) {
        let indices: Vec<char> = term.chars().collect();
        assert_eq!(indices.len(), shape.len(), "rank mismatch on term '{}'", term);
        for (&idx, &dim) in indices.iter().zip(shape) {
            dim_map.insert(idx, dim);
        }
    }
    rhs.chars().map(|c| *dim_map.get(&c).unwrap_or(&1)).collect()
}

/// CPU partial-eval of base-field polynomial at Ext2 challenges. Returns
/// the resulting Ext2 polynomial.
pub fn partial_eval_ext2_cpu(
    evals: &[AlmostGoldilocksField],
    challenges: &[AlmostGoldilocksExt2],
) -> Vec<AlmostGoldilocksExt2> {
    let mut current: Vec<AlmostGoldilocksExt2> =
        evals.iter().map(|&v| AlmostGoldilocksExt2::from_base(v)).collect();
    for &r in challenges {
        let half = current.len() / 2;
        let mut next = Vec::with_capacity(half);
        for j in 0..half {
            let a = current[2 * j];
            let b = current[2 * j + 1];
            next.push(ext2_add(a, ext2_mul(r, ext2_sub(b, a))));
        }
        current = next;
    }
    current
}

// ============================================================================
// GPU forward einsum
// ============================================================================

fn try_run_gpu(equation: &str, inputs: &[&Witness]) -> Option<Vec<Witness>> {
    if inputs.is_empty() || inputs.len() > 2 {
        return None;
    }
    let (lhs, rhs) = equation.split_once("->")?;
    let input_terms: Vec<&str> = lhs.split(',').collect();
    if input_terms.len() != inputs.len() {
        return None;
    }
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
    let out_dims_usize: Vec<usize> = output_chars.iter().map(|c| *dim_map.get(c).unwrap_or(&1)).collect();
    let out_size: usize = out_dims_usize.iter().product::<usize>().max(1);
    let sum_size_check: usize = sum_chars
        .iter()
        .map(|c| *dim_map.get(c).unwrap_or(&1))
        .product::<usize>()
        .max(1);
    if sum_size_check > 1 && out_size < gpu_einsum_min_outputs() {
        return None;
    }
    let out_dims: Vec<i32> = out_dims_usize.iter().map(|&v| v as i32).collect();
    let sum_dims_usize: Vec<usize> =
        sum_chars.iter().map(|c| *dim_map.get(c).unwrap_or(&1)).collect();
    let sum_size: usize = sum_dims_usize.iter().product::<usize>().max(1);
    let sum_dims: Vec<i32> = sum_dims_usize.iter().map(|&v| v as i32).collect();

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

    // `dim_map` holds POWER-OF-TWO PADDED extents, so deriving the output
    // shape from it yields e.g. [1,32,2048] where `run` yields [1,32,1536].
    // The arity is identical, so the mismatch is invisible to a padding check,
    // but downstream ops read logical extents (Add's broadcast matching, claim
    // point construction), and a padded shape makes them disagree with the
    // rest of the graph. Derive the shape exactly as `run` does.
    let actual_shapes: Vec<Vec<usize>> = inputs.iter().map(|w| w.shape.clone()).collect();
    let output_shape: Vec<usize> = einsum_output_shape(equation, &actual_shapes);
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
            &d_a, &mut d_out, out_size, sum_size,
            &out_dims, &strides_a, &sum_dims, &sum_strides_a,
        )
        .is_err()
        {
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
            &d_a, &d_b, &mut d_out, out_size, sum_size,
            &out_dims, &out_strides_a, &out_strides_b,
            &sum_dims, &sum_strides_a, &sum_strides_b,
        )
        .is_err()
        {
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

// ============================================================================
// CPU forward einsum
// ============================================================================

/// Opt-in fast CPU forward for two-input contractions over exactly one
/// summation index (the matmuls, attention products, embedding lookup and LM
/// head): each operand is repacked with the summation axis contiguous, sparse
/// rows (one-hot selectors) are contracted over their nonzeros only. Equal
/// field VALUES to the generic path (canonical words; the generic path's words
/// may be non-canonical), so it is off by default and on for WEIGHTS=real
/// (where the Jacobi generation runs ~100 forward passes) or
/// ZK4_FAST_EINSUM=1; ZK4_FAST_EINSUM=0 forces the generic path.
fn fast_einsum_enabled() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| match std::env::var("ZK4_FAST_EINSUM").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => std::env::var("WEIGHTS").ok().as_deref() == Some("real"),
    })
}

fn einsum_compute_fast(
    equation: &str,
    inputs: &[&[AlmostGoldilocksField]],
    input_shapes: &[Vec<usize>],
) -> Option<Vec<AlmostGoldilocksField>> {
    if inputs.len() != 2 { return None; }
    let (lhs, rhs) = equation.split_once("->")?;
    let terms: Vec<Vec<char>> = lhs.split(',').map(|t| t.chars().collect()).collect();
    let out: Vec<char> = rhs.chars().collect();
    for (t, sh) in terms.iter().zip(input_shapes) {
        if t.len() != sh.len() { return None; }
        let mut seen = HashSet::new();
        if !t.iter().all(|c| seen.insert(*c)) { return None; }
    }
    let mut dim: HashMap<char, usize> = HashMap::new();
    for (t, sh) in terms.iter().zip(input_shapes) {
        for (&c, &d) in t.iter().zip(sh) { dim.insert(c, d.next_power_of_two()); }
    }
    let sums: Vec<char> = dim.keys().copied().filter(|c| !out.contains(c)).collect();
    if sums.len() != 1 { return None; }
    let k = sums[0];
    if !terms.iter().all(|t| t.contains(&k)) { return None; }
    if !out.iter().all(|c| dim.contains_key(c)) { return None; }
    let kd = dim[&k];
    let out_dims: Vec<usize> = out.iter().map(|c| dim[c]).collect();
    let out_size: usize = out_dims.iter().product::<usize>().max(1);
    // Pack operand t as rows over its free chars (in term order), the
    // summation axis contiguous: packed[row * kd + kk].
    struct Packed { rows: Vec<AlmostGoldilocksField>, nz: Vec<Option<Vec<(u32, AlmostGoldilocksField)>>>, free: Vec<char>, free_dims: Vec<usize> }
    let pack = |t: usize| -> Packed {
        let chars = &terms[t];
        let pd: Vec<usize> = chars.iter().map(|c| dim[c]).collect();
        let mut strides = Vec::with_capacity(pd.len());
        let mut st = 1usize;
        for &d in &pd { strides.push(st); st *= d; }
        let kpos = chars.iter().position(|&c| c == k).unwrap();
        let free: Vec<char> = chars.iter().copied().filter(|&c| c != k).collect();
        let free_pos: Vec<usize> = (0..chars.len()).filter(|&i| i != kpos).collect();
        let free_dims: Vec<usize> = free_pos.iter().map(|&i| pd[i]).collect();
        let nrows: usize = free_dims.iter().product::<usize>().max(1);
        let src = inputs[t];
        let kstride = strides[kpos];
        let mut rows = vec![AlmostGoldilocksField(0); nrows * kd];
        rows.par_chunks_mut(kd).enumerate().for_each(|(r, row)| {
            let mut rem = r;
            let mut base = 0usize;
            for (j, &i) in free_pos.iter().enumerate() { base += (rem % free_dims[j]) * strides[i]; rem /= free_dims[j]; }
            for (kk, x) in row.iter_mut().enumerate() {
                let li = base + kk * kstride;
                *x = if li < src.len() { src[li].reduce() } else { AlmostGoldilocksField(0) };
            }
        });
        let nz = rows.par_chunks(kd).map(|row| {
            let n = row.iter().filter(|x| x.0 != 0).count();
            if n * 8 <= kd { Some(row.iter().enumerate().filter(|(_, x)| x.0 != 0).map(|(i, &x)| (i as u32, x)).collect()) } else { None }
        }).collect();
        Packed { rows, nz, free, free_dims }
    };
    let (a, b) = (pack(0), pack(1));
    // Output position -> packed row of each operand.
    let opos: HashMap<char, usize> = out.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let row_map = |p: &Packed| -> Vec<(usize, usize)> {
        let mut m = Vec::new();
        let mut st = 1usize;
        for (c, &d) in p.free.iter().zip(&p.free_dims) { m.push((opos[c], st)); st *= d; }
        m
    };
    let (ma, mb) = (row_map(&a), row_map(&b));
    let res: Vec<AlmostGoldilocksField> = (0..out_size).into_par_iter().with_min_len(64).map(|o| {
        let mut multi = [0usize; 16];
        let mut rem = o;
        for (p, &d) in out_dims.iter().enumerate() { multi[p] = rem % d; rem /= d; }
        let ra: usize = ma.iter().map(|&(p, st)| multi[p] * st).sum();
        let rb: usize = mb.iter().map(|&(p, st)| multi[p] * st).sum();
        let (xa, xb) = (&a.rows[ra * kd..(ra + 1) * kd], &b.rows[rb * kd..(rb + 1) * kd]);
        let mut acc = AlmostGoldilocksField(0);
        if let Some(nz) = &a.nz[ra] {
            for &(i, v) in nz { acc = acc + v * xb[i as usize]; }
        } else if let Some(nz) = &b.nz[rb] {
            for &(i, v) in nz { acc = acc + xa[i as usize] * v; }
        } else {
            for i in 0..kd { acc = acc + xa[i] * xb[i]; }
        }
        acc.reduce()
    }).collect();
    Some(res)
}

fn einsum_compute(
    equation: &str,
    inputs: &[&[AlmostGoldilocksField]],
    input_shapes: &[Vec<usize>],
) -> Vec<AlmostGoldilocksField> {
    if fast_einsum_enabled() {
        if let Some(r) = einsum_compute_fast(equation, inputs, input_shapes) { return r; }
    }
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
    let out_dims: Vec<usize> =
        output_indices.iter().map(|&c| *dim_map.get(&c).unwrap_or(&1)).collect();
    let out_size: usize = out_dims.iter().product::<usize>().max(1);
    let sum_dims: Vec<usize> = sum_indices.iter().map(|&c| *dim_map.get(&c).unwrap_or(&1)).collect();
    let sum_size: usize = sum_dims.iter().product::<usize>().max(1);
    let term_chars: Vec<Vec<char>> = input_terms.iter().map(|t| t.chars().collect()).collect();
    let padded_shapes: Vec<Vec<usize>> = input_shapes
        .iter()
        .map(|shape| shape.iter().map(|&s| s.next_power_of_two()).collect())
        .collect();

    // Precompute, ONCE, each input-term char's (output/sum axis position, stride,
    // padded dim) so the hot loop is pure stride arithmetic instead of
    // per-element `HashMap<char,_>` (SipHash) lookups. The inner body runs
    // `out_size × sum_size × Σ|term|` times, so those map lookups dominated the
    // CPU forward on hidden=4096 matmuls. Output is byte-identical to the map
    // version: same out/sum decomposition, same strides, same `val % dim`, same
    // bounds check, same `agl_mul`/`agl_add` order.
    let onr = out_dims.len();
    let snr = sum_dims.len();
    assert!(onr <= 16 && snr <= 16, "einsum rank > 16 unsupported by fast path");
    let out_pos: HashMap<char, usize> =
        output_indices.iter().enumerate().map(|(p, &c)| (c, p)).collect();
    let sum_pos: HashMap<char, usize> =
        sum_indices.iter().enumerate().map(|(q, &c)| (c, q)).collect();
    // Per char of each term: (from_sum, axis_pos, stride, padded_dim).
    // `axis_pos == usize::MAX` ⇒ char in neither output nor sum (matches the old
    // `unwrap_or(&0)`: contributes index value 0).
    let term_axes: Vec<Vec<(bool, usize, usize, usize)>> = term_chars
        .iter()
        .zip(padded_shapes.iter())
        .map(|(chars, pshape)| {
            let mut axes = Vec::with_capacity(chars.len());
            let mut stride = 1usize;
            for (i, &c) in chars.iter().enumerate() {
                let dim = pshape[i];
                if let Some(&p) = out_pos.get(&c) {
                    axes.push((false, p, stride, dim));
                } else if let Some(&q) = sum_pos.get(&c) {
                    axes.push((true, q, stride, dim));
                } else {
                    axes.push((false, usize::MAX, stride, dim));
                }
                stride *= dim;
            }
            axes
        })
        .collect();

    (0..out_size)
        .into_par_iter()
        .map(|out_idx| {
            let mut out_multi = [0usize; 16];
            let mut remainder = out_idx;
            for p in 0..onr {
                out_multi[p] = remainder % out_dims[p];
                remainder /= out_dims[p];
            }
            let mut sum_multi = [0usize; 16];
            let mut sum = AlmostGoldilocksField(0);
            for sum_idx in 0..sum_size {
                let mut s_remainder = sum_idx;
                for q in 0..snr {
                    sum_multi[q] = s_remainder % sum_dims[q];
                    s_remainder /= sum_dims[q];
                }
                let mut product = AlmostGoldilocksField(1);
                for (t, input) in inputs.iter().enumerate() {
                    let mut linear_idx = 0usize;
                    for &(from_sum, axis, stride, dim) in &term_axes[t] {
                        let val = if axis == usize::MAX {
                            0
                        } else if from_sum {
                            sum_multi[axis]
                        } else {
                            out_multi[axis]
                        };
                        linear_idx += (val % dim) * stride;
                    }
                    if linear_idx < input.len() {
                        product = agl_mul(product, input[linear_idx]);
                    } else {
                        product = AlmostGoldilocksField(0);
                        break;
                    }
                }
                sum = agl_add(sum, product);
            }
            sum
        })
        .collect()
}

/// Broadcast an Ext2 device buffer by doubling `add_dims` times.
/// Input has `n_ext2` Ext2 elements (`2*n_ext2` u64s, interleaved). Returns a
/// new buffer with `n_ext2 << add_dims` Ext2 elements.
fn broadcast_device_buffer_ext2(
    src: &DeviceBuffer<u64>,
    n_ext2: usize,
    add_dims: usize,
) -> DeviceBuffer<u64> {
    if add_dims == 0 {
        let mut d = DeviceBuffer::<u64>::new(n_ext2 * 2).expect("alloc failed");
        let bytes = n_ext2 * 2 * std::mem::size_of::<u64>();
        unsafe {
            memcpy_dtod(
                d.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                bytes,
            )
            .expect("D2D clone failed");
        }
        return d;
    }
    let final_ext2 = n_ext2 << add_dims;
    let mut d_out = DeviceBuffer::<u64>::new(final_ext2 * 2).expect("alloc failed");
    let src_bytes = n_ext2 * 2 * std::mem::size_of::<u64>();
    unsafe {
        memcpy_dtod(
            d_out.as_mut_ptr() as *mut c_void,
            src.as_ptr() as *const c_void,
            src_bytes,
        )
        .expect("D2D copy failed");
    }
    let mut current_ext2 = n_ext2;
    for _ in 0..add_dims {
        let copy_bytes = current_ext2 * 2 * std::mem::size_of::<u64>();
        unsafe {
            memcpy_dtod(
                d_out.as_mut_ptr().add(current_ext2 * 2) as *mut c_void,
                d_out.as_ptr() as *const c_void,
                copy_bytes,
            )
            .expect("D2D copy failed");
        }
        current_ext2 *= 2;
    }
    d_out
}

// ============================================================================
// BasicBlock impl
// ============================================================================

impl BasicBlock for Einsum {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let input_refs: Vec<&[AlmostGoldilocksField]> = inputs
            .iter()
            .map(|w| w.data.as_ref().unwrap().evaluations_ref())
            .collect();
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
        // Tries the GPU einsum1/einsum2 kernels; falls back to the rayon CPU
        // path when the shape is unsupported (3+ inputs, > EINSUM_MAX_NDIM
        // dims, or small-output saturation issue).
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
        assert_eq!(out_claims.len(), 1, "Einsum expects 1 output claim");
        let challenge_point = out_claims[0].point.clone();
        let shapes: Vec<Vec<usize>> = witnesses.iter().map(|w| w.shape.clone()).collect();
        let (degree_one_challenges, high_degree_challenge) =
            compute_einsum_challenges(&self.equation, &shapes, &challenge_point);
        let num_first_rounds = high_degree_challenge.len();
        let num_second_rounds = self.summation_round;
        let total_rounds = num_first_rounds + num_second_rounds;
        let sumcheck_size = 1usize << total_rounds;

        if total_rounds <= gpu_sumcheck_threshold() {
            return self.prove_cpu(
                witnesses,
                edge_ids,
                &degree_one_challenges,
                &high_degree_challenge,
                total_rounds,
                num_second_rounds,
                sumcheck_size,
                false, // small poly — GPU input-prep is fine and fast
                transcript,
            );
        }

        // ----- GPU sumcheck path (host fallback on GPU OOM / KernelFailed) -----
        // Mirrors the fold-tree device-resident fallback (fold/tree.rs): snapshot
        // the transcript, attempt the GPU path, and on ANY GPU failure restore the
        // transcript and re-prove on the CPU. Under partition/cache memory pressure
        // a large head einsum (e.g. full-vocab lm_head) can OOM mid-`prove` —
        // surfacing as KernelFailed in fused_permute_partial_eval / from_device_buffers
        // — which previously aborted the whole proof. All big GPU allocations happen
        // before any transcript write, and prove_cpu emits a byte-identical
        // transcript, so the fallback is sound and verification is unaffected.
        let snapshot = transcript.clone();
        let gpu = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.prove_gpu(
                witnesses,
                edge_ids,
                &degree_one_challenges,
                &high_degree_challenge,
                total_rounds,
                num_second_rounds,
                sumcheck_size,
                transcript,
            )
        }));
        match gpu {
            Ok(r) => {
                // ZK4_EINSUM_CHECK=1: re-prove on the CPU from the same
                // transcript state and require every round message, final
                // eval and claim to agree in the field, and the two
                // transcripts to yield the same next challenge. The GPU path's
                // words may be non-canonical, so bytes can differ while values
                // cannot.
                if std::env::var("ZK4_EINSUM_CHECK").ok().as_deref() == Some("1") {
                    let mut tc = snapshot.clone();
                    let (cp, cc) = self.prove_cpu(
                        witnesses, edge_ids, &degree_one_challenges, &high_degree_challenge,
                        total_rounds, num_second_rounds, sumcheck_size, false, &mut tc);
                    let (gp, gc) = &r;
                    assert_eq!(cp.len(), gp.len(), "einsum check: proof count");
                    for (a, b) in cp.iter().zip(gp.iter()) {
                        assert_eq!(a.round_messages.len(), b.round_messages.len());
                        for (ra, rb) in a.round_messages.iter().zip(b.round_messages.iter()) {
                            assert_eq!(ra.len(), rb.len());
                            for (x, y) in ra.iter().zip(rb.iter()) {
                                assert!(ext2_field_eq(*x, *y), "einsum check: round message differs ({})", self.equation);
                            }
                        }
                        assert!(ext2_field_eq(a.final_eval, b.final_eval), "einsum check: final eval ({})", self.equation);
                    }
                    assert_eq!(cc.len(), gc.len());
                    for (a, b) in cc.iter().zip(gc.iter()) {
                        assert_eq!(a.edge_id, b.edge_id);
                        assert!(ext2_field_eq(a.eval, b.eval), "einsum check: claim eval ({})", self.equation);
                        assert_eq!(a.point.len(), b.point.len());
                        for (x, y) in a.point.iter().zip(b.point.iter()) { assert!(ext2_field_eq(*x, *y)); }
                    }
                    let mut tg = transcript.clone();
                    assert!(ext2_field_eq(tc.challenge_ext2(b"einsum_check"), tg.challenge_ext2(b"einsum_check")),
                        "einsum check: transcripts diverge ({})", self.equation);
                }
                r
            }
            Err(_) => {
                let shapes: Vec<_> = witnesses.iter().map(|w| w.shape.clone()).collect();
                eprintln!(
                    "[einsum] GPU sumcheck prove failed (likely GPU OOM under memory pressure) — host fallback: {} {:?}",
                    self.equation, shapes);
                *transcript = snapshot;
                self.prove_cpu(
                    witnesses,
                    edge_ids,
                    &degree_one_challenges,
                    &high_degree_challenge,
                    total_rounds,
                    num_second_rounds,
                    sumcheck_size,
                    true, // host-only: GPU just OOM'd, don't re-touch it
                    transcript,
                )
            }
        }
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
        let total_rounds = num_first_rounds + num_second_rounds;
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            out_claim.eval,
            total_rounds,
            claims.len(), // num_poly = num_inputs + 1 (eq)
            transcript,
        );
        if !ok {
            return false;
        }
        let one = AlmostGoldilocksExt2::one();
        let eq_eval =
            high_degree_challenge
                .iter()
                .zip(challenges[..num_first_rounds].iter())
                .fold(one, |acc, (hd_j, r_j)| {
                    ext2_mul(
                        acc,
                        ext2_add(
                            ext2_mul(*r_j, *hd_j),
                            ext2_mul(ext2_sub(one, *r_j), ext2_sub(one, *hd_j)),
                        ),
                    )
                });
        let product_eval = claims[..claims.len() - 1]
            .iter()
            .fold(one, |acc, c| ext2_mul(acc, c.eval));
        let expected = ext2_mul(eq_eval, product_eval);
        ext2_field_eq(sumcheck_proofs[0].final_eval, expected)
    }
}

impl Einsum {
    /// CPU sumcheck prover path. Also serves as the host fallback for
    /// [`Self::prove_gpu`]; emits a byte-identical transcript.
    fn prove_cpu(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        degree_one_challenges: &[Vec<AlmostGoldilocksExt2>],
        high_degree_challenge: &[AlmostGoldilocksExt2],
        total_rounds: usize,
        num_second_rounds: usize,
        sumcheck_size: usize,
        force_host: bool,
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let permute_vecs = &self.permute_vecs;
        let num_inputs = permute_vecs.len();
        let num_polys = num_inputs + 1; // inputs + eq
        let mut cpu_polys: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(num_polys);

        for i in 0..num_inputs {
            let witness = witnesses[i];
            let n = get_n(&witness.shape);
            let m = degree_one_challenges[i].len();
            let witness_evals_ref = witness.data.as_ref().unwrap().evaluations_ref();
            let ext2_poly = prepare_input_poly_cpu(
                witness_evals_ref,
                n,
                m,
                &permute_vecs[i],
                &degree_one_challenges[i],
                force_host,
                Some(witness),
            );
            let result_len = ext2_poly.len();
            let result_n = result_len.trailing_zeros() as usize;
            if result_n < total_rounds {
                cpu_polys.push(broadcast_evals_by_doubling_ext2(
                    &ext2_poly,
                    total_rounds - result_n,
                ));
            } else {
                cpu_polys.push(ext2_poly);
            }
        }

        if high_degree_challenge.is_empty() {
            cpu_polys.push(vec![AlmostGoldilocksExt2::one(); sumcheck_size]);
        } else {
            let eq = evaluate_lagrange_basis_ext2(high_degree_challenge);
            if num_second_rounds > 0 {
                cpu_polys.push(broadcast_evals_by_doubling_ext2(&eq, num_second_rounds));
            } else {
                cpu_polys.push(eq);
            }
        }

        let mut cpu_prover =
            CpuLinearSumcheckProverExt2::new(total_rounds, num_polys, transcript);
        let proof = cpu_prover.prove(&mut cpu_polys, transcript);
        let challenges = cpu_prover.challenges.clone();
        let mut claims = Vec::new();
        for i in 0..num_inputs {
            let n_i = get_n(&witnesses[i].shape);
            if n_i == 0 {
                let eval = AlmostGoldilocksExt2::from_base(
                    witnesses[i].data.as_ref().unwrap().index(0),
                );
                claims.push(Claim {
                    edge_id: edge_ids[i],
                    sparse_id: 0,
                    point: vec![],
                    eval,
                });
            } else {
                let point_i_perm: Vec<AlmostGoldilocksExt2> = degree_one_challenges[i]
                    .iter()
                    .chain(challenges.iter())
                    .copied()
                    .collect();
                claims.push(Claim {
                    edge_id: edge_ids[i],
                    sparse_id: 0,
                    point: invert_points_by_ranges(&point_i_perm, &permute_vecs[i]),
                    eval: cpu_prover.final_eval(i),
                });
            }
        }
        (vec![proof], claims)
    }

    /// GPU sumcheck prover path. Returns the same proof/claims as
    /// [`Self::prove_cpu`] (byte-identical transcript). May panic on GPU
    /// failure (OOM/KernelFailed); the caller wraps it in `catch_unwind` and
    /// falls back to `prove_cpu`. All large device allocations happen before
    /// any transcript write, so a caught failure leaves the snapshot pristine.
    fn prove_gpu(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        degree_one_challenges: &[Vec<AlmostGoldilocksExt2>],
        high_degree_challenge: &[AlmostGoldilocksExt2],
        total_rounds: usize,
        num_second_rounds: usize,
        sumcheck_size: usize,
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let permute_vecs = &self.permute_vecs;
        let num_inputs = permute_vecs.len();
        let etime = std::env::var("ZK4_EINSUM_TIMING").is_ok();
        let t_all = std::time::Instant::now();
        let mut gpu_buffers: Vec<DeviceBuffer<u64>> = Vec::with_capacity(num_inputs + 1);
        for i in 0..num_inputs {
            let _t_op = std::time::Instant::now();
            let witness = witnesses[i];
            let n = get_n(&witness.shape);
            let poly_size = 1usize << n;
            let m = degree_one_challenges[i].len();
            let perm_map = compute_perm_map(n, &permute_vecs[i]);
            let needs_permute = !is_identity_perm(&perm_map);
            let witness_evals_ref = witness.data.as_ref().unwrap().evaluations_ref();

            if needs_permute && m > 0 && n > gpu_fused_threshold() {
                // Device in, device out: the operand comes from the weight
                // cache (or one direct upload) and the partial evaluation stays
                // on the device. The previous path downloaded the result and
                // re-uploaded the same words.
                let d_evals = device_operand(witness, witness_evals_ref);
                let d_result = almost_goldilocks_cuda::partial_eval::fused_permute_partial_eval_dev(
                    &d_evals,
                    &degree_one_challenges[i],
                    &permute_vecs[i],
                    n,
                )
                .expect("fused_permute_partial_eval failed");
                let result_ext2_count = d_result.len() / 2;
                let fixed_n = result_ext2_count.trailing_zeros() as usize;
                if fixed_n < total_rounds {
                    let add_dims = total_rounds - fixed_n;
                    gpu_buffers.push(broadcast_device_buffer_ext2(&d_result, result_ext2_count, add_dims));
                } else {
                    gpu_buffers.push(d_result);
                }
            } else {
                // No intermediate Vec<u64>: AlmostGoldilocksField is a
                // transparent u64, and an unpermuted weight comes from the
                // device cache.
                let d_permuted: Arc<DeviceBuffer<u64>> = if needs_permute {
                    let permuted = permute_evals_by_ranges(witness_evals_ref, n, &permute_vecs[i]);
                    Arc::new(DeviceBuffer::<u64>::from_slice(evals_as_u64(&permuted)).expect("upload"))
                } else {
                    device_operand(witness, witness_evals_ref)
                };
                if m > 0 {
                    let output_half = poly_size >> 1;
                    let mut d_output = DeviceBuffer::<AlmostGoldilocksExt2>::new(output_half)
                        .expect("output alloc");
                    let d_r = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(
                        &degree_one_challenges[i],
                    )
                    .expect("r upload");
                    partial_eval_ext2_device_u64(&d_permuted, &mut d_output, &d_r, n, m)
                        .expect("partial_eval_ext2_device failed");
                    let result_ext2_count = poly_size >> m;
                    let mut d_u64 =
                        DeviceBuffer::<u64>::new(result_ext2_count * 2).expect("alloc");
                    unsafe {
                        memcpy_dtod(
                            d_u64.as_mut_ptr() as *mut c_void,
                            d_output.as_ptr() as *const c_void,
                            result_ext2_count * 2 * std::mem::size_of::<u64>(),
                        )
                        .expect("D2D copy");
                    }
                    let fixed_n = result_ext2_count.trailing_zeros() as usize;
                    if fixed_n < total_rounds {
                        let add_dims = total_rounds - fixed_n;
                        gpu_buffers.push(broadcast_device_buffer_ext2(
                            &d_u64,
                            result_ext2_count,
                            add_dims,
                        ));
                    } else {
                        gpu_buffers.push(d_u64);
                    }
                } else {
                    let mut d_ext2 = DeviceBuffer::<u64>::new(poly_size * 2).expect("alloc");
                    AlmostExt2Batch::from_base(&d_permuted, &mut d_ext2)
                        .expect("base→Ext2 failed");
                    if n < total_rounds {
                        let add_dims = total_rounds - n;
                        gpu_buffers.push(broadcast_device_buffer_ext2(&d_ext2, poly_size, add_dims));
                    } else {
                        gpu_buffers.push(d_ext2);
                    }
                }
            }
        }

        let t_ops = t_all.elapsed();
        if high_degree_challenge.is_empty() {
            let one = AlmostGoldilocksExt2::one();
            let raw = [one.c0.0, one.c1.0];
            let d_one = DeviceBuffer::<u64>::from_slice(&raw).expect("upload");
            gpu_buffers.push(broadcast_device_buffer_ext2(&d_one, 1, total_rounds));
        } else {
            let d_r = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(high_degree_challenge)
                .expect("r upload");
            let log_eq = high_degree_challenge.len();
            let eq_size = 1usize << log_eq;
            let (d_buf_a, d_buf_b, result_in_a) =
                ext2_eq_dp_all_device(&d_r, log_eq).expect("ext2_eq_dp_all_device failed");
            let d_eq_ext2 = if result_in_a { &d_buf_a } else { &d_buf_b };
            let mut d_eq_u64 = DeviceBuffer::<u64>::new(eq_size * 2).expect("alloc");
            unsafe {
                memcpy_dtod(
                    d_eq_u64.as_mut_ptr() as *mut c_void,
                    d_eq_ext2.as_ptr() as *const c_void,
                    eq_size * 2 * std::mem::size_of::<u64>(),
                )
                .expect("D2D copy");
            }
            if num_second_rounds > 0 {
                gpu_buffers.push(broadcast_device_buffer_ext2(&d_eq_u64, eq_size, num_second_rounds));
            } else {
                gpu_buffers.push(d_eq_u64);
            }
        }

        let t_eq = t_all.elapsed();
        let buf_refs: Vec<&DeviceBuffer<u64>> = gpu_buffers.iter().collect();
        let gpu_state = GpuSumcheckStateExt2::from_device_buffers(&buf_refs, sumcheck_size)
            .expect("from_device_buffers failed");
        let t_state = t_all.elapsed();
        let mut gpu_prover =
            GpuLinearSumcheckProver::new(total_rounds, gpu_buffers.len(), transcript);
        let proof = gpu_prover.prove_gpu_resident(gpu_state, transcript);
        let challenges = gpu_prover.challenges.clone();
        if etime && t_all.elapsed().as_millis() >= 20 {
            let sizes: Vec<usize> = witnesses.iter().map(|w| get_n(&w.shape)).collect();
            eprintln!("[einsum_t] {} n={:?} rounds={} size={} | operands {:?} eq {:?} state {:?} sumcheck {:?}",
                self.equation, sizes, total_rounds, sumcheck_size,
                t_ops, t_eq - t_ops, t_state - t_eq, t_all.elapsed() - t_state);
        }

        let mut claims = Vec::new();
        for i in 0..num_inputs {
            let n_i = get_n(&witnesses[i].shape);
            if n_i == 0 {
                let eval = AlmostGoldilocksExt2::from_base(witnesses[i].data.as_ref().unwrap().index(0));
                claims.push(Claim {
                    edge_id: edge_ids[i],
                    sparse_id: 0,
                    point: vec![],
                    eval,
                });
            } else {
                let point_i_perm: Vec<AlmostGoldilocksExt2> = degree_one_challenges[i]
                    .iter()
                    .chain(challenges.iter())
                    .copied()
                    .collect();
                claims.push(Claim {
                    edge_id: edge_ids[i],
                    sparse_id: 0,
                    point: invert_points_by_ranges(&point_i_perm, &permute_vecs[i]),
                    eval: gpu_prover.final_eval(i),
                });
            }
        }
        (vec![proof], claims)
    }
}

/// Prepare one input polynomial via fused-permute+partial-eval (GPU) when
/// both `n` and `m` are large enough; otherwise separate permute + partial
/// eval (CPU or GPU partial-eval per the threshold).
fn prepare_input_poly_cpu(
    witness_evals: &[AlmostGoldilocksField],
    n: usize,
    m: usize,
    permute_vec: &[(usize, usize)],
    challenges: &[AlmostGoldilocksExt2],
    // When set, never touch the GPU — used as the host fallback after a GPU
    // OOM/KernelFailed, where re-attempting any device op would just fail again.
    force_host: bool,
    // The operand's witness, when known: lets a model weight come from the
    // device cache (`device_operand`) instead of a fresh upload per proof.
    // Llama-3's lm_head shards (2^27 words, 12-round sumchecks that take this
    // small-sumcheck path) re-uploaded ~1 GB each per proof.
    witness: Option<&Witness>,
) -> Vec<AlmostGoldilocksExt2> {
    let perm_map = compute_perm_map(n, permute_vec);
    let needs_permute = !is_identity_perm(&perm_map);
    // FUSED_MAX_N is the LUT-overflow cap on the fused GPU kernel.
    const FUSED_MAX_N: usize = 28;
    // `force_host` may be promoted to true mid-function: if a device op OOMs
    // (KernelFailed) under memory pressure we don't re-attempt any GPU op —
    // it would just fail again — and finish this input on the host. This makes
    // the prove path degrade to host instead of panicking when the GPU is full
    // (e.g. real 6B/8B + full vocab + many concurrent partitions).
    let mut force_host = force_host;
    if !force_host && needs_permute && m > 0 && n > gpu_fused_threshold() && n <= FUSED_MAX_N {
        if let Some(w) = witness.filter(|w| w.role == Role::Constant) {
            let d = device_operand(w, witness_evals);
            let r = almost_goldilocks_cuda::partial_eval::fused_permute_partial_eval_dev(&d, challenges, permute_vec, n)
                .and_then(|d_out| d_out.read_slice(0, d_out.len()));
            match r {
                Ok(raw) => return raw.chunks_exact(2)
                    .map(|c| AlmostGoldilocksExt2::new(AlmostGoldilocksField(c[0]), AlmostGoldilocksField(c[1])))
                    .collect(),
                Err(e) => {
                    eprintln!("[einsum] fused_permute_partial_eval OOM ({:?}) — host fallback", e);
                    force_host = true;
                }
            }
        } else {
        match fused_permute_partial_eval(witness_evals, challenges, permute_vec, n) {
            Ok(out) => return out,
            Err(e) => {
                eprintln!("[einsum] fused_permute_partial_eval OOM ({:?}) — host fallback", e);
                force_host = true;
            }
        }
        }
    }
    if !force_host && !needs_permute && m > 0 && n > gpu_partial_eval_threshold() {
        if let Some(w) = witness.filter(|w| w.role == Role::Constant) {
            let d = device_operand(w, witness_evals);
            let r = (|| -> almost_goldilocks_cuda::error::Result<Vec<AlmostGoldilocksExt2>> {
                let mut d_out = DeviceBuffer::<AlmostGoldilocksExt2>::new(1usize << (n - 1))?;
                let d_r = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(challenges)?;
                partial_eval_ext2_device_u64(&d, &mut d_out, &d_r, n, m)?;
                almost_goldilocks_cuda::memory::sync_mine()?;
                d_out.read_slice(0, 1usize << (n - m))
            })();
            if let Ok(out) = r { return out; }
        }
    }
    let permuted_owned;
    let permuted: &[AlmostGoldilocksField] = if needs_permute {
        permuted_owned = permute_evals_by_ranges(witness_evals, n, permute_vec);
        &permuted_owned
    } else {
        witness_evals
    };
    if m > 0 {
        if !force_host && n > gpu_partial_eval_threshold() {
            match partial_eval_ext2(permuted, challenges) {
                Ok(out) => out,
                Err(_) => partial_eval_ext2_cpu(permuted, challenges),
            }
        } else {
            partial_eval_ext2_cpu(permuted, challenges)
        }
    } else {
        permuted
            .iter()
            .map(|&v| AlmostGoldilocksExt2::from_base(v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn make_witness(shape: Vec<usize>, vals: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Uint,
            0,
            Role::Input,
        )
    }

    #[test]
    fn run_elementwise_mul() {
        let a = make_witness(vec![4], vec![2, 3, 4, 5]);
        let b = make_witness(vec![4], vec![10, 20, 30, 40]);
        let e = Einsum::new("a,a->a", vec![vec![4], vec![4]], vec![4]);
        let out = e.run(&[&a, &b]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals, &[agl(20), agl(60), agl(120), agl(200)]);
    }

    #[test]
    fn run_matmul_2x2() {
        let a = make_witness(vec![2, 2], vec![1, 2, 3, 4]);
        let b = make_witness(vec![2, 2], vec![5, 6, 7, 8]);
        let e = Einsum::new("ij,jk->ik", vec![vec![2, 2], vec![2, 2]], vec![2, 2]);
        let out = e.run(&[&a, &b]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals, &[agl(23), agl(34), agl(31), agl(46)]);
    }

    #[test]
    fn run_dot_product_scalar_output() {
        let a = make_witness(vec![4], vec![1, 2, 3, 4]);
        let b = make_witness(vec![4], vec![5, 6, 7, 8]);
        let e = Einsum::new("i,i->", vec![vec![4], vec![4]], vec![]);
        let out = e.run(&[&a, &b]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], agl(70));
    }

    /// Full prove→verify roundtrip on a matmul. Forces the CPU path
    /// (total_rounds = 1 ≤ threshold).
    #[test]
    fn matmul_prove_verify_roundtrip_cpu_path() {
        let a = make_witness(vec![2, 2], vec![1, 2, 3, 4]);
        let b = make_witness(vec![2, 2], vec![5, 6, 7, 8]);
        let e = Einsum::new("ij,jk->ik", vec![vec![2, 2], vec![2, 2]], vec![2, 2]);
        let outs = e.run(&[&a, &b]);
        let y = &outs[0];
        let n_y = y.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"einsum");
        let point: Vec<_> = (0..n_y).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut t_prove = Transcript::new(b"einsum-prove");
        let (proofs, claims) = e.prove(&[&a, &b, y], &[0, 1, 2], &[&out_claim], &mut t_prove);
        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 2);

        let mut t_verify = Transcript::new(b"einsum-prove");
        let all = [&claims[0], &claims[1], &out_claim];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(e.verify(&[&a, &b, y], &all, &proof_refs, &mut t_verify));
    }

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    /// Matmul with `n > gpu_sumcheck_threshold()` (default 14) so the GPU
    /// sumcheck path fires. Output is shape [256, 256] → n_y = 16 = 8+8 bits,
    /// summation has 8 bits (k=256) → total_rounds = 16, above threshold.
    #[test]
    fn matmul_prove_verify_roundtrip_gpu_path() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let m = 256usize;
        let k_ = 256usize;
        let n = 256usize;
        // Build small-value-only inputs so the CPU reference is computable
        // in a reasonable time but the GPU sumcheck path triggers.
        let mut a_vals = vec![0u64; m * k_];
        for i in 0..m {
            for j in 0..k_ {
                a_vals[i + j * m] = ((i + j) as u64 % 7) + 1;
            }
        }
        let mut b_vals = vec![0u64; k_ * n];
        for i in 0..k_ {
            for j in 0..n {
                b_vals[i + j * k_] = ((i + j) as u64 % 5) + 1;
            }
        }
        let a = make_witness(vec![m, k_], a_vals);
        let b = make_witness(vec![k_, n], b_vals);
        let e = Einsum::new("ij,jk->ik", vec![vec![m, k_], vec![k_, n]], vec![m, n]);
        let outs = e.run(&[&a, &b]);
        let y = &outs[0];
        let n_y = y.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"big");
        let point: Vec<_> = (0..n_y).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point, eval };

        let mut t_prove = Transcript::new(b"big-prove");
        let (proofs, claims) = e.prove(&[&a, &b, y], &[0, 1, 2], &[&out_claim], &mut t_prove);
        assert_eq!(proofs.len(), 1);
        let mut t_verify = Transcript::new(b"big-prove");
        let all = [&claims[0], &claims[1], &out_claim];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(e.verify(&[&a, &b, y], &all, &proof_refs, &mut t_verify));
    }

    /// run_gpu on a matmul that fits the GPU einsum kernel produces the same
    /// output as the CPU path.
    #[test]
    fn matmul_run_gpu_matches_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let m = 64usize;
        let k_ = 32usize;
        let n = 128usize;
        let mut a_vals = vec![0u64; m * k_];
        for i in 0..m {
            for j in 0..k_ {
                a_vals[i + j * m] = ((i + j) as u64 % 11) + 1;
            }
        }
        let mut b_vals = vec![0u64; k_ * n];
        for i in 0..k_ {
            for j in 0..n {
                b_vals[i + j * k_] = ((i + j) as u64 % 13) + 1;
            }
        }
        let a = make_witness(vec![m, k_], a_vals);
        let b = make_witness(vec![k_, n], b_vals);
        let e = Einsum::new("ij,jk->ik", vec![vec![m, k_], vec![k_, n]], vec![m, n]);
        let cpu = e.run(&[&a, &b]);
        let gpu = e.run_gpu(&[&a, &b]);
        let cpu_evals = cpu[0].data.as_ref().unwrap().evaluations();
        let gpu_evals = gpu[0].data.as_ref().unwrap().evaluations();
        for i in 0..cpu_evals.len() {
            assert_eq!(cpu_evals[i].reduce(), gpu_evals[i].reduce(), "i = {}", i);
        }
    }
}
