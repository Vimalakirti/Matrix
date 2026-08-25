use crate::util::arith::log2_ceil;
use goldilocks_cuda::GoldilocksField;
use ndarray::ArrayD;

/// Compute the total number of multilinear polynomial variables for a given shape.
pub fn shape_to_num_vars(shape: &[usize]) -> usize {
    shape.iter().map(|&s| log2_ceil(s.max(1))).sum()
}

/// Compute the total number of evaluations for a given shape (product of next powers of two).
pub fn shape_to_num_evals(shape: &[usize]) -> usize {
    shape
        .iter()
        .map(|&s| s.max(1).next_power_of_two())
        .product()
}

/// Pad a vector to the next power of two with a fill value.
pub fn pad_to_power_of_two<T: Clone>(data: &[T], fill: T) -> Vec<T> {
    let n = data.len().next_power_of_two();
    let mut padded = data.to_vec();
    padded.resize(n, fill);
    padded
}

/// Compute broadcast shape for two shapes (numpy-style broadcasting).
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    let a_padded: Vec<usize> = {
        let mut v = vec![1; max_len - a.len()];
        v.extend_from_slice(a);
        v
    };
    let b_padded: Vec<usize> = {
        let mut v = vec![1; max_len - b.len()];
        v.extend_from_slice(b);
        v
    };

    for (&da, &db) in a_padded.iter().zip(b_padded.iter()) {
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return None; // incompatible
        }
    }
    Some(result)
}

/// Return output axes that `x` is matched to (not broadcasted),
/// under NumPy-style broadcasting.
/// `out` is the broadcasted output shape.
pub fn matched_axes(x: &[usize], out: &[usize]) -> Option<Vec<usize>> {
    let nx = x.len();
    let no = out.len();

    if nx > no {
        return None;
    }

    let mut matched = Vec::new();

    // right-align, compare from rightmost axis
    for i in 0..no {
        let out_axis = no - 1 - i;
        let d_out = out[out_axis];
        let d_x = if i < nx { x[nx - 1 - i] } else { 1 };

        // matched means exact equality, excluding zero-length output axes
        if d_out != 0 && d_x == d_out && matched.len() < nx {
            matched.push(out_axis);
        }
    }

    matched.reverse();
    Some(matched)
}

/// Pad an ndarray to the next power of two along each dimension.
pub fn pad_to_pow_of_two(arr: &ArrayD<GoldilocksField>, fill: &GoldilocksField) -> ArrayD<GoldilocksField> {
    let shape = arr.shape();
    let new_shape: Vec<usize> = shape.iter().map(|&s| s.next_power_of_two()).collect();

    if shape == new_shape.as_slice() {
        return arr.clone();
    }

    let mut padded = ArrayD::from_elem(ndarray::IxDyn(&new_shape), *fill);

    // Copy elements using flat iteration over the original shape
    for (idx, &val) in arr.indexed_iter() {
        padded[idx.clone()] = val;
    }

    padded
}
