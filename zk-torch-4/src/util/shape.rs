use crate::util::arith::log2_ceil;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use ndarray::ArrayD;

/// Number of multilinear variables needed to index `shape` after rounding each
/// dim up to the next power of two.
pub fn shape_to_num_vars(shape: &[usize]) -> usize {
    shape.iter().map(|&s| log2_ceil(s.max(1))).sum()
}

/// Total number of evaluations once the shape is padded to the next-pow-2
/// hypercube — `prod_i next_pow_2(shape[i])`.
pub fn shape_to_num_evals(shape: &[usize]) -> usize {
    shape
        .iter()
        .map(|&s| s.max(1).next_power_of_two())
        .product()
}

/// Pad `data` up to the next power of two with `fill`. Returns a clone if
/// `data` is already a power-of-two length.
pub fn pad_to_power_of_two<T: Clone>(data: &[T], fill: T) -> Vec<T> {
    let n = data.len().next_power_of_two();
    let mut padded = data.to_vec();
    padded.resize(n, fill);
    padded
}

/// NumPy-style broadcast shape. Returns `None` on incompatibility.
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
            return None;
        }
    }
    Some(result)
}

/// Identify which output-shape axes `x` is NOT broadcasted on — i.e. axes
/// where x's matching dim equals the output's dim, paired right-to-left.
pub fn matched_axes(x: &[usize], out: &[usize]) -> Option<Vec<usize>> {
    let nx = x.len();
    let no = out.len();
    if nx > no {
        return None;
    }
    let mut matched = Vec::new();
    for i in 0..no {
        let out_axis = no - 1 - i;
        let d_out = out[out_axis];
        let d_x = if i < nx { x[nx - 1 - i] } else { 1 };
        if d_out != 0 && d_x == d_out && matched.len() < nx {
            matched.push(out_axis);
        }
    }
    matched.reverse();
    Some(matched)
}

/// Pad an `ndarray::ArrayD<AlmostGoldilocksField>` so every dimension is a
/// power of two, filling new positions with `fill`.
pub fn pad_to_pow_of_two(
    arr: &ArrayD<AlmostGoldilocksField>,
    fill: &AlmostGoldilocksField,
) -> ArrayD<AlmostGoldilocksField> {
    let shape = arr.shape();
    let new_shape: Vec<usize> = shape.iter().map(|&s| s.next_power_of_two()).collect();

    if shape == new_shape.as_slice() {
        return arr.clone();
    }

    let mut padded = ArrayD::from_elem(ndarray::IxDyn(&new_shape), *fill);
    for (idx, &val) in arr.indexed_iter() {
        padded[idx.clone()] = val;
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_helpers() {
        assert_eq!(shape_to_num_vars(&[1, 4, 8]), 0 + 2 + 3);
        assert_eq!(shape_to_num_evals(&[3, 5]), 4 * 8);
        assert_eq!(shape_to_num_evals(&[]), 1);
    }

    #[test]
    fn pad_to_pow2() {
        let v = pad_to_power_of_two::<u32>(&[1, 2, 3, 4, 5], 0);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 0, 0, 0]);
        let v = pad_to_power_of_two::<u32>(&[1, 2, 3, 4], 0);
        assert_eq!(v, vec![1, 2, 3, 4]);
        // `0.next_power_of_two() == 1`, so an empty input pads to a single
        // fill value. This matches zk-torch-3's convention.
        let v = pad_to_power_of_two::<u32>(&[], 7);
        assert_eq!(v, vec![7]);
    }

    #[test]
    fn broadcast_basic() {
        assert_eq!(broadcast_shape(&[4], &[2, 4]), Some(vec![2, 4]));
        assert_eq!(broadcast_shape(&[1, 4], &[3, 1]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 3], &[3, 2]), None);
        assert_eq!(broadcast_shape(&[], &[5]), Some(vec![5]));
    }

    #[test]
    fn matched_axes_returns_unbroadcast_axes() {
        // X[4] in C[2,4] is matched on axis 1 only (axis 0 is broadcast).
        assert_eq!(matched_axes(&[4], &[2, 4]), Some(vec![1]));
        // Both axes matched.
        assert_eq!(matched_axes(&[2, 4], &[2, 4]), Some(vec![0, 1]));
        // Scalar — no matched axes.
        assert_eq!(matched_axes(&[], &[2, 4]), Some(vec![]));
        // Higher-rank input than output is incompatible.
        assert_eq!(matched_axes(&[2, 4, 3], &[4, 3]), None);
    }

    #[test]
    fn pad_to_pow_of_two_arrayd() {
        let a = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 5]),
            (0..15).map(|x| AlmostGoldilocksField(x as u64)).collect(),
        )
        .unwrap();
        let padded = pad_to_pow_of_two(&a, &AlmostGoldilocksField(0));
        assert_eq!(padded.shape(), &[4, 8]);
        for i in 0..3 {
            for j in 0..5 {
                let want = AlmostGoldilocksField((i * 5 + j) as u64);
                assert_eq!(padded[[i, j]], want, "({},{})", i, j);
            }
        }
        // Padded cells are zero.
        for i in 0..4 {
            for j in 0..8 {
                if i >= 3 || j >= 5 {
                    assert_eq!(padded[[i, j]], AlmostGoldilocksField(0));
                }
            }
        }
        // No-op when already power-of-two.
        let b = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 4]),
            (0..8).map(|x| AlmostGoldilocksField(x as u64)).collect(),
        )
        .unwrap();
        let padded2 = pad_to_pow_of_two(&b, &AlmostGoldilocksField(0));
        assert_eq!(padded2, b);
    }
}
