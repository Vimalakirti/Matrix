//! Base-field tensors evaluated at K points (CPU, rayon).
//!
//! A tensor is its padded column-major evaluation table: variable j is bit j
//! of the flat index (first axis lowest), matching zk-torch-4's witnesses.

use crate::morpheus::field8::{eq_table, K8};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use rayon::prelude::*;

#[inline]
fn kmul_base(k: &K8, v: u64) -> K8 { if v == 0 { K8::ZERO } else if v == 1 { *k } else { k.scale(v) } }

/// MLE of `vals` (canonical field words, len 2^n) at `point` (len n).
pub fn mle(vals: &[u64], point: &[K8]) -> K8 {
    let n = point.len();
    assert_eq!(vals.len(), 1usize << n, "mle: table/point mismatch");
    if n == 0 { return K8::from_base(vals[0]); }
    let lo = n / 2;
    let el = eq_table(&point[..lo]);
    let eh = eq_table(&point[lo..]);
    let l = el.len();
    let row = |(h, w): (usize, &K8)| -> K8 {
        let row = &vals[h * l..(h + 1) * l];
        let mut acc = K8::ZERO;
        for (e, &v) in el.iter().zip(row) { if v != 0 { acc = acc.add(&kmul_base(e, v)); } }
        acc.mul(w)
    };
    // Small tables: no thread-pool round trip.
    if n < 12 { return eh.iter().enumerate().map(row).fold(K8::ZERO, |a, b| a.add(&b)); }
    eh.par_iter().enumerate().map(row).reduce(|| K8::ZERO, |a, b| a.add(&b))
}

/// MLE of a K-valued table.
pub fn mle_k(vals: &[K8], point: &[K8]) -> K8 {
    assert_eq!(vals.len(), 1usize << point.len());
    let e = eq_table(point);
    vals.par_iter().zip(e.par_iter()).map(|(a, b)| a.mul(b)).reduce(|| K8::ZERO, |a, b| a.add(&b))
}

/// Partial evaluation: fix variables `fixed[(var, value)]` of an n-variable
/// table; the result is a K table over the remaining variables, in their
/// original relative order.
pub fn partial_eval(vals: &[u64], n: usize, fixed: &[(usize, K8)]) -> Vec<K8> {
    assert_eq!(vals.len(), 1usize << n);
    let mut is_fixed = vec![false; n];
    for &(v, _) in fixed { assert!(!is_fixed[v]); is_fixed[v] = true; }
    let rest: Vec<usize> = (0..n).filter(|&v| !is_fixed[v]).collect();
    let fvars: Vec<usize> = fixed.iter().map(|x| x.0).collect();
    let e = eq_table(&fixed.iter().map(|x| x.1).collect::<Vec<_>>());
    let out_len = 1usize << rest.len();
    // Index of (rest assignment r, fixed assignment f) in the table.
    let rest_off: Vec<usize> = (0..out_len).map(|r| rest.iter().enumerate().fold(0, |a, (j, &v)| a | (((r >> j) & 1) << v))).collect();
    let fix_off: Vec<usize> = (0..e.len()).map(|f| fvars.iter().enumerate().fold(0, |a, (j, &v)| a | (((f >> j) & 1) << v))).collect();
    rest_off.par_iter().map(|&ro| {
        let mut acc = K8::ZERO;
        for (fo, w) in fix_off.iter().zip(&e) {
            let v = vals[ro | fo];
            if v != 0 { acc = acc.add(&kmul_base(w, v)); }
        }
        acc
    }).collect()
}

/// Same, on a K table.
pub fn partial_eval_k(vals: &[K8], n: usize, fixed: &[(usize, K8)]) -> Vec<K8> {
    assert_eq!(vals.len(), 1usize << n);
    let mut is_fixed = vec![false; n];
    for &(v, _) in fixed { is_fixed[v] = true; }
    let rest: Vec<usize> = (0..n).filter(|&v| !is_fixed[v]).collect();
    let fvars: Vec<usize> = fixed.iter().map(|x| x.0).collect();
    let e = eq_table(&fixed.iter().map(|x| x.1).collect::<Vec<_>>());
    let out_len = 1usize << rest.len();
    let rest_off: Vec<usize> = (0..out_len).map(|r| rest.iter().enumerate().fold(0, |a, (j, &v)| a | (((r >> j) & 1) << v))).collect();
    let fix_off: Vec<usize> = (0..e.len()).map(|f| fvars.iter().enumerate().fold(0, |a, (j, &v)| a | (((f >> j) & 1) << v))).collect();
    rest_off.par_iter().map(|&ro| fix_off.iter().zip(&e).fold(K8::ZERO, |a, (fo, w)| a.add(&vals[ro | fo].mul(w)))).collect()
}

/// Canonical field word of a signed integer.
pub fn from_i128(x: i128) -> u64 {
    let q = Q as i128;
    let r = x.rem_euclid(q);
    r as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    #[test]
    fn partial_eval_then_mle_equals_mle() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let n = 9;
        let vals: Vec<u64> = (0..1 << n).map(|_| rng.gen::<u64>() % Q).collect();
        let pt: Vec<K8> = (0..n).map(|_| K8(std::array::from_fn(|_| rng.gen::<u64>() % Q))).collect();
        let full = mle(&vals, &pt);
        assert_eq!(full, mle_k(&vals.iter().map(|&v| K8::from_base(v)).collect::<Vec<_>>(), &pt));
        let fixed = vec![(1, pt[1]), (4, pt[4]), (8, pt[8])];
        let pe = partial_eval(&vals, n, &fixed);
        let rest_pt: Vec<K8> = (0..n).filter(|v| ![1, 4, 8].contains(v)).map(|v| pt[v]).collect();
        assert_eq!(mle_k(&pe, &rest_pt), full);
        let pk = partial_eval_k(&vals.iter().map(|&v| K8::from_base(v)).collect::<Vec<_>>(), n, &fixed);
        assert_eq!(pk, pe);
    }
}
