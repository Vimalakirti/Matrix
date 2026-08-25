//! Dense multilinear polynomial stored as a flat vector of evaluations.
//!
//! Little-endian convention: variable 0 corresponds to bit 0 (LSB) of the
//! evaluation index. `fix_variables` fixes variable 0 first and operates on
//! pairs `[2j, 2j+1]`. This must stay consistent with
//! [`crate::poly::evaluate_lagrange_basis`] and every BasicBlock's indexing.

use std::any::Any;
use std::sync::Arc;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use almost_goldilocks_cuda::memory::DeviceBuffer;
use once_cell::sync::OnceCell;
use rayon::prelude::*;

use crate::poly::MLPoly;
use crate::util::arith::{agl_add, agl_mul, agl_sub, ext2_add, ext2_mul, ext2_sub};

#[derive(Clone, Debug)]
pub struct DenseMLPoly {
    pub n: usize,
    pub evaluations: Vec<AlmostGoldilocksField>,
}

impl DenseMLPoly {
    pub fn new(n: usize, data: Vec<AlmostGoldilocksField>) -> Self {
        let expected = 1usize << n;
        assert!(
            data.len() == expected,
            "DenseMLPoly::new: expected {} evaluations for n={}, got {}",
            expected,
            n,
            data.len()
        );
        Self { n, evaluations: data }
    }

    pub fn zero(n: usize) -> Self {
        Self { n, evaluations: vec![AlmostGoldilocksField(0); 1usize << n] }
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evaluations.is_empty()
    }

    /// Partial evaluation: given `f(x_0, …, x_{n-1})` and `partial_point =
    /// (r_0, …, r_{m-1})`, return `g(x_m, …, x_{n-1}) = f(r_0, …, r_{m-1},
    /// x_m, …)`. Operates from variable 0 (LSB) outward.
    pub fn fix_variables(&self, partial_point: &[AlmostGoldilocksField]) -> DenseMLPoly {
        let m = partial_point.len();
        assert!(m <= self.n, "cannot fix {} variables — poly only has {}", m, self.n);
        if m == 0 {
            return self.clone();
        }
        let mut data = self.evaluations.clone();
        let mut size = 1usize << self.n;
        for &ri in partial_point {
            let half = size / 2;
            for j in 0..half {
                let a = data[2 * j];
                let b = data[2 * j + 1];
                let diff = agl_sub(b, a);
                data[j] = agl_add(a, agl_mul(ri, diff));
            }
            size = half;
        }
        data.truncate(size);
        DenseMLPoly { n: self.n - m, evaluations: data }
    }

    /// Full evaluation at a base-field point. Reuses `fix_variables` to
    /// collapse to a single scalar.
    pub fn evaluate(&self, point: &[AlmostGoldilocksField]) -> AlmostGoldilocksField {
        assert!(
            point.len() >= self.n,
            "evaluate: point has {} coords but poly has {} vars",
            point.len(),
            self.n
        );
        let result = self.fix_variables(&point[..self.n]);
        result.evaluations[0]
    }

    pub fn add_poly(&self, other: &DenseMLPoly) -> DenseMLPoly {
        assert_eq!(self.n, other.n, "add_poly: arity mismatch");
        let evals: Vec<AlmostGoldilocksField> = self
            .evaluations
            .par_iter()
            .zip(other.evaluations.par_iter())
            .map(|(&a, &b)| agl_add(a, b))
            .collect();
        DenseMLPoly { n: self.n, evaluations: evals }
    }

    pub fn sub_poly(&self, other: &DenseMLPoly) -> DenseMLPoly {
        assert_eq!(self.n, other.n, "sub_poly: arity mismatch");
        let evals: Vec<AlmostGoldilocksField> = self
            .evaluations
            .par_iter()
            .zip(other.evaluations.par_iter())
            .map(|(&a, &b)| agl_sub(a, b))
            .collect();
        DenseMLPoly { n: self.n, evaluations: evals }
    }

    pub fn scale(&self, scalar: AlmostGoldilocksField) -> DenseMLPoly {
        let evals: Vec<AlmostGoldilocksField> = self
            .evaluations
            .par_iter()
            .map(|&a| agl_mul(a, scalar))
            .collect();
        DenseMLPoly { n: self.n, evaluations: evals }
    }

    /// Evaluate at an Ext2 point. Lifts each base-field eval to Ext2 first,
    /// then applies the same fix-variables algorithm with Ext2 arithmetic.
    pub fn evaluate_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        assert!(
            point.len() >= self.n,
            "evaluate_ext2: point has {} coords but poly has {} vars",
            point.len(),
            self.n
        );
        let m = self.n;
        if m == 0 {
            return AlmostGoldilocksExt2::from_base(self.evaluations[0]);
        }
        let mut data: Vec<AlmostGoldilocksExt2> = self
            .evaluations
            .iter()
            .map(|&v| AlmostGoldilocksExt2::from_base(v))
            .collect();
        let mut size = 1usize << m;
        for i in 0..m {
            let half = size / 2;
            for j in 0..half {
                let a = data[2 * j];
                let b = data[2 * j + 1];
                let diff = ext2_sub(b, a);
                data[j] = ext2_add(a, ext2_mul(point[i], diff));
            }
            size = half;
        }
        data[0]
    }

    /// MLPoly-naming alias for [`evaluate_ext2`].
    pub fn evaluate_at_point_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        self.evaluate_ext2(point)
    }
}

impl MLPoly for DenseMLPoly {
    fn fix_variables(&self, partial_point: &[AlmostGoldilocksField]) -> Box<dyn MLPoly> {
        Box::new(DenseMLPoly::fix_variables(self, partial_point))
    }

    fn n(&self) -> usize { self.n }
    fn len(&self) -> usize { self.evaluations.len() }

    fn evaluate_at_point(&self, point: &[AlmostGoldilocksField]) -> AlmostGoldilocksField {
        self.evaluate(point)
    }

    fn evaluate_at_point_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        self.evaluate_ext2(point)
    }

    fn evaluations(&self) -> Vec<AlmostGoldilocksField> { self.evaluations.clone() }

    fn try_evaluations_ref(&self) -> Option<&[AlmostGoldilocksField]> { Some(&self.evaluations) }

    fn evaluations_ref(&self) -> &[AlmostGoldilocksField] { &self.evaluations }

    fn index(&self, index: usize) -> AlmostGoldilocksField { self.evaluations[index] }

    fn index_mut(&mut self, index: usize) -> &mut AlmostGoldilocksField {
        &mut self.evaluations[index]
    }

    fn clone_box(&self) -> Box<dyn MLPoly> { Box::new(self.clone()) }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn mul_by_scalar(&self, scalar: AlmostGoldilocksField) -> Box<dyn MLPoly> {
        Box::new(self.scale(scalar))
    }

    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly> {
        let other_dense = other
            .as_any()
            .downcast_ref::<DenseMLPoly>()
            .expect("DenseMLPoly::add: only DenseMLPoly + DenseMLPoly is supported");
        Box::new(self.add_poly(other_dense))
    }
}

// ============================================================================
// Device-resident dense polynomial
// ============================================================================

/// Device-resident dense multilinear polynomial. Backing buffer is reference-
/// counted (cheap clone for the multi-consumer fan-out `Dag::run` does
/// between edges). Any host-needing call lazily downloads the data into
/// `host_cache` once, then serves all subsequent reads from there.
///
/// Used when a witness flows between two GPU-capable basicblocks; the
/// device-resident form lets the next block start kernel work without a
/// round-trip through host memory.
#[derive(Debug)]
pub struct DeviceDenseMLPoly {
    pub n: usize,
    pub buf: Arc<DeviceBuffer<u64>>,
    host_cache: OnceCell<Vec<AlmostGoldilocksField>>,
}

impl DeviceDenseMLPoly {
    pub fn from_device(n: usize, buf: Arc<DeviceBuffer<u64>>) -> Self {
        assert_eq!(buf.len(), 1usize << n, "DeviceDenseMLPoly buffer size mismatch");
        Self { n, buf, host_cache: OnceCell::new() }
    }

    pub fn from_device_buffer(n: usize, buf: DeviceBuffer<u64>) -> Self {
        Self::from_device(n, Arc::new(buf))
    }

    /// Lazy host download. Subsequent calls reuse the cached vec.
    fn ensure_host(&self) -> &[AlmostGoldilocksField] {
        self.host_cache.get_or_init(|| {
            let raw: Vec<u64> = self.buf.to_vec().expect("device->host copy failed");
            raw.into_iter().map(AlmostGoldilocksField).collect()
        })
    }

    pub fn is_host_resident(&self) -> bool {
        self.host_cache.get().is_some()
    }

    /// Materialize host evaluations and consume the cache. Used by the dag
    /// layer to convert a device-resident witness into a host-only one — the
    /// caller wraps the returned vec in a fresh `DenseMLPoly` and drops the
    /// `Arc<DeviceBuffer>` along with this struct.
    pub fn take_host_evals(&mut self) -> Vec<AlmostGoldilocksField> {
        let _ = self.ensure_host();
        std::mem::replace(&mut self.host_cache, OnceCell::new())
            .into_inner()
            .expect("host cache populated by ensure_host")
    }
}

impl Clone for DeviceDenseMLPoly {
    fn clone(&self) -> Self {
        let cache = OnceCell::new();
        if let Some(v) = self.host_cache.get() {
            let _ = cache.set(v.clone());
        }
        Self { n: self.n, buf: Arc::clone(&self.buf), host_cache: cache }
    }
}

impl MLPoly for DeviceDenseMLPoly {
    fn fix_variables(&self, partial_point: &[AlmostGoldilocksField]) -> Box<dyn MLPoly> {
        let host = self.ensure_host();
        let dense = DenseMLPoly { n: self.n, evaluations: host.to_vec() };
        Box::new(dense.fix_variables(partial_point))
    }

    fn n(&self) -> usize { self.n }
    fn len(&self) -> usize { 1usize << self.n }

    fn evaluate_at_point(&self, point: &[AlmostGoldilocksField]) -> AlmostGoldilocksField {
        let host = self.ensure_host();
        let dense = DenseMLPoly { n: self.n, evaluations: host.to_vec() };
        dense.evaluate(point)
    }

    fn evaluate_at_point_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        let host = self.ensure_host();
        let dense = DenseMLPoly { n: self.n, evaluations: host.to_vec() };
        dense.evaluate_ext2(point)
    }

    fn evaluations(&self) -> Vec<AlmostGoldilocksField> {
        self.ensure_host().to_vec()
    }

    fn try_evaluations_ref(&self) -> Option<&[AlmostGoldilocksField]> {
        Some(self.ensure_host())
    }

    fn evaluations_ref(&self) -> &[AlmostGoldilocksField] {
        self.ensure_host()
    }

    fn index(&self, index: usize) -> AlmostGoldilocksField {
        self.ensure_host()[index]
    }

    fn index_mut(&mut self, _index: usize) -> &mut AlmostGoldilocksField {
        panic!("DeviceDenseMLPoly is read-only; the dag layer must ensure_host() and rebuild as a DenseMLPoly before mutation")
    }

    fn clone_box(&self) -> Box<dyn MLPoly> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn mul_by_scalar(&self, scalar: AlmostGoldilocksField) -> Box<dyn MLPoly> {
        let host = self.ensure_host();
        let evals: Vec<AlmostGoldilocksField> =
            host.par_iter().map(|&a| agl_mul(a, scalar)).collect();
        Box::new(DenseMLPoly { n: self.n, evaluations: evals })
    }

    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly> {
        let host = self.ensure_host();
        let other_evals = other.evaluations_ref();
        assert_eq!(host.len(), other_evals.len(), "DeviceDenseMLPoly::add: arity mismatch");
        let evals: Vec<AlmostGoldilocksField> = host
            .par_iter()
            .zip(other_evals.par_iter())
            .map(|(&a, &b)| agl_add(a, b))
            .collect();
        Box::new(DenseMLPoly { n: self.n, evaluations: evals })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::evaluate_lagrange_basis;
    use crate::util::arith::ext2_field_eq;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    #[test]
    fn new_and_len() {
        let p = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        assert_eq!(p.n, 2);
        assert_eq!(p.len(), 4);
    }

    #[test]
    #[should_panic(expected = "expected 4 evaluations for n=2")]
    fn new_rejects_mismatched_length() {
        let _ = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3)]);
    }

    #[test]
    fn zero_poly() {
        let z = DenseMLPoly::zero(3);
        assert_eq!(z.n, 3);
        assert_eq!(z.evaluations.len(), 8);
        for e in z.evaluations {
            assert_eq!(e, agl(0));
        }
    }

    #[test]
    fn fix_one_variable_lsb() {
        // f(x0, x1) with evals [1, 2, 3, 4]:
        //   f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
        // Fixing x0 = r ⇒ g(x1) = f(r, x1):
        //   g(0) = (1-r)*1 + r*2 = 1 + r
        //   g(1) = (1-r)*3 + r*4 = 3 + r
        let p = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let r = agl(5);
        let g = DenseMLPoly::fix_variables(&p, &[r]);
        assert_eq!(g.n, 1);
        assert_eq!(g.evaluations, vec![agl(6), agl(8)]);
    }

    #[test]
    fn fix_no_variables_is_identity() {
        let p = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let g = p.fix_variables(&[]);
        assert_eq!(g.n, p.n);
        assert_eq!(g.evaluations, p.evaluations);
    }

    #[test]
    fn evaluate_at_boolean_points_recovers_evaluations() {
        let p = DenseMLPoly::new(2, vec![agl(10), agl(20), agl(30), agl(40)]);
        for x in 0..4usize {
            let pt = [agl(((x >> 0) & 1) as u64), agl(((x >> 1) & 1) as u64)];
            assert_eq!(p.evaluate(&pt), p.evaluations[x], "x = {}", x);
        }
    }

    /// Cross-check: full evaluation `f(r)` equals
    /// `Σ_x eq(r, x) · f.evaluations[x]`.
    #[test]
    fn evaluate_matches_lagrange_dot_evaluations() {
        let evals = vec![agl(7), agl(11), agl(13), agl(17), agl(19), agl(23), agl(29), agl(31)];
        let p = DenseMLPoly::new(3, evals.clone());
        let r = [agl(2), agl(5), agl(11)];
        let direct = p.evaluate(&r);
        let basis = evaluate_lagrange_basis(&r);
        let mut want = agl(0);
        for i in 0..8 {
            want = agl_add(want, agl_mul(basis[i], evals[i]));
        }
        assert_eq!(direct.reduce(), want.reduce());
    }

    #[test]
    fn evaluate_ext2_lifts_to_extension() {
        // Same poly; verify base-point evaluation lifts correctly through ext2.
        let p = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let r = [
            AlmostGoldilocksExt2::from_base(agl(0)),
            AlmostGoldilocksExt2::from_base(agl(0)),
        ];
        let v = p.evaluate_ext2(&r);
        assert!(ext2_field_eq(v, AlmostGoldilocksExt2::from_base(agl(1))));
        // And at (1, 1) it should be 4.
        let r2 = [
            AlmostGoldilocksExt2::from_base(agl(1)),
            AlmostGoldilocksExt2::from_base(agl(1)),
        ];
        let v2 = p.evaluate_ext2(&r2);
        assert!(ext2_field_eq(v2, AlmostGoldilocksExt2::from_base(agl(4))));
    }

    #[test]
    fn add_sub_scale_linearity() {
        let a = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let b = DenseMLPoly::new(2, vec![agl(10), agl(20), agl(30), agl(40)]);
        let s = a.add_poly(&b);
        let d = b.sub_poly(&a);
        let two_a = a.scale(agl(2));
        assert_eq!(s.evaluations, vec![agl(11), agl(22), agl(33), agl(44)]);
        assert_eq!(d.evaluations, vec![agl(9), agl(18), agl(27), agl(36)]);
        assert_eq!(two_a.evaluations, vec![agl(2), agl(4), agl(6), agl(8)]);
    }

    /// MLPoly trait dispatch (covers the dyn-MLPoly call paths used by the
    /// reducer / sumcheck code).
    // ----- DeviceDenseMLPoly -----

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    #[test]
    fn device_dense_round_trips_via_ensure_host() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let evals: Vec<u64> = (0..8u64).map(|i| i * 7 + 3).collect();
        let dbuf = DeviceBuffer::<u64>::from_slice(&evals).expect("upload");
        let d = DeviceDenseMLPoly::from_device_buffer(3, dbuf);
        assert_eq!(d.n, 3);
        assert!(!d.is_host_resident(), "host cache populates lazily");
        let host = d.evaluations();
        for (i, v) in host.iter().enumerate() {
            assert_eq!(v.0, evals[i]);
        }
        assert!(d.is_host_resident(), "after first read the cache is populated");
    }

    #[test]
    fn device_dense_mlpoly_dispatch_matches_dense() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let evals: Vec<u64> = (1..=4u64).collect();
        let dbuf = DeviceBuffer::<u64>::from_slice(&evals).expect("upload");
        let d = DeviceDenseMLPoly::from_device_buffer(2, dbuf);
        let boxed: Box<dyn MLPoly> = Box::new(d.clone());

        // index/len/n match.
        assert_eq!(boxed.n(), 2);
        assert_eq!(boxed.len(), 4);
        assert_eq!(boxed.index(2).0, 3);

        // evaluate_at_point matches a CPU DenseMLPoly with the same data.
        let cpu = DenseMLPoly::new(2, evals.iter().map(|&v| agl(v)).collect());
        let pt = [agl(7), agl(11)];
        assert_eq!(boxed.evaluate_at_point(&pt), cpu.evaluate(&pt));
    }

    #[test]
    fn device_dense_clone_preserves_cache() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let evals: Vec<u64> = (0..4u64).collect();
        let dbuf = DeviceBuffer::<u64>::from_slice(&evals).expect("upload");
        let d = DeviceDenseMLPoly::from_device_buffer(2, dbuf);
        let _ = d.evaluations(); // populate cache
        assert!(d.is_host_resident());
        let c = d.clone();
        assert!(c.is_host_resident(), "clone copies the populated cache");
    }

    #[test]
    fn device_dense_take_host_evals_returns_cached() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let evals: Vec<u64> = (10..14u64).collect();
        let dbuf = DeviceBuffer::<u64>::from_slice(&evals).expect("upload");
        let mut d = DeviceDenseMLPoly::from_device_buffer(2, dbuf);
        let taken = d.take_host_evals();
        for (i, v) in taken.iter().enumerate() {
            assert_eq!(v.0, evals[i]);
        }
        assert!(!d.is_host_resident(), "take_host_evals consumes the cache");
    }

    #[test]
    fn mlpoly_trait_dispatch_dense() {
        let p = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let boxed: Box<dyn MLPoly> = Box::new(p.clone());
        assert_eq!(boxed.n(), 2);
        assert_eq!(boxed.len(), 4);
        assert_eq!(boxed.index(0), agl(1));
        assert_eq!(boxed.index(3), agl(4));
        let fixed = boxed.fix_variables(&[agl(5)]);
        assert_eq!(fixed.n(), 1);
        let scaled = boxed.mul_by_scalar(agl(3));
        assert_eq!(scaled.evaluations(), vec![agl(3), agl(6), agl(9), agl(12)]);
        let added = boxed.add(&*boxed.clone_box());
        assert_eq!(added.evaluations(), vec![agl(2), agl(4), agl(6), agl(8)]);
    }
}
