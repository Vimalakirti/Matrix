use std::any::Any;
use std::sync::Arc;

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2, DeviceBuffer};
use goldilocks_cuda::partial_eval::partial_eval_ext2_device_u64;
use once_cell::sync::OnceCell;
use rayon::prelude::*;

use crate::poly::MLPoly;
use crate::util::arith::{gl_add, gl_sub, gl_mul, ext2_add, ext2_sub, ext2_mul};

/// Dense multilinear polynomial stored as a vector of evaluations over the Boolean hypercube.
/// In the GPU path, data can optionally be stored in a DeviceBuffer.
#[derive(Clone, Debug)]
pub struct DenseMLPoly {
    pub n: usize,
    pub evaluations: Vec<GoldilocksField>,
}

impl DenseMLPoly {
    pub fn new(n: usize, data: Vec<GoldilocksField>) -> Self {
        let expected = 1usize << n;
        assert!(
            data.len() == expected,
            "DenseMLPoly::new: expected {} evaluations for n={}, got {}",
            expected,
            n,
            data.len()
        );
        Self {
            n,
            evaluations: data,
        }
    }

    pub fn len(&self) -> usize {
        self.evaluations.len()
    }

    pub fn zero(n: usize) -> Self {
        Self {
            n,
            evaluations: vec![GoldilocksField(0); 1 << n],
        }
    }

    /// Fix variables from the left (standard partial evaluation).
    /// Given f(x_1,...,x_N) and partial_point = (r_1,...,r_m),
    /// returns g(x_{m+1},...,x_N) = f(r_1,...,r_m, x_{m+1},...,x_N).
    pub fn fix_variables(&self, partial_point: &[GoldilocksField]) -> DenseMLPoly {
        let m = partial_point.len();
        assert!(m <= self.n, "Cannot fix more variables than available");
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
                let diff = gl_sub(b, a);
                data[j] = gl_add(a, gl_mul(ri, diff));
            }
            size = half;
        }

        data.truncate(size);
        DenseMLPoly {
            n: self.n - m,
            evaluations: data,
        }
    }

    /// Evaluate the polynomial at a full point.
    /// If point has more variables than the polynomial, only the first self.n are used
    /// (handles broadcast/scalar polynomials).
    pub fn evaluate(&self, point: &[GoldilocksField]) -> GoldilocksField {
        assert!(point.len() >= self.n, "Point length must be >= number of variables");
        let result = self.fix_variables(&point[..self.n]);
        result.evaluations[0]
    }

    /// Element-wise addition.
    pub fn add_poly(&self, other: &DenseMLPoly) -> DenseMLPoly {
        assert_eq!(self.n, other.n);
        let evals: Vec<GoldilocksField> = self
            .evaluations
            .par_iter()
            .zip(other.evaluations.par_iter())
            .map(|(&a, &b)| gl_add(a, b))
            .collect();
        DenseMLPoly {
            n: self.n,
            evaluations: evals,
        }
    }

    /// Element-wise subtraction.
    pub fn sub_poly(&self, other: &DenseMLPoly) -> DenseMLPoly {
        assert_eq!(self.n, other.n);
        let evals: Vec<GoldilocksField> = self
            .evaluations
            .par_iter()
            .zip(other.evaluations.par_iter())
            .map(|(&a, &b)| gl_sub(a, b))
            .collect();
        DenseMLPoly {
            n: self.n,
            evaluations: evals,
        }
    }

    /// Scale all evaluations by a scalar.
    pub fn scale(&self, scalar: GoldilocksField) -> DenseMLPoly {
        let evals: Vec<GoldilocksField> = self
            .evaluations
            .par_iter()
            .map(|&a| gl_mul(a, scalar))
            .collect();
        DenseMLPoly {
            n: self.n,
            evaluations: evals,
        }
    }

    /// Evaluate a base-field polynomial at an Ext2 point, returning an Ext2 result.
    /// Uses the same fix_variables algorithm but lifts base field to Ext2.
    pub fn evaluate_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
        assert!(point.len() >= self.n, "Point length must be >= number of variables");
        let m = self.n;
        if m == 0 {
            return GoldilocksExt2::from_base(self.evaluations[0]);
        }

        // Lift base field evaluations to Ext2
        let mut data: Vec<GoldilocksExt2> = self.evaluations.iter().map(|&v| GoldilocksExt2::from_base(v)).collect();
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

    /// Alias for evaluate_ext2 matching the MLPoly naming pattern.
    pub fn evaluate_at_point_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
        self.evaluate_ext2(point)
    }

    /// GPU-accelerated evaluate_ext2 for large polynomials.
    /// Falls back to CPU for small polynomials (n <= 14).
    pub fn evaluate_ext2_gpu(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
        let m = self.n;
        if m <= 14 {
            return self.evaluate_ext2(point);
        }

        // Upload polynomial and point to GPU
        let evals_u64: Vec<u64> = self.evaluations.iter().map(|v| v.0).collect();
        let d_input = DeviceBuffer::<u64>::from_slice(&evals_u64)
            .expect("GPU upload failed");
        let d_r = DeviceBuffer::<GoldilocksExt2>::from_slice(&point[..m])
            .expect("GPU upload failed");

        let output_half = self.evaluations.len() >> 1;
        let mut d_output = DeviceBuffer::<GoldilocksExt2>::new(output_half)
            .expect("alloc failed");

        partial_eval_ext2_device_u64(&d_input, &mut d_output, &d_r, m, m)
            .expect("GPU partial eval failed");

        // Read back single Ext2 result
        let result = d_output.read_slice(0, 1).expect("GPU readback failed");
        result[0]
    }
}

impl MLPoly for DenseMLPoly {
    fn fix_variables(&self, partial_point: &[GoldilocksField]) -> Box<dyn MLPoly> {
        Box::new(DenseMLPoly::fix_variables(self, partial_point))
    }

    fn n(&self) -> usize {
        self.n
    }

    fn len(&self) -> usize {
        self.evaluations.len()
    }

    fn evaluate_at_point(&self, point: &[GoldilocksField]) -> GoldilocksField {
        self.evaluate(point)
    }

    fn evaluate_at_point_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
        self.evaluate_ext2(point)
    }

    fn evaluations(&self) -> Vec<GoldilocksField> {
        self.evaluations.clone()
    }

    fn try_evaluations_ref(&self) -> Option<&[GoldilocksField]> {
        Some(&self.evaluations)
    }

    fn evaluations_ref(&self) -> &[GoldilocksField] {
        &self.evaluations
    }

    fn index(&self, index: usize) -> GoldilocksField {
        self.evaluations[index]
    }

    fn index_mut(&mut self, index: usize) -> &mut GoldilocksField {
        &mut self.evaluations[index]
    }

    fn clone_box(&self) -> Box<dyn MLPoly> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn mul_by_scalar(&self, scalar: GoldilocksField) -> Box<dyn MLPoly> {
        Box::new(self.scale(scalar))
    }

    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly> {
        let other_dense = other.as_any().downcast_ref::<DenseMLPoly>().expect("Can only add DenseMLPoly");
        Box::new(self.add_poly(other_dense))
    }
}

/// Device-resident dense multilinear polynomial.
/// Backing buffer is reference-counted (cheap clone for the multi-consumer fan-out
/// `Dag::run` does between edges). On any host-needing call we lazily download once
/// into the `host_cache` and serve subsequent reads from there.
#[derive(Debug)]
pub struct DeviceDenseMLPoly {
    pub n: usize,
    pub buf: Arc<DeviceBuffer<u64>>,
    host_cache: OnceCell<Vec<GoldilocksField>>,
}

impl DeviceDenseMLPoly {
    pub fn from_device(n: usize, buf: Arc<DeviceBuffer<u64>>) -> Self {
        assert_eq!(buf.len(), 1usize << n, "buffer size mismatch");
        Self { n, buf, host_cache: OnceCell::new() }
    }

    pub fn from_device_buffer(n: usize, buf: DeviceBuffer<u64>) -> Self {
        Self::from_device(n, Arc::new(buf))
    }

    /// Lazy host download. Subsequent calls reuse the cached vec.
    fn ensure_host(&self) -> &[GoldilocksField] {
        self.host_cache.get_or_init(|| {
            let raw: Vec<u64> = self.buf.to_vec().expect("device->host copy failed");
            raw.into_iter().map(GoldilocksField).collect()
        })
    }

    /// True if the host cache has already been populated.
    pub fn is_host_resident(&self) -> bool {
        self.host_cache.get().is_some()
    }

    /// Materialize host evaluations and consume the cache. Used by
    /// `Witness::evict_device_buffer` to convert a device-resident witness
    /// into a host-only one — the caller wraps the returned vec in a fresh
    /// `DenseMLPoly` and drops the `Arc<DeviceBuffer>` along with this struct.
    pub fn take_host_evals(&mut self) -> Vec<GoldilocksField> {
        // Force the host download if not already cached.
        let _ = self.ensure_host();
        // OnceCell::take leaves the cell empty; we own the vec from here.
        std::mem::replace(&mut self.host_cache, OnceCell::new())
            .into_inner()
            .expect("host cache populated by ensure_host")
    }
}

impl Clone for DeviceDenseMLPoly {
    fn clone(&self) -> Self {
        // Cheap: refcount the device buffer; host cache (if any) is also cloned.
        let cache = OnceCell::new();
        if let Some(v) = self.host_cache.get() {
            let _ = cache.set(v.clone());
        }
        Self { n: self.n, buf: Arc::clone(&self.buf), host_cache: cache }
    }
}

impl MLPoly for DeviceDenseMLPoly {
    fn fix_variables(&self, partial_point: &[GoldilocksField]) -> Box<dyn MLPoly> {
        let host = self.ensure_host();
        let dense = DenseMLPoly { n: self.n, evaluations: host.to_vec() };
        Box::new(dense.fix_variables(partial_point))
    }

    fn n(&self) -> usize { self.n }

    fn len(&self) -> usize { 1usize << self.n }

    fn evaluate_at_point(&self, point: &[GoldilocksField]) -> GoldilocksField {
        let host = self.ensure_host();
        let dense = DenseMLPoly { n: self.n, evaluations: host.to_vec() };
        dense.evaluate(point)
    }

    fn evaluate_at_point_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
        let host = self.ensure_host();
        let dense = DenseMLPoly { n: self.n, evaluations: host.to_vec() };
        dense.evaluate_ext2(point)
    }

    fn evaluations(&self) -> Vec<GoldilocksField> {
        self.ensure_host().to_vec()
    }

    fn try_evaluations_ref(&self) -> Option<&[GoldilocksField]> {
        Some(self.ensure_host())
    }

    fn evaluations_ref(&self) -> &[GoldilocksField] {
        self.ensure_host()
    }

    fn index(&self, index: usize) -> GoldilocksField {
        self.ensure_host()[index]
    }

    fn index_mut(&mut self, _index: usize) -> &mut GoldilocksField {
        panic!("DeviceDenseMLPoly is read-only; call Witness::ensure_host() to get a mutable host poly first");
    }

    fn clone_box(&self) -> Box<dyn MLPoly> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any { self }

    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn mul_by_scalar(&self, scalar: GoldilocksField) -> Box<dyn MLPoly> {
        let host = self.ensure_host();
        let evals: Vec<GoldilocksField> = host.par_iter().map(|&a| gl_mul(a, scalar)).collect();
        Box::new(DenseMLPoly { n: self.n, evaluations: evals })
    }

    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly> {
        let host = self.ensure_host();
        let other_evals = other.evaluations_ref();
        assert_eq!(host.len(), other_evals.len());
        let evals: Vec<GoldilocksField> = host.par_iter().zip(other_evals.par_iter())
            .map(|(&a, &b)| gl_add(a, b)).collect();
        Box::new(DenseMLPoly { n: self.n, evaluations: evals })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_ml_poly_new() {
        let poly = DenseMLPoly::new(2, vec![
            GoldilocksField(1),
            GoldilocksField(2),
            GoldilocksField(3),
            GoldilocksField(4),
        ]);
        assert_eq!(poly.n, 2);
        assert_eq!(poly.len(), 4);
    }

    #[test]
    fn test_fix_variables() {
        // f(x1, x2) with evaluations [1, 2, 3, 4]
        // f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
        // fix x1=r: g(x2) = f(r, x2)
        // g(0) = (1-r)*1 + r*2 = 1 + r
        // g(1) = (1-r)*3 + r*4 = 3 + r
        let poly = DenseMLPoly::new(2, vec![
            GoldilocksField(1),
            GoldilocksField(2),
            GoldilocksField(3),
            GoldilocksField(4),
        ]);
        let r = GoldilocksField(5);
        let result = DenseMLPoly::fix_variables(&poly, &[r]);
        assert_eq!(result.n, 1);
        assert_eq!(result.evaluations.len(), 2);
        assert_eq!(result.evaluations[0], GoldilocksField(6)); // 1 + 5 = 6
        assert_eq!(result.evaluations[1], GoldilocksField(8)); // 3 + 5 = 8
    }

    #[test]
    fn test_evaluate() {
        let poly = DenseMLPoly::new(2, vec![
            GoldilocksField(1),
            GoldilocksField(2),
            GoldilocksField(3),
            GoldilocksField(4),
        ]);
        // evaluate at (0,0) should give 1
        let val = poly.evaluate(&[GoldilocksField(0), GoldilocksField(0)]);
        assert_eq!(val, GoldilocksField(1));

        // evaluate at (1,0) should give 2
        let val = poly.evaluate(&[GoldilocksField(1), GoldilocksField(0)]);
        assert_eq!(val, GoldilocksField(2));

        // evaluate at (1,1) should give 4
        let val = poly.evaluate(&[GoldilocksField(1), GoldilocksField(1)]);
        assert_eq!(val, GoldilocksField(4));
    }
}
