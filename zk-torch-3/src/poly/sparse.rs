use std::any::Any;
use std::collections::{HashMap, VecDeque};

use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

use crate::poly::{DenseMLPoly, MLPoly};
use crate::util::arith::{gl_add, gl_mul, gl_sub, ext2_add, ext2_mul, ext2_sub};

/// Selection polynomial for sparse lookups.
/// Maps (input_index, table_index) pairs to entries in a lookup table.
#[derive(Clone, Debug)]
pub struct SelectionPolynomial {
    pub input_num_vars: usize,
    pub table_num_vars: usize,
    pub selection: Vec<(usize, usize)>,
}

impl SelectionPolynomial {
    pub fn empty() -> Self {
        Self {
            input_num_vars: 0,
            table_num_vars: 0,
            selection: vec![],
        }
    }

    pub fn new(input_num_vars: usize, table_num_vars: usize, selection: Vec<(usize, usize)>) -> Self {
        Self {
            input_num_vars,
            table_num_vars,
            selection,
        }
    }

    pub fn to_sparse(&self) -> SparseMLPoly {
        let n = self.input_num_vars + self.table_num_vars;
        let mut evaluations = HashMap::new();
        for &(input_index, table_index) in &self.selection {
            let index = input_index + table_index * (1 << self.input_num_vars);
            evaluations.insert(index, GoldilocksField(1));
        }
        let mut sp = SparseMLPoly::new(n, evaluations);
        sp.selection = self.clone();
        sp
    }
}

/// Sparse multilinear polynomial — stores only non-zero evaluations.
#[derive(Clone, Debug)]
pub struct SparseMLPoly {
    pub n: usize,
    pub evaluations: HashMap<usize, GoldilocksField>,
    pub indices: VecDeque<usize>,
    pub selection: SelectionPolynomial,
}

impl SparseMLPoly {
    pub fn new(n: usize, evals: HashMap<usize, GoldilocksField>) -> Self {
        let indices: VecDeque<usize> = evals.keys().copied().collect();
        Self {
            n,
            evaluations: evals,
            indices,
            selection: SelectionPolynomial::empty(),
        }
    }

    pub fn fix_variables(&self, partial_point: &[GoldilocksField]) -> SparseMLPoly {
        let m = partial_point.len();
        assert!(m <= self.n);
        if m == 0 {
            return self.clone();
        }

        // Convert to dense, fix, convert back to sparse
        let dense = self.to_dense();
        let fixed = DenseMLPoly::fix_variables(&dense, partial_point);

        let mut new_evals = HashMap::new();
        for (i, &v) in fixed.evaluations.iter().enumerate() {
            if v.0 != 0 {
                new_evals.insert(i, v);
            }
        }

        SparseMLPoly::new(self.n - m, new_evals)
    }

    pub fn to_dense(&self) -> DenseMLPoly {
        let size = 1usize << self.n;
        let mut evals = vec![GoldilocksField(0); size];
        for (&idx, &val) in &self.evaluations {
            if idx < size {
                evals[idx] = val;
            }
        }
        DenseMLPoly::new(self.n, evals)
    }

    /// Split into blocks of size 2^block_log for batch commitment.
    pub fn split_into_blocks(&self, block_log: usize) -> Vec<SparseMLPoly> {
        if self.n <= block_log {
            return vec![self.clone()];
        }
        let block_size = 1usize << block_log;
        let num_blocks = 1usize << (self.n - block_log);
        let mut blocks = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let start = b * block_size;
            let mut block_evals = HashMap::new();
            for (&idx, &val) in &self.evaluations {
                if idx >= start && idx < start + block_size {
                    block_evals.insert(idx - start, val);
                }
            }
            blocks.push(SparseMLPoly::new(block_log, block_evals));
        }

        blocks
    }
}

impl MLPoly for SparseMLPoly {
    fn fix_variables(&self, partial_point: &[GoldilocksField]) -> Box<dyn MLPoly> {
        Box::new(SparseMLPoly::fix_variables(self, partial_point))
    }

    fn n(&self) -> usize {
        self.n
    }

    fn len(&self) -> usize {
        1usize << self.n
    }

    fn evaluate_at_point(&self, point: &[GoldilocksField]) -> GoldilocksField {
        assert!(point.len() >= self.n);
        let point = &point[..self.n];
        // O(k*n) sparse evaluation: compute eq(idx, point) per entry instead of O(2^n) full table
        let one = GoldilocksField(1);
        let mut result = GoldilocksField(0);
        for (&idx, &val) in &self.evaluations {
            // eq(idx, point) = Π_i (bit_i * point_i + (1-bit_i) * (1-point_i))
            let mut eq_val = one;
            for i in 0..self.n {
                let bit = (idx >> i) & 1;
                let factor = if bit == 1 { point[i] } else { gl_sub(one, point[i]) };
                eq_val = gl_mul(eq_val, factor);
            }
            result = gl_add(result, gl_mul(eq_val, val));
        }
        result
    }

    fn evaluate_at_point_ext2(&self, point: &[GoldilocksExt2]) -> GoldilocksExt2 {
        assert!(point.len() >= self.n);
        let point = &point[..self.n];
        // O(k*n) sparse evaluation: compute eq(idx, point) per entry instead of O(2^n) full table
        let one = GoldilocksExt2::from_base(GoldilocksField(1));
        let mut result = GoldilocksExt2::zero();
        for (&idx, &val) in &self.evaluations {
            // eq(idx, point) = Π_i (bit_i * point_i + (1-bit_i) * (1-point_i))
            let mut eq_val = one;
            for i in 0..self.n {
                let bit = (idx >> i) & 1;
                let factor = if bit == 1 { point[i] } else { ext2_sub(one, point[i]) };
                eq_val = ext2_mul(eq_val, factor);
            }
            result = ext2_add(result, ext2_mul(eq_val, GoldilocksExt2::from_base(val)));
        }
        result
    }

    fn evaluations(&self) -> Vec<GoldilocksField> {
        let size = 1usize << self.n;
        let mut evals = vec![GoldilocksField(0); size];
        for (&idx, &val) in &self.evaluations {
            if idx < size {
                evals[idx] = val;
            }
        }
        evals
    }

    fn index(&self, index: usize) -> GoldilocksField {
        *self.evaluations.get(&index).unwrap_or(&GoldilocksField(0))
    }

    fn index_mut(&mut self, index: usize) -> &mut GoldilocksField {
        self.evaluations.entry(index).or_insert(GoldilocksField(0))
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
        let evals: HashMap<usize, GoldilocksField> = self
            .evaluations
            .iter()
            .map(|(&k, &v)| (k, gl_mul(v, scalar)))
            .collect();
        Box::new(SparseMLPoly::new(self.n, evals))
    }

    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly> {
        // Convert both to dense and add
        let dense_self = self.to_dense();
        let other_dense = other
            .as_any()
            .downcast_ref::<DenseMLPoly>()
            .cloned()
            .unwrap_or_else(|| {
                let other_sparse = other.as_any().downcast_ref::<SparseMLPoly>().unwrap();
                other_sparse.to_dense()
            });
        Box::new(dense_self.add_poly(&other_dense))
    }
}
