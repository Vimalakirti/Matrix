//! Sparse multilinear polynomial for lookup auxiliaries (range / two-pow).
//!
//! Implements zk-torch-2's z-t-2 form: a [`SelectionPolynomial`] captures a
//! set of `(input_index, table_index)` pairs (one per input row of the
//! lookup), which expands to a [`SparseMLPoly`] over `input_num_vars +
//! table_num_vars` variables whose only nonzero entries are at
//! `input_idx + table_idx · 2^input_num_vars`, valued `1`.
//!
//! This shape composes well with the Ajtai commitment: only the nonzero
//! positions cost anything (§5.5 of `zk-torch-4-plan.md`).

use std::any::Any;
use std::collections::HashMap;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::poly::{DenseMLPoly, MLPoly};
use crate::util::arith::{agl_add, agl_mul, agl_sub, ext2_add, ext2_mul, ext2_sub};

// ============================================================================
// SelectionPolynomial
// ============================================================================

/// A selection polynomial mapping `input_index → table_index`. Used by the
/// range / two-pow lookup protocols (see §5.5).
#[derive(Clone, Debug)]
pub struct SelectionPolynomial {
    pub input_num_vars: usize,
    pub table_num_vars: usize,
    pub selection: Vec<(usize, usize)>,
}

impl SelectionPolynomial {
    pub fn empty() -> Self {
        Self { input_num_vars: 0, table_num_vars: 0, selection: Vec::new() }
    }

    pub fn new(input_num_vars: usize, table_num_vars: usize, selection: Vec<(usize, usize)>) -> Self {
        Self { input_num_vars, table_num_vars, selection }
    }

    /// Expand into the ambient [`SparseMLPoly`]: positions
    /// `input_idx + table_idx · 2^input_num_vars` are set to 1, all others 0.
    pub fn to_sparse(&self) -> SparseMLPoly {
        let n = self.input_num_vars + self.table_num_vars;
        // Pre-size: a selection poly has exactly one nonzero per row, so the
        // map ends at `selection.len()` entries. Without capacity the default
        // HashMap rehashes O(log n) times while growing to millions of entries
        // (a big chunk of ScaleDown/NonNegative witness-gen on CV activations).
        let mut evaluations = HashMap::with_capacity(self.selection.len());
        for &(input_index, table_index) in &self.selection {
            debug_assert!(
                table_index < (1usize << self.table_num_vars),
                "table_index {} out of range for table_num_vars {}",
                table_index,
                self.table_num_vars
            );
            debug_assert!(
                input_index < (1usize << self.input_num_vars),
                "input_index {} out of range for input_num_vars {}",
                input_index,
                self.input_num_vars
            );
            let index = input_index + table_index * (1usize << self.input_num_vars);
            evaluations.insert(index, AlmostGoldilocksField(1));
        }
        let mut sp = SparseMLPoly::new(n, evaluations);
        sp.selection = self.clone();
        sp
    }

    /// Like [`to_sparse`] but does NOT materialize the `evaluations` HashMap —
    /// only `n` + `selection` are set. The DAG's table-commit split
    /// (`Dag::run` post-process) reads ONLY `selection` to build the per-chunk
    /// auxes, so for a will-be-split aux the full-arity HashMap that `to_sparse`
    /// builds is pure wasted work (built then discarded). The returned poly is
    /// "split-pending": its `evaluations` is empty, so it MUST be split (or
    /// re-materialized via `selection.to_sparse()`) before any evaluations /
    /// evaluate / commit access. In the real pipeline `Dag::run` splits it
    /// immediately after the forward loop, before anything reads it.
    pub fn to_sparse_selection_only(&self) -> SparseMLPoly {
        let n = self.input_num_vars + self.table_num_vars;
        let mut sp = SparseMLPoly::new(n, HashMap::new());
        sp.selection = self.clone();
        sp
    }

    /// Pick the cheap split-pending form when `Dag::run` will split this aux —
    /// i.e. splitting is enabled (`NO_SPARSE_SPLIT` unset) AND `table_num_vars >
    /// TABLE_COMMIT_LOG` — which is EXACTLY the split condition, so every
    /// split-pending poly produced here is split before use. Otherwise build the
    /// full `to_sparse` (the aux is consumed directly). Used by the range-check
    /// basicblocks (`NonNegative`, `ScaleDown`, `ScaleUp`) so the wasted
    /// full-arity HashMap is never built for split auxes. Byte-identical
    /// downstream: the chunks (and non-split auxes) are unchanged.
    pub fn to_sparse_dag(&self) -> SparseMLPoly {
        let split_on = std::env::var("NO_SPARSE_SPLIT").is_err();
        if split_on && self.table_num_vars > *crate::TABLE_COMMIT_LOG {
            self.to_sparse_selection_only()
        } else {
            self.to_sparse()
        }
    }
}

// ============================================================================
// SparseMLPoly
// ============================================================================

/// Sparse multilinear polynomial — stores only nonzero evaluations as a
/// `HashMap<flat_index, value>`. The optional `selection` field records the
/// `(input_idx, table_idx)` provenance when the poly came from a lookup
/// `SelectionPolynomial`.
#[derive(Clone, Debug)]
pub struct SparseMLPoly {
    pub n: usize,
    pub evaluations: HashMap<usize, AlmostGoldilocksField>,
    pub selection: SelectionPolynomial,
}

impl SparseMLPoly {
    pub fn new(n: usize, evals: HashMap<usize, AlmostGoldilocksField>) -> Self {
        // NOTE: a previous `indices: VecDeque<usize>` field (a copy of
        // `evals.keys()`) was built here on every construction but never read
        // anywhere — pure O(nnz) hash-walk + allocation per call, on both the
        // forward (`to_sparse`) and the prover (`fix_variables`). Removed.
        Self {
            n,
            evaluations: evals,
            selection: SelectionPolynomial::empty(),
        }
    }

    /// Partial evaluation. Routes through the dense form when `m > 0` because
    /// sparse partial-eval re-densifies after a single round anyway — the
    /// roundtrip is faster than reasoning about set-bit updates directly.
    pub fn fix_variables(&self, partial_point: &[AlmostGoldilocksField]) -> SparseMLPoly {
        let m = partial_point.len();
        assert!(m <= self.n, "cannot fix {} vars — poly has {}", m, self.n);
        if m == 0 {
            return self.clone();
        }
        let dense = self.to_dense();
        let fixed = DenseMLPoly::fix_variables(&dense, partial_point);
        let mut new_evals = HashMap::new();
        for (i, &v) in fixed.evaluations.iter().enumerate() {
            if v.reduce().0 != 0 {
                new_evals.insert(i, v);
            }
        }
        SparseMLPoly::new(self.n - m, new_evals)
    }

    /// Densify into a [`DenseMLPoly`] of size `2^n`. Costly for large `n`.
    pub fn to_dense(&self) -> DenseMLPoly {
        let size = 1usize << self.n;
        let mut evals = vec![AlmostGoldilocksField(0); size];
        for (&idx, &val) in &self.evaluations {
            if idx < size {
                evals[idx] = val;
            }
        }
        DenseMLPoly::new(self.n, evals)
    }

    /// **zk-torch-2 style aux split**: chunk each `(input_idx, table_idx)`
    /// entry's `table_idx` into `block_size`-bit slices, producing one
    /// new `SparseMLPoly` per chunk. Each chunk has `table_num_vars =
    /// min(remaining, block_size)`, so the new aux arity is
    /// `input_num_vars + block_size` (vs the original
    /// `input_num_vars + table_num_vars`).
    ///
    /// Mathematically: `table_idx = Σ_j chunk_j · 2^(j · block_size)`, so
    /// the verifier reconstructs the full table value via
    /// `eval_acc = Σ_j middle_claim_j · 2^(j · block_size)`.
    pub fn split_table_index_into_blocks(&self, block_size: usize) -> Vec<SparseMLPoly> {
        assert!(block_size > 0, "block_size must be > 0");
        let sel = &self.selection;
        if sel.table_num_vars == 0 {
            return vec![self.clone()];
        }
        let num_blocks = (sel.table_num_vars + block_size - 1) / block_size;
        let full_block_mod = 1usize << block_size;
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let offset = i * block_size;
            let start = 1usize << offset;
            let remaining = sel.table_num_vars - offset;
            let this_block_vars = remaining.min(block_size);
            let this_block_mod = 1usize << this_block_vars;
            let block_selection: Vec<(usize, usize)> = sel.selection.iter()
                .map(|&(input_index, table_index)| {
                    let v = (table_index / start) % full_block_mod;
                    (input_index, v % this_block_mod)
                })
                .collect();
            // Note: we use a full `block_size`-wide aux (with zero-padded
            // table_num_vars when the last chunk is shorter) so all chunks
            // have uniform arity = input_num_vars + block_size. Simplifies
            // the fold-tree bucketing.
            let block_sel = SelectionPolynomial::new(sel.input_num_vars, block_size, block_selection);
            blocks.push(block_sel.to_sparse());
        }
        blocks
    }

    /// Slice into blocks of size `2^block_log` for streamed processing
    /// (used by the offline / online commit paths to chunk large sparse polys).
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

    /// Number of nonzero entries.
    pub fn num_nonzero(&self) -> usize {
        self.evaluations.len()
    }
}

impl MLPoly for SparseMLPoly {
    fn fix_variables(&self, partial_point: &[AlmostGoldilocksField]) -> Box<dyn MLPoly> {
        Box::new(SparseMLPoly::fix_variables(self, partial_point))
    }

    fn n(&self) -> usize { self.n }
    fn len(&self) -> usize { 1usize << self.n }

    fn evaluate_at_point(&self, point: &[AlmostGoldilocksField]) -> AlmostGoldilocksField {
        assert!(point.len() >= self.n, "sparse eval: point too short");
        let point = &point[..self.n];
        // O(k·n) sparse evaluation: per nonzero entry, compute the eq factor
        // along the bits of its index, then accumulate.
        let one = AlmostGoldilocksField(1);
        let mut result = AlmostGoldilocksField(0);
        for (&idx, &val) in &self.evaluations {
            let mut eq_val = one;
            for i in 0..self.n {
                let bit = (idx >> i) & 1;
                let factor = if bit == 1 { point[i] } else { agl_sub(one, point[i]) };
                eq_val = agl_mul(eq_val, factor);
            }
            result = agl_add(result, agl_mul(eq_val, val));
        }
        result
    }

    fn evaluate_at_point_ext2(&self, point: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        assert!(point.len() >= self.n, "sparse ext2 eval: point too short");
        let point = &point[..self.n];
        let one = AlmostGoldilocksExt2::one();
        let mut result = AlmostGoldilocksExt2::zero();
        for (&idx, &val) in &self.evaluations {
            let mut eq_val = one;
            for i in 0..self.n {
                let bit = (idx >> i) & 1;
                let factor = if bit == 1 { point[i] } else { ext2_sub(one, point[i]) };
                eq_val = ext2_mul(eq_val, factor);
            }
            result = ext2_add(result, ext2_mul(eq_val, AlmostGoldilocksExt2::from_base(val)));
        }
        result
    }

    fn evaluations(&self) -> Vec<AlmostGoldilocksField> {
        let size = 1usize << self.n;
        let mut evals = vec![AlmostGoldilocksField(0); size];
        for (&idx, &val) in &self.evaluations {
            if idx < size {
                evals[idx] = val;
            }
        }
        evals
    }

    fn index(&self, index: usize) -> AlmostGoldilocksField {
        *self.evaluations.get(&index).unwrap_or(&AlmostGoldilocksField(0))
    }

    fn index_mut(&mut self, index: usize) -> &mut AlmostGoldilocksField {
        self.evaluations.entry(index).or_insert(AlmostGoldilocksField(0))
    }

    fn clone_box(&self) -> Box<dyn MLPoly> { Box::new(self.clone()) }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn mul_by_scalar(&self, scalar: AlmostGoldilocksField) -> Box<dyn MLPoly> {
        let evals: HashMap<usize, AlmostGoldilocksField> = self
            .evaluations
            .iter()
            .map(|(&k, &v)| (k, agl_mul(v, scalar)))
            .collect();
        Box::new(SparseMLPoly::new(self.n, evals))
    }

    fn add(&self, other: &dyn MLPoly) -> Box<dyn MLPoly> {
        // Mixed dense + sparse → densify both, add.
        let dense_self = self.to_dense();
        let other_dense = if let Some(d) = other.as_any().downcast_ref::<DenseMLPoly>() {
            d.clone()
        } else if let Some(s) = other.as_any().downcast_ref::<SparseMLPoly>() {
            s.to_dense()
        } else {
            panic!("SparseMLPoly::add: unknown rhs MLPoly type")
        };
        Box::new(dense_self.add_poly(&other_dense))
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

    /// SelectionPolynomial → SparseMLPoly: each (input, table) maps to the
    /// flat index `input + table · 2^input_num_vars`, valued 1.
    #[test]
    fn selection_to_sparse_indexing() {
        let sel = SelectionPolynomial::new(
            2, // 4 input rows
            3, // 8 table positions
            vec![(0, 5), (1, 0), (2, 7), (3, 3)],
        );
        let sp = sel.to_sparse();
        assert_eq!(sp.n, 5);
        // Position 0 should map to row 0, table 5 → idx 0 + 5*4 = 20.
        assert_eq!(sp.evaluations.get(&20), Some(&agl(1)));
        assert_eq!(sp.evaluations.get(&(1 + 0 * 4)), Some(&agl(1)));
        assert_eq!(sp.evaluations.get(&(2 + 7 * 4)), Some(&agl(1)));
        assert_eq!(sp.evaluations.get(&(3 + 3 * 4)), Some(&agl(1)));
        assert_eq!(sp.evaluations.len(), 4);
        // The provenance is preserved.
        assert_eq!(sp.selection.selection, sel.selection);
    }

    #[test]
    fn sparse_to_dense_round_trip() {
        let mut evals = HashMap::new();
        evals.insert(0, agl(1));
        evals.insert(5, agl(7));
        let sp = SparseMLPoly::new(3, evals);
        let d = sp.to_dense();
        assert_eq!(d.evaluations.len(), 8);
        assert_eq!(d.evaluations[0], agl(1));
        assert_eq!(d.evaluations[5], agl(7));
        // Untouched indices are zero.
        for i in [1, 2, 3, 4, 6, 7] {
            assert_eq!(d.evaluations[i], agl(0), "i={}", i);
        }
    }

    /// Sparse evaluation must agree with dense evaluation at the same point.
    /// This is the core soundness check for the sparse Lagrange-based eval.
    #[test]
    fn sparse_eval_matches_dense_eval() {
        let mut evals = HashMap::new();
        for (idx, v) in [(0, 1), (3, 5), (7, 11), (12, 13), (15, 17)] {
            evals.insert(idx, agl(v));
        }
        let sp = SparseMLPoly::new(4, evals);
        let d = sp.to_dense();
        let r = [agl(2), agl(3), agl(5), agl(7)];
        assert_eq!(sp.evaluate_at_point(&r).reduce(), d.evaluate(&r).reduce());
    }

    /// Sparse Ext2 evaluation matches dense Ext2 evaluation at the same point.
    #[test]
    fn sparse_eval_ext2_matches_dense_eval_ext2() {
        let mut evals = HashMap::new();
        for (idx, v) in [(0, 1u64), (3, 5), (7, 11), (12, 13), (15, 17)] {
            evals.insert(idx, agl(v));
        }
        let sp = SparseMLPoly::new(4, evals);
        let d = sp.to_dense();
        let r = vec![
            AlmostGoldilocksExt2::new(agl(2), agl(0)),
            AlmostGoldilocksExt2::new(agl(3), agl(1)),
            AlmostGoldilocksExt2::new(agl(5), agl(2)),
            AlmostGoldilocksExt2::new(agl(7), agl(3)),
        ];
        assert!(ext2_field_eq(
            sp.evaluate_at_point_ext2(&r),
            d.evaluate_ext2(&r)
        ));
    }

    /// At a Boolean point that matches a stored index, sparse eval should
    /// return that entry's value; anywhere else over the same support, 0.
    #[test]
    fn sparse_eval_at_boolean_point_selects_entry() {
        let mut evals = HashMap::new();
        evals.insert(5, agl(42)); // bit pattern 101 in 3 vars
        let sp = SparseMLPoly::new(3, evals);
        for x in 0..8usize {
            let pt = [
                agl(((x >> 0) & 1) as u64),
                agl(((x >> 1) & 1) as u64),
                agl(((x >> 2) & 1) as u64),
            ];
            let v = sp.evaluate_at_point(&pt);
            if x == 5 {
                assert_eq!(v.reduce(), agl(42));
            } else {
                assert_eq!(v.reduce(), agl(0));
            }
        }
    }

    #[test]
    fn fix_variables_collapses_correctly() {
        // SparseMLPoly fix_variables routes through dense; this just sanity-
        // checks that wiring (and that fix_variables(&[]) is identity).
        let mut evals = HashMap::new();
        evals.insert(0, agl(1));
        evals.insert(2, agl(3));
        let sp = SparseMLPoly::new(2, evals.clone());
        let same = sp.fix_variables(&[]);
        assert_eq!(same.n, 2);
        assert_eq!(same.evaluations.len(), evals.len());

        let g = sp.fix_variables(&[agl(0)]);
        // After fixing x0=0, only indices with bit0=0 survive (idx 0 and 2).
        let dense_g = g.to_dense();
        assert_eq!(dense_g.evaluations, vec![agl(1), agl(3)]);
    }

    /// Lookup-style soundness: for a Selection polynomial mapping row i to
    /// table index t_i, evaluating the resulting SparseMLPoly at (r_input,
    /// r_table) equals Σ_i eq(r_input, i) · eq(r_table, t_i).
    #[test]
    fn sparse_lookup_polynomial_evaluates_correctly() {
        let sel = SelectionPolynomial::new(
            2, // 4 input rows
            2, // 4 table positions
            vec![(0, 1), (1, 3), (2, 0), (3, 2)],
        );
        let sp = sel.to_sparse();
        let r_input = [agl(3), agl(5)];
        let r_table = [agl(7), agl(11)];
        let full_point: Vec<_> = r_input.iter().chain(r_table.iter()).cloned().collect();
        let got = sp.evaluate_at_point(&full_point);

        let basis_in = evaluate_lagrange_basis(&r_input);
        let basis_tb = evaluate_lagrange_basis(&r_table);
        let mut want = agl(0);
        for &(i, t) in &sel.selection {
            want = agl_add(want, agl_mul(basis_in[i], basis_tb[t]));
        }
        assert_eq!(got.reduce(), want.reduce());
    }

    #[test]
    fn split_into_blocks_preserves_content() {
        let mut evals = HashMap::new();
        for (i, v) in [(1, 10), (3, 20), (4, 30), (5, 40), (7, 50)] {
            evals.insert(i, agl(v));
        }
        let sp = SparseMLPoly::new(3, evals);
        let blocks = sp.split_into_blocks(2);
        assert_eq!(blocks.len(), 2);
        // Block 0 covers idx 0..4 → entries (1, 10), (3, 20).
        assert_eq!(blocks[0].n, 2);
        assert_eq!(blocks[0].evaluations.get(&1), Some(&agl(10)));
        assert_eq!(blocks[0].evaluations.get(&3), Some(&agl(20)));
        // Block 1 covers idx 4..8 → entries (4, 30), (5, 40), (7, 50) reindexed.
        assert_eq!(blocks[1].n, 2);
        assert_eq!(blocks[1].evaluations.get(&0), Some(&agl(30)));
        assert_eq!(blocks[1].evaluations.get(&1), Some(&agl(40)));
        assert_eq!(blocks[1].evaluations.get(&3), Some(&agl(50)));
    }

    #[test]
    fn split_into_blocks_no_op_for_small_polys() {
        let mut evals = HashMap::new();
        evals.insert(2, agl(5));
        let sp = SparseMLPoly::new(2, evals);
        let blocks = sp.split_into_blocks(3); // block_log > n
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].n, 2);
    }

    #[test]
    fn split_table_index_into_blocks_reconstructs_table_value() {
        // SelectionPolynomial with input_num_vars = 3, table_num_vars = 10,
        // entries (i, t_i). Split into block_size = 4 chunks → 3 chunks
        // (4 + 4 + 2 bits but we pad to 4 each for uniformity).
        let pairs = vec![
            (0usize, 0b10_1100_0001usize), // t = 705
            (1,     0b01_1010_0110),       // t = 422
            (3,     0b00_0000_1111),       // t = 15
            (5,     0b11_1100_0000),       // t = 960
        ];
        let sel = SelectionPolynomial::new(3, 10, pairs.clone());
        let sp = sel.to_sparse();
        let chunks = sp.split_table_index_into_blocks(4);
        assert_eq!(chunks.len(), 3, "10/4 → ceil = 3 chunks");

        for (input_idx, table_idx) in &pairs {
            let reconstructed: usize = chunks.iter().enumerate().map(|(j, chunk)| {
                let entry = chunk.selection.selection.iter()
                    .find(|&&(i, _)| i == *input_idx)
                    .expect("entry present in chunk");
                let chunk_val = entry.1;
                chunk_val * (1 << (j * 4))
            }).sum();
            assert_eq!(reconstructed, *table_idx,
                       "reconstruct mismatch for input {}: got {} expected {}",
                       input_idx, reconstructed, table_idx);
        }

        // Per-chunk arity = input_num_vars + block_size = 3 + 4 = 7.
        for chunk in &chunks {
            assert_eq!(chunk.n, 7);
            assert_eq!(chunk.selection.table_num_vars, 4);
        }
    }

    #[test]
    fn mlpoly_trait_dispatch_sparse() {
        let sel = SelectionPolynomial::new(2, 2, vec![(0, 0), (3, 3)]);
        let sp = sel.to_sparse();
        let boxed: Box<dyn MLPoly> = Box::new(sp.clone());
        assert_eq!(boxed.n(), 4);
        assert_eq!(boxed.len(), 16);
        assert_eq!(boxed.index(0), agl(1));
        assert_eq!(boxed.index(1), agl(0));
        assert_eq!(boxed.index(15), agl(1));
        let scaled = boxed.mul_by_scalar(agl(5));
        assert_eq!(scaled.index(0), agl(5));
        // Adding sparse + sparse routes through dense and works.
        let added = boxed.add(&*boxed.clone_box());
        assert_eq!(added.evaluations().iter().filter(|v| v.reduce().0 != 0).count(), 2);
    }

    #[test]
    fn num_nonzero_counts_set_entries() {
        let sel = SelectionPolynomial::new(2, 3, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
        let sp = sel.to_sparse();
        assert_eq!(sp.num_nonzero(), 4);
        let empty = SelectionPolynomial::empty().to_sparse();
        assert_eq!(empty.num_nonzero(), 0);
    }
}
