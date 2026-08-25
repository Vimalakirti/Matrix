//! Element-wise [`Add`] / [`Sub`] with NumPy-style broadcasting. Zero-sumcheck
//! claim transform: the verifier checks `a_eval ± b_eval == c_eval` plus that
//! each input claim's point was correctly extracted from the output point.

use std::sync::Arc;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::{AlmostGoldilocksBatch, AlmostGoldilocksField};
use almost_goldilocks_cuda::memory::DeviceBuffer;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{agl_add, agl_sub, ext2_add, ext2_field_eq, ext2_sub, get_n, log2_ceil};
use crate::util::shape::{broadcast_shape, matched_axes};

// ============================================================================
// Broadcast index helpers
// ============================================================================

/// Per-output-dim stride into `input` (zero for broadcast dimensions).
fn broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    let ndims = output_shape.len();
    let in_ndims = input_shape.len();
    let in_padded: Vec<usize> =
        input_shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
    let mut strides = vec![0usize; ndims];
    let mut cumul = 1usize;
    for k in 0..in_ndims {
        let out_dim = ndims - in_ndims + k;
        if input_shape[k] > 1 {
            strides[out_dim] = cumul;
            cumul *= in_padded[k];
        }
    }
    strides
}

/// Project the output claim's Ext2 point onto the input's matched axes
/// (broadcasted axes contribute no bits to the input point).
fn extract_broadcast_point(
    c_shape: &[usize],
    matched: &[usize],
    full_point: &[AlmostGoldilocksExt2],
) -> Vec<AlmostGoldilocksExt2> {
    let mut point = Vec::new();
    let mut bit_offset = 0;
    for dim in 0..c_shape.len() {
        let dim_bits = log2_ceil(c_shape[dim]);
        if matched.contains(&dim) {
            point.extend_from_slice(&full_point[bit_offset..bit_offset + dim_bits]);
        }
        bit_offset += dim_bits;
    }
    point
}

/// Flatten a multi-axis output index into the input flat index using
/// broadcast strides.
fn broadcast_flat_index(flat_idx: usize, c_padded: &[usize], strides: &[usize]) -> usize {
    let mut remaining = flat_idx;
    let mut idx = 0;
    for dim in 0..c_padded.len() {
        let dim_idx = remaining % c_padded[dim];
        remaining /= c_padded[dim];
        idx += dim_idx * strides[dim];
    }
    idx
}

// ============================================================================
// Add
// ============================================================================

#[derive(Clone, Debug)]
pub struct Add;

impl BasicBlock for Add {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "Add expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        let a_evals = a.data.as_ref().unwrap().evaluations_ref();
        let b_evals = b.data.as_ref().unwrap().evaluations_ref();
        let c_shape = broadcast_shape(&a.shape, &b.shape).unwrap_or_else(|| panic!(
            "shapes {:?} and {:?} are not broadcast-compatible; falling back to \
             the left shape here used to underflow broadcast_strides instead",
            a.shape, b.shape));
        let c_n = get_n(&c_shape);
        let c_size = 1usize << c_n;
        let c_padded: Vec<usize> =
            c_shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
        let a_strides = broadcast_strides(&a.shape, &c_shape);
        let b_strides = broadcast_strides(&b.shape, &c_shape);
        let result: Vec<AlmostGoldilocksField> = (0..c_size)
            .map(|i| {
                let ai = broadcast_flat_index(i, &c_padded, &a_strides);
                let bi = broadcast_flat_index(i, &c_padded, &b_strides);
                agl_add(a_evals[ai], b_evals[bi])
            })
            .collect();
        vec![Witness::new(c_shape, result, a.data_type, a.sf, Role::Output)]
    }

    /// GPU path: same-shape only (no broadcast). Uses
    /// `AlmostGoldilocksBatch::add` over device buffers. Broadcast falls
    /// back to CPU because the broadcast logic is index-bound, not arith-
    /// bound — wouldn't gain from a GPU kernel.
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let a = inputs[0];
        let b = inputs[1];
        if a.shape != b.shape {
            return self.run(inputs);
        }
        let n = get_n(&a.shape);
        let size = 1usize << n;
        let d_a = a.as_device_buf();
        let d_b = b.as_device_buf();
        let mut d_out = DeviceBuffer::<u64>::new(size).expect("Add: alloc out failed");
        AlmostGoldilocksBatch::add(&d_a, &d_b, &mut d_out).expect("Add: GPU add failed");
        vec![Witness::new_device(
            a.shape.clone(),
            Arc::new(d_out),
            a.data_type,
            a.sf,
            Role::Output,
        )]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert_eq!(witnesses.len(), 3, "Add expects 2 inputs + 1 output witness");
        assert_eq!(edge_ids.len(), 3, "Add expects 3 edges");
        assert_eq!(out_claims.len(), 1, "Add expects 1 output claim");
        let claim = out_claims[0];
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).expect("matched_axes(a)");
        let b_matched = matched_axes(b_shape, c_shape).expect("matched_axes(b)");

        let a_point = extract_broadcast_point(c_shape, &a_matched, &claim.point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, &claim.point);

        let a_eval = witnesses[0]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&a_point);
        let a_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: a_point,
            eval: a_eval,
        };
        let b_claim = Claim {
            edge_id: edge_ids[1],
            sparse_id: 0,
            point: b_point,
            eval: ext2_sub(claim.eval, a_eval),
        };
        (vec![], vec![a_claim, b_claim])
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        assert_eq!(claims.len(), 3, "Add expects 3 claims (a, b, c)");
        let c_point = &claims[2].point;
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).expect("matched_axes(a)");
        let b_matched = matched_axes(b_shape, c_shape).expect("matched_axes(b)");
        let a_point = extract_broadcast_point(c_shape, &a_matched, c_point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, c_point);
        ext2_field_eq(ext2_add(claims[0].eval, claims[1].eval), claims[2].eval)
            && claims[0].point == a_point
            && claims[1].point == b_point
    }
}

// ============================================================================
// Sub
// ============================================================================

#[derive(Clone, Debug)]
pub struct Sub;

impl BasicBlock for Sub {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2, "Sub expects 2 inputs");
        let a = inputs[0];
        let b = inputs[1];
        let a_evals = a.data.as_ref().unwrap().evaluations_ref();
        let b_evals = b.data.as_ref().unwrap().evaluations_ref();
        let c_shape = broadcast_shape(&a.shape, &b.shape).unwrap_or_else(|| panic!(
            "shapes {:?} and {:?} are not broadcast-compatible; falling back to \
             the left shape here used to underflow broadcast_strides instead",
            a.shape, b.shape));
        let c_n = get_n(&c_shape);
        let c_size = 1usize << c_n;
        let c_padded: Vec<usize> =
            c_shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
        let a_strides = broadcast_strides(&a.shape, &c_shape);
        let b_strides = broadcast_strides(&b.shape, &c_shape);
        let result: Vec<AlmostGoldilocksField> = (0..c_size)
            .map(|i| {
                let ai = broadcast_flat_index(i, &c_padded, &a_strides);
                let bi = broadcast_flat_index(i, &c_padded, &b_strides);
                agl_sub(a_evals[ai], b_evals[bi])
            })
            .collect();
        vec![Witness::new(c_shape, result, a.data_type, a.sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let a = inputs[0];
        let b = inputs[1];
        if a.shape != b.shape {
            return self.run(inputs);
        }
        let n = get_n(&a.shape);
        let size = 1usize << n;
        let d_a = a.as_device_buf();
        let d_b = b.as_device_buf();
        let mut d_out = DeviceBuffer::<u64>::new(size).expect("Sub: alloc out failed");
        AlmostGoldilocksBatch::sub(&d_a, &d_b, &mut d_out).expect("Sub: GPU sub failed");
        vec![Witness::new_device(
            a.shape.clone(),
            Arc::new(d_out),
            a.data_type,
            a.sf,
            Role::Output,
        )]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert_eq!(witnesses.len(), 3, "Sub expects 2 inputs + 1 output witness");
        assert_eq!(edge_ids.len(), 3, "Sub expects 3 edges");
        assert_eq!(out_claims.len(), 1, "Sub expects 1 output claim");
        let claim = out_claims[0];
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).expect("matched_axes(a)");
        let b_matched = matched_axes(b_shape, c_shape).expect("matched_axes(b)");
        let a_point = extract_broadcast_point(c_shape, &a_matched, &claim.point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, &claim.point);
        let a_eval = witnesses[0]
            .data
            .as_ref()
            .unwrap()
            .evaluate_at_point_ext2(&a_point);
        let a_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: a_point,
            eval: a_eval,
        };
        let b_claim = Claim {
            edge_id: edge_ids[1],
            sparse_id: 0,
            point: b_point,
            eval: ext2_sub(a_eval, claim.eval),
        };
        (vec![], vec![a_claim, b_claim])
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        assert_eq!(claims.len(), 3, "Sub expects 3 claims (a, b, c)");
        let c_point = &claims[2].point;
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).expect("matched_axes(a)");
        let b_matched = matched_axes(b_shape, c_shape).expect("matched_axes(b)");
        let a_point = extract_broadcast_point(c_shape, &a_matched, c_point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, c_point);
        ext2_field_eq(ext2_sub(claims[0].eval, claims[1].eval), claims[2].eval)
            && claims[0].point == a_point
            && claims[1].point == b_point
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(agl(v))
    }

    fn flat_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            data.into_iter().map(agl).collect(),
            DataType::Int,
            10,
            Role::Input,
        )
    }

    // ---------- run ----------

    #[test]
    fn add_run_elementwise() {
        let a = flat_witness(vec![4], vec![1, 2, 3, 4]);
        let b = flat_witness(vec![4], vec![10, 20, 30, 40]);
        let out = Add.run(&[&a, &b]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals, &[agl(11), agl(22), agl(33), agl(44)]);
    }

    #[test]
    fn sub_run_elementwise() {
        let a = flat_witness(vec![4], vec![10, 20, 30, 40]);
        let b = flat_witness(vec![4], vec![1, 2, 3, 4]);
        let out = Sub.run(&[&a, &b]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals, &[agl(9), agl(18), agl(27), agl(36)]);
    }

    /// Broadcast `A[4] + B[2,4]`: A is broadcast along dim 0, so
    /// `C[i,j] = A[j] + B[i,j]`. Little-endian flat index is `i + j*2`.
    #[test]
    fn add_run_broadcasts_along_dim0() {
        let a = flat_witness(vec![4], vec![10, 20, 30, 40]);
        let b = flat_witness(
            vec![2, 4],
            vec![1, 101, 2, 102, 3, 103, 4, 104],
        );
        let out = Add.run(&[&a, &b]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(out[0].shape, vec![2, 4]);
        assert_eq!(evals[0].0, 11);  // A[0]+B[0,0] = 10+1
        assert_eq!(evals[1].0, 111); // A[0]+B[1,0] = 10+101
        assert_eq!(evals[2].0, 22);  // A[1]+B[0,1] = 20+2
        assert_eq!(evals[3].0, 122);
    }

    // ---------- prove / verify ----------

    #[test]
    fn add_prove_then_verify_roundtrip() {
        let a = flat_witness(vec![4], vec![1, 2, 3, 4]);
        let b = flat_witness(vec![4], vec![10, 20, 30, 40]);
        let out = Add.run(&[&a, &b]);
        let pt = vec![lift(5), lift(7)];
        let c_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let c_claim = Claim { edge_id: 2, sparse_id: 0, point: pt.clone(), eval: c_eval };

        let mut t = Transcript::new(b"add");
        let (proofs, claims) = Add.prove(&[&a, &b, &out[0]], &[0, 1, 2], &[&c_claim], &mut t);
        assert!(proofs.is_empty());
        assert_eq!(claims.len(), 2);
        // Verifier sees [a_claim, b_claim, c_claim] and checks.
        let mut tv = Transcript::new(b"add");
        let all = [&claims[0], &claims[1], &c_claim];
        assert!(Add.verify(&[&a, &b, &out[0]], &all, &[], &mut tv));
    }

    /// Broadcast add: A matched on dim 1 only; B matched on both dims.
    /// Verifier must accept the projected input points.
    #[test]
    fn add_broadcast_prove_then_verify() {
        let a = flat_witness(vec![4], vec![10, 20, 30, 40]);
        let b = flat_witness(
            vec![2, 4],
            vec![1, 101, 2, 102, 3, 103, 4, 104],
        );
        let out = Add.run(&[&a, &b]);
        let pt = vec![lift(3), lift(5), lift(7)];
        let c_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let c_claim = Claim { edge_id: 2, sparse_id: 0, point: pt.clone(), eval: c_eval };

        let mut t = Transcript::new(b"bc");
        let (_, claims) = Add.prove(&[&a, &b, &out[0]], &[0, 1, 2], &[&c_claim], &mut t);
        // A skips dim 0's bit (axis 0 broadcast); B uses the full point.
        assert_eq!(claims[0].point, vec![pt[1], pt[2]]);
        assert_eq!(claims[1].point, pt);
        // a_eval + b_eval == c_eval.
        assert!(ext2_field_eq(
            ext2_add(claims[0].eval, claims[1].eval),
            c_eval
        ));
        let mut tv = Transcript::new(b"bc");
        let all = [&claims[0], &claims[1], &c_claim];
        assert!(Add.verify(&[&a, &b, &out[0]], &all, &[], &mut tv));
    }

    #[test]
    fn sub_broadcast_prove_then_verify() {
        let a = flat_witness(
            vec![2, 4],
            vec![100, 200, 300, 400, 500, 600, 700, 800],
        );
        let b = flat_witness(vec![4], vec![1, 2, 3, 4]);
        let out = Sub.run(&[&a, &b]);
        let pt = vec![lift(3), lift(5), lift(7)];
        let c_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let c_claim = Claim { edge_id: 2, sparse_id: 0, point: pt.clone(), eval: c_eval };

        let mut t = Transcript::new(b"sub-bc");
        let (_, claims) = Sub.prove(&[&a, &b, &out[0]], &[0, 1, 2], &[&c_claim], &mut t);
        assert!(ext2_field_eq(
            ext2_sub(claims[0].eval, claims[1].eval),
            c_eval
        ));
        let mut tv = Transcript::new(b"sub-bc");
        let all = [&claims[0], &claims[1], &c_claim];
        assert!(Sub.verify(&[&a, &b, &out[0]], &all, &[], &mut tv));
    }

    /// Verifier rejects tampered evals.
    #[test]
    fn add_verify_rejects_wrong_eval() {
        let a = flat_witness(vec![4], vec![1, 2, 3, 4]);
        let b = flat_witness(vec![4], vec![10, 20, 30, 40]);
        let out = Add.run(&[&a, &b]);
        let pt = vec![lift(5), lift(7)];
        let c_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let c_claim = Claim { edge_id: 2, sparse_id: 0, point: pt.clone(), eval: c_eval };
        let mut t = Transcript::new(b"add");
        let (_, mut claims) = Add.prove(&[&a, &b, &out[0]], &[0, 1, 2], &[&c_claim], &mut t);
        claims[0].eval = claims[0].eval + lift(1); // tamper
        let mut tv = Transcript::new(b"add");
        let all = [&claims[0], &claims[1], &c_claim];
        assert!(!Add.verify(&[&a, &b, &out[0]], &all, &[], &mut tv));
    }

    // ---------- GPU ----------

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    #[test]
    fn add_run_gpu_matches_cpu_same_shape() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let a = flat_witness(vec![8], (1..=8u64).collect());
        let b = flat_witness(vec![8], (100..=107u64).collect());
        let cpu = Add.run(&[&a, &b]);
        let gpu = Add.run_gpu(&[&a, &b]);
        let cpu_evals = cpu[0].data.as_ref().unwrap().evaluations();
        let gpu_evals = gpu[0].data.as_ref().unwrap().evaluations();
        for i in 0..cpu_evals.len() {
            assert_eq!(cpu_evals[i].reduce(), gpu_evals[i].reduce(), "i = {}", i);
        }
    }

    #[test]
    fn sub_run_gpu_matches_cpu_same_shape() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let a = flat_witness(vec![8], (100..108u64).collect());
        let b = flat_witness(vec![8], (1..9u64).collect());
        let cpu = Sub.run(&[&a, &b]);
        let gpu = Sub.run_gpu(&[&a, &b]);
        let cpu_evals = cpu[0].data.as_ref().unwrap().evaluations();
        let gpu_evals = gpu[0].data.as_ref().unwrap().evaluations();
        for i in 0..cpu_evals.len() {
            assert_eq!(cpu_evals[i].reduce(), gpu_evals[i].reduce(), "i = {}", i);
        }
    }

    /// Broadcast inputs go through the CPU path even when GPU is requested.
    #[test]
    fn add_run_gpu_broadcast_falls_back_to_cpu() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        let a = flat_witness(vec![4], (1..=4u64).collect());
        let b = flat_witness(vec![2, 4], (100..108u64).collect());
        let gpu = Add.run_gpu(&[&a, &b]);
        assert_eq!(gpu[0].shape, vec![2, 4]);
        // Same result as CPU.
        let cpu = Add.run(&[&a, &b]);
        let cpu_evals = cpu[0].data.as_ref().unwrap().evaluations();
        let gpu_evals = gpu[0].data.as_ref().unwrap().evaluations();
        for i in 0..cpu_evals.len() {
            assert_eq!(cpu_evals[i].reduce(), gpu_evals[i].reduce());
        }
    }
}
