use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksBatch, GoldilocksExt2, GoldilocksField};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_sub, get_n, gl_add, gl_sub, log2_ceil};
use crate::util::shape::{broadcast_shape, matched_axes};

/// Compute broadcast stride mapping from output dimensions to input.
/// For each output dimension, returns the stride in the input's flat index.
/// Broadcast dimensions (input size 1) have stride 0.
fn broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<usize> {
    let ndims = output_shape.len();
    let in_ndims = input_shape.len();
    let in_padded: Vec<usize> = input_shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
    let mut strides = vec![0usize; ndims];
    let mut cumul = 1usize;
    for k in 0..in_ndims {
        let out_dim = ndims - in_ndims + k; // right-aligned
        if input_shape[k] > 1 {
            strides[out_dim] = cumul;
            cumul *= in_padded[k];
        }
    }
    strides
}

/// Extract evaluation point for a broadcast input from the output claim's point.
/// Iterates through all output dimensions, tracking bit offsets, and only includes
/// bits for dimensions where the input is matched (not broadcast).
fn extract_broadcast_point(
    c_shape: &[usize],
    matched: &[usize],
    full_point: &[GoldilocksExt2],
) -> Vec<GoldilocksExt2> {
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

/// Compute broadcast flat index for input given output flat index decomposition.
fn broadcast_flat_index(
    flat_idx: usize,
    c_padded: &[usize],
    strides: &[usize],
) -> usize {
    let mut remaining = flat_idx;
    let mut idx = 0;
    for dim in 0..c_padded.len() {
        let dim_idx = remaining % c_padded[dim];
        remaining /= c_padded[dim];
        idx += dim_idx * strides[dim];
    }
    idx
}

/// Add block: output = input_0 + input_1
#[derive(Clone, Debug)]
pub struct Add;

impl BasicBlock for Add {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let b = inputs[1];
        let a_evals = a.data.as_ref().unwrap().evaluations_ref();
        let b_evals = b.data.as_ref().unwrap().evaluations_ref();

        let c_shape = broadcast_shape(&a.shape, &b.shape).unwrap_or_else(|| a.shape.clone());
        let c_n = get_n(&c_shape);
        let c_size = 1usize << c_n;

        let c_padded: Vec<usize> = c_shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
        let a_strides = broadcast_strides(&a.shape, &c_shape);
        let b_strides = broadcast_strides(&b.shape, &c_shape);

        let result: Vec<GoldilocksField> = (0..c_size)
            .map(|i| {
                let a_idx = broadcast_flat_index(i, &c_padded, &a_strides);
                let b_idx = broadcast_flat_index(i, &c_padded, &b_strides);
                gl_add(a_evals[a_idx], b_evals[b_idx])
            })
            .collect();

        vec![Witness::new(c_shape, result, a.data_type, a.sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Only fast-path the no-broadcast case. Broadcasted Add falls back to CPU.
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
        GoldilocksBatch::add(&d_a, &d_b, &mut d_out).expect("Add: gpu add failed");
        vec![Witness::new_device(a.shape.clone(), Arc::new(d_out), a.data_type, a.sf, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert!(witnesses.len() == 3, "Add expects 2 inputs and 1 output");
        assert!(edge_ids.len() == 3, "Add expects 2 input edges and 1 output edge");
        assert!(out_claims.len() == 1, "Add expects 1 output claim");

        let claim = out_claims[0];
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).unwrap();
        let b_matched = matched_axes(b_shape, c_shape).unwrap();

        let a_point = extract_broadcast_point(c_shape, &a_matched, &claim.point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, &claim.point);

        let a_eval = witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&a_point);
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
        assert!(claims.len() == 3, "Add expects 3 claims"); // [a, b, c]
        let c_point = &claims[2].point;
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).unwrap();
        let b_matched = matched_axes(b_shape, c_shape).unwrap();

        let a_point = extract_broadcast_point(c_shape, &a_matched, c_point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, c_point);

        let a_eval = claims[0].eval;
        let b_eval = claims[1].eval;
        let c_eval = claims[2].eval;

        ext2_add(a_eval, b_eval) == c_eval
            && claims[0].point == a_point
            && claims[1].point == b_point
    }
}

/// Sub block: output = input_0 - input_1
#[derive(Clone, Debug)]
pub struct Sub;

impl BasicBlock for Sub {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0];
        let b = inputs[1];
        let a_evals = a.data.as_ref().unwrap().evaluations_ref();
        let b_evals = b.data.as_ref().unwrap().evaluations_ref();

        let c_shape = broadcast_shape(&a.shape, &b.shape).unwrap_or_else(|| a.shape.clone());
        let c_n = get_n(&c_shape);
        let c_size = 1usize << c_n;

        let c_padded: Vec<usize> = c_shape.iter().map(|&s| s.max(1).next_power_of_two()).collect();
        let a_strides = broadcast_strides(&a.shape, &c_shape);
        let b_strides = broadcast_strides(&b.shape, &c_shape);

        let result: Vec<GoldilocksField> = (0..c_size)
            .map(|i| {
                let a_idx = broadcast_flat_index(i, &c_padded, &a_strides);
                let b_idx = broadcast_flat_index(i, &c_padded, &b_strides);
                gl_sub(a_evals[a_idx], b_evals[b_idx])
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
        GoldilocksBatch::sub(&d_a, &d_b, &mut d_out).expect("Sub: gpu sub failed");
        vec![Witness::new_device(a.shape.clone(), Arc::new(d_out), a.data_type, a.sf, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert!(witnesses.len() == 3, "Sub expects 2 inputs and 1 output");
        assert!(edge_ids.len() == 3, "Sub expects 2 input edges and 1 output edge");
        assert!(out_claims.len() == 1, "Sub expects 1 output claim");

        let claim = out_claims[0];
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).unwrap();
        let b_matched = matched_axes(b_shape, c_shape).unwrap();

        let a_point = extract_broadcast_point(c_shape, &a_matched, &claim.point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, &claim.point);

        let a_eval = witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&a_point);
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
        assert!(claims.len() == 3, "Sub expects 3 claims"); // [a, b, c]
        let c_point = &claims[2].point;
        let a_shape = &witnesses[0].shape;
        let b_shape = &witnesses[1].shape;
        let c_shape = &witnesses[2].shape;
        let a_matched = matched_axes(a_shape, c_shape).unwrap();
        let b_matched = matched_axes(b_shape, c_shape).unwrap();

        let a_point = extract_broadcast_point(c_shape, &a_matched, c_point);
        let b_point = extract_broadcast_point(c_shape, &b_matched, c_point);

        let a_eval = claims[0].eval;
        let b_eval = claims[1].eval;
        let c_eval = claims[2].eval;

        ext2_sub(a_eval, b_eval) == c_eval
            && claims[0].point == a_point
            && claims[1].point == b_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goldilocks_cuda::GoldilocksExt2;
    use crate::dag::DataType;

    #[test]
    fn test_add_block_run() {
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
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
        let add = Add;
        let result = add.run(&[&a, &b]);
        assert_eq!(result.len(), 1);
        let evals = result[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], GoldilocksField(11));
        assert_eq!(evals[1], GoldilocksField(22));
        assert_eq!(evals[2], GoldilocksField(33));
        assert_eq!(evals[3], GoldilocksField(44));
    }

    #[test]
    fn test_sub_block_run() {
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let sub = Sub;
        let result = sub.run(&[&a, &b]);
        assert_eq!(result.len(), 1);
        let evals = result[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], GoldilocksField(9));
        assert_eq!(evals[1], GoldilocksField(18));
        assert_eq!(evals[2], GoldilocksField(27));
        assert_eq!(evals[3], GoldilocksField(36));
    }

    #[test]
    fn test_add_prove_creates_claims() {
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
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
        let add = Add;
        let out = add.run(&[&a, &b]);
        let point = vec![GoldilocksExt2::from_base(GoldilocksField(5)), GoldilocksExt2::from_base(GoldilocksField(7))];
        let actual_out_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 2,
            sparse_id: 0,
            point,
            eval: actual_out_eval,
        };
        let mut transcript = Transcript::new(b"test");
        let (proofs, new_claims) = add.prove(
            &[&a, &b, &out[0]],
            &[0, 1, 2],
            &[&out_claim],
            &mut transcript,
        );
        assert!(proofs.is_empty());
        assert_eq!(new_claims.len(), 2);
        assert_eq!(new_claims[0].edge_id, 0);
        assert_eq!(new_claims[1].edge_id, 1);
        let eval_sum = ext2_add(new_claims[0].eval, new_claims[1].eval);
        assert_eq!(eval_sum, out_claim.eval);
    }

    /// Test broadcast add: A[4] + B[2,4] → C[2,4]
    /// In little-endian MLE: dim 0 (size 2) gets bits 0..1, dim 1 (size 4) gets bits 1..3
    /// A broadcasts along dim 0, so C[i,j] = A[j] + B[i,j]
    #[test]
    fn test_add_broadcast_run() {
        // A has shape [4], B has shape [2, 4]
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
            DataType::Uint,
            0,
            Role::Input,
        );
        // B[2,4]: B[0,j] = j+1, B[1,j] = j+101
        // In little-endian flat: index = i + j*2 where i=dim0, j=dim1
        // B[0,0]=1, B[1,0]=101, B[0,1]=2, B[1,1]=102, B[0,2]=3, B[1,2]=103, B[0,3]=4, B[1,3]=104
        let b = Witness::new(
            vec![2, 4],
            vec![
                GoldilocksField(1), GoldilocksField(101),   // j=0: i=0,1
                GoldilocksField(2), GoldilocksField(102),   // j=1: i=0,1
                GoldilocksField(3), GoldilocksField(103),   // j=2: i=0,1
                GoldilocksField(4), GoldilocksField(104),   // j=3: i=0,1
            ],
            DataType::Uint,
            0,
            Role::Input,
        );

        let add = Add;
        let result = add.run(&[&a, &b]);
        assert_eq!(result[0].shape, vec![2, 4]);
        let evals = result[0].data.as_ref().unwrap().evaluations_ref();
        // C[i,j] = A[j] + B[i,j], flat index = i + j*2
        // C[0,0] = A[0]+B[0,0] = 10+1 = 11
        assert_eq!(evals[0], GoldilocksField(11));
        // C[1,0] = A[0]+B[1,0] = 10+101 = 111
        assert_eq!(evals[1], GoldilocksField(111));
        // C[0,1] = A[1]+B[0,1] = 20+2 = 22
        assert_eq!(evals[2], GoldilocksField(22));
        // C[1,1] = A[1]+B[1,1] = 20+102 = 122
        assert_eq!(evals[3], GoldilocksField(122));
    }

    /// Test broadcast add prove/verify: A[4] + B[2,4] → C[2,4]
    #[test]
    fn test_add_broadcast_prove_verify() {
        let a = Witness::new(
            vec![4],
            vec![GoldilocksField(10), GoldilocksField(20), GoldilocksField(30), GoldilocksField(40)],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b = Witness::new(
            vec![2, 4],
            vec![
                GoldilocksField(1), GoldilocksField(101),
                GoldilocksField(2), GoldilocksField(102),
                GoldilocksField(3), GoldilocksField(103),
                GoldilocksField(4), GoldilocksField(104),
            ],
            DataType::Uint,
            0,
            Role::Input,
        );

        let add = Add;
        let out = add.run(&[&a, &b]);

        // Output has 3 variables: 1 bit for dim 0 (size 2), 2 bits for dim 1 (size 4)
        let point = vec![
            GoldilocksExt2::from_base(GoldilocksField(3)),  // dim 0 bit
            GoldilocksExt2::from_base(GoldilocksField(5)),  // dim 1 bit 0
            GoldilocksExt2::from_base(GoldilocksField(7)),  // dim 1 bit 1
        ];
        let c_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point: point.clone(), eval: c_eval };

        let mut transcript = Transcript::new(b"test_bc");
        let (proofs, claims) = add.prove(
            &[&a, &b, &out[0]],
            &[0, 1, 2],
            &[&out_claim],
            &mut transcript,
        );
        assert!(proofs.is_empty());
        assert_eq!(claims.len(), 2);

        // A matched axes = [1] (dim 1). A's point should be point[1..3] (dim 1's bits).
        assert_eq!(claims[0].point.len(), 2); // A has 2 vars (shape [4])
        assert_eq!(claims[0].point, vec![point[1], point[2]]);

        // B matched axes = [0, 1]. B's point should be full point.
        assert_eq!(claims[1].point.len(), 3);
        assert_eq!(claims[1].point, point);

        // Check A's eval is correct
        let a_eval = a.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert_eq!(claims[0].eval, a_eval);

        // Check a_eval + b_eval = c_eval
        assert_eq!(ext2_add(claims[0].eval, claims[1].eval), c_eval);

        // Verify
        let mut transcript_v = Transcript::new(b"test_bc");
        let all_claims = [&claims[0], &claims[1], &out_claim];
        let ok = add.verify(
            &[&a, &b, &out[0]],
            &all_claims,
            &[],
            &mut transcript_v,
        );
        assert!(ok, "broadcast add verify failed");
    }

    /// Test broadcast sub prove/verify: A[2,4] - B[4] → C[2,4]
    #[test]
    fn test_sub_broadcast_prove_verify() {
        let a = Witness::new(
            vec![2, 4],
            vec![
                GoldilocksField(100), GoldilocksField(200),
                GoldilocksField(300), GoldilocksField(400),
                GoldilocksField(500), GoldilocksField(600),
                GoldilocksField(700), GoldilocksField(800),
            ],
            DataType::Uint,
            0,
            Role::Input,
        );
        let b = Witness::new(
            vec![4],
            vec![GoldilocksField(1), GoldilocksField(2), GoldilocksField(3), GoldilocksField(4)],
            DataType::Uint,
            0,
            Role::Input,
        );

        let sub = Sub;
        let out = sub.run(&[&a, &b]);

        let point = vec![
            GoldilocksExt2::from_base(GoldilocksField(3)),
            GoldilocksExt2::from_base(GoldilocksField(5)),
            GoldilocksExt2::from_base(GoldilocksField(7)),
        ];
        let c_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim { edge_id: 2, sparse_id: 0, point: point.clone(), eval: c_eval };

        let mut transcript = Transcript::new(b"test_bc_sub");
        let (proofs, claims) = sub.prove(
            &[&a, &b, &out[0]],
            &[0, 1, 2],
            &[&out_claim],
            &mut transcript,
        );
        assert!(proofs.is_empty());

        // A matched axes = [0, 1]. A's point = full point.
        assert_eq!(claims[0].point, point);
        // B matched axes = [1]. B's point = point[1..3].
        assert_eq!(claims[1].point, vec![point[1], point[2]]);

        // Check a_eval - b_eval = c_eval
        assert_eq!(ext2_sub(claims[0].eval, claims[1].eval), c_eval);

        // Verify
        let mut transcript_v = Transcript::new(b"test_bc_sub");
        let all_claims = [&claims[0], &claims[1], &out_claim];
        let ok = sub.verify(
            &[&a, &b, &out[0]],
            &all_claims,
            &[],
            &mut transcript_v,
        );
        assert!(ok, "broadcast sub verify failed");
    }
}
