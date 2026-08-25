//! `ChangeShape` — reshape / pad-grow without changing the underlying
//! polynomial. Pure metadata when the new index space is the same size as
//! the old; pad with zeros when growing, truncate when shrinking.

use std::sync::Arc;

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_inv, ext2_mul, ext2_sub, get_n};

#[derive(Clone, Debug)]
pub struct ChangeShape {
    pub target_shape: Vec<usize>,
}

impl BasicBlock for ChangeShape {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ChangeShape expects 1 input");
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let new_n = get_n(&self.target_shape);
        let new_size = 1usize << new_n;

        let mut new_evals = evals.to_vec();
        new_evals.resize(new_size, AlmostGoldilocksField(0));

        vec![Witness::new(
            self.target_shape.clone(),
            new_evals,
            input.data_type,
            input.sf,
            Role::Output,
        )]
    }

    /// Zero-copy reshape when the new index space is the same size as the old
    /// AND the input is device-resident. Otherwise fall back to the CPU path
    /// (pad/truncate paths are rare and tiny — no benefit from a GPU kernel).
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let input = inputs[0];
        let old_n = input.data.as_ref().unwrap().n();
        let new_n = get_n(&self.target_shape);
        if old_n == new_n && input.is_device_resident() {
            let buf = input.device_buf().expect("checked is_device_resident");
            return vec![Witness::new_device(
                self.target_shape.clone(),
                Arc::clone(&buf),
                input.data_type,
                input.sf,
                Role::Output,
            )];
        }
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let claim = out_claims[0];
        let input_n = get_n(&witnesses[0].shape);
        let output_n = claim.point.len();
        let mut point = claim.point.clone();

        if output_n > input_n {
            // Growing: f(r_0, …, r_{n_out-1}) = Π_{j=n_in..n_out} (1 − r_j) · g(r_0..n_in)
            // because the new variables are fixed to 0 in the padding region.
            let one = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(1));
            let mut factor = one;
            for j in input_n..output_n {
                factor = ext2_mul(factor, ext2_sub(one, claim.point[j]));
            }
            let adjusted_eval = ext2_mul(claim.eval, ext2_inv(factor));
            point.truncate(input_n);
            let new_claim = Claim {
                edge_id: edge_ids[0],
                sparse_id: 0,
                point,
                eval: adjusted_eval,
            };
            (vec![], vec![new_claim])
        } else {
            // Shrinking or same arity: pad the output point with zeros for
            // the variables the input still has past the output's arity.
            point.resize(input_n, AlmostGoldilocksExt2::zero());
            let new_claim = Claim {
                edge_id: edge_ids[0],
                sparse_id: 0,
                point,
                eval: claim.eval,
            };
            (vec![], vec![new_claim])
        }
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        _claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        true
    }
}

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

    #[test]
    fn run_same_size_preserves_data() {
        let w = Witness::new(
            vec![4],
            (0..4u64).map(agl).collect(),
            DataType::Int,
            10,
            Role::Input,
        );
        let cs = ChangeShape { target_shape: vec![2, 2] };
        let out = cs.run(&[&w]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        for i in 0..4 {
            assert_eq!(evals[i].0, i as u64);
        }
        assert_eq!(out[0].shape, vec![2, 2]);
    }

    #[test]
    fn run_grow_pads_zeros() {
        let w = Witness::new(
            vec![4],
            (0..4u64).map(agl).collect(),
            DataType::Int,
            10,
            Role::Input,
        );
        let cs = ChangeShape { target_shape: vec![8] };
        let out = cs.run(&[&w]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        for i in 0..4 {
            assert_eq!(evals[i].0, i as u64);
        }
        for i in 4..8 {
            assert_eq!(evals[i].0, 0);
        }
    }

    /// Prove path — growing arity: `g(r_0..n-1) = f(r_0..n-1, 0..0) /
    /// Π (1−r_j)`. Cross-checked by computing the same on the dense data.
    #[test]
    fn prove_grow_consistent_with_direct_eval() {
        // Input has 2 vars (shape [4]); target has 3 vars (shape [8]).
        let in_evals: Vec<_> = (1..=4u64).map(agl).collect();
        let w = Witness::new(vec![4], in_evals.clone(), DataType::Int, 10, Role::Input);
        let cs = ChangeShape { target_shape: vec![8] };
        let out = cs.run(&[&w]);

        let out_point = vec![lift(3), lift(5), lift(7)];
        let out_eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&out_point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point: out_point,
            eval: out_eval,
        };
        let mut t = Transcript::new(b"cs");
        let (proofs, claims) = cs.prove(&[&w, &out[0]], &[0, 1], &[&out_claim], &mut t);
        assert!(proofs.is_empty());
        assert_eq!(claims.len(), 1);
        // The recovered input claim must reproduce f at the truncated point.
        let direct = w.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert!(crate::util::arith::ext2_field_eq(direct, claims[0].eval));
    }

    /// Prove path — shrink/same arity: the input point is the output point
    /// (possibly padded with zeros).
    #[test]
    fn prove_same_arity_passes_through() {
        let w = Witness::new(
            vec![8],
            (0..8u64).map(agl).collect(),
            DataType::Int,
            10,
            Role::Input,
        );
        let cs = ChangeShape { target_shape: vec![2, 4] };
        let out = cs.run(&[&w]);
        let pt = vec![lift(2), lift(3), lift(5)];
        let eval = out[0].data.as_ref().unwrap().evaluate_at_point_ext2(&pt);
        let claim = Claim { edge_id: 1, sparse_id: 0, point: pt, eval };
        let mut t = Transcript::new(b"cs");
        let (proofs, claims) = cs.prove(&[&w, &out[0]], &[0, 1], &[&claim], &mut t);
        assert!(proofs.is_empty());
        assert_eq!(claims[0].point.len(), 3);
    }

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    /// Same-size GPU path zero-copies the device buffer.
    #[test]
    fn run_gpu_same_size_reuses_buffer() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        use almost_goldilocks_cuda::memory::DeviceBuffer;
        let raw: Vec<u64> = (0..4u64).collect();
        let buf = Arc::new(DeviceBuffer::<u64>::from_slice(&raw).expect("upload"));
        let w = Witness::new_device(vec![4], Arc::clone(&buf), DataType::Int, 10, Role::Input);
        let cs = ChangeShape { target_shape: vec![2, 2] };
        let out = cs.run_gpu(&[&w]);
        assert!(out[0].is_device_resident());
        // Underlying buffer is the same pointer.
        assert!(Arc::ptr_eq(&buf, &out[0].device_buf().unwrap()));
    }
}
