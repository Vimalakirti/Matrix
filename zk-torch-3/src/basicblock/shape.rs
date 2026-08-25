use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksField, GoldilocksExt2};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{get_n, ext2_mul, ext2_sub, ext2_inv};

/// ChangeShape block: reshape / pad without changing the underlying polynomial.
#[derive(Clone, Debug)]
pub struct ChangeShape {
    pub target_shape: Vec<usize>,
}

impl BasicBlock for ChangeShape {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let new_n = get_n(&self.target_shape);
        let new_size = 1usize << new_n;

        let mut new_evals = evals.to_vec();
        new_evals.resize(new_size, GoldilocksField(0));

        vec![Witness::new(
            self.target_shape.clone(),
            new_evals,
            input.data_type,
            input.sf,
            Role::Output,
        )]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        let input = inputs[0];
        let old_n = input.data.as_ref().unwrap().n();
        let new_n = get_n(&self.target_shape);

        // Same backing-buffer size — zero-copy reshape.
        if old_n == new_n && input.is_device_resident() {
            let buf = Arc::clone(input.device_buf().unwrap());
            return vec![Witness::new_device(
                self.target_shape.clone(), buf, input.data_type, input.sf, Role::Output,
            )];
        }
        // Pad/truncate paths are rare and tiny; fall back to CPU.
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
            // Growing: output has more variables than input.
            // f(r_0,...,r_{n_out-1}) = Π_{j=n_in}^{n_out-1} (1-r_j) * g(r_0,...,r_{n_in-1})
            // So g(r_0,...,r_{n_in-1}) = f(r_0,...) / Π(1-r_j)
            let one = GoldilocksExt2::from_base(GoldilocksField(1));
            let mut factor = one;
            for j in input_n..output_n {
                let r_j = claim.point[j];
                factor = ext2_mul(factor, ext2_sub(one, r_j));
            }
            let adjusted_eval = ext2_mul(claim.eval, ext2_inv(factor));
            point.truncate(input_n);
            let new_claims = vec![Claim {
                edge_id: edge_ids[0],
                sparse_id: 0,
                point,
                eval: adjusted_eval,
            }];
            (vec![], new_claims)
        } else {
            // Shrinking or same: pad point with zeros
            point.resize(input_n, GoldilocksExt2::zero());
            let new_claims = vec![Claim {
                edge_id: edge_ids[0],
                sparse_id: 0,
                point,
                eval: claim.eval,
            }];
            (vec![], new_claims)
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
