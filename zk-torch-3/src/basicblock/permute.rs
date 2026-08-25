use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksBatch, GoldilocksField};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;

/// Permute block: reorders evaluations according to a variable permutation.
/// Produces both the permuted output and an auxiliary inverse-permutation witness.
#[derive(Clone, Debug)]
pub struct Permute {
    pub ranges: Vec<(usize, usize)>,
}

impl Permute {
    pub fn new(ranges: Vec<(usize, usize)>) -> Self {
        Self { ranges }
    }
}

fn permute_evals(evals: &[GoldilocksField], n: usize, ranges: &[(usize, usize)]) -> Vec<GoldilocksField> {
    let size = 1usize << n;
    assert_eq!(evals.len(), size);

    // Build new variable order
    let mut new_var_order = Vec::with_capacity(n);
    for &(start, end) in ranges {
        for v in start..end {
            new_var_order.push(v);
        }
    }

    // Build mapping
    let mut pos_new = vec![0usize; n];
    for (new_pos, &old_var) in new_var_order.iter().enumerate() {
        pos_new[old_var] = new_pos;
    }

    // Permute
    let mut out = vec![GoldilocksField(0); size];
    for idx_old in 0..size {
        let mut idx_new = 0usize;
        for old_var in 0..n {
            if idx_old & (1 << old_var) != 0 {
                idx_new |= 1 << pos_new[old_var];
            }
        }
        out[idx_new] = evals[idx_old];
    }
    out
}

impl BasicBlock for Permute {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let n = input.data.as_ref().unwrap().n();

        let permuted = permute_evals(evals, n, &self.ranges);

        vec![
            Witness::new(input.shape.clone(), permuted.clone(), input.data_type, input.sf, Role::Output),
            Witness::new(input.shape.clone(), evals.to_vec(), input.data_type, input.sf, Role::Auxiliary),
        ]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let n = input.data.as_ref().unwrap().n();
        let size = 1usize << n;

        // Build perm_map: perm_map[old_var] = new_var, matching CPU permute_evals.
        let mut new_var_order = Vec::with_capacity(n);
        for &(start, end) in &self.ranges {
            for v in start..end {
                new_var_order.push(v);
            }
        }
        let mut perm_map = vec![0i32; n];
        for (new_pos, &old_var) in new_var_order.iter().enumerate() {
            perm_map[old_var] = new_pos as i32;
        }

        // Identity permutation: skip the kernel; alias the buffer.
        let is_identity = perm_map.iter().enumerate().all(|(i, &v)| v == i as i32);
        let d_in = input.as_device_buf();
        let d_out_arc = if is_identity {
            Arc::clone(&d_in)
        } else {
            let mut d_out = DeviceBuffer::<u64>::new(size).expect("Permute: alloc failed");
            GoldilocksBatch::bit_permute(&d_in, &mut d_out, &perm_map).expect("Permute: gpu permute failed");
            Arc::new(d_out)
        };

        // Auxiliary witness is the *original* (unpermuted) evals — Arc-share with input buffer.
        vec![
            Witness::new_device(input.shape.clone(), d_out_arc, input.data_type, input.sf, Role::Output),
            Witness::new_device(input.shape.clone(), d_in, input.data_type, input.sf, Role::Auxiliary),
        ]
    }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let claim = out_claims[0];
        let new_claims = vec![Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: claim.point.clone(),
            eval: claim.eval,
        }];
        (vec![], new_claims)
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
