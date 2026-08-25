
use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksField};
use goldilocks_cuda::bit_decomp::decompose_bits32;

use crate::basicblock::BasicBlock;
use crate::basicblock::scale::BIT_TABLE_VARS;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::dense::DenseMLPoly;

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::get_n;

/// NonNegative block: proves that all values in the input are non-negative.
/// Uses a dense bit decomposition polynomial with 32 bits (BIT_TABLE_VARS=5).
///
/// SOUNDNESS LIMITATION: Only values < 2^32 are correctly decomposed into bits.
/// Values >= 2^32 produce all-zero bit witnesses. The verifier's eval_to_check
/// (in verify_range) will detect this mismatch and reject the proof.
/// To support the full Goldilocks non-negative range [0, p/2], BIT_TABLE_VARS
/// must be increased from 5 to 6 (see scale.rs).
#[derive(Clone, Debug)]
pub struct NonNegative;

impl BasicBlock for NonNegative {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let input_num_vars = get_n(&input.shape);
        let max_value = 1u64 << 32; // 32 bit positions from BIT_TABLE_VARS=5

        let n = 1usize << input_num_vars;

        // Build dense bit polynomial B(x,y) with input_num_vars + 5 variables
        let total_vars = input_num_vars + BIT_TABLE_VARS;
        let total_size = 1usize << total_vars;
        let mut bit_evals = vec![GoldilocksField(0); total_size];

        for i in 0..n {
            let x_i = evals[i].0;
            if x_i < max_value {
                let value = x_i as u32;
                for bit in 0..32u32 {
                    if (value >> bit) & 1 == 1 {
                        bit_evals[i + (bit as usize) * n] = GoldilocksField(1);
                    }
                }
            } else {
                // KNOWN UNSOUND: values >= 2^32 get all-zero bits.
                // Bit reconstruction will yield mc=0 instead of the true input value.
                // The verifier's eval_to_check catches this (mc != input_eval).
                // FIX: increase BIT_TABLE_VARS from 5 → 6 (64-bit decomposition).
                log::warn!(
                    "NonNegative::run(): value {} at index {} exceeds 2^32; \
                     proof will be rejected by eval_to_check",
                    x_i, i
                );
            }
        }

        let bit_poly = DenseMLPoly::new(total_vars, bit_evals);

        vec![Witness::new_dense_poly(
            input.shape.clone(),
            bit_poly,
            DataType::Uint,
            0,
            Role::Auxiliary,
        )]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let input_num_vars = get_n(&input.shape);
        let n = 1usize << input_num_vars;
        let total_size = 1usize << (input_num_vars + BIT_TABLE_VARS);

        let d_input = input.as_device_buf();
        let mut d_bits = DeviceBuffer::<u64>::new(total_size).expect("NonNegative: alloc");
        decompose_bits32(&d_input, &mut d_bits, n).expect("NonNegative: gpu kernel failed");

        let mut bit_shape = input.shape.clone();
        bit_shape.push(1 << BIT_TABLE_VARS);

        vec![Witness::new_device(
            bit_shape,
            Arc::new(d_bits),
            DataType::Uint,
            0,
            Role::Auxiliary,
        )]
    }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Range proof: pass through the claim
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
