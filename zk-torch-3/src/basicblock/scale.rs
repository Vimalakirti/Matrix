use std::sync::Arc;

use goldilocks_cuda::{DeviceBuffer, GoldilocksField};
use goldilocks_cuda::bit_decomp::{memset_zero, scale_down as gpu_scale_down, scale_up as gpu_scale_up};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::dense::DenseMLPoly;
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};
use crate::SF_INT;

/// Number of table variables for the bit decomposition (2^5 = 32 bits → 5 vars).
///
/// SOUNDNESS LIMITATION: 32 bits limits NonNegative range proofs to values < 2^32.
/// Values >= 2^32 produce all-zero bit witnesses, causing the verifier's eval_to_check
/// to reject the proof. To support the full Goldilocks field range [0, p/2], increase
/// this to 6 (64 bits). This doubles the bit polynomial size and adds one sumcheck round.
pub const BIT_TABLE_VARS: usize = 5;

/// ScaleDown block: divide each element by scale factor (integer division).
/// Produces main output + auxiliary bit decomposition witness.
#[derive(Clone, Debug)]
pub struct ScaleDown {
    /// Target output scale factor (set by DagBuilder::scale).
    pub output_sf: usize,
}

impl BasicBlock for ScaleDown {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let sf = *SF_INT;
        let input_num_vars = get_n(&input.shape);

        let n = 1usize << input_num_vars;
        let mut quotients = Vec::with_capacity(n);

        // Build dense bit polynomial B(x,y) with input_num_vars + 5 variables
        let total_vars = input_num_vars + BIT_TABLE_VARS;
        let total_size = 1usize << total_vars;
        let mut bit_evals = vec![GoldilocksField(0); total_size];

        for i in 0..n {
            let int_val = f_to_int(evals[i]);
            let q = if int_val >= 0 {
                int_val / sf as i128
            } else {
                -(((-int_val + sf as i128 - 1) / sf as i128))
            };
            let r = int_val - q * sf as i128;
            quotients.push(int_to_f(q));
            // r is in [0, sf), decompose into bits
            let value = r as u32;
            for bit in 0..32u32 {
                if (value >> bit) & 1 == 1 {
                    bit_evals[i + (bit as usize) * n] = GoldilocksField(1);
                }
            }
        }

        let bit_poly = DenseMLPoly::new(total_vars, bit_evals);

        vec![
            Witness::new(input.shape.clone(), quotients, input.data_type, self.output_sf, Role::Output),
            Witness::new_dense_poly(input.shape.clone(), bit_poly, DataType::Uint, 0, Role::Auxiliary),
        ]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let input_num_vars = get_n(&input.shape);
        let n = 1usize << input_num_vars;
        let sf = *SF_INT as u64;

        let total_vars = input_num_vars + BIT_TABLE_VARS;
        let total_size = 1usize << total_vars; // = n * 32

        let d_input = input.as_device_buf();
        let mut d_quotients = DeviceBuffer::<u64>::new(n).expect("ScaleDown: quotients alloc");
        let mut d_bits = DeviceBuffer::<u64>::new(total_size).expect("ScaleDown: bits alloc");
        gpu_scale_down(&d_input, &mut d_quotients, &mut d_bits, n, sf)
            .expect("ScaleDown: gpu kernel failed");

        // Build a device-resident bit witness with shape `[input_shape, 32]`
        // (32 = 1 << BIT_TABLE_VARS). Caller treats it as Auxiliary.
        let mut bit_shape = input.shape.clone();
        bit_shape.push(1 << BIT_TABLE_VARS);

        vec![
            Witness::new_device(input.shape.clone(), Arc::new(d_quotients), input.data_type, self.output_sf, Role::Output),
            Witness::new_device(bit_shape, Arc::new(d_bits), DataType::Uint, 0, Role::Auxiliary),
        ]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Pass claim through: evaluate input at the output claim point
        let inp_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: out_claims[0].point.clone(),
            eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&out_claims[0].point),
        };
        (vec![], vec![inp_claim])
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

/// ScaleUp block: multiply each element by scale factor.
#[derive(Clone, Debug)]
pub struct ScaleUp {
    /// Target output scale factor (set by DagBuilder::scale).
    pub output_sf: usize,
}

impl BasicBlock for ScaleUp {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let sf = *SF_INT;
        let input_num_vars = get_n(&input.shape);

        let n = 1usize << input_num_vars;
        let mut scaled = Vec::with_capacity(n);

        // Build dense bit polynomial B(x,y) with input_num_vars + 5 variables
        let total_vars = input_num_vars + BIT_TABLE_VARS;
        let total_size = 1usize << total_vars;
        let bit_evals = vec![GoldilocksField(0); total_size];

        for i in 0..n {
            let x_int = f_to_int(evals[i]);
            let y_int = x_int * sf as i128;
            scaled.push(int_to_f(y_int));
            // Remainder is always 0 for ScaleUp (exact scaling)
            // All bits are 0, so bit_evals stays zero for this position
        }

        let bit_poly = DenseMLPoly::new(total_vars, bit_evals);

        vec![
            Witness::new(input.shape.clone(), scaled, input.data_type, self.output_sf, Role::Output),
            Witness::new_dense_poly(input.shape.clone(), bit_poly, DataType::Uint, 0, Role::Auxiliary),
        ]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let input_num_vars = get_n(&input.shape);
        let n = 1usize << input_num_vars;
        let sf = *SF_INT as u64;

        let total_size = 1usize << (input_num_vars + BIT_TABLE_VARS);

        let d_input = input.as_device_buf();
        let mut d_scaled = DeviceBuffer::<u64>::new(n).expect("ScaleUp: alloc");
        gpu_scale_up(&d_input, &mut d_scaled, n, sf).expect("ScaleUp: gpu kernel failed");

        let mut d_bits = DeviceBuffer::<u64>::new(total_size).expect("ScaleUp: bits alloc");
        memset_zero(&mut d_bits, total_size).expect("ScaleUp: bits memset failed");

        let mut bit_shape = input.shape.clone();
        bit_shape.push(1 << BIT_TABLE_VARS);

        vec![
            Witness::new_device(input.shape.clone(), Arc::new(d_scaled), input.data_type, self.output_sf, Role::Output),
            Witness::new_device(bit_shape, Arc::new(d_bits), DataType::Uint, 0, Role::Auxiliary),
        ]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Pass claim through: evaluate input at the output claim point
        let claim = vec![Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: out_claims[0].point.clone(),
            eval: witnesses[0].data.as_ref().unwrap().evaluate_at_point_ext2(&out_claims[0].point),
        }];
        (vec![], claim)
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
