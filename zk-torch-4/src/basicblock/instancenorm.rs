//! [`InstanceNormHelper`] — per-channel scale + offset advice for 3D
//! instance normalization.
//!
//! Output is a packed `[2, C]` witness: row 0 = scale, row 1 = offset.
//! Soundness comes from the downstream `Y = scale · X + offset` decomposition
//! the DAG builder wires (Einsum + Add, both fully proven).

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};

#[derive(Clone, Debug)]
pub struct InstanceNormHelper {
    pub channels: usize,
    pub depth: usize,
    pub height: usize,
    pub width: usize,
    pub eps: f64,
}

impl BasicBlock for InstanceNormHelper {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 3, "InstanceNormHelper expects 3 inputs (X, γ, β)");
        let x = inputs[0];
        let gamma = inputs[1];
        let beta = inputs[2];

        let sf = (1u64 << x.sf) as f64;
        let w_pad = self.width.next_power_of_two();
        let h_pad = self.height.next_power_of_two();
        let d_pad = self.depth.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let spatial_size = self.depth * self.height * self.width;
        let spatial_pad = d_pad * h_pad * w_pad;

        let x_evals = x.data.as_ref().unwrap().evaluations_ref();
        let gamma_evals = gamma.data.as_ref().unwrap().evaluations_ref();
        let beta_evals = beta.data.as_ref().unwrap().evaluations_ref();

        // Packed [2, C] little-endian: scale at idx c, offset at idx c + c_pad.
        let packed_size = 2 * c_pad;
        let mut packed = vec![AlmostGoldilocksField(0); packed_size];

        for c in 0..self.channels {
            let mut sum = 0.0f64;
            for d in 0..self.depth {
                for h in 0..self.height {
                    for w in 0..self.width {
                        let idx = w + h * w_pad + d * w_pad * h_pad + c * spatial_pad;
                        sum += f_to_int(x_evals[idx]) as f64 / sf;
                    }
                }
            }
            let mean = sum / spatial_size as f64;

            let mut var_sum = 0.0f64;
            for d in 0..self.depth {
                for h in 0..self.height {
                    for w in 0..self.width {
                        let idx = w + h * w_pad + d * w_pad * h_pad + c * spatial_pad;
                        let v = f_to_int(x_evals[idx]) as f64 / sf - mean;
                        var_sum += v * v;
                    }
                }
            }
            let var = var_sum / spatial_size as f64;

            let inv_std = 1.0 / (var + self.eps).sqrt();
            let g = f_to_int(gamma_evals[c]) as f64 / sf;
            let b = f_to_int(beta_evals[c]) as f64 / sf;
            let scale = g * inv_std;
            let offset = b - scale * mean;
            packed[c] = int_to_f((scale * sf).round() as i128);
            packed[c + c_pad] = int_to_f((offset * sf).round() as i128);
        }

        let packed_shape = vec![2, self.channels];
        let n = get_n(&packed_shape);
        if packed.len() < (1 << n) {
            packed.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(packed_shape, packed, x.data_type, x.sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        (vec![], vec![])
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

/// `InstanceNorm3D` was the zk-torch-3 name for the same op; keep as an
/// alias so downstream consumers compile unchanged.
pub type InstanceNorm3D = InstanceNormHelper;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn make_witness(shape: Vec<usize>, vals: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(agl).collect(),
            DataType::Uint,
            0,
            Role::Input,
        )
    }

    /// Constant input (var = 0) — scale collapses to γ/√ε ≈ 316 (at γ=1).
    /// Output shape is `[2, C]` packed.
    #[test]
    fn instancenorm_constant_input_produces_2c_packed_output() {
        let norm = InstanceNormHelper {
            channels: 1, depth: 2, height: 2, width: 2, eps: 1e-5,
        };
        let x = make_witness(vec![1, 2, 2, 2], vec![5, 5, 5, 5, 5, 5, 5, 5]);
        let gamma = make_witness(vec![1], vec![1]);
        let beta = make_witness(vec![1], vec![0]);
        let out = norm.run(&[&x, &gamma, &beta]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![2, 1]);
        let data = out[0].data.as_ref().unwrap();
        // Both slots present (group 0 + group 1).
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn instancenorm_prove_is_empty_advice() {
        let norm = InstanceNormHelper {
            channels: 1, depth: 2, height: 2, width: 2, eps: 1e-5,
        };
        let x = make_witness(vec![1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let gamma = make_witness(vec![1], vec![1]);
        let beta = make_witness(vec![1], vec![0]);
        let mut t = Transcript::new(b"in");
        let (proofs, claims) = norm.prove(&[&x, &gamma, &beta], &[0, 1, 2], &[], &mut t);
        assert!(proofs.is_empty());
        assert!(claims.is_empty());
    }
}
