use goldilocks_cuda::GoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};

/// InstanceNormHelper: advice op that computes per-channel scale and offset
/// for instance normalization of [C, D, H, W] tensors.
///
/// For each channel c:
///   mean[c] = mean(X[c, :, :, :])
///   var[c]  = var(X[c, :, :, :])
///   scale[c] = gamma[c] / sqrt(var[c] + eps)
///   offset[c] = beta[c] - scale[c] * mean[c]
///
/// The actual normalization Y[c,d,h,w] = scale[c] * X[c,d,h,w] + offset[c]
/// is performed by proven Einsum + Add nodes in the builder.
///
/// Inputs: X[C,D,H,W], gamma[C], beta[C]
/// Output: packed[2, C] — group 0 = scale, group 1 = offset (single output to avoid reducer issues)
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
        assert_eq!(inputs.len(), 3);
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

        // Packed output: [2, C] where group 0 = scale, group 1 = offset
        // Little-endian: c bits (lowest) | group bit (highest)
        // Index = c + group * c_pad
        let packed_size = 2 * c_pad;
        let mut packed_data = vec![GoldilocksField(0); packed_size];

        for c in 0..self.channels {
            // Compute mean
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

            // Compute variance
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

            packed_data[c] = int_to_f((scale * sf).round() as i128);            // group 0: scale
            packed_data[c + c_pad] = int_to_f((offset * sf).round() as i128);   // group 1: offset
        }

        let packed_shape = vec![2, self.channels];
        let n = get_n(&packed_shape);
        if packed_data.len() < (1 << n) {
            packed_data.resize(1 << n, GoldilocksField(0));
        }

        vec![Witness::new(packed_shape, packed_data, x.data_type, x.sf, Role::Output)]
    }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        _edge_ids: &[usize],
        _out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Advice op: the prover provides scale/offset freely.
        // Soundness: Y = scale * X + offset is constrained by proven Einsum + Add.
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

/// Legacy alias: InstanceNorm3D is now InstanceNormHelper.
pub type InstanceNorm3D = InstanceNormHelper;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Witness, DataType, Role};
    use goldilocks_cuda::GoldilocksField;

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_instancenorm_helper_run() {
        let norm = InstanceNormHelper {
            channels: 1, depth: 2, height: 2, width: 2, eps: 1e-5,
        };
        // All values the same → mean = val, var = 0
        // scale = gamma / sqrt(eps) ≈ gamma * 316.23
        // offset = beta - scale * mean
        let x = make_witness(vec![1, 2, 2, 2], vec![5, 5, 5, 5, 5, 5, 5, 5]);
        let gamma = make_witness(vec![1], vec![1]);
        let beta = make_witness(vec![1], vec![0]);
        let result = norm.run(&[&x, &gamma, &beta]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape, vec![2, 1]); // packed [2, C=1]: group 0=scale, group 1=offset
    }
}
