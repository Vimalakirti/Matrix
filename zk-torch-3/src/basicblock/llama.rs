use goldilocks_cuda::GoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};

/// RMSReciprocal: computes 1/RMS(x) for RMSNorm.
#[derive(Clone, Debug)]
pub struct RMSReciprocal {
    pub dim: usize,
}

impl BasicBlock for RMSReciprocal {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let sf = (1u64 << input.sf) as f64;

        // Output shape: input shape with last dim = 1
        let mut out_shape = input.shape.clone();
        let ndim = out_shape.len();
        out_shape[ndim - 1] = 1;

        // Compute padded sizes and strides (little-endian: first dim has stride 1)
        let padded: Vec<usize> = input.shape.iter().map(|&s| s.next_power_of_two()).collect();
        let d_pad = padded[ndim - 1]; // padded size of last dimension
        let stride_d: usize = padded[..ndim - 1].iter().product(); // stride for last dim
        let num_groups = stride_d; // number of (b, s, ...) groups

        let mut result = Vec::with_capacity(num_groups);

        for g in 0..num_groups {
            let n = d_pad as f64;
            let sum_sq: f64 = (0..d_pad)
                .map(|d| {
                    let idx = g + d * stride_d;
                    let x = f_to_int(evals[idx]) as f64 / sf;
                    x * x
                })
                .sum();

            let rms = (sum_sq / n).sqrt();
            let val = if rms == 0.0 {
                0i128
            } else {
                ((1.0 / rms) * sf).round() as i128
            };
            result.push(int_to_f(val));
        }

        let n_out = get_n(&out_shape);
        result.resize(1 << n_out, GoldilocksField(0));

        vec![Witness::new(
            out_shape,
            result,
            input.data_type,
            input.sf,
            Role::Auxiliary,
        )]
    }

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

/// DivConst: divides each element by a constant.
#[derive(Clone, Debug)]
pub struct DivConst {
    pub divisor: u64,
}

impl BasicBlock for DivConst {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let c = self.divisor as f64;

        let result: Vec<GoldilocksField> = evals
            .iter()
            .map(|&v| {
                let x_int = f_to_int(v) as f64;
                let y_int = (x_int / c).round() as i128;
                int_to_f(y_int)
            })
            .collect();

        vec![Witness::new(
            input.shape.clone(),
            result,
            input.data_type,
            input.sf,
            Role::Output,
        )]
    }

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

/// SoftmaxConst: piecewise linear approximation of softmax.
#[derive(Clone, Debug)]
pub struct SoftmaxConst {
    pub dim: usize,
}

impl BasicBlock for SoftmaxConst {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();

        // Softmax constant: subtract per-row max for numerical stability.
        // softmax(x)_i = exp(x_i - max(row)) / sum(exp(x_j - max(row)))
        // softmax_c[i] = -max(row) for all i in the same row (last dim group).
        let ndim = input.shape.len();
        let padded: Vec<usize> = input.shape.iter().map(|&s| s.next_power_of_two()).collect();
        let d_pad = padded[ndim - 1];
        let stride_d: usize = padded[..ndim - 1].iter().product();
        let num_groups = stride_d;

        let mut result = vec![GoldilocksField(0); evals.len()];

        for g in 0..num_groups {
            // Find max over real (non-padding) elements in last dim
            let mut max_val = i128::MIN;
            for d in 0..self.dim {
                let idx = g + d * stride_d;
                let val = f_to_int(evals[idx]);
                if val > max_val {
                    max_val = val;
                }
            }

            // Compute log-sum-exp normalization:
            // softmax(x_i) = exp(x_i - max - log(sum(exp(x_j - max))))
            // so softmax_c = -(max + log(sum(exp(x_j - max))))
            let sf = (1u64 << input.sf) as f64;
            let sum_exp: f64 = (0..self.dim)
                .map(|d| {
                    let idx = g + d * stride_d;
                    let val = f_to_int(evals[idx]) as f64;
                    ((val - max_val as f64) / sf).exp()
                })
                .sum();
            let log_sum_exp = sum_exp.ln() * sf;
            let c = int_to_f(-(max_val + log_sum_exp.round() as i128));
            for d in 0..d_pad {
                let idx = g + d * stride_d;
                result[idx] = c;
            }
        }

        vec![Witness::new(
            input.shape.clone(),
            result,
            input.data_type,
            input.sf,
            Role::Output,
        )]
    }

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

/// SigmoidConst: piecewise linear approximation of sigmoid.
#[derive(Clone, Debug)]
pub struct SigmoidConst {
    pub segments: usize,
}

impl BasicBlock for SigmoidConst {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let sf = (1u64 << input.sf) as f64;

        // Sigmoid constant: c such that exp((x+c)/sf) = σ(x/sf).
        // σ(t) = 1/(1+exp(-t)), so ln(σ(t)) = -ln(1+exp(-t)) = -softplus(-t).
        // We need (x+c)/sf = ln(σ(x/sf)), so c = sf * ln(σ(x/sf)) - x
        //                                        = -sf * softplus(-x/sf) - x.
        let result: Vec<GoldilocksField> = evals
            .iter()
            .map(|&v| {
                let x_int = f_to_int(v) as f64;
                let x_f = x_int / sf;
                // softplus(-x_f) = ln(1 + exp(-x_f)), numerically stable
                let t = -x_f;
                let sp = if t > 20.0 {
                    t
                } else if t < -20.0 {
                    0.0
                } else {
                    (1.0_f64 + t.exp()).ln()
                };
                let c = -sf * sp - x_int;
                int_to_f(c.round() as i128)
            })
            .collect();

        vec![Witness::new(
            input.shape.clone(),
            result,
            input.data_type,
            input.sf,
            Role::Output,
        )]
    }

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
