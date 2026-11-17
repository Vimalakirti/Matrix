//! LLM-specific advice ops: [`RMSReciprocal`], [`DivConst`], [`SoftmaxConst`],
//! [`SigmoidConst`]. All produce auxiliary values whose soundness is enforced
//! by surrounding DAG operations (Einsum identities, ProductZeroCheck, etc.).

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int, get_n, int_to_f};

/// Public scalar normalization domain; larger scales require linked limbs.
pub const RMS_MEAN_MAX: i128 = (1i128 << 28) - 1;

pub fn floor_sqrt(n: i128) -> i128 {
    assert!(n >= 0);
    let (mut lo, mut hi) = (0i128, 1i128 << 63);
    while lo + 1 < hi {
        let mid = lo + (hi-lo)/2;
        if mid <= n / mid { lo=mid; } else { hi=mid; }
    }
    lo
}

/// Fixed private bucket layout: zero, 28 positive dyadic intervals, and
/// three forbidden padding positions. Every proof uses all 32 positions.
pub fn rms_bucket_bounds(sf: usize) -> Vec<(i128,i128,i128,i128)> {
    assert!(sf<=20,"exact scalar RMS supports activation scales 0..20; larger scales need wider arithmetic");
    let c=1i128 << (3*sf);
    let mut buckets=vec![(0,0,0,0)];
    for j in 0..28 {
        let lo=1i128<<j;let hi=(1i128<<(j+1))-1;
        let r=floor_sqrt(c/lo)+1;
        let u=floor_sqrt(c*hi)+1+(hi+1)/2;
        buckets.push((lo,hi,r,u));
    }
    buckets.resize(32,(-1,-1,0,0));
    buckets
}

/// Public independent bounds; the second-product bound is conditional on
/// the separately proved one-hot bucket, not on the desired final bracket.
pub fn rms_bounds(sf: usize) -> (i128,i128,i128) {
    let buckets=rms_bucket_bounds(sf);
    let r=buckets.iter().map(|b|b.2).max().unwrap();
    let u=buckets.iter().map(|b|b.3).max().unwrap();
    let c=1i128 << (3*sf);
    let d=buckets.iter().filter(|b|b.0>=0).map(|b|b.2*b.3+b.3+b.1/4).max().unwrap().max(c+u+1);
    assert!(RMS_MEAN_MAX*r < (1i128<<59) && d < (1i128<<61));
    (r,u,d)
}

/// Private fixed-shape one-hot advice. Its relation is defined by compile-time
/// bitness, row-sum and selected-interval roots, never by this honest hint.
#[derive(Clone,Debug)]
pub struct RMSBucket;
impl BasicBlock for RMSBucket {
    fn run(&self,inputs:&[&Witness])->Vec<Witness> {
        assert_eq!(inputs.len(),1);let x=inputs[0];assert_eq!(x.shape.len(),1);
        let len=x.shape[0];assert!(len.is_power_of_two());
        let mut values=vec![int_to_f(0);len*32];
        for (i,&v) in x.data.as_ref().unwrap().evaluations_ref().iter().enumerate() {
            let m=f_to_int(v);assert!((0..=RMS_MEAN_MAX).contains(&m));
            let bucket=if m==0 {0} else {128-(m as u128).leading_zeros() as usize};
            values[i+len*bucket]=int_to_f(1);
        }
        vec![Witness::new(vec![len,32],values,crate::dag::DataType::Bool,0,Role::Auxiliary)]
    }
    fn run_gpu(&self,inputs:&[&Witness])->Vec<Witness> {self.run(inputs)}
    fn prove(&self,_:&[&Witness],_:&[usize],_:&[&Claim],_:&mut Transcript)->(Vec<SumcheckProof>,Vec<Claim>) {(vec![],vec![])}
    fn verify(&self,_:&[&Witness],_:&[&Claim],_:&[&SumcheckProof],_:&mut Transcript)->bool {true}
}

/// Nearest nonnegative integer sqrt(Qs^3/m), midpoint ties upward.
/// The zero-statistic extension is exactly r=0, with epsilon exactly zero.
pub fn rounded_rms_reciprocal(m: i128, sf: usize) -> i128 {
    let (bound,_,_)=rms_bounds(sf);
    assert!((0..=RMS_MEAN_MAX).contains(&m), "RMS mean outside public domain");
    if m==0 { return 0; }
    let k=4i128 << (3*sf);
    let (mut lo,mut hi)=(0,bound+1);
    while lo+1<hi {
        let mid=lo+(hi-lo)/2;
        if m*(2*mid-1)*(2*mid-1)<=k {lo=mid;} else {hi=mid;}
    }
    lo
}

// ============================================================================
// RMSReciprocal: 1 / RMS(x) computed per (b, s, ..., :) row.
// ============================================================================

#[derive(Clone, Debug)]
pub struct RMSReciprocal {
    /// Dimension being reduced (= last axis of the input shape).
    pub dim: usize,
}

impl BasicBlock for RMSReciprocal {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "RMSReciprocal expects 1 input");
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let sf = (1u64 << input.sf) as f64;
        let mut out_shape = input.shape.clone();
        let ndim = out_shape.len();
        out_shape[ndim - 1] = 1;

        let padded: Vec<usize> = input.shape.iter().map(|&s| s.next_power_of_two()).collect();
        let d_pad = padded[ndim - 1];
        let stride_d: usize = padded[..ndim - 1].iter().product();
        let num_groups = stride_d;

        // Protocol-side `llama_rms_norm` computes the mean as
        //   x_mean = div_const(x_sum_sq, self.dim)
        // i.e. dividing by the *real* (un-padded) last-axis size. The
        // advice must use the same `self.dim` (NOT the padded count
        // `d_pad`); otherwise the reciprocity check `x_mean·r² ≈ 1`
        // is wrong by a factor of `d_pad / self.dim`, which trips the
        // NonNegative tolerance gate.
        if self.dim == 0 {
            let res: Vec<AlmostGoldilocksField> = evals.iter().map(|&v|
                int_to_f(rounded_rms_reciprocal(f_to_int(v), input.sf))
            ).collect();
            return vec![Witness::new(input.shape.clone(), res, input.data_type, input.sf, Role::Auxiliary)];
        }
        let n = self.dim as f64;
        let mut result = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let sum_sq: f64 = (0..d_pad)
                .map(|d| {
                    let idx = g + d * stride_d;
                    let x = f_to_int(evals[idx]) as f64 / sf;
                    x * x
                })
                .sum();
            let rms = (sum_sq / n).sqrt();
            let val = if rms == 0.0 { 0i128 } else { ((1.0 / rms) * sf).round() as i128 };
            result.push(int_to_f(val));
        }

        let n_out = get_n(&out_shape);
        result.resize(1 << n_out, AlmostGoldilocksField(0));
        vec![Witness::new(out_shape, result, input.data_type, input.sf, Role::Auxiliary)]
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

// ============================================================================
// DivConst: per-element integer division by a constant.
// ============================================================================

#[derive(Clone, Debug)]
pub struct DivConst {
    pub divisor: u64,
}

impl BasicBlock for DivConst {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "DivConst expects 1 input");
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        assert!(self.divisor > 0, "division by zero");
        let c = self.divisor as i128;
        let result: Vec<AlmostGoldilocksField> = evals
            .iter()
            .map(|&v| {
                let x_int = f_to_int(v);
                // Euclidean rounding, including negative exact half ties.
                let y_int = (x_int + c / 2).div_euclid(c);
                int_to_f(y_int)
            })
            .collect();
        vec![Witness::new(input.shape.clone(), result, input.data_type, input.sf, Role::Output)]
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

// ============================================================================
// SoftmaxConst: per-row stability constant for softmax.
// ============================================================================

#[derive(Clone, Debug)]
pub struct SoftmaxConst {
    pub dim: usize,
}

impl BasicBlock for SoftmaxConst {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "SoftmaxConst expects 1 input");
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let ndim = input.shape.len();
        let padded: Vec<usize> = input.shape.iter().map(|&s| s.next_power_of_two()).collect();
        let d_pad = padded[ndim - 1];
        let stride_d: usize = padded[..ndim - 1].iter().product();
        let num_groups = stride_d;
        let mut result = vec![AlmostGoldilocksField(0); evals.len()];
        for g in 0..num_groups {
            let mut max_val = i128::MIN;
            for d in 0..self.dim {
                let idx = g + d * stride_d;
                let v = f_to_int(evals[idx]);
                if v > max_val { max_val = v; }
            }
            let sf = (1u64 << input.sf) as f64;
            let sum_exp: f64 = (0..self.dim)
                .map(|d| {
                    let idx = g + d * stride_d;
                    let v = f_to_int(evals[idx]) as f64;
                    ((v - max_val as f64) / sf).exp()
                })
                .sum();
            let log_sum_exp = sum_exp.ln() * sf;
            let c = int_to_f(-(max_val + log_sum_exp.round() as i128));
            for d in 0..d_pad {
                let idx = g + d * stride_d;
                result[idx] = c;
            }
        }
        vec![Witness::new(input.shape.clone(), result, input.data_type, input.sf, Role::Output)]
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

// ============================================================================
// SigmoidConst: shift to convert exp-as-sigmoid into an exp lookup.
// ============================================================================

#[derive(Clone, Debug)]
pub struct SigmoidConst {
    pub segments: usize,
}

impl BasicBlock for SigmoidConst {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "SigmoidConst expects 1 input");
        let input = inputs[0];
        let evals = input.data.as_ref().unwrap().evaluations_ref();
        let sf = (1u64 << input.sf) as f64;
        let result: Vec<AlmostGoldilocksField> = evals
            .iter()
            .map(|&v| {
                let x_int = f_to_int(v) as f64;
                let x_f = x_int / sf;
                let t = -x_f;
                // softplus(t), numerically stable.
                let sp = if t > 20.0 { t } else if t < -20.0 { 0.0 } else { (1.0 + t.exp()).ln() };
                let c = -sf * sp - x_int;
                int_to_f(c.round() as i128)
            })
            .collect();
        vec![Witness::new(input.shape.clone(), result, input.data_type, input.sf, Role::Output)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DataType;

    fn make_witness(shape: Vec<usize>, vals: Vec<i128>, sf: usize) -> Witness {
        Witness::new(
            shape,
            vals.into_iter().map(int_to_f).collect(),
            DataType::Float,
            sf,
            Role::Input,
        )
    }

    #[test]
    fn rms_reciprocal_dim_1_uniform_input() {
        // x[0..4] = [1, 1, 1, 1] (in sf=0 units). RMS = 1. 1/RMS = 1.
        let x = make_witness(vec![4], vec![1, 1, 1, 1], 0);
        let r = RMSReciprocal { dim: 4 };
        let out = r.run(&[&x]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // First entry is 1, rest are padding-zero.
        assert_eq!(evals[0], int_to_f(1));
    }

    #[test]
    fn div_const_rounds_to_nearest() {
        // sf = 0; divide [10, 7, 3, 4] by 2 → [5, 4, 2, 2] (ties toward positive infinity).
        let x = make_witness(vec![4], vec![10, 7, 3, 4], 0);
        let d = DivConst { divisor: 2 };
        let out = d.run(&[&x]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // Positive half ties round upward; negative halves do too.
        assert_eq!(f_to_int(evals[0]), 5);
        assert_eq!(f_to_int(evals[1]), 4); // 7/2 = 3.5 → 4
        assert_eq!(f_to_int(evals[2]), 2); // 3/2 = 1.5 → 2
        assert_eq!(f_to_int(evals[3]), 2);
    }

    /// Softmax constant `c` is row-uniform — every position in the row gets
    /// the same correction. Verify uniformity.
    #[test]
    fn softmax_const_is_row_uniform() {
        // Row of 4 inputs: x = [0, 1, 2, 3].
        let x = make_witness(vec![4], vec![0, 1, 2, 3], 0);
        let s = SoftmaxConst { dim: 4 };
        let out = s.run(&[&x]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        for i in 1..4 {
            assert_eq!(evals[i], evals[0], "row {} != row 0", i);
        }
    }

    /// Sigmoid constant is per-element — `σ(x/sf) = exp((x + c(x))/sf)` so
    /// `c(x) + x ≤ 0` always (sigmoid is bounded above by 1, so log≤0).
    #[test]
    fn sigmoid_const_log_sigmoid_non_positive() {
        let x = make_witness(vec![4], vec![10, 0, -10, 100], 0);
        let s = SigmoidConst { segments: 1 };
        let out = s.run(&[&x]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // x + c == sf * log(σ(x/sf)) ≤ 0 for every i.
        for i in 0..4 {
            let combined = f_to_int(int_to_f(f_to_int(evals[i]) + f_to_int(x.data.as_ref().unwrap().index(i))));
            assert!(combined <= 0, "i = {}: x + c = {} > 0", i, combined);
        }
    }
}
