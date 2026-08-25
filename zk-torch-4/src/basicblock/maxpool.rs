//! Max-pool advice ops and the [`Replicate2x2`] claim transform.
//!
//! [`MaxPoolHelper`] (2×2) and [`GeneralMaxPoolHelper`] (arbitrary kernel/stride)
//! emit advice outputs whose soundness is enforced by the surrounding DAG:
//! Replicate2x2 + Sub + NonNeg proves `Y ≥ X` at every pool position, which
//! upper-bounds the true max (dominance). True achievability requires a
//! lookup/selection argument — out of scope for the initial port.
//!
//! [`Replicate2x2`] is a pure claim transform — drop bit-0 of W and H in the
//! evaluation point. No sumcheck.

use almost_goldilocks_cuda::field::{AlmostGoldilocksField, ALMOST_GOLDILOCKS_PRIME};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_field_eq, get_n, log2_ceil};

// ---------------------------------------------------------------------------
// Signed comparison helper: treat values > q/2 as negatives.
// ---------------------------------------------------------------------------
fn signed_max(a: u64, b: u64) -> u64 {
    let half_q = ALMOST_GOLDILOCKS_PRIME / 2;
    let a_signed = if a > half_q { a as i128 - ALMOST_GOLDILOCKS_PRIME as i128 } else { a as i128 };
    let b_signed = if b > half_q { b as i128 - ALMOST_GOLDILOCKS_PRIME as i128 } else { b as i128 };
    if a_signed >= b_signed { a } else { b }
}

// ============================================================================
// MaxPoolHelper (regular 2×2)
// ============================================================================

#[derive(Clone, Debug)]
pub struct MaxPoolHelper {
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub pool_h: usize,
    pub pool_w: usize,
}

impl BasicBlock for MaxPoolHelper {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "MaxPoolHelper expects 1 input");
        let x = inputs[0];
        let h_out = self.input_h / self.pool_h;
        let w_out = self.input_w / self.pool_w;
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut y_data = vec![AlmostGoldilocksField(0); out_size];
        let data = x.data.as_ref().unwrap();
        for c in 0..self.channels {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc: Option<u64> = None;
                    for ph in 0..self.pool_h {
                        for pw in 0..self.pool_w {
                            let ih = ho * self.pool_h + ph;
                            let iw = wo * self.pool_w + pw;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let v = data.index(x_idx).reduce().0;
                            acc = Some(match acc {
                                None => v,
                                Some(m) => signed_max(m, v),
                            });
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                    y_data[out_idx] = AlmostGoldilocksField(acc.unwrap_or(0));
                }
            }
        }
        let out_shape = vec![self.channels, h_out, w_out];
        let n = get_n(&out_shape);
        if y_data.len() < (1 << n) {
            y_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, y_data, DataType::Uint, inputs[0].sf, Role::Output)]
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
// GeneralMaxPoolHelper
// ============================================================================

#[derive(Clone, Debug)]
pub struct GeneralMaxPoolHelper {
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

impl BasicBlock for GeneralMaxPoolHelper {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "GeneralMaxPoolHelper expects 1 input");
        let x = inputs[0];
        let h_out = (self.input_h - self.kernel_h) / self.stride_h + 1;
        let w_out = (self.input_w - self.kernel_w) / self.stride_w + 1;
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut y_data = vec![AlmostGoldilocksField(0); out_size];
        let data = x.data.as_ref().unwrap();
        for c in 0..self.channels {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc: Option<u64> = None;
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let ih = ho * self.stride_h + kh;
                            let iw = wo * self.stride_w + kw;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let v = data.index(x_idx).reduce().0;
                            acc = Some(match acc {
                                None => v,
                                Some(m) => signed_max(m, v),
                            });
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                    y_data[out_idx] = AlmostGoldilocksField(acc.unwrap_or(0));
                }
            }
        }
        let out_shape = vec![self.channels, h_out, w_out];
        let n = get_n(&out_shape);
        if y_data.len() < (1 << n) {
            y_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, y_data, DataType::Uint, inputs[0].sf, Role::Output)]
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
// Replicate2x2
// ============================================================================

#[derive(Clone, Debug)]
pub struct Replicate2x2 {
    pub channels: usize,
    pub out_h: usize,
    pub out_w: usize,
}

impl BasicBlock for Replicate2x2 {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "Replicate2x2 expects 1 input");
        let y = inputs[0];
        let h_half = self.out_h / 2;
        let w_half = self.out_w / 2;
        let w_half_pad = w_half.next_power_of_two();
        let h_half_pad = h_half.next_power_of_two();
        let w_out_pad = self.out_w.next_power_of_two();
        let h_out_pad = self.out_h.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut rep_data = vec![AlmostGoldilocksField(0); out_size];
        let data = y.data.as_ref().unwrap();
        for c in 0..self.channels {
            for h in 0..self.out_h {
                for w in 0..self.out_w {
                    let ho = h / 2;
                    let wo = w / 2;
                    let y_idx = wo + ho * w_half_pad + c * w_half_pad * h_half_pad;
                    let rep_idx = w + h * w_out_pad + c * w_out_pad * h_out_pad;
                    rep_data[rep_idx] = data.index(y_idx);
                }
            }
        }
        let out_shape = vec![self.channels, self.out_h, self.out_w];
        let n = get_n(&out_shape);
        if rep_data.len() < (1 << n) {
            rep_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, rep_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let claim = out_claims[0];
        let l_w_out = log2_ceil(self.out_w.max(1));
        let l_h_out = log2_ceil(self.out_h.max(1));
        let l_c = log2_ceil(self.channels.max(1));

        // Drop bit-0 of W and bit-0 of H from the output point.
        let mut y_point = Vec::with_capacity(l_w_out - 1 + l_h_out - 1 + l_c);
        y_point.extend_from_slice(&claim.point[1..l_w_out]);
        y_point.extend_from_slice(&claim.point[l_w_out + 1..l_w_out + l_h_out]);
        y_point.extend_from_slice(&claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c]);

        let y_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: y_point,
            eval: claim.eval,
        };
        (vec![], vec![y_claim])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        claims: &[&Claim],
        _sumcheck_proofs: &[&SumcheckProof],
        _transcript: &mut Transcript,
    ) -> bool {
        let out_claim = claims.last().unwrap();
        let y_claim = &claims[0];
        if !ext2_field_eq(y_claim.eval, out_claim.eval) {
            return false;
        }
        let l_w_out = log2_ceil(self.out_w.max(1));
        let l_h_out = log2_ceil(self.out_h.max(1));
        let l_c = log2_ceil(self.channels.max(1));
        let expected_len = l_w_out - 1 + l_h_out - 1 + l_c;
        if y_claim.point.len() != expected_len { return false; }
        if y_claim.point[..l_w_out - 1] != out_claim.point[1..l_w_out] { return false; }
        let h_start = l_w_out - 1;
        if y_claim.point[h_start..h_start + l_h_out - 1]
            != out_claim.point[l_w_out + 1..l_w_out + l_h_out]
        {
            return false;
        }
        let c_start = h_start + l_h_out - 1;
        if y_claim.point[c_start..] != out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c] {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn maxpool_2x2_takes_block_max() {
        let mp = MaxPoolHelper { channels: 1, input_h: 4, input_w: 4, pool_h: 2, pool_w: 2 };
        let x = make_witness(vec![1, 4, 4], (1..=16u64).collect());
        let y = mp.run(&[&x]);
        let evals = y[0].data.as_ref().unwrap().evaluations_ref();
        // Layout: w stride 1, h stride 4. Block (0,0) = max(X[0,0..1, 0..1]).
        // X is stored linearly: X[0]=1, X[1]=2, X[4]=5, X[5]=6 → block max = 6.
        assert_eq!(evals[0], agl(6));
        assert_eq!(evals[1], agl(8));  // block (1,0)
        assert_eq!(evals[2], agl(14)); // block (0,1)
        assert_eq!(evals[3], agl(16)); // block (1,1)
    }

    /// Signed-max convention: a "negative" Goldilocks value (rep > q/2) is
    /// less than any positive.
    #[test]
    fn maxpool_handles_negatives() {
        let mp = MaxPoolHelper { channels: 1, input_h: 2, input_w: 2, pool_h: 2, pool_w: 2 };
        // Block has 3, -1, 0, -7 — max is 3.
        let q = ALMOST_GOLDILOCKS_PRIME;
        let x = make_witness(vec![1, 2, 2], vec![3, q - 1, 0, q - 7]);
        let y = mp.run(&[&x]);
        assert_eq!(y[0].data.as_ref().unwrap().index(0), agl(3));
    }

    #[test]
    fn general_maxpool_with_stride_and_kernel() {
        // X[1, 4, 4], kernel 3x3, stride 1 → output 2x2.
        let mp = GeneralMaxPoolHelper {
            channels: 1, input_h: 4, input_w: 4,
            kernel_h: 3, kernel_w: 3, stride_h: 1, stride_w: 1,
        };
        let x = make_witness(vec![1, 4, 4], (1..=16u64).collect());
        let y = mp.run(&[&x]);
        // Block (0,0) covers rows 0-2, cols 0-2 → max = 11.
        // Block (1,0) covers rows 0-2, cols 1-3 → max = 12.
        let evals = y[0].data.as_ref().unwrap().evaluations_ref();
        assert_eq!(evals[0], agl(11));
        assert_eq!(evals[1], agl(12));
        assert_eq!(evals[2], agl(15)); // block (0,1)
        assert_eq!(evals[3], agl(16)); // block (1,1)
    }

    #[test]
    fn replicate_2x2_run_and_prove_verify() {
        let rep = Replicate2x2 { channels: 1, out_h: 4, out_w: 4 };
        let y = make_witness(vec![1, 2, 2], vec![6, 8, 14, 16]);
        let outs = rep.run(&[&y]);
        let y_rep = &outs[0];
        let evals = y_rep.data.as_ref().unwrap().evaluations_ref();
        // Y_rep[w, h, c] = Y[w/2, h/2, c].
        assert_eq!(evals[0], agl(6));  // w=0,h=0 → Y[0,0]=6
        assert_eq!(evals[1], agl(6));  // w=1,h=0 → Y[0,0]=6
        assert_eq!(evals[2], agl(8));  // w=2,h=0 → Y[1,0]=8
        assert_eq!(evals[3], agl(8));  // w=3,h=0 → Y[1,0]=8
        assert_eq!(evals[4 + 0], agl(6));  // w=0,h=1 → Y[0,0]=6
        assert_eq!(evals[4 + 4 + 0], agl(14)); // w=0,h=2 → Y[0,1]=14

        // Prove → verify roundtrip.
        let n_rep = y_rep.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"rep");
        let point: Vec<_> = (0..n_rep).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y_rep.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let rep_claim = Claim { edge_id: 1, sparse_id: 0, point, eval };
        let mut t_prove = Transcript::new(b"rep-prove");
        let (proofs, claims) = rep.prove(&[&y, y_rep], &[0, 1], &[&rep_claim], &mut t_prove);
        assert!(proofs.is_empty());
        let direct = y.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert!(ext2_field_eq(claims[0].eval, direct));
        let mut t_verify = Transcript::new(b"rep-prove");
        let all = [&claims[0], &rep_claim];
        assert!(rep.verify(&[&y, y_rep], &all, &[], &mut t_verify));
    }
}
