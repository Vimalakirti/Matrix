use goldilocks_cuda::{GoldilocksField, GOLDILOCKS_PRIME};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{get_n, log2_ceil};

/// MaxPoolHelper: advice op that computes 2×2 max pooling.
/// Input: X[C, H, W] → Output: Y[C, H/pool_h, W/pool_w]
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
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let h_out = self.input_h / self.pool_h;
        let w_out = self.input_w / self.pool_w;
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut y_data = vec![GoldilocksField(0); out_size];

        let p = GOLDILOCKS_PRIME;

        for c in 0..self.channels {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut max_val: u64 = 0;
                    let mut first = true;
                    for ph in 0..self.pool_h {
                        for pw in 0..self.pool_w {
                            let ih = ho * self.pool_h + ph;
                            let iw = wo * self.pool_w + pw;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let v = x.data.as_ref().unwrap().index(x_idx).0;
                            if first {
                                max_val = v;
                                first = false;
                            } else {
                                let v_signed = if v > p / 2 { v as i64 - p as i64 } else { v as i64 };
                                let max_signed = if max_val > p / 2 { max_val as i64 - p as i64 } else { max_val as i64 };
                                if v_signed > max_signed {
                                    max_val = v;
                                }
                            }
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                    y_data[out_idx] = GoldilocksField(max_val);
                }
            }
        }

        let out_shape = vec![self.channels, h_out, w_out];
        let n = get_n(&out_shape);
        if y_data.len() < (1 << n) {
            y_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, y_data, DataType::Uint, 0, Role::Output)]
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

/// GeneralMaxPoolHelper: advice op for general max pooling (arbitrary kernel and stride).
/// Input: X[C, H, W] → Output: Y[C, H_out, W_out]
/// where H_out = (H - kernel_h) / stride_h + 1, W_out = (W - kernel_w) / stride_w + 1.
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
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let h_out = (self.input_h - self.kernel_h) / self.stride_h + 1;
        let w_out = (self.input_w - self.kernel_w) / self.stride_w + 1;
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut y_data = vec![GoldilocksField(0); out_size];

        let p = GOLDILOCKS_PRIME;

        for c in 0..self.channels {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut max_val: u64 = 0;
                    let mut first = true;
                    for kh in 0..self.kernel_h {
                        for kw in 0..self.kernel_w {
                            let ih = ho * self.stride_h + kh;
                            let iw = wo * self.stride_w + kw;
                            let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                            let v = x.data.as_ref().unwrap().index(x_idx).0;
                            if first {
                                max_val = v;
                                first = false;
                            } else {
                                let v_signed = if v > p / 2 { v as i64 - p as i64 } else { v as i64 };
                                let max_signed = if max_val > p / 2 { max_val as i64 - p as i64 } else { max_val as i64 };
                                if v_signed > max_signed {
                                    max_val = v;
                                }
                            }
                        }
                    }
                    let out_idx = wo + ho * w_out_pad + c * w_out_pad * h_out_pad;
                    y_data[out_idx] = GoldilocksField(max_val);
                }
            }
        }

        let out_shape = vec![self.channels, h_out, w_out];
        let n = get_n(&out_shape);
        if y_data.len() < (1 << n) {
            y_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, y_data, DataType::Uint, 0, Role::Output)]
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

/// Replicate2x2: replicates Y[C, H/2, W/2] to Y_rep[C, H, W].
/// Each Y_rep[c, h, w] = Y[c, h/2, w/2].
/// Proof: claim transformation — drop bit-0 of spatial dims.
/// Y_rep MLE is independent of bit-0 in w and h dims (replication).
#[derive(Clone, Debug)]
pub struct Replicate2x2 {
    pub channels: usize,
    pub out_h: usize, // full height H
    pub out_w: usize, // full width W
}

impl BasicBlock for Replicate2x2 {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let y = inputs[0];

        let h_half = self.out_h / 2;
        let w_half = self.out_w / 2;
        let w_half_pad = w_half.next_power_of_two();
        let h_half_pad = h_half.next_power_of_two();
        let w_out_pad = self.out_w.next_power_of_two();
        let h_out_pad = self.out_h.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut rep_data = vec![GoldilocksField(0); out_size];

        for c in 0..self.channels {
            for h in 0..self.out_h {
                for w in 0..self.out_w {
                    let ho = h / 2;
                    let wo = w / 2;
                    let y_idx = wo + ho * w_half_pad + c * w_half_pad * h_half_pad;
                    let rep_idx = w + h * w_out_pad + c * w_out_pad * h_out_pad;
                    rep_data[rep_idx] = y.data.as_ref().unwrap().index(y_idx);
                }
            }
        }

        let out_shape = vec![self.channels, self.out_h, self.out_w];
        let n = get_n(&out_shape);
        if rep_data.len() < (1 << n) {
            rep_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, rep_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        _witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        _transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        // Claim transformation: Y_rep claim → Y claim by dropping bit-0 of w and h dims
        let claim = out_claims[0];
        let y_edge = edge_ids[0];

        let l_w_out = log2_ceil(self.out_w.max(1));
        let l_h_out = log2_ceil(self.out_h.max(1));
        let l_c = log2_ceil(self.channels.max(1));

        // Y_rep point (little-endian): [w0, w1, ..., w_{l_w-1}, h0, h1, ..., h_{l_h-1}, c0, ..., c_{l_c-1}]
        // Y point: [w1, ..., w_{l_w-1}, h1, ..., h_{l_h-1}, c0, ..., c_{l_c-1}]
        let mut y_point = Vec::with_capacity(l_w_out - 1 + l_h_out - 1 + l_c);
        // w dims: skip index 0 (bit-0 of w)
        y_point.extend_from_slice(&claim.point[1..l_w_out]);
        // h dims: skip index l_w_out (bit-0 of h)
        y_point.extend_from_slice(&claim.point[l_w_out + 1..l_w_out + l_h_out]);
        // c dims: keep all
        y_point.extend_from_slice(&claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c]);

        // Y_rep(r) = Y(drop_bit0(r)), so eval is the same
        let y_claim = Claim {
            edge_id: y_edge,
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
        // claims: [y_claim, out_claim (Y_rep)]
        let out_claim = claims.last().unwrap();
        let y_claim = &claims[0];

        // Check eval equality
        if y_claim.eval != out_claim.eval {
            println!("Replicate2x2: eval mismatch");
            return false;
        }

        let l_w_out = log2_ceil(self.out_w.max(1));
        let l_h_out = log2_ceil(self.out_h.max(1));
        let l_c = log2_ceil(self.channels.max(1));

        // Check point transformation: y_claim.point == drop_bit0(out_claim.point)
        let expected_len = l_w_out - 1 + l_h_out - 1 + l_c;
        if y_claim.point.len() != expected_len {
            println!("Replicate2x2: point length mismatch");
            return false;
        }

        // Check w dims (skip bit-0)
        if y_claim.point[..l_w_out - 1] != out_claim.point[1..l_w_out] {
            println!("Replicate2x2: w dim mismatch");
            return false;
        }
        // Check h dims (skip bit-0)
        let h_start = l_w_out - 1;
        if y_claim.point[h_start..h_start + l_h_out - 1] != out_claim.point[l_w_out + 1..l_w_out + l_h_out] {
            println!("Replicate2x2: h dim mismatch");
            return false;
        }
        // Check c dims
        let c_start = h_start + l_h_out - 1;
        if y_claim.point[c_start..] != out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c] {
            println!("Replicate2x2: c dim mismatch");
            return false;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{Witness, DataType, Role};
    use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_maxpool_run() {
        // X[1, 4, 4] → Y[1, 2, 2] with 2×2 max pooling
        let mp = MaxPoolHelper {
            channels: 1, input_h: 4, input_w: 4, pool_h: 2, pool_w: 2,
        };
        // X = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        let x = make_witness(vec![1, 4, 4], vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ]);
        let result = mp.run(&[&x]);
        let y = &result[0];

        assert_eq!(y.shape, vec![1, 2, 2]);
        // Block (0,0): max(1,2,5,6) = 6
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(6));
        // Block (1,0): max(3,4,7,8) = 8
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(8));
        // Block (0,1): max(9,10,13,14) = 14
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(14));
        // Block (1,1): max(11,12,15,16) = 16
        assert_eq!(y.data.as_ref().unwrap().index(3), GoldilocksField(16));
    }

    #[test]
    fn test_replicate_run() {
        // Y[1, 2, 2] → Y_rep[1, 4, 4]
        let rep = Replicate2x2 { channels: 1, out_h: 4, out_w: 4 };
        let y = make_witness(vec![1, 2, 2], vec![6, 8, 14, 16]);
        let result = rep.run(&[&y]);
        let y_rep = &result[0];

        assert_eq!(y_rep.shape, vec![1, 4, 4]);
        // Y_rep[0,0] = Y[0,0] = 6, Y_rep[1,0] = Y[0,0] = 6, Y_rep[0,1] = Y[0,0] = 6, Y_rep[1,1] = Y[0,0] = 6
        assert_eq!(y_rep.data.as_ref().unwrap().index(0), GoldilocksField(6)); // (0,0)
        assert_eq!(y_rep.data.as_ref().unwrap().index(1), GoldilocksField(6)); // (1,0)
        assert_eq!(y_rep.data.as_ref().unwrap().index(4), GoldilocksField(6)); // (0,1) = idx w=0 + h=1*4
        assert_eq!(y_rep.data.as_ref().unwrap().index(5), GoldilocksField(6)); // (1,1)
        assert_eq!(y_rep.data.as_ref().unwrap().index(2), GoldilocksField(8)); // (2,0) → Y[1,0]
        assert_eq!(y_rep.data.as_ref().unwrap().index(3), GoldilocksField(8)); // (3,0) → Y[1,0]
    }

    #[test]
    fn test_replicate_prove_verify() {
        let rep = Replicate2x2 { channels: 1, out_h: 4, out_w: 4 };
        let y = make_witness(vec![1, 2, 2], vec![6, 8, 14, 16]);
        let result = rep.run(&[&y]);
        let y_rep = &result[0];

        // Create claim on Y_rep
        let n_rep = y_rep.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_rep");
        let point: Vec<GoldilocksExt2> = (0..n_rep)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y_rep.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let rep_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_rep_prove");
        let (proofs, claims) = rep.prove(
            &[&y, y_rep],
            &[0, 1],
            &[&rep_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 0);
        assert_eq!(claims.len(), 1);

        // Check that Y claim eval matches Y evaluated at the transformed point
        let y_eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert_eq!(claims[0].eval, y_eval, "Y claim eval should match Y poly evaluation");

        // Verify
        let mut verify_transcript = Transcript::new(b"test_rep_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&rep_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = rep.verify(
            &[&y, y_rep],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "Replicate2x2 prove/verify should pass");
    }
}
