use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_mul, get_n, log2_ceil};

/// SubSample2D: extracts a strided subsample with offset from X[C, H, W].
/// SV[c, oh, ow] = X[c, oh*stride_h + off_h, ow*stride_w + off_w]
///
/// Proof: sumcheck over the input domain. Builds H[x] such that
/// Σ_x H(x) * X(x) = SV(r_out), then verifier checks H(r_in) * X(r_in) = final_eval.
#[derive(Clone, Debug)]
pub struct SubSample2D {
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub offset_h: usize,
    pub offset_w: usize,
}

impl SubSample2D {
    pub fn new(
        channels: usize,
        input_h: usize, input_w: usize,
        stride_h: usize, stride_w: usize,
        offset_h: usize, offset_w: usize,
    ) -> Self {
        let out_h = (input_h - offset_h) / stride_h;
        let out_w = (input_w - offset_w) / stride_w;
        Self { channels, input_h, input_w, out_h, out_w, stride_h, stride_w, offset_h, offset_w }
    }

    /// Create with explicit output size (used when all subsamples must have same output size).
    pub fn new_with_output_size(
        channels: usize,
        input_h: usize, input_w: usize,
        stride_h: usize, stride_w: usize,
        offset_h: usize, offset_w: usize,
        out_h: usize, out_w: usize,
    ) -> Self {
        Self { channels, input_h, input_w, out_h, out_w, stride_h, stride_w, offset_h, offset_w }
    }

    fn l_w_out(&self) -> usize { log2_ceil(self.out_w.max(1)) }
    fn l_h_out(&self) -> usize { log2_ceil(self.out_h.max(1)) }
    fn l_w_in(&self) -> usize { log2_ceil(self.input_w.max(1)) }
    fn l_h_in(&self) -> usize { log2_ceil(self.input_h.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.channels.max(1)) }
}

impl BasicBlock for SubSample2D {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.out_w.next_power_of_two();
        let h_out_pad = self.out_h.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for c in 0..self.channels {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    let ih = oh * self.stride_h + self.offset_h;
                    let iw = ow * self.stride_w + self.offset_w;
                    let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                    let out_idx = ow + oh * w_out_pad + c * w_out_pad * h_out_pad;
                    out_data[out_idx] = x.data.as_ref().unwrap().index(x_idx);
                }
            }
        }

        let out_shape = vec![self.channels, self.out_h, self.out_w];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, GoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, 0, Role::Output)]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let out_claim = out_claims[0];
        let x_edge = edge_ids[0];

        let l_w_in = self.l_w_in();
        let l_h_in = self.l_h_in();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let l_c = self.l_c();

        let n_in = l_w_in + l_h_in + l_c;
        let in_size = 1usize << n_in;

        let w_in_pad = 1usize << l_w_in;
        let h_in_pad = 1usize << l_h_in;

        // Parse r_out: w_out bits | h_out bits | c bits (little-endian)
        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c];

        // Build eq tables for output dimensions
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);

        // Build H[flat_in_idx]: for each output (oh, ow, c), map to input index and accumulate
        let mut h_poly = vec![GoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for oh in 0..self.out_h {
                let ih = oh * self.stride_h + self.offset_h;
                for ow in 0..self.out_w {
                    let iw = ow * self.stride_w + self.offset_w;
                    let in_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                    let val = ext2_mul(
                        ext2_mul(eq_out_w[ow], eq_out_h[oh]),
                        eq_out_c[c],
                    );
                    h_poly[in_idx] = ext2_add(h_poly[in_idx], val);
                }
            }
        }

        // Build X polynomial in Ext2
        let x_data = witnesses[0];
        let x_evals = x_data.data.as_ref().unwrap().evaluations_ref();
        let x_ext2: Vec<GoldilocksExt2> = (0..in_size)
            .map(|i| GoldilocksExt2::from_base(x_evals[i]))
            .collect();

        // Sumcheck: Σ H[i] * X[i] = SV(r_out)
        let mut prover = CpuLinearSumcheckProverExt2::new(n_in, 2, transcript);
        let proof = prover.prove(&mut [h_poly, x_ext2].as_mut_slice(), transcript);

        let challenges = &prover.challenges;

        // Claim on X at the sumcheck challenge point
        let x_eval = x_data.data.as_ref().unwrap().evaluate_at_point_ext2(challenges);
        let x_claim = Claim {
            edge_id: x_edge,
            sparse_id: 0,
            point: challenges.clone(),
            eval: x_eval,
        };

        (vec![proof], vec![x_claim])
    }

    fn verify(
        &self,
        _witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        // claims: [x_claim, out_claim (SubSample2D output)]
        let out_claim = claims.last().unwrap();
        let x_claim = &claims[0];

        let l_w_in = self.l_w_in();
        let l_h_in = self.l_h_in();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let l_c = self.l_c();

        let n_in = l_w_in + l_h_in + l_c;

        // Verify sumcheck: Σ H(x) * X(x) = SV(r_out)
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            out_claim.eval,
            n_in,
            2,
            transcript,
        );
        if !ok {
            println!("SubSample2D sumcheck verification failed");
            return false;
        }

        // Parse r_out
        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c];

        // Parse r_in (sumcheck challenges)
        let r_in_w = &challenges[..l_w_in];
        let r_in_h = &challenges[l_w_in..l_w_in + l_h_in];
        let r_in_c = &challenges[l_w_in + l_h_in..l_w_in + l_h_in + l_c];

        // Compute H(r_in) = factor_c * factor_h * factor_w (factored form)
        let eq_in_c = evaluate_lagrange_basis_ext2(r_in_c);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);
        let mut factor_c = GoldilocksExt2::zero();
        for c in 0..self.channels {
            factor_c = ext2_add(factor_c, ext2_mul(eq_in_c[c], eq_out_c[c]));
        }

        let eq_in_h = evaluate_lagrange_basis_ext2(r_in_h);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let mut factor_h = GoldilocksExt2::zero();
        for oh in 0..self.out_h {
            let ih = oh * self.stride_h + self.offset_h;
            factor_h = ext2_add(factor_h, ext2_mul(eq_in_h[ih], eq_out_h[oh]));
        }

        let eq_in_w = evaluate_lagrange_basis_ext2(r_in_w);
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let mut factor_w = GoldilocksExt2::zero();
        for ow in 0..self.out_w {
            let iw = ow * self.stride_w + self.offset_w;
            factor_w = ext2_add(factor_w, ext2_mul(eq_in_w[iw], eq_out_w[ow]));
        }

        let h_eval = ext2_mul(ext2_mul(factor_c, factor_h), factor_w);

        // Check: H(r_in) * X(r_in) = final_eval
        let expected_final = ext2_mul(h_eval, x_claim.eval);
        if expected_final != sumcheck_proofs[0].final_eval {
            println!("SubSample2D final eval check failed");
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
    use crate::transcript::Transcript;
    use crate::dag::Claim;

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        let data: Vec<GoldilocksField> = data.into_iter().map(GoldilocksField).collect();
        Witness::new(shape, data, DataType::Uint, 0, Role::Input)
    }

    #[test]
    fn test_subsample2d_run_stride2_offset0() {
        // X[1, 4, 4], stride=2, offset=(0,0) → SV[1, 2, 2]
        let ss = SubSample2D::new(1, 4, 4, 2, 2, 0, 0);
        assert_eq!(ss.out_h, 2);
        assert_eq!(ss.out_w, 2);

        // X layout (little-endian: w stride 1, h stride 4):
        // (w,h): (0,0)=1, (1,0)=2, (2,0)=3, (3,0)=4
        //        (0,1)=5, (1,1)=6, (2,1)=7, (3,1)=8
        //        (0,2)=9, (1,2)=10,(2,2)=11,(3,2)=12
        //        (0,3)=13,(1,3)=14,(2,3)=15,(3,3)=16
        let x = make_witness(vec![1, 4, 4], vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ]);
        let result = ss.run(&[&x]);
        let sv = &result[0];
        assert_eq!(sv.shape, vec![1, 2, 2]);

        // SV[0,0] = X[0,0] = 1, SV[1,0] = X[2,0] = 3
        // SV[0,1] = X[0,2] = 9, SV[1,1] = X[2,2] = 11
        assert_eq!(sv.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(sv.data.as_ref().unwrap().index(1), GoldilocksField(3));
        assert_eq!(sv.data.as_ref().unwrap().index(2), GoldilocksField(9));
        assert_eq!(sv.data.as_ref().unwrap().index(3), GoldilocksField(11));
    }

    #[test]
    fn test_subsample2d_run_stride2_offset1() {
        // X[1, 4, 4], stride=2, offset=(1,1) → SV[1, 1, 1]
        let ss = SubSample2D::new(1, 4, 4, 2, 2, 1, 1);
        assert_eq!(ss.out_h, 1);
        assert_eq!(ss.out_w, 1);

        let x = make_witness(vec![1, 4, 4], vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ]);
        let result = ss.run(&[&x]);
        let sv = &result[0];
        assert_eq!(sv.shape, vec![1, 1, 1]);

        // SV[0,0] = X[1, 1] = w=1 + h=1*4 = 6
        assert_eq!(sv.data.as_ref().unwrap().index(0), GoldilocksField(6));
    }

    #[test]
    fn test_subsample2d_prove_verify() {
        // X[2, 4, 4], stride=2, offset=(0,0) → SV[2, 2, 2]
        let ss = SubSample2D::new(2, 4, 4, 2, 2, 0, 0);

        let mut x_data = vec![0u64; 32]; // 2 channels * 4*4 padded
        for c in 0..2 {
            for h in 0..4 {
                for w in 0..4 {
                    x_data[w + h * 4 + c * 16] = (c * 100 + h * 10 + w + 1) as u64;
                }
            }
        }
        let x = make_witness(vec![2, 4, 4], x_data);
        let result = ss.run(&[&x]);
        let sv = &result[0];

        // Create output claim
        let n_sv = sv.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ss");
        let point: Vec<GoldilocksExt2> = (0..n_sv)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = sv.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_ss_prove");
        let (proofs, claims) = ss.prove(
            &[&x, sv],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 1, "SubSample2D should have 1 sumcheck proof");
        assert_eq!(claims.len(), 1, "SubSample2D should produce one X claim");

        // Check that X claim eval matches X evaluated at the challenge point
        let x_eval = x.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert_eq!(claims[0].eval, x_eval, "X claim eval should match X poly evaluation");

        // Verify
        let mut verify_transcript = Transcript::new(b"test_ss_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = ss.verify(
            &[&x, sv],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "SubSample2D prove/verify should pass");
    }

    #[test]
    fn test_subsample2d_prove_verify_with_offset() {
        // X[1, 4, 4], stride=2, offset=(1,0) → SV[1, 1, 2]
        let ss = SubSample2D::new(1, 4, 4, 2, 2, 1, 0);
        assert_eq!(ss.out_h, 1);
        assert_eq!(ss.out_w, 2);

        let x = make_witness(vec![1, 4, 4], vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        ]);
        let result = ss.run(&[&x]);
        let sv = &result[0];

        // SV[0,0] = X[0, 1] = 5, SV[1,0] = X[2, 1] = 7
        assert_eq!(sv.data.as_ref().unwrap().index(0), GoldilocksField(5));
        assert_eq!(sv.data.as_ref().unwrap().index(1), GoldilocksField(7));

        let n_sv = sv.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ss_off");
        let point: Vec<GoldilocksExt2> = (0..n_sv)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = sv.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_ss_off_prove");
        let (proofs, claims) = ss.prove(
            &[&x, sv],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );

        let x_eval = x.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert_eq!(claims[0].eval, x_eval, "X claim eval should match with offset");

        let mut verify_transcript = Transcript::new(b"test_ss_off_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = ss.verify(
            &[&x, sv],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "SubSample2D prove/verify with offset should pass");
    }

    #[test]
    fn test_subsample2d_prove_verify_large_offset() {
        // Test with offset >= stride (the case that broke the carry chain approach)
        // X[1, 8, 8], stride=2, offset=(2,2) → SV[1, 3, 3]
        let ss = SubSample2D::new(1, 8, 8, 2, 2, 2, 2);
        assert_eq!(ss.out_h, 3);
        assert_eq!(ss.out_w, 3);

        let mut x_data = vec![0u64; 64]; // 1 * 8 * 8
        for h in 0..8 {
            for w in 0..8 {
                x_data[w + h * 8] = (h * 10 + w + 1) as u64;
            }
        }
        let x = make_witness(vec![1, 8, 8], x_data);
        let result = ss.run(&[&x]);
        let sv = &result[0];

        // SV[ow, oh] = X[ow*2+2, oh*2+2]
        // SV[0,0] = X[2,2] = idx 2+2*8 = 18 → value 21+2 = 23
        assert_eq!(sv.data.as_ref().unwrap().index(0), GoldilocksField(23));

        let n_sv = sv.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_ss_big");
        let point: Vec<GoldilocksExt2> = (0..n_sv)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = sv.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_ss_big_prove");
        let (proofs, claims) = ss.prove(
            &[&x, sv],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 1);

        let x_eval = x.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert_eq!(claims[0].eval, x_eval, "X claim eval should match with large offset");

        let mut verify_transcript = Transcript::new(b"test_ss_big_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = ss.verify(
            &[&x, sv],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "SubSample2D prove/verify with large offset should pass");
    }
}
