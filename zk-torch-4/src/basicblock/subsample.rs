//! [`SubSample2D`] — strided extraction `SV[c, oh, ow] = X[c, oh·sh + off_h,
//! ow·sw + off_w]`. Same H-polynomial sumcheck pattern as the pad blocks but
//! in the reverse direction (output is smaller than input). H decomposes
//! cleanly per dimension on the verifier side.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, get_n, log2_ceil};

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
        input_h: usize,
        input_w: usize,
        stride_h: usize,
        stride_w: usize,
        offset_h: usize,
        offset_w: usize,
    ) -> Self {
        let out_h = (input_h - offset_h) / stride_h;
        let out_w = (input_w - offset_w) / stride_w;
        Self { channels, input_h, input_w, out_h, out_w, stride_h, stride_w, offset_h, offset_w }
    }

    /// Variant with explicit `out_h` / `out_w` — used by the maxpool kernel
    /// composition where all subsamples must share an output shape.
    pub fn new_with_output_size(
        channels: usize,
        input_h: usize,
        input_w: usize,
        stride_h: usize,
        stride_w: usize,
        offset_h: usize,
        offset_w: usize,
        out_h: usize,
        out_w: usize,
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
        assert_eq!(inputs.len(), 1, "SubSample2D expects 1 input");
        let x = inputs[0];
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.out_w.next_power_of_two();
        let h_out_pad = self.out_h.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let data = x.data.as_ref().unwrap();
        for c in 0..self.channels {
            for oh in 0..self.out_h {
                let ih = oh * self.stride_h + self.offset_h;
                for ow in 0..self.out_w {
                    let iw = ow * self.stride_w + self.offset_w;
                    let x_idx = iw + ih * w_in_pad + c * w_in_pad * h_in_pad;
                    let o_idx = ow + oh * w_out_pad + c * w_out_pad * h_out_pad;
                    out_data[o_idx] = data.index(x_idx);
                }
            }
        }
        let out_shape = vec![self.channels, self.out_h, self.out_w];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Pure gather indexing — no algebraic shortcut. CPU is the documented
        // default (philosophy rule #7).
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let out_claim = out_claims[0];
        let l_w_in = self.l_w_in();
        let l_h_in = self.l_h_in();
        let l_c = self.l_c();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let n_in = l_w_in + l_h_in + l_c;
        let in_size = 1usize << n_in;
        let w_in_pad = 1usize << l_w_in;
        let h_in_pad = 1usize << l_h_in;

        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c];
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);

        // H[in_idx] = eq_out_c[c] · eq_out_h[oh] · eq_out_w[ow] for the
        // unique (c, oh, ow) that maps to this input cell — accumulate, in
        // case stride covers the same input position from multiple outputs
        // (not possible here for valid stride, but the += handles it cleanly).
        let mut h_poly = vec![AlmostGoldilocksExt2::zero(); in_size];
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

        let x_data = witnesses[0].data.as_ref().unwrap();
        let x_evals = x_data.evaluations_ref();
        let x_ext2: Vec<_> = (0..in_size)
            .map(|i| AlmostGoldilocksExt2::from_base(x_evals[i]))
            .collect();
        let mut prover = CpuLinearSumcheckProverExt2::new(n_in, 2, transcript);
        let proof = prover.prove(&mut [h_poly, x_ext2], transcript);
        let challenges = prover.challenges.clone();
        let x_eval = x_data.evaluate_at_point_ext2(&challenges);
        let x_claim = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges,
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
        let out_claim = claims.last().unwrap();
        let x_claim = &claims[0];
        let l_w_in = self.l_w_in();
        let l_h_in = self.l_h_in();
        let l_c = self.l_c();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let n_in = l_w_in + l_h_in + l_c;
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            out_claim.eval,
            n_in,
            2,
            transcript,
        );
        if !ok { return false; }
        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c];
        let r_in_w = &challenges[..l_w_in];
        let r_in_h = &challenges[l_w_in..l_w_in + l_h_in];
        let r_in_c = &challenges[l_w_in + l_h_in..l_w_in + l_h_in + l_c];

        let eq_in_c = evaluate_lagrange_basis_ext2(r_in_c);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);
        let eq_in_h = evaluate_lagrange_basis_ext2(r_in_h);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let eq_in_w = evaluate_lagrange_basis_ext2(r_in_w);
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);

        let mut factor_c = AlmostGoldilocksExt2::zero();
        for c in 0..self.channels {
            factor_c = ext2_add(factor_c, ext2_mul(eq_in_c[c], eq_out_c[c]));
        }
        let mut factor_h = AlmostGoldilocksExt2::zero();
        for oh in 0..self.out_h {
            let ih = oh * self.stride_h + self.offset_h;
            factor_h = ext2_add(factor_h, ext2_mul(eq_in_h[ih], eq_out_h[oh]));
        }
        let mut factor_w = AlmostGoldilocksExt2::zero();
        for ow in 0..self.out_w {
            let iw = ow * self.stride_w + self.offset_w;
            factor_w = ext2_add(factor_w, ext2_mul(eq_in_w[iw], eq_out_w[ow]));
        }
        let h_eval = ext2_mul(ext2_mul(factor_c, factor_h), factor_w);
        ext2_field_eq(ext2_mul(h_eval, x_claim.eval), sumcheck_proofs[0].final_eval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn make_witness(shape: Vec<usize>, data: Vec<u64>) -> Witness {
        Witness::new(
            shape,
            data.into_iter().map(agl).collect(),
            DataType::Uint,
            0,
            Role::Input,
        )
    }

    /// stride=2, offset=0 on X[1, 4, 4] extracts the (0, 2)-grid corners.
    #[test]
    fn run_stride2_offset0_extracts_grid() {
        let ss = SubSample2D::new(1, 4, 4, 2, 2, 0, 0);
        assert_eq!(ss.out_h, 2);
        assert_eq!(ss.out_w, 2);
        let x = make_witness(vec![1, 4, 4], (1..=16u64).collect());
        let out = ss.run(&[&x]);
        let evals = out[0].data.as_ref().unwrap().evaluations_ref();
        // SV[0,0] = X[0,0] = 1, SV[1,0] = X[2,0] = 3.
        assert_eq!(evals[0], agl(1));
        assert_eq!(evals[1], agl(3));
        // SV[0,1] = X[0,2] = 9, SV[1,1] = X[2,2] = 11.
        assert_eq!(evals[2], agl(9));
        assert_eq!(evals[3], agl(11));
    }

    fn run_prove_verify(ss: &SubSample2D, x: &Witness) {
        let outs = ss.run(&[x]);
        let sv = &outs[0];
        let n_sv = sv.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"ss");
        let point: Vec<_> = (0..n_sv).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = sv.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let claim = Claim { edge_id: 1, sparse_id: 0, point, eval };
        let mut t_prove = Transcript::new(b"ss-prove");
        let (proofs, claims) = ss.prove(&[x, sv], &[0, 1], &[&claim], &mut t_prove);
        assert_eq!(proofs.len(), 1);
        let direct = x.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert!(ext2_field_eq(claims[0].eval, direct));
        let mut t_verify = Transcript::new(b"ss-prove");
        let all = [&claims[0], &claim];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(ss.verify(&[x, sv], &all, &proof_refs, &mut t_verify));
    }

    #[test]
    fn prove_verify_no_offset() {
        let ss = SubSample2D::new(2, 4, 4, 2, 2, 0, 0);
        let x = make_witness(vec![2, 4, 4], (1..=32u64).collect());
        run_prove_verify(&ss, &x);
    }

    #[test]
    fn prove_verify_with_offset() {
        // X[1, 4, 4], stride 2, offset (1, 0) → SV[1, 1, 2].
        let ss = SubSample2D::new(1, 4, 4, 2, 2, 1, 0);
        let x = make_witness(vec![1, 4, 4], (1..=16u64).collect());
        run_prove_verify(&ss, &x);
    }

    /// offset > stride was the carry-chain breakdown case in zk-torch-2;
    /// H-poly sumcheck handles it cleanly.
    #[test]
    fn prove_verify_offset_exceeds_stride() {
        let ss = SubSample2D::new(1, 8, 8, 2, 2, 2, 2);
        let x = make_witness(vec![1, 8, 8], (1..=64u64).collect());
        run_prove_verify(&ss, &x);
    }
}
