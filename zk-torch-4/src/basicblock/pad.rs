//! Zero-padding blocks ([`ZeroPad`], [`ZeroPadAsym`], [`ZeroPad3D`]).
//!
//! Common technique: the output's evaluation at `r_out` equals
//! `Σ_x H(x) · X(x)` where `H[i] = eq(r_out, embed(i))` and `embed` is the
//! shape's pad-into-output mapping. The prover runs a CPU Ext2 sumcheck on
//! `H · X`, the verifier checks the final eval by factorizing `H(r_in)` per
//! dimension.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_field_eq, ext2_mul, get_n, log2_ceil};

// ============================================================================
// ZeroPad (symmetric H/W padding)
// ============================================================================

#[derive(Clone, Debug)]
pub struct ZeroPad {
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

impl ZeroPad {
    pub fn new(channels: usize, input_h: usize, input_w: usize, pad_h: usize, pad_w: usize) -> Self {
        Self { channels, input_h, input_w, pad_h, pad_w }
    }

    fn output_h(&self) -> usize { self.input_h + 2 * self.pad_h }
    fn output_w(&self) -> usize { self.input_w + 2 * self.pad_w }
    fn l_w_in(&self) -> usize { log2_ceil(self.input_w.max(1)) }
    fn l_h_in(&self) -> usize { log2_ceil(self.input_h.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.channels.max(1)) }
    fn l_w_out(&self) -> usize { log2_ceil(self.output_w().max(1)) }
    fn l_h_out(&self) -> usize { log2_ceil(self.output_h().max(1)) }
}

impl BasicBlock for ZeroPad {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ZeroPad expects 1 input");
        let x = inputs[0];
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.output_w().next_power_of_two();
        let h_out_pad = self.output_h().next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let data = x.data.as_ref().unwrap();
        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    let out_idx = (w + self.pad_w)
                        + (h + self.pad_h) * w_out_pad
                        + c * w_out_pad * h_out_pad;
                    out_data[out_idx] = data.index(in_idx);
                }
            }
        }
        let out_shape = vec![self.channels, self.output_h(), self.output_w()];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        // Scatter-gather index work; no algebraic shortcut. CPU is the
        // documented default.
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

        // H[in_idx] = eq_out_w[w + pad_w] · eq_out_h[h + pad_h] · eq_out_c[c].
        let mut h_poly = vec![AlmostGoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    h_poly[in_idx] = ext2_mul(
                        ext2_mul(eq_out_w[w + self.pad_w], eq_out_h[h + self.pad_h]),
                        eq_out_c[c],
                    );
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

        let h_eval = h_factored(
            &evaluate_lagrange_basis_ext2(r_in_c),
            &evaluate_lagrange_basis_ext2(r_out_c),
            &evaluate_lagrange_basis_ext2(r_in_h),
            &evaluate_lagrange_basis_ext2(r_out_h),
            &evaluate_lagrange_basis_ext2(r_in_w),
            &evaluate_lagrange_basis_ext2(r_out_w),
            self.channels,
            self.input_h,
            self.input_w,
            self.pad_h,
            self.pad_w,
        );
        ext2_field_eq(ext2_mul(h_eval, x_claim.eval), sumcheck_proofs[0].final_eval)
    }
}

// ============================================================================
// ZeroPadAsym
// ============================================================================

#[derive(Clone, Debug)]
pub struct ZeroPadAsym {
    pub channels: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub pad_h_top: usize,
    pub pad_h_bottom: usize,
    pub pad_w_left: usize,
    pub pad_w_right: usize,
}

impl ZeroPadAsym {
    pub fn new(
        channels: usize,
        input_h: usize,
        input_w: usize,
        pad_h_top: usize,
        pad_h_bottom: usize,
        pad_w_left: usize,
        pad_w_right: usize,
    ) -> Self {
        Self {
            channels,
            input_h,
            input_w,
            pad_h_top,
            pad_h_bottom,
            pad_w_left,
            pad_w_right,
        }
    }

    fn output_h(&self) -> usize { self.input_h + self.pad_h_top + self.pad_h_bottom }
    fn output_w(&self) -> usize { self.input_w + self.pad_w_left + self.pad_w_right }
    fn l_w_in(&self) -> usize { log2_ceil(self.input_w.max(1)) }
    fn l_h_in(&self) -> usize { log2_ceil(self.input_h.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.channels.max(1)) }
    fn l_w_out(&self) -> usize { log2_ceil(self.output_w().max(1)) }
    fn l_h_out(&self) -> usize { log2_ceil(self.output_h().max(1)) }
}

impl BasicBlock for ZeroPadAsym {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ZeroPadAsym expects 1 input");
        let x = inputs[0];
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out_pad = self.output_w().next_power_of_two();
        let h_out_pad = self.output_h().next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let data = x.data.as_ref().unwrap();
        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    let out_idx = (w + self.pad_w_left)
                        + (h + self.pad_h_top) * w_out_pad
                        + c * w_out_pad * h_out_pad;
                    out_data[out_idx] = data.index(in_idx);
                }
            }
        }
        let out_shape = vec![self.channels, self.output_h(), self.output_w()];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

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

        let mut h_poly = vec![AlmostGoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    h_poly[in_idx] = ext2_mul(
                        ext2_mul(eq_out_w[w + self.pad_w_left], eq_out_h[h + self.pad_h_top]),
                        eq_out_c[c],
                    );
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
        let h_eval = h_factored(
            &evaluate_lagrange_basis_ext2(r_in_c),
            &evaluate_lagrange_basis_ext2(r_out_c),
            &evaluate_lagrange_basis_ext2(r_in_h),
            &evaluate_lagrange_basis_ext2(r_out_h),
            &evaluate_lagrange_basis_ext2(r_in_w),
            &evaluate_lagrange_basis_ext2(r_out_w),
            self.channels,
            self.input_h,
            self.input_w,
            self.pad_h_top,
            self.pad_w_left,
        );
        ext2_field_eq(ext2_mul(h_eval, x_claim.eval), sumcheck_proofs[0].final_eval)
    }
}

// ============================================================================
// ZeroPad3D
// ============================================================================

#[derive(Clone, Debug)]
pub struct ZeroPad3D {
    pub channels: usize,
    pub input_d: usize,
    pub input_h: usize,
    pub input_w: usize,
    pub pad_d: usize,
    pub pad_h: usize,
    pub pad_w: usize,
}

impl ZeroPad3D {
    pub fn new(
        channels: usize,
        input_d: usize,
        input_h: usize,
        input_w: usize,
        pad_d: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Self {
        Self {
            channels,
            input_d,
            input_h,
            input_w,
            pad_d,
            pad_h,
            pad_w,
        }
    }

    fn output_d(&self) -> usize { self.input_d + 2 * self.pad_d }
    fn output_h(&self) -> usize { self.input_h + 2 * self.pad_h }
    fn output_w(&self) -> usize { self.input_w + 2 * self.pad_w }
    fn l_w_in(&self) -> usize { log2_ceil(self.input_w.max(1)) }
    fn l_h_in(&self) -> usize { log2_ceil(self.input_h.max(1)) }
    fn l_d_in(&self) -> usize { log2_ceil(self.input_d.max(1)) }
    fn l_c(&self) -> usize { log2_ceil(self.channels.max(1)) }
    fn l_w_out(&self) -> usize { log2_ceil(self.output_w().max(1)) }
    fn l_h_out(&self) -> usize { log2_ceil(self.output_h().max(1)) }
    fn l_d_out(&self) -> usize { log2_ceil(self.output_d().max(1)) }
}

impl BasicBlock for ZeroPad3D {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "ZeroPad3D expects 1 input");
        let x = inputs[0];
        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let d_in_pad = self.input_d.next_power_of_two();
        let w_out_pad = self.output_w().next_power_of_two();
        let h_out_pad = self.output_h().next_power_of_two();
        let d_out_pad = self.output_d().next_power_of_two();
        let c_pad = self.channels.next_power_of_two();
        let out_size = c_pad * d_out_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![AlmostGoldilocksField(0); out_size];
        let data = x.data.as_ref().unwrap();
        for c in 0..self.channels {
            for d in 0..self.input_d {
                for h in 0..self.input_h {
                    for w in 0..self.input_w {
                        let in_idx = w
                            + h * w_in_pad
                            + d * w_in_pad * h_in_pad
                            + c * w_in_pad * h_in_pad * d_in_pad;
                        let out_idx = (w + self.pad_w)
                            + (h + self.pad_h) * w_out_pad
                            + (d + self.pad_d) * w_out_pad * h_out_pad
                            + c * w_out_pad * h_out_pad * d_out_pad;
                        out_data[out_idx] = data.index(in_idx);
                    }
                }
            }
        }
        let out_shape = vec![self.channels, self.output_d(), self.output_h(), self.output_w()];
        let n = get_n(&out_shape);
        if out_data.len() < (1 << n) {
            out_data.resize(1 << n, AlmostGoldilocksField(0));
        }
        vec![Witness::new(out_shape, out_data, DataType::Uint, inputs[0].sf, Role::Output)]
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> { self.run(inputs) }

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
        let l_d_in = self.l_d_in();
        let l_c = self.l_c();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let l_d_out = self.l_d_out();
        let n_in = l_w_in + l_h_in + l_d_in + l_c;
        let in_size = 1usize << n_in;
        let w_in_pad = 1usize << l_w_in;
        let h_in_pad = 1usize << l_h_in;
        let d_in_pad = 1usize << l_d_in;

        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_d = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_d_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out + l_d_out
            ..l_w_out + l_h_out + l_d_out + l_c];
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let eq_out_d = evaluate_lagrange_basis_ext2(r_out_d);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);

        let mut h_poly = vec![AlmostGoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for d in 0..self.input_d {
                for h in 0..self.input_h {
                    for w in 0..self.input_w {
                        let in_idx = w
                            + h * w_in_pad
                            + d * w_in_pad * h_in_pad
                            + c * w_in_pad * h_in_pad * d_in_pad;
                        h_poly[in_idx] = ext2_mul(
                            ext2_mul(eq_out_w[w + self.pad_w], eq_out_h[h + self.pad_h]),
                            ext2_mul(eq_out_d[d + self.pad_d], eq_out_c[c]),
                        );
                    }
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
        let l_d_in = self.l_d_in();
        let l_c = self.l_c();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let l_d_out = self.l_d_out();
        let n_in = l_w_in + l_h_in + l_d_in + l_c;
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
        let r_out_d = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_d_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out + l_d_out
            ..l_w_out + l_h_out + l_d_out + l_c];
        let r_in_w = &challenges[..l_w_in];
        let r_in_h = &challenges[l_w_in..l_w_in + l_h_in];
        let r_in_d = &challenges[l_w_in + l_h_in..l_w_in + l_h_in + l_d_in];
        let r_in_c = &challenges[l_w_in + l_h_in + l_d_in..l_w_in + l_h_in + l_d_in + l_c];

        let factor_c = dim_factor(
            &evaluate_lagrange_basis_ext2(r_in_c),
            &evaluate_lagrange_basis_ext2(r_out_c),
            self.channels,
            0,
        );
        let factor_d = dim_factor(
            &evaluate_lagrange_basis_ext2(r_in_d),
            &evaluate_lagrange_basis_ext2(r_out_d),
            self.input_d,
            self.pad_d,
        );
        let factor_h = dim_factor(
            &evaluate_lagrange_basis_ext2(r_in_h),
            &evaluate_lagrange_basis_ext2(r_out_h),
            self.input_h,
            self.pad_h,
        );
        let factor_w = dim_factor(
            &evaluate_lagrange_basis_ext2(r_in_w),
            &evaluate_lagrange_basis_ext2(r_out_w),
            self.input_w,
            self.pad_w,
        );
        let h_eval = ext2_mul(ext2_mul(factor_c, factor_d), ext2_mul(factor_h, factor_w));
        ext2_field_eq(ext2_mul(h_eval, x_claim.eval), sumcheck_proofs[0].final_eval)
    }
}

// ============================================================================
// Verifier-side H-poly factorization helpers
// ============================================================================

/// Σ_i eq_in[i] · eq_out[i + offset], summed over `n_in` (input-range) values.
fn dim_factor(
    eq_in: &[AlmostGoldilocksExt2],
    eq_out: &[AlmostGoldilocksExt2],
    n_in: usize,
    offset: usize,
) -> AlmostGoldilocksExt2 {
    let mut acc = AlmostGoldilocksExt2::zero();
    for i in 0..n_in {
        acc = ext2_add(acc, ext2_mul(eq_in[i], eq_out[i + offset]));
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn h_factored(
    eq_in_c: &[AlmostGoldilocksExt2],
    eq_out_c: &[AlmostGoldilocksExt2],
    eq_in_h: &[AlmostGoldilocksExt2],
    eq_out_h: &[AlmostGoldilocksExt2],
    eq_in_w: &[AlmostGoldilocksExt2],
    eq_out_w: &[AlmostGoldilocksExt2],
    channels: usize,
    input_h: usize,
    input_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> AlmostGoldilocksExt2 {
    let factor_c = dim_factor(eq_in_c, eq_out_c, channels, 0);
    let factor_h = dim_factor(eq_in_h, eq_out_h, input_h, pad_h);
    let factor_w = dim_factor(eq_in_w, eq_out_w, input_w, pad_w);
    ext2_mul(ext2_mul(factor_c, factor_h), factor_w)
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

    fn run_prove_verify_pad<P: BasicBlock>(p: &P, x: &Witness) {
        let out = p.run(&[x]);
        let y = &out[0];
        let n_y = y.data.as_ref().unwrap().n();
        let mut t_in = Transcript::new(b"setup");
        let point: Vec<_> = (0..n_y).map(|_| t_in.challenge_ext2(b"r")).collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let claim = Claim { edge_id: 1, sparse_id: 0, point, eval };
        let mut t_prove = Transcript::new(b"setup-prove");
        let (proofs, claims) = p.prove(&[x, y], &[0, 1], &[&claim], &mut t_prove);
        assert_eq!(proofs.len(), 1);
        // X claim eval must match a direct evaluation of x at the prover's challenge.
        let direct = x.data.as_ref().unwrap().evaluate_at_point_ext2(&claims[0].point);
        assert!(ext2_field_eq(claims[0].eval, direct));
        let mut t_verify = Transcript::new(b"setup-prove");
        let all = [&claims[0], &claim];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(p.verify(&[x, y], &all, &proof_refs, &mut t_verify));
    }

    #[test]
    fn zeropad_run_embeds_input_with_zero_borders() {
        let pad = ZeroPad::new(1, 2, 2, 1, 1);
        let x = make_witness(vec![1, 2, 2], vec![1, 2, 3, 4]);
        let y = pad.run(&[&x]);
        assert_eq!(y[0].shape, vec![1, 4, 4]);
        let data = y[0].data.as_ref().unwrap();
        // After padding by 1, the interior is shifted by (1, 1).
        // Padded width = 4 (next pow of 2 of 4 is still 4).
        assert_eq!(data.index(0), agl(0));            // (0,0) — pad
        assert_eq!(data.index(1 + 1 * 4), agl(1));    // (1,1) — input (0,0)
        assert_eq!(data.index(2 + 1 * 4), agl(2));    // (2,1) — input (1,0)
        assert_eq!(data.index(1 + 2 * 4), agl(3));    // (1,2) — input (0,1)
        assert_eq!(data.index(2 + 2 * 4), agl(4));    // (2,2) — input (1,1)
        assert_eq!(data.index(3 + 3 * 4), agl(0));    // (3,3) — pad
    }

    #[test]
    fn zeropad_prove_verify_roundtrip() {
        let pad = ZeroPad::new(2, 2, 2, 1, 1);
        let x = make_witness(vec![2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        run_prove_verify_pad(&pad, &x);
    }

    #[test]
    fn zeropad_asym_prove_verify_roundtrip() {
        let pad = ZeroPadAsym::new(2, 2, 2, 0, 1, 0, 1);
        let x = make_witness(vec![2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        run_prove_verify_pad(&pad, &x);
    }

    #[test]
    fn zeropad_3d_prove_verify_roundtrip() {
        let pad = ZeroPad3D::new(1, 2, 2, 2, 1, 1, 1);
        let x = make_witness(vec![1, 2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        run_prove_verify_pad(&pad, &x);
    }
}
