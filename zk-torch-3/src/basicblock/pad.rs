use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, DataType, Role, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_mul, get_n, log2_ceil};

/// ZeroPad: embeds X[C, H, W] into Y[C, H+2*pad_h, W+2*pad_w] with zero boundaries.
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
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out = self.output_w();
        let h_out = self.output_h();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    let out_idx = (w + self.pad_w) + (h + self.pad_h) * w_out_pad + c * w_out_pad * h_out_pad;
                    out_data[out_idx] = x.data.as_ref().unwrap().index(in_idx);
                }
            }
        }

        let out_shape = vec![self.channels, h_out, w_out];
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
        let l_c = self.l_c();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();

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

        // Build H[flat_in_idx] = eq_out_w[w + pad_w] * eq_out_h[h + pad_h] * eq_out_c[c]
        let mut h_poly = vec![GoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    let val = ext2_mul(
                        ext2_mul(eq_out_w[w + self.pad_w], eq_out_h[h + self.pad_h]),
                        eq_out_c[c],
                    );
                    h_poly[in_idx] = val;
                }
            }
        }

        // Build X polynomial in Ext2
        let x_data = witnesses[0];
        let x_evals = x_data.data.as_ref().unwrap().evaluations_ref();
        let x_ext2: Vec<GoldilocksExt2> = (0..in_size)
            .map(|i| GoldilocksExt2::from_base(x_evals[i]))
            .collect();

        // Sumcheck: Σ H[i] * X[i] = Y(r_out)
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
        // claims: [x_claim, out_claim]
        let out_claim = claims.last().unwrap();
        let x_claim = &claims[0];

        let l_w_in = self.l_w_in();
        let l_h_in = self.l_h_in();
        let l_c = self.l_c();
        let l_w_out = self.l_w_out();
        let l_h_out = self.l_h_out();
        let n_in = l_w_in + l_h_in + l_c;

        // Verify sumcheck
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            out_claim.eval,
            n_in,
            2,
            transcript,
        );
        if !ok {
            println!("ZeroPad sumcheck verification failed");
            return false;
        }

        // Parse r_out
        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c];

        // Parse r_in (challenges)
        let r_in_w = &challenges[..l_w_in];
        let r_in_h = &challenges[l_w_in..l_w_in + l_h_in];
        let r_in_c = &challenges[l_w_in + l_h_in..l_w_in + l_h_in + l_c];

        // Compute H(r_in) = factor_c * factor_h * factor_w
        let eq_in_c = evaluate_lagrange_basis_ext2(r_in_c);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);
        let mut factor_c = GoldilocksExt2::zero();
        for c in 0..self.channels {
            factor_c = ext2_add(factor_c, ext2_mul(eq_in_c[c], eq_out_c[c]));
        }

        let eq_in_h = evaluate_lagrange_basis_ext2(r_in_h);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let mut factor_h = GoldilocksExt2::zero();
        for h in 0..self.input_h {
            factor_h = ext2_add(factor_h, ext2_mul(eq_in_h[h], eq_out_h[h + self.pad_h]));
        }

        let eq_in_w = evaluate_lagrange_basis_ext2(r_in_w);
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let mut factor_w = GoldilocksExt2::zero();
        for w in 0..self.input_w {
            factor_w = ext2_add(factor_w, ext2_mul(eq_in_w[w], eq_out_w[w + self.pad_w]));
        }

        let h_eval = ext2_mul(ext2_mul(factor_c, factor_h), factor_w);

        // Check: H(r_in) * X(r_in) = final_eval
        let expected_final = ext2_mul(h_eval, x_claim.eval);
        if expected_final != sumcheck_proofs[0].final_eval {
            println!("ZeroPad final eval check failed");
            return false;
        }

        true
    }
}

/// ZeroPadAsym: embeds X[C, H, W] into Y[C, H+pt+pb, W+pl+pr] with asymmetric zero padding.
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
        channels: usize, input_h: usize, input_w: usize,
        pad_h_top: usize, pad_h_bottom: usize,
        pad_w_left: usize, pad_w_right: usize,
    ) -> Self {
        Self { channels, input_h, input_w, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right }
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
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let w_out = self.output_w();
        let h_out = self.output_h();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    let out_idx = (w + self.pad_w_left) + (h + self.pad_h_top) * w_out_pad + c * w_out_pad * h_out_pad;
                    out_data[out_idx] = x.data.as_ref().unwrap().index(in_idx);
                }
            }
        }

        let out_shape = vec![self.channels, h_out, w_out];
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

        let mut h_poly = vec![GoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for h in 0..self.input_h {
                for w in 0..self.input_w {
                    let in_idx = w + h * w_in_pad + c * w_in_pad * h_in_pad;
                    let val = ext2_mul(
                        ext2_mul(eq_out_w[w + self.pad_w_left], eq_out_h[h + self.pad_h_top]),
                        eq_out_c[c],
                    );
                    h_poly[in_idx] = val;
                }
            }
        }

        let x_data = witnesses[0];
        let x_evals = x_data.data.as_ref().unwrap().evaluations_ref();
        let x_ext2: Vec<GoldilocksExt2> = (0..in_size)
            .map(|i| GoldilocksExt2::from_base(x_evals[i]))
            .collect();

        let mut prover = CpuLinearSumcheckProverExt2::new(n_in, 2, transcript);
        let proof = prover.prove(&mut [h_poly, x_ext2].as_mut_slice(), transcript);

        let challenges = &prover.challenges;

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
        if !ok {
            println!("ZeroPadAsym sumcheck verification failed");
            return false;
        }

        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_c];

        let r_in_w = &challenges[..l_w_in];
        let r_in_h = &challenges[l_w_in..l_w_in + l_h_in];
        let r_in_c = &challenges[l_w_in + l_h_in..l_w_in + l_h_in + l_c];

        let eq_in_c = evaluate_lagrange_basis_ext2(r_in_c);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);
        let mut factor_c = GoldilocksExt2::zero();
        for c in 0..self.channels {
            factor_c = ext2_add(factor_c, ext2_mul(eq_in_c[c], eq_out_c[c]));
        }

        let eq_in_h = evaluate_lagrange_basis_ext2(r_in_h);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let mut factor_h = GoldilocksExt2::zero();
        for h in 0..self.input_h {
            factor_h = ext2_add(factor_h, ext2_mul(eq_in_h[h], eq_out_h[h + self.pad_h_top]));
        }

        let eq_in_w = evaluate_lagrange_basis_ext2(r_in_w);
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let mut factor_w = GoldilocksExt2::zero();
        for w in 0..self.input_w {
            factor_w = ext2_add(factor_w, ext2_mul(eq_in_w[w], eq_out_w[w + self.pad_w_left]));
        }

        let h_eval = ext2_mul(ext2_mul(factor_c, factor_h), factor_w);

        let expected_final = ext2_mul(h_eval, x_claim.eval);
        if expected_final != sumcheck_proofs[0].final_eval {
            println!("ZeroPadAsym final eval check failed");
            return false;
        }

        true
    }
}

/// ZeroPad3D: embeds X[C, D, H, W] into Y[C, D+2*pad_d, H+2*pad_h, W+2*pad_w] with zero boundaries.
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
    pub fn new(channels: usize, input_d: usize, input_h: usize, input_w: usize,
               pad_d: usize, pad_h: usize, pad_w: usize) -> Self {
        Self { channels, input_d, input_h, input_w, pad_d, pad_h, pad_w }
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
        assert_eq!(inputs.len(), 1);
        let x = inputs[0];

        let w_in_pad = self.input_w.next_power_of_two();
        let h_in_pad = self.input_h.next_power_of_two();
        let d_in_pad = self.input_d.next_power_of_two();
        let w_out = self.output_w();
        let h_out = self.output_h();
        let d_out = self.output_d();
        let w_out_pad = w_out.next_power_of_two();
        let h_out_pad = h_out.next_power_of_two();
        let d_out_pad = d_out.next_power_of_two();
        let c_pad = self.channels.next_power_of_two();

        let out_size = c_pad * d_out_pad * h_out_pad * w_out_pad;
        let mut out_data = vec![GoldilocksField(0); out_size];

        for c in 0..self.channels {
            for d in 0..self.input_d {
                for h in 0..self.input_h {
                    for w in 0..self.input_w {
                        let in_idx = w + h * w_in_pad + d * w_in_pad * h_in_pad
                            + c * w_in_pad * h_in_pad * d_in_pad;
                        let out_idx = (w + self.pad_w) + (h + self.pad_h) * w_out_pad
                            + (d + self.pad_d) * w_out_pad * h_out_pad
                            + c * w_out_pad * h_out_pad * d_out_pad;
                        out_data[out_idx] = x.data.as_ref().unwrap().index(in_idx);
                    }
                }
            }
        }

        let out_shape = vec![self.channels, d_out, h_out, w_out];
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

        // Parse r_out: w_out bits | h_out bits | d_out bits | c bits (little-endian)
        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_d = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_d_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out + l_d_out..l_w_out + l_h_out + l_d_out + l_c];

        // Build eq tables for output dimensions
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let eq_out_d = evaluate_lagrange_basis_ext2(r_out_d);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);

        // Build H[flat_in_idx] = eq_out_w[w+pad_w] * eq_out_h[h+pad_h] * eq_out_d[d+pad_d] * eq_out_c[c]
        let mut h_poly = vec![GoldilocksExt2::zero(); in_size];
        for c in 0..self.channels {
            for d in 0..self.input_d {
                for h in 0..self.input_h {
                    for w in 0..self.input_w {
                        let in_idx = w + h * w_in_pad + d * w_in_pad * h_in_pad
                            + c * w_in_pad * h_in_pad * d_in_pad;
                        let val = ext2_mul(
                            ext2_mul(eq_out_w[w + self.pad_w], eq_out_h[h + self.pad_h]),
                            ext2_mul(eq_out_d[d + self.pad_d], eq_out_c[c]),
                        );
                        h_poly[in_idx] = val;
                    }
                }
            }
        }

        // Build X polynomial in Ext2
        let x_data = witnesses[0];
        let x_evals = x_data.data.as_ref().unwrap().evaluations_ref();
        let x_ext2: Vec<GoldilocksExt2> = (0..in_size)
            .map(|i| GoldilocksExt2::from_base(x_evals[i]))
            .collect();

        // Sumcheck: Σ H[i] * X[i] = Y(r_out)
        let mut prover = CpuLinearSumcheckProverExt2::new(n_in, 2, transcript);
        let proof = prover.prove(&mut [h_poly, x_ext2].as_mut_slice(), transcript);

        let challenges = &prover.challenges;

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
        if !ok {
            println!("ZeroPad3D sumcheck verification failed");
            return false;
        }

        // Parse r_out
        let r_out_w = &out_claim.point[..l_w_out];
        let r_out_h = &out_claim.point[l_w_out..l_w_out + l_h_out];
        let r_out_d = &out_claim.point[l_w_out + l_h_out..l_w_out + l_h_out + l_d_out];
        let r_out_c = &out_claim.point[l_w_out + l_h_out + l_d_out..l_w_out + l_h_out + l_d_out + l_c];

        // Parse r_in (challenges)
        let r_in_w = &challenges[..l_w_in];
        let r_in_h = &challenges[l_w_in..l_w_in + l_h_in];
        let r_in_d = &challenges[l_w_in + l_h_in..l_w_in + l_h_in + l_d_in];
        let r_in_c = &challenges[l_w_in + l_h_in + l_d_in..l_w_in + l_h_in + l_d_in + l_c];

        // Compute H(r_in) = factor_c * factor_d * factor_h * factor_w
        let eq_in_c = evaluate_lagrange_basis_ext2(r_in_c);
        let eq_out_c = evaluate_lagrange_basis_ext2(r_out_c);
        let mut factor_c = GoldilocksExt2::zero();
        for c in 0..self.channels {
            factor_c = ext2_add(factor_c, ext2_mul(eq_in_c[c], eq_out_c[c]));
        }

        let eq_in_d = evaluate_lagrange_basis_ext2(r_in_d);
        let eq_out_d = evaluate_lagrange_basis_ext2(r_out_d);
        let mut factor_d = GoldilocksExt2::zero();
        for d in 0..self.input_d {
            factor_d = ext2_add(factor_d, ext2_mul(eq_in_d[d], eq_out_d[d + self.pad_d]));
        }

        let eq_in_h = evaluate_lagrange_basis_ext2(r_in_h);
        let eq_out_h = evaluate_lagrange_basis_ext2(r_out_h);
        let mut factor_h = GoldilocksExt2::zero();
        for h in 0..self.input_h {
            factor_h = ext2_add(factor_h, ext2_mul(eq_in_h[h], eq_out_h[h + self.pad_h]));
        }

        let eq_in_w = evaluate_lagrange_basis_ext2(r_in_w);
        let eq_out_w = evaluate_lagrange_basis_ext2(r_out_w);
        let mut factor_w = GoldilocksExt2::zero();
        for w in 0..self.input_w {
            factor_w = ext2_add(factor_w, ext2_mul(eq_in_w[w], eq_out_w[w + self.pad_w]));
        }

        let h_eval = ext2_mul(ext2_mul(factor_c, factor_d), ext2_mul(factor_h, factor_w));

        let expected_final = ext2_mul(h_eval, x_claim.eval);
        if expected_final != sumcheck_proofs[0].final_eval {
            println!("ZeroPad3D final eval check failed");
            return false;
        }

        true
    }
}

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
    fn test_zeropad_run() {
        // X[1, 2, 2], pad_h=1, pad_w=1 → Y[1, 4, 4]
        let pad = ZeroPad::new(1, 2, 2, 1, 1);
        let x = make_witness(vec![1, 2, 2], vec![1, 2, 3, 4]);
        let result = pad.run(&[&x]);
        let y = &result[0];

        assert_eq!(y.shape, vec![1, 4, 4]);

        // Y layout (little-endian: w bits lowest, then h bits, then c bits):
        // Row 0 (h=0): [0, 0, 0, 0]
        // Row 1 (h=1): [0, 1, 2, 0]
        // Row 2 (h=2): [0, 3, 4, 0]
        // Row 3 (h=3): [0, 0, 0, 0]
        // Index = w + h * 4
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(0)); // (0,0)
        assert_eq!(y.data.as_ref().unwrap().index(1 + 1 * 4), GoldilocksField(1)); // (1,1)
        assert_eq!(y.data.as_ref().unwrap().index(2 + 1 * 4), GoldilocksField(2)); // (2,1)
        assert_eq!(y.data.as_ref().unwrap().index(1 + 2 * 4), GoldilocksField(3)); // (1,2)
        assert_eq!(y.data.as_ref().unwrap().index(2 + 2 * 4), GoldilocksField(4)); // (2,2)
        assert_eq!(y.data.as_ref().unwrap().index(3 + 3 * 4), GoldilocksField(0)); // (3,3)
    }

    #[test]
    fn test_zeropad_prove_verify() {
        let pad = ZeroPad::new(2, 2, 2, 1, 1);
        // X[2, 2, 2]
        let x = make_witness(vec![2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let result = pad.run(&[&x]);
        let y = &result[0];

        // Create output claim
        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_pad");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        // Prove
        let mut prove_transcript = Transcript::new(b"test_pad_prove");
        let (proofs, claims) = pad.prove(
            &[&x, y],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 1);

        // Verify
        let mut verify_transcript = Transcript::new(b"test_pad_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = pad.verify(
            &[&x, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "ZeroPad prove/verify should pass");
    }

    #[test]
    fn test_zeropad_asym_run() {
        // X[1, 2, 2], pad_h=(0,1), pad_w=(0,1) → Y[1, 3, 3]
        let pad = ZeroPadAsym::new(1, 2, 2, 0, 1, 0, 1);
        let x = make_witness(vec![1, 2, 2], vec![1, 2, 3, 4]);
        let result = pad.run(&[&x]);
        let y = &result[0];

        assert_eq!(y.shape, vec![1, 3, 3]);

        // Y layout (little-endian: w stride 1, h stride 4 (padded to 4)):
        // (w,h): (0,0)=1, (1,0)=2, (2,0)=0
        //        (0,1)=3, (1,1)=4, (2,1)=0
        //        (0,2)=0, (1,2)=0, (2,2)=0
        let w_out_pad = 4; // 3.next_power_of_two()
        assert_eq!(y.data.as_ref().unwrap().index(0), GoldilocksField(1));
        assert_eq!(y.data.as_ref().unwrap().index(1), GoldilocksField(2));
        assert_eq!(y.data.as_ref().unwrap().index(2), GoldilocksField(0));
        assert_eq!(y.data.as_ref().unwrap().index(0 + 1 * w_out_pad), GoldilocksField(3));
        assert_eq!(y.data.as_ref().unwrap().index(1 + 1 * w_out_pad), GoldilocksField(4));
    }

    #[test]
    fn test_zeropad_asym_prove_verify() {
        let pad = ZeroPadAsym::new(2, 2, 2, 0, 1, 0, 1);
        let x = make_witness(vec![2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let result = pad.run(&[&x]);
        let y = &result[0];

        let n_y = y.data.as_ref().unwrap().n();
        let mut transcript = Transcript::new(b"test_apad");
        let point: Vec<GoldilocksExt2> = (0..n_y)
            .map(|_| transcript.challenge_ext2(b"ch"))
            .collect();
        let eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&point);
        let out_claim = Claim {
            edge_id: 1,
            sparse_id: 0,
            point,
            eval,
        };

        let mut prove_transcript = Transcript::new(b"test_apad_prove");
        let (proofs, claims) = pad.prove(
            &[&x, y],
            &[0, 1],
            &[&out_claim],
            &mut prove_transcript,
        );
        assert_eq!(proofs.len(), 1);
        assert_eq!(claims.len(), 1);

        let mut verify_transcript = Transcript::new(b"test_apad_prove");
        let mut all_claims: Vec<&Claim> = claims.iter().collect();
        all_claims.push(&out_claim);
        let proofs_ref: Vec<&SumcheckProof> = proofs.iter().collect();
        let verified = pad.verify(
            &[&x, y],
            &all_claims,
            &proofs_ref,
            &mut verify_transcript,
        );
        assert!(verified, "ZeroPadAsym prove/verify should pass");
    }
}
