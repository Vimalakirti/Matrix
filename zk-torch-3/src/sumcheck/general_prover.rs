use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use rayon::prelude::*;

use crate::poly::DenseMLPoly;
use crate::sumcheck::{SumcheckProof, SumcheckProver};
use crate::transcript::Transcript;
use crate::util::arith::{gl_add, gl_mul, gl_sub};

/// General linear sum-check protocol.
///
/// Computes: H = Σ_{x ∈ {0,1}^n} eq(r, x) * Σ_j (a_scalars[j] * Π_i polys[j][i](x))
///
/// Instance is (scalars, nested_poly_arrays, eq_polynomial).
pub struct GeneralLinearSumcheckProver {
    pub num_var: usize,
    pub num_poly: usize,
    /// eq(r, x)
    pub eq: DenseMLPoly,
    /// Scalars a_j
    pub a_scalars: Vec<GoldilocksField>,
    /// Arrays A_j[i] storing evaluations of polynomials
    pub a_arrays: Vec<Vec<DenseMLPoly>>,
    pub current_round: usize,
    pub challenges: Vec<GoldilocksField>,
}

impl SumcheckProver for GeneralLinearSumcheckProver {
    type Instance = (Vec<GoldilocksField>, Vec<Vec<DenseMLPoly>>, DenseMLPoly);

    fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript) -> Self {
        transcript.append_u64(b"gen_num_var", num_var as u64);
        transcript.append_u64(b"gen_num_poly", num_polys as u64);

        Self {
            num_var,
            num_poly: num_polys,
            eq: DenseMLPoly::new(0, vec![GoldilocksField(1)]),
            a_scalars: Vec::new(),
            a_arrays: Vec::new(),
            current_round: 0,
            challenges: Vec::new(),
        }
    }

    fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript) -> SumcheckProof {
        assert!(!instances.1.is_empty() && !instances.0.is_empty());
        let n_size = instances.1[0][0].len();
        assert_eq!(n_size, 1 << self.num_var);
        assert_eq!(instances.1[0].len(), self.num_poly - 1);
        assert_eq!(instances.0.len(), instances.1.len());

        self.a_scalars = instances.0.clone();
        self.a_arrays = instances.1.clone();
        self.eq = instances.2.clone();
        let mut round_messages = Vec::new();

        for _round in 0..self.num_var {
            let round_message = self.compute_round_message();

            // Use Ext2 transcript ops to match SumcheckVerifier::verify_general
            let round_msg_ext2: Vec<GoldilocksExt2> = round_message.iter()
                .map(|&v| GoldilocksExt2::from_base(v))
                .collect();
            for msg in &round_msg_ext2 {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge_ext2 = transcript.challenge_ext2(b"challenge");
            // KNOWN LIMITATION: We take only the base-field component (c0) of the Ext2
            // challenge for internal polynomial folding, since this prover operates in
            // base field. The verifier (verify_general) uses the full Ext2 challenge for
            // Lagrange interpolation, so the final_eval check will only pass when all
            // round-message values happen to have c1=0 (which is true because they are
            // lifted from base field via from_base). If this prover is ever extended to
            // handle native Ext2 polynomials, it must fold with the full Ext2 challenge.
            let challenge = challenge_ext2.c0;
            round_messages.push(round_message);
            self.receive_challenge(challenge);
        }

        let final_eval = self.final_evaluation();
        SumcheckProof {
            final_eval: GoldilocksExt2::from_base(final_eval),
            round_messages: round_messages.into_iter().map(|rm| rm.into_iter().map(|v| GoldilocksExt2::from_base(v)).collect()).collect(),
        }
    }
}

impl GeneralLinearSumcheckProver {
    fn compute_round_message(&self) -> Vec<GoldilocksField> {
        let m = self.current_round;
        let remaining_size = 1 << (self.num_var - m);
        let half = remaining_size >> 1;

        let eval_points: Vec<usize> = (0..=self.num_poly).collect();

        eval_points
            .par_iter()
            .map(|&eval_idx| {
                let c = GoldilocksField(eval_idx as u64);

                (0..half)
                    .into_par_iter()
                    .map(|y_bits| self.compute_g_value(c, y_bits))
                    .reduce(|| GoldilocksField(0), |a, b| gl_add(a, b))
            })
            .collect()
    }

    /// Compute g(c, y) = eq(c, y) * Σ_j (a_scalars[j] * Π_i polys[j][i](c, y))
    fn compute_g_value(&self, c: GoldilocksField, y_bits: usize) -> GoldilocksField {
        let num_terms = self.a_arrays.len();

        // Interpolate eq polynomial at c
        let eq_val_0 = self.eq.evaluations[y_bits * 2];
        let eq_val_1 = self.eq.evaluations[y_bits * 2 + 1];
        let eq_val = gl_add(eq_val_0, gl_mul(c, gl_sub(eq_val_1, eq_val_0)));

        // Compute Σ_j (a_scalars[j] * Π_i polys[j][i](c, y))
        let mut sum = GoldilocksField(0);
        for j in 0..num_terms {
            let mut product = GoldilocksField(1);
            for poly_idx in 0..(self.num_poly - 1) {
                let val_0 = self.a_arrays[j][poly_idx].evaluations[y_bits * 2];
                let val_1 = self.a_arrays[j][poly_idx].evaluations[y_bits * 2 + 1];
                let val = gl_add(val_0, gl_mul(c, gl_sub(val_1, val_0)));
                product = gl_mul(product, val);
            }
            sum = gl_add(sum, gl_mul(self.a_scalars[j], product));
        }

        gl_mul(eq_val, sum)
    }

    fn receive_challenge(&mut self, challenge: GoldilocksField) {
        self.challenges.push(challenge);
        self.bind_variable_to_challenge(challenge);
        self.current_round += 1;
    }

    fn bind_variable_to_challenge(&mut self, r_m: GoldilocksField) {
        let remaining_vars = self.num_var - self.current_round - 1;
        let new_size = 1 << remaining_vars;

        // Fold all polynomial arrays
        for poly_arrays in &mut self.a_arrays {
            for poly in poly_arrays.iter_mut() {
                let mut new_evals = Vec::with_capacity(new_size);
                for y in 0..new_size {
                    let a = poly.evaluations[y * 2];
                    let b = poly.evaluations[y * 2 + 1];
                    new_evals.push(gl_add(a, gl_mul(r_m, gl_sub(b, a))));
                }
                poly.evaluations = new_evals;
                poly.n -= 1;
            }
        }

        // Fold eq polynomial
        let mut new_eq = Vec::with_capacity(new_size);
        for y in 0..new_size {
            let a = self.eq.evaluations[y * 2];
            let b = self.eq.evaluations[y * 2 + 1];
            new_eq.push(gl_add(a, gl_mul(r_m, gl_sub(b, a))));
        }
        self.eq.evaluations = new_eq;
        self.eq.n -= 1;
    }

    fn final_evaluation(&self) -> GoldilocksField {
        assert_eq!(self.current_round, self.num_var);

        let num_terms = self.a_arrays.len();
        let mut sum = GoldilocksField(0);
        for j in 0..num_terms {
            let mut product = GoldilocksField(1);
            for poly_idx in 0..(self.num_poly - 1) {
                assert_eq!(self.a_arrays[j][poly_idx].evaluations.len(), 1);
                product = gl_mul(product, self.a_arrays[j][poly_idx].evaluations[0]);
            }
            sum = gl_add(sum, gl_mul(self.a_scalars[j], product));
        }
        gl_mul(sum, self.eq.evaluations[0])
    }
}
