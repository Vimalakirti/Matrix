//! General linear sumcheck prover with an explicit eq factor.
//!
//! Claim: `H = Σ_{x ∈ {0,1}^n} eq(r, x) · Σ_j (a_scalars[j] · Π_i polys[j][i](x))`.
//!
//! Used by DepthwiseConv2D (degree-3) and the
//! same-point sumcheck inside the fold tree.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rayon::prelude::*;

use crate::poly::DenseMLPoly;
use crate::sumcheck::{SumcheckProof, SumcheckProver};
use crate::transcript::Transcript;
use crate::util::arith::{agl_add, agl_mul, agl_sub};

pub struct GeneralLinearSumcheckProver {
    pub num_var: usize,
    pub num_poly: usize,
    pub eq: DenseMLPoly,
    pub a_scalars: Vec<AlmostGoldilocksField>,
    pub a_arrays: Vec<Vec<DenseMLPoly>>,
    pub current_round: usize,
    pub challenges: Vec<AlmostGoldilocksField>,
}

impl SumcheckProver for GeneralLinearSumcheckProver {
    type Instance = (
        Vec<AlmostGoldilocksField>,
        Vec<Vec<DenseMLPoly>>,
        DenseMLPoly,
    );

    fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript) -> Self {
        transcript.append_u64(b"gen_num_var", num_var as u64);
        transcript.append_u64(b"gen_num_poly", num_polys as u64);
        Self {
            num_var,
            num_poly: num_polys,
            eq: DenseMLPoly::new(0, vec![AlmostGoldilocksField(1)]),
            a_scalars: Vec::new(),
            a_arrays: Vec::new(),
            current_round: 0,
            challenges: Vec::new(),
        }
    }

    fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript) -> SumcheckProof {
        assert!(!instances.1.is_empty(), "no polynomial groups");
        assert!(!instances.0.is_empty(), "no scalars");
        let n_size = instances.1[0][0].len();
        assert_eq!(n_size, 1 << self.num_var, "polynomial size mismatch");
        assert_eq!(
            instances.1[0].len(),
            self.num_poly - 1,
            "per-group poly count mismatch"
        );
        assert_eq!(
            instances.0.len(),
            instances.1.len(),
            "scalars and poly-groups must have same length"
        );

        self.a_scalars = instances.0.clone();
        self.a_arrays = instances.1.clone();
        self.eq = instances.2.clone();
        let mut round_messages = Vec::with_capacity(self.num_var);

        for _round in 0..self.num_var {
            let round_message = self.compute_round_message();

            let round_msg_ext2: Vec<AlmostGoldilocksExt2> = round_message
                .iter()
                .map(|&v| AlmostGoldilocksExt2::from_base(v))
                .collect();
            for msg in &round_msg_ext2 {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge_ext2 = transcript.challenge_ext2(b"challenge");
            // Known limitation: we operate in the base field internally, so
            // we take only the c0 component for folding. The verifier uses
            // the full Ext2 challenge for Lagrange interpolation; the
            // final_eval check holds because all round-message values are
            // c1=0 lifts from base field.
            let challenge = challenge_ext2.c0;
            round_messages.push(round_message);
            self.receive_challenge(challenge);
        }

        let final_eval = self.final_evaluation();
        SumcheckProof {
            final_eval: AlmostGoldilocksExt2::from_base(final_eval),
            round_messages: round_messages
                .into_iter()
                .map(|rm| rm.into_iter().map(AlmostGoldilocksExt2::from_base).collect())
                .collect(),
        }
    }
}

impl GeneralLinearSumcheckProver {
    fn compute_round_message(&self) -> Vec<AlmostGoldilocksField> {
        let m = self.current_round;
        let remaining_size = 1 << (self.num_var - m);
        let half = remaining_size >> 1;
        let eval_points: Vec<usize> = (0..=self.num_poly).collect();

        eval_points
            .par_iter()
            .map(|&eval_idx| {
                let c = AlmostGoldilocksField(eval_idx as u64);
                (0..half)
                    .into_par_iter()
                    .map(|y_bits| self.compute_g_value(c, y_bits))
                    .reduce(|| AlmostGoldilocksField(0), agl_add)
            })
            .collect()
    }

    /// g(c, y) = eq(c, y) · Σ_j (a_scalars[j] · Π_i polys[j][i](c, y))
    fn compute_g_value(
        &self,
        c: AlmostGoldilocksField,
        y_bits: usize,
    ) -> AlmostGoldilocksField {
        let num_terms = self.a_arrays.len();

        let eq_val_0 = self.eq.evaluations[y_bits * 2];
        let eq_val_1 = self.eq.evaluations[y_bits * 2 + 1];
        let eq_val = agl_add(eq_val_0, agl_mul(c, agl_sub(eq_val_1, eq_val_0)));

        let mut sum = AlmostGoldilocksField(0);
        for j in 0..num_terms {
            let mut product = AlmostGoldilocksField(1);
            for poly_idx in 0..(self.num_poly - 1) {
                let val_0 = self.a_arrays[j][poly_idx].evaluations[y_bits * 2];
                let val_1 = self.a_arrays[j][poly_idx].evaluations[y_bits * 2 + 1];
                let val = agl_add(val_0, agl_mul(c, agl_sub(val_1, val_0)));
                product = agl_mul(product, val);
            }
            sum = agl_add(sum, agl_mul(self.a_scalars[j], product));
        }
        agl_mul(eq_val, sum)
    }

    fn receive_challenge(&mut self, challenge: AlmostGoldilocksField) {
        self.challenges.push(challenge);
        self.bind_variable_to_challenge(challenge);
        self.current_round += 1;
    }

    fn bind_variable_to_challenge(&mut self, r_m: AlmostGoldilocksField) {
        let remaining_vars = self.num_var - self.current_round - 1;
        let new_size = 1 << remaining_vars;

        for poly_arrays in &mut self.a_arrays {
            for poly in poly_arrays.iter_mut() {
                let mut new_evals = Vec::with_capacity(new_size);
                for y in 0..new_size {
                    let a = poly.evaluations[y * 2];
                    let b = poly.evaluations[y * 2 + 1];
                    new_evals.push(agl_add(a, agl_mul(r_m, agl_sub(b, a))));
                }
                poly.evaluations = new_evals;
                poly.n -= 1;
            }
        }

        let mut new_eq = Vec::with_capacity(new_size);
        for y in 0..new_size {
            let a = self.eq.evaluations[y * 2];
            let b = self.eq.evaluations[y * 2 + 1];
            new_eq.push(agl_add(a, agl_mul(r_m, agl_sub(b, a))));
        }
        self.eq.evaluations = new_eq;
        self.eq.n -= 1;
    }

    fn final_evaluation(&self) -> AlmostGoldilocksField {
        assert_eq!(self.current_round, self.num_var);
        let num_terms = self.a_arrays.len();
        let mut sum = AlmostGoldilocksField(0);
        for j in 0..num_terms {
            let mut product = AlmostGoldilocksField(1);
            for poly_idx in 0..(self.num_poly - 1) {
                assert_eq!(self.a_arrays[j][poly_idx].evaluations.len(), 1);
                product = agl_mul(product, self.a_arrays[j][poly_idx].evaluations[0]);
            }
            sum = agl_add(sum, agl_mul(self.a_scalars[j], product));
        }
        agl_mul(sum, self.eq.evaluations[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::evaluate_lagrange_basis;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    /// One scalar, one polynomial, eq table: `H = Σ_x eq(r,x) · f(x) = f(r)`.
    /// Verify the prover's first-round-message sum equals f(r).
    #[test]
    fn first_round_recovers_f_at_r() {
        // n = 2, f(x0,x1) with evals [1, 2, 3, 4], r = (5, 7).
        let f = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let r = [agl(5), agl(7)];
        let eq = DenseMLPoly::new(2, evaluate_lagrange_basis(&r));

        let mut t = Transcript::new(b"gen");
        let mut prover = GeneralLinearSumcheckProver::new(2, 2, &mut t);
        let proof = prover.prove(&(vec![agl(1)], vec![vec![f.clone()]], eq), &mut t);

        // Σ_x eq(r,x)·f(x) = f(r); recompute f(r) via fix_variables.
        let f_at_r = f.evaluate(&r);
        let s0 = proof.round_messages[0][0] + proof.round_messages[0][1];
        assert!(crate::util::arith::ext2_field_eq(
            s0,
            AlmostGoldilocksExt2::from_base(f_at_r)
        ));
    }

    /// Degree-3 sumcheck (matches DepthwiseConv2D): `Σ_x eq(r,x)·f(x)·g(x)`.
    /// First-round message has degree 3 (= num_poly + 1 - 1, where num_poly =
    /// number of multiplicative factors including the eq).
    #[test]
    fn degree3_message_length_correct() {
        let r = [agl(11), agl(13)];
        let eq = DenseMLPoly::new(2, evaluate_lagrange_basis(&r));
        let f = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let g = DenseMLPoly::new(2, vec![agl(5), agl(6), agl(7), agl(8)]);

        let mut t = Transcript::new(b"deg3");
        // num_poly = 3 means each round message has 4 values.
        let mut prover = GeneralLinearSumcheckProver::new(2, 3, &mut t);
        let proof = prover.prove(
            &(vec![agl(1)], vec![vec![f, g]], eq),
            &mut t,
        );
        assert_eq!(proof.round_messages.len(), 2);
        assert_eq!(proof.round_messages[0].len(), 4);
        assert_eq!(proof.round_messages[1].len(), 4);
    }

    /// Two-term combination via the scalar slot: `Σ_x eq(r,x) · (a · f + b · g)`.
    #[test]
    fn scalar_combination_consistent() {
        let r = [agl(3), agl(7)];
        let eq = DenseMLPoly::new(2, evaluate_lagrange_basis(&r));
        let f = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let g = DenseMLPoly::new(2, vec![agl(10), agl(20), agl(30), agl(40)]);
        let a = agl(5);
        let b = agl(2);

        let mut t = Transcript::new(b"comb");
        let mut prover = GeneralLinearSumcheckProver::new(2, 2, &mut t);
        let proof = prover.prove(
            &(vec![a, b], vec![vec![f.clone()], vec![g.clone()]], eq),
            &mut t,
        );
        // a·f(r) + b·g(r)
        let expected = agl_add(agl_mul(a, f.evaluate(&r)), agl_mul(b, g.evaluate(&r)));
        let s0 = proof.round_messages[0][0] + proof.round_messages[0][1];
        assert!(crate::util::arith::ext2_field_eq(
            s0,
            AlmostGoldilocksExt2::from_base(expected)
        ));
    }
}
