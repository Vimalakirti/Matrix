use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use rayon::prelude::*;

use crate::poly::DenseMLPoly;
use crate::sumcheck::{SumcheckProof, SumcheckProver};
use crate::transcript::Transcript;
use crate::util::arith::{gl_add, gl_mul, gl_sub};

/// Optimal linear sum-check protocol.
///
/// Computes: H = Σ_{x ∈ {0,1}^n} Π_{i=1}^ℓ p_i(x)
/// where each p_i is a dense multilinear polynomial.
pub struct LinearSumcheckProver {
    pub num_var: usize,
    pub num_poly: usize,
    pub a_arrays: Vec<DenseMLPoly>,
    pub current_round: usize,
    pub challenges: Vec<GoldilocksField>,
}

impl SumcheckProver for LinearSumcheckProver {
    type Instance = Vec<DenseMLPoly>;

    fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript) -> Self {
        transcript.append_u64(b"num_var", num_var as u64);
        transcript.append_u64(b"num_poly", num_polys as u64);

        Self {
            num_var,
            num_poly: num_polys,
            a_arrays: Vec::new(),
            current_round: 0,
            challenges: Vec::new(),
        }
    }

    fn prove(&mut self, instances: &Self::Instance, transcript: &mut Transcript) -> SumcheckProof {
        assert!(!instances.is_empty());
        let n_size = instances[0].len();
        assert_eq!(n_size, 1 << self.num_var);
        assert_eq!(instances.len(), self.num_poly);

        self.a_arrays = instances.clone();
        let mut round_messages = Vec::new();

        for _round in 0..self.num_var {
            let round_message = self.compute_round_message();

            for &msg in &round_message {
                transcript.append_scalar(b"round_message", &msg);
            }

            let challenge = transcript.challenge_scalar(b"challenge");
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

impl LinearSumcheckProver {
    /// Compute prover message for the current round.
    /// Returns evaluations s_m(c) for c ∈ {0, 1, ..., ℓ}.
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

    /// Compute g(c, y) = Π_{i} p_i(c, y) for a specific c and y.
    fn compute_g_value(&self, c: GoldilocksField, y_bits: usize) -> GoldilocksField {
        let mut product = GoldilocksField(1);
        for poly in &self.a_arrays {
            let evals = &poly.evaluations;
            let a = evals[2 * y_bits]; // p(0, y)
            let b = evals[2 * y_bits + 1]; // p(1, y)
            // p(c, y) = a + c * (b - a) = a * (1 - c) + b * c
            let val = gl_add(a, gl_mul(c, gl_sub(b, a)));
            product = gl_mul(product, val);
        }
        product
    }

    /// Receive a challenge and fold all polynomial arrays.
    fn receive_challenge(&mut self, challenge: GoldilocksField) {
        self.challenges.push(challenge);

        // Fold each polynomial
        for poly in &mut self.a_arrays {
            let half = poly.evaluations.len() / 2;
            let mut new_evals = Vec::with_capacity(half);
            for j in 0..half {
                let a = poly.evaluations[2 * j];
                let b = poly.evaluations[2 * j + 1];
                new_evals.push(gl_add(a, gl_mul(challenge, gl_sub(b, a))));
            }
            poly.evaluations = new_evals;
            poly.n -= 1;
        }

        self.current_round += 1;
    }

    /// Get the final evaluation after all rounds.
    fn final_evaluation(&self) -> GoldilocksField {
        let mut product = GoldilocksField(1);
        for poly in &self.a_arrays {
            assert_eq!(poly.evaluations.len(), 1);
            product = gl_mul(product, poly.evaluations[0]);
        }
        product
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_sumcheck_basic() {
        // Test with two polynomials of 2 variables
        // p1 = [1, 2, 3, 4], p2 = [5, 6, 7, 8]
        // H = Σ p1(x) * p2(x) = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let p1 = DenseMLPoly::new(2, vec![
            GoldilocksField(1), GoldilocksField(2),
            GoldilocksField(3), GoldilocksField(4),
        ]);
        let p2 = DenseMLPoly::new(2, vec![
            GoldilocksField(5), GoldilocksField(6),
            GoldilocksField(7), GoldilocksField(8),
        ]);

        let mut transcript = Transcript::new(b"test");
        let mut prover = LinearSumcheckProver::new(2, 2, &mut transcript);

        let mut verify_transcript = Transcript::new(b"test");
        LinearSumcheckProver::new(2, 2, &mut verify_transcript);

        let proof = prover.prove(&vec![p1, p2], &mut transcript);

        // Verify: sum of round_messages[0][0] + round_messages[0][1] should equal H
        let round0_sum = proof.round_messages[0][0] + proof.round_messages[0][1];
        assert_eq!(round0_sum, GoldilocksExt2::from_base(GoldilocksField(70)));
    }
}
