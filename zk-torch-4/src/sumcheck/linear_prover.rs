//! Linear sumcheck prover over the base field (CPU).
//!
//! Claim: `H = Σ_{x ∈ {0,1}^n} Π_i p_i(x)`.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use rayon::prelude::*;

use crate::poly::DenseMLPoly;
use crate::sumcheck::{SumcheckProof, SumcheckProver};
use crate::transcript::Transcript;
use crate::util::arith::{agl_add, agl_mul, agl_sub};

pub struct LinearSumcheckProver {
    pub num_var: usize,
    pub num_poly: usize,
    pub a_arrays: Vec<DenseMLPoly>,
    pub current_round: usize,
    pub challenges: Vec<AlmostGoldilocksField>,
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
        assert!(!instances.is_empty(), "LinearSumcheckProver::prove: no polynomials");
        let n_size = instances[0].len();
        assert_eq!(n_size, 1 << self.num_var, "polynomial size mismatch");
        assert_eq!(instances.len(), self.num_poly, "num_poly mismatch");

        self.a_arrays = instances.clone();
        let mut round_messages = Vec::with_capacity(self.num_var);

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
            final_eval: AlmostGoldilocksExt2::from_base(final_eval),
            round_messages: round_messages
                .into_iter()
                .map(|rm| rm.into_iter().map(AlmostGoldilocksExt2::from_base).collect())
                .collect(),
        }
    }
}

impl LinearSumcheckProver {
    /// Round-`m` message: `s_m(c)` for `c ∈ {0, 1, …, num_poly}`.
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

    fn compute_g_value(
        &self,
        c: AlmostGoldilocksField,
        y_bits: usize,
    ) -> AlmostGoldilocksField {
        let mut product = AlmostGoldilocksField(1);
        for poly in &self.a_arrays {
            let evals = &poly.evaluations;
            let a = evals[2 * y_bits];
            let b = evals[2 * y_bits + 1];
            let val = agl_add(a, agl_mul(c, agl_sub(b, a)));
            product = agl_mul(product, val);
        }
        product
    }

    fn receive_challenge(&mut self, challenge: AlmostGoldilocksField) {
        self.challenges.push(challenge);
        for poly in &mut self.a_arrays {
            let half = poly.evaluations.len() / 2;
            let mut new_evals = Vec::with_capacity(half);
            for j in 0..half {
                let a = poly.evaluations[2 * j];
                let b = poly.evaluations[2 * j + 1];
                new_evals.push(agl_add(a, agl_mul(challenge, agl_sub(b, a))));
            }
            poly.evaluations = new_evals;
            poly.n -= 1;
        }
        self.current_round += 1;
    }

    fn final_evaluation(&self) -> AlmostGoldilocksField {
        let mut product = AlmostGoldilocksField(1);
        for poly in &self.a_arrays {
            assert_eq!(poly.evaluations.len(), 1, "final_evaluation: poly not fully reduced");
            product = agl_mul(product, poly.evaluations[0]);
        }
        product
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    /// First-round message sums to the protocol claim H.
    #[test]
    fn first_round_message_sums_to_claim() {
        // p1·p2 over n=2 variables, expected H = 1·5 + 2·6 + 3·7 + 4·8 = 70.
        let p1 = DenseMLPoly::new(2, vec![agl(1), agl(2), agl(3), agl(4)]);
        let p2 = DenseMLPoly::new(2, vec![agl(5), agl(6), agl(7), agl(8)]);

        let mut transcript = Transcript::new(b"test");
        let mut prover = LinearSumcheckProver::new(2, 2, &mut transcript);
        let proof = prover.prove(&vec![p1, p2], &mut transcript);

        let round0 = &proof.round_messages[0];
        let sum = round0[0] + round0[1];
        assert!(crate::util::arith::ext2_field_eq(
            sum,
            AlmostGoldilocksExt2::from_base(agl(70))
        ));
    }
}
