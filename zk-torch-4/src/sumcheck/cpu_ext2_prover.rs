//! CPU sumcheck prover operating directly on Ext2 polynomials.
//!
//! Drop-in replacement for [`crate::sumcheck::GpuLinearSumcheckProver`] on
//! small polynomials where launch overhead dominates. Uses the same
//! transcript labels (`append_ext2` / `challenge_ext2`) as the GPU prover and
//! the verifier, so proofs interoperate.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_mul, ext2_sub};

pub struct CpuLinearSumcheckProverExt2 {
    pub num_var: usize,
    pub num_poly: usize,
    pub challenges: Vec<AlmostGoldilocksExt2>,
    pub final_evals: Vec<AlmostGoldilocksExt2>,
}

impl CpuLinearSumcheckProverExt2 {
    pub fn new(num_var: usize, num_polys: usize, transcript: &mut Transcript) -> Self {
        transcript.append_u64(b"num_var", num_var as u64);
        transcript.append_u64(b"num_poly", num_polys as u64);
        Self {
            num_var,
            num_poly: num_polys,
            challenges: Vec::new(),
            final_evals: Vec::new(),
        }
    }

    /// Run the sumcheck protocol on `polys` (mutated as the prover folds).
    /// Each polynomial must have `2^num_var` Ext2 evaluations.
    pub fn prove(
        &mut self,
        polys: &mut [Vec<AlmostGoldilocksExt2>],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert_eq!(polys.len(), self.num_poly, "num_poly mismatch");
        assert!(!polys.is_empty());
        assert_eq!(polys[0].len(), 1 << self.num_var, "polynomial size mismatch");

        let mut round_messages = Vec::with_capacity(self.num_var);

        for _round in 0..self.num_var {
            let half = polys[0].len() / 2;
            let num_eval_points = self.num_poly + 1;
            let mut round_msg = vec![AlmostGoldilocksExt2::zero(); num_eval_points];

            for eval_idx in 0..num_eval_points {
                let c = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(eval_idx as u64));
                let mut sum = AlmostGoldilocksExt2::zero();
                for j in 0..half {
                    let mut product = AlmostGoldilocksExt2::one();
                    for poly in polys.iter() {
                        let a = poly[2 * j];
                        let b = poly[2 * j + 1];
                        let val = ext2_add(a, ext2_mul(c, ext2_sub(b, a)));
                        product = ext2_mul(product, val);
                    }
                    sum = ext2_add(sum, product);
                }
                round_msg[eval_idx] = sum;
            }

            for msg in &round_msg {
                transcript.append_ext2(b"round_message", msg);
            }
            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(round_msg);
            self.challenges.push(challenge);

            // Fold all polynomials with the new challenge.
            for poly in polys.iter_mut() {
                let mut new_evals = Vec::with_capacity(half);
                for j in 0..half {
                    let a = poly[2 * j];
                    let b = poly[2 * j + 1];
                    new_evals.push(ext2_add(a, ext2_mul(challenge, ext2_sub(b, a))));
                }
                *poly = new_evals;
            }
        }

        self.final_evals = polys
            .iter()
            .map(|p| {
                assert_eq!(p.len(), 1, "polynomial not fully reduced after prove");
                p[0]
            })
            .collect();
        let mut final_eval = AlmostGoldilocksExt2::one();
        for &e in &self.final_evals {
            final_eval = ext2_mul(final_eval, e);
        }

        SumcheckProof { final_eval, round_messages }
    }

    pub fn final_eval(&self, poly_idx: usize) -> AlmostGoldilocksExt2 {
        self.final_evals[poly_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sumcheck::SumcheckVerifier;
    use crate::util::arith::ext2_field_eq;

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    /// End-to-end prove → verify roundtrip with the verifier sharing the
    /// same transcript label (`test_cpu_ext2`).
    #[test]
    fn prove_then_verify_passes() {
        let p1: Vec<_> = (1..=4u64).map(lift).collect();
        let p2: Vec<_> = (5..=8u64).map(lift).collect();
        let expected = lift(1 * 5 + 2 * 6 + 3 * 7 + 4 * 8);

        let mut t_prove = Transcript::new(b"cpu-ext2-roundtrip");
        let mut prover = CpuLinearSumcheckProverExt2::new(2, 2, &mut t_prove);
        let mut polys = vec![p1, p2];
        let proof = prover.prove(&mut polys, &mut t_prove);

        let mut t_verify = Transcript::new(b"cpu-ext2-roundtrip");
        let (ok, challenges) =
            SumcheckVerifier::verify(&proof, expected, 2, 2, &mut t_verify);
        assert!(ok, "sumcheck verification failed");
        assert_eq!(challenges.len(), 2);
        // Verifier's challenges must agree with the prover's.
        for (a, b) in challenges.iter().zip(prover.challenges.iter()) {
            assert!(ext2_field_eq(*a, *b));
        }
    }

    /// Tampering with the proof flips the verifier outcome.
    #[test]
    fn tampered_proof_fails_verify() {
        let p1: Vec<_> = (1..=4u64).map(lift).collect();
        let p2: Vec<_> = (5..=8u64).map(lift).collect();
        let expected = lift(70);

        let mut t_prove = Transcript::new(b"tamper");
        let mut prover = CpuLinearSumcheckProverExt2::new(2, 2, &mut t_prove);
        let mut polys = vec![p1, p2];
        let mut proof = prover.prove(&mut polys, &mut t_prove);
        // Flip a bit of the first round message.
        proof.round_messages[0][0] = proof.round_messages[0][0] + lift(1);

        let mut t_verify = Transcript::new(b"tamper");
        let (ok, _) =
            SumcheckVerifier::verify(&proof, expected, 2, 2, &mut t_verify);
        assert!(!ok, "tampered proof should fail verification");
    }

    /// Roundtrip with three polynomials (degree-3 round messages).
    #[test]
    fn three_poly_roundtrip() {
        // p1 * p2 * p3 sumcheck.
        let p1: Vec<_> = vec![lift(1), lift(2), lift(3), lift(4)];
        let p2: Vec<_> = vec![lift(2), lift(3), lift(5), lift(7)];
        let p3: Vec<_> = vec![lift(11), lift(13), lift(17), lift(19)];
        // Expected H = Σ p1·p2·p3 = 1·2·11 + 2·3·13 + 3·5·17 + 4·7·19 = 22 + 78 + 255 + 532 = 887
        let expected = lift(887);

        let mut t_prove = Transcript::new(b"three-poly");
        let mut prover = CpuLinearSumcheckProverExt2::new(2, 3, &mut t_prove);
        let mut polys = vec![p1, p2, p3];
        let proof = prover.prove(&mut polys, &mut t_prove);

        // First round message must sum to the claim.
        let s0 = proof.round_messages[0][0] + proof.round_messages[0][1];
        assert!(ext2_field_eq(s0, expected));

        let mut t_verify = Transcript::new(b"three-poly");
        let (ok, _) =
            SumcheckVerifier::verify(&proof, expected, 2, 3, &mut t_verify);
        assert!(ok);
    }

    /// Larger n (cube of size 16) exercises multiple folding rounds.
    #[test]
    fn larger_arity_roundtrip() {
        let n_var = 4;
        let size = 1 << n_var;
        let p1: Vec<_> = (0..size as u64).map(|i| lift(i * 3 + 1)).collect();
        let p2: Vec<_> = (0..size as u64).map(|i| lift(i * 5 + 7)).collect();
        let mut expected = AlmostGoldilocksExt2::zero();
        for i in 0..size {
            expected = expected + p1[i] * p2[i];
        }

        let mut t_prove = Transcript::new(b"larger");
        let mut prover = CpuLinearSumcheckProverExt2::new(n_var, 2, &mut t_prove);
        let mut polys = vec![p1, p2];
        let proof = prover.prove(&mut polys, &mut t_prove);

        let mut t_verify = Transcript::new(b"larger");
        let (ok, _) =
            SumcheckVerifier::verify(&proof, expected, n_var, 2, &mut t_verify);
        assert!(ok);
    }
}
