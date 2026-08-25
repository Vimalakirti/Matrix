use goldilocks_cuda::GoldilocksExt2;

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_sub, ext2_mul};

/// CPU linear sumcheck prover operating directly on Ext2 polynomials.
///
/// Uses append_ext2/challenge_ext2 transcript pattern matching the GPU prover
/// and verifier, allowing it to be used as a drop-in replacement for
/// GpuLinearSumcheckProver on small polynomials where GPU launch overhead
/// dominates.
pub struct CpuLinearSumcheckProverExt2 {
    pub num_var: usize,
    pub num_poly: usize,
    pub challenges: Vec<GoldilocksExt2>,
    pub final_evals: Vec<GoldilocksExt2>,
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

    /// Prove sumcheck on Ext2 polynomials entirely on CPU.
    ///
    /// `polys` is a mutable slice of polynomials, each with 2^num_var elements.
    /// Each round computes P(0), P(1), P(2) for the round polynomial, then folds.
    pub fn prove(
        &mut self,
        polys: &mut [Vec<GoldilocksExt2>],
        transcript: &mut Transcript,
    ) -> SumcheckProof {
        assert_eq!(polys.len(), self.num_poly);
        assert!(!polys.is_empty());
        assert_eq!(polys[0].len(), 1 << self.num_var);

        let mut round_messages = Vec::with_capacity(self.num_var);

        for _round in 0..self.num_var {
            let half = polys[0].len() / 2;

            // Compute round message: [P(0), P(1), P(2)]
            // P(k) = sum_j prod_i poly_i(2j + k_interp)
            // where k_interp means evaluating the line through (poly[2j], poly[2j+1]) at k
            let num_eval_points = self.num_poly + 1; // degree = num_poly - 1, need num_poly + 1 points
            let mut round_msg = vec![GoldilocksExt2::zero(); num_eval_points];

            for eval_idx in 0..num_eval_points {
                let c = GoldilocksExt2::from_base(goldilocks_cuda::GoldilocksField(eval_idx as u64));
                let mut sum = GoldilocksExt2::zero();

                for j in 0..half {
                    let mut product = GoldilocksExt2::one();
                    for poly in polys.iter() {
                        let a = poly[2 * j];     // poly(0, j)
                        let b = poly[2 * j + 1]; // poly(1, j)
                        // poly(c, j) = a + c * (b - a)
                        let val = ext2_add(a, ext2_mul(c, ext2_sub(b, a)));
                        product = ext2_mul(product, val);
                    }
                    sum = ext2_add(sum, product);
                }

                round_msg[eval_idx] = sum;
            }

            // Append to transcript (must match GPU prover / verifier pattern)
            for msg in &round_msg {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge = transcript.challenge_ext2(b"challenge");
            round_messages.push(round_msg);
            self.challenges.push(challenge);

            // Fold: poly[j] = poly[2j] * (1 - r) + poly[2j+1] * r
            //                = poly[2j] + r * (poly[2j+1] - poly[2j])
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

        // Final evaluations
        self.final_evals = polys.iter().map(|p| {
            assert_eq!(p.len(), 1);
            p[0]
        }).collect();

        let mut final_eval = GoldilocksExt2::one();
        for &e in &self.final_evals {
            final_eval = ext2_mul(final_eval, e);
        }

        SumcheckProof {
            final_eval,
            round_messages,
        }
    }

    pub fn final_eval(&self, poly_idx: usize) -> GoldilocksExt2 {
        self.final_evals[poly_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
    use crate::transcript::Transcript;
    use crate::sumcheck::SumcheckVerifier;

    #[test]
    fn test_cpu_ext2_prover_basic() {
        // Two Ext2 polys of 2 variables (4 elements each)
        let p1: Vec<GoldilocksExt2> = vec![
            GoldilocksExt2::from_base(GoldilocksField(1)),
            GoldilocksExt2::from_base(GoldilocksField(2)),
            GoldilocksExt2::from_base(GoldilocksField(3)),
            GoldilocksExt2::from_base(GoldilocksField(4)),
        ];
        let p2: Vec<GoldilocksExt2> = vec![
            GoldilocksExt2::from_base(GoldilocksField(5)),
            GoldilocksExt2::from_base(GoldilocksField(6)),
            GoldilocksExt2::from_base(GoldilocksField(7)),
            GoldilocksExt2::from_base(GoldilocksField(8)),
        ];

        // Expected sum: 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        let expected_sum = GoldilocksExt2::from_base(GoldilocksField(70));

        let mut transcript = Transcript::new(b"test_cpu_ext2");
        let mut prover = CpuLinearSumcheckProverExt2::new(2, 2, &mut transcript);
        let mut polys = vec![p1, p2];
        let proof = prover.prove(&mut polys, &mut transcript);

        // Verify: s(0) + s(1) == 70
        let round0_sum = ext2_add(proof.round_messages[0][0], proof.round_messages[0][1]);
        assert_eq!(round0_sum, expected_sum);

        // Also verify with SumcheckVerifier
        let mut verify_transcript = Transcript::new(b"test_cpu_ext2");
        let (verified, _challenges) = SumcheckVerifier::verify(
            &proof, expected_sum, 2, 2, &mut verify_transcript,
        );
        assert!(verified, "CPU Ext2 proof should verify");
    }
}
