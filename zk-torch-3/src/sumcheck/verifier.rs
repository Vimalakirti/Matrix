use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_sub, ext2_mul, ext2_inv};

/// Sumcheck verifier — runs on CPU (lightweight).
pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verify a sumcheck proof.
    ///
    /// Returns (verified, challenges, final_eval) if the proof is valid.
    /// The verifier checks:
    /// 1. For each round i: s_i(0) + s_i(1) = claimed_sum (or previous round's evaluation)
    /// 2. The final evaluation matches the prover's claim
    pub fn verify(
        proof: &SumcheckProof,
        claimed_sum: GoldilocksExt2,
        num_var: usize,
        num_poly: usize,
        transcript: &mut Transcript,
    ) -> (bool, Vec<GoldilocksExt2>) {
        transcript.append_u64(b"num_var", num_var as u64);
        transcript.append_u64(b"num_poly", num_poly as u64);

        if proof.round_messages.len() != num_var {
            return (false, vec![]);
        }

        let mut challenges = Vec::with_capacity(num_var);
        let mut current_sum = claimed_sum;

        for round in 0..num_var {
            let round_msg = &proof.round_messages[round];

            // Check: round message has exactly num_poly + 1 evaluations (degree check)
            if round_msg.len() != num_poly + 1 {
                return (false, vec![]);
            }
            let s0 = round_msg[0];
            let s1 = round_msg[1];
            let sum = ext2_add(s0, s1);

            if sum != current_sum {
                return (false, vec![]);
            }

            // Append round message to transcript
            for msg in round_msg {
                transcript.append_ext2(b"round_message", msg);
            }

            // Get challenge
            let challenge = transcript.challenge_ext2(b"challenge");
            challenges.push(challenge);

            // Interpolate round polynomial at challenge point
            current_sum = interpolate_and_evaluate_ext2(round_msg, challenge);
        }

        // Check final evaluation
        let verified = current_sum == proof.final_eval;
        (verified, challenges)
    }

    /// Verify a general sumcheck proof (same algorithm, different transcript labels).
    pub fn verify_general(
        proof: &SumcheckProof,
        claimed_sum: GoldilocksExt2,
        num_var: usize,
        num_poly: usize,
        transcript: &mut Transcript,
    ) -> (bool, Vec<GoldilocksExt2>) {
        transcript.append_u64(b"gen_num_var", num_var as u64);
        transcript.append_u64(b"gen_num_poly", num_poly as u64);

        if proof.round_messages.len() != num_var {
            return (false, vec![]);
        }

        let mut challenges = Vec::with_capacity(num_var);
        let mut current_sum = claimed_sum;

        for round in 0..num_var {
            let round_msg = &proof.round_messages[round];

            // Check: round message has exactly num_poly + 1 evaluations (degree check)
            if round_msg.len() != num_poly + 1 {
                return (false, vec![]);
            }
            let s0 = round_msg[0];
            let s1 = round_msg[1];
            let sum = ext2_add(s0, s1);

            if sum != current_sum {
                return (false, vec![]);
            }

            for msg in round_msg {
                transcript.append_ext2(b"round_message", msg);
            }

            let challenge = transcript.challenge_ext2(b"challenge");
            challenges.push(challenge);

            current_sum = interpolate_and_evaluate_ext2(round_msg, challenge);
        }

        let verified = current_sum == proof.final_eval;
        (verified, challenges)
    }
}

/// Lagrange interpolation over Ext2: given evaluations at {0, 1, ..., d}, evaluate at x.
fn interpolate_and_evaluate_ext2(evals: &[GoldilocksExt2], x: GoldilocksExt2) -> GoldilocksExt2 {
    let d = evals.len();
    let mut result = GoldilocksExt2::zero();

    for i in 0..d {
        let mut basis = GoldilocksExt2::one();
        let xi = GoldilocksExt2::from_base(GoldilocksField(i as u64));
        for j in 0..d {
            if j != i {
                let xj = GoldilocksExt2::from_base(GoldilocksField(j as u64));
                let num = ext2_sub(x, xj);
                let den = ext2_sub(xi, xj);
                let den_inv = ext2_inv(den);
                basis = ext2_mul(basis, ext2_mul(num, den_inv));
            }
        }
        result = ext2_add(result, ext2_mul(basis, evals[i]));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_ext2() {
        // Linear function: f(x) = 3 + 2x (base field values embedded in Ext2)
        // f(0) = 3, f(1) = 5
        let evals = vec![
            GoldilocksExt2::from_base(GoldilocksField(3)),
            GoldilocksExt2::from_base(GoldilocksField(5)),
        ];
        let result = interpolate_and_evaluate_ext2(
            &evals,
            GoldilocksExt2::from_base(GoldilocksField(2)),
        );
        assert_eq!(result, GoldilocksExt2::from_base(GoldilocksField(7))); // 3 + 2*2 = 7
    }
}
