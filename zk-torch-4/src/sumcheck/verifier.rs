//! Sumcheck verifier — runs on CPU and is lightweight. Mirrors zk-torch-3.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_inv, ext2_mul, ext2_sub};

pub struct SumcheckVerifier;

impl SumcheckVerifier {
    /// Verify a sumcheck proof against `claimed_sum`. Returns
    /// `(verified, challenges)`; on failure, `challenges` is empty.
    pub fn verify(
        proof: &SumcheckProof,
        claimed_sum: AlmostGoldilocksExt2,
        num_var: usize,
        num_poly: usize,
        transcript: &mut Transcript,
    ) -> (bool, Vec<AlmostGoldilocksExt2>) {
        Self::verify_inner(proof, claimed_sum, num_var, num_poly, transcript, false)
    }

    /// "General" sumcheck verification — same algorithm, different
    /// transcript labels (matches `GeneralLinearSumcheckProver`'s
    /// `gen_num_var` / `gen_num_poly` prefixes).
    pub fn verify_general(
        proof: &SumcheckProof,
        claimed_sum: AlmostGoldilocksExt2,
        num_var: usize,
        num_poly: usize,
        transcript: &mut Transcript,
    ) -> (bool, Vec<AlmostGoldilocksExt2>) {
        Self::verify_inner(proof, claimed_sum, num_var, num_poly, transcript, true)
    }

    fn verify_inner(
        proof: &SumcheckProof,
        claimed_sum: AlmostGoldilocksExt2,
        num_var: usize,
        num_poly: usize,
        transcript: &mut Transcript,
        general: bool,
    ) -> (bool, Vec<AlmostGoldilocksExt2>) {
        let (var_label, poly_label): (&[u8], &[u8]) = if general {
            (b"gen_num_var", b"gen_num_poly")
        } else {
            (b"num_var", b"num_poly")
        };
        transcript.append_u64(var_label, num_var as u64);
        transcript.append_u64(poly_label, num_polys_as_u64(num_poly));

        if proof.round_messages.len() != num_var {
            return (false, vec![]);
        }

        let mut challenges = Vec::with_capacity(num_var);
        let mut current_sum = claimed_sum;

        for round in 0..num_var {
            let round_msg = &proof.round_messages[round];
            if round_msg.len() != num_poly + 1 {
                return (false, vec![]);
            }
            let s0 = round_msg[0];
            let s1 = round_msg[1];
            let sum = ext2_add(s0, s1);
            if !crate::util::arith::ext2_field_eq(sum, current_sum) {
                return (false, vec![]);
            }
            for msg in round_msg {
                transcript.append_ext2(b"round_message", msg);
            }
            let challenge = transcript.challenge_ext2(b"challenge");
            challenges.push(challenge);
            current_sum = interpolate_and_evaluate_ext2(round_msg, challenge);
        }

        let verified = crate::util::arith::ext2_field_eq(current_sum, proof.final_eval);
        (verified, challenges)
    }
}

#[inline]
fn num_polys_as_u64(n: usize) -> u64 {
    n as u64
}

/// Lagrange-interpolate `evals` (defined at `x = 0, 1, …, d − 1`) and
/// evaluate at `x`.
pub(crate) fn interpolate_and_evaluate_ext2(
    evals: &[AlmostGoldilocksExt2],
    x: AlmostGoldilocksExt2,
) -> AlmostGoldilocksExt2 {
    let d = evals.len();
    let mut result = AlmostGoldilocksExt2::zero();
    for i in 0..d {
        let mut basis = AlmostGoldilocksExt2::one();
        let xi = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(i as u64));
        for j in 0..d {
            if j != i {
                let xj = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(j as u64));
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
    fn interpolation_at_known_points_recovers_inputs() {
        // f(x) = 7 + 11x + 13x^2 — evaluate at 0, 1, 2.
        let vals: Vec<u64> = (0u64..3)
            .map(|x| 7 + 11 * x + 13 * x * x)
            .collect();
        let evals: Vec<_> = vals
            .iter()
            .map(|&v| AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v)))
            .collect();
        for x_raw in 0u64..3 {
            let x = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(x_raw));
            let v = interpolate_and_evaluate_ext2(&evals, x);
            assert!(
                crate::util::arith::ext2_field_eq(
                    v,
                    AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(vals[x_raw as usize])),
                ),
                "x={}",
                x_raw
            );
        }
    }

    #[test]
    fn interpolation_at_off_grid_point() {
        // f(x) = 3 + 2x → f(2) = 7
        let evals = vec![
            AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(3)),
            AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(5)),
        ];
        let result = interpolate_and_evaluate_ext2(
            &evals,
            AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2)),
        );
        assert!(crate::util::arith::ext2_field_eq(
            result,
            AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(7))
        ));
    }
}
