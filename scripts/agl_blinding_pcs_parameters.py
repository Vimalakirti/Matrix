#!/usr/bin/env python3
"""Candidate Almost-Goldilocks parameters for Morpheus's masked RLC.

This script uses the standard-deviation convention

    D_sigma(x) proportional to exp(-||x||_2^2 / (2 sigma^2)).

The masking and same-point RLC challenges are one protocol phase.  A
coordinate fork therefore cancels the mask directly and needs relaxed
binding at 2*B_Z, rather than at 4*B_Z as in the older RLC-then-binary-
terminal construction.

The Module-SIS estimate mirrors the Euclidean conversion and MATZOV cost
model in SuperNeo Appendix D.8.  A final submission should rerun the pinned
Sage lattice-estimator version and record its commit hash.
"""

import math


Q = (2**64 - 2**32 + 1) - 32
RING_DEGREE = 64
MODULE_RANK = 42

MESSAGE_COEFFICIENTS = 2**33
HIDING_RING_COORDINATES = 2**17
HIDING_ALPHABET_SIZE = 3
HIDING_COEFFICIENTS = HIDING_RING_COORDINATES * RING_DEGREE
LINK_MASK_COEFFICIENTS = 2**17
COMMITMENT_COEFFICIENTS = MESSAGE_COEFFICIENTS + HIDING_COEFFICIENTS
TOTAL_COEFFICIENTS = COMMITMENT_COEFFICIENTS + LINK_MASK_COEFFICIENTS

SOURCE_BOUND = 2
DECOMPOSITION_DIGITS = 13
ACCUMULATED_BOUND = SOURCE_BOUND**DECOMPOSITION_DIGITS
EXPANSION_FACTOR = 128
MAX_FOLDED_CLAIMS = 50
CHALLENGE_BITS = 128

REPETITIONS = 2
GAUSSIAN_STDDEV_RATIO = 8
LIKELIHOOD_TAIL_EXPONENT = 95
RESPONSE_TAIL_BITS = 150
MAX_ATTEMPTS = 512


def root_hermite_factor(beta: int) -> float:
    if beta <= 40:
        return 1.01295
    return (
        beta / (2 * math.pi * math.e) * (math.pi * beta) ** (1 / beta)
    ) ** (1 / (2 * (beta - 1)))


def block_size(delta: float) -> int:
    for beta in range(40, 2**16 + 1):
        if root_hermite_factor(beta) < delta:
            return beta
    raise ValueError("required block size exceeds estimator search range")


def matzov_log2_cost(beta: int, lattice_dimension: int) -> float:
    slope = 0.29613500308205365
    intercept = 20.387885985467914
    progressive = 1 / (1 - 2 ** (-slope))
    dimensions_for_free = max(
        beta * math.log(4 / 3) / math.log(beta / (2 * math.pi * math.e)),
        0,
    )
    log_gate_cost = (
        math.log2(progressive)
        + slope * (beta - dimensions_for_free)
        + intercept
    )
    log_svp_calls = math.log2(progressive * max(lattice_dimension - beta, 1))
    log_lll_cost = 3 * math.log2(lattice_dimension)
    largest = max(log_lll_cost, log_svp_calls + log_gate_cost)
    return largest + math.log2(
        2 ** (log_lll_cost - largest)
        + 2 ** (log_svp_calls + log_gate_cost - largest)
    )


def sis_estimate(euclidean_bound: float):
    if euclidean_bound >= Q:
        return None
    module_dimension = MODULE_RANK * RING_DEGREE
    log_q = math.log2(Q)
    log_bound = math.log2(euclidean_bound)
    lattice_dimension = math.floor(2 * module_dimension * log_q / log_bound)
    log_delta = (
        log_bound - (module_dimension / lattice_dimension) * log_q
    ) / (lattice_dimension - 1)
    delta = 2**log_delta
    beta = block_size(delta)
    return {
        "bits": matzov_log2_cost(beta, lattice_dimension),
        "beta": beta,
        "lattice_dimension": lattice_dimension,
        "delta": delta,
    }


def main():
    assert (
        MAX_FOLDED_CLAIMS
        * EXPANSION_FACTOR
        * (SOURCE_BOUND - 1)
        < ACCUMULATED_BOUND
    )

    joint_shift_l2 = ACCUMULATED_BOUND * math.sqrt(
        REPETITIONS * TOTAL_COEFFICIENTS
    )
    sigma = GAUSSIAN_STDDEV_RATIO * joint_shift_l2

    rejection_constant = math.exp(
        math.sqrt(2 * LIKELIHOOD_TAIL_EXPONENT)
        / GAUSSIAN_STDDEV_RATIO
        + 1 / (2 * GAUSSIAN_STDDEV_RATIO**2)
    )
    rejection_error = (
        2 * math.exp(-LIKELIHOOD_TAIL_EXPONENT) / rejection_constant
    )
    accepted_response_error = (
        2
        * math.exp(-LIKELIHOOD_TAIL_EXPONENT)
        / (1 - 2 * math.exp(-LIKELIHOOD_TAIL_EXPONENT))
    )
    acceptance_lower_bound = (
        1 - 2 * math.exp(-LIKELIHOOD_TAIL_EXPONENT)
    ) / rejection_constant

    response_tail = 2 ** (-RESPONSE_TAIL_BITS)
    tail_factor = math.sqrt(
        2
        * math.log(
            2 * REPETITIONS * TOTAL_COEFFICIENTS / response_tail
        )
    )
    response_bound = math.ceil(sigma * tail_factor)

    # A masked-RLC coordinate fork gives z-z' below 2*B_Z and a challenge
    # difference in C-C.  SuperNeo Theorem 2 reduces (2*B_Z,C)-relaxed
    # binding to MSIS with coefficient bound 4*T*(2*B_Z) = 8*T*B_Z.
    msis_coefficient_bound = 8 * EXPANSION_FACTOR * response_bound
    msis_euclidean_bound = math.sqrt(
        COMMITMENT_COEFFICIENTS
    ) * msis_coefficient_bound
    link_msis_euclidean_bound = math.sqrt(
        LINK_MASK_COEFFICIENTS
    ) * msis_coefficient_bound

    retry_failure = (1 - acceptance_lower_bound + response_tail) ** MAX_ATTEMPTS
    interactive_knowledge_error = (
        MAX_FOLDED_CLAIMS / 2 ** (CHALLENGE_BITS * REPETITIONS)
    )

    output_bits = MODULE_RANK * RING_DEGREE * math.log2(Q)
    hiding_entropy = (
        HIDING_RING_COORDINATES * math.log2(HIDING_ALPHABET_SIZE)
    )
    hiding_advantage_log2 = -(hiding_entropy - output_bits) / 2

    print(f"q = {Q}")
    print(f"Phi = X^64 + 1, d = {RING_DEGREE}, kappa = {MODULE_RANK}")
    print(f"D_total = {TOTAL_COEFFICIENTS}")
    print(f"D_commitment = {COMMITMENT_COEFFICIENTS}")
    print(f"D_link = {LINK_MASK_COEFFICIENTS}")
    print(
        f"hiding ring coordinates = {HIDING_RING_COORDINATES}, "
        f"alphabet size = {HIDING_ALPHABET_SIZE}"
    )
    print(
        f"b = {SOURCE_BOUND}, k = {DECOMPOSITION_DIGITS}, "
        f"B = {ACCUMULATED_BOUND}"
    )
    print(f"T = {EXPANSION_FACTOR}, |E| <= {MAX_FOLDED_CLAIMS}")
    print(
        f"tau = {REPETITIONS}, alpha_std = {GAUSSIAN_STDDEV_RATIO}, "
        f"eta_rs = {LIKELIHOOD_TAIL_EXPONENT}"
    )
    print(f"joint shift l2 <= {joint_shift_l2:.12g}")
    print(f"sigma_std = {sigma:.12g}")
    print(f"B_Z = {response_bound}")
    print(
        f"M = {rejection_constant:.12g}, "
        f"acceptance >= {acceptance_lower_bound:.9f}"
    )
    print(f"log2 rejection error <= {math.log2(rejection_error):.6f}")
    print(
        "log2 accepted-response error <= "
        f"{math.log2(accepted_response_error):.6f}"
    )
    print(
        "log2 retry-hybrid rejection error <= "
        f"{math.log2(MAX_ATTEMPTS * rejection_error):.6f}"
    )
    print(f"log2 retry failure <= {math.log2(retry_failure):.6f}")
    print(
        "log2 interactive masked-RLC knowledge error <= "
        f"{math.log2(interactive_knowledge_error):.6f}"
    )
    print(f"MSIS infinity bound = {msis_coefficient_bound}")
    print(f"L MSIS Euclidean bound = {msis_euclidean_bound:.12g}")
    print(f"L bound/q = {msis_euclidean_bound / Q:.6f}")
    print(f"L estimate = {sis_estimate(msis_euclidean_bound)}")
    print(
        f"A_link MSIS Euclidean bound = "
        f"{link_msis_euclidean_bound:.12g}"
    )
    print(f"A_link bound/q = {link_msis_euclidean_bound / Q:.6f}")
    print(f"A_link estimate = {sis_estimate(link_msis_euclidean_bound)}")
    print(f"log2 commitment-hiding advantage <= {hiding_advantage_log2:.6f}")


if __name__ == "__main__":
    main()

