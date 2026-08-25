//! Masked random linear combination `Π_mRLC` — the weak, commitment-consistent
//! stage that closes the opening.
//!
//! For each repetition `ℓ ∈ [τ]` the prover samples a discrete Gaussian mask
//! `U_ℓ`, publishes `D_ℓ = F_ξ(U_ℓ)`, receives ring challenges `β_{ℓ,e}`, and
//! answers with `Z_ℓ = U_ℓ + Σ_e β_{ℓ,e}·W_e` subject to one joint rejection
//! rule. The verifier checks
//!
//! ```text
//!   ‖Z_ℓ‖_∞ < B_Z
//!   F_ξ(Z_ℓ) = D_ℓ + Σ_e β_{ℓ,e}·(C_e, a_e)
//! ```
//!
//! Extraction forks on one challenge coordinate: differencing two accepting
//! transcripts leaves `F_ξ(Z−Z') = (β−β')·(C_e, a_e)`, and strong sampling makes
//! the difference invertible, so normalizing yields a relaxed witness bounded by
//! `2B_Z`. That is why relaxed binding is required at `2B_Z` rather than `B_Z`.
//!
//! ## Why the evaluation half is ring-valued
//!
//! `F_ξ(W) = (L(W), Ev(W,ξ))` has to be a homomorphism for *ring* scalars, since
//! `β_{ℓ,e}` is a ring element. `L` is, but the field-valued multilinear
//! evaluation is not: multiplying a witness by a ring challenge convolves within
//! each 64-coefficient ring element, so `MLE(β·W, ξ) ≠ β·MLE(W, ξ)`. This is the
//! same fact that makes the existing fold tree carry `y'` as a prover claim
//! rather than deriving it from `Σ γ_i y_i`.
//!
//! So `Ev` here is the *ring-linear* evaluation
//! `Ev(W,ξ) = Σ_j χ_hi(j)·W_j ∈ R`, where `W_j` is the `j`-th ring element and
//! `χ_hi` is the equality weight over the high `A − log d` variables. It is
//! ring-linear by construction. The link's field-valued terminal `a_e` is then a
//! public linear functional of it — `a_e = ⟨Ev(W,ξ), χ_lo(ξ)⟩` — so the two
//! stages are tied together by a check the verifier can run itself, and the
//! prover sends the ring element per commitment.
//!
//! That consistency check is the seam between the strong and weak reductions,
//! and it is the piece the write-up currently states as a single field-valued
//! `a_e`. Flagged rather than papered over: the composition proof needs to cover
//! it.

use almost_goldilocks_cuda::ajtai::{self, RingChallenge, RingCommitment, Seed, KAPPA, RING_DIM};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2 as Ext2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::transcript::Transcript;

/// Modulus, as `u128` for host-side ring arithmetic.
const Q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;

/// Repetitions. `τ = 2` per the parameter table.
pub const TAU: usize = 2;

/// Ring expansion factor `T_exp` for `{-1,0,1,2}^64` challenges.
pub const T_EXP: u64 = 128;

/// Parameters for one opening, derived from the group's layout.
#[derive(Clone, Copy, Debug)]
pub struct MaskParams {
    /// Number of commitments merged by this opening.
    pub n_commit: usize,
    /// Ambient dimension in coefficients.
    pub ambient: usize,
    /// `B` — the power of two above `B★ = |I|·T_exp·(B_x−1)+1`.
    pub b: u64,
    /// Gaussian width ratio `γ = σ / T_sh`. Higher means fewer rejections and a
    /// larger `B_Z`; the parameter chain has room for it when `β₂/q` is small.
    pub gamma: u64,
    /// `σ`, rounded to an integer.
    pub sigma: f64,
    /// Accepted-response bound.
    pub b_z: u128,
    /// Retry cap.
    pub r_max: usize,
}

impl MaskParams {
    /// Derive from the layout: `B★ = |I|·T_exp·(B_x−1)+1`, `σ = γ·B·√(τD)`, and
    /// `B_Z` from a union bound over the `τD` target coefficients.
    pub fn derive(n_commit: usize, ambient: usize, b_x: u64, gamma: u64) -> Self {
        let b_star = (n_commit as u64) * T_EXP * (b_x - 1) + 1;
        let b = (b_star + 1).next_power_of_two();
        let d = ambient as f64;
        let t_sh = (b as f64) * ((TAU as f64) * d).sqrt();
        let sigma = (gamma as f64) * t_sh;
        // tail budget 2^-150 over tau*D coefficients
        let tail = (2.0 * (2.0 * (TAU as f64) * d).ln() + 2.0 * 150.0 * 2f64.ln()).sqrt();
        let b_z = (sigma * tail).ceil() as u128;
        Self { n_commit, ambient, b, gamma, sigma, b_z, r_max: 512 }
    }

    /// `β_∞ = 8·T_exp·B_Z` — the Module-SIS coefficient bound relaxed binding
    /// needs. Reported so a run can state the security it actually achieved.
    pub fn beta_inf(&self) -> u128 {
        8 * (T_EXP as u128) * self.b_z
    }

    /// `β₂ = √D·β_∞`, which must stay below `q` or relaxed binding is vacuous.
    pub fn beta_2(&self) -> f64 {
        (self.ambient as f64).sqrt() * (self.beta_inf() as f64)
    }

    pub fn binding_is_meaningful(&self) -> bool {
        self.beta_2() < Q as f64
    }

    /// Rejection constant `M = exp(√(2η)/γ + 1/(2γ²))` at `η = 95`, and the
    /// acceptance probability it implies.
    pub fn rejection_constant(&self) -> (f64, f64) {
        let eta = 95.0f64;
        let g = self.gamma as f64;
        let m = ((2.0 * eta).sqrt() / g + 1.0 / (2.0 * g * g)).exp();
        (m, (1.0 - 2.0 * (-eta).exp()) / m)
    }
}

/// One repetition's published data.
#[derive(Clone, Debug)]
pub struct MaskResponse {
    /// `L(U_ℓ)`.
    pub d_commit: RingCommitment,
    /// `Ev(U_ℓ, ξ)` as a ring element.
    pub d_eval: Vec<u64>,
    /// The accepted `Z_ℓ`, as signed coefficients.
    pub z: Vec<i128>,
}

/// A complete masked-RLC opening.
#[derive(Clone, Debug)]
pub struct MrlcProof {
    pub responses: Vec<MaskResponse>,
    /// Ring-valued terminal evaluations, one per commitment. The link's
    /// field-valued `a_e` is recovered from these by a public functional.
    pub a_ring: Vec<Vec<u64>>,
    /// Index of the accepted attempt, published so the verifier can rebuild the
    /// challenge derivation. Rejected candidates are erased.
    pub retry_index: usize,
}

/// Centered representative of a canonical field element.
fn centered(v: u64) -> i128 {
    let x = (v as u128) % Q;
    if x > Q / 2 { x as i128 - Q as i128 } else { x as i128 }
}

fn canonical(v: i128) -> u64 {
    let m = Q as i128;
    let r = ((v % m) + m) % m;
    r as u64
}

/// Discrete Gaussian over the integers, by rejection against a uniform envelope.
///
/// Correctness first: this is a straightforward sampler adequate for testing the
/// protocol, not the constant-time GPU sampler the prover will use. `σ` here is
/// ~10^9, so the practical sampler must be a device kernel; the interface takes
/// the randomness source so that swap is local.
pub fn sample_gaussian<F: FnMut() -> u64>(sigma: f64, n: usize, mut rng: F) -> Vec<i128> {
    let bound = (sigma * 12.0) as i128;
    let span = (2 * bound + 1) as u128;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u = (rng() as u128) % span;
        let x = u as i128 - bound;
        // accept with probability exp(-x^2 / 2σ²)
        let p = (-((x as f64) * (x as f64)) / (2.0 * sigma * sigma)).exp();
        let t = (rng() >> 11) as f64 / (1u64 << 53) as f64;
        if t < p {
            out.push(x);
        }
    }
    out
}

/// Ring-linear evaluation `Ev(W,ξ) = Σ_j χ_hi(j)·W_j ∈ R`.
///
/// `chi_hi` weights ring elements, so multiplying the witness by a ring
/// challenge commutes with this map — which the field-valued multilinear
/// evaluation does not do.
pub fn ring_eval(coeffs: &[i128], chi_hi: &[u64]) -> Vec<u64> {
    let n_ring = coeffs.len() / RING_DIM;
    assert_eq!(chi_hi.len(), n_ring, "one weight per ring element");
    let mut acc = vec![0i128; RING_DIM];
    for j in 0..n_ring {
        let w = chi_hi[j] as i128;
        if w == 0 {
            continue;
        }
        for k in 0..RING_DIM {
            acc[k] = (acc[k] + coeffs[j * RING_DIM + k] * w) % (Q as i128);
        }
    }
    acc.iter().map(|v| canonical(*v)).collect()
}

/// Negacyclic multiply of a ring element by a small challenge.
pub fn ring_mul_challenge(v: &[u64], beta: &RingChallenge) -> Vec<u64> {
    let mut acc = vec![0i128; RING_DIM];
    for (i, vi) in v.iter().enumerate() {
        let x = centered(*vi);
        for (j, bj) in beta.coeffs.iter().enumerate() {
            let c = *bj as i128;
            if c == 0 {
                continue;
            }
            let k = i + j;
            let (slot, sign) = if k < RING_DIM { (k, 1i128) } else { (k - RING_DIM, -1i128) };
            acc[slot] = (acc[slot] + sign * x * c) % (Q as i128);
        }
    }
    acc.iter().map(|v| canonical(*v)).collect()
}

/// `β·W` over a full witness, ring element by ring element.
pub fn scale_witness(coeffs: &[i128], beta: &RingChallenge) -> Vec<i128> {
    let n_ring = coeffs.len() / RING_DIM;
    let mut out = vec![0i128; coeffs.len()];
    for j in 0..n_ring {
        let mut acc = vec![0i128; RING_DIM];
        for i in 0..RING_DIM {
            let x = coeffs[j * RING_DIM + i];
            if x == 0 {
                continue;
            }
            for (m, bm) in beta.coeffs.iter().enumerate() {
                let c = *bm as i128;
                if c == 0 {
                    continue;
                }
                let k = i + m;
                let (slot, sign) = if k < RING_DIM { (k, 1i128) } else { (k - RING_DIM, -1i128) };
                acc[slot] += sign * x * c;
            }
        }
        out[j * RING_DIM..(j + 1) * RING_DIM].copy_from_slice(&acc);
    }
    out
}

/// Commit a signed wide vector via the GPU wide-commit kernel.
fn commit_signed(seed: Seed, coeffs: &[i128], col_offset: u64) -> Option<RingCommitment> {
    let canon: Vec<u64> = coeffs.iter().map(|v| canonical(*v)).collect();
    ajtai::commit_wide(seed, &canon, col_offset, None).ok()
}

/// Sample ring challenges `β_{ℓ,e} ∈ {-1,0,1,2}^64` from the transcript.
fn sample_betas(transcript: &mut Transcript, n_commit: usize) -> Vec<Vec<RingChallenge>> {
    (0..TAU)
        .map(|_| {
            (0..n_commit)
                .map(|_| {
                    let mut coeffs = [0i8; RING_DIM];
                    let words = transcript.challenge_vector(b"mrlc-beta", RING_DIM / 8 + 1);
                    let mut bits = 0u64;
                    let mut have = 0usize;
                    let mut wi = 0usize;
                    for c in coeffs.iter_mut() {
                        if have < 2 {
                            bits = words[wi].0;
                            wi = (wi + 1) % words.len();
                            have = 32;
                        }
                        *c = (bits & 3) as i8 - 1; // {-1,0,1,2} shifted: 0..3 -> -1..2
                        bits >>= 2;
                        have -= 1;
                    }
                    RingChallenge::from_coeffs_unchecked(coeffs)
                })
                .collect()
        })
        .collect()
}

/// Joint rejection rule over the whole `τ`-tuple.
///
/// `acc(z) = min{1, p_σ(z) / (M·p_σ(z − S))}`, which in log form is
/// `-(2⟨z,S⟩ − ‖S‖²)/(2σ²) − ln M`. The shift is treated as one joint vector, so
/// the norm entering `γ` is its Euclidean norm across all repetitions.
fn accept(z: &[i128], s: &[i128], sigma: f64, m: f64, u01: f64) -> bool {
    let mut dot = 0f64;
    let mut s2 = 0f64;
    for (zi, si) in z.iter().zip(s.iter()) {
        dot += (*zi as f64) * (*si as f64);
        s2 += (*si as f64) * (*si as f64);
    }
    let ratio = (-(2.0 * dot - s2) / (2.0 * sigma * sigma)).exp() / m;
    u01 < ratio.min(1.0)
}

/// Run the masked RLC.
///
/// `witnesses` are the committed coefficient vectors (centered), `commitments`
/// their published commitments, `chi_hi` the ring-element equality weights at
/// `ξ`, and `a_field` the link's terminal values, which are checked against the
/// ring-valued evaluations rather than trusted.
pub fn prove_mrlc<F: FnMut() -> u64>(
    seed: Seed,
    witnesses: &[Vec<i128>],
    params: &MaskParams,
    chi_hi: &[u64],
    transcript: &mut Transcript,
    mut rng: F,
) -> Option<MrlcProof> {
    let n_commit = witnesses.len();
    let ambient = params.ambient;
    let (m, _) = params.rejection_constant();

    let a_ring: Vec<Vec<u64>> = witnesses.iter().map(|w| ring_eval(w, chi_hi)).collect();
    for a in &a_ring {
        for v in a {
            transcript.append_u64(b"mrlc-a-ring", *v);
        }
    }

    for retry in 0..params.r_max {
        // Fresh masks and a fresh nonce per attempt. Rejected candidates are
        // erased: publishing them would leak their correlation with the witness.
        let mut fork = transcript.clone();
        fork.append_u64(b"mrlc-nonce", retry as u64);

        let masks: Vec<Vec<i128>> = (0..TAU)
            .map(|_| sample_gaussian(params.sigma, ambient, &mut rng))
            .collect();
        let d: Vec<(RingCommitment, Vec<u64>)> = masks
            .iter()
            .map(|u| {
                let c = commit_signed(seed, u, 0)?;
                Some((c, ring_eval(u, chi_hi)))
            })
            .collect::<Option<Vec<_>>>()?;

        for (c, e) in &d {
            for row in c.rows.iter() {
                for v in row.iter() {
                    fork.append_u64(b"mrlc-D", *v);
                }
            }
            for v in e {
                fork.append_u64(b"mrlc-Deval", *v);
            }
        }

        let betas = sample_betas(&mut fork, n_commit);

        let mut shifts = Vec::with_capacity(TAU);
        let mut zs = Vec::with_capacity(TAU);
        for l in 0..TAU {
            let mut s = vec![0i128; ambient];
            for (e, w) in witnesses.iter().enumerate() {
                let scaled = scale_witness(w, &betas[l][e]);
                for (acc, v) in s.iter_mut().zip(scaled.iter()) {
                    *acc += v;
                }
            }
            let z: Vec<i128> = masks[l].iter().zip(s.iter()).map(|(u, si)| u + si).collect();
            shifts.push(s);
            zs.push(z);
        }

        // Joint rule over the concatenated tuple.
        let z_all: Vec<i128> = zs.iter().flatten().copied().collect();
        let s_all: Vec<i128> = shifts.iter().flatten().copied().collect();
        let u01 = (rng() >> 11) as f64 / (1u64 << 53) as f64;
        if !accept(&z_all, &s_all, params.sigma, m, u01) {
            continue;
        }
        if z_all.iter().any(|v| v.unsigned_abs() >= params.b_z) {
            continue;
        }

        *transcript = fork;
        return Some(MrlcProof {
            responses: (0..TAU)
                .map(|l| MaskResponse {
                    d_commit: d[l].0.clone(),
                    d_eval: d[l].1.clone(),
                    z: zs[l].clone(),
                })
                .collect(),
            a_ring,
            retry_index: retry,
        });
    }
    None
}

/// Verify a masked-RLC opening against the published commitments and the link's
/// terminal values.
pub fn verify_mrlc(
    seed: Seed,
    commitments: &[RingCommitment],
    proof: &MrlcProof,
    params: &MaskParams,
    chi_hi: &[u64],
    chi_lo: &[u64],
    a_field: &[Ext2],
    transcript: &mut Transcript,
) -> bool {
    let n_commit = commitments.len();
    if proof.a_ring.len() != n_commit || proof.responses.len() != TAU {
        return false;
    }

    for a in &proof.a_ring {
        for v in a {
            transcript.append_u64(b"mrlc-a-ring", *v);
        }
    }

    // The link's field-valued terminal is a public functional of the ring one.
    // Without this the two stages are not tied together and a prover could
    // answer the RLC about a different polynomial than the link certified.
    for (e, a) in proof.a_ring.iter().enumerate() {
        let mut acc = 0i128;
        for (k, v) in a.iter().enumerate() {
            acc = (acc + centered(*v) * (chi_lo[k] as i128)) % (Q as i128);
        }
        if canonical(acc) != a_field[e].c0.0 || a_field[e].c1.0 != 0 {
            return false;
        }
    }

    transcript.append_u64(b"mrlc-nonce", proof.retry_index as u64);
    for r in &proof.responses {
        for row in r.d_commit.rows.iter() {
            for v in row.iter() {
                transcript.append_u64(b"mrlc-D", *v);
            }
        }
        for v in &r.d_eval {
            transcript.append_u64(b"mrlc-Deval", *v);
        }
    }
    let betas = sample_betas(transcript, n_commit);

    for (l, r) in proof.responses.iter().enumerate() {
        // norm bound
        if r.z.iter().any(|v| v.unsigned_abs() >= params.b_z) {
            return false;
        }
        // commitment half: L(Z) == L(U) + Σ β_e C_e
        let lz = match commit_signed(seed, &r.z, 0) {
            Some(c) => c,
            None => return false,
        };
        let mut expect = r.d_commit.clone();
        for (e, c) in commitments.iter().enumerate() {
            let scaled = scale_commitment(c, &betas[l][e]);
            expect = add_commitments(&expect, &scaled);
        }
        if !commitments_eq(&lz, &expect) {
            return false;
        }
        // evaluation half, in the ring: Ev(Z) == Ev(U) + Σ β_e a_e
        let ez = ring_eval(&r.z, chi_hi);
        let mut expect_e = r.d_eval.clone();
        for (e, a) in proof.a_ring.iter().enumerate() {
            let scaled = ring_mul_challenge(a, &betas[l][e]);
            for (acc, v) in expect_e.iter_mut().zip(scaled.iter()) {
                *acc = canonical(centered(*acc) + centered(*v));
            }
        }
        if ez != expect_e {
            return false;
        }
    }
    true
}

fn scale_commitment(c: &RingCommitment, beta: &RingChallenge) -> RingCommitment {
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        let row: Vec<u64> = c.rows[i].to_vec();
        let scaled = ring_mul_challenge(&row, beta);
        out.rows[i].copy_from_slice(&scaled);
    }
    out
}

fn add_commitments(a: &RingCommitment, b: &RingCommitment) -> RingCommitment {
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            out.rows[i][r] = ((a.rows[i][r] as u128 + b.rows[i][r] as u128) % Q) as u64;
        }
    }
    out
}

fn commitments_eq(a: &RingCommitment, b: &RingCommitment) -> bool {
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            if (a.rows[i][r] as u128) % Q != (b.rows[i][r] as u128) % Q {
                return false;
            }
        }
    }
    true
}

/// Helper: the equality weights `ξ` splits into, high part over ring elements
/// and low part within a ring element.
pub fn split_chi(xi: &[Ext2]) -> (Vec<u64>, Vec<u64>) {
    let log_d = RING_DIM.trailing_zeros() as usize;
    let hi_vars = xi.len() - log_d;
    let base_eq = |pt: &[Ext2]| -> Vec<u64> {
        let mut t = vec![1u64];
        for &r in pt.iter().rev() {
            // Only the base component is used; ξ is drawn from Ext2 but the
            // ring-linear evaluation needs base-field weights.
            let rb = r.c0.0 % (Q as u64);
            let om = canonical(1 - centered(rb));
            let mut nt = vec![0u64; t.len() * 2];
            for (i, v) in t.iter().enumerate() {
                nt[i] = canonical(centered(*v) * centered(om) % (Q as i128));
                nt[t.len() + i] = canonical(centered(*v) * centered(rb) % (Q as i128));
            }
            t = nt;
        }
        t
    };
    (base_eq(&xi[..hi_vars]), base_eq(&xi[hi_vars..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn small_params(n_commit: usize, ambient: usize) -> MaskParams {
        // A small sigma keeps the test fast; the structure under test is the
        // protocol, not the concrete parameter set.
        let mut p = MaskParams::derive(n_commit, ambient, 2, 32);
        p.sigma = 4096.0;
        p.b_z = 1 << 40;
        p
    }

    #[test]
    fn parameters_track_the_documented_chain() {
        let p = MaskParams::derive(50, 1 << 23, 2, 8);
        assert_eq!(p.b, 8192, "B* = 50*128+1 = 6401 -> B = 8192");
        let (m, acc) = p.rejection_constant();
        assert!((m - 5.645).abs() < 0.01, "M = {}", m);
        assert!((acc - 0.1771).abs() < 0.001, "acceptance = {}", acc);
        assert!(p.binding_is_meaningful(), "beta_2 must stay below q");
    }

    #[test]
    fn raising_gamma_trades_acceptance_against_binding_margin() {
        let lo = MaskParams::derive(50, 1 << 23, 2, 8);
        let hi = MaskParams::derive(50, 1 << 23, 2, 32);
        let (_, acc_lo) = lo.rejection_constant();
        let (_, acc_hi) = hi.rejection_constant();
        assert!(acc_hi > acc_lo * 3.0, "gamma 8 -> 32 should raise acceptance sharply");
        assert!(hi.beta_2() > lo.beta_2(), "and cost binding margin");
    }

    #[test]
    fn ring_evaluation_is_linear_in_ring_challenges() {
        // The property the whole construction rests on, and the one the
        // field-valued multilinear evaluation does NOT have.
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        let n_ring = 4;
        let w: Vec<i128> = (0..n_ring * RING_DIM)
            .map(|_| (rng.next() % 7) as i128 - 3)
            .collect();
        let chi: Vec<u64> = (0..n_ring).map(|_| rng.next() % 1000).collect();

        let mut coeffs = [0i8; RING_DIM];
        for c in coeffs.iter_mut() {
            *c = (rng.next() % 4) as i8 - 1;
        }
        let beta = RingChallenge::from_coeffs_unchecked(coeffs);

        let lhs = ring_eval(&scale_witness(&w, &beta), &chi);
        let rhs = ring_mul_challenge(&ring_eval(&w, &chi), &beta);
        assert_eq!(lhs, rhs, "Ev(beta*W) must equal beta*Ev(W)");
    }

    #[test]
    fn honest_opening_verifies() {
        let mut rng = Rng(0xC0FF_EE00_1234_5678);
        let seed = Seed([3, 1, 4, 1, 5, 9, 2, 6]);
        let n_ring = 8usize;
        let ambient = n_ring * RING_DIM;
        let n_commit = 3usize;

        let witnesses: Vec<Vec<i128>> = (0..n_commit)
            .map(|_| (0..ambient).map(|_| (rng.next() % 3) as i128 - 1).collect())
            .collect();
        let commitments: Vec<RingCommitment> = witnesses
            .iter()
            .map(|w| commit_signed(seed, w, 0).expect("commit"))
            .collect();

        let chi_hi: Vec<u64> = (0..n_ring).map(|_| rng.next() % 997).collect();
        let chi_lo: Vec<u64> = (0..RING_DIM).map(|_| rng.next() % 997).collect();
        let params = small_params(n_commit, ambient);

        let a_ring: Vec<Vec<u64>> = witnesses.iter().map(|w| ring_eval(w, &chi_hi)).collect();
        let a_field: Vec<Ext2> = a_ring
            .iter()
            .map(|a| {
                let mut acc = 0i128;
                for (k, v) in a.iter().enumerate() {
                    acc = (acc + centered(*v) * (chi_lo[k] as i128)) % (Q as i128);
                }
                Ext2::new(AlmostGoldilocksField(canonical(acc)), AlmostGoldilocksField(0))
            })
            .collect();

        let mut tp = Transcript::new(b"mrlc");
        let proof = prove_mrlc(seed, &witnesses, &params, &chi_hi, &mut tp, || rng.next())
            .expect("prover should accept within the retry cap");

        let mut tv = Transcript::new(b"mrlc");
        assert!(
            verify_mrlc(seed, &commitments, &proof, &params, &chi_hi, &chi_lo, &a_field, &mut tv),
            "honest opening must verify"
        );
    }

    #[test]
    fn tampered_response_is_rejected() {
        let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);
        let seed = Seed([7, 7, 7, 7, 1, 2, 3, 4]);
        let n_ring = 4usize;
        let ambient = n_ring * RING_DIM;
        let n_commit = 2usize;

        let witnesses: Vec<Vec<i128>> = (0..n_commit)
            .map(|_| (0..ambient).map(|_| (rng.next() % 3) as i128 - 1).collect())
            .collect();
        let commitments: Vec<RingCommitment> = witnesses
            .iter()
            .map(|w| commit_signed(seed, w, 0).expect("commit"))
            .collect();
        let chi_hi: Vec<u64> = (0..n_ring).map(|_| rng.next() % 997).collect();
        let chi_lo: Vec<u64> = (0..RING_DIM).map(|_| rng.next() % 997).collect();
        let params = small_params(n_commit, ambient);

        let a_ring: Vec<Vec<u64>> = witnesses.iter().map(|w| ring_eval(w, &chi_hi)).collect();
        let a_field: Vec<Ext2> = a_ring
            .iter()
            .map(|a| {
                let mut acc = 0i128;
                for (k, v) in a.iter().enumerate() {
                    acc = (acc + centered(*v) * (chi_lo[k] as i128)) % (Q as i128);
                }
                Ext2::new(AlmostGoldilocksField(canonical(acc)), AlmostGoldilocksField(0))
            })
            .collect();

        let mut tp = Transcript::new(b"mrlc-bad");
        let mut proof =
            prove_mrlc(seed, &witnesses, &params, &chi_hi, &mut tp, || rng.next()).expect("prove");

        proof.responses[0].z[5] += 1;
        let mut tv = Transcript::new(b"mrlc-bad");
        assert!(
            !verify_mrlc(seed, &commitments, &proof, &params, &chi_hi, &chi_lo, &a_field, &mut tv),
            "a modified response must fail both linear checks"
        );
    }

    #[test]
    fn inconsistent_terminal_value_is_rejected() {
        // The seam between the link and the masked RLC: if the ring-valued
        // evaluation does not agree with the link's field-valued a_e, the two
        // stages are talking about different polynomials.
        let mut rng = Rng(0x0102_0304_0506_0708);
        let seed = Seed([9, 9, 9, 9, 9, 9, 9, 9]);
        let n_ring = 4usize;
        let ambient = n_ring * RING_DIM;
        let witnesses: Vec<Vec<i128>> =
            vec![(0..ambient).map(|_| (rng.next() % 3) as i128 - 1).collect()];
        let commitments: Vec<RingCommitment> =
            witnesses.iter().map(|w| commit_signed(seed, w, 0).unwrap()).collect();
        let chi_hi: Vec<u64> = (0..n_ring).map(|_| rng.next() % 997).collect();
        let chi_lo: Vec<u64> = (0..RING_DIM).map(|_| rng.next() % 997).collect();
        let params = small_params(1, ambient);

        let mut tp = Transcript::new(b"seam");
        let proof =
            prove_mrlc(seed, &witnesses, &params, &chi_hi, &mut tp, || rng.next()).expect("prove");

        // a_field that does not match the ring evaluation
        let bogus = vec![Ext2::new(AlmostGoldilocksField(12345), AlmostGoldilocksField(0))];
        let mut tv = Transcript::new(b"seam");
        assert!(!verify_mrlc(
            seed, &commitments, &proof, &params, &chi_hi, &chi_lo, &bogus, &mut tv
        ));
    }
}
