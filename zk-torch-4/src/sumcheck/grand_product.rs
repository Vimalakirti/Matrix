//! Grand-product argument for a β-linear multiset leaf, à la VerfCNN's
//! `prove_product_beta_linear_gkr` (IEEE S&P 2026). Proves
//!
//!   P = Π_{i=0}^{N-1} ( β · v[i] + idx[i] )
//!
//! where `v` is a committed value vector and `idx` is a PUBLIC index vector
//! (both length `N = 2^n`), `β` a Fiat-Shamir challenge. This is the
//! output-extraction primitive used by the grand-product conv-binding
//! baseline (behind `ZK4_CONV_GRANDPRODUCT`) so we can measure our bit-affine
//! masked-view binding against the VerfCNN-style partition inside the *same*
//! lattice PCS.
//!
//! Layered product tree (Thaler §4.6): `W_0[i] = β·v[i] + idx[i]`,
//! `W_ℓ[j] = W_{ℓ-1}[2j]·W_{ℓ-1}[2j+1]`, `W_n = P`. A claim on `W_ℓ(r)`
//! reduces to a claim on `W_{ℓ-1}` via the degree-3 sumcheck
//!   W̃_ℓ(r) = Σ_x eq(r,x) · W_{ℓ-1}(x,0) · W_{ℓ-1}(x,1)
//! (the pair bit is the LSB), whose two child evaluations are line-combined
//! into one claim at a fresh challenge `t`. After `n` layers the bottom claim
//! is `W_0(r_0) = β·v(r_0) + idx(r_0)`; the caller opens `v` at `r_0` and the
//! verifier reconstructs `idx(r_0)` from the public index MLE.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use serde::{Deserialize, Serialize};

use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_add, ext2_mul, ext2_sub, ext2_field_eq};

/// One layer's reduction: the degree-3 sumcheck plus the two child
/// evaluations the verifier needs to check the product and line-combine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrandProductLayer {
    pub sumcheck: SumcheckProof,
    pub child0: AlmostGoldilocksExt2, // W_{ℓ-1}(r', 0)
    pub child1: AlmostGoldilocksExt2, // W_{ℓ-1}(r', 1)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrandProductProof {
    pub product: AlmostGoldilocksExt2,
    /// Top (W_n → W_{n-1}) first, bottom (W_1 → W_0) last. Length `n`.
    pub layers: Vec<GrandProductLayer>,
}

impl GrandProductProof {
    /// Flatten into a `Vec<SumcheckProof>` so a grand-product proof can travel
    /// through the `BasicBlock::prove` channel (which only carries
    /// `SumcheckProof`s). Layout: `[product_carrier, layer0.sumcheck,
    /// layer0.child0_carrier, layer0.child1_carrier, layer1.sumcheck, …]`,
    /// where a *carrier* is a 0-round `SumcheckProof` whose `final_eval` holds
    /// a transported scalar. Consumes `1 + 3·n` slots for `n` layers.
    pub fn flatten(&self) -> Vec<SumcheckProof> {
        let carrier = |x: AlmostGoldilocksExt2| SumcheckProof { final_eval: x, round_messages: Vec::new() };
        let mut out = Vec::with_capacity(1 + 3 * self.layers.len());
        out.push(carrier(self.product));
        for l in &self.layers {
            out.push(l.sumcheck.clone());
            out.push(carrier(l.child0));
            out.push(carrier(l.child1));
        }
        out
    }

    /// Inverse of [`flatten`]. Reads `n` layers from `proofs[start..]` and
    /// returns the reconstructed proof plus the number of slots consumed.
    pub fn unflatten(proofs: &[SumcheckProof], start: usize, n: usize) -> (GrandProductProof, usize) {
        let product = proofs[start].final_eval;
        let mut layers = Vec::with_capacity(n);
        let mut k = start + 1;
        for _ in 0..n {
            let sumcheck = proofs[k].clone();
            let child0 = proofs[k + 1].final_eval;
            let child1 = proofs[k + 2].final_eval;
            k += 3;
            layers.push(GrandProductLayer { sumcheck, child0, child1 });
        }
        (GrandProductProof { product, layers }, 1 + 3 * n)
    }
}

/// Result of a grand-product proof: the value-vector claim `(point, eval)` the
/// caller must open against `v`'s commitment.
pub struct GrandProductClaim {
    pub point: Vec<AlmostGoldilocksExt2>,
    pub eval: AlmostGoldilocksExt2,
}

fn build_leaf(
    v: &[AlmostGoldilocksField],
    idx: &[AlmostGoldilocksField],
    beta: AlmostGoldilocksExt2,
) -> Vec<AlmostGoldilocksExt2> {
    assert_eq!(v.len(), idx.len());
    assert!(v.len().is_power_of_two(), "grand product needs power-of-two length");
    v.iter()
        .zip(idx.iter())
        .map(|(&vi, &ii)| {
            ext2_add(
                ext2_mul(beta, AlmostGoldilocksExt2::from_base(vi)),
                AlmostGoldilocksExt2::from_base(ii),
            )
        })
        .collect()
}

/// Prove `Π_i (β·v[i] + idx[i]) = P`. Returns the proof plus the value-vector
/// claim to open against `v`'s commitment.
pub fn prove_grand_product(
    v: &[AlmostGoldilocksField],
    idx: &[AlmostGoldilocksField],
    beta: AlmostGoldilocksExt2,
    transcript: &mut Transcript,
) -> (GrandProductProof, GrandProductClaim) {
    let leaf = build_leaf(v, idx, beta);
    let n = leaf.len().trailing_zeros() as usize;

    // Build the product-tree layers W_0..W_n. layer_vals[ℓ] has 2^{n-ℓ} elems.
    let mut layer_vals: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(n + 1);
    layer_vals.push(leaf);
    for l in 1..=n {
        let prev = &layer_vals[l - 1];
        let mut cur = Vec::with_capacity(prev.len() / 2);
        for j in 0..prev.len() / 2 {
            cur.push(ext2_mul(prev[2 * j], prev[2 * j + 1]));
        }
        layer_vals.push(cur);
    }
    let product = layer_vals[n][0];
    transcript.append_ext2(b"gp_product", &product);

    // Reduce top-down. Hold a claim (point r, eval c) on W_ℓ.
    let mut r: Vec<AlmostGoldilocksExt2> = Vec::new();
    let mut c = product;
    let mut layers = Vec::with_capacity(n);
    for l in (1..=n).rev() {
        let num_var = n - l; // W_ℓ has (n-ℓ) vars → sumcheck over that many
        let child = &layer_vals[l - 1];
        let half = child.len() / 2;
        // A[j] = W_{ℓ-1}(x=j, pair=0), B[j] = W_{ℓ-1}(x=j, pair=1).
        let a: Vec<AlmostGoldilocksExt2> = (0..half).map(|j| child[2 * j]).collect();
        let b: Vec<AlmostGoldilocksExt2> = (0..half).map(|j| child[2 * j + 1]).collect();
        let eq = evaluate_lagrange_basis_ext2(&r);
        debug_assert_eq!(eq.len(), 1 << num_var);

        let mut prover = CpuLinearSumcheckProverExt2::new(num_var, 3, transcript);
        let sumcheck = prover.prove(&mut [eq, a, b].as_mut_slice(), transcript);
        let r_prime = prover.challenges.clone();
        let child0 = prover.final_eval(1); // A(r') = W_{ℓ-1}(r', 0)
        let child1 = prover.final_eval(2); // B(r') = W_{ℓ-1}(r', 1)

        // Line-combine: new claim on W_{ℓ-1} at point (t, r') with the pair
        // bit as LSB. W̃_{ℓ-1}(t, r') = child0 + t·(child1 - child0).
        transcript.append_ext2(b"gp_child0", &child0);
        transcript.append_ext2(b"gp_child1", &child1);
        let t = transcript.challenge_ext2(b"gp_line");
        c = ext2_add(child0, ext2_mul(t, ext2_sub(child1, child0)));
        let mut next_point = Vec::with_capacity(num_var + 1);
        next_point.push(t);
        next_point.extend_from_slice(&r_prime);
        r = next_point;

        layers.push(GrandProductLayer { sumcheck, child0, child1 });
    }

    (
        GrandProductProof { product, layers },
        GrandProductClaim { point: r, eval: c },
    )
}

/// Verify a grand-product proof. Returns `Some((point, eval))` — the value-
/// vector claim on `v` the caller must check against `v`'s commitment — or
/// `None` on any inconsistency. `idx` is the public index vector; the verifier
/// reconstructs `idx(r_0)` itself.
pub fn verify_grand_product(
    proof: &GrandProductProof,
    idx: &[AlmostGoldilocksField],
    beta: AlmostGoldilocksExt2,
    expected_product: AlmostGoldilocksExt2,
    transcript: &mut Transcript,
) -> Option<GrandProductClaim> {
    if !ext2_field_eq(proof.product, expected_product) {
        return None;
    }
    let n = idx.len().trailing_zeros() as usize;
    if idx.len() != (1 << n) || proof.layers.len() != n {
        return None;
    }
    transcript.append_ext2(b"gp_product", &proof.product);

    let mut r: Vec<AlmostGoldilocksExt2> = Vec::new();
    let mut c = proof.product;
    for (li, l) in (1..=n).rev().enumerate() {
        let num_var = n - l;
        let layer = &proof.layers[li];
        let (ok, r_prime) =
            SumcheckVerifier::verify(&layer.sumcheck, c, num_var, 3, transcript);
        if !ok {
            return None;
        }
        // final_eval must equal eq(r, r') · child0 · child1.
        let eq_rr = eq_eval(&r, &r_prime);
        let expected = ext2_mul(ext2_mul(eq_rr, layer.child0), layer.child1);
        if !ext2_field_eq(expected, layer.sumcheck.final_eval) {
            return None;
        }
        transcript.append_ext2(b"gp_child0", &layer.child0);
        transcript.append_ext2(b"gp_child1", &layer.child1);
        let t = transcript.challenge_ext2(b"gp_line");
        c = ext2_add(layer.child0, ext2_mul(t, ext2_sub(layer.child1, layer.child0)));
        let mut next_point = Vec::with_capacity(num_var + 1);
        next_point.push(t);
        next_point.extend_from_slice(&r_prime);
        r = next_point;
    }

    // Bottom: c == β·v(r_0) + idx(r_0). Reconstruct idx(r_0); return the
    // v-claim the caller opens. eval = (c − idx(r_0)) / β would need β⁻¹; we
    // instead hand back the point and let the caller supply v(r_0), then the
    // caller checks c == β·v(r_0) + idx(r_0). Return the point + the residual
    // target so the caller's opened value can be checked without inverting β.
    let idx_ext: Vec<AlmostGoldilocksExt2> =
        idx.iter().map(|&i| AlmostGoldilocksExt2::from_base(i)).collect();
    let idx_at_r = mle_eval(&idx_ext, &r);
    // Caller checks: β·v(r) + idx_at_r == c. We stash `c` as the eval field and
    // the point; the conv verifier does the β·v + idx == c check after opening.
    let _ = idx_at_r;
    Some(GrandProductClaim { point: r, eval: c })
}

/// eq(a, b) = Π_i ((1−a_i)(1−b_i) + a_i b_i) for two equal-length points.
fn eq_eval(a: &[AlmostGoldilocksExt2], b: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    assert_eq!(a.len(), b.len());
    let one = AlmostGoldilocksExt2::one();
    let two = AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(2));
    let mut prod = one;
    for i in 0..a.len() {
        let term = ext2_add(
            ext2_sub(one, ext2_add(a[i], b[i])),
            ext2_mul(two, ext2_mul(a[i], b[i])),
        );
        prod = ext2_mul(prod, term);
    }
    prod
}

/// MLE of a value table evaluated at `r`: Σ_x eq(r,x)·table[x].
fn mle_eval(table: &[AlmostGoldilocksExt2], r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
    let eq = evaluate_lagrange_basis_ext2(r);
    assert_eq!(eq.len(), table.len());
    let mut acc = AlmostGoldilocksExt2::zero();
    for i in 0..table.len() {
        acc = ext2_add(acc, ext2_mul(eq[i], table[i]));
    }
    acc
}

/// Public helper: β·v(r) + idx(r), the value the bottom claim must equal.
/// Used by both the primitive's tests and the conv verifier after it opens v.
pub fn beta_linear_leaf_eval(
    beta: AlmostGoldilocksExt2,
    v_at_r: AlmostGoldilocksExt2,
    idx: &[AlmostGoldilocksField],
    r: &[AlmostGoldilocksExt2],
) -> AlmostGoldilocksExt2 {
    let idx_ext: Vec<AlmostGoldilocksExt2> =
        idx.iter().map(|&i| AlmostGoldilocksExt2::from_base(i)).collect();
    ext2_add(ext2_mul(beta, v_at_r), mle_eval(&idx_ext, r))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agl(x: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(x)
    }

    fn v_at(v: &[AlmostGoldilocksField], r: &[AlmostGoldilocksExt2]) -> AlmostGoldilocksExt2 {
        let ext: Vec<AlmostGoldilocksExt2> =
            v.iter().map(|&x| AlmostGoldilocksExt2::from_base(x)).collect();
        mle_eval(&ext, r)
    }

    fn naive_product(
        v: &[AlmostGoldilocksField],
        idx: &[AlmostGoldilocksField],
        beta: AlmostGoldilocksExt2,
    ) -> AlmostGoldilocksExt2 {
        build_leaf(v, idx, beta)
            .into_iter()
            .fold(AlmostGoldilocksExt2::one(), ext2_mul)
    }

    #[test]
    fn honest_grand_product_verifies() {
        let v: Vec<_> = [3u64, 7, 2, 9, 4, 1, 6, 5].iter().map(|&x| agl(x)).collect();
        let idx: Vec<_> = (0..8u64).map(agl).collect();
        let mut tp = Transcript::new(b"gp_test");
        let beta = tp.challenge_ext2(b"beta");

        let mut prove_t = Transcript::new(b"gp_run");
        let _ = prove_t.challenge_ext2(b"beta"); // keep transcripts aligned
        let (proof, claim) = prove_grand_product(&v, &idx, beta, &mut prove_t);

        // Product correctness.
        assert!(ext2_field_eq(proof.product, naive_product(&v, &idx, beta)));

        // Verify.
        let mut verify_t = Transcript::new(b"gp_run");
        let _ = verify_t.challenge_ext2(b"beta");
        let vclaim = verify_grand_product(&proof, &idx, beta, proof.product, &mut verify_t)
            .expect("honest proof must verify");

        // Bottom claim: c == β·v(r) + idx(r), and the point matches.
        assert_eq!(vclaim.point, claim.point);
        let expect = beta_linear_leaf_eval(beta, v_at(&v, &vclaim.point), &idx, &vclaim.point);
        assert!(ext2_field_eq(vclaim.eval, expect), "bottom β·v+idx check");
    }

    #[test]
    fn flatten_unflatten_roundtrip_verifies() {
        let v: Vec<_> = [3u64, 7, 2, 9, 4, 1, 6, 5].iter().map(|&x| agl(x)).collect();
        let idx: Vec<_> = (0..8u64).map(agl).collect();
        let mut tp = Transcript::new(b"gp_test");
        let beta = tp.challenge_ext2(b"beta");
        let mut prove_t = Transcript::new(b"gp_run");
        let _ = prove_t.challenge_ext2(b"beta");
        let (proof, _) = prove_grand_product(&v, &idx, beta, &mut prove_t);
        let n = proof.layers.len();

        // Flatten → unflatten must reproduce a proof that still verifies.
        let flat = proof.flatten();
        assert_eq!(flat.len(), 1 + 3 * n);
        let (rebuilt, consumed) = GrandProductProof::unflatten(&flat, 0, n);
        assert_eq!(consumed, flat.len());

        let mut verify_t = Transcript::new(b"gp_run");
        let _ = verify_t.challenge_ext2(b"beta");
        assert!(verify_grand_product(&rebuilt, &idx, beta, rebuilt.product, &mut verify_t).is_some());
    }

    #[test]
    fn wrong_product_rejected() {
        let v: Vec<_> = [3u64, 7, 2, 9].iter().map(|&x| agl(x)).collect();
        let idx: Vec<_> = (0..4u64).map(agl).collect();
        let mut tp = Transcript::new(b"gp_test");
        let beta = tp.challenge_ext2(b"beta");
        let mut prove_t = Transcript::new(b"gp_run");
        let _ = prove_t.challenge_ext2(b"beta");
        let (proof, _) = prove_grand_product(&v, &idx, beta, &mut prove_t);

        let mut verify_t = Transcript::new(b"gp_run");
        let _ = verify_t.challenge_ext2(b"beta");
        let bad = ext2_add(proof.product, AlmostGoldilocksExt2::one());
        assert!(verify_grand_product(&proof, &idx, beta, bad, &mut verify_t).is_none());
    }

    #[test]
    fn tampered_value_breaks_bottom_check() {
        // A prover who lies about v (but keeps a consistent proof for the
        // TRUE product) is caught at the bottom β·v(r)+idx(r) check once the
        // (honest) commitment opening supplies the real v(r).
        let v: Vec<_> = [3u64, 7, 2, 9, 4, 1, 6, 5].iter().map(|&x| agl(x)).collect();
        let v_tampered: Vec<_> = [3u64, 7, 2, 9, 4, 1, 6, 4].iter().map(|&x| agl(x)).collect();
        let idx: Vec<_> = (0..8u64).map(agl).collect();
        let mut tp = Transcript::new(b"gp_test");
        let beta = tp.challenge_ext2(b"beta");
        let mut prove_t = Transcript::new(b"gp_run");
        let _ = prove_t.challenge_ext2(b"beta");
        let (proof, _) = prove_grand_product(&v, &idx, beta, &mut prove_t);

        let mut verify_t = Transcript::new(b"gp_run");
        let _ = verify_t.challenge_ext2(b"beta");
        let vclaim = verify_grand_product(&proof, &idx, beta, proof.product, &mut verify_t)
            .expect("proof is internally valid");
        // The committed (tampered) value opens to a different v(r):
        let opened = beta_linear_leaf_eval(beta, v_at(&v_tampered, &vclaim.point), &idx, &vclaim.point);
        assert!(!ext2_field_eq(vclaim.eval, opened), "tampered v must fail bottom check");
    }
}
