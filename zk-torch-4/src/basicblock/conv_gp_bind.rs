//! Standalone grand-product conv output-binding, for a head-to-head comparison
//! against our production bit-affine masked-view binding (sumcheck "C" inside
//! [`conv2d_prove`]). VerfCNN (IEEE S&P 2026) binds the valid conv output to the
//! full 1D convolution advice with a multiset-partition grand product; we bind
//! it with a degree-3 masked-view sumcheck. This module reproduces the VerfCNN
//! binding on our exact Conv2D layout so both can be microbenchmarked inside the
//! same lattice PCS (see the `conv_binding_ablation_bench` test).
//!
//! NOTHING here touches the production `conv2d_prove`/`conv2d_verify` path, the
//! DAG builder, or `out_arity`. It is a `#[cfg(test)]` child module of `conv`,
//! so it can reach `Conv2D`'s private geometry helpers (`view_exponent`,
//! `s_full`, `view_point`, …) and the private `conv2d_*` functions the
//! benchmark calls, without exposing anything.
//!
//! ## What the grand-product binding proves (VerfCNN Part 2, our layout)
//! `Y_full[d,m]` (committed advice) is the full 1D conv, stored padded to
//! `[c_out_pad, s_full_pad]`. `Y[d,ho,wo]` (committed output) is the valid crop,
//! stored padded to `[c_out_pad, h_out_pad, w_out_pad]`. Over `Y_full`'s linear
//! domain `D = c_out_pad·s_full_pad` we prove the multiset partition
//!
//!   Π_{Y}(β·Y[i]+idxY[i]) · Π_{K}(β·K[i]+idxK[i]) = Π_{Y_full}(β·Y_full[i]+idxAll[i])
//!
//! at a shared Fiat–Shamir β, then check `P_Y·P_K == P_Yfull` (Schwartz–Zippel
//! over β ⇒ the partition holds). The `K` ("leftover") leg is exactly the real
//! `Y_full` coefficients no valid output lands on. Every padded slot uses the
//! `value=0 ⇒ idx=1` no-op so `β·0+1 = 1` and drops out of the product.

use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;

use crate::basicblock::conv::Conv2D;
use crate::dag::Witness;
use crate::poly::dense::DenseMLPoly;
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::grand_product::{
    beta_linear_leaf_eval, prove_grand_product, verify_grand_product, GrandProductProof,
};
use crate::sumcheck::{CpuLinearSumcheckProverExt2, SumcheckProof, SumcheckVerifier};
use crate::transcript::Transcript;
use crate::util::arith::{ext2_field_eq, ext2_mul, log2_ceil};

type AGField = AlmostGoldilocksField;
type Ext2 = AlmostGoldilocksExt2;

// ============================================================================
// Grand-product binding (VerfCNN-style multiset partition)
// ============================================================================

/// The three (value, idx) vectors of the multiset partition. `idx` vectors are
/// PUBLIC (pure conv geometry); `val` vectors are the committed witnesses.
struct BindVectors {
    y_val: Vec<AGField>,
    y_idx: Vec<AGField>,
    k_val: Vec<AGField>,
    k_idx: Vec<AGField>,
    yfull_val: Vec<AGField>,
    yfull_idx: Vec<AGField>,
}

/// Build the Y / K / Y_full (value, idx) vectors. Deterministic from the conv
/// geometry plus the two committed witnesses; both prover and verifier call it
/// (the verifier only consumes the PUBLIC idx vectors + the K values, which in
/// a real system would be a committed "leftover" advice edge).
fn build_bind_vectors(conv: &Conv2D, y: &Witness, yfull: &Witness) -> BindVectors {
    let c_out_pad = conv.c_out.next_power_of_two();
    let s_full = conv.s_full();
    let s_full_pad = s_full.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let w_out_pad = conv.w_out.next_power_of_two();
    let d_domain = c_out_pad * s_full_pad; // |Y_full| linear domain

    let yfull_ev = yfull.data.as_ref().unwrap().evaluations();
    let y_ev = y.data.as_ref().unwrap().evaluations();
    debug_assert_eq!(yfull_ev.len(), d_domain);
    debug_assert_eq!(y_ev.len(), c_out_pad * h_out_pad * w_out_pad);

    // --- Y_full leg: value = committed Y_full, idx = linear position (real) ---
    let mut yfull_idx = vec![AlmostGoldilocksField(1); d_domain];
    for d in 0..conv.c_out {
        for m in 0..s_full {
            let i = d * s_full_pad + m;
            yfull_idx[i] = AlmostGoldilocksField(i as u64);
        }
    }
    let yfull_val = yfull_ev.clone();

    // --- Y leg: value = committed Y, idx = exponent position in Y_full ---
    let y_len = c_out_pad * h_out_pad * w_out_pad;
    let mut y_idx = vec![AlmostGoldilocksField(1); y_len];
    for d in 0..conv.c_out {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let lin = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                let e = conv.view_exponent(ho, wo);
                y_idx[lin] = AlmostGoldilocksField((d * s_full_pad + e) as u64);
            }
        }
    }
    let y_val = y_ev.clone();

    // --- K leg: the real Y_full coefficients NO valid output lands on ---
    let mut used = vec![false; d_domain];
    for d in 0..conv.c_out {
        for ho in 0..conv.h_out {
            for wo in 0..conv.w_out {
                let e = conv.view_exponent(ho, wo);
                used[d * s_full_pad + e] = true;
            }
        }
    }
    let mut k_val = Vec::new();
    let mut k_idx = Vec::new();
    for d in 0..conv.c_out {
        for m in 0..s_full {
            let i = d * s_full_pad + m;
            if !used[i] {
                k_val.push(yfull_ev[i]);
                k_idx.push(AlmostGoldilocksField(i as u64));
            }
        }
    }
    // Partition sanity: #valid + #leftover == #real Y_full coefficients.
    debug_assert_eq!(
        conv.c_out * conv.h_out * conv.w_out + k_idx.len(),
        conv.c_out * s_full,
        "grand-product bind: valid+leftover must tile the real Y_full domain"
    );
    let k_pad = k_idx.len().max(1).next_power_of_two();
    k_val.resize(k_pad, AlmostGoldilocksField(0));
    k_idx.resize(k_pad, AlmostGoldilocksField(1));

    BindVectors { y_val, y_idx, k_val, k_idx, yfull_val, yfull_idx }
}

/// A full grand-product binding proof: the three products + the three
/// grand-product proofs. `k_idx` is the PUBLIC leftover-index vector (a real
/// verifier reconstructs it from geometry; carried here so the size accounting
/// and verify path are explicit).
pub(crate) struct GpBindProof {
    /// [P_Y, P_K, P_Yfull].
    pub products: [Ext2; 3],
    pub y: GrandProductProof,
    pub k: GrandProductProof,
    pub yfull: GrandProductProof,
    pub k_idx: Vec<AGField>,
}

/// Prove the VerfCNN multiset partition binding `Y`/`K` against `Y_full`.
/// Draws a shared β from `transcript`, runs three grand products (Y, K, Y_full),
/// and returns the proof. The value-claim points each grand product exits with
/// are re-derived (and "opened") by the verifier.
pub(crate) fn grand_product_bind_prove(
    conv: &Conv2D,
    y: &Witness,
    yfull: &Witness,
    transcript: &mut Transcript,
) -> GpBindProof {
    let bv = build_bind_vectors(conv, y, yfull);
    let beta = transcript.challenge_ext2(b"gp_bind_beta");

    let (py, _) = prove_grand_product(&bv.y_val, &bv.y_idx, beta, transcript);
    let (pk, _) = prove_grand_product(&bv.k_val, &bv.k_idx, beta, transcript);
    let (pyf, _) = prove_grand_product(&bv.yfull_val, &bv.yfull_idx, beta, transcript);

    debug_assert!(
        ext2_field_eq(ext2_mul(py.product, pk.product), pyf.product),
        "grand-product bind: P_Y·P_K must equal P_Yfull on honest data"
    );

    GpBindProof {
        products: [py.product, pk.product, pyf.product],
        y: py,
        k: pk,
        yfull: pyf,
        k_idx: bv.k_idx,
    }
}

/// Verify a grand-product binding: mirror β, verify the three grand products,
/// check each bottom claim `c == β·v(point) + idx(point)` where `v(point)` is
/// obtained by evaluating the corresponding witness MLE at the claim point
/// (this stands in for the PCS opening — the masked-view side is measured the
/// same way, so the comparison is symmetric), and check `P_Y·P_K == P_Yfull`.
pub(crate) fn grand_product_bind_verify(
    conv: &Conv2D,
    y: &Witness,
    yfull: &Witness,
    proof: &GpBindProof,
    transcript: &mut Transcript,
) -> bool {
    let bv = build_bind_vectors(conv, y, yfull);
    let beta = transcript.challenge_ext2(b"gp_bind_beta");

    // Y leg — open Y's committed MLE at the bottom point.
    let cy = match verify_grand_product(&proof.y, &bv.y_idx, beta, proof.products[0], transcript) {
        Some(c) => c,
        None => return false,
    };
    let vy = y.data.as_ref().unwrap().evaluate_at_point_ext2(&cy.point);
    if !ext2_field_eq(cy.eval, beta_linear_leaf_eval(beta, vy, &bv.y_idx, &cy.point)) {
        return false;
    }

    // K leg — the leftover advice; open its MLE table directly.
    let ck = match verify_grand_product(&proof.k, &proof.k_idx, beta, proof.products[1], transcript) {
        Some(c) => c,
        None => return false,
    };
    let k_poly = DenseMLPoly::new(log2_ceil(bv.k_val.len()), bv.k_val.clone());
    let vk = k_poly.evaluate_at_point_ext2(&ck.point);
    if !ext2_field_eq(ck.eval, beta_linear_leaf_eval(beta, vk, &proof.k_idx, &ck.point)) {
        return false;
    }

    // Y_full leg — open Y_full's committed MLE at the bottom point.
    let cyf =
        match verify_grand_product(&proof.yfull, &bv.yfull_idx, beta, proof.products[2], transcript) {
            Some(c) => c,
            None => return false,
        };
    let vyf = yfull.data.as_ref().unwrap().evaluate_at_point_ext2(&cyf.point);
    if !ext2_field_eq(cyf.eval, beta_linear_leaf_eval(beta, vyf, &bv.yfull_idx, &cyf.point)) {
        return false;
    }

    // Multiset partition: P_Y · P_K == P_Yfull.
    ext2_field_eq(ext2_mul(proof.products[0], proof.products[1]), proof.products[2])
}

// ============================================================================
// Masked-view binding (our production sumcheck "C"), isolated for comparison
// ============================================================================

/// A binding-only reproduction of the production sumcheck C: one degree-3
/// sumcheck plus the Y_full opening value it exits with. Reproduced here (not
/// carved out of `conv2d_prove`) so we can time it in isolation without
/// touching production; it is byte-for-byte the same construction as the C
/// block inside `conv2d_prove`.
pub(crate) struct MaskedViewProof {
    pub proof_c: SumcheckProof,
    /// Y_full(σ̃(r'_spatial), r'_d) — the view opening the verifier checks.
    pub yfull_eval: Ext2,
}

/// Prove `Y(r_star) = Σ_x eq(r_star,x)·mask(x)·view(Y_full)(x)` at the (given,
/// already-random) self point `r_star = (r_spatial | r_d)`.
pub(crate) fn masked_view_bind_prove(
    conv: &Conv2D,
    yfull: &Witness,
    r_star: &[Ext2],
    transcript: &mut Transcript,
) -> MaskedViewProof {
    let l_spatial_out = conv.l_spatial_out();
    let l_d = conv.l_d();
    let l_out_n = l_spatial_out + l_d;
    assert_eq!(r_star.len(), l_out_n);

    let c_out_pad = conv.c_out.next_power_of_two();
    let h_out_pad = conv.h_out.next_power_of_two();
    let w_out_pad = conv.w_out.next_power_of_two();
    let s_full_pad = conv.s_full().next_power_of_two();
    let yfull_slice = yfull.data.as_ref().unwrap().evaluations_ref();

    let eq_star = evaluate_lagrange_basis_ext2(r_star);
    let mut mask_tab = vec![Ext2::zero(); 1 << l_out_n];
    let mut view_tab = vec![Ext2::zero(); 1 << l_out_n];
    for d in 0..c_out_pad {
        for ho in 0..h_out_pad {
            for wo in 0..w_out_pad {
                let x_lin = wo + ho * w_out_pad + d * w_out_pad * h_out_pad;
                let e = conv.view_exponent(ho, wo);
                view_tab[x_lin] = Ext2::from_base(yfull_slice[d * s_full_pad + e]);
                if d < conv.c_out && ho < conv.h_out && wo < conv.w_out {
                    mask_tab[x_lin] = Ext2::one();
                }
            }
        }
    }

    let mut prover_c = CpuLinearSumcheckProverExt2::new(l_out_n, 3, transcript);
    let proof_c = prover_c.prove(&mut [eq_star, mask_tab, view_tab].as_mut_slice(), transcript);
    MaskedViewProof { proof_c, yfull_eval: prover_c.final_eval(2) }
}

/// Verify the masked-view binding against the incoming self-claim value
/// `y_self_eval`, then "open" Y_full at the bit-affine view point (the same PCS
/// stand-in the grand-product side uses).
pub(crate) fn masked_view_bind_verify(
    conv: &Conv2D,
    yfull: &Witness,
    r_star: &[Ext2],
    y_self_eval: Ext2,
    proof: &MaskedViewProof,
    transcript: &mut Transcript,
) -> bool {
    let l_spatial_out = conv.l_spatial_out();
    let l_d = conv.l_d();
    let l_out_n = l_spatial_out + l_d;

    let (ok, challenges_c) =
        SumcheckVerifier::verify(&proof.proof_c, y_self_eval, l_out_n, 3, transcript);
    if !ok {
        return false;
    }
    // final_eval = eq(r_star, r') · mask(r') · Y_full(σ̃(r'_spatial), r'_d).
    let eq_final = super::eq_points_ext2(r_star, &challenges_c);
    let mask_final = super::conv2d_mask_mle_eval(conv, &challenges_c);
    let expected = ext2_mul(ext2_mul(eq_final, mask_final), proof.yfull_eval);
    if !ext2_field_eq(expected, proof.proof_c.final_eval) {
        return false;
    }
    // "Open" Y_full at the reconstructed bit-affine view point.
    let mut view_pt = conv.view_point(&challenges_c[..l_spatial_out]);
    view_pt.extend_from_slice(&challenges_c[l_spatial_out..]);
    let opened = yfull.data.as_ref().unwrap().evaluate_at_point_ext2(&view_pt);
    ext2_field_eq(opened, proof.yfull_eval)
}

// ============================================================================
// Size accounting (Ext2 element counts → bytes; Ext2 = 2×u64 = 16 bytes)
// ============================================================================

const EXT2_BYTES: usize = 16;

/// Ext2 elements in one sumcheck proof: all round messages + the final eval.
fn sumcheck_ext2_count(p: &SumcheckProof) -> usize {
    p.round_messages.iter().map(|r| r.len()).sum::<usize>() + 1
}

/// Ext2 elements in one grand-product proof: per layer, its sumcheck + the two
/// child evaluations. (The single product scalar is counted once by the caller.)
fn gp_ext2_count(p: &GrandProductProof) -> usize {
    p.layers
        .iter()
        .map(|l| sumcheck_ext2_count(&l.sumcheck) + 2)
        .sum::<usize>()
}

pub(crate) fn masked_view_proof_bytes(p: &MaskedViewProof) -> usize {
    sumcheck_ext2_count(&p.proof_c) * EXT2_BYTES
}

pub(crate) fn gp_bind_proof_bytes(p: &GpBindProof) -> usize {
    let ext2 = gp_ext2_count(&p.y) + gp_ext2_count(&p.k) + gp_ext2_count(&p.yfull) + 3; // +3 products
    ext2 * EXT2_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basicblock::conv::FlattenKernel;
    use crate::basicblock::BasicBlock;
    use crate::dag::{DataType, Role};
    use std::time::Instant;

    fn f(x: u64) -> AGField {
        AlmostGoldilocksField(x)
    }

    /// Deterministic small (%2) input/weight witnesses + the conv outputs.
    /// Returns (Y, Y_full). Uses FlattenKernel so the flat-with-dilation W
    /// layout is exactly what `Conv2D::run` expects.
    fn make_conv_io(
        conv: &Conv2D,
        kernel_h: usize,
        kernel_w: usize,
    ) -> (Witness, Witness) {
        let c_in = conv.c_in;
        let c_out = conv.c_out;
        let h = conv.input_h;
        let w = conv.input_w;
        let w_pad = w.next_power_of_two();

        // X[c_in, h, w] little-endian (w | h | c). %2 magnitudes.
        let mut x_data = Vec::with_capacity(c_in * h * w);
        for c in 0..c_in {
            for ih in 0..h {
                for iw in 0..w {
                    x_data.push((c + ih + iw) as u64 % 2);
                }
            }
        }
        let x = Witness::new(vec![c_in, h, w], x_data.into_iter().map(f).collect(), DataType::Uint, 0, Role::Input);

        // raw kernel W[c_out, c_in, kh, kw] little-endian (kw | kh | c | d).
        let mut w_data = Vec::with_capacity(c_out * c_in * kernel_h * kernel_w);
        for d in 0..c_out {
            for c in 0..c_in {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        w_data.push((d + c + kh + kw) as u64 % 2);
                    }
                }
            }
        }
        let w_raw = Witness::new(
            vec![c_out, c_in, kernel_h, kernel_w],
            w_data.into_iter().map(f).collect(),
            DataType::Uint,
            0,
            Role::Input,
        );
        let fk = FlattenKernel {
            s_w: w_pad,
            kh: kernel_h,
            kw: kernel_w,
            c_out,
            c_in,
            dilation_h: 1,
            dilation_w: 1,
        };
        let wf = fk.run(&[&w_raw]).remove(0);

        let mut out = conv.run(&[&x, &wf]);
        let yfull = out.remove(1);
        let y = out.remove(0);
        (y, yfull)
    }

    fn median(mut xs: Vec<u128>) -> u128 {
        xs.sort_unstable();
        xs[xs.len() / 2]
    }

    // ---- Correctness: honest grand-product binding verifies ----

    fn honest_gp_bind_case(conv: &Conv2D, kh: usize, kw: usize) {
        let (y, yfull) = make_conv_io(conv, kh, kw);

        let mut tp = Transcript::new(b"gp_bind_case");
        let proof = grand_product_bind_prove(conv, &y, &yfull, &mut tp);

        let mut tv = Transcript::new(b"gp_bind_case");
        assert!(
            grand_product_bind_verify(conv, &y, &yfull, &proof, &mut tv),
            "honest grand-product bind must verify"
        );
    }

    #[test]
    fn gp_bind_honest_3x3_stride1() {
        // Non-pow2 5×5 input, c_in=2, c_out=3, 3×3 stride-1.
        let conv = Conv2D::new(2, 3, 3, 3, 5, 5);
        honest_gp_bind_case(&conv, 3, 3);
    }

    #[test]
    fn gp_bind_honest_strided() {
        // 3×3 stride-2 on a 7×7 input, c_in=2, c_out=3.
        let conv = Conv2D::new_strided(2, 3, 3, 3, 7, 7, 2, 2);
        honest_gp_bind_case(&conv, 3, 3);
    }

    #[test]
    fn gp_bind_tampered_y_rejected() {
        let conv = Conv2D::new(2, 3, 3, 3, 5, 5);
        let (y, yfull) = make_conv_io(&conv, 3, 3);

        let mut tp = Transcript::new(b"gp_bind_tamper");
        let proof = grand_product_bind_prove(&conv, &y, &yfull, &mut tp);

        // Flip one real output coefficient in Y: its committed MLE now opens to
        // a different value, so the Y-leg bottom check β·v(r)+idx(r)==c fails.
        let w_out_pad = conv.w_out.next_power_of_two();
        let h_out_pad = conv.h_out.next_power_of_two();
        let mut ev = y.data.as_ref().unwrap().evaluations();
        let flip = 0usize; // (d=0, ho=0, wo=0) — a real output
        let _ = (w_out_pad, h_out_pad);
        ev[flip] = AlmostGoldilocksField(ev[flip].0 + 1);
        let y_tampered = Witness::new(
            y.shape.clone(),
            ev,
            DataType::Uint,
            0,
            Role::Output,
        );

        let mut tv = Transcript::new(b"gp_bind_tamper");
        assert!(
            !grand_product_bind_verify(&conv, &y_tampered, &yfull, &proof, &mut tv),
            "tampered Y must fail the grand-product bind"
        );
    }

    // ---- Correctness: honest masked-view binding verifies ----

    #[test]
    fn masked_view_honest_3x3_stride1() {
        let conv = Conv2D::new(2, 3, 3, 3, 5, 5);
        let (y, yfull) = make_conv_io(&conv, 3, 3);
        let l_out_n = conv.l_spatial_out() + conv.l_d();

        let mut tp = Transcript::new(b"mv_case");
        let r_star: Vec<Ext2> = (0..l_out_n).map(|_| tp.challenge_ext2(b"r")).collect();
        let y_self_eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&r_star);
        let proof = masked_view_bind_prove(&conv, &yfull, &r_star, &mut tp);

        let mut tv = Transcript::new(b"mv_case");
        let r_star_v: Vec<Ext2> = (0..l_out_n).map(|_| tv.challenge_ext2(b"r")).collect();
        assert_eq!(r_star, r_star_v);
        assert!(
            masked_view_bind_verify(&conv, &yfull, &r_star_v, y_self_eval, &proof, &mut tv),
            "honest masked-view bind must verify"
        );
    }

    // ---- Microbenchmark ----

    struct Shape {
        name: &'static str,
        conv: Conv2D,
        kh: usize,
        kw: usize,
    }

    #[test]
    fn conv_binding_ablation_bench() {
        const REPS: usize = 5;
        // Representative ResNet-ish shapes (all pow2 conv strides).
        let shapes = vec![
            Shape {
                name: "(a) stage1 3x3 s1  c=64->64  8x8",
                conv: Conv2D::new(64, 64, 3, 3, 8, 8),
                kh: 3,
                kw: 3,
            },
            Shape {
                // Stem: worst case for Y_full size. c_out reduced 64->16 to keep
                // the 2^19-ish grand product within the test time budget.
                name: "(b) stem 7x7 s2    c=3->16   38x38",
                conv: Conv2D::new_strided(3, 16, 7, 7, 38, 38, 2, 2),
                kh: 7,
                kw: 7,
            },
            Shape {
                name: "(c) stage2 3x3 s1  c=128->128 8x8",
                conv: Conv2D::new(128, 128, 3, 3, 8, 8),
                kh: 3,
                kw: 3,
            },
        ];

        println!();
        println!("=== Conv output-binding ablation: masked-view (ours) vs grand-product (VerfCNN) ===");
        println!("Isolation: BINDING-ONLY for both. The shared sumchecks (1,B,2,3,4) that");
        println!("reduce Y/Y_full to the X,W claims are IDENTICAL for both bindings and are");
        println!("excluded. Masked-view = production sumcheck C (1 degree-3 sumcheck + 1 Y_full");
        println!("open). Grand-product = 3 layered product trees (Y, K, Y_full) + 3 witness opens.");
        println!("Prover/verifier times are medians of {REPS}. 'Open' = MLE eval standing in for");
        println!("the PCS opening, counted in verifier time on BOTH sides. Proof bytes = Ext2");
        println!("elements x 16 (round messages + final evals; grand-product adds 2 child evals");
        println!("per layer + 3 product scalars).");
        println!();
        println!(
            "{:<34} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
            "shape", "MV prv us", "MV vrf us", "MV bytes", "GP prv us", "GP vrf us", "GP bytes"
        );
        println!("{}", "-".repeat(34 + 3 + 33 + 3 + 33));

        for sh in &shapes {
            let conv = &sh.conv;
            let (y, yfull) = make_conv_io(conv, sh.kh, sh.kw);
            let l_out_n = conv.l_spatial_out() + conv.l_d();

            // ---- masked-view (ours) ----
            let r_star: Vec<Ext2> = {
                let mut t = Transcript::new(b"mv_r");
                (0..l_out_n).map(|_| t.challenge_ext2(b"r")).collect()
            };
            let y_self_eval = y.data.as_ref().unwrap().evaluate_at_point_ext2(&r_star);

            let mut mv_prv = Vec::with_capacity(REPS);
            let mut mv_vrf = Vec::with_capacity(REPS);
            let mut mv_bytes = 0usize;
            for _ in 0..REPS {
                let mut tp = Transcript::new(b"mv_bench");
                let t0 = Instant::now();
                let proof = masked_view_bind_prove(conv, &yfull, &r_star, &mut tp);
                mv_prv.push(t0.elapsed().as_micros());

                let mut tv = Transcript::new(b"mv_bench");
                let t1 = Instant::now();
                let ok = masked_view_bind_verify(conv, &yfull, &r_star, y_self_eval, &proof, &mut tv);
                mv_vrf.push(t1.elapsed().as_micros());
                assert!(ok, "masked-view bench must verify [{}]", sh.name);
                mv_bytes = masked_view_proof_bytes(&proof);
            }

            // ---- grand-product (VerfCNN baseline) ----
            let mut gp_prv = Vec::with_capacity(REPS);
            let mut gp_vrf = Vec::with_capacity(REPS);
            let mut gp_bytes = 0usize;
            for _ in 0..REPS {
                let mut tp = Transcript::new(b"gp_bench");
                let t0 = Instant::now();
                let proof = grand_product_bind_prove(conv, &y, &yfull, &mut tp);
                gp_prv.push(t0.elapsed().as_micros());

                let mut tv = Transcript::new(b"gp_bench");
                let t1 = Instant::now();
                let ok = grand_product_bind_verify(conv, &y, &yfull, &proof, &mut tv);
                gp_vrf.push(t1.elapsed().as_micros());
                assert!(ok, "grand-product bench must verify [{}]", sh.name);
                gp_bytes = gp_bind_proof_bytes(&proof);
            }

            println!(
                "{:<34} | {:>10} {:>10} {:>10} | {:>10} {:>10} {:>10}",
                sh.name,
                median(mv_prv),
                median(mv_vrf),
                mv_bytes,
                median(gp_prv),
                median(gp_vrf),
                gp_bytes
            );
        }
        println!();
    }
}
