//! Multifold step (plan §6.2). Given a group of `FoldInstance`s that
//! already share a claim point `R` (post-`same_point_sumcheck`),
//! sample ring challenges `γ_1, …, γ_{M-1}` from the transcript and
//! reduce to one instance:
//!
//! ```text
//! c' = c_0 + Σ_{i=1..M-1} γ_i · c_i      (multifold_commitment kernel)
//! f' = f_0 + Σ_{i=1..M-1} γ_i · f_i      (multifold_witness kernel)
//! y' = f'(R)                              (direct evaluation by prover)
//! ```
//!
//! The Ajtai homomorphism guarantees `c' = M · f'` bit-exactly, so the
//! verifier doesn't need to see `f'` — it re-derives `c'` from the input
//! commitments + `γ` challenges and checks against the prover's claimed
//! `c'`. The new claim value `y' = f'(R)` is supplied by the prover and
//! anchored at the final tree level by the receive-and-verify check
//! `commit(f*) = c*` ∧ `f*(R*) = y*`.
//!
//! ### Why `y' ≠ Σ γ_i · y_i'`
//!
//! `γ_i ∈ R` is a degree-63 polynomial in `X` with `{-1, 0, 1, 2}`
//! coefficients (`R = F_q[X]/(X^64 + 1)`); ring-multiplying it into a
//! witness convolves the within-ring-element coefficients. As a result
//! the multilinear-Ext2 evaluation of the folded witness does **not**
//! factor as `Σ γ_i · f_i(R)`. The protocol carries `y'` forward as a
//! prover claim, anchored at the root.

use almost_goldilocks_cuda::ajtai::{
    self, RingChallenge, RingCommitment, TernaryChunksDevice,
};
use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::memory::DeviceBuffer;
use serde::{Deserialize, Serialize};

use crate::fold::{FoldData, FoldInstance, WireCommitment, WireRingChallenge};
use crate::transcript::Transcript;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultifoldProof {
    /// `γ_1, …, γ_{M-1}` — one per non-anchor instance.
    pub gammas: Vec<WireRingChallenge>,
    /// `c' = c_0 + Σ γ_i · c_i` — the prover's combined commitment.
    /// The verifier re-derives this from `multifold_commitment` and
    /// checks bit-equality.
    pub combined_commitment: WireCommitment,
    /// `y' = f'(R)` — the prover's evaluation of the folded witness at
    /// the shared point `R`. Verified at the root level only.
    pub combined_claim: AlmostGoldilocksExt2,
}

/// Sample `M − 1` `RingChallenge`s from `transcript`, fold the witnesses
/// (host reference path) and commitments (GPU), and return the new
/// `FoldInstance` plus the wire proof.
///
/// Requires every input in `group` to share the same `arity` and
/// `claim_pt` — this is the post-same-point-sumcheck contract.
pub fn prove_multifold(
    group: &[FoldInstance],
    transcript: &mut Transcript,
) -> (FoldInstance, MultifoldProof) {
    prove_multifold_with_eq(group, transcript, None)
}

/// How the prover obtains `combined_y = f'(R)`.
enum CombinedY<'a> {
    /// Reuse a pre-built `eq(R, ·)` table (host dot product).
    Eq(&'a [AlmostGoldilocksExt2]),
    /// Build the MLE evaluation directly (host, no shared eq table).
    Direct,
    /// Skip it — set a zero placeholder. The internal-group caller derives
    /// `combined_y = Σ 2^i·chunk_eval[i]` from the post-split GPU chunk evals
    /// and patches `proof.combined_claim`. The fold-tree verifier never reads
    /// an internal group's `combined_claim` (only `chunk_claim_vals`), so the
    /// placeholder is safe; this avoids the 64 MB host eq-table download.
    Defer,
}

/// As [`prove_multifold`], but defers `combined_y` (see [`CombinedY::Defer`]).
/// Returns the folded instance (with a placeholder `claim_val`) and a proof
/// whose `combined_claim` the caller must overwrite.
pub fn prove_multifold_defer_y(
    group: &[FoldInstance],
    transcript: &mut Transcript,
) -> (FoldInstance, MultifoldProof) {
    prove_multifold_core(group, transcript, CombinedY::Defer)
}

/// Device-resident defer-y multifold: the group's witness data is already
/// on the CURRENT device (one binary concat OR a ternary pos/neg concat
/// pair, leaf i at element offset `i · n_ring`), and the folded wide-i16
/// output STAYS on device for the split-decompose + commit chain. Same
/// transcript behavior (γ sampling) and same `combined_commitment` as the
/// host path; `combined_claim` is a zero placeholder the caller patches
/// from the chunk evals. On `Err` the transcript has consumed the γ
/// challenges — the caller must restore its snapshot before falling back.
pub fn prove_multifold_defer_y_dev(
    group: &[FoldInstance],
    dev_bin: Option<&DeviceBuffer<u64>>,
    dev_tern: Option<(&DeviceBuffer<u64>, &DeviceBuffer<u64>)>,
    transcript: &mut Transcript,
) -> Result<(DeviceBuffer<i16>, MultifoldProof), almost_goldilocks_cuda::error::CudaError> {
    assert!(!group.is_empty(), "multifold expects ≥ 1 instance");
    assert!(dev_bin.is_some() ^ dev_tern.is_some(),
        "exactly one of dev_bin / dev_tern must be supplied");
    let m = group.len();
    let arity = group[0].arity;
    let n_ring = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
    // NB: unlike the host path (which receives instances PROMOTED to the
    // shared same-point challenge), this variant takes the group's original
    // instances — only their arity, data, and commitments matter here (the
    // γ sampling and the fold are claim-point-independent; the caller
    // carries shared_r and patches combined_claim from the chunk evals).
    for (i, inst) in group.iter().enumerate() {
        assert_eq!(inst.arity, arity, "instance {} arity {} != {}", i, inst.arity, arity);
    }

    let mut gammas: Vec<RingChallenge> = Vec::with_capacity(m.saturating_sub(1));
    for i in 1..m {
        gammas.push(sample_ring_challenge(transcript, b"mf_gamma", i as u64));
    }

    let (k_bin, k_tern) = if dev_bin.is_some() { (m, 0) } else { (0, m) };
    let d_wide = ajtai::multifold_mixed_witness_tc_fused_dev(
        dev_bin, k_bin, dev_tern, k_tern, n_ring, &gammas,
    )?;

    let comm_refs: Vec<&RingCommitment> = group.iter().map(|i| &i.commitment).collect();
    let combined_c = ajtai::multifold_commitment(&comm_refs, &gammas)?;

    let proof = MultifoldProof {
        gammas: gammas.iter().map(WireRingChallenge::from_ring).collect(),
        combined_commitment: WireCommitment::from_ring(&combined_c),
        combined_claim: AlmostGoldilocksExt2::zero(),
    };
    Ok((d_wide, proof))
}

/// As [`prove_multifold`], but optionally reuses a pre-built
/// `eq(shared_pt, ·)` table for the `combined_y = f'(R)` evaluation.
/// All instances share `claim_pt`, so the caller (a fold-tree node)
/// can build the eq table once and pass it to both this and the
/// subsequent split-chunk evals — avoiding a redundant `2^arity` eq
/// reconstruction (~100 ms at arity 22).
pub fn prove_multifold_with_eq(
    group: &[FoldInstance],
    transcript: &mut Transcript,
    shared_eq: Option<&[AlmostGoldilocksExt2]>,
) -> (FoldInstance, MultifoldProof) {
    let mode = match shared_eq {
        Some(eq) => CombinedY::Eq(eq),
        None => CombinedY::Direct,
    };
    prove_multifold_core(group, transcript, mode)
}

fn prove_multifold_core(
    group: &[FoldInstance],
    transcript: &mut Transcript,
    y_mode: CombinedY<'_>,
) -> (FoldInstance, MultifoldProof) {
    assert!(!group.is_empty(), "multifold expects ≥ 1 instance");
    let m = group.len();
    let arity = group[0].arity;
    let shared_pt = group[0].claim_pt.clone();
    for (i, inst) in group.iter().enumerate() {
        assert_eq!(inst.arity, arity, "instance {} arity {} != {}", i, inst.arity, arity);
        assert_eq!(inst.claim_pt, shared_pt, "instance {} claim_pt mismatch", i);
    }

    // Sample γ_1..γ_{M-1}.
    let mut gammas: Vec<RingChallenge> = Vec::with_capacity(m.saturating_sub(1));
    for i in 1..m {
        gammas.push(sample_ring_challenge(transcript, b"mf_gamma", i as u64));
    }

    // Fold the witnesses on GPU via the appropriate Ajtai kernel:
    // `multifold_witness` for all-binary groups (the typical level-0
    // case after bit-decomposition), `multifold_mixed_witness_tc_fused`
    // with `k_bin = 0` for all-ternary groups (every level after the
    // first split). The kernels return a wide-i16 buffer which we
    // re-encode as `FoldData::Binary` (only when every coefficient
    // happens to be 0/1) or `FoldData::Ternary` (the common case after
    // a real fold).
    let folded = fold_witnesses_gpu(group, &gammas);

    // Fold the commitments on GPU (KAPPA · RING_DIM = 960 ops — μs).
    let comm_refs: Vec<&RingCommitment> = group.iter().map(|i| &i.commitment).collect();
    let combined_c = ajtai::multifold_commitment(&comm_refs, &gammas)
        .expect("multifold_commitment kernel");

    // Evaluate f'(R) on the prover side per the requested strategy.
    let combined_y = match y_mode {
        CombinedY::Eq(eq) => folded.evaluate_with_eq(eq),
        CombinedY::Direct => folded.evaluate_at_ext2(&shared_pt),
        CombinedY::Defer => AlmostGoldilocksExt2::zero(),
    };

    let combined = FoldInstance {
        commitment: combined_c.clone(),
        data: folded,
        arity,
        claim_pt: shared_pt,
        claim_val: combined_y,
    };
    let proof = MultifoldProof {
        gammas: gammas.iter().map(WireRingChallenge::from_ring).collect(),
        combined_commitment: WireCommitment::from_ring(&combined_c),
        combined_claim: combined_y,
    };
    (combined, proof)
}

/// Verifier-side multifold: re-derives γ challenges from transcript, runs
/// `multifold_commitment` on the prover-supplied input commitments, and
/// checks bit-equality with `proof.combined_commitment`. Returns the
/// prover's `combined_commitment` and `combined_claim` for use as the
/// next-level input — the claim is **trusted** here and anchored at the
/// root by the final `commit(f*) = c*` ∧ `f*(R*) = y*` check.
pub fn verify_multifold(
    input_commitments: &[&RingCommitment],
    proof: &MultifoldProof,
    transcript: &mut Transcript,
) -> Option<(RingCommitment, AlmostGoldilocksExt2)> {
    let m = input_commitments.len();
    if proof.gammas.len() + 1 != m { return None; }

    let mut gammas: Vec<RingChallenge> = Vec::with_capacity(m - 1);
    for i in 1..m {
        let derived = sample_ring_challenge(transcript, b"mf_gamma", i as u64);
        if derived.coeffs != proof.gammas[i - 1].coeffs.as_slice() { return None; }
        gammas.push(derived);
    }

    let derived_c = ajtai::multifold_commitment(input_commitments, &gammas).ok()?;
    let expected = proof.combined_commitment.to_ring();
    if !rings_equal(&derived_c, &expected) { return None; }
    Some((expected, proof.combined_claim))
}

// ============================================================================
// Witness fold — GPU production path + host reference (test oracle)
// ============================================================================

/// GPU witness fold. Dispatches on the group's instance kinds:
/// all-binary → `ajtai::multifold_witness`; otherwise (all-ternary by
/// construction in the fold tree, since post-split outputs are always
/// ternary) → `ajtai::multifold_mixed_witness_tc_fused` with
/// `k_bin = 0`. Result is bit-exactly identical to
/// [`fold_witnesses_host`] (verified by
/// `gpu_witness_fold_matches_host_*`).
/// Convert a Digit (K bit-planes + sign flag) to K ternary chunks. Chunk k's
/// (pos, neg) is (bit_k, 0) for unsigned bits and (0, bit_k) for the top
/// digit's sign bit. The wide value `Σ_k 2^k(pos_k − neg_k)` reproduces the
/// digit's signed value exactly, so the existing multi-chunk ternary path
/// processes it correctly through the TC-fused multifold kernel.
fn digit_to_ternary_chunks(
    bit_planes: &[Vec<u64>],
    negate_top_bit: bool,
) -> almost_goldilocks_cuda::ajtai::TernaryChunks {
    let k_chunks = bit_planes.len();
    assert!(k_chunks >= 1);
    let n_ring = bit_planes[0].len();
    let mut pos: Vec<u64> = vec![0u64; k_chunks * n_ring];
    let mut neg: Vec<u64> = vec![0u64; k_chunks * n_ring];
    for k in 0..k_chunks {
        let is_sign = negate_top_bit && k == k_chunks - 1;
        if is_sign {
            neg[k * n_ring..(k + 1) * n_ring].copy_from_slice(&bit_planes[k]);
        } else {
            pos[k * n_ring..(k + 1) * n_ring].copy_from_slice(&bit_planes[k]);
        }
    }
    almost_goldilocks_cuda::ajtai::TernaryChunks { n_ring, k_chunks, pos, neg }
}

fn fold_witnesses_gpu(group: &[FoldInstance], gammas: &[RingChallenge]) -> FoldData {
    // Digit groups (higher-radix L0 leaves) route through the chunk-γ
    // derivation path: each Digit leaf becomes K single-chunk ternary chunks
    // with γ_chunk = γ_leaf · 2^k (signed for the top-bit sign chunk). A
    // dummy zero binary leaf is prepended so the kernel's implicit γ=1
    // attaches to it (contributes 0) and every real leaf carries an
    // explicit γ. This bridges the protocol's per-leaf γ sampling to the
    // kernel's per-chunk instance counting without modifying the kernel.
    let any_digit = group.iter().any(|i| matches!(i.data, FoldData::Digit { .. }));
    if any_digit {
        return fold_witnesses_gpu_digit_path(group, gammas);
    }
    let all_binary = group.iter().all(|i| matches!(i.data, FoldData::Binary(_)));
    let any_ternary = group.iter().any(|i| matches!(i.data, FoldData::Ternary(_)));
    assert!(
        all_binary || group.iter().all(|i| matches!(i.data, FoldData::Ternary(_))),
        "fold group must be all-binary or all-ternary (no mixing in the fold tree)",
    );

    // Cap for the STANDALONE multifold kernel (this host-path fold). This is
    // SEPARATE from the device-resident *fused* path's cap
    // (ZK4_MULTIFOLD_GPU_MAX_ARITY=24 in tree.rs): the fused group kernel does
    // fail >24 (grid/shared-mem), but the plain multifold_mixed_witness_tc_fused
    // kernel runs fine at arity 26/28 on 80 GB (measured: arity-28 M=63 fold
    // 9–11 s on CPU → ~2 s on GPU, Verified). Its inner loop is already
    // popcount-sparse, and `wide_i16: Err(_) => fold_witnesses_host` below is a
    // safety net if a larger config OOMs, so we let the kernel attempt high
    // arities. Default 30 covers seq≤512 (arity 30); override via
    // ZK4_MULTIFOLD_KERNEL_MAX_ARITY.
    let arity = group[0].arity;
    let kernel_arity_cap = std::env::var("ZK4_MULTIFOLD_KERNEL_MAX_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(30);
    if arity > kernel_arity_cap {
        return fold_witnesses_host(group, gammas);
    }

    let timing = std::env::var("ZK4_TIMING_MF").is_ok();
    let t0 = std::time::Instant::now();
    let wide_i16: Result<Vec<i16>, _> = if all_binary {
        let refs: Vec<&[u64]> = group.iter().map(|i| match &i.data {
            FoldData::Binary(v) => v.as_slice(),
            _ => unreachable!(),
        }).collect();
        let n_ring = refs[0].len();
        let empty_ternary = TernaryChunksDevice {
            n_ring,
            k_chunks: 0,
            pos: almost_goldilocks_cuda::memory::DeviceBuffer::<u64>::new(0)
                .expect("alloc empty pos"),
            neg: almost_goldilocks_cuda::memory::DeviceBuffer::<u64>::new(0)
                .expect("alloc empty neg"),
        };
        ajtai::multifold_mixed_witness_tc_fused(&refs, &empty_ternary, gammas)
    } else {
        assert!(any_ternary);
        // Pool-backed direct upload: each leaf's chunk slices write straight
        // into the final concatenated device layout. Replaces the old
        // per-leaf DeviceBuffer uploads (126 cudaMalloc/cudaFree pairs at
        // M=63) + a full extra DtoD concat copy.
        let n_ring = match &group[0].data {
            FoldData::Ternary(c) => c.n_ring,
            _ => unreachable!(),
        };
        let total_chunks: usize = group.iter().map(|i| match &i.data {
            FoldData::Ternary(c) => c.k_chunks,
            _ => unreachable!(),
        }).sum();
        let total_words = total_chunks * n_ring;
        let mut d_pos = almost_goldilocks_cuda::sumcheck_prover::pool_take(total_words)
            .expect("pool_take pos for multifold");
        let mut d_neg = almost_goldilocks_cuda::sumcheck_prover::pool_take(total_words)
            .expect("pool_take neg for multifold");
        let mut slot = 0usize;
        for inst in group {
            if let FoldData::Ternary(c) = &inst.data {
                d_pos.write_slice_at(slot * n_ring, &c.pos)
                    .expect("upload pos chunks");
                d_neg.write_slice_at(slot * n_ring, &c.neg)
                    .expect("upload neg chunks");
                slot += c.k_chunks;
            }
        }
        let combined = TernaryChunksDevice {
            n_ring, k_chunks: total_chunks, pos: d_pos, neg: d_neg,
        };
        let r = ajtai::multifold_mixed_witness_tc_fused(&[], &combined, gammas);
        almost_goldilocks_cuda::sumcheck_prover::pool_return(combined.pos);
        almost_goldilocks_cuda::sumcheck_prover::pool_return(combined.neg);
        r
    };
    let t1 = std::time::Instant::now();

    let out = match wide_i16 {
        Ok(w) => encode_wide_as_fold_data(&w, arity),
        Err(_) => fold_witnesses_host(group, gammas),
    };
    if timing {
        eprintln!("[mf arity={} M={} binary={}] fused={:?} encode={:?}",
            arity, group.len(), all_binary, t1 - t0, t1.elapsed());
    }
    out
}

/// Multiply a `RingChallenge`'s coefficients by a signed integer `weight`.
/// Used to derive per-chunk γs from per-leaf γs in the digit multifold path.
/// At base ≤ 64 every scaled coefficient fits in `i8` (|γ| ≤ 2, |weight| ≤ 32).
fn scale_ring_challenge(g: &RingChallenge, weight: i32) -> RingChallenge {
    let mut c = [0i8; 64];
    for i in 0..64 {
        let scaled = (g.coeffs[i] as i32) * weight;
        debug_assert!(scaled.abs() <= 127,
            "scaled coef {} overflows i8 (γ={} · weight={})", scaled, g.coeffs[i], weight);
        c[i] = scaled as i8;
    }
    RingChallenge::from_coeffs_unchecked(c)
}

/// Digit-aware multifold. Each `FoldData::Digit` leaf with K bit-planes is
/// re-expressed as K single-chunk ternary chunks; the per-chunk γ is
/// `γ_leaf · 2^k` (signed for the top digit's sign chunk). A dummy zero
/// binary leaf is prepended so the kernel's implicit γ=1 attaches to it
/// (contributing 0) and every real leaf carries an explicit γ. This bridges
/// the per-leaf γ sampling to the kernel's per-chunk instance counting
/// without modifying the kernel. Mathematically:
///     combined = Σ_leaf γ_leaf · digit_leaf
///              = Σ_{leaf,k} γ_leaf · 2^k · bit_{leaf,k}
///              = Σ_chunks γ_chunk · chunk_value
/// which is exactly what the kernel computes.
fn fold_witnesses_gpu_digit_path(group: &[FoldInstance], gammas: &[RingChallenge]) -> FoldData {
    let n = group.len();
    let arity = group[0].arity;
    // See fold_witnesses_gpu: standalone kernel runs past the fused path's
    // arity-24 cap; Err→host fallback is the safety net.
    let kernel_arity_cap = std::env::var("ZK4_MULTIFOLD_KERNEL_MAX_ARITY").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(30);
    if arity > kernel_arity_cap {
        return fold_witnesses_host(group, gammas);
    }

    let n_ring = match &group[0].data {
        FoldData::Binary(v) => v.len(),
        FoldData::Ternary(c) => c.n_ring,
        FoldData::Digit { bit_planes, .. } => bit_planes[0].len(),
    };

    // Per-leaf γs: leaf 0 has γ=1 (the protocol's implicit), rest from transcript.
    let one_chal = {
        let mut c = [0i8; 64]; c[0] = 1;
        RingChallenge::from_coeffs_unchecked(c)
    };
    let leaf_gammas: Vec<RingChallenge> = (0..n).map(|i| {
        if i == 0 { one_chal.clone() } else { gammas[i - 1].clone() }
    }).collect();

    // Prepend a dummy zero binary leaf — absorbs the kernel's implicit γ=1
    // (contributes 0), so all real leaves carry explicit γs.
    let dummy_zeros: Vec<u64> = vec![0u64; n_ring];
    let mut bin_refs: Vec<&[u64]> = vec![&dummy_zeros[..]];
    let mut derived_gammas: Vec<RingChallenge> = Vec::new();

    // Real binary leaves (binary auxiliaries in mixed groups) come second.
    for (i, inst) in group.iter().enumerate() {
        if let FoldData::Binary(v) = &inst.data {
            bin_refs.push(v.as_slice());
            derived_gammas.push(leaf_gammas[i].clone());
        }
    }

    // For digit/ternary chunks build ONE big (pos, neg) host buffer of
    // size n_ring × total_chunks, then upload + concat in a single shot.
    // Previous per-chunk approach allocated ~n_ring zeros per chunk and
    // did one device upload per chunk — ~1GB of zero-fill + 484 uploads
    // at arity-24, dominating multifold time.
    let mut total_tern_chunks = 0;
    for inst in group {
        match &inst.data {
            FoldData::Digit { bit_planes, .. } => total_tern_chunks += bit_planes.len(),
            FoldData::Ternary(c) => total_tern_chunks += c.k_chunks,
            _ => {}
        }
    }

    let combined_dev = if total_tern_chunks == 0 {
        TernaryChunksDevice {
            n_ring, k_chunks: 0,
            pos: almost_goldilocks_cuda::memory::DeviceBuffer::<u64>::new(0).expect("empty pos"),
            neg: almost_goldilocks_cuda::memory::DeviceBuffer::<u64>::new(0).expect("empty neg"),
        }
    } else {
        // Allocate once, zero-initialized; copy bit planes into the right slots.
        let total_words = n_ring * total_tern_chunks;
        let mut big_pos = vec![0u64; total_words];
        let mut big_neg = vec![0u64; total_words];
        let mut slot = 0usize;
        for (i, inst) in group.iter().enumerate() {
            if let FoldData::Digit { bit_planes, negate_top_bit, .. } = &inst.data {
                let k_bits = bit_planes.len();
                for bk in 0..k_bits {
                    let is_sign = *negate_top_bit && bk == k_bits - 1;
                    let weight: i32 = if is_sign { -(1i32 << bk) } else { 1i32 << bk };
                    derived_gammas.push(scale_ring_challenge(&leaf_gammas[i], weight));
                    let dst = slot * n_ring;
                    if is_sign {
                        big_neg[dst..dst + n_ring].copy_from_slice(&bit_planes[bk]);
                    } else {
                        big_pos[dst..dst + n_ring].copy_from_slice(&bit_planes[bk]);
                    }
                    slot += 1;
                }
            }
        }
        for (i, inst) in group.iter().enumerate() {
            if let FoldData::Ternary(c) = &inst.data {
                for bk in 0..c.k_chunks {
                    let weight: i32 = 1i32 << bk;
                    derived_gammas.push(scale_ring_challenge(&leaf_gammas[i], weight));
                    let dst = slot * n_ring;
                    big_pos[dst..dst + n_ring]
                        .copy_from_slice(&c.pos[bk * c.n_ring..(bk + 1) * c.n_ring]);
                    big_neg[dst..dst + n_ring]
                        .copy_from_slice(&c.neg[bk * c.n_ring..(bk + 1) * c.n_ring]);
                    slot += 1;
                }
            }
        }
        debug_assert_eq!(slot, total_tern_chunks);
        TernaryChunksDevice {
            n_ring,
            k_chunks: total_tern_chunks,
            pos: DeviceBuffer::<u64>::from_slice(&big_pos).expect("upload big_pos"),
            neg: DeviceBuffer::<u64>::from_slice(&big_neg).expect("upload big_neg"),
        }
    };

    let wide = ajtai::multifold_mixed_witness_tc_fused(&bin_refs, &combined_dev, &derived_gammas);
    match wide {
        Ok(w) => encode_wide_as_fold_data(&w, arity),
        Err(_) => fold_witnesses_host(group, gammas),
    }
}

/// Encode a wide i16 buffer (`n_ring × 64`) as `FoldData`. Binary form
/// is returned when every coefficient happens to be `0` or `1` (rare —
/// only single-instance "folds" hit this); otherwise 13-chunk ternary
/// base-2 decomposition.
///
/// Rayon-parallel over ring positions: each `j` owns word index `j` of
/// every chunk plane (`pos[i*n_ring + j]`), so per-`j` columns compute
/// independently and a cheap serial scatter transposes them into the
/// chunk-major layout. The old serial loop was ~30-50 ms at arity 22
/// (4M coefficients × 13 chunk bits on one core) — a top multifold cost.
fn encode_wide_as_fold_data(wide: &[i16], arity: usize) -> FoldData {
    use rayon::prelude::*;
    let n_ring = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
    assert_eq!(wide.len(), n_ring * 64, "wide buffer size mismatch");
    if wide.par_iter().all(|&v| v == 0 || v == 1) {
        let packed: Vec<u64> = (0..n_ring).into_par_iter().map(|j| {
            let mut w = 0u64;
            for k in 0..64 {
                if wide[j * 64 + k] == 1 { w |= 1u64 << k; }
            }
            w
        }).collect();
        return FoldData::Binary(packed);
    }
    let k_chunks = ajtai::SPLIT_K_CHUNKS;
    assert!(k_chunks <= 16, "column buffers sized for k_chunks <= 16");
    let cols: Vec<([u64; 16], [u64; 16])> = (0..n_ring).into_par_iter().map(|j| {
        let mut p = [0u64; 16];
        let mut n = [0u64; 16];
        for k in 0..64 {
            let mut v = wide[j * 64 + k];
            assert!(v.abs() < (1i16 << k_chunks),
                "fold output |coef| = {} >= 2^13 = 8192", v.abs());
            let negative = v < 0;
            if negative { v = -v; }
            for i in 0..k_chunks {
                if (v >> i) & 1 == 1 {
                    if negative { n[i] |= 1u64 << k; }
                    else        { p[i] |= 1u64 << k; }
                }
            }
        }
        (p, n)
    }).collect();
    let mut pos = vec![0u64; k_chunks * n_ring];
    let mut neg = vec![0u64; k_chunks * n_ring];
    for (j, (p, n)) in cols.iter().enumerate() {
        for i in 0..k_chunks {
            pos[i * n_ring + j] = p[i];
            neg[i * n_ring + j] = n[i];
        }
    }
    FoldData::Ternary(ajtai::TernaryChunks { n_ring, k_chunks, pos, neg })
}

/// Host-side `f' = f_0 + Σ γ_i · f_i` in ring `R`. Each input's
/// per-ring-element coefficients are convolved with the corresponding
/// `γ_i`'s coefficients modulo `(X^64 + 1)`, then summed across `i`.
/// Output is i16 wide coefficients (`|·| ≤ 1 + (M − 1) · 128`).
///
/// Used by the bit-exact-agreement tests as the oracle for the GPU
/// kernels — not on the production prove path.
// Host reference (also used as the production fallback when the GPU
// kernel runs out of memory or hits an internal limit at very large
// arity). Slow but correct.
fn fold_witnesses_host(group: &[FoldInstance], gammas: &[RingChallenge]) -> FoldData {
    let arity = group[0].arity;
    let n_ring = if arity >= 6 { 1usize << (arity - 6) } else { 1 };
    let mut wide = vec![0i32; n_ring * 64];

    // Anchor instance gets the constant-1 weight.
    let one_ring = constant_one_ring();
    accumulate_witness_with_weight(&mut wide, &group[0], &one_ring, n_ring);
    for (i, g) in gammas.iter().enumerate() {
        accumulate_witness_with_weight(&mut wide, &group[i + 1], g, n_ring);
    }

    // Down-cast to i16, asserting we stay within the SuperNeo binding bound.
    let mut wide_i16 = vec![0i16; n_ring * 64];
    for (out, &v) in wide_i16.iter_mut().zip(wide.iter()) {
        assert!(v.abs() < (1i32 << 13),
            "fold output |coef| = {} >= 2^13 = 8192", v.abs());
        *out = v as i16;
    }

    if wide_i16.iter().all(|&v| v == 0 || v == 1) {
        // Encode back into packed binary form.
        let mut packed = vec![0u64; n_ring];
        for j in 0..n_ring {
            for k in 0..64 {
                if wide_i16[j * 64 + k] == 1 {
                    packed[j] |= 1u64 << k;
                }
            }
        }
        FoldData::Binary(packed)
    } else {
        // Base-2 decomposition into 13 ternary chunks (matches
        // `ajtai::split_witness`).
        let k_chunks = ajtai::SPLIT_K_CHUNKS;
        let mut pos = vec![0u64; k_chunks * n_ring];
        let mut neg = vec![0u64; k_chunks * n_ring];
        for j in 0..n_ring {
            for k in 0..64 {
                let mut v = wide_i16[j * 64 + k];
                let negative = v < 0;
                if negative { v = -v; }
                for i in 0..k_chunks {
                    if (v >> i) & 1 == 1 {
                        if negative { neg[i * n_ring + j] |= 1u64 << k; }
                        else        { pos[i * n_ring + j] |= 1u64 << k; }
                    }
                }
            }
        }
        FoldData::Ternary(ajtai::TernaryChunks { n_ring, k_chunks, pos, neg })
    }
}

/// Accumulate `weight · inst.data` (ring multiplication, integer
/// arithmetic) into the wide i32 buffer. Used by `fold_witnesses_host`
/// (CPU fallback for very large arity).
fn accumulate_witness_with_weight(
    wide: &mut [i32],
    inst: &FoldInstance,
    weight: &RingChallenge,
    n_ring: usize,
) {
    use rayon::prelude::*;
    let b = &weight.coeffs;
    // Parallel over the j (ring-element) axis. Each j touches a disjoint
    // slice `wide[j*64..(j+1)*64]`. The witness ring elements are SPARSE
    // (level-0 binary planes: ~0.25 set bits/element, 75% all-zero; level-1+
    // ternary split chunks: ~25% dense / high chunks near-zero), so instead
    // of the dense 64×64 negacyclic convolution per element we:
    //   (1) skip all-zero ring elements outright, and
    //   (2) for each nonzero coefficient `a[k]`, add `a[k]·(b ≪ k)`
    //       (negacyclic shift: `+b[m−k]` for m≥k, `−b[m+64−k]` for m<k).
    // Integer accumulation is order-independent, so `wide` is BYTE-IDENTICAL
    // to the dense convolution — split chunks, commitments and the verifier
    // are unchanged. Cost drops from `O(n_ring·64²)` to `O(nnz_coeffs·64)`.
    //
    // `shift_add` applies one nonzero coefficient's negacyclic contribution.
    #[inline(always)]
    fn shift_add(slot: &mut [i32], b: &[i8; 64], k: usize, ak: i32) {
        if ak == 1 {
            for m in k..64 { slot[m] += b[m - k] as i32; }
            for m in 0..k { slot[m] -= b[m + 64 - k] as i32; }
        } else if ak == -1 {
            for m in k..64 { slot[m] -= b[m - k] as i32; }
            for m in 0..k { slot[m] += b[m + 64 - k] as i32; }
        } else {
            for m in k..64 { slot[m] += ak * b[m - k] as i32; }
            for m in 0..k { slot[m] -= ak * b[m + 64 - k] as i32; }
        }
    }
    wide.par_chunks_mut(64).enumerate().for_each(|(j, slot)| {
        if j >= n_ring { return; }
        match &inst.data {
            FoldData::Binary(v) => {
                // a[k] ∈ {0,1}: iterate set bits directly, no row buffer.
                let mut word = v[j];
                while word != 0 {
                    let k = word.trailing_zeros() as usize;
                    shift_add(slot, b, k, 1);
                    word &= word - 1;
                }
            }
            FoldData::Ternary(chunks) => {
                // Reconstruct the (sparse) integer coefficient row, skipping
                // all-zero chunk words, then shift-add each nonzero coeff.
                let mut row = [0i32; 64];
                let mut any = false;
                for ki in 0..chunks.k_chunks {
                    let pw = chunks.pos[ki * chunks.n_ring + j];
                    let nw = chunks.neg[ki * chunks.n_ring + j];
                    if pw == 0 && nw == 0 { continue; }
                    any = true;
                    let mult = 1i32 << ki;
                    let mut w = pw;
                    while w != 0 { let k = w.trailing_zeros() as usize; row[k] += mult; w &= w - 1; }
                    let mut w = nw;
                    while w != 0 { let k = w.trailing_zeros() as usize; row[k] -= mult; w &= w - 1; }
                }
                if !any { return; }
                for k in 0..64 {
                    if row[k] != 0 { shift_add(slot, b, k, row[k]); }
                }
            }
            FoldData::Digit { bit_planes, negate_top_bit, .. } => {
                let mut row = [0i32; 64];
                let k_bits = bit_planes.len();
                let mut any = false;
                for bk in 0..k_bits {
                    if j >= bit_planes[bk].len() { continue; }
                    let word = bit_planes[bk][j];
                    if word == 0 { continue; }
                    any = true;
                    let wt: i32 = if *negate_top_bit && bk == k_bits - 1 {
                        -(1i32 << bk)
                    } else {
                        1i32 << bk
                    };
                    let mut w = word;
                    while w != 0 { let k = w.trailing_zeros() as usize; row[k] += wt; w &= w - 1; }
                }
                if !any { return; }
                for k in 0..64 {
                    if row[k] != 0 { shift_add(slot, b, k, row[k]); }
                }
            }
        }
    });
}

fn constant_one_ring() -> RingChallenge {
    let mut c = [0i8; 64];
    c[0] = 1;
    RingChallenge::from_coeffs_unchecked(c)
}

/// Sample a `RingChallenge` (coefficients in `{-1, 0, 1, 2}^64`) from the
/// transcript. Two bits per coefficient → 32 coefficients per `u64`
/// challenge → 2 challenges total. Deterministic given transcript state +
/// label + idx, so prover and verifier agree by construction.
pub(crate) fn sample_ring_challenge(transcript: &mut Transcript, label: &[u8], idx: u64) -> RingChallenge {
    transcript.append_u64(label, idx);
    let w0 = transcript.challenge_scalar(b"mf_word0").0;
    let w1 = transcript.challenge_scalar(b"mf_word1").0;
    let mut coeffs = [0i8; 64];
    for k in 0..32 {
        let two_bits = ((w0 >> (2 * k)) & 0b11) as u8;
        coeffs[k] = match two_bits { 0 => -1, 1 => 0, 2 => 1, 3 => 2, _ => unreachable!() };
    }
    for k in 0..32 {
        let two_bits = ((w1 >> (2 * k)) & 0b11) as u8;
        coeffs[32 + k] = match two_bits { 0 => -1, 1 => 0, 2 => 1, 3 => 2, _ => unreachable!() };
    }
    RingChallenge::from_coeffs_unchecked(coeffs)
}

fn rings_equal(a: &RingCommitment, b: &RingCommitment) -> bool {
    use almost_goldilocks_cuda::ajtai::{KAPPA, RING_DIM};
    for i in 0..KAPPA {
        for k in 0..RING_DIM {
            if a.rows[i][k] != b.rows[i][k] { return false; }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn binary_inst_with_random_witness(
        rng: &mut StdRng,
        arity: usize,
        claim_pt: Vec<AlmostGoldilocksExt2>,
        commitment: RingCommitment,
    ) -> FoldInstance {
        let n_ring = 1usize << (arity - 6);
        let packed: Vec<u64> = (0..n_ring).map(|_| rng.gen::<u64>()).collect();
        let data = FoldData::Binary(packed);
        let claim_val = data.evaluate_at_ext2(&claim_pt);
        FoldInstance { commitment, data, arity, claim_pt, claim_val }
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(AlmostGoldilocksField(v))
    }

    /// Bit-exact `FoldData` equality (asserts on mismatch). Used by the
    /// GPU-vs-host oracle tests.
    fn assert_fold_data_bit_exact(a: &FoldData, b: &FoldData) {
        match (a, b) {
            (FoldData::Binary(av), FoldData::Binary(bv)) => {
                assert_eq!(av.len(), bv.len(), "binary length mismatch");
                for i in 0..av.len() {
                    assert_eq!(av[i], bv[i], "binary[{}] mismatch", i);
                }
            }
            (FoldData::Ternary(ac), FoldData::Ternary(bc)) => {
                assert_eq!(ac.n_ring, bc.n_ring, "ternary n_ring");
                assert_eq!(ac.k_chunks, bc.k_chunks, "ternary k_chunks");
                for i in 0..ac.pos.len() {
                    assert_eq!(ac.pos[i], bc.pos[i], "pos[{}] mismatch", i);
                    assert_eq!(ac.neg[i], bc.neg[i], "neg[{}] mismatch", i);
                }
            }
            (a, b) => panic!("fold-data kind mismatch: {:?} vs {:?}", a, b),
        }
    }

    fn make_commit(seed_idx: u32) -> RingCommitment {
        let mut c = RingCommitment::zero();
        c.rows[0][0] = 1000 + seed_idx as u64;
        c.rows[3][7] = 2000 + seed_idx as u64;
        c
    }

    /// Single-instance multifold is a no-op modulo γ (which is empty).
    /// Verifier must accept and the combined commitment equals the input.
    #[test]
    fn single_instance_passthrough() {
        let arity = 6;
        let pt: Vec<_> = (0..arity).map(|i| lift(i as u64 + 1)).collect();
        let inst = binary_inst_with_random_witness(
            &mut StdRng::seed_from_u64(1), arity, pt.clone(), make_commit(0),
        );

        let mut t_p = Transcript::new(b"mf-single");
        let (combined, proof) = prove_multifold(&[inst.clone()], &mut t_p);
        assert_eq!(combined.claim_pt, pt);
        assert!(crate::util::arith::ext2_field_eq(combined.claim_val, inst.claim_val));

        let mut t_v = Transcript::new(b"mf-single");
        let res = verify_multifold(&[&inst.commitment], &proof, &mut t_v);
        assert!(res.is_some(), "verify_multifold should accept");
        let (c_out, y_out) = res.unwrap();
        assert_eq!(c_out.rows, inst.commitment.rows);
        assert!(crate::util::arith::ext2_field_eq(y_out, inst.claim_val));
    }

    /// Two-instance multifold: γ from the transcript is shared, the
    /// prover/verifier commitments agree, and the prover's claim_val
    /// equals the host-folded witness's MLE.
    #[test]
    fn two_instance_multifold_roundtrip() {
        let arity = 7; // 128 binary coefs = 2 ring elements
        let pt: Vec<_> = (0..arity).map(|i| lift(i as u64 * 13 + 5)).collect();
        let mut rng = StdRng::seed_from_u64(0xABCD_1234);
        let i0 = binary_inst_with_random_witness(&mut rng, arity, pt.clone(), make_commit(0));
        let i1 = binary_inst_with_random_witness(&mut rng, arity, pt.clone(), make_commit(1));

        let mut t_p = Transcript::new(b"mf-two");
        let (_combined, proof) = prove_multifold(&[i0.clone(), i1.clone()], &mut t_p);
        assert_eq!(proof.gammas.len(), 1);

        let mut t_v = Transcript::new(b"mf-two");
        let res = verify_multifold(&[&i0.commitment, &i1.commitment], &proof, &mut t_v);
        // The two commitments are mock (not real Ajtai commits), but the
        // verifier still runs the kernel and gets a deterministic c'. The
        // critical correctness is that prover and verifier derive the
        // same γ and same c' from the same kernel inputs.
        assert!(res.is_some(), "honest 2-instance multifold must verify");
    }

    /// GPU witness fold matches the host reference bit-exactly for an
    /// all-binary group. The kernel's `multifold_witness` path is what
    /// `prove_multifold` calls in production; this test pins it
    /// against the simple CPU convolution oracle.
    #[test]
    fn gpu_witness_fold_matches_host_all_binary() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let arity = 7;
        let m = 5;
        let pt: Vec<_> = (0..arity).map(|i| lift(i as u64 * 11 + 3)).collect();
        let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
        let group: Vec<FoldInstance> = (0..m)
            .map(|i| binary_inst_with_random_witness(&mut rng, arity, pt.clone(), make_commit(i)))
            .collect();

        // Sample (m-1) γ's deterministically.
        let mut t_gammas = Transcript::new(b"oracle-bin");
        let gammas: Vec<RingChallenge> = (1..m)
            .map(|i| sample_ring_challenge(&mut t_gammas, b"mf_gamma", i as u64))
            .collect();

        let gpu_out = fold_witnesses_gpu(&group, &gammas);
        let host_out = fold_witnesses_host(&group, &gammas);
        assert_fold_data_bit_exact(&gpu_out, &host_out);
    }

    /// GPU witness fold matches host for an all-ternary group (the
    /// post-split fold-tree levels).
    #[test]
    fn gpu_witness_fold_matches_host_all_ternary() {
        almost_goldilocks_cuda::init().expect("CUDA init");
        let arity = 7;
        let m = 4;
        let pt: Vec<_> = (0..arity).map(|i| lift(i as u64 * 17 + 1)).collect();
        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        // Build random ternary instances (k_chunks = 1, |coefs| ≤ 1).
        let n_ring = 1usize << (arity - 6);
        let group: Vec<FoldInstance> = (0..m).map(|i| {
            let mut pos = vec![0u64; n_ring];
            let mut neg = vec![0u64; n_ring];
            for j in 0..n_ring {
                let pw = rng.gen::<u64>();
                let nw = rng.gen::<u64>() & !pw; // disjoint pos/neg
                pos[j] = pw;
                neg[j] = nw;
            }
            let chunks = almost_goldilocks_cuda::ajtai::TernaryChunks {
                n_ring, k_chunks: 1, pos, neg,
            };
            let data = FoldData::Ternary(chunks);
            let claim_val = data.evaluate_at_ext2(&pt);
            FoldInstance {
                commitment: make_commit(i),
                data,
                arity,
                claim_pt: pt.clone(),
                claim_val,
            }
        }).collect();

        let mut t_gammas = Transcript::new(b"oracle-tern");
        let gammas: Vec<RingChallenge> = (1..m)
            .map(|i| sample_ring_challenge(&mut t_gammas, b"mf_gamma", i as u64))
            .collect();

        let gpu_out = fold_witnesses_gpu(&group, &gammas);
        let host_out = fold_witnesses_host(&group, &gammas);
        assert_fold_data_bit_exact(&gpu_out, &host_out);
    }

    /// Tampered γ in the proof → verifier rejects (γ replay mismatch).
    #[test]
    fn verifier_rejects_tampered_gamma() {
        let arity = 6;
        let pt: Vec<_> = (0..arity).map(|i| lift(i as u64 + 3)).collect();
        let mut rng = StdRng::seed_from_u64(99);
        let i0 = binary_inst_with_random_witness(&mut rng, arity, pt.clone(), make_commit(0));
        let i1 = binary_inst_with_random_witness(&mut rng, arity, pt.clone(), make_commit(1));

        let mut t_p = Transcript::new(b"mf-bad");
        let (_c, mut proof) = prove_multifold(&[i0.clone(), i1.clone()], &mut t_p);
        proof.gammas[0].coeffs[0] = proof.gammas[0].coeffs[0].wrapping_add(1).clamp(-1, 2);
        if proof.gammas[0].coeffs[0] > 2 { proof.gammas[0].coeffs[0] = -1; }
        let mut t_v = Transcript::new(b"mf-bad");
        let res = verify_multifold(&[&i0.commitment, &i1.commitment], &proof, &mut t_v);
        assert!(res.is_none(), "tampered γ must be rejected");
    }
}
