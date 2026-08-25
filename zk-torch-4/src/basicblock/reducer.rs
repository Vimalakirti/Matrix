//! [`Reducer`] — collapses multiple claims about the same polynomial into one
//! via the random-linear-combination sumcheck
//! `Σ_i α^i · eq(r_i, x) · f(x) = Σ_i α^i · v_i`.
//!
//! Both prover paths (CPU and GPU) are kept, with an explicit threshold-based
//! dispatch at the call site (`Reducer::prove` here). The GPU path uses
//! `GpuSumcheckStateExt2::from_device_buffers` so the per-claim eq tables
//! never round-trip through host memory.

use std::os::raw::c_void;
use std::sync::OnceLock;

use almost_goldilocks_cuda::eq_lagrange::ext2_eq_dp_all_device;
use almost_goldilocks_cuda::extension::{AlmostExt2Batch, AlmostGoldilocksExt2};
use almost_goldilocks_cuda::memory::{memcpy_dtod, DeviceBuffer};
use almost_goldilocks_cuda::sumcheck_prover::GpuSumcheckStateExt2;

use crate::basicblock::BasicBlock;
use crate::dag::{Claim, Witness};
use crate::poly::evaluate_lagrange_basis_ext2;
use crate::sumcheck::{
    CpuLinearSumcheckProverExt2, GpuLinearSumcheckProver, SumcheckProof, SumcheckVerifier,
};
use crate::transcript::Transcript;
use crate::util::arith::{
    calc_pow_vec_ext2, ext2_add, ext2_field_eq, ext2_mul, ext2_sub, get_n,
};

/// Polynomials with `n` ≤ this threshold use the CPU prover (GPU launch
/// overhead dominates). Override with `ZK_GPU_SUMCHECK_THRESHOLD`.
fn gpu_sumcheck_threshold() -> usize {
    static VAL: OnceLock<usize> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("ZK_GPU_SUMCHECK_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(14)
    })
}

/// Per-phase thread-time counters for `prove_with_cached_buffers`, summed
/// across all reducer jobs in an acc-update. Reported (and reset) by the
/// streaming accumulator under ZK4_TIMING. Microseconds.
pub static REDUCER_EQ_BUILD_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static REDUCER_EQ_COPY_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static REDUCER_SCALE_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static REDUCER_SUMCHECK_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct Reducer;

impl BasicBlock for Reducer {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        assert_eq!(inputs.len(), 1, "Reducer expects 1 input");
        vec![inputs[0].clone()]
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        assert_eq!(witnesses.len(), 1, "Reducer expects 1 input");
        let alpha = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, out_claims.len());

        let x = witnesses[0];
        let n = x.data.as_ref().unwrap().n();
        let size = 1usize << n;

        if n <= gpu_sumcheck_threshold() {
            return self.prove_cpu(x, edge_ids, out_claims, &alphas, n, size, transcript);
        }
        self.prove_gpu(x, edge_ids, out_claims, &alphas, n, size, transcript)
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        assert_eq!(witnesses.len(), 1, "Reducer expects 1 input");
        let alpha = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, claims.len() - 1);
        let x = witnesses[0];

        // claimed_sum = Σ α^i · v_i over the K original claims (claims[..K-1]).
        let mut claimed_sum = AlmostGoldilocksExt2::zero();
        for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
            claimed_sum = ext2_add(claimed_sum, ext2_mul(claim.eval, alphas[i]));
        }

        let n_verify = get_n(&x.shape);
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            claimed_sum,
            n_verify,
            2,
            transcript,
        );
        if !ok {
            return false;
        }

        // Final-eval check: proof.final_eval == x(R) · Σ_i α^i · eq(R, r_i).
        let one = AlmostGoldilocksExt2::one();
        let mut eq_eval = AlmostGoldilocksExt2::zero();
        for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
            let mut eq = one;
            for j in 0..challenges.len() {
                let r_j = challenges[j];
                let p_j = claim.point[j];
                let term = ext2_add(
                    ext2_mul(r_j, p_j),
                    ext2_mul(ext2_sub(one, r_j), ext2_sub(one, p_j)),
                );
                eq = ext2_mul(eq, term);
            }
            eq_eval = ext2_add(eq_eval, ext2_mul(alphas[i], eq));
        }
        let x_eval = claims[claims.len() - 1].eval;
        let expected = ext2_mul(x_eval, eq_eval);
        ext2_field_eq(sumcheck_proofs[0].final_eval, expected)
    }
}

impl Reducer {
    /// Streaming-friendly variant of `prove_gpu`: accepts a pre-lifted
    /// Ext2 witness buffer (`d_x_ext2`) so the caller can cache it
    /// across many calls on the same polynomial. Saves the per-call
    /// `from_base` lift, which is the dominant cost (~55% of the GPU
    /// path's wall time at n=24 — see profiling notes in the streaming
    /// accumulator). All other steps (sample α, eq tables, sumcheck)
    /// are identical to `prove_gpu`.
    ///
    /// Caller MUST ensure `d_x_ext2` length is exactly `size * 2` (u64s)
    /// where `size = 1 << x_n`, AND that it holds the Ext2 lift of the
    /// witness polynomial whose claims are being reduced.
    pub fn prove_with_cached_x_ext2(
        &self,
        d_x_ext2: &DeviceBuffer<u64>,
        x_n: usize,
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let n = x_n;
        let size = 1usize << n;
        assert_eq!(
            d_x_ext2.len(),
            size * 2,
            "prove_with_cached_x_ext2: d_x_ext2 length mismatch",
        );
        let alpha = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, out_claims.len());

        // Build combined eq buffer (Σ α^i · eq(r_i, ·)) on device.
        // Method #1: zero-init via `cudaMemset` (single GPU launch) instead
        // of `vec![0u64; size*2]` + `copy_from_slice` (host alloc + PCIe
        // upload). Saves ~30 ms per call at n=22, ~120 ms at n=24, summing
        // to ~500 ms saved per Llama stream-update across 142 weights.
        let mut d_acc =
            DeviceBuffer::<u64>::new(size * 2).expect("Reducer: acc alloc failed");
        d_acc.zero().expect("Reducer: acc zero-init failed");
        for (idx, claim) in out_claims.iter().enumerate() {
            let d_r = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&claim.point)
                .expect("Reducer: r upload failed");
            let log_n = claim.point.len();
            let (d_buf_a, d_buf_b, result_in_a) =
                ext2_eq_dp_all_device(&d_r, log_n).expect("ext2_eq_dp_all_device failed");
            let d_eq_ext2 = if result_in_a { &d_buf_a } else { &d_buf_b };
            let mut d_eq_u64 =
                DeviceBuffer::<u64>::new(size * 2).expect("Reducer: eq u64 alloc failed");
            unsafe {
                memcpy_dtod(
                    d_eq_u64.as_mut_ptr() as *mut c_void,
                    d_eq_ext2.as_ptr() as *const c_void,
                    size * 2 * std::mem::size_of::<u64>(),
                )
                .expect("Reducer: eq D2D copy failed");
            }
            AlmostExt2Batch::scale_accumulate(alphas[idx], &d_eq_u64, &mut d_acc)
                .expect("Reducer: scale_accumulate failed");
        }

        // Skip the witness lift — use the cached d_x_ext2 directly.
        let buf_refs: Vec<&DeviceBuffer<u64>> = vec![d_x_ext2, &d_acc];
        let gpu_state = GpuSumcheckStateExt2::from_device_buffers(&buf_refs, size)
            .expect("Reducer: from_device_buffers failed");

        let mut prover = GpuLinearSumcheckProver::new(n, 2, transcript);
        let proof = prover.prove_gpu_resident(gpu_state, transcript);
        let challenges = prover.challenges.clone();

        let claim_x = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges,
            eval: prover.final_eval(0),
        };
        (vec![proof], vec![claim_x])
    }

    /// Method #2 variant: accepts BOTH a pre-lifted `d_x_ext2` AND a
    /// pre-allocated `d_acc` (size `2*2^x_n` u64s). The caller owns
    /// `d_acc` across many calls — this fn just `cudaMemset`s it to
    /// zero at the start of each call and accumulates Σ α^i · eq(r_i,·)
    /// into it. Skips the per-call `cudaMalloc(size*2)` that the
    /// non-cached variant pays.
    ///
    /// Caller MUST ensure both buffers have length `2 * (1 << x_n)`.
    pub fn prove_with_cached_buffers(
        &self,
        d_x_ext2: &DeviceBuffer<u64>,
        d_acc: &mut DeviceBuffer<u64>,
        x_n: usize,
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let n = x_n;
        let size = 1usize << n;
        assert_eq!(
            d_x_ext2.len(),
            size * 2,
            "prove_with_cached_buffers: d_x_ext2 length mismatch",
        );
        assert_eq!(
            d_acc.len(),
            size * 2,
            "prove_with_cached_buffers: d_acc length mismatch",
        );
        let alpha = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, out_claims.len());

        use std::sync::atomic::Ordering;
        let timing = std::env::var("ZK4_TIMING").ok().as_deref() == Some("1");

        // Method #1: zero-init via cudaMemset (no PCIe).
        d_acc.zero().expect("Reducer: acc zero-init failed");

        for (idx, claim) in out_claims.iter().enumerate() {
            let t_eq = std::time::Instant::now();
            let d_r = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&claim.point)
                .expect("Reducer: r upload failed");
            let log_n = claim.point.len();
            let (d_buf_a, d_buf_b, result_in_a) =
                ext2_eq_dp_all_device(&d_r, log_n).expect("ext2_eq_dp_all_device failed");
            let d_eq_ext2 = if result_in_a { &d_buf_a } else { &d_buf_b };
            if timing { REDUCER_EQ_BUILD_US.fetch_add(t_eq.elapsed().as_micros() as u64, Ordering::Relaxed); }
            // Accumulate the eq table directly from its Ext2 buffer —
            // Ext2 is repr(C) of two u64s, so no copy / no u64 scratch
            // alloc (was a full 2^n D2D copy + cudaMalloc per claim).
            let t_sc = std::time::Instant::now();
            AlmostExt2Batch::scale_accumulate_from_ext2(alphas[idx], d_eq_ext2, d_acc)
                .expect("Reducer: scale_accumulate_from_ext2 failed");
            if timing { REDUCER_SCALE_US.fetch_add(t_sc.elapsed().as_micros() as u64, Ordering::Relaxed); }
        }

        let t_sm = std::time::Instant::now();
        let buf_refs: Vec<&DeviceBuffer<u64>> = vec![d_x_ext2, &*d_acc];
        let gpu_state = GpuSumcheckStateExt2::from_device_buffers(&buf_refs, size)
            .expect("Reducer: from_device_buffers failed");

        let mut prover = GpuLinearSumcheckProver::new(n, 2, transcript);
        let proof = prover.prove_gpu_resident(gpu_state, transcript);
        let challenges = prover.challenges.clone();
        if timing { REDUCER_SUMCHECK_US.fetch_add(t_sm.elapsed().as_micros() as u64, Ordering::Relaxed); }

        let claim_x = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges,
            eval: prover.final_eval(0),
        };
        (vec![proof], vec![claim_x])
    }

    /// Factored-eq (Gruen) variant of [`Self::prove_with_cached_buffers`]
    /// for the streaming accumulator: same inputs, byte-identical proof
    /// (round messages + final_eval) and new claim, but the two eq tables
    /// are never materialized or accumulated — a [`GpuReducerFactoredState`]
    /// keeps factored suffix stages + host prefix scalars and folds only
    /// the shared witness. `d_acc` is unused here (kept in the signature so
    /// the caller's per-edge cache plumbing is identical); pass any buffer.
    ///
    /// Exactly TWO claims (prior + incoming), the streaming reducer's case.
    pub fn prove_with_cached_buffers_factored(
        &self,
        d_x_ext2: &DeviceBuffer<u64>,
        x_n: usize,
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        use almost_goldilocks_cuda::sumcheck_prover::GpuReducerFactoredState;
        assert_eq!(out_claims.len(), 2,
            "factored reducer handles exactly 2 claims (prior + incoming)");
        let n = x_n;
        assert_eq!(d_x_ext2.len(), (1usize << n) * 2,
            "prove_with_cached_buffers_factored: d_x_ext2 length mismatch");

        // Transcript: identical schedule to GpuLinearSumcheckProver
        // (reducer_alpha, then num_var/num_poly header, then per round
        // 3 round-message appends + 1 challenge), so the verifier
        // (verify_with_point → SumcheckVerifier::verify) replays unchanged.
        let alpha = transcript.challenge_ext2(b"reducer_alpha");
        let r0 = &out_claims[0].point;
        let r1 = &out_claims[1].point;
        let mut state = GpuReducerFactoredState::new(d_x_ext2, r0, r1, alpha)
            .expect("GpuReducerFactoredState::new failed");

        transcript.append_u64(b"num_var", n as u64);
        transcript.append_u64(b"num_poly", 2u64);
        let mut round_messages: Vec<Vec<AlmostGoldilocksExt2>> = Vec::with_capacity(n);
        let mut challenges: Vec<AlmostGoldilocksExt2> = Vec::with_capacity(n);
        for _ in 0..n {
            let msg = state.compute_round_message().expect("factored round msg");
            for m in &msg { transcript.append_ext2(b"round_message", m); }
            let c = transcript.challenge_ext2(b"challenge");
            round_messages.push(msg.to_vec());
            state.fold(c).expect("factored fold");
            challenges.push(c);
        }

        let x_eval = state.f_final().expect("factored f_final");
        let final_eval = ext2_mul(x_eval, state.eq_combined_final());
        let proof = SumcheckProof { final_eval, round_messages };
        let claim_x = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges,
            eval: x_eval,
        };
        (vec![proof], vec![claim_x])
    }

    /// Same as `<Reducer as BasicBlock>::verify`, but also returns the
    /// sumcheck challenges (= the new aggregated claim's point) when the
    /// proof verifies. Used by the streaming accumulator's verifier to
    /// reconstruct the new accumulated `Claim` after each reducer step.
    /// Returns `None` if any check fails.
    ///
    /// `claims[..K]` are the K input claims to reduce; `claims[K]`
    /// supplies `x_eval = new_claim.eval` (its `point` field is ignored,
    /// since this method's job is to derive that point).
    pub fn verify_with_point(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> Option<Vec<AlmostGoldilocksExt2>> {
        assert_eq!(witnesses.len(), 1, "Reducer expects 1 input");
        let alpha = transcript.challenge_ext2(b"reducer_alpha");
        let alphas = calc_pow_vec_ext2(alpha, claims.len() - 1);
        let x = witnesses[0];

        let mut claimed_sum = AlmostGoldilocksExt2::zero();
        for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
            claimed_sum = ext2_add(claimed_sum, ext2_mul(claim.eval, alphas[i]));
        }

        let n_verify = get_n(&x.shape);
        let (ok, challenges) = SumcheckVerifier::verify(
            sumcheck_proofs[0],
            claimed_sum,
            n_verify,
            2,
            transcript,
        );
        if !ok {
            return None;
        }

        let one = AlmostGoldilocksExt2::one();
        let mut eq_eval = AlmostGoldilocksExt2::zero();
        for (i, claim) in claims[..claims.len() - 1].iter().enumerate() {
            let mut eq = one;
            for j in 0..challenges.len() {
                let r_j = challenges[j];
                let p_j = claim.point[j];
                let term = ext2_add(
                    ext2_mul(r_j, p_j),
                    ext2_mul(ext2_sub(one, r_j), ext2_sub(one, p_j)),
                );
                eq = ext2_mul(eq, term);
            }
            eq_eval = ext2_add(eq_eval, ext2_mul(alphas[i], eq));
        }
        let x_eval = claims[claims.len() - 1].eval;
        let expected = ext2_mul(x_eval, eq_eval);
        if !ext2_field_eq(sumcheck_proofs[0].final_eval, expected) {
            return None;
        }
        Some(challenges)
    }

    /// CPU path: build `eq_combined = Σ α^i · eq(r_i, x)` on host, lift the
    /// base-field witness to Ext2, and feed both into the CPU sumcheck prover.
    fn prove_cpu(
        &self,
        x: &Witness,
        edge_ids: &[usize],
        out_claims: &[&Claim],
        alphas: &[AlmostGoldilocksExt2],
        n: usize,
        size: usize,
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let mut eq_combined = vec![AlmostGoldilocksExt2::zero(); size];
        for (idx, claim) in out_claims.iter().enumerate() {
            let eq_table = evaluate_lagrange_basis_ext2(&claim.point);
            for j in 0..size {
                eq_combined[j] = ext2_add(eq_combined[j], ext2_mul(alphas[idx], eq_table[j]));
            }
        }

        let x_evals = x.data.as_ref().unwrap().evaluations_ref();
        let x_ext2: Vec<AlmostGoldilocksExt2> = x_evals
            .iter()
            .map(|&v| AlmostGoldilocksExt2::from_base(v))
            .collect();

        let mut prover = CpuLinearSumcheckProverExt2::new(n, 2, transcript);
        let mut polys = vec![x_ext2, eq_combined];
        let proof = prover.prove(&mut polys, transcript);
        let challenges = prover.challenges.clone();

        let claim_x = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges,
            eval: prover.final_eval(0),
        };
        (vec![proof], vec![claim_x])
    }

    /// GPU path: build the eq-table and the Ext2-lifted witness on-device,
    /// then hand both off to `GpuSumcheckStateExt2::from_device_buffers` —
    /// no host round-trip for the per-claim eq tables.
    fn prove_gpu(
        &self,
        x: &Witness,
        edge_ids: &[usize],
        out_claims: &[&Claim],
        alphas: &[AlmostGoldilocksExt2],
        n: usize,
        size: usize,
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        let profile = std::env::var("ZK_REDUCER_PROFILE").ok().as_deref() == Some("1");
        let t_total = std::time::Instant::now();
        let mut t_acc_alloc = std::time::Duration::ZERO;
        let mut t_eq = std::time::Duration::ZERO;
        let mut t_acc_eq_combine = std::time::Duration::ZERO;
        let mut t_x_lift = std::time::Duration::ZERO;
        let mut t_sumcheck = std::time::Duration::ZERO;

        // Accumulator for Σ α^i · eq(r_i, x), Ext2-packed as [c0, c1, c0, c1, ...].
        // Method #1: cudaMemset on device (single GPU launch) — no host
        // zero-vec alloc and no PCIe upload.
        let s = std::time::Instant::now();
        let mut d_acc =
            DeviceBuffer::<u64>::new(size * 2).expect("Reducer: acc alloc failed");
        d_acc.zero().expect("Reducer: acc zero-init failed");
        if profile { almost_goldilocks_cuda::synchronize().ok(); }
        t_acc_alloc += s.elapsed();

        for (idx, claim) in out_claims.iter().enumerate() {
            let s = std::time::Instant::now();
            // Upload r_i and compute eq(r_i, ·) on GPU.
            let d_r = DeviceBuffer::<AlmostGoldilocksExt2>::from_slice(&claim.point)
                .expect("Reducer: r upload failed");
            let log_n = claim.point.len();
            let (d_buf_a, d_buf_b, result_in_a) =
                ext2_eq_dp_all_device(&d_r, log_n).expect("ext2_eq_dp_all_device failed");
            let d_eq_ext2 = if result_in_a { &d_buf_a } else { &d_buf_b };
            if profile { almost_goldilocks_cuda::synchronize().ok(); }
            t_eq += s.elapsed();

            let s = std::time::Instant::now();
            // The eq result is a DeviceBuffer<AlmostGoldilocksExt2>; reinterpret
            // as `2 · size` u64s for scale_accumulate.
            let mut d_eq_u64 =
                DeviceBuffer::<u64>::new(size * 2).expect("Reducer: eq u64 alloc failed");
            unsafe {
                memcpy_dtod(
                    d_eq_u64.as_mut_ptr() as *mut c_void,
                    d_eq_ext2.as_ptr() as *const c_void,
                    size * 2 * std::mem::size_of::<u64>(),
                )
                .expect("Reducer: eq D2D copy failed");
            }
            AlmostExt2Batch::scale_accumulate(alphas[idx], &d_eq_u64, &mut d_acc)
                .expect("Reducer: scale_accumulate failed");
            if profile { almost_goldilocks_cuda::synchronize().ok(); }
            t_acc_eq_combine += s.elapsed();
        }

        let s = std::time::Instant::now();
        // Upload the base-field witness and lift to Ext2 on-device.
        let d_x_base = x.as_device_buf();
        let mut d_x_ext2 =
            DeviceBuffer::<u64>::new(size * 2).expect("Reducer: x_ext2 alloc failed");
        AlmostExt2Batch::from_base(&d_x_base, &mut d_x_ext2)
            .expect("Reducer: base→Ext2 failed");
        if profile { almost_goldilocks_cuda::synchronize().ok(); }
        t_x_lift += s.elapsed();

        let s = std::time::Instant::now();
        let buf_refs: Vec<&DeviceBuffer<u64>> = vec![&d_x_ext2, &d_acc];
        let gpu_state = GpuSumcheckStateExt2::from_device_buffers(&buf_refs, size)
            .expect("Reducer: from_device_buffers failed");

        let mut prover = GpuLinearSumcheckProver::new(n, 2, transcript);
        let proof = prover.prove_gpu_resident(gpu_state, transcript);
        let challenges = prover.challenges.clone();
        if profile { almost_goldilocks_cuda::synchronize().ok(); }
        t_sumcheck += s.elapsed();

        if profile {
            let total = t_total.elapsed();
            eprintln!(
                "[reducer-gpu n={:>2}] total {:>7.2}ms = alloc {:>6.2} + eq {:>7.2} + combine {:>7.2} + x_lift {:>6.2} + sumcheck {:>7.2}",
                n,
                total.as_secs_f64() * 1e3,
                t_acc_alloc.as_secs_f64() * 1e3,
                t_eq.as_secs_f64() * 1e3,
                t_acc_eq_combine.as_secs_f64() * 1e3,
                t_x_lift.as_secs_f64() * 1e3,
                t_sumcheck.as_secs_f64() * 1e3,
            );
        }

        let claim_x = Claim {
            edge_id: edge_ids[0],
            sparse_id: 0,
            point: challenges,
            eval: prover.final_eval(0),
        };
        (vec![proof], vec![claim_x])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;
    use crate::dag::{DataType, Role};
    use crate::poly::DenseMLPoly;

    fn agl(v: u64) -> AlmostGoldilocksField {
        AlmostGoldilocksField(v)
    }

    fn lift(v: u64) -> AlmostGoldilocksExt2 {
        AlmostGoldilocksExt2::from_base(agl(v))
    }

    fn build_witness(evals: Vec<u64>) -> Witness {
        let n = evals.len().trailing_zeros() as usize;
        let poly = DenseMLPoly::new(n, evals.into_iter().map(agl).collect());
        Witness::new_dense_poly(
            vec![1usize << n],
            poly,
            DataType::Int,
            10,
            Role::Auxiliary,
        )
    }

    /// Force the CPU path by keeping `n` below the default threshold (14).
    #[test]
    fn reducer_cpu_prove_then_verify() {
        let evals: Vec<u64> = (1..=8u64).collect();
        let w = build_witness(evals);
        let f = w.data.as_ref().unwrap();

        // Three claims at three different points.
        let r1 = vec![lift(3), lift(5), lift(7)];
        let r2 = vec![lift(11), lift(13), lift(17)];
        let r3 = vec![lift(2), lift(4), lift(6)];
        let v1 = f.evaluate_at_point_ext2(&r1);
        let v2 = f.evaluate_at_point_ext2(&r2);
        let v3 = f.evaluate_at_point_ext2(&r3);
        let c1 = Claim { edge_id: 0, sparse_id: 0, point: r1, eval: v1 };
        let c2 = Claim { edge_id: 0, sparse_id: 0, point: r2, eval: v2 };
        let c3 = Claim { edge_id: 0, sparse_id: 0, point: r3, eval: v3 };

        let mut t_prove = Transcript::new(b"red-cpu");
        let (proofs, new_claims) = Reducer.prove(
            &[&w],
            &[0],
            &[&c1, &c2, &c3],
            &mut t_prove,
        );
        assert_eq!(proofs.len(), 1);
        assert_eq!(new_claims.len(), 1);

        let mut t_verify = Transcript::new(b"red-cpu");
        let all = [&c1, &c2, &c3, &new_claims[0]];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(Reducer.verify(&[&w], &all, &proof_refs, &mut t_verify));

        // The reduced claim must evaluate f at the prover's challenge point.
        let direct = f.evaluate_at_point_ext2(&new_claims[0].point);
        assert!(ext2_field_eq(direct, new_claims[0].eval));
    }

    fn cuda_ready() -> bool {
        almost_goldilocks_cuda::init().is_ok()
    }

    /// Force the GPU path with `n = 15 > 14`. Bit-exact prove → verify
    /// roundtrip against the same transcript label.
    #[test]
    fn reducer_gpu_prove_then_verify() {
        if !cuda_ready() {
            eprintln!("skipping GPU test: CUDA not available");
            return;
        }
        // Make sure the threshold is the default (don't honor a user-set
        // override that would knock us back to CPU mid-test).
        std::env::remove_var("ZK_GPU_SUMCHECK_THRESHOLD");
        assert!(gpu_sumcheck_threshold() == 14);

        let n_var = 15;
        let size = 1usize << n_var;
        let evals: Vec<u64> = (0..size as u64).map(|i| i.wrapping_mul(7) + 3).collect();
        let w = build_witness(evals);
        let f = w.data.as_ref().unwrap();

        let r1: Vec<_> = (0..n_var).map(|i| lift(i as u64 * 31 + 5)).collect();
        let r2: Vec<_> = (0..n_var).map(|i| lift(i as u64 * 17 + 11)).collect();
        let v1 = f.evaluate_at_point_ext2(&r1);
        let v2 = f.evaluate_at_point_ext2(&r2);
        let c1 = Claim { edge_id: 0, sparse_id: 0, point: r1, eval: v1 };
        let c2 = Claim { edge_id: 0, sparse_id: 0, point: r2, eval: v2 };

        let mut t_prove = Transcript::new(b"red-gpu");
        let (proofs, new_claims) = Reducer.prove(&[&w], &[0], &[&c1, &c2], &mut t_prove);
        assert_eq!(new_claims.len(), 1);

        let mut t_verify = Transcript::new(b"red-gpu");
        let all = [&c1, &c2, &new_claims[0]];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(Reducer.verify(&[&w], &all, &proof_refs, &mut t_verify));

        // Reduced claim evaluates f at the new point.
        let direct = f.evaluate_at_point_ext2(&new_claims[0].point);
        assert!(ext2_field_eq(direct, new_claims[0].eval));
    }

    /// Tampered final eval is rejected.
    #[test]
    fn reducer_verify_rejects_tampered_proof() {
        let evals: Vec<u64> = (1..=4u64).collect();
        let w = build_witness(evals);
        let f = w.data.as_ref().unwrap();
        let r1 = vec![lift(3), lift(5)];
        let v1 = f.evaluate_at_point_ext2(&r1);
        let c1 = Claim { edge_id: 0, sparse_id: 0, point: r1, eval: v1 };
        let mut t_prove = Transcript::new(b"red-tamper");
        let (mut proofs, new_claims) = Reducer.prove(&[&w], &[0], &[&c1], &mut t_prove);
        proofs[0].final_eval = proofs[0].final_eval + lift(1);
        let mut t_verify = Transcript::new(b"red-tamper");
        let all = [&c1, &new_claims[0]];
        let proof_refs: Vec<&SumcheckProof> = proofs.iter().collect();
        assert!(!Reducer.verify(&[&w], &all, &proof_refs, &mut t_verify));
    }

    /// The factored-eq reducer must produce a BYTE-IDENTICAL proof (round
    /// messages + final_eval) and new claim to the materialized
    /// `prove_with_cached_buffers`, and verify against the unchanged
    /// verifier. n = 15 forces the real GPU sizes.
    #[test]
    fn reducer_factored_matches_materialized() {
        if !cuda_ready() { eprintln!("skipping: no CUDA"); return; }
        use almost_goldilocks_cuda::memory::DeviceBuffer;
        use almost_goldilocks_cuda::extension::AlmostExt2Batch;
        let n = 15usize;
        let size = 1usize << n;
        let evals: Vec<u64> = (0..size as u64).map(|i| (i * 2654435761) % 97 + 1).collect();
        let w = build_witness(evals);
        let f = w.data.as_ref().unwrap();

        let r0: Vec<_> = (0..n).map(|i| lift(i as u64 * 7 + 3)).collect();
        let r1: Vec<_> = (0..n).map(|i| lift(i as u64 * 11 + 5)).collect();
        let v0 = f.evaluate_at_point_ext2(&r0);
        let v1 = f.evaluate_at_point_ext2(&r1);
        let c0 = Claim { edge_id: 0, sparse_id: 0, point: r0, eval: v0 };
        let c1 = Claim { edge_id: 0, sparse_id: 0, point: r1, eval: v1 };
        let out: &[&Claim] = &[&c0, &c1];

        // Build the cached lifted-witness buffer once (shared by both paths).
        let d_x_base = w.as_device_buf();
        let mut d_x_ext2 = DeviceBuffer::<u64>::new(size * 2).unwrap();
        AlmostExt2Batch::from_base(&d_x_base, &mut d_x_ext2).unwrap();

        // Materialized path.
        let mut d_acc = DeviceBuffer::<u64>::new(size * 2).unwrap();
        let mut t_mat = Transcript::new(b"red-fac-eq");
        let (mat_proofs, mat_claims) = Reducer.prove_with_cached_buffers(
            &d_x_ext2, &mut d_acc, n, &[0], out, &mut t_mat);

        // Factored path.
        let mut t_fac = Transcript::new(b"red-fac-eq");
        let (fac_proofs, fac_claims) = Reducer.prove_with_cached_buffers_factored(
            &d_x_ext2, n, &[0], out, &mut t_fac);

        // Byte-identical round messages, final_eval, and new claim.
        assert_eq!(mat_proofs[0].round_messages.len(), fac_proofs[0].round_messages.len());
        for (rm, rf) in mat_proofs[0].round_messages.iter().zip(&fac_proofs[0].round_messages) {
            assert_eq!(rm.len(), rf.len());
            for (a, b) in rm.iter().zip(rf) {
                assert!(ext2_field_eq(*a, *b), "round message diverged");
            }
        }
        assert!(ext2_field_eq(mat_proofs[0].final_eval, fac_proofs[0].final_eval),
            "final_eval diverged");
        assert!(ext2_field_eq(mat_claims[0].eval, fac_claims[0].eval), "claim eval diverged");
        assert_eq!(mat_claims[0].point.len(), fac_claims[0].point.len());
        for (a, b) in mat_claims[0].point.iter().zip(&fac_claims[0].point) {
            assert!(ext2_field_eq(*a, *b), "claim point diverged");
        }

        // Factored proof verifies against the unchanged verifier.
        let mut t_v = Transcript::new(b"red-fac-eq");
        let all = [&c0, &c1, &fac_claims[0]];
        let pr: Vec<&SumcheckProof> = fac_proofs.iter().collect();
        assert!(Reducer.verify_with_point(&[&w], &all, &pr, &mut t_v).is_some(),
            "factored reducer proof must verify");
    }
}
