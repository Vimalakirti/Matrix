# Independent targeted review — Q20 mean/RMS and CNN means

## Verdict and pinned scope

**No remaining blocking mathematical/source defect found in the declared bounded increment after the repairs below.** This approves only the local mean/RMS construction and its specified domains. It is not whole-model/full-proof certification, a performance or accuracy result, or one of the five complete independent paper acceptance reviews.

Reviewed source: `<candidate>`, based on frozen `<exp-base>`.

Final 16-file delta manifest: `<evidence>/source-manifest.json`.

**Manifest SHA-256: `60b90a1c996cdb0c86becb35005563e289b0e8c2abfbf68a331d530e3de70e3a`.**

I recomputed all16 candidate hashes and corresponding base hashes: all match. `git diff --check` passes. This supersedes the earlier SF15-only and15-file Q20 review checkpoints, which remain intermediate records. I did not edit candidate or base source.

## Mathematical/source assessment

### Division and means

The separately bounded operands and residual `rho=a-c*y+floor(c/2) ∈ [0,c-1]` define exactly one Euclidean nearest quotient, with ties toward positive infinity. Public divisor/input/quotient bounds make the entire residual envelope smaller than the field modulus, not merely the honest result. Exact integer witness generation avoids f64 precision and sign-tie ambiguity. Compile-time scanning covers raw DivConst nodes as well as convenience wrappers; scalar shape[] reductions are supported.

Transformer mean inputs are masked before reduction; logical rather than padded counts are used. Input/reduction bounds lift sums non-circularly. All seven former duplicated inclusive tolerance blocks use the repaired primitive. Generic ReduceMean retains its column-major contract.

### Private dyadic bucket lifting through Q20

The new RMSBucket source is a fixed-shape private Boolean tensor `[len,32]`: one zero bucket,28 positive dyadic intervals, and3 forbidden tails. Actual range roots enforce bitness and row sum1; selected lower/upper interval roots force exactly the bucket containing m. For nonnegative m, tail intervals[-1,-1] cannot be selected. The selected bounds are computed by Einsum against immutable public tables. The source is committed, not public; shape, planes and table layout do not depend on the private selected bucket.

Let C=Qs^3. For the selected interval[L,H], code defines `R=floor(sqrt(C/L))+1`, `U=floor(sqrt(C*H))+1+ceil(H/2)`, with zero bounds in the zero bucket. It independently proves r≤R_selected, u=m*r≤U_selected and their global caps. The first product lifts globally; the second t=u*r lifts conditionally on the already-proved unique bucket. This is a legitimate conditional bound, not circular inference from the desired reciprocal bracket.

At SF20 I independently recomputed:

- Rmax=1,073,741,825;
- Umax=17,592,320,229,376;
- M*Rmax=288,230,375,346,405,375<2^59;
- Dmax=1,630,510,897,461,400,881<2^61.

The extra exact division `h=RD(m-3,4)=floor((m-1)/4)` is separately constrained by the live compile-time queue, including when introduced by another guard. The residuals

    (1-b)*(C-t+u-h-1) and (1-z)*(t+u+h-C)

are equivalent to the inclusive lower/strict upper reciprocal midpoint bracket. Their signed envelopes and range representatives cannot differ by a nonzero multiple of p=18,446,744,069,414,584,289. Zero indicators are independently defined, so m=0 forces r=0; positive m with r=0 is accepted exactly when m>4C. Negative, adjacent, wrong-bucket and wrong-flag values do not create another solution.

I inspected source registration/dispatch, statement tag40, advice commitment, source bit typing and actual NonNegative-root traversal. The queue reaches both newly generated quarter DivConst and RMSBucket nodes. Partial lookup chunks use `min(tcl,remaining_bits)` rather than rounding61 bits to64. The dedicated real chunk test checks61/62/63-bit reconstruction bounds.

### Local RMS output completeness

The new `min(old_global_bound,2^31-1)` bound is complete, not an unexplained restriction. For m≥1, rounded squares zi and rounded mean imply

    xi² < Qs*m*(3n/2+1/2).

With r≤sqrt(Qs³/m)+1/2, n≤2^20, Qs≤2^20 and M<2^28,

    |xi*r/Qs| < 2^20*1255+8192.

After rounding, integer cap1,315,971,073 suffices, strictly below2^31−1. For m=0 the output is zero. Independently bounded gain and output quotients keep the gain product below2^61 and the full rounding residual below2^62<p. Unrelated graph ScaleDown nodes are not certified by this argument.

### CNN mean layout repairs

LeNet's 2×2 average now uses four explicit-output-size SubSample2D gathers and exact division by4, rather than a mutable quarter convolution kernel. The authenticated row-major mask agrees with both CPU gather indexing and PIOP `cv.rs` mappings; the sum magnitude is at most4*(2^30−1). Tests cover original6×28×28 and16×10×10, non-square3×4×6, explicit batched inputs, signed rounding and poisoned padding. Folded3D tensors cannot reveal an unknown original logical batch count; documentation correctly treats represented folded batches as logical rather than claiming to recover omitted metadata.

**A substantive ResNet blocker was found during final review and repaired.** Calling column-major ReduceMean directly on row-major `[C,H,W]` mixed channels: the simple buffer `[0,1,2,3,100,101,102,103]` incorrectly gave[51,52], rather than per-channel means[2,102]. The final adapter reinterprets the same raw buffer as `[W,H,C]` (or `[W,H,C,B]`) before reducing axes0,1. This is a metadata view, not a transpose. The mask now aligns with the actual spatial/channel layout. The final tests include that exact regression,7×7 and non-square spatial shapes, explicit batch2 and poisoned padding. Existing FC/nonpower-batch full-model issues are not silently certified.

## Independent checks and inspected Rust evidence

- Independently executed `check_q20.py`: **127,930 groups passed**, all scales0–20, randomized/boundary means, sign/adjacent reciprocal mutants, wrong neighboring/zero/forbidden buckets, quarter identities, product/comparison budgets and local-output lift bounds. Its sqrt-plus-midpoint oracle differs from the production binary-search witness.
- Earlier independent division/reciprocal arithmetic script passed587,349 groups; it remains supporting arithmetic evidence, not native proof evidence.
- Inspected `typecheck4.log`: all-target Rust modified-build check completed. The isolated build omits CUDA build/link commands; this is not an unmodified production build.
- Inspected `cpu-scales-0-20-final.log`: **189 PASS test-body invocations** (nine CPU bodies at21 activation scales). The tests use real CPU operators and emitted-range predicates, including both CNN repairs and all dyadic boundary batches. I directly compared the final cpu4 test file to production: the only differences are `#[test] fn` becoming callable `pub fn`; bodies are unchanged.
- Inspected `cpu-partial-chunks-final.log`: original61/62/63-bit chunk test passed.
- Inspected production and PIOP build logs: production is blocked at missing nvcc; genuine PIOP linkage fails on unresolved CUDA symbols. **No PIOP replay execution or full accepted cryptographic proof is claimed.**

Two negative PIOP harness defects found in the earlier review remain repaired: mutants are installed after honest prover preparation with legal selectors, and negative reciprocal mutants use legal Boolean zero-flag hints rather than failing source typing before verification. New wrong-bucket replay mutants also remain legal one-hot sources. These tests are authored/typechecked, not runtime-validated while native linkage is unavailable.

## Remaining conditions and limits

1. Execute real positive/negative PIOP replay and authenticated full PCS proofs with genuine native linkage. CPU range evaluation is not verifier execution.
2. Scales21+ require a separately proved wider implementation; rejection is explicit.
3. Full RMS scalar-domain completeness requires declared32-plane request sources at every scale; default16 planes admit a narrower intersection. General DivConst bounds do not promise default-source completeness. No secret-dependent width enlargement was introduced.
4. Mean/reciprocal/padding/CNN-layout changes are substantive. Historical f64 outputs, checkpoint accuracy and timings cannot be reused; new accuracy comparisons and cost/source/range/mask census remain required.
5. InstanceNorm3D, other normalizers, attention causality, unrelated rescales, complete CNN/FC semantics and full cryptographic composition remain outside this targeted approval.
6. The intended DAG must be independently registered/pinned. The patch does not certify an arbitrarily modified post-compile DAG as an intended model; the special structural validator remains exp-specific.

No external publication, installation, GPU work or frozen-source modification was performed by this reviewer.

## Final evidence-only addendum

After source freeze I inspected `piop-replay-build-final.log` and `production-check-final.log` for the final replay4/source snapshot. They confirm, respectively, actual linker failure on unresolved native CUDA symbols and production build-script failure at missing `/usr/local/cuda/bin/nvcc`; both commands returned101 according to the executing worker. No replay test body executed. Final typecheck4 returned0, the189 CPU invocations returned0, and the original partial-chunk body returned0.

The updated `test-body-identity.json` now identifies nine unchanged production test bodies (production SHA-256 `d910a44c98def1520d5daf9febeecf72f615124f08f2e503c26ab45d66fb2505`, adapted SHA-256 `80388ef03858ddda404f5b4a6dc59e826af7c026270386dcedf2552a6fffaf06`). My direct source comparison already verified the visibility-only adaptation. The author-run patch replay record reports `git apply --check`, application, and all443 source-file hashes matching; generated cache files are excluded and retained separately in the workspace inventory. The16-file source-manifest hash remains unchanged. This addendum changes no source verdict or unexecuted-proof/accuracy limitation.


Export presentation note: private workspace path labels were normalized; mathematical and verification conclusions are unchanged. Original document SHA256s are recorded in `export-provenance.json` (relative to the evidence directory). Raw terminal logs and machine receipts remain byte-preserved.
