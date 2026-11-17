# Independent targeted review — bounded InstanceNorm3D

## Scoped verdict and source identity

**No remaining blocking mathematical or source-wiring defect found in the declared bounded InstanceNorm increment.** This is a targeted construction review, not a complete paper acceptance review, whole-U-Net certification, runtime/accuracy result, or cryptographic proof-execution receipt.

Reviewed candidate: `/workspace/shared/morpheus-instance-norm-repair`, based on frozen `/workspace/shared/morpheus-mean-rms-q20`. The ten-file delta is pinned by `reviewed-source-manifest.json`, SHA-256:

`53a7767e95c6ee627670ad784773e30dbc740c104fad614980f8070051393c5f`.

I did not modify candidate or base source. The no-index whitespace check emitted no diagnostics (exit1 is the expected no-index result for differing trees, not a clean-tree exit0 claim).

## Construction assessment

The original helper's packed affine tensor was unconstrained; its output equation alone did not establish statistics or gamma/beta identity. The supported builder now removes that helper entirely and constructs the specified finite program from real arithmetic and lookup roots. Legacy InstanceNorm nodes fail at builder compilation, Graph registration, and statement encoding, including manually assembled legacy DAGs.

The finite semantics are explicit and consistent with the source: rounded logical mean; centered per-element squared/rescaled values; rounded logical variance; exact public integer epsilon; the inherited exact rounded reciprocal, including its zero extension; then quantized affine scale, scaled mean, offset, and output in the documented order. This is not real-valued population variance, not one-stage square-sum rounding, and not separately rounded centered output. No clipping is used.

The IEEE epsilon conversion uses exact mantissa/exponent arithmetic. It rejects nonfinite/negative/out-of-domain values, permits zero (including negative zero), and deliberately allows positive epsilon to quantize to zero. At SF10/SF16/SF20 binary f64 `1e-5` becomes0/1/10. Equivalent epsilon arguments intentionally have the same canonical integer semantics.

CNN row-major `[C,D,H,W]` is correctly reinterpreted as column-major `[W,H,D,C]` without transposition. Masks precede both statistics reductions and follow centering and the output; parameter/statistic channel masks are present. Gamma and beta have ordinary arithmetic consumers and remain claimed model sources rather than advice-only pruned inputs.

## Independent lifting and completeness checks

For Q=2^s, s0..20, K=2^28−1, e0..K, G=2^24−1, the stated public bounds X,R,A,T are independent of observed witness values. Rounded means remain within the input interval. Centered squares obey d²≤4X²<2^48, rounded squares and variance lie in[0,K−e], and the statistic lies in[0,K]. With n≤2^24, |sum x|<2^47 and sum z<2^52, below the exact DivConst input bound2^54−1.

For every rescale the source now checks an envelope at least as large as `P+Q*Y+floor(Q/2)+(Q−1)`, bounding the entire residual rather than assuming the honest product cannot wrap. Independently recomputed maxima across scales:

- affine raw product bound:144115162440270832<2^57;
- full residual envelope:288230324883162095<2^58<p;
- scaled-input quotient bound T:274844336130<2^38.

Thus bounded quotients/selectors cannot counterfeit an integer rounding relation by a nonzero field-modulus multiple. Interval-root representatives also lift non-circularly because their combined range is below p. The reciprocal reuses the previously reviewed, separately constrained private-bucket construction; it is not replaced by a floating hint or inferred from an accepted final bracket.

Extending only generic DivConst's c cap to2^24 is valid: with B=2^54−1 and Y=floor(B/c)+1, the complete residual envelope `B+cY+floor(c/2)+(c−1)<2^56<p`. Other mean/RMS row caps and the existing RMS output lemma are unchanged. Spatial counts128³/256³ are algebraically covered; neither was executed as a full-volume model.

The committed request sources are bounded input, mean/variance advice, reciprocal, quarter-division advice, and Boolean zero/bucket flags. Large affine ScaleDown quotients are derived edges, not new committed dense advice. Public req_planes_log5 (32 signed planes) suffices for the stated logical domain; raw ignored padding must still fit its declared source type. No secret-adaptive width change occurs. The CPU tests inspect actual committed-source widths and gamma/beta retention; source registration and PIOP root/walk code agree with the argument.

## Independent checks and inspected evidence

- Executed `check_profile.py`: **2,675,503 arithmetic assertion groups passed**. Coverage includes all scales0..20, public-bound endpoints/random epsilon, signed/extreme/constant/random rows, exact quotient ±1 mutants, reciprocal sign/neighbor mutants, IEEE/Fraction epsilon comparisons,10k random extended divisors, and symbolic128³/256³ histograms. The reciprocal oracle uses integer square root plus midpoint comparison, not the production binary-search algorithm. This script is arithmetic evidence, not Rust or PIOP execution.
- Independently reran the final CPU binary at scales0/16/20: **24/24 unchanged test-body invocations passed**, exit0. Binary SHA256 `01848376604dbbe3801a7f524a5a5e62b4d34c3994ecc8669c99349f304c2166`; retained log `cpu-final-independent-scales-0-16-20.log`.
- Direct comparison confirms all eight CPU test bodies are identical to production except visibility/test attributes; arithmetic implementation, PIOP registration and statement code match the final CPU2 copy. Its binary has no unresolved CUDA/agl_/mpiop symbols; dynamic dependencies are ordinary libc/libm/libgcc.
- Inspected author `cpu-scales-final.log`:168 passes (eight bodies ×21 scales), and `typecheck-final.log`: all-target modified-build check finished successfully. Adaptations omit native build/link and expose test bodies; they are not an unmodified production build.
- Authored genuine PIOP replay covers changed mean/variance/reciprocal/affine quotients, model parameters and the actual epsilon literal with legal selectors. It is typechecked, not executed: native linkage remains blocked on actual CUDA/agl_mpiop symbols. No successful GPU implementation or proof was stubbed.

During review I identified and the author repaired: a too-small written residual guard term (changed to conservative2Q), malformed negative tests that previously panicked on a source-length mismatch before the intended rejection, ambiguous selection of a unit literal rather than the actual epsilon, and parameter-mutant choices that could change the source layout. I also requested defense-in-depth legacy registration rejection, now present. Earlier intermediate logs are not the final evidence.

## Remaining boundaries

This establishes a precisely restricted local numerical relation, not full-model lowering correctness. The profile allows input magnitudes just below8 atQ20 and32 atSF16 for small epsilon; checkpoint/activation compatibility is not presumed. Historical timings or accuracy cannot be transferred. Same-checkpoint numerical comparisons, task accuracy, whole-model census, full authenticated PCS proofs, native production build/GPU execution, and full-volume memory/performance remain unestablished. This review must not be counted as one of the five complete independent acceptance reviews.
