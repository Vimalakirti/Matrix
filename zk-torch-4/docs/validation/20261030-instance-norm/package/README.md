# InstanceNorm repair — exact finite statistics and parameter binding

## Ready-to-integrate source

- Candidate `/workspace/shared/morpheus-instance-norm-repair`, unchanged after independent review.
- Base `/workspace/shared/morpheus-mean-rms-q20`; apply `instance-norm.patch` ONLY after that frozen increment.
- Ten-file delta manifest SHA256 `789c2a5d0ae6c02e80fbff9f423cd8de27e1eb161f6ea08e444e4c4a4eb64b42`.
- Patch SHA256 `7ed94e85df0680d035482384faa339a1ce0c77b0f8c01caf365938488b650843`.
- Fresh `git apply --check`/application passed; all446 source hashes matched; all10 reviewer hashes matched; whitespace diff-check emitted no diagnostics. See `patch-apply-validation.txt`.
- No remote publication, merge, deployment or CUDA installation was performed. The parent owns publication/commit identity; inherited local HEAD is not a candidate identifier.

## What actually changed

The old InstanceNorm packed scale/offset advice did not prove means, variance, reciprocal, gamma or beta. The named U-Net builder now composes actual constrained arithmetic: rounded logical channel mean, centered per-element square rescale, rounded variance, exact public integer epsilon, existing exact reciprocal, quantized gamma*reciprocal scale, beta−scaled-mean offset, and final affine output. Every multiplication/rescale has explicit independent bounds and a full-residual no-wrap proof. CNN row-major indexing is repaired by a reversed metadata view; spatial/channel padding is excluded before statistics and zeroed at output. Gamma/beta are now claimed arithmetic model sources, not pruned advice-only inputs.

Raw legacy InstanceNorm nodes reject at builder compilation, Graph registration and statement encoding. Existing names/call sites stay intact. No opcode or new advice primitive was introduced. General constant division alone extends c≤2^24 with the same exact residual budget; transformer mean/RMS caps remain unchanged. InstanceNorm n≤2^24 algebraically includes128³/256³, without a full-volume execution claim.

Full profile: `zk-torch-4/docs/morpheus_instance_norm_profile.md` in candidate/patch. SF0..20 includesSF16/Q20; no clipping or source-width changes inferred from secret data. Epsilon1e-5 quantizes exactly to0/1/10 atSF10/16/20, no silent minimum-one substitution. This is deliberately changed finite arithmetic, not old f64 equivalence or trained-model compatibility. Logical |x| is just under8 atQ20 and32 atSF16 for small epsilon; gamma/beta bounds and source intersections are explicit. Public32-plane request encoding suffices for the full logical domain.

## Executed checks, not inflated proof claims

| Check | Result |
|---|---|
| Final original all-target Rust source/tests via disclosed native-build omission | PASS, `typecheck-final.log` |
| Eight original InstanceNorm CPU test bodies, everySF0..20 |168 PASS invocations, `cpu-scales-final.log` |
| Inherited14 exp/basic-graph bodies, everySF0..30 |434 PASS invocations, `exp-regression-scales-0-30.log` |
| Inherited nine mean/RMS/CNN-mean bodies, everySF0..20 |189 PASS invocations, `mean-rms-regression.log` |
| Original partial-chunk structural test |PASS61/62/63-bit reconstruction, `partial-chunks-regression.log` |
| Independent exact arithmetic audit |2,675,503 assertion groups passed, `independent/check_profile.out` |
| Independent final CPU rerunSF0/16/20 |24/24 invocations passed, independent final log |
| Independent mathematical/source review |No remaining scoped blocker, `independent/FINAL_INSTANCE_REVIEW.md` |
| Fresh patch apply and complete source identity |PASS446 hashes, `patch-apply-validation.txt` |
| Unmodified production all-target check |BLOCKED exit101 at missing `/usr/local/cuda/bin/nvcc`, `production-check.log` |
| Actual new PIOP replay runner build |BLOCKED exit101 unresolved CUDA/agl_mpiop symbols, `piop-replay-build.log`; ZERO replay bodies executed |
| Full authenticated proof, GPU execution, model accuracy, benchmark transfer |NOT ESTABLISHED |

The targeted CPU tests cover exact IEEE epsilon conversion/ties; all declared scales; non-square/non-power-of-two row-major tensors with poisoned spatial/channel padding; positive/zero epsilon and constant channels; domain extrema; genuine gamma/beta dependency and actual committed-source retention/width; neighboring statistic/reciprocal/affine quotient mutants; out-of-domain rejection without clipping; malformed public profiles; manual legacy DAG registration rejection; scalar divisors2^20,2^20+1,2^21,2^24 and rejection2^24+1.

CPU negative cases evaluate emitted range/rescale predicates. They are NOT a cryptographic verifier replay. The separately authored PIOP test supplies legal malicious selectors, updates claimed outputs consistently, preserves model source layout for parameter mutations, and identifies the actual epsilon literal. It typechecks but has not run due to native linkage. No stubs, ignored linker symbols or fake success implementations were used.

The independent scoped review must NOT be counted among the five complete independent paper acceptance reviews. Open requirements still include whole-model lowering/census, numerical accuracy with fixed checkpoints, full proof/GPU/runtime evidence, and other normalization/causality repairs.

The final candidate therefore has791 author-run targeted CPU test invocations (168 new+189 mean/RMS regressions+434 exp/basic-graph regressions), plus the partial-chunk structural test; these are not a full cargo-test suite. The independent24 rerun and portable rerun are separate repetitions, not extra distinct cases.

## Build-only adaptations and reproduction

Historical checks used official Rust/Cargo1.99.1 installed under `/workspace/shared/morpheus-toolchain`; the portable recipe uses the caller’s already available Rust/Cargo on PATH. Fresh validation copies omit CUDA build/link commands and static FFI linking, expose unchanged test bodies, and add runners. FFI declarations remain: any reachable GPU dependency correctly fails linkage. `cpu2-`, `typecheck2-`, `replay-` and `regression-adaptations.{json,patch}` record all changes. `test-body-identity.json` checks all eight production bodies are identical except test visibility. The successful CPU binary is `01848376604dbbe3801a7f524a5a5e62b4d34c3994ecc8669c99349f304c2166`, with no unresolved CUDA/agl_/mpiop symbols; see `cpu-binary-audit.log`.

```sh
# Use an already available Rust/Cargo on PATH; no installation is performed.
# SOURCE is the reconstructed/pinned source directory. Start in this bundle.
SOURCE=/path/to/pinned-source
WORK=$(mktemp -d)
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR="$WORK/target"
python prepare_harness.py "$SOURCE" "$WORK/cpu" cpu-instance
cargo build --offline --locked --manifest-path "$WORK/cpu/zk-torch-4/Cargo.toml" --bin cpu_instance_norm
for sf in $(seq 0 20); do ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_instance_norm" || exit $?; done
# A separate fresh typecheck copy: mode typecheck; cargo check --locked --all-targets.
# Inherited mean/RMS and partial chunks: mode cpu-normalization, bins
# cpu_normalization and cpu_partial_chunks. In that same copy, bins
# cpu_basicblock_exp and cpu_graph_exp run the14 original tests atSF0..30.
# --offline requires dependencies already in Cargo's cache; no registry/vendor
# completeness is implied by this validation bundle.
```

An intermediate rebuild hit ENOSPC; only author/root-owned regenerable incremental caches were removed, then builds used CARGO_INCREMENTAL=0. Intermediate compile errors and malformed negative-test construction were repaired and final logs retained separately; they are not hidden or counted as final passes. Final source remains frozen.

## Portable evidence export

`export-manifest.json` lists exact path/bytes/SHA256 for every packaged text file (excluding itself). `python verify_export.py --source SOURCE` checks both bundle and all446 source hashes. `python reconstruct_source.py EXACT_Q20_BASE NEW_DEST` applies the patch in a fresh copy and verifies every source hash. Intermediate attempts remain in the workspace evidence directory; the export contains final logs only. The fresh packaged CPU wrapper is also exercised separately atSF0/16/20; see `portable-scales-0-16-20.log`.

The final shared-builder/Graph regression requested before publication ran all14 inherited exp/basic-graph bodies at everySF0..30:434/434 passed. `exp-regression-body-identity.json` verifies complete bodies unchanged; `exp-regression-receipt.json` checks every body passed exactly31 times. No candidate source changed. Native CUDA/PIOP blockers remain unchanged.
