# Exp, padding and literal-binding repair: scoped validation

Frozen source parent: `f192323f0725fed17a27dee21442fcba25460d2f`.
The enclosing commit adds exactly the ten files listed in `source-manifest.json`
to that parent plus this additive evidence directory. The ten-file manifest means
changed paths, not ten new files. The Rust bundle verifies all 441 non-cache source
files (437 parent blobs plus four added paths), including those ten changed files.
This evidence directory is additional documentation, not part of that tested source
inventory. No other in-progress numerical changes are included.

## Results and limits

- Real Rust CPU execution: **434 passing invocations**, consisting of **14 distinct
  original test bodies at each of 31 activation scales (0–30)**. This is not 434
  distinct tests and not a normal full `cargo test` run.
- Complete `cargo check --locked --all-targets` passed only in a disposable copy
  with the CUDA build-script compilation/link directives omitted. This includes
  Rust binaries and test code; it is **not** production build/link verification.
- The unmodified production check failed with exit 101 at missing
  `/usr/local/cuda/bin/nvcc`.
- **Five original PIOP replay test bodies never executed**: real GPU FFI symbols
  prevented the standalone runner from linking. No GPU implementation was stubbed.
- Independent exhaustive Python scalar domains at SF10 (181405 inputs) and SF15
  (5803172 inputs), all-scale/width endpoints, interval/tie and padding checks pass.
  `scalar-test-output.log` also records the earlier 24 parser, seven semantic,
  two unit-reference and 10048 UBSan host-division checks. These are not Rust
  verifier or CUDA execution.

`rust/REPORT.md` identifies exact toolchain, adaptations, failures and successful
portable replay. `rust/SHA256SUMS` authenticates the bundled evidence bytes, not a
third-party attestation. `source-review.md` is a historical independent targeted
source review with a subsequent borrow-fix addendum; its pre-Rust statements are
superseded only by the narrow Rust results above. It is **not** one of the five
whole-paper independent reviews or an acceptance/merge approval.

## Reproduce without installing anything

From the repository root, with an existing Rust/Cargo toolchain and locked registry
dependencies available:

```sh
EVIDENCE=zk-torch-4/docs/validation/20261030-exp/rust
(cd "$EVIDENCE" && sha256sum -c SHA256SUMS)
python3 "$EVIDENCE/verify_sources.py" "$PWD"
bash "$EVIDENCE/run_checks.sh" "$PWD" /path/to/new-validation-work
python3 zk-torch-4/scripts/bench/morpheus/test_exp_profile.py
```

The new work directory must not exist and must be outside the checkout. The wrapper
copies the source, changes only disclosed validation-only build/test access gates,
and keeps the original test bodies and assertions. No CUDA toolkit is installed or
downloaded by these scripts. The successful portable replay and exact copy changes
are retained under `rust/logs/` and `rust/*-adaptations.{json,patch}`.

## Deliberate semantic changes and remaining obligations

Default exp selector width is eight, with explicit widths 1–8 supported. Inputs
outside the finite nonpositive domain fail rather than silently clamp. Immutable
public literal values and helper guards are bound by canonical statement codec 2;
wire protocol remains 3. Logical exp values in the unit-corrected eight-bit domain
are preserved by the scalar comparison, but padding is explicitly zeroed before
and after exp. Downstream reductions and historical four-bit clamping results can
change. See `../../morpheus_exp_profile.md` for exact bounds and source-level costs.

No full authenticated PCS/proof acceptance, CUDA kernel, model accuracy,
performance, normalization/causal correctness or general compiler/no-wrap claim is
established. Actual full-verifier tampering tests and fresh whole-model measurement
remain open. The draft must not be treated as ready to merge.
