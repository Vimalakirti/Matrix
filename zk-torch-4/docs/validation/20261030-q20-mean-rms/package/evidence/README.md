# Mean/RMS repair — frozen Q0..Q20 candidate and honest validation

## Deliverables and identity

- Source: `<candidate>`.
- Base: frozen `<exp-base>` (ten-file exp candidate).
  Base and `<earlier-private-input-base>` were not modified.
- Apply `mean-rms.patch` to that base, not to an earlier mixed checkout.
- Final16-file delta manifest SHA256:
  `60b90a1c996cdb0c86becb35005563e289b0e8c2abfbf68a331d530e3de70e3a`.
- Patch SHA256:
  `6c36b21a613c705de390ad6bc2a54f55cea15775e5de6d57bbabb43e529e2186`.
- `full-source-manifest.json` lists443 required source files, excluding5
  inherited generated Python cache files. `complete-workspace-inventory.json`
  retains the original448-file inventory; caches are NOT publication requirements.
- Fresh patch application and all443 SHA256 checks passed; see
  `patch-apply-validation.txt`. `git diff --check` passed.

The local repository HEAD is inherited and is NOT claimed to identify this
uncommitted candidate. Parent owns any later publication and commit receipt.
No external PR write, merge, deployment, installation, or paid compute occurred.

## Substantive repair

1. Every DivConst gets exact integer Euclidean-rounding constraints at compile
   time, including raw-node users. Half ties go toward positive infinity.
2. All named transformer mean/RMS paths and generic ReduceMean use exact means,
   explicit public bounds, and logical padding rules. RMS epsilon stays exactly0;
   m0 deterministically returns r0. The weak optional tolerance relation is gone.
3. Private fixed-shape32-position one-hot dyadic selection independently bounds
   m,r,u=m*r before t=u*r. Exact quarter brackets define the reciprocal through
   activation scales0..20, includingSF16/Q20, with non-circular integer lifting.
   Internal exp Qe remains separate. Source layouts do not reveal private buckets.
4. Local RMS square/output rescale quotients and gain products are bounded.
5. LeNet average pooling uses four explicit-sized row-major gathers and exact
   division. ResNet GAP now reinterprets row-major storage before column-major
   reduction, repairing actual channel mixing ([51,52] ->[2,102] in its minimal
   regression). These are real numerical changes, not retroactive validation of
   old f64 outputs, accuracy or performance.

Exact equations, zero/tie policy, source-width rules, padding conventions,
no-wrap proof, bounds and changed-output examples:
`zk-torch-4/docs/morpheus_mean_rms_profile.md` in the candidate.
Full bounded RMS completeness uses public `REQ_PLANES_LOG=5` (32 signed planes).
Default16 planes remains a narrower source-domain intersection; no secret-derived
width choice or silent enlargement was introduced.

## Executed validation, carefully scoped

Original environment: official Rust/Cargo 1.99.1. Use a compatible Rust toolchain already installed locally; no installation is included or required by this export wrapper.

| Check | Actual result |
|---|---|
| Original all-target Rust source/tests, isolated CUDA-build-script omission only | PASS exit0, `typecheck4.log` |
| Nine original CPU graph test bodies at every scale0..20 | 189 PASS invocations, exit0, `cpu-scales-0-20-final.log` |
| Original PIOP partial-chunk structure test at widths61/62/63 | PASS, `cpu-partial-chunks-final.log` |
| Inherited14 exp/basic-graph bodies, all scales0..30 on FINAL candidate |434 PASS invocations, exit0, `exp-regression-scales-0-30.log` |
| Independent Q20 arithmetic/source review |127,930 arithmetic groups; no remaining scoped blocker |
| Unmodified production `cargo check --locked --all-targets` |BLOCKED exit101: missing `/usr/local/cuda/bin/nvcc`, `production-check-final.log` |
| Real PIOP prover/verifier replay runner build |BLOCKED exit101: unresolved actual agl_mpiop/CUDA symbols, `piop-replay-build-final.log`; ZERO replay bodies executed |
| Full authenticated PCS proof, GPU runtime, model accuracy, timings |NOT ESTABLISHED |

CPU cases include signed/large mean half ties and neighboring advice, reciprocal
zero/sign/neighbor/over-bound mutations, every dyadic boundary atSF16/SF20,
wrong legal one-hot/forbidden/two-hot/nonbinary bucket hints, all six shared
transformer norm paths at widths3/768/1024/4096/8192, LeNet original/non-square/
batched pooling layouts, and ResNet row-major/spatial-padding regressions.
Together these are624 targeted CPU test-body invocations, NOT ordinary full
`cargo test` suite execution. The189 normalization invocations include630 small named norm graphs,126 LeNet pooling cases,
and84 ResNet mean cases; they are NOT full model inference runs.

CPU negative tests directly evaluate emitted range predicates after running real
CPU operators with changed advice. They are NOT cryptographic verifier acceptance
or rejection receipts. Separate genuine PIOP mutation tests are authored and
all-target typechecked but remain unexecuted due to the actual linkage blocker.

## Validation-copy adaptations and reproduction

`prepare_harness.py` wraps the preserved `prepare_base_harness.py`. Fresh copies
only: skip CUDA build/link commands, remove the static FFI library link attribute
for CPU runners, expose original test modules/functions, and add small runners.
Extern declarations stay intact. NO GPU stubs, fake successful proofs, ignored
unresolved symbols, source arithmetic edits, or hidden excluded test targets.
`morpheus-mean-rms-q20-{typecheck4,cpu4,replay4}-adaptations.{json,patch}` records
every difference. `test-body-identity.json` verifies all nine CPU bodies unchanged
except test visibility. `cpu-binary-audit-final.log` records binary hashes,
no unresolved agl/CUDA symbols, and dynamic dependencies for successful binaries.

```sh
# Set CANDIDATE to the reconstructed full source tree, not export/source.
CANDIDATE="$(pwd)/reproduce/candidate"
WORK="$(pwd)/reproduce"
python3 evidence/prepare_harness.py "$CANDIDATE" "$WORK/cpu" cpu-normalization
export CARGO_TARGET_DIR="$WORK/target"
cargo build --locked --manifest-path "$WORK/cpu/zk-torch-4/Cargo.toml" \
  --bin cpu_normalization --bin cpu_partial_chunks
for sf in $(seq 0 20); do
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_normalization" || exit $?
done
"$CARGO_TARGET_DIR/debug/cpu_partial_chunks"
# Separate full-source copy for all-target adapted typecheck:
python3 evidence/prepare_harness.py "$CANDIDATE" "$WORK/typecheck" typecheck
cargo check --locked --all-targets \
  --manifest-path "$WORK/typecheck/zk-torch-4/Cargo.toml"
```

Expected caught panics and range warnings from negative cases are retained in
logs, not suppressed or counted as successes independently of assertions.
Earlier SF15/intermediate Q20 attempts remain separate records. Independent
review found and prompted repairs to malformed PIOP test construction, scalar
ReduceMean shape handling, and the real ResNet layout defect before finalization.

## Remaining gates

This is a targeted numerical increment, not one of the five complete independent
paper acceptance reviews. InstanceNorm3D/UNet statistics, SoftmaxConst/SigmoidConst,
unrelated ScaleDowns, causal attention, complete workload lowering/source/mask
census, calibrated model accuracy and all full-proof/performance claims remain
separate. Scales21..30 need a wider arithmetic profile; this does not exclude the
reportedSF16 or requestedQ20. Folded3D CNN tensors have no recoverable original
logical batch count: represented folded batches are logical; explicit4D batch
padding is masked. Full FC/non-power-of-two batched inference is not certified.

Dependent execution requires an authorized suitable CUDA build/link environment
(and real hardware for GPU tests). No installation was attempted or bypassed.


Export presentation note: private workspace path labels were normalized; mathematical and verification conclusions are unchanged. Original document SHA256s are recorded in `export-provenance.json` (relative to the evidence directory). Raw terminal logs and machine receipts remain byte-preserved.
