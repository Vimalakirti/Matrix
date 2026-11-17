# Scoped CNN layout repair — frozen local deliverables

No remote publication, installation, GPU stub or proof-success substitution.
The inherited Git HEAD is NOT the identity of these uncommitted source snapshots.

## Source and portable patches

- Q20-based source: `Q20-based source`.
  Apply `q20-layout.patch` to the frozen Q20 base named in `q20-base-manifest.json`.
  13 changed/new files;446 source files excluding inherited Git metadata/caches.
- Preferred finite-normalization integration:
  `reconstructed joint source`.
  Apply `joint-layout.patch` to `morpheus-selected-normalization-joint`, checked
  against `joint-base-manifest.json`.14 changed/new files;454 source files.
- Both patches already include the batched-convolution worker's builder-only
  explicit4D geometry/view delta. They do NOT include its cv.rs PIOP extension.
  Avoid applying that same builder delta twice during integration.
- Every file hash and patch SHA256 is in the manifests / `export-receipt.log`.
  Fresh application and exact source-hash roundtrip passed for both bases.
  `git diff --check` passed. Old inventory including inherited `.git` is retained
  as historical inventory, explicitly not a publication requirement.

Production specification, formulas, input conventions, exact public source-cost
census and explicit remaining gates:
`zk-torch-4/docs/morpheus_cnn_layout_profile.md`.

## Executed checks

| Check | Actual outcome |
|---|---|
| Final original CPU test bodies, every SF0..20 |210 PASS body invocations:10×21 |
| Inherited selected-normalization joint regressions on same frozen source |978 PASS:434 exp,186 selected normalization,189 mean/RMS,168 InstanceNorm,1 partial-chunk |
| Actual B3 pad→plain/strided/dilated Conv→distinct bias→maxpool chains |PASS within those bodies, including original Conv2D::run/FlattenKernel::run |
| Nonzero LeNet flatten+400→120→84→10 FC; ResNet GAP/FC B1/3/5 |PASS vs independent logical references, poison padding |
| Full LeNet graph all-zero CPU smoke; representative ResNet head census |PASS, not accuracy validation |
| Selected finite-sigmoid CNN SiLU joint integration |PASS vs exact dyadic integer reference at every SF0..20 |
| Original source + all targets/tests, isolated CUDA-build omission only |PASS, final-all-target-typecheck.log |
| Unmodified production cargo check --locked --all-targets |BLOCKED101: /usr/local/cuda/bin/nvcc absent |
| Actual PIOP replay runner build |BLOCKED101: real agl/CUDA externs unresolved; ZERO replay bodies executed |
| Full authenticated PCS/GPU proof, full-model accuracy/timings |NOT ESTABLISHED |

`executed-test-summary.json` lists all210 CNN bodies/log hashes;
`inherited-executed-test-summary.json` lists the additional978 inherited bodies.
Total1188 targeted CPU invocations passed on the exact frozen source. Negative logical
shape tests deliberately catch panics; the expected stderr is retained. Old
intermediate census runs failed because template scale metadata was used before
forward execution; the final census runs actual forwards first. Logs remain.
No failed/intermediate run is counted as a pass.

The10 final CPU bodies include9 original CNN layout/census tests plus1 joint-only
finite activation test. `test-body-identity.json` checks unchanged test bodies
except visibility. `final-cpu-adaptations.{json,patch}` records all6 differences:
CUDA build/link omission, FFI link attribute omission, test visibility, runner.
Extern declarations remain intact. No arithmetic or mapping source is rewritten
for the harness. `cpu-binary-audit.log` includes binary SHA256, unresolved-symbol
listing and dynamic libraries (no unresolved agl/CUDA reference in that successful
CPU binary). The successful binaries do not execute `Dag::run`'s GPU dispatch;
they execute the original production BasicBlock CPU operators over the real DAG.

## Source-cost result and limitations

Public literals ARE committed sources and are bound to the public statement.
Flatten's16384 entries and all0/1 masks each cost one raw bit-source coordinate.
Single-image LeNet adds29712 public bit coordinates vs Q20. Complete PUBLIC
literal subsets in executed representative graphs cost1642512 raw coordinates
(LeNet) and36392960 (B3/C2048/7x7/O1000 head); each subset packs into one ternary
source. `public-source-census-sf20.log` lists all46 literal shapes/rows/planes.
These counts DO NOT include the complete model/request/advice/selector/mask-bank/
carry lifetime. Whole-workload parameter compatibility remains UNVERIFIED.

## Independent targeted review

Read `independent-review/REPORT.md` and its
reproducible Python evidence. Independent checks passed158 flatten shapes,
428 non-Boolean MLE identities,36 bias shapes and144 pad/pool geometry cases.
The reviewer found an equal-padded-size/logical-shape mismatch; the source now
rejects it, and CPU negative tests confirm the rejection. It also surfaced the
pre-existing logical maxpool-attainment proof gap and the public-commitment cost.

A second independent reviewer checked the final explicit output masks and all
three imported Conv builder geometry changes at builder SHA256
`c38fbd735363f1a15d05fe22fb2403bcac6f07aac59d9bf0bda509f9dfa7fc5e`.
Its36-case/36552-assertion index/source checks passed; evidence:
`independent-wrapper-review/check_wrapper_integration.py`
and adjacent results. This closes the narrow frontend/view recheck, NOT logical
maxpool achievability, actual PIOP/native execution or a full independent paper review.

## Remaining substantive gates

- Logical maxpool constraints still establish dominance, not window attainment;
  all-zero window / claimed1 remains a real pre-existing proof counterexample.
  Final output masks here repair padding only. Separate maxpool repair underway.
- Actual tensor-batched Conv2D PIOP still needs the separate cv.rs extension and
  real replay. Deferred batching of multiple proof requests is a distinct axis.
- This repair does not certify arbitrary FC integer rescale soundness, VGG's
  separate flatten/head logic, all workload lowering, models, accuracy or costs.
- The finite SiLU integration requires the selected normalization increment;
  the Q20 base's old SigmoidConst is not made sound by a layout wrapper.
- Poison±2^40/±2^60 CPU mask tests do not fit a32-plane private-source profile;
  authenticated admission would need a PUBLIC64-plane profile. No source-width
  setting is chosen from private tensor values.

## Reproduce the honest CPU harness

```sh
# Rust/Cargo 1.99.1 must be installed and available on PATH.
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0
SOURCE=/absolute/path/to/reconstructed-source
WORK=/absolute/path/to/new-validation-workspace
BUNDLE=/absolute/path/to/this-evidence-bundle
export CARGO_TARGET_DIR="$WORK-target"
python "$BUNDLE/prepare_harness.py" "$SOURCE" "$WORK" cpu
cargo build --locked --manifest-path "$WORK/zk-torch-4/Cargo.toml" --bin cpu_cnn_cpu
for sf in $(seq 0 20); do
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_cnn_cpu" || exit $?
done
```

Use a separate new `typecheck` copy and `cargo check --locked --all-targets` for
all-target type checking. Mode `replay` intentionally retains real PIOP external
symbols: without native libraries the runner must fail to link, not pretend pass.


## Portable bundle entry points

This directory is self-contained and hash-inventoried. Historical logs
retain actual local execution paths, but executable recipes require only an
existing Rust/Cargo installation on PATH and user-supplied source/work paths.

```sh
python verify_bundle.py --source /absolute/path/to/frozen-joint-source
python reconstruct.py /absolute/path/to/frozen-selected-normalization-base /absolute/path/to/NEW-source
python run_regressions.py /absolute/path/to/NEW-source /absolute/path/to/NEW-external-validation
python run_independent.py /absolute/path/to/NEW-source /absolute/path/to/NEW-independent-validation
```

Both reconstruction and harness creation reject destinations inside the resolved
source (including symlink aliases). Reconstruction also rejects existing or
bundle-internal destinations, verifies exact base bytes, copies only those checked
manifest files, applies the patch and verifies all454 final source hashes.
Original independent reports/scripts are preserved; portable reruns change only
the old fixed source path in an EXTERNAL copied script and record that adaptation.

Historical raw reports/logs retain original provenance paths and source identities. The frozen production profile document also retains its base-path provenance; no portable script selects that path. User-facing recipes use supplied source/work arguments. All historical execution logs are in `logs/`; pre-freeze manifests/deltas are in `history/`, not canonical source identity.


Final portability checks passed:14 resolved source/bundle/existing/symlink
containment guards reject before copying; source/base hashes unchanged. Fresh
portable reconstruction verified454 exact source files. Fresh portable CPU
harnesses match all289/294 Rust files in the actually executed CNN/inherited
harnesses. Independent scripts reran successfully from external copies with the
parameterized source argument. Receipts are `portable-guard-verification.json`,
`portable-harness-code-identity.json`, and `portable-independent-rerun/`.
