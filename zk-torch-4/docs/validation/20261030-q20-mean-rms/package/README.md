# Exact Q0–Q20 mean/RMS repair: portable source and evidence

This package is a **16-file source overlay**, not a standalone repository. It
applies to the preceding exp repair at commit
`e2d82ccfcc61bad6311fab053a146b7e68f7064e`
(tree `9b61e33bcf68774ae7c2306aa517650280add661`).
No publication or merge of this Q20 overlay is claimed by this package.

## Identity and contents

| Item | Identity / purpose |
|---|---|
| `source/` | The 16 unchanged final source files, with repository-relative paths |
| `evidence/source-manifest.json` | SHA256 `60b90a1c996cdb0c86becb35005563e289b0e8c2abfbf68a331d530e3de70e3a`; 16 delta files plus expected preceding-base hashes |
| `evidence/full-source-manifest.json` | SHA256 `9119fd68440756e222fdbd8f1ff6814f9d1b51a9e74d241c1e160ea580c81ccb`; **443 required source files**, no generated caches |
| `evidence/complete-workspace-inventory.json` | Original 448-file inventory retained, including five historical Python caches; caches are **not required publication files** |
| `evidence/mean-rms.patch` | SHA256 `6c36b21a613c705de390ad6bc2a54f55cea15775e5de6d57bbabb43e529e2186` |
| `evidence/independent/` | Scoped independent review and integer-check scripts; not a complete S&P review |
| `evidence/recovery-20261030/` | Fresh integrity, patch-replay and CPU rerun receipts |
| `export-manifest.json` | SHA256 identity and byte length of every other package file |

The source overlay and source manifests are byte-for-byte frozen. Portable
editorial copies replace private workspace path labels in prose. Raw logs and
historical machine receipts retain their original bytes, including historical
host paths; `evidence/export-provenance.json` records this distinction.

## What was actually executed

| Check | Recorded result |
|---|---|
| Final all-target Rust check with isolated CUDA build-script omission | PASS; `evidence/typecheck4.log` |
| Nine original normalization CPU bodies at every activation SF0–20 | 189 PASS invocations; `evidence/cpu-scales-0-20-final.log` |
| Original 61/62/63-bit partial-chunk structure test | PASS; `evidence/cpu-partial-chunks-final.log` |
| Fourteen inherited exp/basic-graph CPU bodies at SF0–30 | 434 PASS invocations; `evidence/exp-regression-scales-0-30.log` |
| Independent Q20 integer checker | 127,930 groups PASS; `evidence/independent/check-q20-output.txt` |
| Recovery rerun | Another 189 normalization invocations, original chunk test, and 127,930 integer groups PASS; recovery receipts |
| Unmodified production build | **BLOCKED:** missing native CUDA compiler; `evidence/production-check-final.log` |
| Genuine PIOP prover/verifier runner build | **BLOCKED:** unresolved actual CUDA/native symbols; `evidence/piop-replay-build-final.log`; **zero replay bodies executed** |
| Authenticated full PCS proof, GPU runtime, full-model accuracy and timings | **NOT ESTABLISHED** |

The original targeted CPU total is **624 test-body invocations**, not a full
`cargo test` pass and not 624 different tests. Recovery reruns are repeated
validation, not new distinct test coverage. Earlier numbered logs are retained
as historical attempts and are not added to final totals.

All CPU runners execute actual CPU operators and unchanged selected production
test bodies. Only CUDA build/link setup and test visibility/runner entry points
are adapted. No GPU success stub, ignored unresolved symbol, or simulated
cryptographic acceptance is supplied. Negative CPU cases evaluate emitted range
predicates; they do not establish cryptographic verifier rejection.

## Portable reproduction

Prerequisites: Python 3, Git, and an already installed compatible Rust/Cargo
(original environment used Rust/Cargo 1.99.1). Cargo dependencies must be
available normally or cached. Nothing here installs CUDA or accepts any license.
Run these commands from the package directory. The `reproduce` directory and
its children must be new; source and base trees are never overwritten.

```sh
set -eu
# First check the package's file identities.
python3 evidence/verify_export.py

# Obtain the preceding exp commit through your normal authorized repository
# workflow, then set BASE to that full checkout. The script checks old hashes.
BASE=/path/to/exp-base-checkout
WORK="$PWD/reproduce"
mkdir -p "$WORK"
python3 evidence/reconstruct.py --base "$BASE" --output "$WORK/candidate"
python3 evidence/verify_export.py --source "$WORK/candidate"
python3 evidence/independent/check_q20.py

# Isolated CPU validation copy: full candidate source remains untouched.
python3 evidence/prepare_harness.py "$WORK/candidate" "$WORK/cpu" cpu-normalization
export CARGO_TARGET_DIR="$WORK/target"
cargo build --locked --manifest-path "$WORK/cpu/zk-torch-4/Cargo.toml" \
  --bin cpu_normalization --bin cpu_partial_chunks \
  --bin cpu_basicblock_exp --bin cpu_graph_exp
for sf in $(seq 0 20); do
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_normalization"
done
"$CARGO_TARGET_DIR/debug/cpu_partial_chunks"
for sf in $(seq 0 30); do
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_basicblock_exp"
  ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/cpu_graph_exp"
done
python3 evidence/record_adaptations.py "$WORK/candidate" "$WORK/cpu" "$WORK/cpu-receipt"

# Separate adaptation containing ONLY the CUDA-build omission for typechecking.
python3 evidence/prepare_harness.py "$WORK/candidate" "$WORK/typecheck" typecheck
cargo check --locked --all-targets \
  --manifest-path "$WORK/typecheck/zk-torch-4/Cargo.toml"
```

To attempt the **unmodified production** check in a separately authorized
suitable native environment, run:

```sh
cargo check --locked --all-targets \
  --manifest-path "$WORK/candidate/zk-torch-4/Cargo.toml"
```

The authored genuine PIOP replay tests remain in
`zk-torch-4/src/morpheus/piop/exp_tests.rs`. The separate build-blocker
reproduction below intentionally supplies **no native implementation or stub**;
it will fail linking if retained CUDA symbols cannot actually resolve. The
runner may only be executed after a genuine successful native link:

```sh
python3 evidence/prepare_harness.py "$WORK/candidate" "$WORK/replay" cpu-replay
cargo build --locked --manifest-path "$WORK/replay/zk-torch-4/Cargo.toml" \
  --bin cpu_exp_replay
# Only after that command truly succeeds:
"$CARGO_TARGET_DIR/debug/cpu_exp_replay"
```

## Numerical and review boundaries

The source profile defines actual exact mean half-tie behavior, zero/sign and
private-bucket RMS constraints, bounded no-wrap arithmetic through **SF20**, and
CNN pooling/layout corrections. It is a changed numerical program: historical
f64 outputs, accuracy and performance cannot be reused as evidence.

Read `source/zk-torch-4/docs/morpheus_mean_rms_profile.md` and
`evidence/independent/FINAL_Q20_REVIEW.md` for full domains and limitations.
Full RMS-domain source completeness requires the declared public 32-plane
request layout; default 16-plane sources retain a narrower domain intersection.
SF21+ requires a wider proven arithmetic profile. InstanceNorm, Softmax/Sigmoid,
attention causality, unrelated rescales and full-model composition remain
separate. This package is **not one of the five requested complete independent
S&P acceptance reviews**.

For repository publication, place this evidence under
`zk-torch-4/docs/validation/20261030-q20-mean-rms/` without replacing the older
exp evidence. Apply `source/` as a repository-root overlay (or use the patch),
not as a nested repository. Final remote commit, file identity, PR and CI
verification belong to the separate publication receipt.
