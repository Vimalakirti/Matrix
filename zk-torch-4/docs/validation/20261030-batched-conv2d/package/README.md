# Conv2D tensor batching — minimal publication evidence

This export contains **no complete source tree**, native binary or GPU stub.
It preserves the 3-file source patch, exact source manifests, test-body identity
receipts, original success/failure logs, scoped independent reports/tests, and
portable reconstruction/validation tools. Historical log paths are retained as
original evidence; executable scripts require explicit paths or use their own
relative location, with no private workspace defaults.

## Exact source boundary

- Published frozen CNN base: commit `7241f377bebd47e911e4a587bdab1b0c7f89baf5`.
- Base: 454 canonical source files; manifest SHA256
  `da3f4f1bae5927b06547d61bd061725a46634db1cb944df469da532084770113`.
- Final: 455 canonical source files; manifest SHA256
  `a57be8f20529556584fef5a97d8bf24c54d77809835890bd4e2678848c227579`.
- Patch SHA256:
  `6e755beca69b513a068b3337c3eae5b64c1ee5cc1435b62121967d8689b54666`.

Only `zk-torch-4/src/morpheus/piop/cv.rs`, `piop/mod.rs`, and new
`piop/cv_batch_tests.rs` change. The builder is unchanged from the published
CNN base, which already includes the three explicit-batch Conv builder fixes.
`SOURCE_IDENTITY.json` and `evidence/overlay-manifest.json` retain exact hashes;
the latter intentionally uses its original `final_sha256` field.

The original local full archive remains untouched, SHA256
`47bfbf050cf7283ac0b017c004a7865d5c5cc56f6c792114a7ac920c67525511`.
Its 989 listed artifact entries plus its own manifest are **990 regular files**:
454 base-source files, 455 final-source files, 55 evidence files, 24 independent
review files, README and artifact manifest. This export omits both duplicated
source trees (909 files, 18,887,789 uncompressed bytes). See
`ORIGINAL_ARCHIVE_INVENTORY.json`. Do not add that full archive to repository docs.

## What passed, and what did not

- Disclosed build-only adapted all-target Rust typecheck: passed.
- Three genuine unchanged Conv CPU test-body invocations collectively contain
  32 forward dimension/stride/dilation cases and the helper/source checks below.
- Actual normal/PP reduction-table helpers: eight K-field cases, with PP
  alpha=0/1/nonbase extension points; three actual source-binding fixtures;
  collision and integer-index-overflow rejection checks.
- Inherited CNN CPU tests: 10 bodies × SF0..20 = **210 executed passes**;
  combined with the three Conv invocations, **213 test-body invocations**, not
  45 separate tests or a rerun of every earlier project test.
- Independent narrow checks: 147,192 convolution assertions; 96 flat-source map
  cases; padding-mask mutants; 36,552 wrapper assertions; exact-source audit.
- Native all-target build: **blocked**, missing CUDA toolkit/nvcc.
- Authored masked-PIOP verifier/link-row replay: **linker-blocked** on retained
  CUDA extern symbols; **zero replay bodies executed**.
- Native PCS/CUDA/full-model proofs: **not executed**. No whole-project or
  five-independent-review acceptance is claimed.

`evidence/VALIDATION.json` distinguishes these stages. Caught panics in the CPU
log are explicit invalid-geometry tests. Tensor batching is not deferred-proof
batching. `evidence/RELATION.md` gives the exact normal/PP equations, logical
B/C/D masks, signed-integer no-wrap assumptions, exponent no-wrap, and costs.
The unresolved source-bypass issue for four other convolution families is
preserved in `evidence/REMAINING_UNFLATTEN_FAMILIES.md` for the next repair.

## Verify and reconstruct without duplicate source snapshots

Use a checkout of the exact published CNN base above, or another checkout whose
454 canonical paths match the authenticated base manifest. Additional published
evidence paths are neither copied nor authenticated as source. Supply a fresh,
non-overlapping output path:

```sh
export PYTHONDONTWRITEBYTECODE=1
python3 verify_export.py /path/to/this-export
python3 reconstruct.py --base /path/to/exact-cnn-base \
  --destination /tmp/conv-reconstructed
python3 verify_export.py /path/to/this-export --source /tmp/conv-reconstructed
python3 test_reconstruction_safety.py --base /path/to/exact-cnn-base
```

Reconstruction checks pinned base/final manifest and patch identities, validates
all base bytes, applies exact unified hunks without shell/fuzz, verifies all 455
result hashes **before writing**, and then verifies written bytes. It rejects
existing/overlapping destinations, source symlinks, unsafe paths, and altered
base bytes. It never writes to the base. Canonical manifests authenticate file
contents, not Git metadata or executable modes.

`export-manifest.json` contains `{files:[{path,sha256,bytes}]}`, excluding only
itself. `verify_export.py` checks exact export inventory, byte counts and hashes;
optional `--expected-manifest-sha` pins that manifest. Optional `--source` checks
all 455 canonical paths without treating additional publication files as source.

## Re-run honest CPU checks; preserve proof blockers

With Rust/Cargo configured and dependencies cached (`--offline`):

```sh
bash evidence/run_checks.sh /tmp/conv-reconstructed /tmp/conv-harness \
  /tmp/conv-target /tmp/conv-logs
```

This wrapper exposes unchanged test bodies and removes only CUDA build/link
requirements, retaining every native extern declaration. No FFI definition is
supplied. If CPU checks pass but PIOP linkage remains blocked, it returns **3**,
not a green overall pass. Original diffs and test-body identities are included.
Do not share a target directory with another concurrent build.

```sh
python3 evidence/prepare_inherited_cpu.py /tmp/conv-reconstructed /tmp/conv-inherited
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/conv-target cargo build --offline \
  --manifest-path /tmp/conv-inherited/zk-torch-4/Cargo.toml --bin cpu_conv_inherited
for sf in $(seq 0 20); do
  ZK4_SF_LOG=$sf RAYON_NUM_THREADS=2 /tmp/conv-target/debug/cpu_conv_inherited \
    /tmp/conv-reconstructed/zk-torch-4/bench_config.yaml || exit $?
done
```

Independent scripts below reproduce arithmetic/source checks, not Rust proofs:

```sh
python3 independent-review/check_relations.py
python3 independent-review/check_fallback.py
python3 independent-review/check_mask_mutants.py
python3 independent-review/check_wrapper_integration.py --root /tmp/conv-reconstructed
python3 independent-review/check_joint_overlay.py --base /path/to/exact-cnn-base \
  --root /tmp/conv-reconstructed --manifest evidence/base-source-manifest.json \
  --supplement /path/to/exact-cnn-base \
  --reviewed-hashes independent-review/reviewed-source-hashes.json
```

The CLI-only portability changes and their rechecks are recorded separately in
`independent-review/publication-cli-inputs.patch` and its receipt. Run scripts in
a working copy if preserving the exported evidence bytes exactly: independent
check scripts write fresh result JSON alongside themselves. Original logs are
not rewritten or dropped by this export; publishers may losslessly encode them
only if byte-perfect decoding is independently verified.
