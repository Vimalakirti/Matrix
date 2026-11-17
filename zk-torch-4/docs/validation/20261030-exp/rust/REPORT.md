# PR1 Rust validation evidence

**Date:** 2026-10-30. This bundle contains reproducible, scoped validation—not a full verifier or CUDA pass.

| Check | Observed result |
|---|---|
| Unmodified production `cargo check --locked --all-targets` | **Blocked**, exit 101: CUDA toolkit not found at `/usr/local/cuda/bin/nvcc` |
| Isolated CUDA-build-script-only `cargo check --locked --all-targets` | **Passed**, including all Rust binary targets and test code |
| Original CPU ExpHelper/TwoPow and DAG-builder tests | **434 passes:** 14 distinct unchanged test bodies × activation scales 0–30 |
| Five original CPU PIOP replay test bodies | **Not executed**: real CUDA FFI symbols prevented linking |

The first all-target check exposed E0596 in `src/bin/morpheus_piop_e2e.rs:287`. The candidate owner repaired the mutable borrow and moved the immutable keys binding after it. The complete all-target check then passed. Both failure and final success logs are retained. The final CPU harness was rebuilt and all 434 invocations repeated on the repaired snapshot.

## Scope and provenance

Published base: `f192323f0725fed17a27dee21442fcba25460d2f`; base tree: `96844dc3236c37f2983c42d238864dcda809d855`. The tested candidate adds the nine-file exp repair plus the binary borrow fix. `source-sha256.json` requires the 441 non-cache files from the tested snapshot, including every recorded Rust source, build script, manifest and lockfile. `observed-inventory-sha256.json` retains the original 446-file observed inventory. Exactly five generated Python `__pycache__/*.pyc` files are excluded from reproduction (listed in `excluded-generated-cache.json`); they are not Rust compiler inputs and need not exist in a clean checkout. No actual Rust/source identity checks were relaxed. These hashes identify tested bytes, not a claim that the candidate was published. Additional files may exist in a checkout; they are not claimed as tested by this inventory.

Official toolchain: **rustc 1.99.1** (`58cf8f9c35b355199727627d07ae34a4e2846d67`, 2026-10-13), **cargo 1.99.1** (`5f94df4789f005f9a352888e8355ffc645b7ed0e`, 2026-08-27), `x86_64-unknown-linux-gnu`, LLVM 23.1.1. Installation followed [official Rust guidance](https://rust-lang.org/tools/install/), using `https://sh.rustup.rs` over verified HTTPS and `-y --no-modify-path --profile minimal --default-toolchain stable`. Installer SHA-256: `9502b514362a21096210d498f143b8cdacda365d3dfcf45e8f5541e089ccafc0`. Installation was workspace-local, without system/profile changes. Dependencies used the existing Cargo.lock and `--locked`.

## Every source adaptation

All adaptations occur in disposable copies; production source files are not modified.

1. **Type-check copy:** dependency `build.rs` omits CUDA discovery/compilation/archive/link directives. Header rebuild tracking remains. No Rust module, target, test, or `cfg` selection is removed. This is a **modified-build type check**, not a production build/link.
2. **CPU runner copy:** same build-script omission; remove only the FFI static-library `#[link]` annotation. All external declarations remain unchanged. Any retained unresolved GPU call still fails linking.
3. Expose the original `basicblock::exp::tests` and `dag::builder::tests` modules by changing their `#[cfg(test)]`/visibility gates; replace selected `#[test] fn` declarations with `pub fn`; add two sequential runner binaries. Original function bodies, arithmetic, assertions, helpers and production implementations are unchanged.
4. **Blocked PIOP attempt:** use the same build/link omission and expose only `piop::exp_tests` to a standalone runner. Linking failed on `agl_mpiop_*` and `cuda_*`; it supplies **no PIOP pass evidence**.

Exact diffs and before/after SHA-256s accompany the successful final CPU and type-check copies. `test-body-sha256.json` records 14 executed and five blocked original test bodies. The 14 executed bodies were independently compared byte-for-byte with their adapted counterparts. This is standalone test-body execution, **not** a normal full `cargo test` suite.

No GPU functions were stubbed, no fake zero results were returned, no unresolved-symbol errors were suppressed, and no tracked binary was executed. Successful runners used real Rust field/Witness/polynomial/DAG implementations. Unused GPU code was discarded by the linker. The binary audit found no unresolved `agl_`/`cuda` symbols or CUDA dynamic dependency in the successful graph runner. Negative domain tests intentionally catch panics; the corresponding printed panic messages are expected and the processes exited 0.

Coverage includes decomposition/selector recovery, domain rejection, TwoPow behavior, actual exp CPU DAG output at zero, padding masks, eager request-versus-model roles, and DAG construction/compilation. It does **not** establish normalization/causal correctness, authenticated PCS/full proof acceptance, CUDA kernel behavior, performance or model accuracy.

## Reproduce

Prerequisites: Python 3.9+, Bash, a C linker, an available Rust/Cargo toolchain (recorded version above), registry access or cached locked dependencies. Scripts install nothing. Use a source checkout matching the manifest and a **new work directory outside that checkout**:

```sh
python3 verify_sources.py /path/to/source
bash run_checks.sh /path/to/source /path/to/new-validation-work
```

`run_checks.sh` verifies source identity, creates isolated copies, executes all-target type checking, builds both CPU runners, and executes all scales. It resets relevant profile overrides and uses an empty runtime directory so an unrelated `config.yaml` is not selected. Logs and adaptation manifests stay in the supplied work directory. Portable-wrapper execution on a fresh cache-free source copy is recorded separately below; the original logs remain unchanged.

For the unmodified production boundary, use an untouched copy and run:

```sh
cargo check --locked --all-targets --manifest-path /path/to/source/zk-torch-4/Cargo.toml
```

For the historically blocked PIOP attempt, `prepare_harness.py SOURCE NEW_COPY cpu-replay` creates the exact standalone adaptation; then `cargo run --locked --manifest-path NEW_COPY/zk-torch-4/Cargo.toml --bin cpu_exp_replay` attempts linking. Do not treat link failure as a test result or substitute GPU stubs.

## Actual portable-wrapper replay

The exported `run_checks.sh` was actually executed to terminal **exit 0** on a fresh cache-free copy with the already installed official toolchain and `CARGO_NET_OFFLINE=true` (cached locked registry dependencies). No installation or CUDA download occurred.

- Required identity verification: **441 source files and 19 original test bodies passed**.
- Complete isolated modified-build all-target check: **passed**.
- Both CPU runners freshly built; all scales 0–30 completed: **434 passes**.
- Generated CPU and type-check adaptation manifests are byte-identical to the bundled expected manifests. Assertions/test bodies are unchanged; no GPU stubs.
- Logs: `logs/portable-wrapper-{terminal,typecheck,cpu-build,cpu-scales,toolchain}.log`.

The original 446-file inventory is retained solely as historical observation; the five `.pyc` omissions are explicit and do not exclude any Rust/compiler source input. Extra unrecorded checkout files are neither required by the identity verifier nor claimed as part of the recorded tested snapshot.

## Bundle integrity

`SHA256SUMS` covers bundle files. Logs have private workspace prefixes replaced with `<validation>`/`<toolchain>`; diagnostics, outcomes and test output are otherwise retained. No CUDA packages were downloaded for this evidence. No external publication, merge, deployment or paid GPU work occurred during validation.
