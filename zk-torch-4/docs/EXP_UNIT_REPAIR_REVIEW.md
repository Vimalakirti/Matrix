# Independent review: fixed-point exponential unit repair

Reviewed 2026-10-30 against integration baseline reported as published PR #1 head `7b50735`. Narrow delta: `zk-torch-4/src/dag/builder.rs`, `zk-torch-4/src/lib.rs` (unit-test configuration only), and `scripts/audit/test_exp_unit_reference.py`.

## Disposition

Suitable for a draft repair increment with uncompiled Rust regression explicitly disclosed. No introduced static compile/type issue identified. This is not a nonlinear soundness or causal-mask repair and is not one of the five whole-paper reviews.

## Numerical correctness

The cubic exp approximation's constant term carries scale factor 15. `Witness::new` accepts raw field coefficients; the previous raw coefficient 1 encoded 2^-15, not one. Raw coefficient 32768 correctly encodes one. At x=0, ExpHelper emits r=0,k=0; TwoPow returns 32768 at sf15; nonconstant terms vanish; their product is 2^30 at sf30, which rescales exactly to 2^SF_LOG. The repair is therefore correct for this missing-unit defect. Padding is still explicitly zero-padded as before.

This changes outputs and invalidates reuse of historical timings/accuracy outputs without rerun. It does not establish global approximation error, functional helper correctness, canonical bound constants, causal noninterference, or end-to-end verifier acceptance.

## Rust regression inspection

`exp_zero_is_one_at_output_scale_cpu_forward` traverses the actual constructed DAG's public `topo`, invokes `BasicBlock::run` imported through the module's existing glob import, and updates owned returned witnesses. This matches the repository's CPU `Dag::run` logic. The input shape `[2]` avoids padding ambiguity; the final data access and field comparison match available polynomial and field interfaces. It bypasses the benchmark hook and explicit GPU backend dispatch.

**Harness issue found and repaired during review:** `lib.rs` previously treated process `args[1]` as an explicit YAML path, so filtered unit-test invocations could fail before test logic. The final reviewed delta makes `CONFIG_FILE` choose `config.yaml` under `cfg!(test)` and makes `explicit` false in that same unit-test mode. I rechecked both changes: they are type-compatible, preserve production parsing/fail-closed behavior, and remove this identified argument-parsing blocker for `cargo test --lib`. They do not change non-test-compiled library behavior in integration tests. The focused command remains NOT RUN, and Rust/CUDA dependencies still require their build environment even for CPU-forward logic.

## Checks actually executed by reviewer

```
cd /path/to/morpheus
python3 zk-torch-4/scripts/audit/test_exp_unit_reference.py -v
git diff --check
```

Both scalar tests pass, including sf1 through sf29. Diff whitespace check passes. These are mathematical-model tests, not Rust graph execution. Rust compilation, the new Rust test, GPU execution and verifier tampering tests remain NOT RUN due unavailable toolchain/environment.

Follow-up review at 02:06 UTC: re-read the final `src/lib.rs` diff and reran `git diff --check`; no new static compile concern found.
