# Independent implementation review — 2026-10-30

Independent AI-assisted source review, separate from the integration author. This is a targeted implementation review, **not one of the five whole-paper reviews**.

Reviewed the integrated repair against its pinned source baseline `903d6e27be1d4a58c2a36a88917283dc2595b1da`. No repository code was edited by this reviewer. No remote mutations, GPU runs, installations or deployment.

## Disposition

**Suitable for an explicitly unverified draft repair PR; not approved for merge, production use, complete inference correctness, or benchmark claims.** The definite introduced blockers found below have been repaired and rechecked. No other definite compile error was identified in the static changed-code scan, which is not compilation.

## Findings and resolution

1. **Blocking compile error in initial boundary proposal, repaired.** `link.rs` inserted verifier-only shape checks into `prove`, referencing undefined `pf`/`c_src` and returning `None` from a `LinkOutput` function. Integration author moved this block into `verify`. Rechecked final `link.rs:366–370`: inputs and return type now match; no stray guard remains in `prove`.
2. **Whole-table parsing crash on malformed measurement, repaired.** `paper_tables.parse` converted request times before checking admission; changing a valid fixture's `prove: 1.00s` to `prove: ..s` reproduced a `ValueError`. Final `paper_tables.py:42–57` performs admission before conversions and catches conversion failures for optional fields. Rejected records preserve fields used by offline tables.
3. **Oversized integer crash in admission, repaired.** A 5,000-digit request count reproduced Python's integer-digit-limit `ValueError`. Final `evidence_admission.py` bounds count/batch token lengths and batch cardinality before conversion. Tests now cover both direct admission and table-parser entrypoints.
4. **AR metadata is not a complete program validator; scope clarified, not solved.** `ArStatement::validate` checks signed-bound roots, public constants, canonical selector arrays and shift, but does not independently establish embedding/rank wiring, canonical arithmetic, causal masks or numerical helper correctness for an arbitrary supplied graph. Final module documentation explicitly requires a trusted independently pinned DAG, says arbitrary caller DAGs are outside the guarantee, and says `ar: None` proves no AR claim. Range-root output indexing was also changed to `.first()` rejection rather than a malformed-node panic. Do not present this as complete AR soundness.

## Reviewer checks actually run

```
cd /path/to/morpheus
python3 -m unittest discover -s zk-torch-4/scripts/bench/morpheus -p test_evidence_admission.py -v
git diff --check
```

Result: **23 Python tests pass**, including parser-level malformed measurements, malformed parts/registration, oversized counts and valid fixtures. `git diff --check` passes. The reviewer independently reproduced findings 2 and 3 before repair and inspected their final changes.

The integration author reports 10,048 host-extracted CUDA integer-division cases against Python big integers; this reviewer statically inspected the exact fallback and guard logic, but did not independently rerun that suite. Host extraction is not NVCC compilation or GPU verification.

## Required before acceptance beyond draft

- Rust compilation/unit tests and NVCC compilation remain unexecuted: no toolchain available in this workspace. Resolve actual compiler findings before merge.
- Actual verifier adversarial tests are still needed: absent authentication at request/final boundaries, malformed in-memory proof shapes and statements, AR descriptor/public-value/proof-reuse mutations, firstargmax ties and cross-shard cases, ScaleUp tampering, and accepted honest witnesses.
- Existing wire-helper unit tests are not full verifier tests and were not compiled here. Handwritten scalar model tests do not establish implementation behavior.
- Require pinned canonical graph/program/statement validation, nonlinear normalization/advice linkage, causal masking, and upstream no-wrap bounds before full inference claims. These are known residual correctness obligations, not fixed by this draft.
- Log admission is not cryptographic proof or source/binary/device provenance certification. Historical rows without receipts remain inadmissible; fresh real measurements are not yet available.
