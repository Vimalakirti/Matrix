# Independent review: isolated private-input/lifecycle candidate

Reviewed 2026-10-30 04:04 UTC. Candidate: the isolated private-input/lifecycle draft. Baseline: frozen published round-3 implementation, commit `2413eb8841b13caf3116d23f01d44bde1e5bed34`.

This targeted code review is **not one of the five whole-paper reviews**. No code or remote edits were made. The paused numerical/public-literal work was not resumed.

## Disposition

Suitable for an explicitly unverified draft increment within its narrow input-identity/lifecycle scope. No missing dependency on the paused literal/numerical changes or introduced source-ordering/API blocker identified. **Not approved for merge, runtime correctness, complete inference correctness or performance claims.** Rust/CUDA compilation and actual verifier tests remain unexecuted.

## Isolation verified

Compared every candidate file's bytes against the frozen baseline (excluding git/Python cache artifacts), and checked for missing baseline files. Exactly **11 files** differ; no baseline file is missing:

- `zk-torch-4/src/morpheus/{request,driver,wire,coins}.rs`
- `zk-torch-4/scripts/bench/morpheus/{evidence_admission.py,paper_tables.py,run_extra.sh,run_models.sh,test_evidence_admission.py}`
- `zk-torch-4/docs/{MORPHEUS_REPAIR_STATUS.md,PRIVATE_INPUT_REGISTRATION.md}`

Dag, basicblock, PIOP and statement files are byte-identical to the frozen baseline. In particular, no unfinished exp rewrite, padding operation, literal-registry field or new literal codec dependency is present. The request code uses existing Graph.edge_meta and public-map interfaces, both already available in the baseline. The new docs explicitly state that this isolated increment adds no public-literal registry or numerical constraints.

## Source/API checks

- Hidden inputs are producer-free Role::Input edges absent from the caller-selected public map. Model parameters and hidden inputs are retained as committed even when only consumed by advice; this binds identity, not the correctness of that advice.
- Packing remains `[model, input, fresh, carry]`. Monolithic raw/device/replica/link order is `[model,input,fresh]`; deferred request order is `[input,fresh,carry]`, using `input.start` for its local link remapping. WG and selectors stay in fresh sources.
- InputReg preserves exact raw payload plus suffix; RegisteredInput exports expected commitments and public program/layout descriptors, not private values. Both descriptor equality and exact mode-adjusted input commitment prefix are checked by verify_request. Empty input requires an explicit empty descriptor. Existing Graph/layout metadata are sufficient; no literal-registry dependency remains.
- Driver's only prove/verify_request call sites supply the new input registration consistently, reusing one input for repeated proofs. These are not independent inference samples.
- Expected model/input/fresh commitment groups are tagged separately in request transcripts. Final queues retain verified request digests that already bind each input identity. The existing trusted/verifier-owned queue precondition remains necessary.
- Request/finalization calls handle the fallible Coins opening API. Finalization reserves its opening before taking/extending model.raw, preserving the registration on immediate duplicate/lifetime rejection. Input allocation length matches the allocated suffix-owner range. Checked request/model/input owner bounds are disjoint.
- Coins clones share mutex-protected lifecycle state; opening reservation, one-time model registration and bounded input-source allocation are coherent with their production call sites. Repository scan found no stale seed-field access or unhandled old begin_opening call. Same-seed reconstruction and low-level replay remain explicitly outside the high-level lifecycle guard, as documented.
- Input-prefix/helper tests use matching types; the packing fixture uses min_lg=9, satisfying p<=min_lg for its p=9 WG/carry entries. Those Rust tests remain unexecuted.

## Checks actually run on the clean candidate

```
python3 -m unittest discover -s zk-torch-4/scripts/bench/morpheus -p test_evidence_admission.py
bash -n zk-torch-4/scripts/bench/morpheus/run_models.sh zk-torch-4/scripts/bench/morpheus/run_extra.sh
git diff --check
```

**24 Python admission/parser tests pass; shell syntax and diff checks pass.** These do not execute or validate Rust/CUDA proof code. Protocol-3 wire/transcript/receipt changes are consistent; old/missing receipt versions are rejected.

## Explicit remaining limitations

Prepared still contains ProverData, and preparation requires witnesses; a separate public-only remote-verifier constructor is not demonstrated. Model digit-width leakage, independent authentication of registry descriptors, queue provenance, complete cryptographic verification, normalization/advice linkage, causal semantics, no-wrap bounds and compiler correctness remain separate obligations. Stable input commitments reveal intended repeated-input linkability. Real mono/deferred proof acceptance and adversarial input/prefix/layout/lifecycle mutation tests, followed by Rust/CUDA compilation, remain mandatory before stronger acceptance claims.

## Version consistency follow-up (04:06 UTC)

Rechecked final wire.rs `PROTOCOL_VERSION = 3`, both request/final transcript prefixes using that constant before challenges, and the driver emitting that same constant. Wire magic is v3 and harness/admission expectations require 3. The updated documentation correctly distinguishes unchanged canonical statement codec version 1 and parameter profile `morpheus-ternary-v1` from protocol/wire version 3; no numerical dependency is implied. The adjusted admission test explicitly rejects missing versions and both versions 1 and 2. All 24 Python tests and diff check pass again.

## Prover-registration encapsulation recommendation

Currently public ModelReg/InputReg raw/commitment/descriptor fields permit callers to mutate a retained opening or rebind its commitment while reusing a hiding suffix. This does not bypass an independently trusted verifier descriptor, but weakens misuse resistance of the intended immutable-registration lifecycle. Recommended narrow hardening: make raw/c/statement/layout private, preserve read-only public() and add source_count() for the driver's count reporting; timing metadata may remain public. This is an API encapsulation recommendation, not a newly found verifier acceptance bypass. Lower-level APIs and reconstructing a backend from the same seed remain outside the guard regardless.

## Final encapsulation/freeze review (04:08 UTC)

The recommended narrow hardening is implemented and statically rechecked: ModelReg/InputReg raw, commitment, statement and layout fields are private; secs remains public; public() returns read-only borrowed descriptors and source_count() exposes a count. Both driver raw.len consumers now use source_count(). Repository caller scan found no external registration struct construction or stale private-field access; all internal mutation stays within request.rs. Public expected-descriptor fields intentionally remain constructible for the independently authenticated registry boundary. Updated docs accurately describe this distinction. `git diff --check` passes.

**Freeze disposition:** no unresolved introduced source/API blocker identified for this isolated candidate. Suitable for publication only as an explicitly unverified draft increment. This is not merge approval or a claim of Rust/GPU/runtime verification, and the separately paused numerical work remains excluded.
