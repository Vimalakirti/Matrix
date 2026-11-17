# Independent registered-model / public-statement review

Reviewed 2026-10-30 02:34–02:43 UTC, integration delta after exp-unit baseline `7e16a0c`. Separate reviewer; not an implementation author and not one of five whole-paper reviewers. Static source review only, with `git diff --check` passing. Rust/CUDA compilation and all newly authored Rust tests are NOT RUN.

## Disposition

Suitable for an explicitly unverified draft increment; the registered source-layout identity gap found during review has been repaired and statically rechecked. No definite introduced Rust type error identified in inspected source; this is not compilation, cryptographic certification, or complete inference-correctness approval.

## Positive source correspondences

- Monolithic packing isolates model sources, with raw buffers, resident/replica authentication buffers, commitments and link source ordering consistently `[model, request]`. Model payload and hiding suffix are copied verbatim from registration. Fresh request bank and source suffixes are sampled separately. Deferred ordering remains `[request, carry]`, with link remapping based on request start.
- Verifier requires externally supplied expected model commitments in both modes; monolithic commitment prefix must exactly match. Empty model requires an explicit `RegisteredModel` with empty commitments and matching program/layout descriptors; registration skips an empty commitment call and request WG guarantees a nonempty request group. API callers formerly passing `None` must update.
- Statement encoder enumerates all 39 BasicBlockType variants without a catch-all, including stored public operator fields and Einsum's read-only cached schedule. It includes all Dag structural fields, edge type/shape/scale/role/commitment metadata, sorted public-map values, ordered output claims, graph config, profile/ring/key identity and the captured convolution protocol switch. Private tensor payloads and Coins are not read by the encoder.
- Every encoder word is split into two 32-bit limbs before field absorption, avoiding modulo-q aliases of raw u64 metadata. Finite f64 epsilon bits are bound, nonfinite values rejected. Structural/index validation and canonical public field-value checks are present.
- Full public request words enter the transcript before graph challenges. Finalization uses stable program words and ordered per-request digests; those digests originate from the successful request verifier's transcript containing that request's public inputs/outputs, model identity and carry. Version-2 proof magic and separate request/final prefixes prevent accidental v1/mode reuse.
- Convolution protocol selection is snapshotted once into Graph and used for classification/pruning/reduction, rather than rereading a mutable environment setting after binding.
- LINK_ROWS causes verifier rejection. Empty final queues are rejected, resolving the formerly possible empty-buffer path. Request identifiers yield openings 1..2^32-1; batch identifiers yield openings 2^32..2^33-1, so the new bounds remove the old boundary overlap.

## Review findings / required boundaries

### Registered source-layout identity gap — repaired during review

Initial `ModelReg.statement` stored only `statement::model_words`, which omits source layout. Same graph metadata can yield different value-derived model digit widths/packing while retaining the same source count. The initial identity/count assertions did not prevent reinterpretation under a changed model layout.

Final reviewed code adds exact `ModelReg.layout` for model-only tensor descriptors: model source range, ell/min_lg, edge/n/p/bit flag, and every piece source/base/m/p/plane0/hb/hbits field. It checks this before proving/finalizing. New borrowed `RegisteredModel` carries commitments, program words and layout words from the independent registry; both verifier entrypoints call `registration_matches`, comparing all three against the trusted local Prepared instance. All four driver verifier call sites now use `m.public()` consistently. This resolves the identified gap at source level, without pretending that the registry is authenticated by the prover. The source-layout descriptor still exposes existing value-derived model widths; this change does not remove that leakage.

### Queue provenance and lifecycle

`verify_final` does not reverify requests or derive queue entries from raw request proofs. Its public `QueueEntry` arguments must be retained outputs of successful `verify_request` for the same trusted program/model, not arbitrary network records. Binding a supplied digest is not verification of that digest's provenance. Final QueueEntry/verify_final documentation now states this requirement explicitly. The driver follows this requirement by building vqueue from successful verification.

Identifier range checks establish domain separation, not uniqueness. Callers must still ensure nonrepeated j/b within the private-seed lifecycle; the existing counter is not a duplicate-ID registry. Queue replay/duplicate policy remains application-owned.

### Scope limits

- The codec binds a trusted program; it does not prove that the program implements intended inference. Advice normalization, causal masking, upstream no-wrap bounds and private-input integration remain independent obligations.
- Model registration is local trusted setup, not network authentication of a model owner. ModelReg fields are public; verifier must independently pin expected program/layout/commitments.
- `model_words` contains mode/carry/config identity. A ModelReg cannot be reused unchanged between differing protocol modes merely because underlying weights agree; this is conservative identity separation.
- Encoder and registry helper tests are authored but uncompiled. The helper tests do not execute a proof verifier or GPU backend.

## Acceptance evidence still required

Compile all changed Rust modules and CUDA dependencies. Run encoder mutation tests; real mono/deferred proof tests with changed registry commitments, model payload/suffix, missing/reordered prefixes, empty models, fresh per-request randomness with stable model prefix, program/layout changes, public input/output changes, carry/digest/queue order mutations, LINK_ROWS, absent authentication and boundary identifiers. Re-run final exact changes after any repair. Historical benchmark rows are not evidence for the new wire/packing/protocol behavior.

Final follow-up: inspected new RegisteredModel signatures, all local caller sites and exact layout equality checks; reran `git diff --check` successfully. A pure registered-descriptor mutation test has now been authored and reviewed: it changes each representative program/layout word and checks truncation, through the helper used by registration_matches. It checks exact descriptor equality, not full encoder/verification behavior. No Rust tests were executed.

## Final admission/helper follow-up (02:41 UTC)

Reviewed `registered_descriptor_matches` factoring and its representative word-mutation test; no static type issue found. Driver, both benchmark harnesses and admission fixtures consistently require `protocol=2`. Missing/legacy protocol receipts are rejected, so prior v1 runs cannot become current evidence solely by carrying the earlier auth markers. Independently ran all **24 Python admission/parser tests**, which pass; `git diff --check` also passes. No unresolved introduced source blocker identified in this reviewed delta, subject to the explicit uncompiled/runtime limitations and trust boundaries above. The codec's final scale-width and request-value-reuse edits were subsequently reviewed as recorded below.

## Final frozen-code follow-up (02:43 UTC)

Inspected the two final codec edits. `SF_LOG > 30` rejects before dereferencing the existing i32-shift-based `SF_FLOAT`, making its supported domain explicitly 0..=30 and avoiding a negative/overflowing scale in this encoding path. This is not a claim that all earlier graph-construction entrypoints validate arbitrary scale configuration. The new `model_identity_allows_new_request_values_but_binds_metadata` test coherently checks unchanged model identity but changed request encoding for new public input/output values, and changed identity for shape, role and chunk-width metadata. Its Graph construction/type use matches existing tests; it remains uncompiled. Re-ran `git diff --check`, passed. These final two code edits are covered by this review; no additional introduced source blocker identified.
