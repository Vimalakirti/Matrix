# Independent registered-statement binding audit

Date: 2026-10-30 UTC. This is an independent source diagnostic, not formal PR approval, an executed end-to-end proof test, or inference certification.

## Audited source and provenance

Published repair reference: `7b50735a2052513069ea39475fb8dd0c9bdd9625`, tree `ae83cf8d58d2516e489c48d3466a768e8f1b04c0`.
Baseline: `903d6e27be1d4a58c2a36a88917283dc2595b1da`.
The relevant source blobs inspected in the integration snapshot are:

| Path under zk-torch-4 | Audited repair blob | Baseline blob |
| --- | --- | --- |
| src/morpheus/request.rs | 346076421dd025d58e73ef74b06b645f34e0fe25 | 188ea23bb88793dd77d062fe910f9e1f54932cf7 |
| src/morpheus/piop/mod.rs | a5c609446f7c4cf350ce015814b211dbf02c8fc0 | e6ae2c7c4769d541ff04a8aab51f9f3de967ac28 |
| src/morpheus/driver.rs | a8dbcc7baf23e1939b09bfb5a2b7314d665a6ee8 | 0146a14ecede17881a6ca9753fe2d174bda38c0d |

A concurrent four-file ExpHelper/unit draft did not change these three audited blobs. Line references below describe the audited repair blobs, not a prospective fix.

## Finding and scope

Monolithic verification does not enforce an externally registered model identity. This is a pre-existing fixed-model API/conformance gap, not an exhibited algebraic forgery against a prover-selected commitment statement. Public inputs and claimed outputs are constrained when independently supplied through the trusted verifier graph. Private inputs have no separate expected-registration hook.

A statement-adaptive committed-witness knowledge theorem can remain valid. Fixed-model inference integrity requires an authenticated registration comparison absent in monolithic mode. Deferred mode has a model-commitment hook, but the application must authenticate and pin it; commitment authentication is not external registry authentication.

## Source trace

- `request.rs:113-150`: `Prepared` contains graph, prover data, layout and source ranges. Monolithic preparation sets `model=0..0`, packing model/input tensors with request-local data.
- `request.rs:182-193`: `register()` commits model sources with randomized suffixes. `driver.rs` invokes it only for deferred mode.
- `request.rs:216-239`: request transcript absorbs layout, optional model commitments, request commitments and request number; AR mode additionally absorbs designated public arrays.
- `request.rs:454-497`: monolithic verification requires `model=None`, takes commitments from `out.c_src`, and invokes `Walk::new(&pp.g,None)`. It never reads `pp.pd`.
- `piop/mod.rs:84-176`: `Graph` retains model-role flags, not model weight payloads. This source has no `g.consts` field.
- `piop/mod.rs:731-746`: output and public-input roots use concrete `Graph.outputs` and `Graph.public` arrays.
- `request.rs:577-608`: deferred finalization uses externally supplied model commitments. Application acceptance must retain the same pinned model and verifier-produced pending queue through successful finalization.

## Concrete verifier-view equivalence

Consider identical multiplication DAGs with public input x=1:

- Intended model A has weight w=2.
- Substitute model B has weight w=3.
- The weights have identical declared width/type and packing layout.
- The statement claims output y'=3 and supplies an honest B proof.

An A-based verifier with claimed-output root y'=3 has the same graph, public input, output claim, layout and options as B verification. Its differing model payload is in `pp.pd`, which verification does not read. Every verifier operation is therefore identical to honest B verification, so correctness entails identical acceptance.

This is a source-level acceptance-equivalence argument, not an executed GPU proof. It does NOT claim acceptance while the output root remains fixed at 2. `Graph.outputs` is the claimed output, not independent recomputation of intended M(x). The benchmark prepares outputs from the same honest witness, which does not establish external fixed-model identity binding.

## Public and private inputs; transcript boundary

Independently pinned `g.public` values are constrained through root MLE equations. Do not report arbitrary public-input substitution against an unchanged trusted `Prepared`. A private `Role::Input` omitted from `public` has no separate immutable expected-input commitment API in either mode; deferred registration covers model sources only.

Trusted DAG and configuration are separate application assumptions. The current request transcript does not generally encode the entire DAG/configuration/all public inputs and outputs. This is full-statement transcript conformance work, not independently demonstrated fixed-statement forgery. A repair should use canonical full public-statement encoding; an external statement digest requires its own explicit binding assumptions/reduction. It is not automatically equivalent to the paper's injective full-prefix oracle interface.

The paper's registered-integrity experiment expressly requires an immutable authenticated model/input/configuration record and rejection on mismatch. Current monolithic execution does not implement that interface. An application registry trust boundary must not be misreported as an algebraic commitment forgery.

## Smallest privacy-preserving repair

1. Separate immutable model sources from fresh request/mask sources in both modes.
2. Register once and retain the exact full model openings, including randomized hiding suffixes.
3. Require externally pinned expected model commitments and registration metadata in both verifier modes, before graph challenges.
4. Monolithic mode opens registered model sources together with fresh request sources. Deferred mode retains its carry/finalization structure and the same pinned model across both verification stages.
5. Keep graph masks, request suffixes and opening randomness fresh. Do not recompute randomized model commitments with fresh suffixes and require byte equality.
6. For private-input registered integrity, separately register input sources or prove their equality to the registered sources. Whole-request commitment equality is unsuitable because fresh W_G and proof-local data are co-packed there.
7. Bind canonical program/configuration identity and public statement before challenges.

Do not substitute exposing private weights as public roots, deterministic suffixes, or an additional hash of prover-selected commitments.

## Required acceptance tests

- Same model registration accepts two fresh requests with distinct request randomness.
- Model B proof rejects under expected model A; preserve x and use B's claimed output to avoid confusing output-root enforcement with model binding.
- Correct model registration succeeds in monolithic and deferred modes; incorrect or missing registration fails closed.
- Deferred request/finalization reject model mismatches; application cannot report final acceptance before successful finalization.
- Independently fixed public input mutation rejects; private-input registry mismatch rejects once that API is implemented.
- Registration commitments retain their original suffixes; request masks/suffixes and opening randomness remain fresh.

No Rust compiler, Cargo, CUDA compiler/runtime or GPU execution was available during this audit. No end-to-end acceptance tests were executed. Baseline comparison confirms the model-binding gap predates the repair draft. No implementation code was modified by this audit.

## Subsequent repair draft status

The subsequent wire-v2 draft separates model sources in both modes, retains
registered raw model payloads and hiding suffixes, and requires externally
expected model commitments. Monolithic verification compares the proof's model
prefix to that expectation. Request masks and suffixes remain fresh. Full public
statement encoding is absorbed before challenges; registration reuse also checks
the immutable program/type identity. These are source-level repairs, not executed
proof-test results. The privacy/registration preconditions and private-input gap
above remain applicable. See MORPHEUS_REPAIR_STATUS.md for current test limitations.
