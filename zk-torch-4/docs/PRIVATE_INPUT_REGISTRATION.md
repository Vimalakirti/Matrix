# Registered private-input binding (wire-v3 draft)

This is a source-level API/protocol repair, not an executed inference proof or
complete inference-correctness claim. The parent published base was
`2413eb8841b13caf3116d23f01d44bde1e5bed34`; this document describes the subsequent
uncompiled draft.

## What is bound

Every nonpublic leaf `Role::Input` is placed into a distinct registered-input
source group. Such inputs remain committed even if their only consumers are
currently-unconstrained advice nodes. All declared model parameters likewise
remain committed. This preserves identity; it does not make advice functional.
Public inputs retain their public-root checks. This isolated increment adds no
new public-literal registry or numerical constraints.
Neither private input values nor private model values are converted into public
literal arrays.

`register_inputs` produces prover-side `InputReg` (raw sources, hiding suffixes,
commitments, public program/type identity and exact input-source layout).
The raw sources, commitments and descriptor storage inside prover registration
objects are private fields, preventing external callers from accidentally
rebinding new payloads to old hiding suffixes. Read-only `public()` and
`source_count()` accessors expose only the expected public metadata.
The `RegisteredInput` view contains only commitments and public descriptors.
`verify_request` requires that independently expected view in addition to the
expected `RegisteredModel`. Both descriptors must be authenticated by the
application out of band, not supplied as trusted identity by a prover's proof.
The request API has no missing-input/existential fallback. An empty private-input
group still has an explicit descriptor with an empty commitments list.

Sources are packed in order: model, private input, fresh request, optional carry.
The fresh group includes graph masks, lookup selectors and public request
sources. Monolithic openings use model+input+fresh; deferred request openings use
input+fresh+carry and defer only model rows. A verifier compares the exact input
commitment prefix to its expected registry. Request transcripts bind the typed
model/input/fresh groups separately before challenges. Deferred finalization
binds verifier-owned request digests and carries; those digests already bind each
expected input identity. The final queue must come from successful verification
under the same program/model/layout, never an untrusted deserialized queue.

## Reuse and lifecycle

The same registered input may be proven repeatedly by reusing its exact raw
source and suffix. A different input requires a new input registration. Program,
shape, source-role and input-layout changes require a corresponding new trusted
descriptor; request public value changes do not alone change registration identity.
Graph masks and fresh request suffixes are regenerated for every unique opening.

The guarded Coins API now consumes opening identifiers once (shared by clones),
permits one model registration per Coins instance, and allocates disjoint input
suffix owners with an explicit total budget of 2^20 input sources. Model owners
occupy `[2^63,2^63+2^24)`, input owners start at `2^62`, and bounded request owners
stay below `2^56`. Reservations are never rolled back after later failure.
Finalization reserves its opening before moving registered model raw sources, so
a duplicate-opening rejection leaves the registered source object intact.
Reconstructing a new Coins instance from the same seed remains outside this
in-memory uniqueness guard; production must use fresh private seeds and must not
reuse deterministic test seeds. Low-level addressed stream rereads remain allowed
for exact fallback/replay and are not fresh protocol invocations.

## Validation status and limitations

Two input-specific Rust tests were authored: prefix replacement/reordering/
truncation/overflow/empty-group/reuse, and actual layout packing with disjoint
model/input/fresh/carry groups plus descriptor mutations. These tests have NOT
been compiled or run. Existing Python parser/scalar and host-extracted division
checks do not establish actual Rust verifier acceptance or rejection.

Required runtime tests include accepted same-input repeated proofs with different
fresh coins; rejection of changed expected input commitments, payload/suffix,
input layout or source order; both monolithic and deferred modes; model/input
cross-substitution; malformed descriptors; empty input/model groups; and lifecycle
errors through clones/concurrent callers. Full authenticated proof fixtures and
Rust/CUDA build results remain missing.

The verifier routines do not read private ProverData values, but the current
`Prepared` object still embeds ProverData and the preparation API requires witness
arrays. No separate public-only remote verifier construction/deployment has yet
been demonstrated. Model digit widths remain witness-derived public registration
metadata (explicit leakage); private-input widths use declared request types.
Numerical normalization, RMS sign/mean ties, causal semantics, all integer no-wrap
obligations and compiler correctness remain separate requirements.

## Version and isolation scope

This isolated increment deliberately leaves the canonical public-statement codec
at its existing version 1: its DAG/configuration/type/value field encoding has not
changed. The protocol/wire semantics are version 3. Request transcripts absorb
`[PROTOCOL_VERSION, 0, defer]` and final transcripts absorb
`[PROTOCOL_VERSION, 1, defer]` before any challenges, followed by the existing
full statement words and layout. Requests additionally distinguish model, input
and fresh-source groups with separate typed prefixes. Thus version 3 is not merely
an unbound wire header or log label. The driver emits the same protocol constant;
harness expectations and table admission require 3. The parameter profile name
`morpheus-ternary-v1` is unchanged: it identifies parameters, not wire semantics.

No exponential, public-literal-registry, tensor-padding, numerical DAG, PIOP or
canonical statement-codec edits from the separately paused work are included.
Normalization correctness therefore remains exactly as unresolved in the published
base, apart from this increment's independent identity/lifecycle improvements.
