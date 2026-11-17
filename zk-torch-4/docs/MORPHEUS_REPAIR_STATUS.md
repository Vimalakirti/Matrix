# Morpheus repair draft — not inference certification

Base: `903d6e27be1d4a58c2a36a88917283dc2595b1da` (main rechecked 2026-10-30 UTC).
This is an implementation repair draft, not a reproduction of paper measurements,
not a complete soundness audit, and not ready to merge without the checks below.

## Changes

- Both request and deferred-finalization verifiers require authentication, even
  when diagnostic proof production disables it. Shape/canonical-field guards
  cover decoded and directly constructed proof values before indexing.
- ScaleUp adds the missing exact remainder equation. This fixes its local field
  relation, not the full signed integer no-wrap argument.
- An explicit AR descriptor binds public selector arrays and constants, validates
  contiguous vocabulary shards, unique one-hot selections and suffix shift.
  Logit bounds and vocabulary-scaled/index-offset rank comparisons implement
  lowest-index argmax under the checked modulus budget. Public tensor widths
  derive from their already-public values; private widths remain fixed.
- Sequential suffix generation replaces an unrelated randomly drawn suffix.
  This changes witness-generation cost; historical timings are not applicable.
- Benchmark table admission rejects missing authentication/verification, debug
  flags, wrong counts/IDs/batches, malformed measurements and failed processes.
  This is log admission, not cryptographic provenance or a benchmark rerun.
- CUDA div_z validates each floating estimate by exact multiplication before
  subtraction, bounds correction work, and falls back to restoring division.

## Trust boundary and unresolved correctness

`ArStatement::validate` is NOT a canonical numerical-program validator. The
application must independently pin the trusted DAG, graph wiring, model and
public statement, and explicitly select the AR API. Generic `run` / `ar: None`
proves no AR claim. The descriptor does not establish that arbitrary caller DAGs
connect embedding and selectors to the declared rank equations. It does not
repair unconstrained nonlinear normalization advice, complete causal masking,
all intermediate no-wrap bounds, source/program codec coverage, deterministic
key expansion versus a fresh-uniform setup theorem, or private-seed lifecycle.

In particular, InstanceNorm/Softmax/Sigmoid/RMS need specified deterministic
numerical relations and constrained advice. Choosing an integer approximation
must not be described as preserving existing f64 outputs. The sequential
witness may fail final validation until genuine causal/nonlinear relations are
fixed. No task-accuracy equality, accepted inference proof, GPU timings,
whole-process memory, or paper-row provenance is claimed here.

## Reproducible checks run on this draft

From repository root:

```sh
python3 -m unittest discover -s zk-torch-4/scripts/bench/morpheus -p test_evidence_admission.py -v
python3 zk-torch-4/scripts/audit/test_semantics_reference.py
python3 zk-torch-4/scripts/audit/test_div_z_host.py
bash -n zk-torch-4/scripts/bench/morpheus/run_models.sh zk-torch-4/scripts/bench/morpheus/run_extra.sh
git diff --check
```

Results: 23 synthetic log-admission tests pass; seven independent scalar-model
checks pass; 10,048 boundary/random cases of host-extracted `div_z` and
`div_z_exact` agree with Python arbitrary-precision division (g++ C++17, UBSan).
Shell syntax and whitespace checks pass. Host extraction replaces CUDA qualifiers
and two integer intrinsics; it is neither nvcc compilation nor GPU execution.
Scalar-model checks do not invoke Rust or verify actual proofs.

No Rust compiler/Cargo, nvcc, CUDA toolkit/runtime or GPU was available in the
execution environment. Standard alternate Rust locations were also absent.
No packages were installed and no GPU job was submitted. Added Rust wire tests
are unexecuted. An independent source review caught and corrected a misplaced
link guard before publication; this is not a substitute for compilation.

## Required before merge / correctness claims

1. Build using the project's Rust/CUDA environment (`cd zk-torch-4 && cargo build
   --release`); run Rust unit tests and check external constructors for added
   `Instance.ar` and `Prepared.ar` fields. Audit all warnings/errors.
2. Execute full verifier tests rejecting absent auth at request AND finalization;
   malformed decoded and in-memory proof shapes must reject without panic.
3. Execute complete proof mutations for ScaleUp (x=0,S=2,y=1), altered selectors,
   descriptor-reuse, noncanonical padding, shard boundaries, negative/tied logits,
   changed bound constants and range roots. Scalar tests are insufficient.
4. Compile with nvcc and compare actual GPU div_z/sampler outputs to exact CPU
   references across boundaries and fallback paths; check sanitizer results.
5. Complete numerical/causal/no-wrap repairs, prove canonical program binding,
   then rerun source/mask/query census and authenticated performance workloads.
   Persist full accepted proof bytes, binary/source identities and run manifests.

## Subsequent fixed-point exp unit repair

The cubic's constant term was raw integer `1` tagged `sf=15`. `Witness::new`
does not scale its coefficients, so this represented `2^-15`, not one. It is
now raw `32768`. At input zero, the other cubic terms vanish and TwoPow(0)
is 32768; after the final sf30-to-output rescale, the old result was 0 at sf10
(or 1 at sf15), while the repaired result is 1024 (or 32768). This intentionally
changes numerical outputs; old benchmark measurements are not evidence for it.

`python3 zk-torch-4/scripts/audit/test_exp_unit_reference.py` passes two scalar
unit checks, including all output scales 1 through 29. The new Rust test
`exp_zero_is_one_at_output_scale_cpu_forward` executes the actual graph's CPU
operators without benchmark hooks/backend overrides, but remains UNEXECUTED
until the Rust/CUDA dependency build can be satisfied. This is a unit-value bug
repair only: it does not constrain ExpHelper's canonical remainder, pin arbitrary
private model constants, repair softmax normalization, or prove causal masking.

The library test build now ignores libtest filters/flags as YAML arguments;
production binaries still reject missing explicitly requested config files.
After supplying the real build dependencies, the focused check is:
`cd zk-torch-4 && cargo test --lib exp_zero_is_one_at_output_scale_cpu_forward`.
This command has NOT run in the current environment.

## Subsequent wire-v2 registered-model and statement repair

Base for this increment: published commit `7e16a0cdc74f3d8e03914a45a29529e543062f03`.
Wire magic changes to `MRPHSv2`; v1 proofs are intentionally incompatible. This is
one protocol-version change, not an additional parameter/security profile.

Both modes now isolate model sources from request sources and require a trusted
external `RegisteredModel` descriptor: commitments, canonical program identity,
and exact model-only source layout. Even model-free graphs need the descriptor,
with an empty commitments list.
The prover reuses the exact registered model payload **and hiding suffix**;
monolithic proof sources have a model prefix checked against that external list.
Fresh graph masks and request hiding suffixes are not registered or reused.
The local benchmark constructs its own registration only to demonstrate the API;
it does not authenticate the identity of an external model owner.

The request transcript now absorbs a typed, length-delimited full public
statement: operator kinds/parameters, topology/wiring, shape and padding metadata,
roles/types/scales, setup/key/profile/configuration, and public input/claimed
output arrays. Every encoded u64 is split into 32-bit limbs before field absorption.
No digest substitutes for that full prefix and no private tensor payload is
exported. Finalization binds the immutable program/type portion and verified
request queue digests. The convolution protocol choice is snapshotted in Graph,
not reread from a mutable environment during reduction. Diagnostic LINK_ROWS
proofs cannot be accepted by request/final verifiers. Empty final queues reject.
Opening IDs are bounded so request/final domains do not overlap.

**Still required:** the verifier must independently pin the intended DAG,
registered model commitments and public inputs/claimed outputs, and own its
verified deferred queue. This does not prove that a compiler is correct or fix
unconstrained numerical advice. Private Role::Input values omitted from public
still lack an independently expected private-input registration hook. Existing
model digit widths are witness-derived public registration metadata: their
leakage is explicit here, not a claim of fixed-bound zero knowledge. Replacing
that with public declared per-edge bounds remains separate work.

Rust registry-identity helper tests and Rust statement-codec tests are
added but UNEXECUTED. Full mono/deferred verifier tests must mutate expected model
identity, payload/suffix, missing/reordered prefix, graph/config/public values,
and queue identity; test an empty-model descriptor and stable model/fresh request
sources across requests. No successful proof, Rust build, CUDA run or new timing
measurement is claimed.

The registration fixes graph/configuration/shapes and public input/output edge
identities, but excludes request-specific public input/output/AR token values.
Those values are fully bound separately by each request transcript. New requests
of the same shape can reuse registration; dynamic shapes require a new registered
descriptor. Model source layout is separately compared by both verifier APIs.
Per-seed request/batch IDs must remain unique across calls; numeric domain bounds
do not by themselves prevent callers from deliberately reusing addressed coins.

Benchmark expectations/policy receipts now explicitly require `protocol=2`;
legacy/missing-version receipts are rejected rather than admitted as evidence
for the changed relation. This does not retroactively authenticate old logs.
The wire-v2 statement codec supports global `SF_LOG` from 0 through 30 and rejects
larger scales before reading the existing signed-shift-based `SF_FLOAT` helper;
this is an explicit supported-domain restriction, not a silent scale change.

## Subsequent wire-v3 private-input/lifecycle increment

The next draft closes the previously documented *expected private-input identity
hook* at source level through mandatory `InputReg` / `RegisteredInput`, separate
immutable input sources, and exact verifier prefix/descriptor checks. See
PRIVATE_INPUT_REGISTRATION.md for semantics, privacy, reuse, queue provenance and
remaining public-only verifier deployment limitations. This supersedes the earlier
statement that the request API has no private-input hook; it does not supersede
the outstanding normalization/compiler/causal/runtime-proof obligations.

Opening IDs are now consumed once per shared Coins backend; model registration
is once per backend and private-input registrations allocate distinct suffix
owners. Wire and benchmark receipts change to version 3. Old/missing-version
receipts do not become evidence for this changed relation. All Rust tests remain
unexecuted until the real Rust/CUDA build prerequisites are available.
