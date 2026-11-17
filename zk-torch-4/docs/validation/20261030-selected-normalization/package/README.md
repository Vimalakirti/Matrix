# Selected normalization + causal attention — frozen joint validation

## Identity and publication order

Final candidate: `/workspace/shared/morpheus-selected-normalization-joint`.
Base: frozen `/workspace/shared/morpheus-instance-norm-repair`, subsequently
published as `8afbe14aec6c476221fe745cf898db7cc94ab47d`. The18-file delta is
`selected-normalization-after-instance.patch`; do not apply it directly to Q20
without the earlier InstanceNorm patch. All450 required source files match the
full-source manifest after a fresh actual patch application. This local HEAD is
inherited and is NOT the identity of the uncommitted numerical candidate.

- Delta manifest SHA256: a74fdff49a60074fcc7de14ebc9439c457dac64bc78246f8bebb6d4db7c460b7
- Patch SHA256: 866617b806b622a81928e086adbfebf711a8ad96eb146de4453495649c1c4220
- Source export: `export/zk-torch-4/...` with `export/manifest.json`.
- The publisher owns the eventual remote commit receipt; no remote write was
  performed by the author of this increment.

## Actual repair

Private fixed-shape one-hot selection plus membership/dominance roots makes the
maximum canonical over an immutable selected set. Future and padded coordinates
are excluded BEFORE both max and denominator. Existing repaired finite E_s has
proved0<=E_s<=Qs and E_s(0)=Qs, including its explicit zero tail. Exact bounded
variable denominator division replaces free log advice, with full signed
residual no-wrap guards and deterministic nearest/upward ties. Sigmoid is the
same finite two-logit relation, not unconstrained softplus advice. Legacy helpers
reject at compile, graph-registration and statement-codec entrypoints.

All former SoftmaxConst callsites migrated. GPT2/J, nanoGPT and Llama2/3 retain
causal attention using exact support rather than additive-100. Whisper decoder
self-attention is newly corrected to causal; encoder remains bidirectional and
rectangular crossattention includes every logical encoder key. A real attention
CPU regression tests both earlier decoder rows across future-state changes and
keeps encoder/crossattention noncausal controls. Stale nanoGPT accuracy/equal-cost
comments are removed rather than silently transferred to the new relation.

The source document `docs/morpheus_selected_normalization.md` is normative for
finite equations, ranges, ties, zero policy, proof of E bounds, concrete SF30
wrap attack, practical SF16/Q20/context512/4096 budgets, source widths, and cost.
It preserves ratios only of selected finite exponent values before final rounding.
Each weight error versus that finite ratio is<=1/(2Qs); exact mass1 is not
promised. No real-softmax equivalence or historical task accuracy is claimed.
H adds one declared bit plane per padded score; ratio advice uses the registered
request planes. New max/ratio certificates add14 NonNegative and2 PZC nodes
beyond the existing exp recipe, plus actual mask/reduction/product/broadcast
arithmetic. This is not a one-proof-row-per-operator estimate or a GPU timing.
Context4096 CPU normalization cases are single rows, not full32-head inference.

## Executed final joint-source evidence

| Check | Actual result |
|---|---|
| All-target source/tests with only native-build omission |PASS, `joint-finaltypecheck.log` |
| Original6 new CPU bodies at scales0..30 |186 PASS invocations |
| Inherited14 exp/basic-graph bodies at scales0..30 |434 PASS invocations |
| Inherited9 mean/RMS bodies at scales0..20 |189 PASS invocations |
| Inherited8 InstanceNorm bodies at scales0..20 |168 PASS invocations |
| Original partial-width structure body |1 PASS invocation |
| Independent author's scalar relation tests |185014 assertions passed |
| Independent scoped reviewer arithmetic |1351857 assertions passed |
| Independent joint source/Whisper review |No remaining scoped blocker |
| Unmodified production `cargo check --locked --all-targets` |BLOCKED exit101: missing `/usr/local/cuda/bin/nvcc` |
| Real PIOP replay runner build |BLOCKED exit101: actual unresolved agl/CUDA symbols; ZERO bodies executed |
| Authenticated PCS proof, GPU runtime, model accuracy or benchmarks |NOT ESTABLISHED |

The978 CPU invocations are separated by family in `final-joint-cpu-summary.json`;
they are not an ordinary full cargo test pass. Negative cases recompute real CPU
operators and inspect actual emitted range/PZC predicates. Captured warnings and
expected caught panics remain in logs. PIOP source-mutation tests are authored
and typechecked: the new test changes linked advice while downstream values
stay honest, so it tests source/constraint linkage, not a fully recomputed
coherent forgery. It has NOT been executed under native proof linkage.

The independent review is targeted, not one of five whole-paper acceptances.
`test-body-identity.json` verifies inherited bodies unchanged and final CPU-copy
adaptations limited to test visibility. `final-cpu-binaries.sha256` and
`final-binary-link-audit.txt` preserve executable hashes and link dependencies.
No GPU symbols were stubbed, ignored or replaced with fake success.

## Reproduce without environment-specific script locations

Three adjacent scripts are bundled: `prepare_harness.py`,
`prepare_mean_rms_harness.py`, `prepare_base_harness.py`. The top wrapper uses only
its adjacent helpers; a fresh portability regeneration reproduced ALL final CPU
copy bytes. Use installed Rust/Cargo1.99.1 (the executed toolchain) and a fresh
destination. Original source is never edited by the validation scripts.

```sh
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/your/fresh/target
python prepare_harness.py /your/final/source /your/fresh/cpu-copy cpu-joint
cargo build --locked --manifest-path /your/fresh/cpu-copy/zk-torch-4/Cargo.toml \
  --bin cpu_softmax --bin cpu_instance_norm --bin cpu_normalization \
  --bin cpu_basicblock_exp --bin cpu_graph_exp --bin cpu_partial_chunks
for sf in $(seq 0 30); do
  for b in cpu_softmax cpu_basicblock_exp cpu_graph_exp; do
    ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/$b" || exit $?
  done
done
for sf in $(seq 0 20); do
  for b in cpu_normalization cpu_instance_norm; do
    ZK4_SF_LOG=$sf "$CARGO_TARGET_DIR/debug/$b" || exit $?
  done
done
"$CARGO_TARGET_DIR/debug/cpu_partial_chunks"
# Separate fresh copies: mode typecheck + cargo check --locked --all-targets;
# mode cpu-replay + build cpu_exp_replay (expected actual native-link blocker).
python reference.py
python independent-review/check.py /your/final/source/zk-torch-4
```

## Remaining gates

Full native proofs/GPU execution need authorized genuine CUDA linkage/hardware.
Whole-model finite arithmetic, unrelated generic ScaleDown bounds, task accuracy,
first-argmax/generation comparison and complete source/range/mask/runtime census
remain open. CNN row-major sigmoid/SiLU padding adaptation is being handled in a
separate layout increment; transformer column-major masks are not represented as
universal CNN-layout correctness. No approval-dependent install was attempted.

## Self-contained bundle verification and guarded reconstruction

The canonical `export-manifest.json` lives inside this bundle. Its `files` list
pins SHA256 and byte count of every other bundle file; the manifest intentionally
does not hash itself. `verify_bundle.py` checks the exact bundle inventory and,
with `--source`, all450 required final-source hashes. Additional prior validation
artifacts in a published repository are permitted by the source hash check.

```sh
python verify_bundle.py
python reconstruct.py /your/frozen/InstanceNorm/base /your/new/reconstructed/source
python verify_bundle.py --source /your/new/reconstructed/source
```

Reconstruction checks all446 expected base-source hashes before copying and
applies the actual18-file patch, then verifies all450 result hashes. Its resolved
DEST must be new and outside both BASE and the evidence bundle. The harness's
base preparer likewise rejects a resolved DEST equal to or inside SOURCE before
copying, including symlink-alias destinations. The included guard/roundtrip
receipt records executed safe reconstruction and direct/symlink unsafe cases.

This canonical internal manifest supersedes the earlier external
`portable-evidence-manifest.json` (SHA256d1e39c1e676e5c93b41e2924179e29a9abe694890524ebd966d6ea738790f411),
which described the first168-file export before these evidence-only wrappers.
Raw logs, production source, reviewed arithmetic and original test bodies did
not change. The previous source/patch manifest hashes remain valid.
