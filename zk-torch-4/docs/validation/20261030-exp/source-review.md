# Independent exp/literal integration review

Reviewed 2026-10-30 04:32–04:36 UTC. Candidate the frozen source candidate, based on published private-input/lifecycle commit `f192323f0725fed17a27dee21442fcba25460d2f`. Targeted integration/code review, **not one of the five whole-paper reviews**. No repository edits, remote writes, installation or GPU work by this reviewer.

## Disposition

The reviewed integration is coherent at source level, with both identified source issues repaired and rechecked. Suitable only for an explicitly unverified draft increment; Rust/CUDA compilation, real positive/negative proof verification and fresh cost census remain unexecuted. The independent scalar/math review and Python results are separate evidence, not compiler or GPU execution.

## Findings

1. **Rust test compile blocker found and repaired.** New `piop/exp_tests.rs` invoked trait method `BasicBlockType::run` without importing `BasicBlock`. Author added `use crate::basicblock::{BasicBlock, ExpHelper};`. Rechecked final import and call sites. This prevents the identified missing-trait resolution error; no compilation occurred.
2. **Non-Unicode environment override silently defaults — repaired.** Initial `exp_bits()` uses `Err(_) => K_BITS`, treating a non-Unicode ZK4_EXP_BITS as absence despite the stated fail-closed profile policy. Final code distinguishes NotPresent (default8) from NotUnicode (panic), and retains parse/range rejection. Rechecked the exact match arms. This is profile-selection robustness, not a demonstrated proof acceptance bypass.

## Integration correspondences checked

- Repository scan found one local Dag struct literal and one DagBuilder constructor/destructure; all receive the new literal/guard registries. Graph's sole constructor initializes frozen public literals and validates helper guard wiring. New crate-visible fields intentionally affect downstream manual struct construction; supported local construction uses DagBuilder.
- public_literal canonicalizes/stores values separately and makes the leaf Role::Input. Graph.public receives these values before the unchanged private-input preparation classifier runs. Therefore literal leaves are excluded from InputReg; learned param values remain private model commitments. Public roots bind actual witness values to frozen literal values even if a caller mutates a witness.
- Canonical statement codec version 2 encodes literal membership and **values** regardless of request-value inclusion, so model_words freezes literals while still excluding variable request values/private weights. It validates leaf roles/public-map equality and includes registered helper guards. Existing protocol-3 request/final prefixes bind these codec-2 words before challenges; an old codec-1 statement is not transcript-equivalent.
- Eager Add/Sub with request/literal dependencies now stay Role::Output rather than incorrectly becoming model parameters. Two Constant operands may retain trusted model-only preprocessing; code/tests/docs do not claim those precomputed outputs are proved functions of separately registered weight leaves.
- Compiler inserts domain/remainder range roots even for standalone ExpHelper. Graph/codec validation checks helper width, scale/shape, lookup widths, frozen bounds, subtraction wiring and noncommitted root/remainder edges. This is not a general canonical-compiler validator for arbitrary adversarial DAGs; the trusted-program boundary remains essential.
- exp builder performs checked supported logical-shape sizing, explicit pre-mask for padded inputs and post-mask for padded outputs using public shape-derived literals. It attaches the five bounded quotient intervals in addition to four helper roots. Numerical no-wrap/tie reasoning was independently reviewed by the scalar specialist; this review checked integration and API wiring rather than re-proving the exhaustive arithmetic analysis.
- Authored replay tests traverse actual CPU BasicBlock operators and actual PIOP prover/verifier reductions, then inspect emitted rows. They do not perform PCS/authentication and remain unexecuted. Test fixtures use appropriate layout APIs and inherited module imports after the trait fix. Shapes `[5]` and `[3,3]` exercise nontrivial padding.
- Documentation explicitly distinguishes logical-profile equivalence from changed padded outputs and historical four-bit clamping, and retains full normalization, causal denominator, RMS/mean, generic no-wrap, private-verifier deployment and compiler obligations. No full-model accuracy/performance claim is supported here.

## Checks actually run by this reviewer

```
git diff --check
python3 -m py_compile zk-torch-4/scripts/bench/morpheus/test_exp_profile.py
```

Both pass. Parent reports completed exhaustive Python logical domains SF10=181405 and SF15=5803172, endpoints/widths/padding cases, plus 24 parser, 7 semantic, 2 unit-reference and 10048 host-division cases. These reports are not mislabeled as Rust compilation or GPU execution; the parent holds their exact logs.

## Required before stronger acceptance

Compile the nine-file increment with supported Rust/CUDA dependencies, execute authored graph and PIOP tests, and run full verifier tampering tests for helper domains, compensated k/r, literal replacement, modular quotient counterfeits and padded claims. Re-run old/new full-model outputs and source/mask/query costs. The padding semantics intentionally change downstream values where old reductions consumed invalid pads; preserving historical accuracy/timings cannot be assumed.

Final scope check at 04:35 UTC: request.rs, coins.rs, driver.rs and wire.rs are byte-identical to the reviewed private-input/lifecycle candidate; only the declared nine checkpoint files change. Final diff check passes. No unresolved introduced integration/API blocker identified in this targeted review. Draft-only disposition remains; no Rust/GPU execution or merge approval is implied.

## Addendum — tenth-file baseline borrow repair (04:49 UTC)

Reviewed only the new `zk-torch-4/src/bin/morpheus_piop_e2e.rs` delta at lines 280–289, plus the called `PcsGpu::dev_sources` implementation (`pcs_gpu.rs:185–188`) and owned `DevSource` definition (`auth_gpu.rs:85`). Parent reports an actual all-target E0596 on the previous immutable receiver; that compiler log is retained by the Rust worker, not independently reproduced here.

The repair correctly uses `pcs.as_mut()` for `dev_sources(&mut self, ...)`, and moves the immutable `keys` borrow until after that mutable call. `dev_sources` transfers resident source buffers via `mem::take` into an owned `Vec<DevSource>`; neither its return type nor `DevSource` carries a borrow of `pcs`. Thus the temporary mutable borrow ends before the subsequent `pcs.as_ref()` key borrow. The `Some` branch does not mutate keys, and the `None` branch retains the owned fallback key lifetime through authentication. No later mutable access to `pcs` conflicts with the key reference. This preserves the intended one-shot consumption of resident physical columns; no transcript, parameter, numerical, or authentication-check change is introduced.

No targeted lifetime or semantic blocker found. This source-only approval covers the tenth-file compile repair; it does not assert the ongoing fresh-copy all-target check has passed or that CUDA execution occurred. The earlier nine-file-only scope statement applies to the preceding checkpoint, and is superseded by this explicitly reviewed tenth file. No source edits made by this reviewer.
