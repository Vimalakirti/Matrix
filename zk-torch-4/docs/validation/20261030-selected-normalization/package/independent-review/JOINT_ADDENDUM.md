# Joint-source / Whisper addendum

Reviewed `/workspace/shared/morpheus-selected-normalization-joint` after the
InstanceNorm overlay and Whisper call-site repair. This remains a targeted source
and arithmetic review, not a whole-paper acceptance or production-proof result.

## Integration result

No remaining integration blocker found.

- InstanceNorm implementation and test bodies are byte-identical to the frozen
  `morpheus-instance-norm-repair` versions.
- Exact DivConst's public cap2^24 remains intact; its numerator/quotient/residual
  roots are not overwritten by selected-normalization edits.
- The compiler's live guard queue retains both new selected hint branches,
  legacy SoftmaxConst/SigmoidConst rejection, legacy InstanceNorm rejection,
  DivConst, RMSReciprocal and RMSBucket defining guards.
- Graph construction and statement encoding reject both legacy advice families.
  New selected-normalization statement tags44/45 survive.
- Both implementation/test modules and both appended PIOP test bodies survive.
  The new bit-source classification is preserved.
- Normalization primitive and DAG source hashes exactly match the earlier
  independently checked snapshot; no arithmetic relation changed in the merge.

## Whisper semantics and regression

Encoder self-attention explicitly passes `causal=false`; decoder self-attention
passes `causal=true`. Cross-attention remains noncausal and uses its distinct
query/key logical extents. This prevents fixing decoder causality by incorrectly
making the encoder or rectangular cross-attention causal.

The sixth CPU body executes actual self/cross-attention operators with identity
weights, checks the immutable support arrays, compares BOTH earlier decoder
outputs across a later hidden-state mutation, and separately requires encoder
and cross-attention outputs to retain later-key dependency. It is more than an
inspection of final-zero weights.

Two candidate test perturbations were rejected during review:

1. Future3Q yields a later-row self-QK product9Q^2>p/2 at SF30, crossing the
   inherited centered integer lift even though first-row assertions may pass.
2. Future0 versusQ is within the raw-product bound but finite rounding yields
   the same first output50 at SF5; that particular `assert_ne` is flaky as a
   noncausal-dependency detector, not evidence of an incorrect relation.

The final test uses future-2Q versus2Q. It has max logical self-score magnitude
4Q^2 and a conservative6Q^2<p/2 raw attention bound asserted in the test, through
SF30. This perturbation retains a distinguishable noncausal output at all31
scales in the finite-profile precheck. The original fixed padding poison777
also remains far below the field centered threshold. The test's meaningful
negative control is therefore not relying on unrelated field wrap or an output
change that quantization can erase.

The full-attention regression is still a CPU operator/predicate test, not an
end-to-end proof of all generic ScaleDown bounds. Parent's fresh exact-source
execution/typecheck receipts must be reported separately; this review does not
claim their outcome in advance.

## Reviewed hashes (SHA256)

- basicblock/normalization.rs:
  b2aea13b3e878ec538053f888db32ae35c0b705478539fcddb85689a1769c68a
- dag/softmax.rs:
  25ab153de789d32698c9db4588ae9eced2b80f1f4f9ea76a16caf7f489ae0365
- dag/whisper.rs:
  5b62aa3ddb6141722d42ae378210199c6e58be7d6203a5b8bce7c14c428a7bb8
- dag/softmax_tests.rs:
  dac56f7535ac4757fde7ae4805ffd647c3e6f9738e459fed0d7a04280e88cf2e
- dag/instancenorm.rs:
  07d72d0841a73d056a0faeb82fe61aa11229a77d8bb8cfc557a668de85068d5a
- dag/builder.rs:
  ff0f9c2affc45b33ac92d3590a235a99dee612b115041a0cb64ca7ee4b34dfde
- morpheus/statement.rs:
  0187b23c6ae8c3ecef9e5affb8bef97d82fa4b75deafea7fa99f9bceb5c2aec7
- morpheus/piop/mod.rs:
  be8a94bfb75c54bc249956ccc3455bfaa0782c1387dd285a76773d17af114667
