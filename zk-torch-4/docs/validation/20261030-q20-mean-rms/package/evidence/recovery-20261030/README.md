# Q20 recovery revalidation — 2026-10-30

No candidate or exp-base source was edited. This is a recovery/integrity and CPU
reexecution receipt, not a new independent S&P review or cryptographic acceptance.

## Frozen candidate

- Candidate: `<candidate>`
- Delta manifest SHA256: `60b90a1c996cdb0c86becb35005563e289b0e8c2abfbf68a331d530e3de70e3a`
- Full443-source manifest SHA256: `9119fd68440756e222fdbd8f1ff6814f9d1b51a9e74d241c1e160ea580c81ccb`
- Patch SHA256: `6c36b21a613c705de390ad6bc2a54f55cea15775e5de6d57bbabb43e529e2186`

`integrity.json` verifies all443 source hashes, all16 delta/base hashes, original
nine normalization test-body identities (visibility-only exposure in cpu4), and
binary hashes against the original audit. `git diff --check` returned0.
`adaptation-integrity.json` verifies every actual cpu4/replay4/typecheck4 delta
against the declared manifests:12/5/1 files respectively, zero mismatches.
`patch-replay.json` records a fresh isolated exp-clean copy receiving
`git apply --check` and `git apply`, both exit0, then all443 resulting hashes
matching the frozen candidate.

## Fresh executions

- Independent integer checker:127,930 groups PASS, s0..20 (`check-q20.log`).
- Existing verified cpu_normalization binary:21 invocations (ZK4_SF_LOG=0..20),
  nine original production bodies each,189 PASS lines, every process exit0
  (`cpu-scales-0-20.log`).
- Existing verified cpu_partial_chunks binary: original61/62/63-bit lookup chunk
  body PASS, process exit0 (`partial-chunks.log`).
- Terminal shell exit0 recorded by execution tool; human-readable completion in
  `terminal-results.txt`.

Original evidence additionally contains an all-target build-adapted Rust
check,434 exp/basic CPU test-body invocations, and a narrowly scoped independent
mathematical/source review. These are not claimed to have been rerun here.

## Unchanged remaining gates

Unmodified production compilation previously stopped at missing
`/usr/local/cuda/bin/nvcc`. Genuine PIOP replay build previously failed unresolved
actual native/CUDA linkage, with zero replay bodies executed. No installation,
GPU stubs, linker suppression, external publication, or retry of that unchanged
blocker was attempted in this recovery pass. Native proof acceptance, full
model accuracy and performance remain unestablished. This targeted repair is
not one of the five requested complete independent S&P acceptance reviews.

The established source review is `../independent/FINAL_Q20_REVIEW.md`; its
no-blocker verdict is limited to the bounded mean/RMS increment. Other model
semantics (InstanceNorm, Softmax/Sigmoid, attention causality, unrelated rescale,
full source/mask census) remain separate workstreams.


Export presentation note: private workspace path labels were normalized; mathematical and verification conclusions are unchanged. Original document SHA256s are recorded in `export-provenance.json` (relative to the evidence directory). Raw terminal logs and machine receipts remain byte-preserved.
