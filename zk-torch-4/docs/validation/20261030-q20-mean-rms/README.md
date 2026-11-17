# Bounded Q0–Q20 mean/RMS and CNN means: source and exact evidence

This is a separate increment after exp checkpoint
`e2d82ccfcc61bad6311fab053a146b7e68f7064e` (tree
`9b61e33bcf68774ae7c2306aa517650280add661`). The 16-file numerical repair is
identified by source-manifest SHA256
`60b90a1c996cdb0c86becb35005563e289b0e8c2abfbf68a331d530e3de70e3a`.
The frozen tested candidate contains **443 required non-cache source files**.
Publication adds evidence without changing those tested bytes or dropping the
preceding exp checkpoint's 32 evidence files.

## Observed results, not full proof acceptance

- Modified-build `cargo check --locked --all-targets`: passed in a disposable
  copy with CUDA build-script compilation/link directives omitted.
- Nine unchanged normalization CPU test bodies at scales 0–20: **189 passing
  invocations**. One original 61/62/63-bit partial-lookup-chunk structure test
  also passed; it does not run a cryptographic prover/verifier.
- Fourteen inherited exp/basic-graph bodies rerun on the final Q20 source at
  scales 0–30: **434 passing invocations**.
- Total: **624 targeted CPU invocations**, not 624 distinct tests and not a
  complete `cargo test` run. Recovery repeats are not added to this total.
- Independent arithmetic checker: **127,930 groups** passed. The independent
  targeted source review found and prompted repairs, including an actual
  ResNet channel-layout defect; it is not a whole-paper acceptance review.
- Unmodified production check remains **blocked at missing nvcc**. The genuine
  PIOP replay runner remains **blocked by unresolved native CUDA symbols**;
  **zero prover/verifier replay bodies executed**. All failure logs are retained.
- No GPU success stubs, ignored link failures, full authenticated PCS proofs,
  model accuracy, timings or complete inference correctness are claimed.

## Lossless text transport — no log lines removed

The canonical portable package is 74 files / 23,141,616 bytes. Its original
`export-manifest.json` SHA256 is
`90ec3b867e49734a89913da2d764d1d2f0d1dd073d73c5b3f065eeb7b9ccda49`.
Most bytes are repeated warnings from deliberately invalid CPU witnesses. Rather
than remove warnings or commit binary archives, oversized logs are stored as
UTF-8 line dictionaries plus the complete ordered line-index sequence in
`package/**/*.log.lines.json`. Exact line endings, final newlines, ordering,
byte lengths and original SHA256s are preserved. All other package files use
identity storage. The canonical export itself was not modified.

`transport-manifest.json` retains every original filename and hash and identifies
its stored representation. The standard-library-only decoder verifies encoded
storage hashes, reconstructs every byte, then verifies every original size/hash.
It refuses an existing output directory and performs no installation or tests.
The decoded package's own verifier independently checks its original manifest.

From this evidence directory:

```sh
python3 decode_bundle.py /path/to/new-q20-evidence
python3 /path/to/new-q20-evidence/evidence/verify_export.py --source /path/to/q20-checkout
```

Read the decoded `README.md` for complete portable reproduction commands and
`evidence/README.md` for detailed execution scope. The original source overlay,
patch, harness scripts, mathematical checkers, adaptations, audits, all 25 raw
original/recovery logs, failure receipts and independent review are included.
The visible `package/README.md` is the unmodified canonical document; its commands
are intended for the **decoded** directory, not the encoded transport tree.

For source reconstruction use a checkout of exactly the preceding exp checkpoint;
additive evidence directories are allowed and preserved. The old exp checkpoint's
441-source manifest intentionally identifies that older source, not this changed
Q20 source. Use this package's 443-source manifest for the final Q20 candidate.

## Deliberate numerical changes and remaining boundaries

The new exact signed mean rule uses Euclidean nearest rounding, half ties toward
positive infinity, logical padding and logical reduction counts. Bounded RMS
supports activation scales 0–20 (including SF16 and Q20), exact epsilon zero and
statistic-zero result zero. Its fixed-shape private 32-position one-hot dyadic
advice is independently constrained; bucket choice does not select public source
widths. Full stated RMS-domain completeness requires publicly declared 32-plane
request sources; default 16 planes gives a narrower intersection.

LeNet pooling uses exact four-gather means. ResNet GAP now interprets row-major
buffers correctly before reduction. These, rounding and quantized-statistic
changes alter numerical outputs; historical f64 accuracy/timings cannot be
relabelled as evidence for this program. See the production
`zk-torch-4/docs/morpheus_mean_rms_profile.md` for equations and limits.

InstanceNorm/UNet statistics, Softmax/Sigmoid, attention causality, unrelated
rescales, general compiler/lowering validation, full model/FC/batch semantics,
source/range/mask cost census and fresh accuracy/performance remain open. Scales
21+ require wider linked arithmetic. The trusted intended DAG must be pinned
independently. This draft is not merge approval or one of the five whole-paper
independent acceptance reviews.
