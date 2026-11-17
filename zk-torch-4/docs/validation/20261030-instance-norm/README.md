# Bounded InstanceNorm3D: exact statistics and affine parameter binding

This is a separate increment after Q20 checkpoint
`1fee3d6017484c5bd23a7267d6603130cb70c50b` (tree
`0bbccd2f4ddde09a96603c2d43dc18b9fedf6ab0`). The ten-file source delta is
identified by SHA256
`789c2a5d0ae6c02e80fbff9f423cd8de27e1eb161f6ea08e444e4c4a4eb64b42`.
The frozen tested candidate has **446 required non-cache source files**. The
publication preserves all 109 evidence files from the preceding two checkpoints
and adds this evidence separately, without changing the tested source bytes.

## Observed validation and strict boundaries

- All-target Rust type checking passed only in an isolated copy with native
  CUDA build-script compilation/link directives omitted.
- Eight unchanged InstanceNorm CPU bodies at scales 0–20: **168 passes**.
- Nine inherited mean/RMS/CNN-mean bodies at scales 0–20: **189 passes**.
- Fourteen inherited exp/basic-graph bodies at scales 0–30: **434 passes** on
  this final source, with test-body identity and per-body pass counts retained.
- One original 61/62/63-bit partial-lookup-chunk structural body passed.
- Total: **792 targeted CPU invocations**, not 792 distinct tests or a full
  `cargo test` suite. Independent and portable 24-invocation repeats are not
  additional distinct coverage.
- Independent arithmetic checking passed **2,675,503 assertion groups**. The
  independent source review found and prompted concrete repairs and found no
  remaining blocker within the stated bounded increment. It is not one of the
  five complete independent paper acceptance reviews.
- Unmodified production checking remains **blocked at missing nvcc**. Genuine
  PIOP replay linking remains **blocked on actual native CUDA/agl_mpiop symbols**;
  **zero prover/verifier replay bodies executed**. Final failure logs remain.

CPU negative cases evaluate actual emitted arithmetic/range predicates, not
cryptographic verifier rejection. No GPU success stubs or ignored unresolved
symbols are supplied. No full authenticated PCS proof, GPU execution, complete
U-Net/full-volume inference, checkpoint accuracy or performance is established.

## Exact package bytes and lossless text transport

The canonical export has 47 files / 8,640,241 bytes; its original
`export-manifest.json` SHA256 is `8ac3db2904d3644bf01955ed2998c4c288932236622bf97462167c0ff0c0a714`.
`package/` stores its unmodified ordinary text files. Oversized logs are represented
by UTF-8 line dictionaries plus the **complete ordered index sequence** in
`*.log.lines.json`; no warning, negative output, line ending or trailing newline
is removed. All original filenames, byte lengths and SHA256 hashes remain in
`transport-manifest.json`, alongside encoded-storage identities.

The standard-library-only decoder checks every stored representation and every
reconstructed original size/hash, refuses an existing output directory and does
not install/build/run anything. Its output was compared byte-for-byte with the
canonical package, then verified using that package's own manifest verifier.
From this evidence directory:

```sh
python3 decode_bundle.py /path/to/new-instance-evidence
python3 /path/to/new-instance-evidence/verify_export.py --source /path/to/instance-checkout
```

Read the decoded `README.md` for portable source reconstruction, isolated harness
commands and final log identities. The visible `package/README.md` is the canonical
README: its verification/reproduction commands apply to the **decoded** package.
Use a checkout of the exact preceding Q20 commit as the reconstruction base;
additive evidence directories are preserved. Earlier 441/443-source manifests
identify their earlier frozen checkpoints, not the changed final 446-source tree.

## Deliberate finite numerical semantics

The named InstanceNorm3D builder now constructs constrained rounded logical
channel means, centered per-element square rounding, rounded variance, exact
integer epsilon, the bounded exact reciprocal, quantized affine scale,
beta-minus-scaled-mean offset, and affine output. Gamma/beta have actual proved
arithmetic consumers rather than detached advice-only inputs. Legacy unchecked
InstanceNorm helper nodes are rejected at compilation, graph registration and
statement encoding. Row-major CNN buffers are viewed with reversed shape metadata;
spatial/channel padding is excluded from statistics and zeroed at output.

The profile supports activation scales 0–20 and spatial count at most 2^24.
Only the generic constant-divisor cap widens to 2^24, with a separate residual
no-wrap proof; transformer mean/RMS row caps remain unchanged. Algebraic coverage
of 128³/256³ counts is not an executed full-volume memory/runtime claim.
IEEE epsilon conversion is exact integer rounding of the supplied finite binary
value. For f64 1e-5, epsilon units are 0/1/10 at scales 10/16/20; positive epsilon
can quantize to zero, without a hidden minimum-one replacement. Statistic zero
uses reciprocal zero. The operation order deliberately differs from the former
floating helper and from separately rounded centered-output normalization.

Public input/gain/bias bounds and 32-plane source requirements are explicit;
logical input magnitude is just under 8 at Q20 and 32 at SF16 for small epsilon.
This is not a claim that arbitrary checkpoints/activations fit the domain.
See `zk-torch-4/docs/morpheus_instance_norm_profile.md` for the exact equations,
source completeness and non-circular full-residual bounds.

Whole-model lowering/census, other advice/normalization, attention causality,
unrelated rescales, full proof composition, checkpoint comparisons and fresh
accuracy/performance remain open. Historical measurements cannot be relabelled
as this new program. The PR remains draft; no merge or acceptance is implied.
