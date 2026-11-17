# Selected finite normalization and causal attention: scoped evidence

This separate 18-file increment follows InstanceNorm checkpoint
`8afbe14aec6c476221fe745cf898db7cc94ab47d` (tree
`f5c398236b3562f7155afa5156b951857121c390`). Delta-manifest SHA256:
`a74fdff49a60074fcc7de14ebc9439c457dac64bc78246f8bebb6d4db7c460b7`.
The frozen joint candidate has **450 required non-cache source files**; all 159
prior evidence files are retained separately. The source delta does not contain
later CNN-layout work.

## Results actually observed

On this final joint source, isolated modified-build all-target type checking
passed with only native CUDA build-script compilation/link directives omitted.
The original selected CPU test bodies passed as follows:

| Family | Invocations |
|---|---:|
| Six new selected-normalization/attention bodies × scales 0–30 | 186 |
| Fourteen inherited exp/basic-graph bodies × scales 0–30 | 434 |
| Nine inherited mean/RMS/CNN-mean bodies × scales 0–20 | 189 |
| Eight inherited InstanceNorm bodies × scales 0–20 | 168 |
| Original partial-lookup-chunk structure body | 1 |
| **Total** | **978** |

These are 978 targeted test-body invocations, **not** 978 distinct tests or an
ordinary full `cargo test` suite. Test bodies/implementations remain unchanged in
the disclosed standalone harness copies. Independent arithmetic checking passed
**1,351,857 assertions**; the author scalar checker separately reports 185,014.
The independent source review plus joint/Whisper addendum found no remaining
blocker in the defined scope after concrete source/test repairs. Neither is one
of the five complete independent paper acceptance reviews.

Unmodified production checking remains **blocked at missing nvcc**; actual PIOP
replay linking remains **blocked on unresolved native agl/CUDA symbols**, with
**zero replay bodies executed**. CPU mutation tests recompute original operators
and inspect emitted range/PZC predicates. The authored PIOP mutation changes
linked advice while other witnesses stay honest: it is a linkage test, not a
fully recomputed coherent forgery test, and remains unexecuted. No GPU stubs,
ignored linker symbols, full authenticated PCS proof, accuracy or timing pass
is supplied.

## Deliberate relation, finite approximation and source costs

Immutable public selected sets exclude future and padded coordinates **before
both maximum and denominator**, not merely after output. Private fixed-shape
one-hot advice is constrained by bitness, supported membership, row sum and
maximum dominance. The denominator sums selected finite exponent values only.
Exact bounded variable division uses independently bounded numerator,
denominator, quotient and the complete signed residual interval. Sigmoid uses
the same finite two-logit relation. Legacy free log advice/additive causal masks
fail closed.

The local ratio is formed from the existing finite cubic exp, including its zero
tail, **not real exp**. Before final rounding it preserves those finite ratios;
each normalized weight's rounding error is at most 1/(2Q). Exact row mass one,
unchanged logits, task accuracy and historical measurements are not promised.
Public residual-budget checks admit the documented SF16/Q20 profiles; they
reject a concrete SF30/context-9 modular counterfeit while still allowing some
smaller SF30 profiles. Larger unsupported budgets need linked wider arithmetic.

The maximum/ratio certificates add **14 NonNegative and 2 ProductZeroCheck
roots** beyond the existing exp recipe, plus real masking, reduction, broadcast
and product arithmetic. Both hints add fixed-shape O(padded scores) committed
entries: H uses one declared private bit plane, while ratio advice uses registered
request planes. Public 32-plane sources are complete for the new hints at supported
scales; default 16 planes is not complete at SF16. Sigmoid uses two finite-exp
evaluations per cell. These are source-level costs, not a proof-row count or GPU
benchmark. Context-4096 CPU normalization examples are single rows, not complete
32-head inference.

GPT/nanoGPT/Llama causal routing now uses exact support. Whisper decoder
self-attention is corrected to causal; its encoder remains bidirectional and
rectangular cross-attention remains noncausal over logical encoder keys. The
CPU regression compares both earlier decoder rows under a future-state change
and separately retains noncausal encoder/cross-attention dependency controls.
This is a substantive semantic change, not unchanged historical outputs.

## Lossless reproducible evidence transport

The canonical portable package has 172 files / 8,771,729 bytes. Its internal
`export-manifest.json` SHA256 is `144cae910c833066057e78c2f39a8b0a733f0cdac5e2095bcfa363974347364e`.
Oversized logs use UTF-8 line dictionaries plus the complete ordered index
sequence in `package/**/*.log.lines.json`. No warning, negative output, line
ending, trailing newline or original filename is removed from reconstruction.
All other files are stored unchanged. `transport-manifest.json` records original
and stored byte identities; the standard-library-only decoder verifies both and
refuses an existing output directory. No installation/build/test is performed
by decoding.

From this evidence directory:

```sh
python3 decode_bundle.py /path/to/new-selected-evidence
python3 /path/to/new-selected-evidence/verify_bundle.py --source /path/to/selected-checkout
```

Read the decoded README for exact fresh-copy reconstruction/harness commands.
The visible `package/README.md` is the canonical README, whose commands apply to
the **decoded** package. Prior checkpoint source manifests intentionally describe
older snapshots; use this package's 450-source manifest for this joint candidate.
All final raw success/failure logs, adaptations, test-body identities and scoped
independent reviews are reconstructible byte-for-byte.

## Still open

Whole-model lowering and arithmetic, unrelated generic ScaleDown/no-wrap,
first-argmax/generation correspondence, complete source/range/mask census,
checkpoint numerical/task accuracy and runtime measurements remain open. The
CNN row-major sigmoid/SiLU and other layout work is a separate increment; a
transformer column-major mask is not universal CNN-layout correctness. Local
selected-set noninterference is not a complete one-shot/incremental model theorem.
Actual native PIOP/full PCS/GPU execution still needs genuine authorized linkage
and hardware. This remains a draft, not merge approval or whole-paper acceptance.
