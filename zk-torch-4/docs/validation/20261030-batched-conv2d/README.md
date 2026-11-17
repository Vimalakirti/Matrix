# Tensor-batched Conv2D: scoped source/field evidence

This separate **three-file** increment follows CNN layout checkpoint
`7241f377bebd47e911e4a587bdab1b0c7f89baf5` (tree
`a10cf80803f913bebea1f46eb8a4695a41691481`). The frozen candidate has
**455 required source files**; all **610 prior evidence paths** are preserved.
The already published CNN builder changes are not applied again. This is tensor
batching within a single Conv2D, not the deferred batching of independent proofs.

## What changed

Normal and polynomial-product (PP) reductions now bind logical B/C/D and spatial
padding consistently, including non-power-of-two batches. The kernel remains
shared rather than gaining a batch dimension. Masked terminal links use the
same public logical masks as the contractions. PP full coefficients are bound
without masking away malicious padded advice; direct nonnegative exponent
powers support alpha=0. The exact crop relation supports non-power-of-two
strides without assuming a bit-affine map.

FlattenKernel bypass now requires matching geometry and an equal-arity view
chain. Otherwise the normal path links the actual flat edge through a public
sparse tap map, retaining truncate/regrow effects. Checked geometry rejects
scatter collisions/overflow/out-of-bounds indices. A related source-registration
condition retains PP full output advice even when auxiliary outputs are present.

This is field convolution. Signed integer interpretation additionally requires
independently authenticated no-wrap bounds such as C*KH*KW*M_X*M_W < q/2.
Neither arbitrary trained weights nor complete model arithmetic is certified.

## Evidence actually obtained

Three original Conv CPU test bodies passed, containing 32 production-forward
cases, eight actual field-helper identity cases at non-Boolean extension-field
points, three kernel-source fixtures and two collision/overflow rejections.
All 210 inherited CNN layout/census/finite-activation body invocations passed
again at scales 0–20. These are **213 body invocations**, not 213 distinct tests
or a full cargo suite. Other earlier CPU families retain their checkpointed
results; no fresh execution on this Conv snapshot is inferred.

Modified-build all-target typechecking passes. Unmodified production checking
remains blocked at missing nvcc, and the genuine masked-PIOP replay executable
remains linker-blocked on actual agl/CUDA externs: **zero replay bodies executed**.
There are no GPU stubs, ignored unresolved symbols, PCS proof, GPU, calibrated
accuracy or performance passes.

Independent field/index checking passed 80 cases / 147,192 assertions plus 96
fallback/source-map cases. Additional deterministic mask-mutant checks expose
removed B/C/D masks, cross-batch cancellation attempts and padded full advice.
The independent review found and prompted concrete source-identity repairs;
its final reviewed hashes and 455-file overlay check are preserved. This is
scoped algebra/source review, not one of five complete independent acceptances.

## Cost and remaining gates

Batch support is not constant prover work. Normal batch folding reads
O(Bp*Cp*Hp*Wp) input cells; non-power-of-two B and D add masked-axis reductions.
PP commits Bp*Dp*Lp full coefficients and uses explicit exponent/crop maps. Its
local polynomial identity bound includes (Lp-1+log2(Bp)+log2(Dp))/|K| plus separate
sumcheck/authentication terms; it is not total protocol security. Historical
performance rows cannot be reused unchanged.

The inherited Conv3D, DepthwiseConv2D, ConvTranspose2D and ConvTranspose3D paths
still use the old producer bypass and have concrete truncate/regrow source-map
counterexamples. This increment's Conv2D review does not apply to them. Maxpool
attainment remains separately open. Complete model lowering, integer ranges,
source/lifetime census, real native PIOP/PCS execution and measurements remain
unestablished.

## Complete lossless evidence transport

The canonical package has 107 files / 748,607 bytes.
Internal `export-manifest.json` SHA256:
`661f53e0335c4fd3a77fd131ce45e65ed71c9c381c8e853a171b8fcbac8908a2`.

The canonical package omits duplicate full source trees; exact base/final
manifests and the three-file patch reconstruct them. All original success/failure
logs are retained. Oversized logs use complete ordered UTF-8 line dictionaries,
not shortened output. `transport-manifest.json` identifies original paths,
byte lengths and SHA256 hashes alongside stored representations. The
standard-library decoder validates exact reconstruction and refuses existing
or publication-internal destinations; it installs/builds/runs nothing.

```sh
python3 decode_bundle.py /path/to/new-conv-evidence
python3 /path/to/new-conv-evidence/verify_export.py /path/to/new-conv-evidence --source /path/to/conv-checkout
```

The visible `package/README.md` contains canonical reproduction commands for the
**decoded** bundle. Use the exact CNN base for reconstruction; prior additive
evidence does not change its canonical source identity. Earlier checkpoint
source manifests are historical, not claims about this source. This stays an
open draft; no merge, approval or whole-paper acceptance is implied.
