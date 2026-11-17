# CNN logical layout: scoped evidence

This separate 14-file increment follows selected-normalization checkpoint
`517dbcf9fed670b4c7357a6ff265c8695a46df0a` (tree
`e9c6a0d36adf971e202e71529d7226bf3f1b5f86`). The frozen joint candidate has
**454 required non-cache source files**. All **334 prior evidence files** are
retained separately. This increment includes the explicit-batch **builder**
geometry/view changes, but does not include the separate Conv2D `cv.rs` proof
extension or a logical maxpool-attainment repair.

## Results actually observed

On this exact joint source, ten original CNN layout/census/finite-activation CPU
test bodies passed at every scale 0–20: **210 invocations**. The inherited tests
were rerun on the same source: 434 exp/basic-graph, 186 selected-normalization,
189 mean/RMS, 168 InstanceNorm, and one partial-lookup-chunk body. The combined
**1,188 targeted invocations** are neither 1,188 distinct tests nor a full
`cargo test` suite. Original implementation/test bodies are unchanged apart
from disclosed test visibility in disposable standalone harness copies.

The actual small B3 pad→plain/strided/dilated Conv→bias→pool chains pass their
logical references, including the production CPU convolution operators. Nonzero
LeNet flatten/classifier and ResNet head B1/3/5 cases check poisoned padding.
An all-zero full-LeNet graph smoke and representative public-literal census are
not trained-model accuracy validation. CNN SiLU tests use the already repaired
selected finite sigmoid, not legacy unchecked sigmoid advice.

Modified-build all-target typechecking passes with only CUDA build-script
compilation/link directives omitted. The unmodified production build remains
blocked at missing `nvcc`; genuine PIOP runner linking remains blocked on actual
native agl/CUDA symbols. **Zero PIOP replay bodies executed.** No GPU stub,
ignored linker symbol, full authenticated PCS proof, GPU result, accuracy or
performance pass is supplied.

Independent checks cover 158 flatten shapes, 428 non-Boolean MLE stage
identities, 36 bias shapes, 144 explicit-batch pad/pool layout cases, and a
separate 36-case/36,552-assertion frontend/view check. They are scoped
algebra/index/source reviews, not native proof execution or any of the five
complete independent paper acceptance reviews.

## Deliberate semantic changes and costs

A logical row-major flatten is now a staged authenticated public gather rather
than physical-buffer truncation. Registered equal-arity views align generic
column-major operations with CNN row-major channels. LeNet FC input/output masks,
ResNet GAP/head logical B and C checks, convolution-bias helpers, explicit-batch
pad/pool views and CNN SiLU routing preserve declared logical coordinates.
Streaming ResNet now carries explicit `[B,3,H,W]` metadata. These changes do not
certify whole-network import/lowering or raw ONNX weight-byte compatibility.

Public maps/masks are **committed public sources**, not free metadata. LeNet's
flatten factors occupy 16,384 one-bit coordinates, and its stated incremental
layout maps/masks add 29,712 coordinates relative to Q20. Executed public-literal
subsets total 1,642,512 raw coordinates for full LeNet and 36,392,960 for the
B3/C2048/7×7/O1000 ResNet head. These exclude the full model/request/advice/
selector/mask-bank/carry lifetime and do not certify whole-workload parameter
compatibility. Historical cost/timing rows must not be reused unchanged.

Padding poison ±2^40/±2^60 tests mask independence, not admission under a
32-plane request profile: authenticated sources containing those values require
an explicitly public 64-plane profile. Test FC bounds are documented separately
and do not establish arbitrary integer-rescale/no-wrap soundness.

## Lossless reproducible evidence transport

The canonical package has 273 files / 9,740,419 bytes.
Its internal `export-manifest.json` SHA256 is
`815ae4e95ece97b9a0d74d251bde4b8b77ad11d9d81b46684cb4e25f8e3256f7`.
Delta-manifest SHA256: `8babeddcf2d1dc8b01018bd701eeb9170547228b666d3aed05990482f8246abb`.

Oversized logs use UTF-8 line dictionaries plus the complete ordered index
sequence. No warning, negative output, line ending, trailing newline or original
filename is removed from reconstruction. All other files are stored unchanged.
`transport-manifest.json` records original and stored byte identities; the
standard-library-only decoder verifies both and refuses an existing output
directory. Decoding performs no installation, build or test.

```sh
python3 decode_bundle.py /path/to/new-cnn-layout-evidence
python3 /path/to/new-cnn-layout-evidence/verify_bundle.py --source /path/to/cnn-layout-checkout
```

Read the decoded README for safe fresh-copy reconstruction and harness commands.
The visible `package/README.md` is the canonical README; its commands apply to
the **decoded** package. Earlier evidence source inventories identify historical
checkpoints, not the current 454-source candidate.

## Still open

Logical maxpool currently proves dominance, not attainment: all-zero input with
claimed maximum one remains a substantive pre-existing proof counterexample.
The output masks in this increment repair padding only. Tensor-batched Conv2D
PIOP remains a separate proof-map extension; batching independent proof requests
is a different operation. No batch proof support is claimed here.

Native PIOP/full PCS/GPU execution, arbitrary FC no-wrap/rescale, all model
lowerings (including VGG's separate flatten/head), complete source/lifetime
census, trained-model accuracy and current-profile measurements remain open.
This is an authorized draft implementation increment, not merge approval or
whole-paper acceptance.
