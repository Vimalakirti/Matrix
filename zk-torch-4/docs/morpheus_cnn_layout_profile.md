# CNN logical layout repair (not a full-model/proof acceptance)

This increment changes the registered computation. Old accuracy, timings,
source counts and proofs do not validate the corrected graph. The frozen Q20
base is `/workspace/shared/morpheus-mean-rms-q20`; selected finite sigmoid is a
separate prerequisite for the YOLO SiLU integration test.

## Exact layouts and public maps

CNN raw storage is row-major, last axis fastest, each axis padded to a power of
two. Einsum/Add interpret the first axis as fastest. Reversing shape metadata
with a same-arity ChangeShape therefore reinterprets the SAME polynomial without
transposing data. Shape changes, masks and gathers are registered graph nodes,
not private preprocessing. No shape or map depends on a tensor value.

A raw ChangeShape [16,5,5] -> [400] truncates 1024 physical entries to512; it does
not remove internal 8x8 spatial padding. `flatten_row_major` instead sets
N=product(logical dimensions), P=nextpow2(N). For each reversed axis a of logical
extent d_a and logical stride s_a, its immutable public factor is

    M_a(i,j) = [i<N] [j = floor(i/s_a) mod d_a].

The staged binary contractions `abc,za->zbc`, `zbc,zb->zc`, `zc,zc->z` select
exactly the row-major logical input for every i<N, and give zero for i>=N.
Each stage removes one original axis. Axes missing from an operand are free-once,
so the existing PIOP's point maps and legacy Einsum partial evaluation agree.
The first-axis-fastest interpretation is used consistently by the CPU operator
and the registered PIOP; no new unauthenticated mapping or CPU-only proof path
exists. All-power-of-two shapes use a same-arity view.

For LeNet the map shapes are [400,5], [400,5], [400,16], with padded row lengths
4096,4096,8192. These contain1200 one bits across16384 field slots. The three
sumcheck cubes have3,12,13 variables (rather than a four-factor19-variable cube).
Intermediate padded lengths are65536,8192,512; they are derived, not committed
advice. Numerical flatten is selection only and introduces no arithmetic wrap.

## Classifiers, biases and explicit batches

LeNet FC weights are explicitly COLUMN-MAJOR [out,in] = [120,400],[84,120],[10,84],
with weight(o,i) at o+nextpow2(out)*i. This does not claim raw ONNX row-major byte
compatibility; an importer must encode those bytes. The compact vector is masked
before every contraction and after each bias addition, so poisoned output/weight/
bias padding cannot enter the following120/84-wide contraction.

ResNet GAP reverses CNN axes before spatial mean. Its classifier retains public
logical B and validates logical features/B before viewing the result as [C,B].
Input/output masks exclude feature, class and B padding. Vector bias [classes]
is explicitly viewed as [classes,1] for batched Add. Both explicit [B,C,H,W]
and the documented folded representation are supported at the head; a folded
[Bp*Cp,H,W] tensor cannot reveal a previously discarded logical B.

`add_channel_bias_row_major` repairs the older direct Add([C,H,W],[C,1,1])
which put channel bias in low spatial bits. It uses [W,H,(D),C,B] views, accepts
[C], [C,1,..], full single-image row-major bias, or a full tensor bias matching x,
and applies an authenticated logical output mask. Call sites repaired: all
ResNet convolutions, LeNet, VGG, YOLO, PointPillar convolution helpers, and the
UNet3D final convolution. PointPillar's separate column-major VFE linear bias is
not changed. These call-site repairs do not certify the whole named networks.

Explicit [B,C,H,W] now survives pad/maxpool through registered mask/fold/restore
nodes. Padding is masked BEFORE advice/range predicates and again at the output;
the latter is necessary because honest zero-padding alone does not bind advice.
The three Conv2D builders derive B from axis0 (not B*C/Cp) and retain explicit
output shape with a real ChangeShape wrapper, leaving original Conv outputs,
self-claims and auxiliary edges intact. The streaming ResNet driver now declares
[B,3,H,W] rather than erasing B into Bp*4. Conv2D batch PIOP extension remains a
separate increment; this file does not claim that it was executed.

YOLO's two SiLU call sites use `silu_row_major`: reverse shape -> generic SiLU
-> restore shape. The finite sigmoid's input/output masks then address the true
CNN logical coordinates. Other sigmoid call sites are transformer column-major
paths and are unchanged. The Q20 base's legacy SigmoidConst is not repaired by
this wrapper alone; use the separate selected finite-normalization increment.

## Public does not mean zero commitment cost

Graph::new marks literal leaves committed. `tensor_specs` declares their width
from already-public values, and Walk also binds those values to the registered
program. Each 0/1 map/mask therefore occupies ONE raw bit-source coordinate per
padded field slot (even if its Witness data type is Float). The16384 flatten slots
are16384 committed PUBLIC bit coordinates, not free memory or a private model.

Relative to Q20's original single-image LeNet, the new binary maps add16384 bit
coordinates, convolution-bias masks add8192+4096, and six FC input/output masks
add512+128+128+128+128+16=1040: total29712 additional public bit coordinates.
No historical performance/source census may be reused unchanged.

The executable census prints every literal edge, shape, padded row length,
plane count, bit-typed flag and raw-coordinate count. With genuine CPU forwards
(all-zero representative weights/inputs only), public literals alone are:

| Representative graph | Literal leaves | Field slots | Raw coordinates | Public-only packed sources |
|---|---:|---:|---:|---:|
| Full LeNet-5 graph |33|91152|1642512|1|
| ResNet GAP/FC head B3,C2048,7x7,O1000 |13|1650688|36392960|1|

The ternary profile's source payload capacity is131497984 coordinates, source
cap262144, energy cap2^38. These public subsets fit one source and are below the
energy cap by capacity. THIS IS NOT a complete workload parameter-compatibility
or lifetime-cap certificate: model/request/advice digits, selector chunks,
graph/carry banks, partition duplication, all CNN body masks, grouped packing,
proof lifetime and rejection accounting still need the complete census. For
example a B3 ResNet stem [64,112,112] bias mask alone adds4*64*128*128=4194304
public bit coordinates. Public masks can materially change costs.

## Numerical/test bounds and validation limits

CPU layout regressions run the original production operator bodies, comparing
independently indexed logical references and checking emitted local range/PZC
predicates. They are not cryptographic acceptance/rejection. Exact signed nearest
rounding uses floor((a+floor(d/2))/d), ties toward positive infinity.

At SF0..20 the LeNet classifier test has initial |x|<=11Q, two nonzero weights
per output in {-Q,+Q}, and |bias|<=2Q. Successive logical magnitudes are <=24Q,
50Q,102Q; the largest raw dot-product bound is100Q^2<p/2. ResNet head test inputs
have |x|<=37Q+1 and at most5 unit-Q weights, giving <=185Q^2+5Q<p/2. Means stay
inside the existing2^30-1 mean-input domain. These are TEST bounds, not new
certificates for every FC rescale or arbitrary model weights.

Padding poison ±2^40/±2^60 tests algebraic mask independence, not admission into
an unchanged32-plane request encoding. Those magnitudes need an explicitly
public64-plane request profile if included in an authenticated source. No private
payload-derived width or shape has been introduced. The PIOP mutation tests use
an explicit64-plane configuration; they are authored and typechecked, but CUDA
linkage prevents execution in this environment.

Important remaining blockers: MaxPoolHelper/GeneralMaxPoolHelper's existing
constraints prove dominance only, not that the output is attained in the window.
An all-zero logical window with claimed maximum1 satisfies those old inequalities.
The output masks here repair padding only, NOT that logical-max proof defect.
A separate selected-element/product-zero repair is required. Batch Conv2D PIOP,
full authenticated PCS proofs, GPU execution, calibrated model accuracy, FC integer
soundness outside the stated local tests, other model lowerings (including VGG's
own flatten/batch heads), and complete source/mask/lifetime census remain open.
