# Exact window MaxPool relation and validation limits

## Registered semantics

`maxpool2d` and `maxpool_general` now prove the **maximum**, not merely an upper
bound. Their output uses floor dimensions `(input-kernel)/stride+1`; only full
windows participate. Nonoverlapping pooling also accepts nondivisible spatial
sizes and drops the incomplete right/bottom window. Strides/kernel/extents must
be positive, a full window must exist, and padded size arithmetic is checked.
The old specialized `Replicate2x2` is not used for arbitrary pool sizes.

`maxpool_general_padded` implements symmetric implicit-negative-infinity padding
by **public valid-window exclusion**. It uses neither a finite sentinel nor a
trusted zero-valued boundary. Every logical output window must contain at least
one real input; empty windows are rejected. The output remains finite and
attainable by a real selected input. YOLO's actual `sppf_pool` stage uses this
API for all three k5/s1/p2 pools, fixing all-negative SiLU border windows.

Conversely, an explicit `pad`/`pad_asym` followed by ordinary pooling deliberately
includes those zeros as logical input values. This is a distinct operation.
Per-axis storage padding is never selected. Explicit `[B,C,H,W]` input/output
masks preserve nonpower batch/channel semantics; the folded `[C,H,W]` interface
continues to treat every declared channel as logical. Pad metadata preserves
input fixed-point scale. A computed output view gives a standalone wrapper a
public PIOP terminal root rather than returning only consumed committed advice.

## Constraints and source classification

Compilation attaches constraints to **all** `MaxPoolHelper`,
`GeneralMaxPoolHelper`, and `MaskedMaxPoolHelper` nodes, including raw
`add_gkr_node` users. A new `MaxPoolSelector` receives the defining constraints
even when created directly. Its inputs are x, claimed y, and immutable public
support A; a private/request mask is rejected. The support literal and extraction
geometry are included in registered statement data (operator tags 46 and 47).

Let R be padded output size, K kernel area, and Kp=next_power_of_two(K). H has
shape `[R,Kp]`, with output positions contiguous. For each output position:

* `H*(1-H)=0`, `H*(1-A)=0`;
* `sum H=V`, where V is the immutable logical-output indicator;
* `y=sum H*x_k` (attainment; padded output is therefore zero);
* each supported input has a canonical signed field lift in `[-B,B]`;
* each supported `y-x_k` has a 55-bit nonnegative certificate.

H is a committed **bit-typed** request source (`p=0`, one digit plane), not a
private model weight. Max outputs remain ordinary committed request advice.
The selector is not exposed; any tied maximum is allowed. The honest hint picks
the first tie, but canonical tie-index selection is not a relation claim.
Support slices mask inputs **before** signed interval/order predicates, so
excluded values do not restrict the numeric domain. Every arithmetic relation
above is a PIOP-consumed Einsum/Sub/Add/PZC/range relation, not an advice check.

This is a compiler fix under the existing verifier-trusted/registered-DAG
assumption. Manually removing guards makes a different graph/model; there is
no new independent pool-guard structural registry analogous to the exp registry.
The statement codec is not itself proof that an arbitrary graph implements a
user's intended model.

## Integer lifting and source-plane limits

The field prime is p=18,446,744,069,414,584,289 and
B=2^54-1=18,014,398,509,481,983. Each input interval uses paired 55-bit roots
u=x+B and v=B-x. Each root is at most T=2^55-1; the entire sum envelope
2T=72,057,594,037,927,934 is less than p. Since u+v=2B in the field, it is the
same equality over integers, forcing the intended signed field lift.

Boolean H and `Kp<p` prevent a wrapped row count. One-hot attainment makes y
one bounded input, not a large sum. Thus every supported y-x is in [-2B,2B].
A negative difference has least field residue at least
p-2B=18,410,715,272,395,620,323, greater than T, and fails the dominance root.
With PIOP tcl=4 a 55-bit root is exactly 13 four-bit chunks plus one three-bit
chunk; weighted recomposition spans [0,2^55-1], not an extra rounded 56 bits.

These bounds concern canonical signed **field lifts**, not uniqueness of raw
ternary digit encodings. Full-domain honest source completeness needs at least
54 source planes; the supported power-of-two profile `req_planes_log=6` gives
64. Profiles with 16 or 32 planes support only their representable subset.
64-plane raw digit expansions can alias modulo p; no new canonical digit
normalization claim is made. CPU poison tests are not proof that smaller source
profiles accept 60-bit padding. Computed 55-bit interval residues are range
roots, not 55-bit private dense source tensors.

## Representative actual graph census

For ordinary `[1,3,5]` input, k2x3/s1x2 (K=6), R=4, Kp=8:

| Item | Count |
|---|---:|
| Graph nodes / edges | 99 / 126 |
| Private dense sources | 3: input x, max y, Boolean H |
| H cells | 32 |
| 55-bit range roots | 18 (3K) |
| PZC roots | 4 |
| Registered public literals / cells | 26 / 240 |
| Committed Bool-declared edges, including public literals | 9 |
| Canonical model statement words in toy census | 10,780 |

The graph census is executed by the CPU tests; it is not a source-packing or
cryptographic runtime measurement. At req_planes_log=6, private dense tensor
payload is `32*64+4*64+32=2,336` raw digits. At tcl=4 the lookup-selector payload
is 252 Shout chunk-selector tensors and `18*4*(13*16+8)=15,552` bits,
before public sources, padding/alignment, W_G,
PCS source packing and transcript masks. This cost is substantial and must not
be advertised as the old dominance-only cost. No end-to-end latency or memory
claim is established here.

## Evidence and unexecuted stages

On the final combined CNN + batched-Conv + MaxPool source, nine original CPU
operator/emitted-constraint bodies pass at every SF_LOG=0..30 (279 invocations).
The inherited 978 numerical bodies and 210 CNN bodies also pass, plus the three
Conv forward/helper/source-fixture bodies (32 forward cases, 8 field-helper cases,
and source/overflow rejection checks): **1,470 CPU body invocations total**.
An independent reviewer additionally reran the earlier unchanged MaxPool binary
at SF30 and all nine passed. Covered attacks include the original all-zero
forged-y=1, simultaneous nonmax output/selection, zero/two/nonBoolean selectors,
storage/kernel padding, raw public support with +/-2^60 excluded values, private
mask rejection, signed boundaries, invalid/empty geometry, B3C3 odd shapes,
overlap/stride, public output roots and actual three-stage YOLO SPPF negative
borders. Pad and pad_asym compile-time/runtime scale metadata are exercised.

These tests execute original CPU operators and evaluate the **actual emitted**
NonNegative/PZC predicates. They are not PIOP verifier or PCS proof receipts.
Two authored PIOP replay tests include positive runs, source/output/public-mask
mutations, a fixed honest bit layout for nonBoolean mutation (not a host typing
assertion), and a forged-y=1 graph with every downstream witness recomputed.
Their execution status must be reported separately: native CUDA linkage remains
required by the repository and is unavailable in this environment. No CUDA
installation, EULA acceptance, FFI stubs, or GPU-result substitution was used.

Harness scripts preserve original test/operator bodies and record adaptation
hashes. They omit native build/link declarations only in isolated validation
copies; any reachable missing CUDA symbol still fails at link time. A first
census attempt retaining a GPU-cache destructor failed at `cuda_free`; the CPU
census was narrowed to Graph/source metadata rather than hiding that symbol.

## Follow-on PillarMaxPool correction

PillarMaxPool uses the same exact relation over [N,T,D] with a T-by-1 kernel,
then returns [N,D] without changing logical point membership. All declared
points are real members, including explicit zeros. Storage point/channel/feature
padding is excluded. Its raw helper receives compilation-time selector and
attainment constraints; its wrapper preserves fixed-point scale and returns a
computed terminal view. Numeric maxima use ordinary request source planes, not
the erroneous old one-bit classification. The private H selector remains one-bit.
The same signed-lift bound and source-plane completeness limits apply.

This does not repair ScatterToBEV or GatherFromGrid, certify PointPillar as a
whole, or claim native PIOP/PCS execution. Its new authored replay is separately
unexecuted pending native linkage; CPU emitted-constraint and source-type checks
are recorded in the follow-on evidence rather than added to older counts.
