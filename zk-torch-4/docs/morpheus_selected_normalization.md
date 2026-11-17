# Exact selected finite-exp normalization (new registered numerical program)

This increment replaces unconstrained `SoftmaxConst`/`SigmoidConst` log advice,
not real-valued softmax. It builds on the frozen Q0..Q20 mean/RMS candidate and
canonical unclipped exp profile. No historical model accuracy, logits, runtime,
GPU execution or authenticated proof is implied. Legacy log advice and additive
`causal_mask` now fail closed; named former call sites use this explicit relation.

## Public profile and mathematical relation

Let Q=2^s, 0<=s<=30. Exp internally retains Qe=32768 and its registered exact Ls,
selector width1..8 (default8), cubic constants, operation order, positive-infinity
half ties, and zero TwoPow tail for k>=16. No clipping or transcendental advice is
introduced. The same registered graph/profile must be used by prover/verifier;
a new numerical program has a different identity, never the historical one.

For each row, immutable public A_j in{0,1} selects logical keys; all key/group/
query padding is excluded. Every logical row has a selected key. Padded rows have
A=0. Selected scores have signed range[-B,B], B=2^54-1; excluded field residues
are multiplied by0 before numerical bounds and cannot cause earlier-row rejection.
Logical tensor shape, column-major layout and all exact A bits are frozen public
literals. A custom `selected_softmax` support must exclude padding and have a
nonempty support in every logical row. The causal constructor sets A_{i,j}=1
exactly for logical keys j<=i. This is replacement/exclusion, not an additive
large-negative sentinel.

A fixed-shape private dense hint H has entries0 or1, sum_j H_j=V (public row
validity), and H_j(1-A_j)=0. Define m=sum H_j*s_j. Prove

    -Bexp <= A_j*(s_j-m) <= 0,
    Bexp=2^b*Ls-floor(Ls/2)-1.

The selector therefore chooses an allowed maximum, not merely an upper bound;
ties may choose any maximum without changing m. The lower bound declares the
finite spread domain; out-of-domain rows reject rather than silently saturate.
For padded rows H=m=0. Set

    t_j=A_j*(s_j-m), e_j=A_j*E_s(t_j),
    D=sum_j e_j+(1-V)*Q,
    w_j=floor((2Q*e_j+D)/(2D)).

Every forbidden weight and every padded-row weight is exactly0. The denominator
for padded rows is the fixed public Q and numerators are0. Both canonical max
and denominator are independent of future/padded scores. Zeroing weights only
after a contaminated max or denominator does not establish this property.

Sigmoid is this same selected finite two-logit softmax on[0,x], returning its
x component. Input/output padding is masked. At x=0 its output is round(Q/2)
(ties toward positive infinity, including s0 where this equals1).

## Exp positivity is proved for the actual integer polynomial

Canonical remainder conversion gives re in[-16384,16384]. With RD nearest,

    a=RD(re^2,Qe), b6=RD(5461*re,Qe), c=16384+b6,
    z=RD(c*a,Qe), h=Qe+re+z.

Bounds are a in[0,8192], c in[13653,19115], z in[0,4779], hence
16384<=h<=53931<2Qe. For k=0, re<=0. Put u=-re. If u=0 then z=0.
For u>=1, c<=Qe/2, a<=u^2/Qe+1/2, and
z<=u^2/(2Qe)+3/4<=u/4+3/4<=u. Thus h<=Qe.
For k>=1, T[k]<=Qe/2 and T[k]*h<Qe^2 (or T[k]=0).
The final denominator is2^(30-s), so in every admitted case

    0<=E_s(t)<=Q, E_s(0)=Q.

These facts do not assume real exp, monotonicity or strictly positive tails.
The actual zero tail is retained. A selected maximum has t=0, proving D>=Q;
D<=KQ where K is the public maximum number of selected keys per row.
The exp interval roots plus normalization's e range roots enforce the needed
bounds in the actual graph; they are not host-only observed maxima.

## Exact variable division and no-wrap

Raw `NormalizedRatio` nodes also receive compile-time defining roots. Independently
prove e in[0,Q], D in[Q,Dmax=KQ], w in[0,Q]. Define derived field residuals

    rho=2*Q*e+D-2*D*w,
    tau=2D-1-rho.

Both residuals receive nonnegative lookup roots with width
k=bit_length(2Dmax-1), T=2^k-1. The public checked profile requires

    2T<p,
    2Q^2+Dmax<p,
    Dmax*(2Q-1)+T<p,
    p=18446744069414584289.

All host budget operations are checked. Widths>=62 reject. The integer expression
F=2Q*e+D-2D*w lies in[-Dmax(2Q-1),2Q^2+Dmax]. Therefore F-rho cannot equal a
nonzero multiple of p: the entire residual interval, not just the accepted output,
is bounded. Next rho+tau=2D-1 lifts since both sum bounds are<p. Consequently
0<=rho<2D and w is the unique nearest quotient with ties toward positive infinity.
The source quotient bound is essential; the positive lower denominator is also
proved. The denominator is a derived masked sum, not independently free advice.

Practical profiles (actual residual width / largest conservative residual):

| Scale | Maximum selected keys | Dmax | Bits | Envelope |
|---|---:|---:|---:|---:|
| SF16 |512|33554432|26|4398080065535|
| SF16 |4096|268435456|29|35184640524287|
| SF20 |512|536870912|30|1125900443713535|
| SF20 |4096|4294967296|33|9007203549708287|
| SF20 |1048576|1099511627776|41|2305844108725321727|

SF30 supports small profiles (K7 passes), not arbitrary contexts. Concrete wrap
regression: Q=2^30, eight allowed logits0 and one allowed -16Ls give eight e=Q,
one e=0, D=8Q. A false tail weight w=Q has rho mod p=4Q-31 and tau=12Q+30,
both within[0,2D). It passes naive bounded remainder/quotient tests but is wrong.
The public budget rejects this K9 profile; linked limbs would be needed to admit
it. Rejecting this profile is not evidence that all SF30 normalization is unsafe.

## Integer lifting of the max and support

Paired integer_interval roots force each H to0/1. The number of columns is<2^54,
so sum H=V lifts and gives exactly one allowed selected coordinate per active
row. H membership is enforced through actual ProductZeroCheck roots. Multiplying
by H or A is exact selection; m equals a selected signed score. The signed
selected-score interval has width2^55-2; paired range sums are<2^56<p. The
max-difference interval is within[-2B,2B], Bexp<2^40, so range representatives
cannot conceal wrap. Empty rows have no selected H and m0. All range nodes are
registered in dag.range; PZC outputs are self-claim roots.

## Ratios, rounding, and semantic limits

Before output rounding, normalized finite weights e_j/D preserve the ratios of
this canonical finite E_s, including its zero tail. No logarithm must be proved.
After rounding, |w_j/Q-e_j/D|<=1/(2Q). Thus row mass differs from Q by at most
K/2 integer units (and normalized L1 rounding error<=K/(2Q)). Exact row sumQ is
NOT promised or falsely constrained. Nor do two rounded weights preserve their
ratio exactly. Finite exp is not generally translation-invariant, so max must
be canonical even if denominator selection is already correct.

These statements do NOT prove real exp ratios, historical f64-log-helper outputs,
accuracy, unchanged logits/argmax/generation, or finite-profile equivalence to the
original benchmark. Required accuracy evaluation remains same checkpoint/input
corpus, pinned old/new profile at every used scale, node and mass errors, logits,
argmax/sequence disagreement, task accuracy, and fresh timing/source census.

Causal noninterference is an exact local theorem of these selected normalization
and weighted-sum relations. Full one-shot-versus-incremental model correspondence
still requires every other layer/layout/scale relation to be checked, including
known whole-graph ScaleDown/no-wrap obligations. This repair alone is not that
full-model theorem.

## Implementation, source and cost

`basicblock/normalization.rs` supplies only two hints: private one-hot H (same
full padded shape, no index-dependent layout) and w. `dag/softmax.rs` supplies
all defining roots and public support validation; the live compiler queue adds
guards even for raw hint nodes. Softmax/SigmoidConst raw nodes now reject.
`basicblock/mod.rs`, PIOP advice classification and statement tags44/45 register
these hints. All numerical bounds/masks/literals and entire operation graph are
bound by the public statement identity. The intended compiled DAG must still be
independently registered; arbitrary graph edits are not trusted model edits.

Both hints add O(number of padded scores) committed entries; H is declared private
bit-typed (one plane), with explicit paired0..1 roots retained. w needs at least s+1 signed digit planes in the worst case. Existing default
16-plane sources cannot represent Q at SF16: public req_planes_log5 (32 planes)
is source-complete for these new hints at all supported scales. All scalar domain
claims are intersected with the registered source width; never enlarge a source
layout depending on secret values. Public literals do not become private weights.

Beyond the existing exp recipe, max guards add six NonNegative roots (H, selected
score and canonical difference pairs), two PZC roots (support, row sum), and
linear selection/reduction arithmetic. Ratio guards add eight NonNegative roots
(three interval pairs plus rho/tau) and linear arithmetic with one variable
product D*w. The constructor adds selected-score/shift/exp masks, reductions,
public padded-row divisor and denominator broadcast. These are actual PIOP
sumchecks/source chunks, not a claim of one proof row per operator. No giant
exp, division or max lookup table is introduced; complexity is O(R*Kpad), with
public fixed shapes and widths. Sigmoid uses two finite exp evaluations per cell.

CPU tests directly evaluate original operators and emitted range/PZC predicates;
authored PIOP replay tests check actual reduction/link rows but are not PCS proof
acceptance. Report executed versus unexecuted stages separately.

## Whisper causal routing correction

The old shared Whisper self-attention constructor was noncausal for both encoder
and decoder. It now takes an explicit public construction choice: encoderfalse,
decodertrue. Decoder self-attention uses causal selected support; encoder
self-attention remains bidirectional. Crossattention retains all logical encoder
keys and independent decoder/encoder sequence dimensions (no triangular mask).
This repairs another substantive model-semantics defect, not an unchanged-output
claim. The actual attention CPU regression changes a future decoder hidden state
and checks both earlier rows unchanged, verifies encoder/crossattention still
depend on that later key, and inspects the exact authenticated masks for3x3 and
2x3 cases. Its values have independent raw-product envelope6Q^2<p/2 throughSF30,
avoiding the unrelated generic ScaleDown/no-wrap gap in this particular test.

The PIOP source-mutation test changes linked selector/ratio advice while keeping
downstream values honest; it tests source/constraint linkage, not a completely
recomputed coherent forgery. CPU negative tests separately recompute operators
and directly inspect emitted range/PZC predicates. Neither is a PCS proof receipt.
