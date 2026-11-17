# Exact bounded InstanceNorm3D (new finite numerical program)

This repairs the named `instancenorm3d` path used by 3D U-Net. It does **not**
rename the model or certify the old helper's outputs. The old packed scale/offset
helper had empty proof logic: proving only `y=scale*x+offset` left statistics and
both affine parameters unrelated to the output. The new path contains no such
helper. It emits ordinary arithmetic, canonical rescale lookups, exact constant
divisions and the already-guarded Q0..Q20 reciprocal. Raw legacy helper nodes
are rejected by compilation, PIOP Graph registration, and statement encoding.

## Public program, epsilon and order

Let Q=2^s, 0<=s<=20, K=2^28-1, and n=D*H*W in [1,2^24]. This includes SF16,
Q20, and spatial counts for 64³, 128³ and 256³. Supporting a count algebraically
is **not** an executed full-volume model, proof, memory or timing result.
The graph, logical dimensions, masks, bounds, integer epsilon and operation
order are immutable registered program data, not prover-selected constants.
`gamma[C]` and `beta[C]` remain model parameters, with real proved consumers.

`RD(a,d)=floor((a+floor(d/2))/d)` for d>=1: nearest integer, exact half ties toward
positive infinity, also for negative a. Per logical channel:

    mu = RD(sum_i x_i, n)
    d_i = x_i-mu
    z_i = RD(d_i*d_i, Q)
    v = RD(sum_i z_i, n)
    m = v+e
    r = nearest_nonnegative_integer(sqrt(Q^3/m)), ties upward, m=0 -> r=0
    a = RD(gamma*r, Q)
    t = RD(a*mu, Q)
    o = beta-t
    y_i = RD(a*x_i, Q)+o

This deliberately specifies per-element square rounding BEFORE the variance
mean. It is not `RD(sum d_i²,nQ)`, not a real-valued floating population variance,
and not the separately rounded centered-output program `RD(a*d_i,Q)+beta`.
Constant channels yield exactly beta, including the e=0 branch. Nonconstant
low-amplitude channels whose statistic rounds to zero also yield beta.

The explicit API `instancenorm3d_units(...,e)` accepts integer0<=e<=K. The retained
`instancenorm3d(...,eps:f64)` converts the exact nonnegative finite IEEE-754 value
`eps` to `e=round_ties_up(eps*Q)` using its mantissa/exponent bits and integer
arithmetic. No f64 multiplication, sqrt or ambient rounding mode defines the
relation. Negative/nonfinite/out-of-domain inputs reject; positive epsilon
may round to zero, **without** being silently replaced by one. For binary f64
`1e-5`, e is0 atSF10,1 atSF16 and10 atSF20. The registered semantic epsilon is
exactly e/Q. Distinct f64 arguments that quantize to the same e intentionally
select the same program. This is not a claim to preserve exact real epsilon.

## Domain and no-wrap argument

Use public G=2^24-1 and

    X = floor(sqrt((K-e)*Q))/2, rounded down again to an integer
    R = the independent global reciprocal bound from rms_bounds(s)
    A = ceil(G*R/Q)
    T = ceil(A*X/Q).

Logical inputs satisfy |x|<=X; logical gamma and beta satisfy |parameter|<=G.
The graph proves |mu|<=X, |d|<=2X, 0<=z,v<=K-e, 0<=m<=K, |a|<=A, |t|<=T,
|o|<=G+T, |RD(a*x,Q)|<=T and |y|<=G+2T. These are actual two-sided lookup
roots, not observed-witness maxima or clipping instructions. Out-of-domain
values reject. Full-domain input magnitude reaches just under8 real units at
Q20 and32 atSF16 for small e; the fixed G permits gain/bias just under16 real
units atQ20 and256 atSF16. These are finite profile restrictions, not a claim
that every trained checkpoint or activation satisfies them.

The field modulus is p=18446744069414584289. A two-sided interval[l,h] has
h-l<2^62; the two range representatives have total<2^63<p, uniquely lifting
the field value. Independently proved input bounds give |sum x|<2^47,
d²<=4X²<2^48, and sum z<2^52 for n<=2^24. Each is below p/2 and the division
input bound B=2^54-1. The integer reciprocal is the existing private32-bucket
construction, with no circular inference from its desired final bracket;
see `morpheus_mean_rms_profile.md` for its exact strict midpoint/zero rules.

For every rescale, raw product P and quotient bound Y are independent. Its
selector satisfies rho in[0,Q-1] and the field equation

    raw-Q*quotient+floor(Q/2)-rho = 0.

The WHOLE integer residual has magnitude at most P+QY+2Q. Code checks this
for squares, gamma*r, a*mu and a*x; the largest affine raw-product bound is
144115162440270832<2^57, with full residual<2^59<p. The independent maximum
T over s0..20 is274844336130<2^38. Offset/output sums and their interval widths
also remain far below p. Thus modular aliases cannot masquerade as rounded
integer products. Honest rounding always lies within these bounds, including
signed half ties. The final interval does not serve as a circular product bound.

### Shared constant-divisor extension

Only the generic DivConst divisor cap changes from2^20 to2^24; transformer
mean/RMS row caps and their local-output lemma remain unchanged. DivConst still
proves |a|<=B, |y|<=floor(B/c)+1 and rho=a-c*y+floor(c/2) in[0,c-1]. For all
1<=c<=2^24, c*|y|<=B+c and the full residual envelope
B+c*(floor(B/c)+1)+floor(c/2)+(c-1)<2^56<p. All new spatial means therefore
reuse a genuinely exact single division; they are not approximated by a
sequence of smaller means. Scalar high-divisor tests do not allocate volumes.

## Storage, parameters, source completeness and roots

CNNs store `[C,D,H,W]` row-major, while Einsum/Add/mask use column-major axes.
The builder reinterprets metadata as `[W,H,D,C]` without moving buffer elements,
then restores `[C,D,H,W]` at output. Logical spatial AND channel padding is
masked before summation, after centering, and at final output. Gamma/beta and
the reciprocal statistic have their own logical channel masks. Poisoned pads
cannot affect statistics; output padding is exactly zero. This also avoids the
old affine operation's disagreement with CNN channel layout.

Gamma/beta now flow through ordinary multiplications/subtractions and range
roots, so `prune_unclaimed` cannot discard them as advice-only sources. Parameter
identity is still authenticated by the existing model commitment/registered
statement machinery; a different parameter is a different model, not a permitted
prover choice inside a fixed model proof. Epsilon/masks/bounds are `program_literals`,
not mutable model weights. The graph is independently pinned as the intended
program, as elsewhere; this does not identify arbitrary arithmetic DAGs as U-Net.

No new dense advice primitive is introduced. Existing committed sources include
request x, mean/variance DivConst outputs, reciprocal and its quarter-division/
zero-flag/private-bucket advice. The affine ScaleDown quotients are derived
(not newly committed dense advice). Public `req_planes_log=5` (32 signed ternary
planes) suffices for the full declared logical numerical domain. Raw input
padding, although semantically ignored, must still fit its declared source
encoding. Model/public source widths follow the existing registration rules.
Default16 planes supports only the corresponding smaller source intersection;
no width is inferred from secret values or silently enlarged. Wide range roots
use existing partial-width sparse chunks; no dense enormous lookup is allocated.

## Evidence boundary and changed outputs

CPU tests execute real Rust graph operations and directly evaluate emitted
range/rescale predicates. They cover row-major non-square/non-power-of-two
shapes, poisoned padding, constant/zero channels, extremal domains, signed ties,
parameter dependence, invalid-domain rejection, and changed mean/variance/
reciprocal/affine quotients. PIOP replay tests with legal malicious selectors
are separately authored and typechecked; without real native linkage they are
**not executed cryptographic verifier receipts**. No CUDA/GPU success is stubbed.

This finite program changes real arithmetic and, for some layouts, previously
incorrect channel application. Example atSF10, e0, channel x=[0,1], gamma=Q,
beta0: mu1, each centered square rounds0, so m0 and the new output is[0,0];
this is not the old f64 normalizer. Same-checkpoint/input corpus comparisons,
per-node errors, logits/argmax/task accuracy, whole-model source/range/mask census,
full authenticated proofs, GPU/runtime/memory and historical benchmark transfer
remain unestablished. No accuracy or old-performance claim follows from these
local tests or the exact finite arithmetic proof.
