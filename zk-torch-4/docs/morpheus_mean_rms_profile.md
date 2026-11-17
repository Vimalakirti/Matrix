# Exact bounded mean and RMS profile (new numerical program)

This increment follows the frozen exp candidate, not the old benchmark numerical
relation. It makes all `DivConst` and mean-input `RMSReciprocal(dim=0)` nodes
created by `DagBuilder::compile` acquire real arithmetic/range roots. Legacy
row-f64 `RMSReciprocal(dim>0)` graphs fail compilation. There is no optional
weak tolerance mode: `ZK4_EXACT_RMS` and `WHISPER_RECIP_TOL` no longer select the
named models' semantics. This is not an end-to-end inference or performance claim.

## Scope and exact public rules

Activation scale is **Qs=2^s**, separate from exp's internal Qe=2^15.
Scalar RMS accepts s=0..20, including the reported SF16 and configurable Q20;
s=21..30 explicitly rejects pending wider linked arithmetic. Mean division is scale-independent. Epsilon in the current named
transformer paths was absent and remains **exactly zero**; no positive epsilon
is silently inserted. A zero statistic yields r=0. This is a deterministic
extension, not IEEE division by zero and not standard positive-epsilon RMSNorm.

* `div_const(a,c)` returns floor((a+floor(c/2))/c), with Euclidean division.
  Thus nearest ties go toward positive infinity, for negative inputs too.
* Logical means first authenticate/mask padding, then sum logical coordinates,
  and divide by the actual logical count (not its next power of two).
* RMS squares each logical input and rounds each square by Qs, then rounds the
  logical sum by the logical last-axis count. Its resulting integer statistic
  is m. No real-valued mean or variance is substituted.
* r is the nearest nonnegative integer sqrt(Qs^3/m), ties upward, with m=0 ->0.
* Gain/output order remains h=RoundDiv(x*r,Qs), out=RoundDiv(h*gain,Qs).
  Both local rescale quotients and the gain are now explicitly ranged.
* LayerNorm first computes its rounded logical mean, subtracts it, masks the
  centered tensor, and invokes the same RMS path, then adds its bias.

Covered current call sites: `gpt2_layer_norm`, `gptj_layer_norm`,
`bert_layer_norm`, `whisper_layer_norm`/`whisper_rms_norm`, `llama_rms_norm`,
`llama3_rms_norm`, generic `reduce_mean` (including ResNet global average pooling), and LeNet
2x2 average pooling. Thus shared transformer families
and all seven former duplicated inclusive-mean blocks are covered, rather than
one toy site. ResNet CNN buffers are row-major, unlike Einsum: its head now
reinterprets [C,H,W] as [W,H,C] (and explicit[B,C,H,W] as[W,H,C,B]) before
spatial reduction. This repairs a genuine channel-mixing error, not merely an
axis-label change. For row-major[2,2,2] values[0,1,2,3,100,101,102,103],
the correct outputs are[2,102]; the old column-major reduction gave[51,52].
This substantive output change also requires fresh accuracy evidence.
LeNet pooling uses four explicitly sized gathers and exact
division by4; its authenticated mask uses the CNN ROW-MAJOR layout and covers
logical channel/spatial padding and explicit4D batch padding. A folded3D
[b_pad*c_pad,H,W] shape has lost its original logical batch count: as in the
previous convolution convention, all represented folded batches are logical.
No hidden original batch count is inferred or claimed to be masked. Its sum bound is4*(2^30-1)<2^32.
Explicit output sizes avoid the old offset-one gather truncation; no mutable
quarter-kernel model parameter remains. For s>=2 this preserves the old exact
quarter-kernel arithmetic on valid nonwrapping inputs; s0/1 gains defined mean
semantics instead of the old shift-underflow failure. Other LeNet quadratic/conv
operations are not certified by repairing this mean. InstanceNorm3D/UNet's unrelated affine advice, SoftmaxConst,
SigmoidConst, whole-graph ScaleDown, attention causality, and full-model numerical
translation are NOT certified by this increment.

## Enforced domains and source encoding

The actual field modulus is p=18446744069414584289. `integer_interval(v,l,h)`
adds two real NonNegative roots on v-l and h-v, both with
k=max(1,bit_length(h-l)). Require h-l<2^62. Their integer sum is at most
2(2^k-1)<2^63<p, hence equals h-l, proving the interval non-circularly.
All constants use `program_literals`, not mutable `Role::Constant` weights.

Division: 1<=c<=2^24 (extended by the InstanceNorm increment), |a|<=B=2^54-1, |y|<=floor(B/c)+1.
Mean inputs: |x|<=2^30-1 and reduction count<=2^20; padded coordinates are
masked before summation. Hence every sum has magnitude<2^50<p/2. Checked
axis-count multiplication rejects overflow and duplicate ReduceMean axes.

RMS: M=2^28-1, 0<=m<=M. Define integer public bounds
R=max_j R_j and U=max_j U_j from the fixed dyadic table below.
Input squares additionally enforce |x|<=floor(sqrt(M*Qs)) and square-rounding
outputs in [0,M]. Each raw square<=M*Qs<2^48; the sum of <=2^20 such rounded
squares is<2^48. The square's bounded ScaleDown quotient and bounded selector
therefore lift to its unique integer rounding equation. This is a local repair,
not an assumption that arbitrary field multiplication is integer multiplication.

The source codec uses ternary signed digit planes: value=sum(2^b*d_b),
d_b in {-1,0,1}. Default `req_planes_log=4` means16 planes and magnitude at most
65535, NOT a sign bit plus15 magnitude bits. **Full RMS-domain completeness
requires public req_planes_log=5 (32 planes) at every scale**, since m itself is
committed DivConst advice, not merely because SF15 r can exceed65535. Default16
retains a smaller declared-source intersection. The larger general DivConst
numerical bound does not promise all its values fit the default source layout.
No request widths are inferred from secret values or silently enlarged. All
finite-domain claims are intersected with declared source bounds. Public literal
widths remain public, and u/t/bracket intermediates are derived arithmetic,
not newly committed wide advice.

Wide NonNegative witnesses require the existing split sparse representation
when row bits plus table bits do not fit a machine index. The constructor
explicitly rejects the incompatible `NO_SPARSE_SPLIT`/table-width configuration;
there is no enormous 2^62 dense table allocation.

## Defining equations and integer lifting

Division enforces rho=a-c*y+floor(c/2) in [0,c-1]. Thus
c*y-floor(c/2)<=a<c*y+c-floor(c/2), selecting exactly one y. Independent bounds
make the entire residual equation magnitude<2^56<p, excluding modular aliases.
In particular c=2,a=1 accepts only y=1, and a=-1 accepts only y=0.

For RMS, use C=Qs^3 and fixed private one-hot bucket advice with32 positions:
zero [0,0], the28 intervals [L_j,H_j]=[2^j,2^(j+1)-1], and3 forbidden
padding positions [-1,-1]. Every row, including zero/padding rows, has exactly
the same public source shape and widths. Prove all32 flags in [0,1], their
row sum exactly1, and selected L<=m<=H. Since m>=0, the three forbidden
positions cannot be selected. The selector is private committed bit advice,
not a public group id or a secret-dependent source shape.

For each positive bucket use authenticated public bounds

    R_j=floor(sqrt(C/L_j))+1
    U_j=floor(sqrt(C*H_j))+1+ceil(H_j/2).

The zero bucket has R=U=0. Prove independently 0<=r<=R_selected,
u=m*r, and 0<=u<=U_selected (with public global caps too), before t=u*r.
The first product lifts globally: M*Rmax=288230375346405375<2^59 at SF20.
Once one-hot selection and its input bounds are established, the second
product satisfies t<=R_j*U_j in that unique bucket. This is a conditional
integer lifting argument, NOT an assumed unproved correlation between m/r/u
and NOT an inference from the final reciprocal bracket.

Use z=[m=0], b=[r=0]. ReLUHelper(v-1) only supplies a hint; each flag is defined
by 0<=flag<=1, flag*v=0, and1<=v+flag<=max(M,Rmax)+1. Enforce z*r=0 too.
Define one extra exact division h=RoundDiv(m-3,4)=floor((m-1)/4).
Then ceil(m/4)=h+1 even when m=0. The live compile-time guard queue ensures
this newly created DivConst and the newly created RMSBucket are also defined.
Require

    0 <= (1-b)*(C-t+u-h-1) <= Dmax
    0 <= (1-z)*(t+u+h-C)   <= Dmax
    Dmax=max(C+Umax+1, max_j(R_j*U_j+U_j+floor(H_j/4))) <2^61.

At SF20, Rmax=1073741825, Umax=17592320229376, and
Dmax=1630510897461400881<2^61. Both signed residuals have magnitude<2^61;
the residual-plus-range-representative envelope is<2^62<p. The PIOP uses
`min(tcl,remaining_bits)` for each lookup chunk, so a61-bit range ends with
one actual bit at tcl4; it is not silently widened to64 bits. Actual table
sizes, source regions, and reconstruction shifts use those partial widths.

For m=0, z=1 forces r=0. For positive m and r=0, the upper bracket becomes
floor((m-1)/4)>=C, i.e. m>4C, exactly the valid nearest rounding case.
Otherwise the reduced brackets are exactly equivalent to
m(2r-1)^2<=4C<m(2r+1)^2. The upper endpoint is strict, so ties select the
larger reciprocal. Honest mr<=sqrt(m*C)+m/2<=U_j proves completeness of the
selected U bound independently. Witness generation binary-searches the exact
integer predicate, never f64. All bucket constants/budgets are computed with
checked supported-scale i128 arithmetic.

The local output uses Hmax=min(ceil(Xmax*Rmax/Qs),2^31-1), gain bound2^30-1,
and output bound ceil(Hmax*(2^30-1)/Qs). The cap2^31-1 is complete, not an
unadvertised accuracy restriction: zi=RD(xi²/Qs)>=0, S=sum zi, m=RD(S,n)
give xi²<Qs*m*(3n/2+1/2) when m>=1. Since r<=sqrt(Qs³/m)+1/2,

    |xi*r/Qs| < Qs*sqrt(3n/2+1/2)+|xi|/(2Qs)
               <=2^20*1255+8192.

After nearest rounding, a conservative integer cap1315971073<2^31-1
suffices, using n<=2^20 and |xi|<=sqrt(M*Qs). If m=0, r=0 and h=0.
Thus the final gain product is<2^61, and its full rounded residual is<2^62<p.
The first x*r product is globally<2^55, even without bucket correlation.
These local quotient bounds exclude modular aliases in the RMS output path.

## Authenticity and remaining verification boundary

A live compile-time scan attaches guards even to raw `add_gkr_node` division,
mean-reciprocal and bucket nodes, including advice newly introduced by guards. Every range node is included in `dag.range`, consumed by
the existing PIOP lookup walk; every arithmetic operand is traversed rather than
silently treated as model precomputation. The full node graph, literal arrays,
range-node list, model identity and layout are included in the registered public
statement. As elsewhere in the repository, the intended DAG must be independently
pinned: this is not a claim that an arbitrarily edited post-compile DAG describes
a trusted model. The special exp-only structural validator remains exp-only.

CPU forward/range-predicate tests are not cryptographic verifier execution.
The PIOP replay tests in `morpheus/piop/exp_tests.rs` include the old mean ties,
negative reciprocal, zero, neighboring-value and private-bucket mutants, but require genuine
native linkage to execute. No missing GPU operation may be stubbed into success.

## Numerical changes and required evidence

Old DivConst used f64 rounding away from zero: negative exact half ties change;
large integers also no longer suffer f64 precision loss. The old default RMS
helper computed unrounded real squares/mean and used a nonunique tolerance gate.
The new program uses its explicitly rounded in-circuit statistic instead. At
SF10, row [23,23] has rounded squares1, mean1, r32768 and unit-gain output736;
the old real-statistic helper rounds r to45590 and gives output1024. This is a
material change, not an assertion that old helper outputs were proved correct.
Very small rows can quantize to m=0 and now deterministically produce zero.
Padding poison is ignored rather than included in a supposedly logical mean.

Historical accuracy/timing results are not evidence for this changed program.
Before model-equivalence or inference-quality claims: pin same checkpoint and
inputs, old/new profiles at each actual scale, per-node errors, normalization
errors, logits, argmax/sequence differences, task accuracy, and new source/range/
mask census. None is silently inferred from scalar or synthetic CPU tests.
