# Tensor-batched Conv2D relation and validation scope

This is tensor batching inside one Conv2D. It is **not** the deferred-proof
batching protocol, whose authentication, transcript, opening and accumulation
security obligations remain separate.

## Registered layout and exact semantics

Let `B,C,D,H,W,Kh,Kw` be **logical** positive dimensions. Pad every axis
independently to the next power of two and put bits in the order
`w,h,c,b` (least to most significant); kernel bits are `kw,kh,c,d`.
A shared kernel has no batch coordinate. An explicit input `[B,C,H,W]`
preserves logical B through registered `ChangeShape` nodes; a legacy folded
`[Bp*Cp,H,W]` shape alone cannot recover an unrecorded original B.
The public Conv2D `batch` parameter, already in the statement codec, is B.

For logical output indices,

`Y[b,d,oh,ow] = Σ_{c<C,kh<Kh,kw<Kw} X[b,c,sh*oh+dh*kh,sw*ow+dw*kw] W[d,c,kh,kw]`.

Every output outside logical `B,D,Ho,Wo` is zero, regardless of nonzero
committed padding in X or W. The relation is over the base field Fq (embedded
in K=Fq[Theta]/(Theta^8-3)). The code performs modular arithmetic. To interpret
this equality as a signed-integer convolution one separately needs a no-wrap
bound, e.g. `C*Kh*Kw*M_X*M_W < q/2` with authenticated signed bounds
`|X|≤M_X`, `|W|≤M_W`, plus corresponding output range interpretation. This
extension does not silently assume or prove those integer range bounds.

## Normal reduction

For output challenge `r=(r_w,r_h,r_d,r_b)`, define

`X_b[c,i] = Σ_{b<B} eq(r_b,b) X[b,c,i]`,
`W_d[c,k] = Σ_{d<D} eq(r_d,d) W[d,c,k]`,
`A[c,k] = Σ_{oh<Ho,ow<Wo} eq(r_h,oh)eq(r_w,ow) X_b[c,i(o,k)]`.

The first masked sumcheck proves `Y~(r)=Σ_{c,k}[c<C] W_d[c,k] A[c,k]`.
The logical tap restrictions live in the spatial relation constructing A;
therefore poisoned padded taps cannot contribute. Kernel and patch terminal
claims are reduced with the same public D and B masks, respectively, rather
than replacing a masked sum with an unrestricted evaluation. For power-of-two
B/D the corresponding identity map becomes a direct point fixation.

A `FlattenKernel` producer may be bypassed only when its stride, tap sizes,
channels and dilations agree with the registered Conv2D. Otherwise the normal
path uses the actual flat edge and a public tap map `k -> dh*kh*Wp+dw*kw`.
The flat edge's padded length must agree with Conv2D's memory-index strides.
This keeps a differently parameterized producer in the proof instead of
silently proving another graph. Only equal-arity ChangeShape nodes may be skipped; truncate/regrow chains keep their actual flat source. Noninjective or out-of-bounds FlattenKernel scatters fail closed, with checked index arithmetic. Direct flat-weight inputs of the expected
extent are also supported.

## Polynomial-product (PP) reduction

Put `S=Hp*Wp`, `p=iw+Wp*ih`, and `j=dh*kh*Wp+dw*kw`. Define

`P_{b,c}(T)=Σ_{ih<H,iw<W} X[b,c,ih,iw] T^(S-1-p)`,
`Q_{d,c}(T)=Σ_{kh<Kh,kw<Kw} W_flat[d,c,j] T^j`.

For every logical b,d, full coefficients F satisfy

`Σ_m F[b,d,m] T^m = Σ_{c<C} P_{b,c}(T) Q_{d,c}(T)`.

All F slots outside logical b,d or its full coefficient support are zero.
The full coefficient count is `L=S+s_kernel-1`, padded to Lp. The exponent
sum is ordinary integer addition (at most L-1), **not** reduction modulo Lp,
q, or a cyclic transform length. Thus the field polynomial has no exponent
wrap/aliasing. Fq coefficients may still wrap as explained above.

At independently sampled alpha,r_b,r_d, the verifier binds the unrestricted
MLE of F in b,d to the RHS masked to b<B,d<D,c<C and real spatial/tap indices.
Leaving F unrestricted is important: this pins its padded slots to zero.
Powers are computed directly, without alpha inverse, so alpha=0 is defined.
The polynomial identity contributes at most `(Lp-1+log2(Bp)+log2(Dp))/|K|`
by a total-degree Schwartz-Zippel bound, in addition to the separate masked
sumcheck and authentication terms. This is a local relation bound, not a
claim about total protocol security.

The crop map is the exact public relation

`(oh,ow) -> E=S-1-sh*Wp*oh-sw*ow`.

For a valid output, `F[b,d,E]=Y[b,d,oh,ow]`. It has no cyclic carry ambiguity:
`p-(sh*Wp*oh+sw*ow)=j` implies the intended 2D tap because horizontal input
and tap offsets stay inside one width-Wp row. The public exact relation also
supports non-power-of-two strides; it does not depend on a bit-affine shortcut.
The output crop additionally masks logical B,D,Ho,Wo.

## Costs (not constant-time batching)

Normal prover batch contraction reads O(Bp*Cp*Hp*Wp) cells, then performs the
existing spatial contractions on the folded image. A non-power-of-two B adds
`log2(Bp)` variables to the input-map sumcheck; a power-of-two B can be fixed.
Similarly non-power-of-two D requires a masked kernel-axis reduction. The
first sumcheck has `log2(Khp*Kwp*Cp)` variables and a public C mask when needed.
The fallback flat-kernel source map adds a map reduction rather than a new
commitment.

PP commits Bp*Dp*Lp full coefficients. Its full-coefficient, channel,
input-link and kernel-link sumchecks have respectively `log2(Lp)`,
`log2(Cp)`, `log2(Hp*Wp*Bp)`, and `log2(Kflatp*Dp)` variables. The exact crop
map sums over the full coefficient bits and, when needed, masked B/D bits;
it trades the old bit-affine view's cheaper output-spatial reduction for
uniform public index correctness. Batch-dependent prover memory/work must
be counted, not described as independent of B.

## Verification status

Authored Rust CPU forward/semantic and PIOP/link-row replay tests are in
`src/morpheus/piop/cv_batch_tests.rs`. They are not PCS proofs. A disposable
build-only harness retained every native extern declaration, omitted CUDA
build/link directives and exposed unchanged test bodies. Library typecheck
passed; the executable failed at native-symbol linkage. Therefore **zero
replay test bodies executed**, and there is no native CPU, CUDA or end-to-end
acceptance claim from that attempt. Independent standalone field/index
checks are narrower algebraic evidence; see the independent review receipt.

The separately linked CPU-only runner did execute genuine, unchanged production
forward bodies: 32 dimension/stride/dilation cases, eight cases invoking the
actual production reduction-table helpers at non-Boolean K points (PP alpha
0/1/nonbase), three actual flat-source fixtures, and collision/overflow guards.
This is meaningful source/field/forward evidence but does not turn the blocked
masked-PIOP runner into an executed proof.
