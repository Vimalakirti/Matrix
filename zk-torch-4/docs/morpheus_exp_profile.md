# Canonical finite exp and immutable public literals

This is a numerical-correctness repair, not a model-accuracy result, a complete
normalization repair, or certification of all graph arithmetic. Rust/CUDA tests
were authored but **not executed** in the editing environment (no toolchain).
No GPU run or benchmark acceptance is claimed.

## Declared profile

Activation scale remains `Qs=2^s`, `0<=s<=30` (default s=10); the cubic uses
`Qe=32768`. `EXP_LS[s]` is an exact registered integer, independently checked
against 100-digit Decimal rounding of ln(2)*Qs. It is not verifier-time f64.
Default selector width is now **8**, configurable in `1..=8`. Invalid environment
values and unsupported domains fail closed. There is no saturation or index clamp.
For declared width b, let `L=EXP_LS[s]`, `h=floor(L/2)` and
`B=2^b*L-h-1`. Logical inputs must be in `[-B,0]`.

`k=floor((-x+h)/L)`, `r=x+L*k`, `rho=h-r`.
The graph proves `0<=-x<=B`, `0<=rho<=L-1`, and the selector proves
`0<=k<2^b`. Thus k/r are unique, including ties. Four explicit NonNegative
roots implement these interval checks. Compilation adds them even to standalone
raw ExpHelper nodes. Graph construction and statement encoding validate their
registered wiring, literal values, widths, and noncommitted derived edges.

The exact existing cubic operation order is retained:

1. Rescale r from s to 15, producing re (exact upscale, rounded downscale).
2. a=RD(re*re,Qe); b6=RD(5461*re,Qe); c=b6+16384.
3. hpoly=Qe+re+RD(c*a,Qe).
4. T[k]=Qe>>k for k<=15, otherwise0.
5. E=RD(T[k]*hpoly,2^(30-s)).

RD rounds nearest with ties toward positive infinity, including negatives. The
previous missing-unit repair (integer32768, not1) remains in place.

### Necessary quotient bounds

A ScaleDown remainder check **alone is not integer division**: modulo p one may
choose a different bounded remainder and a huge quotient. The exp recipe now
adds five interval-root pairs: re in[-16384,16384], a in[0,8192], b6 in[-2731,2731],
cubic quotient in[0,4779], final E in[-2Qs,2Qs]. The last interval is a conservative
signed certificate; logical outputs are nonnegative. These are actual graph
constraints, not observed-value assertions.

For all supported scales, the helper decomposition interval is below2^40 and
paired range sums below2^40, far below p=0xfffffffeffffffe1. With the above bounds,
internal polynomial products are below2^31 and every exp ScaleDown residual
interval is below2^33. Hence zero field residual cannot conceal a nonzero multiple
of p. The bounded input operands, operation order and quotient roots are all
needed for this conclusion; it does not extend to arbitrary ScaleDown nodes.

## Explicit padding-profile change

For nonpower-of-two logical shapes, public shape-derived 0/1 masks are multiplied
before and after exp. Logical coordinates are unchanged; padded input/output
coordinates are canonically zero. The mask uses the repository's column-major
padded layout. Missing/zero/unsupported logical extents and indexing overflow fail
closed, rather than inventing logical dimensions. Shape and exact mask values are
included in the registered program identity.

This intentionally changes the padded program. Previously SoftmaxConst could put
positive values into padding even when logical exp inputs were nonpositive, or
zero-padded cubic constants could yield negative padded exp outputs. Downstream
reductions may change if they consumed those old padded values. This is **not** a
claim of unchanged full-model logits, task accuracy, causal independence or
correct normalization advice.

## Public literals and eager arithmetic

`DagBuilder::public_literal` records immutable canonical values separately from
private `param` weights and variable request inputs. Literals use Role::Input and
are automatically present in Graph.public, so private-input registration does
not classify them as hidden request data. Public-root rows authenticate their
values. Statement codec v2 freezes literal VALUES into both program and request
identities; it rejects deletion/replacement in the request public map. Private
weight payloads remain excluded and require their existing commitment registration.
Cubic coefficients, exp range constants and builder masks use this registry.

Eager Add/Sub no longer promote request-dependent values to Role::Constant merely
because data is present. Only two semantic Role::Constant operands retain
model-only precomputation. Such precomputation belongs to the trusted registered
model setup, not a proof of the relationship between separate weight leaves.
Other constructors/whole-model preprocessing are not certified by this patch.

## Cost and evidence scope

Per exp recipe, the new interval certificates add14 NonNegative nodes,14 Sub nodes
and14 literal leaves (42 edges), in addition to migrating the three cubic
constants to public literals. Nonpower-of-two recipes additionally add two
pointwise mask Einsums and two mask literal leaves (four edges). Standalone raw
helpers receive only the four helper roots/Subs/literals. Default k lookup grows
from16 to256 entries, implemented using configured selector chunks (default tcl4).
For s10,b8,tcl4, helper guard widths18,18,10,10 use16 selector chunks; quotient
interval widths16,14,13,13,13 use40 chunks (two roots per interval). Mask Einsums
also incur their actual PIOP sumchecks/rows; no one-row-per-operator claim is made.

Independent scalar enumeration covered all181405 inputs at s10 and5803172 at s15,
all31 scales/eight widths at domain endpoints, and shifted k/r negative cases.
On the accepted logical domain, outputs equal the old **unit-corrected eight-bit**
profile. They intentionally differ from the old four-bit silent-clamp behavior:
for the -100 sentinel, corrected-old4/new8 is876/0 at s10 and27177/0 at s15.
The new padding profile is separately tested against the historical pad defects.

Authored Rust tests cover CPU graph forward values, eager-role classification,
standalone helper guards, literal identity and deletion, selector-width mismatch,
PIOP prover/verifier replay and emitted-link-row checking, compensated k/r,
modular quotient counterfeits, literal mutations and nonzero claimed padding.
These are **unexecuted**. Even when run, replay/link-row tests are not end-to-end
PCS/authentication acceptance tests. Production positive/negative tests and a
fresh source/row/mask census remain required before performance claims.

## Remaining whole-graph integer-correspondence gate

Generic division/rescaling outside this exp recipe remains a separate obligation.
Constructively: derive public interval bounds from registered leaf digit widths
and proved range roots; propagate integer expressions (including products) without
assuming centered field lifts; add bounded quotient/output roots for every divide
or rescale; bound the entire residual interval. If it contains a nonzero multiple
of p, reject that profile or use linked limbs/chunking. Do not infer no-wrap from
the field equation itself. Preserve exact rounding/tie/zero policies explicitly.
The exp modular quotient counterexample is a regression and evidence that this
whole-graph pass is necessary. Softmax/Sigmoid/InstanceNorm advice, RMS reciprocal
sign/zero/tie policy, mean division and causal denominator independence remain
open; this patch does not resolve them.
