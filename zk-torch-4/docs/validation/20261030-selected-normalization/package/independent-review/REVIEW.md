# Independent selected-normalization scope review

Reviewed locally on 2026-10-30. Scope: the new selected finite softmax/sigmoid
relation, canonical supported max, integer division/no-wrap proof, padding and
future-score exclusion, source/statement wiring, and added tests. This is NOT
one of five independent whole-paper acceptances, a model-equivalence review,
a historical-accuracy endorsement, GPU execution, or an authenticated PCS proof.

## Result

No remaining blocker found in the reviewed scope after two concrete findings
were addressed:

1. H originally used the full signed request-plane allocation. Its producer is
   now explicitly declared bit-typed in `tensor_specs` (one private plane), while
   real paired H-in-[0,1] graph roots remain. The documentation reflects this.
2. The new PIOP test used input100*Q but only32 signed source planes, invalid at
   high supported scales. Its public test configuration now declares64 planes;
   the legal one-hot mutation remains bit-typed. It does not count source-fit
   constructor failure as verifier rejection.

I independently verified the selected maximum needs BOTH supported membership
and dominance. The implementation proves bitness, supported H, sumH=V and bounded
selected differences. The one-hot sum has an independent integer lift because
the column count<p; selected scores independently have a signed lift. Canonical
m is identical across all maximum ties, including private hint tie choices.

The denominator reduction consumes e=A*E after exclusion, not unmasked E.
Raw excluded scores are also masked before the numeric-domain constraints.
Padded rows have fixed divisorQ and numerator0. Column-major flattening preserves
the last-axis reduction and every original logical/padded group coordinate.
Sigmoid masks its original padding before the two-logit construction and after
extracting the x weight.

The actual polynomial bounds establish E(0)=Q and0<=E<=Q without assuming real
exp, strict positivity, or monotonicity. The canonical maximum therefore implies
D>=Q despite the retained underflow tail. The finite ratio and output-rounding
claims are appropriately distinguished from real-exp/historical semantics.

The variable-D residual proof is non-circular: e,D,w are independently bounded;
the full F-rho interval is narrower than p; and rho+tau=2D-1 independently lifts.
The profile checks are sufficient for the unique nearest quotient with upward
half ties. They admit practical SF16/Q20 contexts and reject the concrete SF30
context9 counterfeit, while still admitting smaller SF30 contexts.

The live compile queue supplies actual arithmetic/range/PZC roots for both new
raw hint primitives. Morpheus `Walk::roots` walks every NonNegative descriptor
and every ProductZeroCheck node, so these constraints are not just annotations.
Advice values are linked sources, not host-asserted semantics. Public masks and
bounds are program literals frozen into statement identity. Legacy log advice
is rejected by compiler, Graph construction and statement encoding. Registration
still assumes the intended compiled DAG; this is explicitly documented rather
than a claimed compiler-equivalence theorem.

## Independently executed arithmetic evidence

`check.py`, written independently without importing implementation/reference
helpers, passed1,351,857 assertions:
-20,100 exhaustive small-prime modular lift/division tuples across accepted public
 budgets (including the slack from power-of-two residual lookup widths);
-1,298,988 additional actual-prime endpoint/random tuples across SF0..30 and
 accepted context profiles;
-32,769 actual integer cubic remainder values;
-the concrete SF30 wrap-counterfeit equation.

`results.txt` records source hashes for the reviewed normalization primitive,
DAG implementation and exact relation document. Parent's separately executed CPU
body logs were inspected, not reclassified as independent verifier execution.
PIOP replay remains authored/unexecuted until genuine native linkage is available.
The replay mutation currently changes a linked source while retaining other
honest intermediate witnesses; it tests actual source/constraint linkage, not
an adversary that recomputes every downstream witness consistently. CPU mutation
checks do recompute downstream operators and evaluate actual emitted range/PZC
predicates. Keep these evidence scopes distinct.
