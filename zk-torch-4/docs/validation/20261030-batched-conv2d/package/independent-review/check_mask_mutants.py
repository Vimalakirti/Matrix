#!/usr/bin/env python3
"""Small deterministic mutant-killing evidence, distinct from source execution."""
from pathlib import Path
import json
from check_relations import Q,eq
B=C=D=3;P=4
eb,ed=eq([7,11]),eq([13,17])
x=[[1+b*3+c if b<B and c<C else 1001+b*7+c for c in range(P)]for b in range(P)]
w=[[2+d*5+c if d<D and c<C else 2003+d*11+c for c in range(P)]for d in range(P)]
def contraction(bn,cn,dn):
 return sum(eb[b]*ed[d]*x[b][c]*w[d][c] for b in range(bn) for c in range(cn) for d in range(dn))%Q
want=contraction(B,C,D)
mutants={'drop_B_mask':contraction(P,C,D),'drop_C_mask':contraction(B,P,D),'drop_D_mask':contraction(B,C,P),'drop_all_masks':contraction(P,P,P)}
assert all(v!=want for v in mutants.values())
# Equal/opposite batch output errors vanish under a naive batch total but not rb.
assert 1+(-1)==0
assert (eb[0]-eb[1])%Q!=0
# Full advice needs unmasked b,d check. A coefficient poisoned at b3,d3,m1
# changes its polynomial identity although its masked crop never reads it.
alpha=19;delta=eb[3]*ed[3]*alpha%Q
assert delta!=0
result={'reference':want,'rejected_mutants':mutants,'equal_opposite_batch_detected':True,'padded_full_advice_delta':delta,'scope':'Independent arithmetic mutants only, not production proof acceptance'}
Path(__file__).with_name('mask-mutants-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
