#!/usr/bin/env python3
"""Independent flat-kernel fallback and source-identity checks."""
from pathlib import Path
import itertools,json,random
from check_relations import Q,pad,log,eq,mle,point,fold,ck
import check_relations as base
rng=random.Random(0xF1A7)
cases=0
for C,D,KH,KW,H,W,DH,DW in itertools.product([1,3],[1,3],[1,2],[1,2,3],[4],[8],[1,2],[1,2]):
 if (KH-1)*DH>=H or (KW-1)*DW>=W:continue
 cp,dp,kph,kpw=map(pad,[C,D,KH,KW]);sk=log(KH)+log(KW)
 J=(KH-1)*DH*pad(W)+(KW-1)*DW+1;JP=pad(J);lk=log(J)
 wf=[rng.randrange(Q) for _ in range(dp*cp*JP)]
 rd=point(log(D));xc=point(log(C));xk=point(sk)
 wd=fold(wf,lk+log(C)+log(D),lk+log(C),rd,D)
 # source virtual W table assignment, low k = kw + KH stride.
 vt=[0]*(cp*kph*kpw)
 pairs=[]
 for kh,kw in itertools.product(range(KH),range(KW)):
  k=kw+kpw*kh;j=kh*DH*pad(W)+kw*DW;pairs.append((k,j))
  for c in range(cp):vt[k+(c<<sk)]=wd[j+(c<<lk)]
 # Rel::table(public xk) over actual flat j.
 tj=[0]*JP;ek=eq(xk);ed=eq(rd);ec=eq(xc)
 for k,j in pairs:tj[j]=(tj[j]+ek[k])%Q
 want=sum(wf[j+JP*(c+cp*d)]*tj[j]*ec[c]*ed[d] for d in range(D) for c in range(cp) for j in range(JP))
 ck(mle(vt,xk+xc),want,'flat virtual kernel source map')
 cases+=1
# Same extent, different flatten map: fallback must use WF, not raw W.
x=list(range(1,9));w=[2,3]
# Conv DW2 requires J3 / pad4; FK DW3 requires J4 / pad4.
wf=[2,0,0,3]
actual=[x[o]*wf[0]+x[o+2]*wf[2] for o in range(6)]
bypass=[x[o]*w[0]+x[o+2]*w[1] for o in range(6)]
assert actual!=bypass
# Matching FK DW2 gives [2,0,3,0]; truncate to 2 then pad to 4 erases w1.
original_wf=[2,0,3,0];after_view=original_wf[:2]+[0,0]
assert [x[o]*after_view[0]+x[o+2]*after_view[2] for o in range(6)]==actual
report={'scope':'Independent finite-field flat virtual-kernel/source-map algebra, not production Rust execution','cases':cases,'assertions':base.checks,'same_extent_mismatched_fk':{'actual':actual,'bad_bypass':bypass},'truncate_then_restore_fk':{'original_wf':original_wf,'after_view':after_view,'actual':actual,'bad_bypass':bypass}}
Path(__file__).with_name('fallback-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
