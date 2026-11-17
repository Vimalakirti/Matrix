#!/usr/bin/env python3
"""Independent finite-field equations/index review; no production Rust execution."""
from pathlib import Path
import hashlib, itertools, json, random
Q=(1<<64)-(1<<32)-31
rng=random.Random(0xBADC0FFEE)
checks=0

def ck(a,b,why):
 global checks
 checks+=1
 assert a%Q==b%Q, (why,a%Q,b%Q)
def log(n): return (max(1,n)-1).bit_length()
def pad(n): return 1<<log(n)
def eq(r):
 e=[1]
 for t in r: e=[v*(1-t)%Q for v in e]+[v*t%Q for v in e]
 return e
def point(n): return [rng.randrange(2,Q-1) for _ in range(n)]
def mle(v,r): return sum(a*b for a,b in zip(v,eq(r)))%Q
def fold(v,n,off,r,logical):
 # Index split exactly reviewed masked_axis_fold, independently interpreted.
 low=(1<<off)-1; e=eq(r)
 return [sum(e[c]*v[(i&low)|((i>>off)<<(off+len(r)))|(c<<off)] for c in range(logical))%Q for i in range(1<<(n-len(r)))]
def partial(v,n,fixed):
 # Generic reference assignment sum, not binary folding implementation.
 free=[i for i in range(n) if i not in fixed]
 ans=[0]*(1<<len(free))
 for i,vv in enumerate(v):
  wt=1
  for j,r in fixed.items(): wt=wt*(r if i>>j&1 else 1-r)%Q
  k=sum(((i>>j)&1)<<a for a,j in enumerate(free))
  ans[k]=(ans[k]+vv*wt)%Q
 return ans

def case(B,C,D,H,W,KH,KW,SH,SW,DH,DW):
 global checks
 HO=(H-1-DH*(KH-1))//SH+1; WO=(W-1-DW*(KW-1))//SW+1
 bp,cp,dp,hp,wp,khp,kwp=map(pad,[B,C,D,H,W,KH,KW]);hop,wop=map(pad,[HO,WO])
 lw,lh,lc,ld,lb,lkh,lkw=map(log,[W,H,C,D,B,KH,KW]);si=lw+lh;sk=lkh+lkw
 xv=[0]*(bp*cp*hp*wp);wv=[0]*(dp*cp*khp*kwp)
 def xi(b,c,h,w): return w+wp*(h+hp*(c+cp*b))
 def wi(d,c,h,w): return w+kwp*(h+khp*(c+cp*d))
 for b,c,h,w in itertools.product(range(bp),range(cp),range(hp),range(wp)):
  xv[xi(b,c,h,w)]=(rng.randrange(-7,8) if b<B and c<C and h<H and w<W else rng.randrange(1000,2000))%Q
 for d,c,h,w in itertools.product(range(dp),range(cp),range(khp),range(kwp)):
  wv[wi(d,c,h,w)]=(rng.randrange(-7,8) if d<D and c<C and h<KH and w<KW else rng.randrange(2000,3000))%Q
 y=[0]*(bp*dp*hop*wop)
 for b,d,h,w in itertools.product(range(B),range(D),range(HO),range(WO)):
  y[w+wop*(h+hop*(d+dp*b))]=sum(xv[xi(b,c,h*SH+kh*DH,w*SW+kw*DW)]*wv[wi(d,c,kh,kw)] for c in range(C) for kh in range(KH) for kw in range(KW))%Q
 rw,rh,rd,rb=map(point,[log(WO),log(HO),ld,lb]);ew,eh,ed,eb=map(eq,[rw,rh,rd,rb])
 folded=fold(xv,si+lc+lb,si+lc,rb,B)
 generic=partial(xv,si+lc+lb,{si+lc+j:r for j,r in enumerate(rb)}) if B==bp else None
 if generic is not None:
  for a,b in zip(folded,generic): ck(a,b,'full axis fold')
 # Exact sequential contraction ordering from source, w first then h.
 cur=folded;sizes=[lw,lh,lc]
 for axis,(out_n,k_n,in_n,stride,dilation,er,ib,kb) in enumerate([(WO,KW,W,SW,DW,ew,lw,lkw),(HO,KH,H,SH,DH,eh,lh,lkh)]):
  m=[[] for _ in range(1<<kb)]
  for o,k in itertools.product(range(out_n),range(k_n)):
   i=o*stride+k*dilation
   if i<in_n:m[k].append((i,er[o]))
  pre=1<<sum(sizes[:axis]);post=1<<sum(sizes[axis+1:]);ni=1<<ib;nk=1<<kb
  cur=[sum(coef*cur[(ph*ni+i)*pre+pl] for i,coef in m[k])%Q for ph in range(post) for k in range(nk) for pl in range(pre)]
  sizes[axis]=kb
 at=cur;wt=fold(wv,sk+lc+ld,sk+lc,rd,D)
 z=mle(y,rw+rh+rd+rb)
 ck(z,sum(at[i]*wt[i] for i in range(len(at)) if (i>>sk)<C),'normal first stage')
 xk=point(sk);xc=point(lc);ek=eq(xk);ec=eq(xc)
 # Normal patch child must equal exact map to X, including masked b.
 tw=[0]*wp;th=[0]*hp
 for o,k in itertools.product(range(WO),range(KW)):tw[o*SW+k*DW]=(tw[o*SW+k*DW]+ew[o]*eq(xk[:lkw])[k])%Q
 for o,k in itertools.product(range(HO),range(KH)):th[o*SH+k*DH]=(th[o*SH+k*DH]+eh[o]*eq(xk[lkw:])[k])%Q
 rhs=sum(xv[xi(b,c,h,w)]*eb[b]*ec[c]*tw[w]*th[h] for b in range(B) for c in range(cp) for h in range(hp) for w in range(wp))
 ck(mle(at,xk+xc),rhs,'normal patch source link')
 rhs=sum(wv[wi(d,c,h,w)]*ed[d]*ec[c]*ek[w+kwp*h] for d in range(D) for c in range(cp) for h in range(khp) for w in range(kwp))
 ck(mle(wt,xk+xc),rhs,'normal kernel source link')
 # Polynomial product and full-convolution coefficient/crop identity.
 S=hp*wp;J=(KH-1)*DH*wp+(KW-1)*DW+1;JP=pad(J);L=S+J-1;LP=pad(L)
 wf=[rng.randrange(3000,4000) for _ in range(dp*cp*JP)]
 taps=[kh*DH*wp+kw*DW for kh in range(KH) for kw in range(KW)]
 assert len(set(taps))==len(taps);checks+=1
 for d,c,kh,kw in itertools.product(range(D),range(C),range(KH),range(KW)):
  wf[kw*DW+kh*DH*wp+JP*(c+cp*d)]=wv[wi(d,c,kh,kw)]
 full=[0]*(bp*dp*LP)
 for b,d,c,h,w,kh,kw in itertools.product(range(B),range(D),range(C),range(H),range(W),range(KH),range(KW)):
  p=w+wp*h;j=kw*DW+kh*DH*wp;m=S-1-p+j
  assert 0<=m<L;checks+=1
  idx=m+LP*(d+dp*b)
  full[idx]=(full[idx]+xv[xi(b,c,h,w)]*wf[j+JP*(c+cp*d)])%Q
 for b,d,h,w in itertools.product(range(B),range(D),range(HO),range(WO)):
  E=S-1-h*SH*wp-w*SW
  ck(y[w+wop*(h+hop*(d+dp*b))],full[E+LP*(d+dp*b)],'PP coefficient crop')
  for kh,kw in itertools.product(range(KH),range(KW)):
   assert w*SW+kw*DW<W<=wp;checks+=1
   assert E==S-1-(h*SH+kh*DH)*wp-(w*SW+kw*DW)+(kh*DH*wp+kw*DW);checks+=1
 # Exact crop table is an MLE functional, even for stride 3.
 crop=[0]*LP
 for h,w in itertools.product(range(HO),range(WO)):
  E=S-1-h*SH*wp-w*SW
  crop[E]=(crop[E]+eh[h]*ew[w])%Q
 rhs=sum(crop[m]*ed[d]*eb[b]*full[m+LP*(d+dp*b)] for b in range(B) for d in range(D) for m in range(LP))
 ck(z,rhs,'PP exact crop map')
 for alpha in [0,1,rng.randrange(2,Q-1)]:
  am=[pow(alpha,m,Q) for m in range(LP)]
  ap=[am[S-1-p] if p%wp<W and p//wp<H else 0 for p in range(S)]
  aj=[am[j] if j in taps else 0 for j in range(JP)]
  f=fold(xv,si+lc+lb,si+lc,rb,B)
  wd=fold(wf,log(J)+lc+ld,log(J)+lc,rd,D)
  px=[sum(ap[p]*f[p+S*c] for p in range(S))%Q for c in range(cp)]
  pw=[sum(aj[j]*wd[j+JP*c] for j in range(JP))%Q for c in range(cp)]
  lhs=sum(am[m]*eb[b]*ed[d]*full[m+LP*(d+dp*b)] for b in range(bp) for d in range(dp) for m in range(LP))
  ck(lhs,sum(px[c]*pw[c] for c in range(C)),'PP polynomial identity')
  xpa=partial(xv,si+lc+lb,{si+j:r for j,r in enumerate(xc)})
  wpa=partial(wf,log(J)+lc+ld,{log(J)+j:r for j,r in enumerate(xc)})
  ck(mle(px,xc),sum(ap[p]*eb[b]*xpa[p+S*b] for b in range(B) for p in range(S)),'PP X child map')
  ck(mle(pw,xc),sum(aj[j]*ed[d]*wpa[j+JP*d] for d in range(D) for j in range(JP)),'PP W child map')
 # Generic masked-axis folding at non-high offset (transpose/shared helper).
 off=rng.randrange(sk+lc+ld+1);width=rng.randrange(sk+lc+ld-off+1);nr=point(width);logical=rng.randrange(1,(1<<width)+1)
 fv=fold(wv,sk+lc+ld,off,nr,logical)
 er=eq(nr);free=[i for i in range(sk+lc+ld) if not off<=i<off+width]
 for i,v in enumerate(fv):
  want=0
  for j,val in enumerate(wv):
   k=(j>>off)&((1<<width)-1)
   ix=sum(((j>>v)&1)<<a for a,v in enumerate(free))
   if ix==i and k<logical:want+=val*er[k]
  ck(v,want,'masked axis offset')
 return len(y)

def main():
 geometries=[(1,1,1,1,1,1,1,1),(3,5,2,2,1,1,1,1),(5,7,3,2,2,3,1,2),(7,5,2,2,3,2,2,1),(6,7,2,3,1,2,2,2)]
 total=0;cases=0
 for B,C,D in itertools.product([1,2,3,5],[1,3],[1,3]):
  for geom in geometries:
   total+=case(B,C,D,*geom);cases+=1
 # Nonmatching FlattenKernel is a concrete actual-DAG mismatch.
 x=list(range(1,9));w=[2,3]
 wf=[2,0,3,0]
 actual=[x[o]*wf[0]+x[o+1]*wf[1] for o in range(7)]
 bypass=[x[o]*w[0]+x[o+1]*w[1] for o in range(7)]
 assert actual!=bypass
 report={'scope':'Independent CPU field/index equations, not native Rust PIOP execution','seed':'0xBADC0FFEE','prime':Q,'cases':cases,'assertions':checks,'padded_output_slots':total,'batches':[1,2,3,5],'channels':[1,3],'alpha_cases':[0,1,'non-Boolean random'],'stride_cases':[1,2,3],'dilation_cases':[1,2],'mismatched_flatten_kernel_counterexample':{'X':x,'W':w,'FlattenKernel_dilation_w':2,'Conv2D_dilation_w':1,'actual_output':actual,'normal_unflatten_output':bypass}}
 Path(__file__).with_name('results.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
