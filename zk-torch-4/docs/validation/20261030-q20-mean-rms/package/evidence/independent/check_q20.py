from math import isqrt
import random,json
P=18446744069414584289; M=(1<<28)-1
rng=random.Random(897533);groups=0
for s in range(21):
 C=1<<(3*s);Q=1<<s
 bs=[(0,0,0,0)]+[(1<<j,(1<<(j+1))-1,isqrt(C//(1<<j))+1,isqrt(C*((1<<(j+1))-1))+1+(1<<(j))) for j in range(28)]+[(-1,-1,0,0)]*3
 R=max(x[2] for x in bs);U=max(x[3] for x in bs);D=max(max(r*u+u+hi//4 for lo,hi,r,u in bs if lo>=0),C+U+1)
 assert M*R<1<<59 and D<1<<61
 assert 32*U<P and 32*R<P
 for lo,hi,r,u in bs:
  if lo>=0:assert r*u+u+hi//4<=D
 def acc(m,r,bidx):
  if not(0<=m<=M and 0<=r<=R):return False
  lo,hi,rb,ub=bs[bidx]
  if not lo<=m<=hi or r>rb:return False
  u=m*r
  if not 0<=u<=ub:return False
  z=int(m==0);b=int(r==0)
  if z*r:return False
  h=(m-1)//4;t=u*r
  dl=(1-b)*(C-t+u-h-1);dh=(1-z)*(t+u+h-C)
  return 0<=dl<=D and 0<=dh<=D
 ms=set(range(1000))|{0,1,M}|{min(M,max(0,(1<<j)+delta)) for j in range(29) for delta in [-2,-1,0,1,2]}|{rng.randrange(M+1) for _ in range(5000)}
 for m in ms:
  t=isqrt(C//m) if m else 0;r=(t+(m*(2*t+1)**2<=4*C)) if m else 0
  bi=m.bit_length();assert acc(m,r,bi),(s,m,r,bi)
  for wrong in {r-1,r+1,-r,0,R} - {r}:assert not acc(m,wrong,bi),(s,m,r,wrong)
  for badbucket in {0,28,29,30,31,max(0,bi-1),min(31,bi+1)}-{bi}:assert not acc(m,r,badbucket),(s,m,r,badbucket)
  groups+=1
 # Norm output completeness and no-wrap (independent coarse proof).
 X=isqrt(M*Q);H=min((X*R+Q-1)//Q,(1<<31)-1);W=(1<<30)-1
 assert Q*1255+8193 < (1<<31)-1
 assert X*R+Q*H+Q<P
 Y=(H*W+Q-1)//Q
 assert H*W+Q*Y+Q<P
 assert 2*Y<1<<62
print(json.dumps({'status':'pass','groups':groups,'scales':'0..20','private_buckets':32,'scope':'integer constraints and bounds only'}))
