from math import isqrt
import random,json
P=18446744069414584289; M=(1<<28)-1
rng=random.Random(998109)
checks=0
for s in range(16):
 A=1<<(3*s);K=4*A;R=isqrt(A)+1;U=isqrt(M*A)+1+(M+1)//2;V=4*U*R+4*U+M
 assert V<1<<62 and M*R<P and U*R<P and V+(1<<62)-1<P
 assert 2*((1<<V.bit_length())-1)<P
 def accept(m,r,z,b):
  if not (0<=m<=M and 0<=r<=R and z in (0,1) and b in (0,1)):return False
  if z*m or z*r or b*r or m+z<1 or r+b<1:return False
  u=m*r
  if u>U:return False
  t=u*r;lo=(1-b)*(K-(4*t-4*u+m));hi=(1-z)*(4*t+4*u+m-K-1)
  return 0<=lo<=V and 0<=hi<=V
 ms={0,1,2,M,M-1, min(M,4*A),min(M,4*A+1)}|set(range(1000))|{rng.randrange(M+1) for _ in range(3000)}
 for m in ms:
  if not m:r=0
  else:
   t=isqrt(A//m);r=t+(m*(2*t+1)**2<=K)
  assert accept(m,r,int(m==0),int(r==0)),(s,m,r)
  for wrong in {r-1,r+1,-r,0,R} - {r}:
   for z in range(2):
    for b in range(2):assert not accept(m,wrong,z,b),(s,m,r,wrong,z,b)
  for z in range(2):
   for b in range(2):assert accept(m,r,z,b)==(z==int(m==0) and b==int(r==0))
  checks+=1
B=(1<<54)-1
for d in list(range(1,257))+[768,1024,4096,8192,1<<20]:
 for a in list(range(-1000,1001))+[-B,-B+1,B,B-1]:
  q=(a+d//2)//d
  assert -B//d-1<=q<=B//d+1
  assert 0<=a-d*q+d//2<d
  assert not 0<=a-d*(q-1)+d//2<d and not 0<=a-d*(q+1)+d//2<d
  checks+=1
print(json.dumps({'assertion_groups':checks,'scales':list(range(16)),'max_mean':M,'status':'pass','scope':'independent integer equations, not native PIOP execution'}))
