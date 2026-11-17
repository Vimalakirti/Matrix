#!/usr/bin/env python3
"""Independent algebra/index check; NOT Rust execution or a cryptographic replay."""
from itertools import product
from math import prod
import random,json,pathlib,hashlib
Q=2**64-2**32-31
rng=random.Random(2026103007)
def pad(n):return 1<<(n-1).bit_length()
def bits(n):return (n-1).bit_length()
def ix(coords,dims):
 s=1;o=0
 for c,d in zip(coords,dims):o+=c*s;s*=pad(d)
 return o
def coords(i,dims):
 o=[]
 for d in dims:o.append(i%pad(d));i//=pad(d)
 return o
def mle(values,r):
 v=[x%Q for x in values]
 for t in r:v=[(v[2*j]*(1-t)+v[2*j+1]*t)%Q for j in range(len(v)//2)]
 assert len(v)==1
 return v[0]
def eq(r,i):
 v=1
 for j,t in enumerate(r):v=v*(t if i>>j&1 else 1-t)%Q
 return v
def stage(inp,idims,axis_shape,count,axis_stride,first):
 """Independent source equation, dense plain coordinate contraction."""
 n=pad(count);d=pad(axis_shape);remain=idims[1:] if first else idims[2:]
 odims=[count]+remain
 out=[0]*prod(map(pad,odims))
 table=[0]*(n*d)
 for i in range(count):table[i+n*((i//axis_stride)%axis_shape)]=1
 for out_ix in range(len(out)):
  oc=coords(out_ix,odims);z=oc[0];tail=oc[1:]
  acc=0
  for a in range(d):
   ic=[a]+tail if first else [z,a]+tail
   acc+=inp[ix(ic,idims)]*table[z+n*a]
  out[out_ix]=acc%Q
 return out,odims,table
cases=0;mle_checks=0
shapes=[list(s) for n in range(1,4) for s in product(range(1,6),repeat=n)]
shapes += [[16,5,5],[3,2,3,5],[3,3,2,3]]
for shape in shapes:
 reverse=shape[::-1];count=prod(shape);physical=prod(map(pad,shape))
 raw=[777777 if i%2 else -888888 for i in range(physical)]
 logical=[]
 for logical_i in range(count):
  rem=logical_i;co=[]
  for d in reverse:co.append(rem%d);rem//=d
  val=rng.randrange(-10000,10000);raw[ix(co,reverse)]=val;logical.append(val%Q)
 cur=[v%Q for v in raw];dims=reverse;stride=1
 for axis,d in enumerate(reverse):
  prev,pdims=cur,dims
  cur,dims,table=stage(prev,pdims,d,count,stride,axis==0)
  if prod(map(pad,dims))<=2048:
   # Check the PIOP relation at non-Boolean challenges. First stage fixes
   # output free-once axes; later stages sum the common z with eq(r_z,z).
   ro=[rng.randrange(2,Q) for _ in range(sum(map(bits,dims)))]
   zbits=bits(count);abits=bits(d);tail_r=ro[zbits:];rz=ro[:zbits]
   lhs=mle(cur,ro);rhs=0
   if axis==0:
    for a in range(pad(d)):
     ar=[(a>>j)&1 for j in range(abits)]
     rhs+=mle(prev,ar+tail_r)*mle(table,rz+ar)
   else:
    for z in range(pad(count)):
     zr=[(z>>j)&1 for j in range(zbits)]
     for a in range(pad(d)):
      ar=[(a>>j)&1 for j in range(abits)]
      rhs+=eq(rz,z)*mle(prev,zr+ar+tail_r)*table[z+pad(count)*a]
   assert lhs==rhs%Q,(shape,axis)
   mle_checks+=1
  stride*=d
 assert cur==logical+[0]*(pad(count)-count),shape
 cases+=1
# Independent row-major channel-bias coordinate checks. Shape reversal is a
# polynomial identity; Add's matched axes retain C and optionally spatial/B.
bias_cases=0
for b,c,h,w in product([1,3,5],[1,3,5],[2,3],[3,5]):
 bs,cs,hs,ws=map(pad,[b,c,h,w]);N=bs*cs*hs*ws
 raw=[-777]*N;expected=[0]*N
 for bi,ci,hi,wi in product(range(b),range(c),range(h),range(w)):
  i=wi+ws*(hi+hs*(ci+cs*bi));v=1000*bi+100*ci+10*hi+wi
  raw[i]=v;expected[i]=v+101*(ci+1)
 # Column-major Add after [W,H,C,B] reinterpretation, then exact logical mask.
 actual=[]
 for i in range(N):
  wi,hi,ci,bi=coords(i,[w,h,c,b]);mask=wi<w and hi<h and ci<c and bi<b
  actual.append((raw[i]+101*(ci+1)) if mask else 0)
 assert actual==expected
 bias_cases+=1
out={'flatten_shapes':cases,'non_boolean_mle_stage_checks':mle_checks,'channel_bias_shapes':bias_cases,'claim':'independent Python algebra/index checks only; no Rust PIOP or PCS execution'}
pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
