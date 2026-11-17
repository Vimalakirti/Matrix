#!/usr/bin/env python3
"""Independently authored arithmetic/lift checks; no imported implementation."""
from random import Random
from hashlib import sha256
from pathlib import Path
P=18446744069414584289
checks=0
# Small prime analogues test *modular* accept iff exact nearest at every legal
# e,D,w, including the power-of-two range slack used by the real implementation.
for p in [17,31,61,127,251,509]:
 for q in [1,2,4,8,16]:
  for K in range(1,17):
   dm=q*K;t=(1<<(2*dm-1).bit_length())-1
   if not (2*t<p and 2*q*q+dm<p and dm*(2*q-1)+t<p):continue
   for e in range(q+1):
    for d in range(q,dm+1):
     wanted=(2*q*e+d)//(2*d)
     for w in range(q+1):
      rho=(2*q*e+d-2*d*w)%p;tau=(2*d-1-rho)%p
      assert ((rho<=t and tau<=t)==(w==wanted)),(p,q,K,e,d,w)
      checks+=1
print('PASS exhaustive small-field modular division/lift tuples:',checks)
# Actual independent endpoint/random residual checking at every activationSF.
rng=Random(4391)
for s in range(31):
 q=1<<s
 for K in [1,2,3,7,8,9,31,512,4096,1<<20]:
  dm=q*K;t=(1<<(2*dm-1).bit_length())-1
  safe=2*t<P and 2*q*q+dm<P and dm*(2*q-1)+t<P
  if not safe:continue
  for _ in range(1000):
   d=rng.randrange(q,dm+1);e=rng.randrange(q+1);want=(2*q*e+d)//(2*d)
   for w in set([0,q,want,max(0,want-1),min(q,want+1)]):
    rho=(2*q*e+d-2*d*w)%P;tau=(2*d-1-rho)%P
    assert ((rho<=t and tau<=t)==(w==want))
    checks+=1
print('PASS actual-prime profiles/endpoints/random modular tuples; combined:',checks)
# Integer polynomial exact enumeration independent of imported reference.
def rnd(n,d):return (n+d//2)//d
for r in range(-16384,16385):
 a=rnd(r*r,32768);c=16384+rnd(5461*r,32768);z=rnd(c*a,32768);h=32768+r+z
 assert 0<=a<=8192 and 13653<=c<=19115 and 0<=z<=4779 and 16384<=h<=53931
 if r<=0:assert z<=-r and h<=32768
 checks+=1
q=1<<30;d=8*q;rho=(d-2*d*q)%P;tau=2*d-1-rho
assert (rho,tau)==(4*q-31,12*q+30) and 0<=rho<2*d and 0<=tau<2*d
print('PASS exact32769 remainder polynomial bounds and SF30/context9 wrap attack; total checks:',checks)
# Usage: python check.py [REPOSITORY_ROOT_OR_ZK_TORCH_4_DIR]
# Omit the argument to inspect the current repository directory.
import sys
if len(sys.argv)>2:
 raise SystemExit("usage: check.py [repository-root-or-zk-torch-4-dir]")
root=Path(sys.argv[1] if len(sys.argv)==2 else '.').resolve()
if (root/'zk-torch-4').is_dir():root=root/'zk-torch-4'
if not (root/'src/basicblock/normalization.rs').is_file():
 raise SystemExit(f"not a selected-normalization source tree: {root}")
for rel in ['src/basicblock/normalization.rs','src/dag/softmax.rs','docs/morpheus_selected_normalization.md']:
 print('SHA256',rel,sha256((root/rel).read_bytes()).hexdigest())
