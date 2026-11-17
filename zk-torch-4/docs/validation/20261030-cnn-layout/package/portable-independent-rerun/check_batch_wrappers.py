#!/usr/bin/env python3
"""Independent pad/pool wrapper layout check. Does not prove max achievability."""
from itertools import product
from pathlib import Path
import json

def pad(n):return 1<<(n-1).bit_length()
def idx(b,c,h,w,cp,hp,wp):return w+wp*(h+hp*(c+cp*b))
cases=0
for B,C,H,W in product([1,3,5],[1,3,5],[4,6],[6,10]):
 BP,CP,HP,WP=map(pad,[B,C,H,W])
 for poison in [2**40,-2**40]:
  raw=[poison]*(BP*CP*HP*WP)
  logical={}
  for b,c,h,w in product(range(B),range(C),range(H),range(W)):
   value=b*100+c*17+h*3-w-5
   raw[idx(b,c,h,w,CP,HP,WP)]=value
   logical[b,c,h,w]=value
  # Reversed-shape logical mask, then unchanged folded bytes.
  masked=list(raw)
  for b,c,h,w in product(range(BP),range(CP),range(HP),range(WP)):
   if b>=B or c>=C or h>=H or w>=W:masked[idx(b,c,h,w,CP,HP,WP)]=0
  PH,PW=H+2,W+2;PHP,PWP=pad(PH),pad(PW)
  folded_pad=[0]*(BP*CP*PHP*PWP)
  for row,h,w in product(range(BP*CP),range(H),range(W)):
   folded_pad[(w+1)+PWP*((h+1)+PHP*row)]=masked[w+WP*(h+HP*row)]
  # Restore explicit [B,C,PH,PW] is an equal-arity raw view.
  assert len(folded_pad)==BP*CP*PHP*PWP
  for b,c,h,w in product(range(BP),range(CP),range(PHP),range(PWP)):
   ref=logical.get((b,c,h-1,w-1),0)
   assert folded_pad[idx(b,c,h,w,CP,PHP,PWP)]==ref
  for K,S in [(2,2),(3,1)]:
   OH,OW=(PH-K)//S+1,(PW-K)//S+1;OHP,OWP=pad(OH),pad(OW)
   folded_out=[0]*(BP*CP*OHP*OWP)
   for row,oh,ow in product(range(BP*CP),range(OH),range(OW)):
    folded_out[ow+OWP*(oh+OHP*row)]=max(folded_pad[(ow*S+kw)+PWP*((oh*S+kh)+PHP*row)] for kh,kw in product(range(K),range(K)))
   for b,c,oh,ow in product(range(BP),range(CP),range(OHP),range(OWP)):
    ref=0
    if b<B and c<C and oh<OH and ow<OW:
     ref=max(logical.get((b,c,oh*S+kh-1,ow*S+kw-1),0) for kh,kw in product(range(K),range(K)))
    assert folded_out[idx(b,c,oh,ow,CP,OHP,OWP)]==ref
   cases+=1
out={'wrapper_geometry_cases':cases,'batch':[1,3,5],'channels':[1,3,5],'pool_kernels_strides':[[2,2],[3,1]],'poison':[2**40,-2**40],'claim':'Independent layout/honest-forward integer reference, not cryptographic maxpool achievability'}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
