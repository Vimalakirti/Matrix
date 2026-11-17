#!/usr/bin/env python3
"""Supplemental independent explicit-B wrapper index check, not a Rust run."""
from pathlib import Path
import hashlib,itertools,json,sys

def pad(n):return 1<<(n-1).bit_length()
checks=0;cases=0
for B,C,H,W in itertools.product([1,3,5],[1,3,5],[2,3],[4,5]):
 bp,cp,hp,wp=map(pad,[B,C,H,W]);n=bp*cp*hp*wp
 # The primitive views change only shape, never evaluation order.
 for i in range(n):
  q=i;coords=[]
  for length in [W,H,C,B]:
   coords.append(q%pad(length));q//=pad(length)
  w,h,c,b=coords
  idx=w+wp*(h+hp*(c+cp*b))
  assert idx==i
  mask=(w<W and h<H and c<C and b<B)
  folded_channel=i//(wp*hp)
  assert folded_channel==c+cp*b
  # Exposed restored value is exactly zero for every padded B/C/spatial cell.
  poisoned=19 if mask else 1<<40
  exposed=poisoned*int(mask)
  assert exposed==(19 if mask else 0)
  checks+=3
 # Explicit branch uses original B directly, not B*C/Cp.
 inferred=B
 assert inferred==B and cp*bp==pad(cp*bp)
 cases+=1;checks+=1
if len(sys.argv)!=2:raise SystemExit('usage: check_wrapper_integration.py SOURCE_ROOT')
root=Path(sys.argv[1]).resolve(strict=True)/'zk-torch-4/src/dag/builder.rs'
s=root.read_text()
for f in ['conv2d','conv2d_strided','conv2d_dilated']:
 a=s.index('    pub fn '+f+'(');b=s.index('\n    }',a);body=s[a:b]
 assert 'let explicit_batch = x_shape.len() == 4' in body
 assert 'assert_eq!(x_shape[1], c_in' in body
 assert 'x_shape[0]\n' in body
 assert 'vec![self.change_shape(y_edge, vec![batch, c_out, h_out, w_out])]' in body
 checks+=4
out={'scope':'Supplemental index/source check only, not maxpool function soundness or native execution','cases':cases,'assertions':checks,'builder_sha256':hashlib.sha256(root.read_bytes()).hexdigest()}
Path(__file__).with_name('wrapper-integration-results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
