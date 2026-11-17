#!/usr/bin/env python3
"""Exact-source overlay audit. Does not execute or certify the whole project."""
from pathlib import Path
import argparse,hashlib,json,re
out=Path(__file__).parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',type=Path,required=True,help='Frozen CNN base repository root')
parser.add_argument('--root',type=Path,required=True,help='Combined convolution repository root')
parser.add_argument('--supplement',type=Path,required=True,help='Previously reviewed CNN wrapper source root; frozen base also works if those bodies are identical')
parser.add_argument('--manifest',type=Path,required=True,help='Frozen 454-source base manifest JSON')
parser.add_argument('--reviewed-hashes',type=Path,default=out / 'reviewed-source-hashes.json',help='Previously recorded reviewed source hashes JSON')
args=parser.parse_args()
base=args.base
joint=args.root
supplement=args.supplement
manifest_path=args.manifest
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def all_files(root):return {str(x.relative_to(root)):sha(x) for x in root.rglob('*') if x.is_file() and '.git' not in x.relative_to(root).parts}
assert sha(manifest_path)=='da3f4f1bae5927b06547d61bd061725a46634db1cb944df469da532084770113'
original=json.loads(manifest_path.read_text());orig={x['path']:x['sha256'] for x in original}
a,b=all_files(base),all_files(joint)
assert len(orig)==454 and all(a.get(p)==v for p,v in orig.items())
changes=sorted(p for p in set(a)|set(b) if a.get(p)!=b.get(p))
expected=['zk-torch-4/src/morpheus/piop/cv.rs','zk-torch-4/src/morpheus/piop/cv_batch_tests.rs','zk-torch-4/src/morpheus/piop/mod.rs']
assert changes==expected
old_review=json.loads(args.reviewed_hashes.read_text())
for f in expected[:2]:assert b[f]==old_review[f]
mod='zk-torch-4/src/morpheus/piop/mod.rs'
expected_mod=(base/mod).read_text().replace('#[cfg(test)]\nmod exp_tests;', '#[cfg(test)]\nmod exp_tests;\n#[cfg(test)]\nmod cv_batch_tests;').replace('&& nd.outputs.len() == 2 { committed[nd.outputs[1]]','&& nd.outputs.len() >= 2 { committed[nd.outputs[1]]')
assert expected_mod==(joint/mod).read_text()
for module in ['exp_tests','cv_batch_tests','cnn_layout_tests']:
 assert expected_mod.count('mod '+module+';')==1
 assert '#[cfg(test)]\nmod '+module+';' in expected_mod
builder='zk-torch-4/src/dag/builder.rs'
assert (base/builder).read_bytes()==(joint/builder).read_bytes()
def body(s,name):
 pat=r'    (?:pub(?:\([^\)]*\))? )?fn '+name+r'\('
 m=re.search(pat,s);assert m,name
 end=s.index('\n    }',m.start())+6
 return s[m.start():end]
bodies={}
for name in ['mask','fold_per_channel2d_input','restore_per_channel2d_output','conv2d','conv2d_strided','conv2d_dilated']:
 old=body((supplement/builder).read_text(),name);new=body((joint/builder).read_text(),name)
 assert old==new,name
 bodies[name]=hashlib.sha256(new.encode()).hexdigest()
# Canonical source universe is exactly original 454 paths + one new test file.
# Record inherited bytecode separately; never silently treat it as source.
paths=sorted(set(orig)|{expected[1]})
canonical=[{'path':p,'sha256':b[p]}for p in paths]
mp=out/'joint-reviewed-source-manifest.json';mp.write_text(json.dumps(canonical,indent=2)+'\n')
extras={p:b[p] for p in sorted(set(b)-set(paths))}
assert all('/__pycache__/' in p and p.endswith('.pyc') and a.get(p)==b[p] for p in extras)
result={'scope':'Exact overlay and scoped Conv2D/wrapper source recheck only; not whole-project acceptance','base_manifest_sha256':sha(manifest_path),'canonical_source_files':len(canonical),'combined_manifest_sha256':sha(mp),'changed_paths':changes,'cv_byte_identical_to_review':True,'test_file_byte_identical_to_review':True,'mod_exact_two_edits_and_preserves_existing_test_modules':True,'builder_byte_identical_to_frozen_joint':True,'previously_reviewed_wrapper_and_conv_function_bodies_identical':bodies,'inherited_nonsource_bytecode_excluded':extras,'native_piop_status':'Linker blocked; no native PIOP execution inferred by this audit'}
(out/'joint-overlay-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
