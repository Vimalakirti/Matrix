#!/usr/bin/env python3
"""Expose only unchanged inherited CNN CPU tests in a fresh base harness."""
import subprocess,sys,re,json,hashlib,difflib
from pathlib import Path
src,dst=map(Path,sys.argv[1:]);subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_harness.py')),str(src),str(dst)],check=True)
manifest=[];patch=[];names=[]
for module in ['cnn_layout_tests','cnn_finite_activation_tests']:
    rel=Path('zk-torch-4/src/dag/mod.rs');p=dst/rel;old=p.read_text();new=old.replace(f'#[cfg(test)]\nmod {module};',f'pub mod {module};');assert old!=new;p.write_text(new)
    patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/'+str(rel),tofile='after/'+str(rel)))
    rel=Path(f'zk-torch-4/src/dag/{module}.rs');p=dst/rel;old=p.read_text();fs=re.findall(r'#\[test\]\nfn (\w+)\(',old);new=old.replace('#[test]\nfn ','pub fn ');p.write_text(new)
    assert new.replace('pub fn ','fn ')==old.replace('#[test]\n','')
    manifest.append({'path':str(rel),'original_sha256':hashlib.sha256(old.encode()).hexdigest(),'exposed_sha256':hashlib.sha256(new.encode()).hexdigest(),'unchanged_function_bodies':True,'test_names':fs})
    patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='before/'+str(rel),tofile='after/'+str(rel)))
    names.extend((module,f) for f in fs)
runner='fn main() {\n'+''.join(f'println!("RUN {m}::{f}");zk_torch_4::dag::{m}::{f}();println!("PASS {m}::{f}");\n' for m,f in names)+'}\n'
(dst/'zk-torch-4/src/bin/cpu_conv_inherited.rs').write_text(runner)
(dst/'inherited-adaptations.json').write_text(json.dumps(manifest,indent=2)+'\n');(dst/'inherited-adaptations.patch').write_text(''.join(patch))
print(json.dumps({'unchanged_inherited_test_bodies':len(names)}))

# Complete source-to-harness delta, including generated runners and module visibility.
all_changes=[];all_patch=[]
for f in sorted((dst/'zk-torch-4').rglob('*'))+sorted((dst/'almost-goldilocks-cuda-rs').rglob('*')):
    if not f.is_file(): continue
    rel=f.relative_to(dst);oldfile=src/rel
    old=oldfile.read_bytes() if oldfile.exists() else b'';new=f.read_bytes()
    if old==new: continue
    all_changes.append({'path':str(rel),'original_sha256':hashlib.sha256(old).hexdigest() if oldfile.exists() else None,'adapted_sha256':hashlib.sha256(new).hexdigest()})
    all_patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='original/'+str(rel),tofile='adapted/'+str(rel)))
(dst/'complete-inherited-adaptations.json').write_text(json.dumps(all_changes,indent=2)+'\n')
(dst/'complete-inherited-adaptations.patch').write_text(''.join(all_patch))
