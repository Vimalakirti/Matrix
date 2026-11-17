#!/usr/bin/env python3
"""Fresh validation copy only: omit native build/link, expose unchanged CPU test bodies."""
from pathlib import Path
import sys,subprocess,re,json,hashlib,difflib
src,dst=map(Path,sys.argv[1:3]);mode=sys.argv[3]
subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_base_harness.py')),str(src),str(dst), 'cpu-graph' if mode in ['cpu-instance','cpu-normalization'] else mode],check=True)
if mode=='cpu-instance':
 p=dst/'zk-torch-4/src/dag/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod instancenorm_tests;','// VALIDATION ONLY: original bodies exposed.\npub mod instancenorm_tests;'))
 p=dst/'zk-torch-4/src/dag/instancenorm_tests.rs';s=p.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);p.write_text(s.replace('#[test]\nfn ','pub fn '))
 (dst/'zk-torch-4/src/bin/cpu_instance_norm.rs').write_text('fn main(){\n'+''.join(f'println!("RUN {n}");zk_torch_4::dag::instancenorm_tests::{n}();println!("PASS {n}");\n' for n in names)+'}\n')
if mode=='cpu-normalization':
 p=dst/'zk-torch-4/src/dag/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod normalization_tests;','// VALIDATION ONLY: original bodies exposed.\npub mod normalization_tests;'))
 p=dst/'zk-torch-4/src/dag/normalization_tests.rs';s=p.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);p.write_text(s.replace('#[test]\nfn ','pub fn '))
 (dst/'zk-torch-4/src/bin/cpu_normalization.rs').write_text('fn main(){\n'+''.join(f'println!("RUN {n}");zk_torch_4::dag::normalization_tests::{n}();println!("PASS {n}");\n' for n in names)+'}\n')
 p=dst/'zk-torch-4/src/morpheus/piop/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod exp_tests;','// VALIDATION ONLY: partial chunk structural test.\npub mod exp_tests;'))
 p=dst/'zk-torch-4/src/morpheus/piop/exp_tests.rs';p.write_text(p.read_text().replace('#[test]\nfn ','pub fn '))
 (dst/'zk-torch-4/src/bin/cpu_partial_chunks.rs').write_text('fn main(){zk_torch_4::morpheus::piop::exp_tests::nonnegative_partial_chunks_preserve_61_62_63_bit_widths();println!("PASS nonnegative_partial_chunks_preserve_61_62_63_bit_widths");}\n')
manifest=[];patch=[]
for p in sorted(dst.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(dst);old=src/rel
 if old.exists() and old.read_bytes()==p.read_bytes():continue
 manifest.append(dict(path=str(rel),original_sha256=hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None,adapted_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 patch.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile='original/'+str(rel),tofile='adapted/'+str(rel)))
(dst.parent/(dst.name+'-adaptations.json')).write_text(json.dumps(manifest,indent=2)+'\n')
(dst.parent/(dst.name+'-adaptations.patch')).write_text(''.join(patch))
