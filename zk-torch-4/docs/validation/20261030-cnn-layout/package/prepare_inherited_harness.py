#!/usr/bin/env python3
import pathlib,subprocess,sys,re,json,hashlib,difflib
src,dst=map(pathlib.Path,sys.argv[1:3]);mode=sys.argv[3]
subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('prepare_mean_rms_harness.py')),str(src),str(dst),'cpu-normalization' if mode in ['cpu-softmax','cpu-joint'] else mode],check=True)
if mode in ['cpu-softmax','cpu-joint']:
 r=dst/'zk-torch-4/src';p=r/'dag/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod softmax_tests;','// VALIDATION ONLY: original CPU test bodies exposed.\npub mod softmax_tests;'))
 p=r/'dag/softmax_tests.rs';s=p.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);p.write_text(s.replace('#[test]\nfn ','pub fn '))
 (r/'bin/cpu_softmax.rs').write_text('fn main(){\n'+''.join(f'println!("RUN {n}");zk_torch_4::dag::softmax_tests::{n}();println!("PASS {n}");\n' for n in names)+'}\n')
if mode=='cpu-joint':
 r=dst/'zk-torch-4/src';p=r/'dag/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod instancenorm_tests;','// VALIDATION ONLY: original CPU test bodies exposed.\npub mod instancenorm_tests;'))
 p=r/'dag/instancenorm_tests.rs';s=p.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);p.write_text(s.replace('#[test]\nfn ','pub fn '))
 (r/'bin/cpu_instance_norm.rs').write_text('fn main(){\n'+''.join(f'println!("RUN {n}");zk_torch_4::dag::instancenorm_tests::{n}();println!("PASS {n}");\n' for n in names)+'}\n')
manifest=[];diff=[]
for p in sorted(dst.rglob('*')):
 if not p.is_file() or '.git' in p.parts:continue
 rel=p.relative_to(dst);old=src/rel
 if old.exists() and old.read_bytes()==p.read_bytes():continue
 manifest.append(dict(path=str(rel),original_sha256=hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None,adapted_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile='original/'+str(rel),tofile='adapted/'+str(rel)))
(dst.parent/(dst.name+'-adaptations.json')).write_text(json.dumps(manifest,indent=2)+'\n');(dst.parent/(dst.name+'-adaptations.patch')).write_text(''.join(diff))
