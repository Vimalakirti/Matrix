#!/usr/bin/env python3
import pathlib,sys,subprocess,hashlib,json,difflib,re
src,dst=map(pathlib.Path,sys.argv[1:3]);mode=sys.argv[3]
assert mode in ['typecheck','cpu','replay']
subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('prepare_base_harness.py')),str(src),str(dst),'typecheck'],check=True,stdout=subprocess.DEVNULL)
if mode!='typecheck':
 p=dst/'almost-goldilocks-cuda-rs/src/ffi.rs';s=p.read_text();s=s.replace('#[link(name = "almost_goldilocks_cuda_wrapper", kind = "static")]','// VALIDATION ONLY: no native CUDA library and no FFI implementation/stubs.')
 p.write_text(s)
 section='dag' if mode=='cpu' else 'morpheus/piop'
 p=dst/f'zk-torch-4/src/{section}/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod cnn_layout_tests;','// VALIDATION ONLY: expose original test bodies.\npub mod cnn_layout_tests;'))
 p=dst/f'zk-torch-4/src/{section}/cnn_layout_tests.rs';s=p.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);p.write_text(s.replace('#[test]\nfn ','pub fn '))
 module=section.replace('/','::')
 (dst/'zk-torch-4/src/bin'/f'cpu_cnn_{mode}.rs').write_text('fn main(){\n'+''.join(f'println!("RUN {n}");zk_torch_4::{module}::cnn_layout_tests::{n}();println!("PASS {n}");\n' for n in names)+'}\n')
if mode=='cpu' and (dst/'zk-torch-4/src/dag/cnn_finite_activation_tests.rs').exists():
 p=dst/'zk-torch-4/src/dag/mod.rs';p.write_text(p.read_text().replace('#[cfg(test)]\nmod cnn_finite_activation_tests;','// VALIDATION ONLY: expose unchanged integration tests.\npub mod cnn_finite_activation_tests;'))
 p=dst/'zk-torch-4/src/dag/cnn_finite_activation_tests.rs';s=p.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);p.write_text(s.replace('#[test]\nfn ','pub fn '))
 p=dst/'zk-torch-4/src/bin/cpu_cnn_cpu.rs';s=p.read_text();end=s.rfind('}');p.write_text(s[:end]+''.join(f'println!("RUN {n}");zk_torch_4::dag::cnn_finite_activation_tests::{n}();println!("PASS {n}");\n' for n in names)+s[end:])
manifest=[];patch=[]
for p in sorted(dst.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(dst);old=src/rel
 if old.exists() and old.read_bytes()==p.read_bytes():continue
 manifest.append(dict(path=str(rel),original_sha256=hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None,adapted_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 patch.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],p.read_text().splitlines(True),fromfile='original/'+str(rel),tofile='adapted/'+str(rel)))
(dst.parent/(dst.name+'-adaptations.json')).write_text(json.dumps(manifest,indent=2)+'\n')
(dst.parent/(dst.name+'-adaptations.patch')).write_text(''.join(patch))
