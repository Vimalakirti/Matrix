#!/usr/bin/env python3
"""Reproduce isolated CUDA-free validation copies. Never alters input snapshot.
Usage: python prepare_harness.py SOURCE DEST MODE
MODE: typecheck, cpu-replay, cpu-basicblock, cpu-graph
CPU modes supply NO GPU implementation/stub: any retained GPU reference must
fail at link time. Bodies of selected original tests are unchanged; only their
module visibility/cfg(test) gates and #[test] attributes become callable.
"""
import pathlib, shutil, sys, re, hashlib, json, difflib
src,dst=map(pathlib.Path,sys.argv[1:3]); mode=sys.argv[3]
assert mode in ['typecheck','cpu-replay','cpu-basicblock','cpu-graph']
# Resolve aliases BEFORE creating/copying anything: a destination inside its
# source (including through a symlink) recursively copies the new copy itself.
src=src.resolve(strict=True);dst=dst.resolve(strict=False)
if not src.is_dir():raise SystemExit('SOURCE must be a directory')
if dst==src or src in dst.parents:
    raise SystemExit('DEST must be outside resolved SOURCE')
bundle=pathlib.Path(__file__).resolve().parent
if dst==bundle or bundle in dst.parents:raise SystemExit('DEST must be outside resolved bundle')
if dst.exists():raise SystemExit('DEST must not already exist')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
f=dst/'almost-goldilocks-cuda-rs/build.rs';s=f.read_text()
a=s.index('    let (nvcc, cuda_lib_path) = find_cuda_paths();');b=s.index('    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE");')
f.write_text(s[:a]+'    // VALIDATION ONLY: CUDA compilation and linker directives omitted.\n'+s[b:])
if mode!='typecheck':
    f=dst/'almost-goldilocks-cuda-rs/src/ffi.rs';s=f.read_text();tag='#[link(name = "almost_goldilocks_cuda_wrapper", kind = "static")]';assert s.count(tag)==1
    f.write_text(s.replace(tag,'// VALIDATION ONLY: no native library supplied; any reachable FFI must fail linking.'))
    if mode=='cpu-replay':
        f=dst/'zk-torch-4/src/morpheus/piop/mod.rs';s=f.read_text();tag='#[cfg(test)]\nmod exp_tests;';assert tag in s
        f.write_text(s.replace(tag,'// VALIDATION ONLY: expose unchanged CPU test bodies to standalone runner.\npub mod exp_tests;'))
        f=dst/'zk-torch-4/src/morpheus/piop/exp_tests.rs';s=f.read_text();names=re.findall(r'#\[test\]\nfn (\w+)\(',s);s=s.replace('#[test]\nfn ','pub fn ')
        mod='morpheus::piop::exp_tests';binary='cpu_exp_replay'
    else:
        f=dst/'zk-torch-4/src/basicblock/exp.rs';s=f.read_text();s=s.replace('#[cfg(test)]\nmod tests {','// VALIDATION ONLY: expose original CPU test bodies.\npub mod tests {');names=re.findall(r'#\[test\]\n    fn (\w+)\(',s);s=s.replace('#[test]\n    fn ','pub fn ')
        mod='basicblock::exp::tests';binary='cpu_basicblock_exp'
    f.write_text(s)
    runner='fn main() {\n'
    for n in names: runner+=f'    println!("RUN {n}");\n    zk_torch_4::{mod}::{n}();\n    println!("PASS {n}");\n'
    runner+='}\n';(dst/f'zk-torch-4/src/bin/{binary}.rs').write_text(runner)
    if mode=='cpu-graph':
        f=dst/'zk-torch-4/src/dag/builder.rs';s=f.read_text();s=s.replace('#[cfg(test)]\nmod tests {','// VALIDATION ONLY: expose original CPU graph test bodies.\npub mod tests {');names=re.findall(r'#\[test\]\n    fn (\w+)\(',s);s=s.replace('#[test]\n    fn ','pub fn ');f.write_text(s)
        runner='fn main() {\n'
        for n in names: runner+=f'    println!("RUN {n}");\n    zk_torch_4::dag::builder::tests::{n}();\n    println!("PASS {n}");\n'
        runner+='}\n';(dst/'zk-torch-4/src/bin/cpu_graph_exp.rs').write_text(runner)
manifest=[];diff=[]
for f in sorted(dst.rglob('*')):
    if not f.is_file():continue
    rel=f.relative_to(dst);old=src/rel;data=f.read_bytes()
    if old.exists() and data==old.read_bytes():continue
    manifest.append({'path':str(rel),'original_sha256':hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None,'adapted_sha256':hashlib.sha256(data).hexdigest()})
    diff.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],f.read_text().splitlines(True),fromfile='original/'+str(rel),tofile='adapted/'+str(rel)))
(dst.parent/(dst.name+'-adaptations.json')).write_text(json.dumps(manifest,indent=2)+'\n')
(dst.parent/(dst.name+'-adaptations.patch')).write_text(''.join(diff))
print(json.dumps(manifest,indent=2))
