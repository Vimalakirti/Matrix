#!/usr/bin/env python3
"""Run210 CNN +978 inherited unchanged CPU test bodies on a frozen joint source.
Needs existing Rust/Cargo and dependency availability; never installs CUDA or stubs.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
from verify_bundle import verify_bundle,verify_source
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
source=a.source.resolve(strict=True);work=a.work.resolve(strict=False);bundle=Path(__file__).resolve().parent
for root,label in [(source,'SOURCE'),(bundle,'bundle')]:
 if work==root or root in work.parents:raise SystemExit(f'WORK must be outside resolved {label}')
if work.exists():raise SystemExit('WORK must be new')
verify_bundle(bundle);verify_source(bundle,source)
work.mkdir(parents=True);env=os.environ.copy();env.update(CARGO_INCREMENTAL='0',CARGO_PROFILE_DEV_DEBUG='0',CARGO_TARGET_DIR=str(work/'target'))
def run(args,log,extra=None):
 e=env.copy();e.update(extra or {})
 with (work/log).open('w') as f:
  result=subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,env=e)
 if result.returncode:raise SystemExit(f'FAILED ({result.returncode}); see {work/log}')
 return (work/log).read_text()
run(['rustc','--version'],'rust-version.log');run(['cargo','--version'],'cargo-version.log')
run([sys.executable,bundle/'prepare_harness.py',source,work/'cnn','cpu'],'cnn-adaptations.log')
run([sys.executable,bundle/'prepare_inherited_harness.py',source,work/'inherited','cpu-joint'],'inherited-adaptations.log')
bins=['cpu_basicblock_exp','cpu_graph_exp','cpu_normalization','cpu_partial_chunks','cpu_softmax','cpu_instance_norm']
run(['cargo','build','--locked','--manifest-path',work/'cnn/zk-torch-4/Cargo.toml','--bin','cpu_cnn_cpu'],'cnn-build.log')
args=['cargo','build','--locked','--manifest-path',work/'inherited/zk-torch-4/Cargo.toml']
for b in bins:args+=['--bin',b]
run(args,'inherited-build.log')
summary=[]
for binary,label,scales,count in [('cpu_cnn_cpu','cnn',range(21),10),('cpu_basicblock_exp','exp-basic',range(31),6),('cpu_graph_exp','exp-graph',range(31),8),('cpu_softmax','softmax',range(31),6),('cpu_normalization','mean-rms',range(21),9),('cpu_instance_norm','instance',range(21),8),('cpu_partial_chunks','partial-chunks',[0],1)]:
 for sf in scales:
  name=f'{label}-sf{sf}.log';text=run([work/'target/debug'/binary],name,{'ZK4_SF_LOG':str(sf)})
  passes=sum(x.startswith('PASS ') for x in text.splitlines())
  if passes!=count:raise SystemExit(f'wrong PASS count {passes}, expected{count}: {name}')
  summary.append({'group':label,'sf':sf,'pass_bodies':passes,'log':name,'sha256':hashlib.sha256((work/name).read_bytes()).hexdigest()})
total=sum(x['pass_bodies'] for x in summary);assert total==1188
(work/'summary.json').write_text(json.dumps({'total':total,'checks':summary},indent=2)+'\n')
print('PASS',total,'CPU body invocations; no PIOP/native/full-proof runtime claimed; logs',work)
