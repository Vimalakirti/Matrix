#!/usr/bin/env python3
"""Reproduce one declared public-data fit, or verify its exact artifact hashes.
Installs/downloads nothing directly. scikit-learn provides its bundled public dataset.
Never changes the frozen benchmark, searches seeds, or overwrites an output folder.
"""
import argparse,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;out=a.output.resolve();pins=json.loads((root/'expected_hashes.json').read_text())
if not a.verify_only:
 if out.exists():raise SystemExit('Refusing existing output directory; choose a fresh path')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 subprocess.run([sys.executable,str(root/'train_export.py'),'--output',str(out)],env=env,check=True)
failures=[]
for n,h in pins['files'].items():
 f=out/n
 actual=hashlib.sha256(f.read_bytes()).hexdigest()if f.is_file()else'MISSING'
 if actual!=h:failures.append({'file':n,'expected':h,'actual':actual})
report=json.loads((out/'report.json').read_text())
for k,v in [('float_correct',435),('integer_correct',435),('test_count',450),('selected_indices',pins['selected_indices'])]:
 if report.get(k)!=v:failures.append({'field':k,'expected':v,'actual':report.get(k)})
result={'python':platform.python_version(),'reported_versions':report.get('versions'),'files_checked':len(pins['files']),'exact_match':not failures,'failures':failures,'caveat':'Rerun convergence diagnostics do not recover missing original diagnostics.'}
print(json.dumps(result,indent=2))
if failures:raise SystemExit('Reproduction differs. Preserve output; do not tune seeds or replace the frozen benchmark.')
