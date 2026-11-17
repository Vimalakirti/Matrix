#!/usr/bin/env python3
"""Apply the exact Q20 patch to a verified exp base without changing that base."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check(tree, records, key='sha256'):
    bad=[]
    for row in records:
        p=tree/row['path']; actual=sha(p) if p.is_file() else None
        if actual!=row[key]: bad.append(row['path'])
    if bad: raise SystemExit('Source hash mismatch: '+', '.join(bad))

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True,help='full checkout of the preceding frozen exp candidate')
p.add_argument('--output',type=Path,required=True,help='new directory; must not already exist')
a=p.parse_args(); ev=Path(__file__).resolve().parent; base=a.base.resolve(); output=a.output.resolve()
if not base.is_dir():p.error('--base must exist')
if output.exists():p.error('--output must not exist')
if output==base or base in output.parents:p.error('--output must be outside --base')
delta=json.loads((ev/'source-manifest.json').read_text()); full=json.loads((ev/'full-source-manifest.json').read_text())
check(base,delta,'base_sha256')
shutil.copytree(base,output,ignore=shutil.ignore_patterns('.git','target','__pycache__','*.pyc'))
for args in (['git','apply','--check',str(ev/'mean-rms.patch')],['git','apply',str(ev/'mean-rms.patch')]):
    subprocess.run(args,cwd=output,check=True)
check(output,full)
print(json.dumps({'status':'pass','delta_files':len(delta),'required_source_files':len(full),'patch_sha256':sha(ev/'mean-rms.patch')},sort_keys=True))
