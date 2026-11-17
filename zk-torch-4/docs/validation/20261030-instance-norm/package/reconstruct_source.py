#!/usr/bin/env python3
"""Apply the frozen InstanceNorm patch to a new copy of its exact Q20 base."""
import argparse,hashlib,json,pathlib,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('base',type=pathlib.Path);p.add_argument('destination',type=pathlib.Path);a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
assert a.base.is_dir() and not a.destination.exists(),'requires existing base and NEW destination'
a.base=a.base.resolve(strict=True);a.destination=a.destination.resolve()
assert a.destination != a.base and a.base not in a.destination.parents,'destination must be outside base tree'
for item in json.loads((root/'source-manifest.json').read_text()):
 if item['base_sha256'] is not None:assert hashlib.sha256((a.base/item['path']).read_bytes()).hexdigest()==item['base_sha256'],item['path']
shutil.copytree(a.base,a.destination,ignore=shutil.ignore_patterns('.git','target','__pycache__'))
patch=(root/'instance-norm.patch').resolve()
for cmd in [['git','apply','--check',str(patch)],['git','apply',str(patch)]]:subprocess.run(cmd,cwd=a.destination,check=True)
for rel,h in json.loads((root/'full-source-manifest.json').read_text()).items():assert hashlib.sha256((a.destination/rel).read_bytes()).hexdigest()==h,rel
print('PASS reconstructed446-source snapshot')
