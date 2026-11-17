#!/usr/bin/env python3
"""Rerun preserved independent Python checks in a NEW external work directory.
Only the historical fixed source-file path is parameterized; arithmetic unchanged.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
from verify_bundle import verify_bundle,verify_source
p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('work',type=Path);a=p.parse_args()
source=a.source.resolve(strict=True);work=a.work.resolve(strict=False);bundle=Path(__file__).resolve().parent
for root in [source,bundle]:
 if work==root or root in work.parents:raise SystemExit('WORK must be outside resolved source and bundle')
if work.exists():raise SystemExit('WORK must be new')
verify_bundle(bundle);verify_source(bundle,source);work.mkdir(parents=True)
files=['independent-review/check_maps.py','independent-review/check_batch_wrappers.py','independent-wrapper-review/check_wrapper_integration.py'];adaptations=[]
for rel in files:
 original=bundle/rel;target=work/original.name;text=original.read_text()
 target.write_text(text);adaptations.append({'original':rel,'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'executable_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'only_change':'none (portable script already accepts source argument)'})
 with (work/(target.stem+'.log')).open('w') as log:subprocess.run([sys.executable,str(target)]+([str(source)] if original.name=='check_wrapper_integration.py' else []),cwd=work,stdout=log,stderr=subprocess.STDOUT,check=True)
(work/'portability-adaptations.json').write_text(json.dumps(adaptations,indent=2)+'\n')
print('PASS independent arithmetic/index checks; not Rust/PIOP/full-model execution; results',work)
