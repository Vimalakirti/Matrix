#!/usr/bin/env python3
"""Record all validation-copy changes; never writes to either source tree."""
from pathlib import Path
import argparse,difflib,hashlib,json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source',type=Path);p.add_argument('adapted',type=Path);p.add_argument('output_prefix',type=Path)
a=p.parse_args();src=a.source.resolve();dst=a.adapted.resolve();prefix=a.output_prefix.resolve()
paths=[Path(str(prefix)+suffix)for suffix in ['.json','.patch','-test-body-identity.json']]
if any(x.exists()for x in paths):p.error('output files must not already exist')
if src in prefix.parents or dst in prefix.parents:p.error('output must be outside source and adapted trees')
records=[];diff=[]
for f in sorted(dst.rglob('*')):
 if not f.is_file()or '.git'in f.parts or 'target'in f.parts or '__pycache__'in f.parts or f.suffix=='.pyc':continue
 rel=f.relative_to(dst);old=src/rel
 if old.exists()and old.read_bytes()==f.read_bytes():continue
 records.append(dict(path=str(rel),original_sha256=hashlib.sha256(old.read_bytes()).hexdigest()if old.exists()else None,adapted_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 diff.extend(difflib.unified_diff(old.read_text().splitlines(True)if old.exists()else [],f.read_text().splitlines(True),fromfile='original/'+str(rel)if old.exists()else '/dev/null',tofile='adapted/'+str(rel)))
prefix.parent.mkdir(parents=True,exist_ok=True)
paths[0].write_text(json.dumps(records,indent=2)+'\n');paths[1].write_text(''.join(diff))
rel='zk-torch-4/src/dag/normalization_tests.rs';orig=src/rel;adapt=dst/rel
same=orig.read_text().replace('#[test]\nfn ','pub fn ')==adapt.read_text()
if same:
 paths[2].write_text(json.dumps(dict(original_sha256=hashlib.sha256(orig.read_bytes()).hexdigest(),adapted_sha256=hashlib.sha256(adapt.read_bytes()).hexdigest(),only_change='#[test] fn to pub fn for nine original test bodies'),indent=2)+'\n')
print(json.dumps({'changed_files':len(records),'normalization_visibility_only':same}))
