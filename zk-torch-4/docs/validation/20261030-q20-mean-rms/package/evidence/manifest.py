#!/usr/bin/env python3
"""Regenerate a portable candidate manifest/patch in a NEW evidence directory."""
from pathlib import Path
import argparse,hashlib,json,difflib
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('base',type=Path);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();base=a.base.resolve();src=a.source.resolve();out=a.output.resolve()
if out.exists():p.error('output must not already exist (frozen receipts are not overwritten)')
if src in out.parents or base in out.parents:p.error('output must be outside source and base')
out.mkdir(parents=True);files=[];patch=[];allfiles=[];inventory=[]
for f in sorted(src.rglob('*')):
 if not f.is_file()or '.git'in f.parts or 'target'in f.parts:continue
 rel=f.relative_to(src);data=f.read_bytes();r={'path':str(rel),'sha256':hashlib.sha256(data).hexdigest()};inventory.append(r)
 if '__pycache__'in f.parts or f.suffix=='.pyc':continue
 allfiles.append(r);old=base/rel
 if old.exists()and old.read_bytes()==data:continue
 files.append(dict(r,base_sha256=hashlib.sha256(old.read_bytes()).hexdigest()if old.exists()else None))
 patch.extend(difflib.unified_diff(old.read_text().splitlines(True)if old.exists()else [],data.decode().splitlines(True),fromfile='a/'+str(rel)if old.exists()else '/dev/null',tofile='b/'+str(rel)))
for name,rows in [('source-manifest.json',files),('full-source-manifest.json',allfiles),('complete-workspace-inventory.json',inventory)]:
 (out/name).write_text(json.dumps(rows,indent=2)+'\n')
(out/'mean-rms.patch').write_text(''.join(patch))
print(json.dumps({'changed_files':len(files),'required_source_files':len(allfiles),'inventory_files':len(inventory)}))
