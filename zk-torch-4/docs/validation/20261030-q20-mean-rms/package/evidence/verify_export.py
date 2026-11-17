#!/usr/bin/env python3
"""Verify portable bundle identity and optionally a reconstructed full candidate."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path)
a=p.parse_args();root=Path(__file__).resolve().parent.parent;ev=root/'evidence'
def check(folder,rows):
 bad=[]
 for r in rows:
  f=folder/r['path']
  if not f.is_file()or hashlib.sha256(f.read_bytes()).hexdigest()!=r['sha256']:bad.append(r['path'])
 if bad:raise SystemExit('Hash mismatches: '+', '.join(bad))
manifest=json.loads((root/'export-manifest.json').read_text());check(root,manifest['files'])
actual={str(f.relative_to(root))for f in root.rglob('*')if f.is_file()and f.name!='export-manifest.json'}
expected={r['path']for r in manifest['files']}
if actual!=expected:raise SystemExit('Unexpected/missing export files: '+', '.join(sorted(actual^expected)))
delta=json.loads((ev/'source-manifest.json').read_text());check(root/'source',delta)
full=json.loads((ev/'full-source-manifest.json').read_text())
assert len(full)==443 and all('__pycache__'not in r['path']and not r['path'].endswith('.pyc')for r in full)
if a.source:check(a.source.resolve(),full)
print(json.dumps({'status':'pass','export_files':len(manifest['files']),'overlay_files':len(delta),'required_source_files':len(full),'full_candidate_verified':bool(a.source)},sort_keys=True))
