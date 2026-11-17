#!/usr/bin/env python3
"""Verify this evidence bundle and, optionally, the frozen source snapshot."""
import argparse,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path);a=p.parse_args();root=pathlib.Path(__file__).resolve().parent
m=json.loads((root/'export-manifest.json').read_text());seen=set()
for item in m['files']:
 rel=pathlib.PurePosixPath(item['path']);assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen;seen.add(str(rel))
 data=(root/rel).read_bytes();assert len(data)==item['bytes'],str(rel);assert hashlib.sha256(data).hexdigest()==item['sha256'],str(rel)
print(f'PASS {len(seen)} evidence byte/hash records')
if a.source:
 files=json.loads((root/'full-source-manifest.json').read_text())
 for rel,h in files.items():assert hashlib.sha256((a.source/rel).read_bytes()).hexdigest()==h,rel
 print(f'PASS {len(files)} frozen source hashes')
