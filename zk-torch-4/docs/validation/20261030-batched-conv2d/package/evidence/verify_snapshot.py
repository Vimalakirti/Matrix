#!/usr/bin/env python3
"""Verify the exact canonical source tree, rejecting unsafe manifest paths."""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('manifest',type=Path);a=p.parse_args();root=a.root.resolve(strict=True)
items=json.loads(a.manifest.read_text());expected={}
for entry in items:
    rel=entry['path'];q=PurePosixPath(rel)
    if q.is_absolute() or '..' in q.parts or str(q)!=rel or rel in expected: raise SystemExit('unsafe or duplicate manifest path: '+rel)
    f=root/rel
    if f.is_symlink() or root not in f.resolve(strict=True).parents or not f.is_file(): raise SystemExit('unsafe source: '+rel)
    actual=hashlib.sha256(f.read_bytes()).hexdigest()
    if actual!=entry['sha256']:raise SystemExit('hash mismatch: '+rel)
    expected[rel]=actual
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not any(x in p.relative_to(root).parts for x in ['.git','__pycache__','target'])}
if actual!=set(expected):raise SystemExit('file-set mismatch: '+repr(sorted(actual.symmetric_difference(expected))))
print(json.dumps({'verified_files':len(expected),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}))
