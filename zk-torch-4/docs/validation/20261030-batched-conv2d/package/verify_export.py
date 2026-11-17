#!/usr/bin/env python3
"""Verify exact export inventory/bytes/hashes, optionally 455 canonical sources."""
import argparse,hashlib,json,re,sys
sys.dont_write_bytecode=True
from pathlib import Path,PurePosixPath

def sha(b):return hashlib.sha256(b).hexdigest()
def safe(s):
    p=PurePosixPath(s)
    if not s or not p.parts or p.is_absolute() or str(p)!=s or '..' in p.parts or '\\' in s or '\x00' in s:raise ValueError('unsafe manifest path')
    return s

def read(root,rel):
    f=root/rel
    for p in [f,*f.parents]:
        if p==root:break
        if p.is_symlink():raise ValueError('symlink in manifest path: '+rel)
    if not f.is_file() or root not in f.resolve(strict=True).parents:raise ValueError('missing or escaping file: '+rel)
    return f.read_bytes()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--expected-manifest-sha');a=p.parse_args();root=a.root.resolve(strict=True)
    raw=read(root,'export-manifest.json');mh=sha(raw)
    if a.expected_manifest_sha and mh!=a.expected_manifest_sha:raise ValueError('export manifest hash mismatch')
    obj=json.loads(raw);files=obj['files'];expected=set();total=0
    for x in files:
        rel=safe(x['path'])
        if rel in expected or rel=='export-manifest.json' or not re.fullmatch('[0-9a-f]{64}',x['sha256']):raise ValueError('duplicate/invalid manifest entry')
        b=read(root,rel)
        if type(x['bytes']) is not int or len(b)!=x['bytes'] or sha(b)!=x['sha256']:raise ValueError('byte/hash mismatch: '+rel)
        expected.add(rel);total+=len(b)
    actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() or f.is_symlink()}
    if actual!=expected|{'export-manifest.json'}:raise ValueError('export inventory mismatch: '+repr(sorted(actual.symmetric_difference(expected|{'export-manifest.json'}))))
    source_files=None
    if a.source:
        source=a.source.resolve(strict=True);rows=json.loads(read(root,'evidence/final-source-manifest.json'));seen=set()
        if len(rows)!=455:raise ValueError('expected 455 canonical source files')
        for x in rows:
            rel=safe(x['path'])
            if rel in seen or sha(read(source,rel))!=x['sha256']:raise ValueError('source hash mismatch: '+rel)
            seen.add(rel)
        source_files=len(seen)
    print(json.dumps({'verified_export_files':len(files),'verified_export_bytes_excluding_manifest':total,'export_manifest_sha256':mh,'canonical_source_files_verified':source_files,'source_extra_paths_are_not_authenticated':bool(a.source)}))
if __name__=='__main__':main()
