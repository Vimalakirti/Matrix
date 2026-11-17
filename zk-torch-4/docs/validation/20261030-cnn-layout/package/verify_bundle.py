#!/usr/bin/env python3
"""Verify a portable CNN evidence bundle and optional reconstructed source."""
import argparse,hashlib,json
from pathlib import Path

def safe_file(root,relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError(f'unsafe manifest path: {relative}')
    out=(root/rel).resolve(strict=True)
    if root not in out.parents or not out.is_file():raise ValueError(f'file escapes root: {relative}')
    return out

def check_rows(root,rows,byte_counts=False):
    root=root.resolve(strict=True);seen=set()
    for row in rows:
        rel=row['path']
        if rel in seen:raise ValueError(f'duplicate manifest path: {rel}')
        seen.add(rel);data=safe_file(root,rel).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError(f'SHA256 mismatch: {rel}')
        if byte_counts and len(data)!=row['bytes']:raise ValueError(f'length mismatch: {rel}')
    return seen

def verify_bundle(root):
    root=root.resolve(strict=True);m=json.loads((root/'export-manifest.json').read_text());expected=check_rows(root,m['files'],True)
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=expected|{'export-manifest.json'}:raise ValueError('bundle inventory differs from export manifest')
    return len(expected)

def verify_source(bundle,source,variant='joint'):
    return len(check_rows(source,json.loads((bundle/f'{variant}-source-manifest.json').read_text())))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--variant',choices=['joint','q20'],default='joint');a=p.parse_args();root=Path(__file__).resolve().parent
    print('PASS bundle',verify_bundle(root))
    if a.source:print('PASS source',verify_source(root,a.source,a.variant))
